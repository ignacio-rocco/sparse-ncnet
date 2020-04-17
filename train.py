import faiss
import os
from os.path import exists, join, basename
from collections import OrderedDict
import numpy as np
import numpy.random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import Dataset

from lib.dataloader import DataLoader # modified dataloader
from lib.model import ImMatchNet
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict, normalize_image_dict_caffe
from lib.torch_util import save_checkpoint, str_to_bool
from lib.torch_util import BatchTensorToVars, str_to_bool
from lib.sparse import get_scores

from lib.sparse import corr_and_add
import torch.nn.functional as F
from lib.model import featureL2Norm

import argparse


# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.manual_seed(10)
if use_cuda:
    torch.cuda.manual_seed(10)
np.random.seed(10)

print('Sparse-NCNet training script')

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--dataset_image_path', type=str, default='datasets/ivd', help='path to IVD dataset')
parser.add_argument('--dataset_csv_path', type=str, default='datasets/ivd/image_pairs/', help='path to IVD training csv')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[3,3], help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16,1], help='channels in neigh. cons')
parser.add_argument('--result_model_fn', type=str, default='sparsencnet', help='trained model filename')
parser.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--fe_finetune_params',  type=int, default=0, help='number of layers to finetune')
parser.add_argument('--k',  type=int, default=10, help='number of nearest neighs')
parser.add_argument('--symmetric_mode',  type=int, default=1, help='use symmetric mode')
parser.add_argument('--bn',  type=int, default=0, help='use batch norm')
parser.add_argument('--random_affine',  type=int, default=0, help='use affine data augmentation')
parser.add_argument('--feature_extraction_cnn',  type=str, default='resnet101', help='type of feature extractor')
parser.add_argument('--relocalize', type=int, default=0)
parser.add_argument('--change_stride', type=int, default=0)


args = parser.parse_args()
print(args)

# Create model
print('Creating CNN model...')
model = ImMatchNet(use_cuda=use_cuda,
                       checkpoint=args.checkpoint,
                       ncons_kernel_sizes=args.ncons_kernel_sizes,
                       ncons_channels=args.ncons_channels,
                       sparse=True,
                       symmetric_mode=bool(args.symmetric_mode),
                       feature_extraction_cnn=args.feature_extraction_cnn,
                       bn=bool(args.bn),
                       k=args.k)

if args.change_stride:
    model.FeatureExtraction.model[-1][0].conv1.stride=(1,1)
    model.FeatureExtraction.model[-1][0].conv2.stride=(1,1)
    model.FeatureExtraction.model[-1][0].downsample[0].stride=(1,1)

def eval_model_fn(batch):
    # feature extraction
    if args.relocalize:
        feature_A_2x = model.FeatureExtraction(batch['source_image'])
        feature_B_2x = model.FeatureExtraction(batch['target_image'])

        feature_A = F.max_pool2d(feature_A_2x, kernel_size=3, stride=2, padding=1)
        feature_B = F.max_pool2d(feature_B_2x, kernel_size=3, stride=2, padding=1)
    else:
        feature_A = model.FeatureExtraction(batch['source_image'])
        feature_B = model.FeatureExtraction(batch['target_image'])
        
    feature_A = featureL2Norm(feature_A)
    feature_B = featureL2Norm(feature_B)
    
    fs1, fs2 = feature_A.shape[-2:]
    fs3, fs4 = feature_B.shape[-2:]
    
    corr4d = corr_and_add(feature_A, feature_B, k = model.k)
    corr4d = model.NeighConsensus(corr4d)
    
    return corr4d

# Set which parts of the model to train
if args.fe_finetune_params>0:
    for i in range(args.fe_finetune_params):
        for p in model.FeatureExtraction.model[-1][-(i+1)].parameters(): 
            p.requires_grad=True

print('Trainable parameters:')
for i,p in enumerate(filter(lambda p: p.requires_grad, model.parameters())): 
    print(str(i+1)+": "+str(p.shape))
    
# Optimizer
print('using Adam optimizer')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
cnn_image_size=(args.image_size,args.image_size)

Dataset = ImagePairDataset
train_csv = 'train_pairs.csv'
test_csv = 'val_pairs.csv'
if args.feature_extraction_cnn == 'd2': #.startswith('d2'):
    normalization_tnf = normalize_image_dict_caffe
else:
    normalization_tnf = NormalizeImageDict(['source_image','target_image'])
    
batch_preprocessing_fn = BatchTensorToVars(use_cuda=use_cuda)   

# Dataset and dataloader
dataset = Dataset(transform=normalization_tnf,
                  dataset_image_path=args.dataset_image_path,
                  dataset_csv_path=args.dataset_csv_path,
                  dataset_csv_file = train_csv,
                  output_size=cnn_image_size,
                  random_affine = bool(args.random_affine))

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, 
                        num_workers=0)

dataset_test = Dataset(transform=normalization_tnf,
                       dataset_image_path=args.dataset_image_path,
                       dataset_csv_path=args.dataset_csv_path,
                       dataset_csv_file=test_csv,
                       output_size=cnn_image_size)

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)
    
# Define checkpoint name
checkpoint_name = os.path.join(args.result_model_dir,
                               datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")+'_'+args.result_model_fn + '.pth.tar')

print('Checkpoint name: '+checkpoint_name)    
    
# Train
best_test_loss = float("inf")

def weak_loss(model,batch,normalization='softmax',alpha=30):
    if normalization is None:
        normalize = lambda x: x
    elif normalization=='softmax':     
        normalize = lambda x: torch.nn.functional.softmax(x,1)
    elif normalization=='l1':
        normalize = lambda x: x/(torch.sum(x,dim=1,keepdim=True)+0.0001)

    b = batch['source_image'].size(0)
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
        
    # positive
    start.record()
    corr4d = model(batch)
    mid.record()

    # compute matching scores    
    scores_A = get_scores(corr4d, k = args.k)
    scores_B = get_scores(corr4d, reverse=True, k = args.k)
    score_pos = (scores_A + scores_B)/2
    
    end.record()
    torch.cuda.synchronize()
    model_time = start.elapsed_time(mid)/1000
    loss_time = mid.elapsed_time(end)/1000
    #print('model: {:.2f}: loss: {:.2f}'.format(model_time,loss_time))

    # negative
    batch['source_image']=batch['source_image'][np.roll(np.arange(b),-1),:] # roll
    corr4d = model(batch)
            
    # compute matching scores    
    scores_A = get_scores(corr4d, k = args.k)
    scores_B = get_scores(corr4d, reverse=True, k = args.k)
    score_neg = (scores_A + scores_B)/2
    
    # loss
    loss = score_neg - score_pos    
    return loss

loss_fn = lambda model,batch: weak_loss(model,batch,normalization='softmax')


# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode=='train':            
            optimizer.zero_grad()
        tnf_batch = batch_preprocessing_fn(batch)
        
        loss = loss_fn(model,tnf_batch)
        
        loss_np = loss.item()
        
        epoch_loss += loss_np
        if mode=='train':
            loss.backward()
            optimizer.step()
        else:
            loss=None
        if batch_idx % log_interval == 0:
            print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss_np))
    epoch_loss /= len(dataloader)
    print(mode.capitalize()+' set: Average loss: {:.4f}'.format(epoch_loss))
    return epoch_loss

train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

print('Starting training...')

model.FeatureExtraction.eval()

for epoch in range(1, args.num_epochs+1):
    train_loss[epoch-1] = process_epoch('train',epoch,eval_model_fn,loss_fn,optimizer,dataloader,batch_preprocessing_fn,log_interval=1)
    test_loss[epoch-1] = process_epoch('test',epoch,eval_model_fn,loss_fn,optimizer,dataloader_test,batch_preprocessing_fn,log_interval=1)
      
    # remember best loss
    is_best = test_loss[epoch-1] < best_test_loss
    best_test_loss = min(test_loss[epoch-1], best_test_loss)
    save_checkpoint({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, is_best,checkpoint_name)

print('Done!')
