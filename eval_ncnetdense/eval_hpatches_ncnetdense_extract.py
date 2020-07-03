import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

import os
from os.path import exists, join, basename
from collections import OrderedDict

import sys
sys.path.append('..')

from lib.model import ImMatchNet, MutualMatching
from lib.normalization import NormalizeImageDict
from lib.torch_util import str_to_bool
from lib.point_tnf import normalize_axis,unnormalize_axis,corr_to_matches
from lib.plot import plot_image

import numpy as np
import numpy.random
from skimage.io import imread
from scipy.io import loadmat
from scipy.io import savemat

import argparse

print('NCNetDense evaluation script - HSequences dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, default='../trained_models/ncnet_ivd.pth.tar')
parser.add_argument('--hseq_path', type=str, default='../datasets/hpatches/hpatches-sequences-release')
parser.add_argument('--k_size', type=int, default=2)
parser.add_argument('--image_size', type=int, default=1600)
parser.add_argument('--softmax', type=str_to_bool, default=False)
parser.add_argument('--matching_both_directions', type=str_to_bool, default=True)
parser.add_argument('--flip_matching_direction', type=str_to_bool, default=False)
parser.add_argument('--experiment_name', type=str, default='ncnet_resnet101_3200k2_softmax0')
parser.add_argument('--symmetric_mode', type=str_to_bool, default=True)
parser.add_argument('--nchunks', type=int, default=1)
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--skip_up_to', type=str, default='')
parser.add_argument('--feature_extraction_cnn', type=str, default='resnet101')
parser.add_argument('--change_stride', type=int, default=1)
parser.add_argument('--benchmark', type=int, default=0)

args = parser.parse_args()

image_size = args.image_size
k_size = args.k_size
matching_both_directions = args.matching_both_directions
flip_matching_direction = args.flip_matching_direction

# Load pretrained model
half_precision=True # use for memory saving

print(args)
    
model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=args.checkpoint,
                   half_precision=half_precision,
                   feature_extraction_cnn=args.feature_extraction_cnn,
                   relocalization_k_size=args.k_size,
                   symmetric_mode=args.symmetric_mode)

if args.change_stride:
    scale_factor = 0.0625
#    import pdb;pdb.set_trace()
    model.FeatureExtraction.model[-1][0].conv1.stride=(1,1)
    model.FeatureExtraction.model[-1][0].conv2.stride=(1,1)
    model.FeatureExtraction.model[-1][0].downsample[0].stride=(1,1)
else:
    scale_factor = 0.0625/2

imreadth = lambda x: torch.Tensor(imread(x).astype(np.float32)).transpose(1,2).transpose(0,1)
normalize = lambda x: NormalizeImageDict(['im'])({'im':x})['im']

# allow rectangular images. Does not modify aspect ratio.
if k_size==1:
    resize = lambda x: nn.functional.upsample(Variable(x.unsqueeze(0).cuda(),volatile=True),
                size=(int(x.shape[1]/(np.max(x.shape[1:])/image_size)),int(x.shape[2]/(np.max(x.shape[1:])/image_size))),mode='bilinear')
else:
    resize = lambda x: nn.functional.upsample(Variable(x.unsqueeze(0).cuda(),volatile=True),
                size=(int(np.floor(x.shape[1]/(np.max(x.shape[1:])/image_size)*scale_factor/k_size)/scale_factor*k_size),
                      int(np.floor(x.shape[2]/(np.max(x.shape[1:])/image_size)*scale_factor/k_size)/scale_factor*k_size)),mode='bilinear')

padim = lambda x,h_max: torch.cat((x,x.view(-1)[0].clone().expand(1,3,h_max-x.shape[2],x.shape[3])/1e20),dim=2) if x.shape[2]<h_max else x

# Get shortlists for each query image
dataset_path=args.hseq_path
seq_names = sorted(os.listdir(dataset_path))

N=int((image_size*scale_factor/k_size)*np.floor((image_size*scale_factor/k_size)*(3/4)))
if matching_both_directions:
    N=2*N
    
do_softmax = args.softmax

plot=False

seq_names=np.array(seq_names)
seq_names_split = np.array_split(seq_names,args.nchunks)
seq_names_chunk = seq_names_split[args.chunk_idx]

seq_names_chunk=list(seq_names_chunk)
if args.skip_up_to!='':
    seq_names_chunk = seq_names_chunk[seq_names_chunk.index(args.skip_up_to)+1:]

if args.benchmark:
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    seq_names_chunk = [seq_names_chunk[0]]
    indices = [2 for i in range(21)]
    first_iter=True
else:
    indices = range(2, 7)
    
for seq_name in seq_names_chunk:
    # load query image
    
    # load database image
    for idx in indices:
        #import pdb;pdb.set_trace()
        src_fn = os.path.join(args.hseq_path,seq_name,'1.ppm')
        src=imreadth(src_fn)
        hA,wA=src.shape[-2:]
        src=resize(normalize(src))
        hA_,wA_=src.shape[-2:]
    
        tgt_fn = os.path.join(args.hseq_path,seq_name,'{}.ppm'.format(idx))
        tgt=imreadth(tgt_fn)
        hB,wB=tgt.shape[-2:]
        tgt=resize(normalize(tgt))        
        
        if args.benchmark:
            start.record()
            
        with torch.no_grad():
            if k_size>1:
                corr4d,delta4d=model({'source_image':src,'target_image':tgt})
            else:
                corr4d=model({'source_image':src,'target_image':tgt})
                delta4d=None
            if args.benchmark:
                mid.record()
            
            # reshape corr tensor and get matches for each point in image B
            batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()

            # pad image and plot
            if plot:
                h_max=int(np.max([src.shape[2],tgt.shape[2]]))
                im=plot_image(torch.cat((padim(src,h_max),padim(tgt,h_max)),dim=3),return_im=True)
                plt.imshow(im)

            if matching_both_directions:
                (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size)
                (xA2_,yA2_,xB2_,yB2_,score2_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size,invert_matching_direction=True)
                xA_=torch.cat((xA_,xA2_),1)
                yA_=torch.cat((yA_,yA2_),1)
                xB_=torch.cat((xB_,xB2_),1)
                yB_=torch.cat((yB_,yB2_),1)
                score_=torch.cat((score_,score2_),1)
                # sort in descending score (this will keep the max-score instance in the duplicate removal step)
                sorted_index=torch.sort(-score_)[1].squeeze()
                xA_=xA_.squeeze()[sorted_index].unsqueeze(0)
                yA_=yA_.squeeze()[sorted_index].unsqueeze(0)
                xB_=xB_.squeeze()[sorted_index].unsqueeze(0)
                yB_=yB_.squeeze()[sorted_index].unsqueeze(0)
                score_=score_.squeeze()[sorted_index].unsqueeze(0)
                # remove duplicates
                concat_coords=np.concatenate((xA_.cpu().data.numpy(),yA_.cpu().data.numpy(),xB_.cpu().data.numpy(),yB_.cpu().data.numpy()),0)
                _,unique_index=np.unique(concat_coords,axis=1,return_index=True)
                xA_=xA_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
                yA_=yA_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
                xB_=xB_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
                yB_=yB_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
                score_=score_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
            elif flip_matching_direction:
                (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size,invert_matching_direction=True)
            else:
                (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size)

            # recenter
            if k_size>1:
                yA_=yA_*(fs1*k_size-1)/(fs1*k_size)+0.5/(fs1*k_size)
                xA_=xA_*(fs2*k_size-1)/(fs2*k_size)+0.5/(fs2*k_size)
                yB_=yB_*(fs3*k_size-1)/(fs3*k_size)+0.5/(fs3*k_size)
                xB_=xB_*(fs4*k_size-1)/(fs4*k_size)+0.5/(fs4*k_size)    
            else:
                yA_=yA_*(fs1-1)/fs1+0.5/fs1
                xA_=xA_*(fs2-1)/fs2+0.5/fs2
                yB_=yB_*(fs3-1)/fs3+0.5/fs3
                xB_=xB_*(fs4-1)/fs4+0.5/fs4
                
        if args.benchmark:
            end.record()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)/1000
            processing_time = start.elapsed_time(mid)/1000
            post_processing_time = mid.elapsed_time(end)/1000
            max_mem = torch.cuda.max_memory_allocated()/1024/1024
            if first_iter:
                first_iter=False
                ttime = []
                mmem = []
            else:
                ttime.append(total_time)
                mmem.append(max_mem)
            print('cnn: {:.2f}, pp: {:.2f}, total: {:.2f}, max mem: {:.2f}MB'.format(processing_time,
                                                               post_processing_time,
                                                               total_time,
                                                               max_mem))
        
        xA = xA_.view(-1).data.cpu().float().numpy()*wA
        yA = yA_.view(-1).data.cpu().float().numpy()*hA
        xB = xB_.view(-1).data.cpu().float().numpy()*wB
        yB = yB_.view(-1).data.cpu().float().numpy()*hB
        score = score_.view(-1).data.cpu().float().numpy()
        
        keypoints_A=np.stack((xA,yA),axis=1)
        keypoints_B=np.stack((xB,yB),axis=1)
        
        Npts=len(xA)
        if Npts>0:
            # plot top N matches
            if plot:
                c=numpy.random.rand(Npts,3)
                for i in range(Npts):       
                    if score[i]>0.75:
                        ax = plt.gca()
                        ax.add_artist(plt.Circle((float(xA[i])*src.shape[3],float(yA[i])*src.shape[2]), radius=3, color=c[i,:]))
                        ax.add_artist(plt.Circle((float(xB[i])*tgt.shape[3]+src.shape[3] ,float(yB[i])*tgt.shape[2]), radius=3, color=c[i,:]))


        matches_file = '{}/{}_{}.npz.{}'.format(seq_name,'1',idx,args.experiment_name)
        
        if not args.benchmark:
            with open(os.path.join(args.hseq_path,matches_file), 'wb') as output_file:
                np.savez(
                        output_file,
                        keypoints_A=keypoints_A,
                        keypoints_B=keypoints_B,
                        scores=score
                )

            print(matches_file)
            
        del corr4d,delta4d,src,tgt
        del xA,xB,yA,yB,score
        del xA_,xB_,yA_,yB_,score_
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

if args.benchmark:
    print('{}x{},{:.4f},{:.4f}'.format(
    wA_,
    hA_,
    torch.tensor(ttime).mean(),
    torch.tensor(mmem).mean()))