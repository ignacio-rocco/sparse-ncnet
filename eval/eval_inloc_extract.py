import faiss
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
from os.path import exists, join, basename
from collections import OrderedDict

import sys
sys.path.append('..')

from lib.model import ImMatchNet, MutualMatching
from lib.normalization import imreadth, resize, normalize
from lib.torch_util import str_to_bool
from lib.point_tnf import normalize_axis,unnormalize_axis,corr_to_matches
from lib.sparse import get_matches_both_dirs, torch_to_me, me_to_torch
from lib.relocalize import relocalize, relocalize_soft, eval_model_reloc

import numpy as np
import numpy.random
from scipy.io import loadmat
from scipy.io import savemat

import argparse

print('Sparse-NCNet evaluation script - InLoc dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='../trained_models/sparsencnet_k10.pth.tar')
parser.add_argument('--inloc_shortlist', type=str, default='../datasets/inloc/shortlists/densePE_top100_shortlist_cvpr18.mat')
parser.add_argument('--pano_path', type=str, default='../datasets/inloc/', help='path to InLoc panos - should contain CSE3,CSE4,CSE5,DUC1 and DUC2 folders')
parser.add_argument('--query_path', type=str, default='../datasets/inloc/query/iphone7/', help='path to InLoc queries')
parser.add_argument('--k_size', type=int, default=1)
parser.add_argument('--image_size', type=int, default=3200)
parser.add_argument('--experiment_name', type=str, default='sparsencnet_3200_hard_soft')
parser.add_argument('--symmetric_mode', type=str_to_bool, default=True)
parser.add_argument('--nchunks', type=int, default=1)
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--skip_up_to', type=str, default='')
parser.add_argument('--relocalize', type=int, default=1)
parser.add_argument('--reloc_type', type=str, default='hard_soft')
parser.add_argument('--reloc_hard_crop_size', type=int, default=2)
parser.add_argument('--change_stride', type=int, default=1)
parser.add_argument('--benchmark', type=int, default=0)
parser.add_argument('--no_ncnet', type=int, default=0)
parser.add_argument('--Npts', type=int, default=2000)
parser.add_argument('--n_queries', type=int, default=356)
parser.add_argument('--n_panos', type=int, default=10)

args = parser.parse_args()

print(args)

chp_args = torch.load(args.checkpoint)['args']
model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=args.checkpoint,
                   ncons_kernel_sizes=chp_args.ncons_kernel_sizes,
                   ncons_channels=chp_args.ncons_channels,
                   sparse=True,
                   symmetric_mode=bool(chp_args.symmetric_mode),
                   feature_extraction_cnn=chp_args.feature_extraction_cnn,
                   bn=bool(chp_args.bn),
                   k=chp_args.k,
                   return_fs=True)

# Generate output folder path
output_folder = args.inloc_shortlist.split('/')[-1].split('.')[0]+'_'+args.experiment_name
print('Output matches folder: '+output_folder)

scale_factor = 0.0625
if args.relocalize==1:
    scale_factor = scale_factor/2
if args.change_stride==1:
    scale_factor = scale_factor*2
elif args.change_stride==2:
    scale_factor = scale_factor*4
    
if args.change_stride>=1:
    model.FeatureExtraction.model[-1][0].conv1.stride=(1,1)
    model.FeatureExtraction.model[-1][0].conv2.stride=(1,1)
    model.FeatureExtraction.model[-1][0].downsample[0].stride=(1,1)
if args.change_stride>=2:
    model.FeatureExtraction.model[-2][0].conv1.stride=(1,1)
    model.FeatureExtraction.model[-2][0].conv2.stride=(1,1)
    model.FeatureExtraction.model[-2][0].downsample[0].stride=(1,1)

# Get shortlists for each query image
shortlist_fn = args.inloc_shortlist

dbmat = loadmat(shortlist_fn)
db = dbmat['ImgList'][0,:]

query_fn_all=np.squeeze(np.vstack(tuple([db[q][0] for q in range(len(db))])))
pano_fn_all=np.vstack(tuple([db[q][1] for q in range(len(db))]))

Nqueries=args.n_queries
Npanos=args.n_panos

try:
    os.mkdir('../datasets/inloc/matches/')
except FileExistsError:
    pass

try:
    os.mkdir('../datasets/inloc/matches/'+output_folder)
except FileExistsError:
    pass

queries_idx = np.arange(Nqueries)
queries_idx_split = np.array_split(queries_idx,args.nchunks)
queries_idx_chunk = queries_idx_split[args.chunk_idx]

queries_idx_chunk=list(queries_idx_chunk)

if args.skip_up_to!='':
    queries_idx_chunk = queries_idx_chunk[queries_idx_chunk.index(args.skip_up_to)+1:]

if args.benchmark:
    start = torch.cuda.Event(enable_timing=True)
    match = torch.cuda.Event(enable_timing=True)
    reloc = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    queries_idx_chunk = [queries_idx_chunk[0]]
    indices = [0 for i in range(21)]
    first_iter=True
else:
    indices = range(Npanos)

for q in queries_idx_chunk:
    print(q)
    matches=numpy.zeros((1,Npanos,args.Npts,5))
    # load query image
    src_fn = os.path.join(args.query_path,db[q][0].item())
    src=imreadth(src_fn)
    hA,wA=src.shape[-2:]
    src=resize(normalize(src), args.image_size, scale_factor)
    hA_,wA_=src.shape[-2:]
    
    # load database image
    for idx in indices:
        tgt_fn = os.path.join(args.pano_path,db[q][1].ravel()[idx].item())
        tgt=imreadth(tgt_fn)
        hB,wB=tgt.shape[-2:]
        tgt=resize(normalize(tgt), args.image_size, scale_factor)
        hB_,wB_=tgt.shape[-2:]

        if args.benchmark:
            start.record()
            
        with torch.no_grad():
            if args.benchmark:
                corr4d, feature_A_2x, feature_B_2x, fs1, fs2, fs3, fs4, fe_time, cnn_time = eval_model_reloc(
                    model,
                    {'source_image':src,
                     'target_image':tgt},
                    args
                )
            else:
                corr4d, feature_A_2x, feature_B_2x, fs1, fs2, fs3, fs4 = eval_model_reloc(
                    model,
                    {'source_image':src,
                     'target_image':tgt},
                    args
                )

            delta4d=None
            if args.benchmark:
                match.record()

            xA_, yA_, xB_, yB_, score_ = get_matches_both_dirs(corr4d, fs1, fs2, fs3, fs4)
            
            if args.Npts is not None:
                matches_idx_sorted = torch.argsort(-score_.view(-1))
                N_matches = min(args.Npts, matches_idx_sorted.shape[0])
                matches_idx_sorted = matches_idx_sorted[:N_matches]
                score_ = score_[:,matches_idx_sorted]
                xA_ = xA_[:,matches_idx_sorted]
                yA_ = yA_[:,matches_idx_sorted]
                xB_ = xB_[:,matches_idx_sorted]
                yB_ = yB_[:,matches_idx_sorted]
            
            if args.benchmark:
                reloc.record()
                
            if args.relocalize:
                if args.reloc_type=='hard':
                    xA_, yA_, xB_, yB_, score_ = relocalize(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x)
                elif args.reloc_type=='hard_soft':
                    xA_, yA_, xB_, yB_, score_ = relocalize(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x)
                    xA_, yA_, xB_, yB_, score_ = relocalize_soft(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x, upsample_positions=False)
                elif args.reloc_type=='soft':
                    xA_, yA_, xB_, yB_, score_ = relocalize_soft(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x, upsample_positions=True)
                    
                fs1,fs2,fs3,fs4=2*fs1,2*fs2,2*fs3,2*fs4
                
            yA_=(yA_+0.5)/(fs1)
            xA_=(xA_+0.5)/(fs2)
            yB_=(yB_+0.5)/(fs3)
            xB_=(xB_+0.5)/(fs4)

        if args.benchmark:
            end.record()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)/1000
            processing_time = start.elapsed_time(match)/1000
            match_processing_time = match.elapsed_time(reloc)/1000
            reloc_processing_time = reloc.elapsed_time(end)/1000
            max_mem = torch.cuda.max_memory_allocated()/1024/1024
            if first_iter:
                first_iter=False
                ttime = []
                mmem = []
            else:
                ttime.append(total_time)
                mmem.append(max_mem)
            print('fe: {:.2f}, cnn: {:.2f}, pp: {:.2f}, reloc: {:.2f}, total: {:.2f}, max mem: {:.2f}MB'.format(fe_time, cnn_time,
                                                               match_processing_time,
                                                               reloc_processing_time,
                                                               total_time,
                                                               max_mem))


        xA = xA_.view(-1).data.cpu().float().numpy()
        yA = yA_.view(-1).data.cpu().float().numpy()
        xB = xB_.view(-1).data.cpu().float().numpy()
        yB = yB_.view(-1).data.cpu().float().numpy()
        score = score_.view(-1).data.cpu().float().numpy()
        
        matches[0,idx,:,0]=xA
        matches[0,idx,:,1]=yA
        matches[0,idx,:,2]=xB
        matches[0,idx,:,3]=yB
        matches[0,idx,:,4]=score

        del corr4d,delta4d,tgt, feature_A_2x, feature_B_2x
        del xA,xB,yA,yB,score
        del xA_,xB_,yA_,yB_,score_
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        
        print(">>>"+str(idx))
        
    if not args.benchmark:
        matches_file=os.path.join('matches/',output_folder,str(q+1)+'.mat')
        savemat(matches_file,{'matches':matches,'query_fn':db[q][0].item(),'pano_fn':pano_fn_all},do_compression=True)
        print(matches_file)
    del src
        
if args.benchmark:
    print('{}x{},{:.4f},{:.4f}'.format(
    wA_,
    hA_,
    torch.tensor(ttime).mean(),
    torch.tensor(mmem).mean()))