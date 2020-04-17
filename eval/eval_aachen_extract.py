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

print('Sparse-NCNet evaluation script - Aachen dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='../trained_models/sparsencnet_k10.pth.tar')
parser.add_argument('--aachen_path', type=str, default='../datasets/Aachen-Day-Night')
parser.add_argument('--k_size', type=int, default=1)
parser.add_argument('--image_size', type=int, default=3200)
parser.add_argument('--experiment_name', type=str, default='sparsencnet_3200_hard')
parser.add_argument('--symmetric_mode', type=str_to_bool, default=True)
parser.add_argument('--nchunks', type=int, default=1)
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--skip_up_to', type=str, default='')
parser.add_argument('--relocalize', type=int, default=1)
parser.add_argument('--reloc_type', type=str, default='hard')
parser.add_argument('--change_stride', type=int, default=1)
parser.add_argument('--benchmark', type=int, default=0)
parser.add_argument('--no_ncnet', type=int, default=0)
parser.add_argument('--Npts', type=int, default=8000)
parser.add_argument('--image_pairs', type=str, default='all')

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

try:
    os.mkdir(os.path.join(args.aachen_path,'matches'))
except FileExistsError:
    pass

try:
    os.mkdir(os.path.join(args.aachen_path,'matches',args.experiment_name))
except FileExistsError:
    pass

# Get shortlists for each query image
if args.image_pairs=='all':
    pair_names_fn = os.path.join(args.aachen_path,'image_pairs_to_match.txt')
elif args.image_pairs=='queries':
    pair_names_fn = os.path.join(args.aachen_path,'query_pairs_to_match.txt')
    
with open(pair_names_fn) as f:
    pair_names = [line.rstrip('\n') for line in f]

pair_names=np.array(pair_names)
pair_names_split = np.array_split(pair_names,args.nchunks)
pair_names_chunk = pair_names_split[args.chunk_idx]

pair_names_chunk=list(pair_names_chunk)
if args.skip_up_to!='':
    pair_names_chunk = pair_names_chunk[pair_names_chunk.index(args.skip_up_to)+1:]

if args.benchmark:
    start = torch.cuda.Event(enable_timing=True)
    match = torch.cuda.Event(enable_timing=True)
    reloc = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    pair_names_chunk = [pair_names_chunk[0]]
    indices = [2 for i in range(21)]
    first_iter=True
else:
    indices = range(2, 7)

for pair in pair_names_chunk:
    src_fn = os.path.join(args.aachen_path,'images','images_upright',pair.split(' ')[0])
    src=imreadth(src_fn)
    hA,wA=src.shape[-2:]
    src=resize(normalize(src), args.image_size, scale_factor)
    hA_,wA_=src.shape[-2:]

    tgt_fn = os.path.join(args.aachen_path,'images','images_upright',pair.split(' ')[1])
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
            elif args.reloc_type=='hard_hard':
                xA_, yA_, xB_, yB_, score_ = relocalize(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x)
                xA_, yA_, xB_, yB_, score_ = relocalize(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x, upsample_positions=False)
                
            fs1,fs2,fs3,fs4=2*fs1,2*fs2,2*fs3,2*fs4

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

    YA,XA=torch.meshgrid(torch.arange(fs1),torch.arange(fs2))
    YB,XB=torch.meshgrid(torch.arange(fs3),torch.arange(fs4))
    
    YA = YA.contiguous()
    XA = XA.contiguous()
    YB = YB.contiguous()
    XB = XB.contiguous()
    
    YA=(YA+0.5)/(fs1)*hA
    XA=(XA+0.5)/(fs2)*wA
    YB=(YB+0.5)/(fs3)*hB
    XB=(XB+0.5)/(fs4)*wB
    
    XA = XA.view(-1).data.cpu().float().numpy()
    YA = YA.view(-1).data.cpu().float().numpy()
    XB = XB.view(-1).data.cpu().float().numpy()
    YB = YB.view(-1).data.cpu().float().numpy()

    keypoints_A=np.stack((XA,YA),axis=1)
    keypoints_B=np.stack((XB,YB),axis=1)
    
#     idx_A = (yA_*fs2+xA_).long().view(-1,1)
#     idx_B = (yB_*fs4+xB_).long().view(-1,1)
    idx_A = (yA_*fs2+xA_).view(-1,1)
    idx_B = (yB_*fs4+xB_).view(-1,1)
    score = score_.view(-1,1)
    
    matches = torch.cat((idx_A,idx_B,score),dim=1).cpu().numpy()
    
    kp_A_fn = src_fn+'.'+args.experiment_name
    kp_B_fn = tgt_fn+'.'+args.experiment_name
    
    if not args.benchmark and not os.path.exists(kp_A_fn):
        with open(kp_A_fn, 'wb') as output_file:
            np.savez(output_file,keypoints=keypoints_A)
            
    if not args.benchmark and not os.path.exists(kp_B_fn):
        with open(kp_B_fn, 'wb') as output_file:
            np.savez(output_file,keypoints=keypoints_B)
            
    matches_fn = pair.replace('/','-').replace(' ','--')+'.'+args.experiment_name
    matches_path = os.path.join(args.aachen_path,'matches',args.experiment_name,matches_fn)
        
    if not args.benchmark:
        with open(matches_path, 'wb') as output_file:
            np.savez(output_file,matches=matches)
        print(matches_fn)

    del corr4d,delta4d,src,tgt, feature_A_2x, feature_B_2x
    del xA_,xB_,yA_,yB_,score_
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
        
if args.benchmark:
    print('{}x{},{:.4f},{:.4f}'.format(
    wA_,
    hA_,
    torch.tensor(ttime).mean(),
    torch.tensor(mmem).mean()))