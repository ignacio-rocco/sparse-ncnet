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
from lib.sparse import get_matches_both_dirs, torch_to_me, me_to_torch, unique
from lib.relocalize import relocalize, relocalize_soft, eval_model_reloc

import numpy as np
import numpy.random
from scipy.io import loadmat
from scipy.io import savemat

import argparse

print('Sparse-NCNet evaluation script - HPatches Sequences dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='../trained_models/sparsencnet_k10.pth.tar')
parser.add_argument('--hseq_path', type=str, default='../datasets/hpatches/hpatches-sequences-release')
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
                   return_fs=True,
                   change_stride=args.change_stride
                  )

scale_factor = 0.0625
if args.relocalize==1:
    scale_factor = scale_factor/2
if args.change_stride==1:
    scale_factor = scale_factor*2

# Get shortlists for each query image
dataset_path=args.hseq_path
seq_names = sorted(os.listdir(dataset_path))

seq_names=np.array(seq_names)
seq_names_split = np.array_split(seq_names,args.nchunks)
seq_names_chunk = seq_names_split[args.chunk_idx]

seq_names_chunk=list(seq_names_chunk)
if args.skip_up_to!='':
    seq_names_chunk = seq_names_chunk[seq_names_chunk.index(args.skip_up_to)+1:]

if args.benchmark:
    start = torch.cuda.Event(enable_timing=True)
    match = torch.cuda.Event(enable_timing=True)
    reloc = torch.cuda.Event(enable_timing=True)
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
        src_fn = os.path.join(args.hseq_path,seq_name,'1.ppm')
        src=imreadth(src_fn)
        hA,wA=src.shape[-2:]
        src=resize(normalize(src), args.image_size, scale_factor)
        hA_,wA_=src.shape[-2:]

        tgt_fn = os.path.join(args.hseq_path,seq_name,'{}.ppm'.format(idx))
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
#             if args.relocalize:
#                 N_matches = min(int(args.Npts*1.25), matches_idx_sorted.shape[0])
#             else:
#                 N_matches = min(args.Npts, matches_idx_sorted.shape[0])
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
            fs1,fs2,fs3,fs4=2*fs1,2*fs2,2*fs3,2*fs4
            # relocalization stage 1:
            if args.reloc_type.startswith('hard'):
                xA_, yA_, xB_, yB_, score_ = relocalize(xA_,
                                                        yA_,
                                                        xB_,
                                                        yB_,
                                                        score_,
                                                        feature_A_2x,
                                                        feature_B_2x,
                                                        crop_size=args.reloc_hard_crop_size)
                if args.reloc_hard_crop_size==3:
                    _,uidx = unique(yA_.double()*fs2*fs3*fs4+xA_.double()*fs3*fs4+yB_.double()*fs4+xB_.double(),return_index=True)
                    xA_=xA_[:,uidx]
                    yA_=yA_[:,uidx]
                    xB_=xB_[:,uidx]
                    yB_=yB_[:,uidx]
                    score_=score_[:,uidx]
            elif args.reloc_type=='soft':
                xA_, yA_, xB_, yB_, score_ = relocalize_soft(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x)
                    
            # relocalization stage 2:
            if args.reloc_type=='hard_soft':
                xA_, yA_, xB_, yB_, score_ = relocalize_soft(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x, upsample_positions=False)
            
            elif args.reloc_type=='hard_hard':
                xA_, yA_, xB_, yB_, score_ = relocalize(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x, upsample_positions=False)
                
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


        xA = xA_.view(-1).data.cpu().float().numpy()*wA
        yA = yA_.view(-1).data.cpu().float().numpy()*hA
        xB = xB_.view(-1).data.cpu().float().numpy()*wB
        yB = yB_.view(-1).data.cpu().float().numpy()*hB
        score = score_.view(-1).data.cpu().float().numpy()

        keypoints_A=np.stack((xA,yA),axis=1)
        keypoints_B=np.stack((xB,yB),axis=1)
                    
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

        del corr4d,delta4d,src,tgt, feature_A_2x, feature_B_2x
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
