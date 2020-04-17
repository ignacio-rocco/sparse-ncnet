import torch 
import MinkowskiEngine as ME
from collections import defaultdict
from .point_tnf import normalize_axis
import math
from .knn import knn_faiss
import numpy as np

def sparse_corr(feature_A,
                feature_B,
                k=10,
                coords_A=None,
                coords_B=None,
                reverse=False,
                ratio=False,
                sparse_type='torch',
                return_indx=False,
                fsize = None,
                bidx = None):

    b,ch=feature_B.shape[:2]
    
    if fsize is None:
        hA, wA = feature_A.shape[2:]
        hB, wB = feature_B.shape[2:]
    else:
        hA, wA = fsize
        hB, wB = fsize
        
    feature_A = feature_A.view(b,ch,-1)
    feature_B = feature_B.view(b,ch,-1)

    nA = feature_A.shape[2]
    nB = feature_B.shape[2]
    
    with torch.no_grad():
        dist_squared, indx = knn_faiss(feature_B, feature_A, k)
    
    if bidx is None: bidx = torch.arange(b).view(b,1,1)
    bidx = bidx.expand_as(indx).contiguous()

    if feature_A.requires_grad:
        corr = (feature_A.permute(1,0,2).unsqueeze(2) * \
                feature_B.permute(1,0,2)[:,bidx.view(-1),indx.view(-1)].view(ch,b,k,nA)).sum(dim=0).contiguous()
    else:
        corr = 1-dist_squared/2  # [b,k,nA]
        

    if ratio:
        corr_ratio=corr/corr[:,:1,:]

    if coords_A is None:
        YA,XA=torch.meshgrid(torch.arange(hA),torch.arange(wA))
        YA=YA.contiguous()
        XA=XA.contiguous()
        yA=YA.view(-1).unsqueeze(0).unsqueeze(0).expand(b,k,nA).contiguous().view(-1,1)
        xA=XA.view(-1).unsqueeze(0).unsqueeze(0).expand(b,k,nA).contiguous().view(-1,1)
    else:
        yA,xA = coords_A
        yA=yA.view(-1).unsqueeze(0).unsqueeze(0).expand(b,k,nA).contiguous().view(-1,1)
        xA=xA.view(-1).unsqueeze(0).unsqueeze(0).expand(b,k,nA).contiguous().view(-1,1)

    if coords_B is None:
        YB,XB=torch.meshgrid(torch.arange(hB),torch.arange(wB))
        YB=YB.contiguous()
        XB=XB.contiguous()
        yB=YB.view(-1)[indx.view(-1).cpu()].view(-1,1)
        xB=XB.view(-1)[indx.view(-1).cpu()].view(-1,1)
    else:
        yB,xB = coords_B
        yB=yB.view(-1)[indx.view(-1).cpu()].view(-1,1)
        xB=xB.view(-1)[indx.view(-1).cpu()].view(-1,1)

    bidx = bidx.view(-1,1)
    corr=corr.view(-1,1)
    if ratio: corr_ratio = corr_ratio.view(-1,1)

    if reverse:
        yA,xA,yB,xB=yB,xB,yA,xA
        hA,wA,hB,wB=hB,wB,hA,wA

    if sparse_type == 'me':
        coords = torch.cat((bidx, yA, xA, yB, xB),dim=1).int()
        scorr = ME.SparseTensor(corr, coords)

        if ratio: scorr_ratio = ME.SparseTensor(corr_ratio,coords)

    elif sparse_type == 'torch':
        coords = torch.cat((bidx, yA, xA, yB, xB),dim=1).long().to(corr.device).t()
        scorr = torch.sparse.FloatTensor(coords,corr,torch.Size([b,hA,wA,hB,wB,1]))

        if ratio: scorr_ratio = torch.sparse.FloatTensor(coords,corr_ratio,torch.Size([b,hA,wA,hB,wB,1]))
            
    elif sparse_type == 'raw':
        coords = torch.cat((bidx, yA, xA, yB, xB),dim=1).int()
        scorr = (corr, coords)

        if ratio: scorr_ratio = (corr_ratio,coords)

    else:
        raise ValueError('sparse type {} not recognized'.format(sparse_type))

    if ratio: return scorr, scorr_ratio
    if return_indx: return scorr, indx
    return scorr

def torch_to_me(sten):
    sten = sten.coalesce()
    indices = sten.indices().t().int().cpu()
    
    return ME.SparseTensor(sten.values(), indices)

def me_to_torch(sten):
    values = sten.feats
    indices = sten.coords.t().long().to(values.device)
    
    sten = torch.sparse.FloatTensor(indices,values).coalesce()
    
    return sten

def corr_and_add(
    feature_A,
    feature_B,
    k=10,
    coords_A=None,
    coords_B=None,
    Npts=None):
    
    # compute sparse correlation from A to B
    scorr = sparse_corr(
        feature_A,
        feature_B,
        k=k,
        ratio=False, sparse_type='raw')
    
    # compute sparse correlation from B to A
    scorr2 = sparse_corr(
        feature_B,
        feature_A,
        k=k,
        ratio=False,
        reverse=True, sparse_type='raw')
    
    scorr = ME.SparseTensor(scorr[0],scorr[1])
    scorr2 = ME.SparseTensor(scorr2[0],scorr2[1],coords_manager=scorr.coords_man,force_creation=True)
    
    scorr = ME.MinkowskiUnion()(scorr,scorr2)
    
    return scorr

def transpose_me(sten):
    return ME.SparseTensor(sten.feats.clone(),
                           sten.coords[:,[0,3,4,1,2]].clone())

def transpose_torch(sten):
    sten = sten.coalesce()
    indices = sten.indices()[[0,3,4,1,2],:]
    values = sten.values()
    
    return torch.sparse.FloatTensor(indices,values).coalesce()

def get_scores(corr, reverse=False, k=10):
    if reverse:
        c=[3,4]
    else:
        c=[1,2]

    coords = corr.indices()
    values = corr.values().squeeze().clone()
    
    #knn = KNN(k=k, transpose_mode=False)
    batch_size = coords[:1,:].max()+1
    feature_size = coords[1:,:].max()+1
    
    loss = []
    for b in range(batch_size):
        batch_indices = torch.nonzero(coords[0,:].view(-1)==b).view(-1)
        ref = coords[c,:][:,batch_indices]
        uniq_coords = torch.unique(ref,dim=1)
        dist, idx = knn_faiss(ref.unsqueeze(0).float(), uniq_coords.unsqueeze(0).float(), k)
        #dist, idx = knn(ref.unsqueeze(0), uniq_coords.unsqueeze(0))
        mask = (dist == 0)
        #import pdb; pdb.set_trace()
        curr_vals = values[batch_indices[idx]]*mask
        zeros = torch.zeros((1,feature_size**2-curr_vals.shape[1],curr_vals.shape[2]),device=curr_vals.device)
        curr_vals_extended = torch.cat((curr_vals,zeros),dim=1)
        max_vals = torch.softmax(curr_vals_extended,dim=1).max(dim=1)[0].mean().unsqueeze(0)
        loss.append(max_vals)
                
    scores = torch.cat(loss,dim=0).mean()
    
    return scores


def unique(ar, return_index=False, return_inverse=False,
           return_counts=False):

    ar = ar.view(-1)

    perm = ar.argsort()
    aux = ar[perm]

    mask = torch.zeros(aux.shape, dtype=torch.bool)
    mask[0] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = aux[mask]
    if not return_index and not return_inverse and not return_counts:
        return ret

    ret = ret,
    if return_index:
        ret += perm[mask],
    if return_inverse:
        imask = torch.cumsum(mask) - 1
        inv_idx = torch.zeros(mask.shape, dtype=torch.int64)
        inv_idx[perm] = imask
        ret += inv_idx,
    if return_counts:
        nonzero = torch.nonzero(mask)[0]
        idx = torch.zeros(nonzero.shape[0] + 1, dtype=nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],
    return ret
    

def get_matches(out, reverse=True, fsize=40, scale='centered'):
    if isinstance(fsize,tuple):
        fs1, fs2, fs3, fs4 = fsize
    else:
        fs1, fs2, fs3, fs4 = fsize, fsize, fsize, fsize
        
    if reverse:
        c=[3,4]
        fh, fw = fs3, fs4
    else:
        c=[1,2]
        fh, fw = fs1, fs2

    coords = out.coords[:,c].cuda()
    feats = out.feats
    sorted_idx = torch.argsort(-feats,dim=0).view(-1)
    coords = coords[sorted_idx]
    
    coords_idx = coords[:,0]*fw+coords[:,1]
    _, matches_idx = unique(coords_idx, return_index=True)
    matches_idx = sorted_idx[matches_idx]
    
    matches_scores = feats[matches_idx].t()
    matches = out.coords.to(out.device)[matches_idx,1:]
            
    if scale=='centered':
        yA = normalize_axis(matches[:,0]+1,fs1).unsqueeze(0).to(out.device)
        xA = normalize_axis(matches[:,1]+1,fs2).unsqueeze(0).to(out.device)
        yB = normalize_axis(matches[:,2]+1,fs3).unsqueeze(0).to(out.device)
        xB = normalize_axis(matches[:,3]+1,fs4).unsqueeze(0).to(out.device)
    elif scale=='positive':
        yA = (matches[:,0].float()/(fs1-1)).unsqueeze(0).to(out.device)
        xA = (matches[:,1].float()/(fs2-1)).unsqueeze(0).to(out.device)
        yB = (matches[:,2].float()/(fs3-1)).unsqueeze(0).to(out.device)
        xB = (matches[:,3].float()/(fs4-1)).unsqueeze(0).to(out.device) 
    elif scale=='none':
        yA = (matches[:,0].float()).unsqueeze(0).to(out.device)
        xA = (matches[:,1].float()).unsqueeze(0).to(out.device)
        yB = (matches[:,2].float()).unsqueeze(0).to(out.device)
        xB = (matches[:,3].float()).unsqueeze(0).to(out.device)
    
    return xA,yA,xB,yB,matches_scores


def get_matches_both_dirs(corr4d, fs1, fs2, fs3, fs4):
    corr4d = torch_to_me(corr4d)
    (xA_,yA_,xB_,yB_,score_)=get_matches(corr4d, fsize=(fs1, fs2, fs3, fs4), reverse=False, scale='none')
     
    (xA2_,yA2_,xB2_,yB2_,score2_)=get_matches(corr4d, fsize=(fs1, fs2, fs3, fs4), reverse=True, scale='none')
    # fuse matches
    xA_=torch.cat((xA_,xA2_),1)
    yA_=torch.cat((yA_,yA2_),1)
    xB_=torch.cat((xB_,xB2_),1)
    yB_=torch.cat((yB_,yB2_),1)
    score_=torch.cat((score_,score2_),1)
    # remove duplicates
    all_matches = torch.cat((xA_,yA_,xB_,yB_),dim=0)
    _, matches_idx = np.unique(all_matches.cpu().numpy(),axis=1,return_index=True)
    score_ = score_[:,matches_idx]
    xA_,yA_,xB_,yB_ = all_matches[:1,matches_idx], all_matches[1:2,matches_idx], all_matches[2:3,matches_idx], all_matches[3:,matches_idx]
    return xA_, yA_, xB_, yB_, score_
