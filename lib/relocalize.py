import torch
import torchvision.ops as O
import torch.nn.functional as F
from lib.model import featureL2Norm, corr_dense
from lib.sparse import corr_and_add

def relocalize(xA_,yA_,xB_,yB_,score_,feature_A_2x, feature_B_2x, N_matches=None, upsample_positions=True, crop_size = 2):
    assert crop_size==3 or crop_size==2
    
    if N_matches is None:
        N_matches = xA_.shape[1]
    else:
        idx = torch.argsort(-score_.view(-1))
        N_matches = min(N_matches, idx.shape[0])
        idx = idx[:N_matches]
        score_ = score_[:,idx]
        xA_ = xA_[:,idx]
        yA_ = yA_[:,idx]
        xB_ = xB_[:,idx]
        yB_ = yB_[:,idx]
    
    if upsample_positions:
        xA_ = xA_*2
        yA_ = yA_*2
        xB_ = xB_*2
        yB_ = yB_*2

    coords_A = torch.cat(
        (torch.zeros(1,N_matches).to(xA_.device),
        xA_-(crop_size%2),
        yA_-(crop_size%2),
        xA_+1,
        yA_+1),
        dim = 0
        ).t()

    coords_B = torch.cat(
        (torch.zeros(1,N_matches).to(xB_.device),
        xB_-(crop_size%2),
        yB_-(crop_size%2),
        xB_+1,
        yB_+1),
        dim = 0
        ).t()
    
    ch = feature_A_2x.shape[1]
    feature_A_local = O.roi_pool(feature_A_2x,
                                 coords_A,
                                 output_size=(crop_size,crop_size)).view(N_matches,ch,-1,1)
    feature_B_local = O.roi_pool(feature_B_2x,
                                 coords_B,
                                 output_size=(crop_size,crop_size)).view(N_matches,ch,1,-1)
    
    deltaY, deltaX = torch.meshgrid(torch.linspace(-(crop_size%2),1,crop_size),
                                    torch.linspace(-(crop_size%2),1,crop_size))
    
    deltaX = deltaX.contiguous().view(-1).to(xA_.device)
    deltaY = deltaY.contiguous().view(-1).to(xA_.device)
    
    corr_local = (feature_A_local * feature_B_local).sum(dim=1)
    
    delta_A_idx = torch.argmax(corr_local.max(dim=2,keepdim=True)[0],dim=1)
    delta_B_idx = torch.argmax(corr_local.max(dim=1,keepdim=True)[0],dim=2)
    
    xA_ = xA_ + deltaX[delta_A_idx].t()
    yA_ = yA_ + deltaY[delta_A_idx].t()
    xB_ = xB_ + deltaX[delta_B_idx].t()
    yB_ = yB_ + deltaY[delta_B_idx].t()
    
    return xA_, yA_, xB_, yB_, score_

def relocalize_soft(xA_,yA_,xB_,yB_,score_, feature_A_2x, feature_B_2x, N_matches=None, sigma=10, upsample_positions=True):
    if N_matches is None:
        N_matches = xA_.shape[1]
    else:
        idx = torch.argsort(-score_.view(-1))
        N_matches = min(N_matches, idx.shape[0])
        idx = idx[:N_matches]
        score_ = score_[:,idx]
        xA_ = xA_[:,idx]
        yA_ = yA_[:,idx]
        xB_ = xB_[:,idx]
        yB_ = yB_[:,idx]
    
    if upsample_positions:
        xA_ = xA_*2
        yA_ = yA_*2
        xB_ = xB_*2
        yB_ = yB_*2
        
    coords_A = torch.cat(
        (torch.zeros(1,N_matches).to(xA_.device),
        xA_-1,
        yA_-1,
        xA_+1,
        yA_+1),
        dim = 0
        ).t()

    coords_B = torch.cat(
        (torch.zeros(1,N_matches).to(xB_.device),
        xB_-1,
        yB_-1,
        xB_+1,
        yB_+1),
        dim = 0
        ).t()

    ch = feature_A_2x.shape[1]
    feature_A_local = O.roi_pool(feature_A_2x,coords_A,output_size=(3,3)) 
    feature_B_local = O.roi_pool(feature_B_2x,coords_B,output_size=(3,3)) 
    
    deltaY, deltaX = torch.meshgrid(torch.linspace(-1,1,3),torch.linspace(-1,1,3))

    deltaX = deltaX.contiguous().to(xA_.device).unsqueeze(0)
    deltaY = deltaY.contiguous().to(xA_.device).unsqueeze(0)
    
    corrA_B = (feature_A_local[:,:,1:2,1:2] * feature_B_local).sum(dim=1).mul(sigma).view(N_matches,-1).softmax(dim=1).view(N_matches,3,3)
    corrB_A = (feature_B_local[:,:,1:2,1:2] * feature_A_local).sum(dim=1).mul(sigma).view(N_matches,-1).softmax(dim=1).view(N_matches,3,3)
        
    deltaX_B = (corrA_B * deltaX).view(N_matches,-1).sum(dim=1).unsqueeze(0)
    deltaY_B = (corrA_B * deltaY).view(N_matches,-1).sum(dim=1).unsqueeze(0)

    deltaX_A = (corrB_A * deltaX).view(N_matches,-1).sum(dim=1).unsqueeze(0)
    deltaY_A = (corrB_A * deltaY).view(N_matches,-1).sum(dim=1).unsqueeze(0)
        
    xA_ = xA_ + deltaX_A
    yA_ = yA_ + deltaY_A
    xB_ = xB_ + deltaX_B
    yB_ = yB_ + deltaY_B
    
    return xA_, yA_, xB_, yB_, score_

# redefine forward function for evaluation with relocalization
def eval_model_reloc(model, batch, args=None):
    
    benchmark = False if args is None else args.benchmark
    relocalize = True if args is None else args.relocalize
    reloc_hard_crop_size = 2 if args is None else args.reloc_hard_crop_size
    no_ncnet = False if args is None else args.no_ncnet
    
    if benchmark:
        start = torch.cuda.Event(enable_timing=True)
        mid = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    # feature extraction
    if relocalize:
        feature_A_2x = model.FeatureExtraction.model(batch['source_image'])
        feature_B_2x = model.FeatureExtraction.model(batch['target_image'])
        
        if reloc_hard_crop_size==3:
            feature_A = F.max_pool2d(feature_A_2x, kernel_size=3, stride=2, padding=1)
            feature_B = F.max_pool2d(feature_B_2x, kernel_size=3, stride=2, padding=1)
        elif reloc_hard_crop_size==2:
            feature_A = F.max_pool2d(feature_A_2x, kernel_size=2, stride=2, padding=0)
            feature_B = F.max_pool2d(feature_B_2x, kernel_size=2, stride=2, padding=0)

        feature_A_2x = featureL2Norm(feature_A_2x)
        feature_B_2x = featureL2Norm(feature_B_2x)
    else:
        feature_A = model.FeatureExtraction.model(batch['source_image'])
        feature_B = model.FeatureExtraction.model(batch['target_image'])
        feature_A_2x, feature_B_2x = None, None
        
    feature_A = featureL2Norm(feature_A)
    feature_B = featureL2Norm(feature_B)
    
    fs1, fs2 = feature_A.shape[-2:]
    fs3, fs4 = feature_B.shape[-2:]
    
    if benchmark:
        mid.record()
    
    
    if no_ncnet:
        corr4d = None
    else:
        corr4d = corr_and_add(feature_A, feature_B, k = model.k, Npts=None)
        corr4d = model.NeighConsensus(corr4d)
        
    if benchmark:
        end.record()
        torch.cuda.synchronize()
        fe_time = start.elapsed_time(mid)/1000
        cnn_time = mid.elapsed_time(end)/1000
        return corr4d, feature_A_2x, feature_B_2x, fs1, fs2, fs3, fs4, fe_time, cnn_time
    
    return corr4d, feature_A_2x, feature_B_2x, fs1, fs2, fs3, fs4
