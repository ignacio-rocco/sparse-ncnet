import faiss
import torch

res = faiss.StandardGpuResources()  # use a single GPU

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def knn_faiss(feature_B, feature_A, k):
    b,ch,nA = feature_A.shape
    if b==1:
        feature_A = feature_A.view(ch,-1).t().contiguous()
        feature_B = feature_B.view(ch,-1).t().contiguous()
        index_cpu = faiss.IndexFlatL2(ch)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        torch.cuda.synchronize()
        feature_B_ptr = swig_ptr_from_FloatTensor(feature_B)
        index.add_c(feature_B.shape[0], feature_B_ptr)
        dist, indx = search_index_pytorch(index, feature_A, k=k)
        dist = dist.t().unsqueeze(0).contiguous()
        indx = indx.t().unsqueeze(0).contiguous()
    else:
        feature_A = feature_A.view(b,ch,-1).permute(0,2,1).contiguous()
        feature_B = feature_B.view(b,ch,-1).permute(0,2,1).contiguous()
        dist = []
        indx = []
        for i in range(b):
            index_cpu = faiss.IndexFlatL2(ch)
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            torch.cuda.synchronize()
            feature_B_ptr = swig_ptr_from_FloatTensor(feature_B[i])
            index.add_c(feature_B[i].shape[0], feature_B_ptr)
            dist_i, indx_i = search_index_pytorch(index, feature_A[i], k=k)
            dist_i = dist_i.t().unsqueeze(0).contiguous()
            indx_i = indx_i.t().unsqueeze(0).contiguous()
            dist.append(dist_i)
            indx.append(indx_i)
        dist = torch.cat(dist,dim=0)
        indx = torch.cat(indx,dim=0)
    return dist, indx

# The knn_faiss_ivf function was working slower for small image resolutions
# I leave it here for future reference or for other people who might find it useful

def knn_faiss_ivf(feature_B, feature_A, k):
    b,ch,nA = feature_A.shape
    if b==1:
        feature_A = feature_A.view(ch,-1).t().contiguous()
        feature_B = feature_B.view(ch,-1).t().contiguous()
        
        quantizer = faiss.IndexFlatL2(ch)  # the other index
        index_cpu = faiss.IndexIVFFlat(quantizer, ch, 100, faiss.METRIC_L2)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        torch.cuda.synchronize()
        feature_B_ptr = swig_ptr_from_FloatTensor(feature_B)
        index.train(feature_B.cpu().numpy())
        index.add_c(feature_B.shape[0], feature_B_ptr)
        
        dist, indx = search_index_pytorch(index, feature_A, k=k)
        dist = dist.t().unsqueeze(0).contiguous()
        indx = indx.t().unsqueeze(0).contiguous()
    else:
        feature_A = feature_A.view(b,ch,-1).permute(0,2,1).contiguous()
        feature_B = feature_B.view(b,ch,-1).permute(0,2,1).contiguous()
        dist = []
        indx = []
        for i in range(b):
            quantizer = faiss.IndexFlatL2(ch)  # the other index
            index_cpu = faiss.IndexIVFFlat(quantizer, ch, 100, faiss.METRIC_L2)
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            torch.cuda.synchronize()
            feature_B_ptr = swig_ptr_from_FloatTensor(feature_B[i])
            index.train(feature_B[i].cpu().numpy())
            index.add_c(feature_B[i].shape[0], feature_B_ptr)
            
            dist_i, indx_i = search_index_pytorch(index, feature_A[i], k=k)
            dist_i = dist_i.t().unsqueeze(0).contiguous()
            indx_i = indx_i.t().unsqueeze(0).contiguous()
            dist.append(dist_i)
            indx.append(indx_i)
        dist = torch.cat(dist,dim=0)
        indx = torch.cat(indx,dim=0)
    return dist, indx