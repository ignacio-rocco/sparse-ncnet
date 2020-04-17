import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.io import imread
import numpy as np

def normalize_caffe(image):
    if image.ndim==3: return normalize_caffe(image.unsqueeze(0)).squeeze(0)
    # RGB -> BGR
    image = image[:, [2,1,0], :, :]
    # Zero-center by mean pixel
    mean = torch.tensor([103.939, 116.779, 123.68], device = image.device)
    image = image - mean.view(1,3,1,1)
    return image

normalize_image_dict_caffe = lambda x: {k:normalize_caffe(v) if k in ['source_image',
                                                                      'target_image'] else v for k,v in x.items()}

class NormalizeImageDict(object):
    """
    
    Normalizes Tensor images in dictionary
    
    Args:
        image_keys (list): dict. keys of the images to be normalized
        normalizeRange (bool): if True the image is divided by 255.0s
    
    """
    
    def __init__(self,image_keys,normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange=normalizeRange
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0                
            sample[key] = self.normalize(sample[key])
        return  sample
    
def normalize_image(image, forward=True, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        im_size = image.size()
        mean=torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
        std=torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
        if image.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        if isinstance(image,torch.autograd.variable.Variable):
            mean = Variable(mean,requires_grad=False)
            std = Variable(std,requires_grad=False)
        if forward:
            if len(im_size)==3:
                result = image.sub(mean.expand(im_size)).div(std.expand(im_size))
            elif len(im_size)==4:
                result = image.sub(mean.unsqueeze(0).expand(im_size)).div(std.unsqueeze(0).expand(im_size))
        else:
            if len(im_size)==3:
                result = image.mul(std.expand(im_size)).add(mean.expand(im_size))
            elif len(im_size)==4:
                result = image.mul(std.unsqueeze(0).expand(im_size)).add(mean.unsqueeze(0).expand(im_size))
                
        return  result
    
imreadth = lambda x: torch.Tensor(imread(x).astype(np.float32)).transpose(1,2).transpose(0,1)
normalize = lambda x: NormalizeImageDict(['im'])({'im':x})['im']

# allow rectangular images. Does not modify aspect ratio.
resize = lambda x, image_size, scale_factor: F.interpolate(x.unsqueeze(0).cuda(),
                size=(int(np.floor(x.shape[1]/(np.max(x.shape[1:])/image_size)*scale_factor)/scale_factor),
                      int(np.floor(x.shape[2]/(np.max(x.shape[1:])/image_size)*scale_factor)/scale_factor)),mode='bilinear', align_corners=False)

padim = lambda x, h_max: torch.cat((x,x.view(-1)[0].clone().expand(1,3,h_max-x.shape[2],x.shape[3])/1e20),dim=2) if x.shape[2]<h_max else x