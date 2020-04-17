import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def plot_image(im,batch_idx=0,return_im=False):
    if im.dim()==4:
        im=im[batch_idx,:,:,:]
    mean=Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1))
    std=Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1))
    if im.is_cuda:
        mean=mean.cuda()
        std=std.cuda()
    im=im.mul(std).add(mean)*255.0
    im=im.permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    if return_im:
        return im
    plt.imshow(im)
    plt.show()
    
def imshow_image(image, preprocessing=None):
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        mean = np.array([103.939, 116.779, 123.68])
        image = image + mean.reshape([3, 1, 1])
        # RGB -> BGR
        image = image[:: -1, :, :]
    elif preprocessing == 'torch':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std.reshape([3, 1, 1]) + mean.reshape([3, 1, 1])
        image *= 255.0
    else:
        raise ValueError('Unknown preprocessing parameter.')
    image = np.transpose(image, [1, 2, 0])
    image = np.round(image).astype(np.uint8)
    return image

def save_plot(filename):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches = 'tight',
        pad_inches = 0)
    
padim = lambda x,h_max: torch.cat((x,x.view(-1)[0].clone().expand(1,3,h_max-x.shape[2],x.shape[3])/1e20),dim=2) if x.shape[2]<h_max else x