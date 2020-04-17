import os
import errno
import numpy as np

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

def create_file_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise