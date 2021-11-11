import paddle.io
from PIL import Image
import paddle.vision.transforms as transforms
import numpy as np
import random
from torchvision_paddle.Lambda import Lambda

class BaseDataset(paddle.io.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize

    if opt.resize_or_crop == 'scale_width_and_crop': # we scale the shorter side into 256

        if w<h:
            new_w = opt.loadSize
            new_h = opt.loadSize * h // w
        else:
            new_h=opt.loadSize
            new_w = opt.loadSize * w // h

    if opt.resize_or_crop=='crop_only':
        pass


    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
    #    transform_list.append(Lambda(lambda img: __scale_width(img, opt.loadSize, method)))  ## Here , We want the shorter side to match 256, and Resize will finish it.
        transform_list.append(transforms.Resize(256))

    if 'crop' in opt.resize_or_crop:
        if opt.isTrain:
            transform_list.append(Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))
        else:
            if opt.test_random_crop:
                transform_list.append(transforms.RandomCrop(opt.fineSize))
            else:
                transform_list.append(transforms.CenterCrop(opt.fineSize))

    ## when testing, for ablation study, choose center_crop directly.


    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(Lambda(lambda img: __flip(img, params['flip'])))



    if normalize:
        transform_list += [transforms.Normalize(mean=(127.5,127.5,127.5),std=(127.5,127.5,127.5),data_format='HWC')]

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize(mean=(127.5,127.5,127.5),std=(127.5,127.5,127.5),data_format='HWC')

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
