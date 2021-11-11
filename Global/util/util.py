from __future__ import print_function
from PIL import Image
import numpy as np
import os
from .PSNR_SSIM import PSNR
from .PSNR_SSIM import SSIM
from .FID import FID
from .LPIPS import LPIPSMetric as LPIPS
import paddle

class IsSave():
    def __init__(self,border=2):
        self.border=border
        self.PSNR=float('-inf')
        self.SSIM=float('-inf')
        self.FID=float('inf')
        self.LPIPS=float('inf')

    def initial(self):
        self.PSNR = float('-inf')
        self.SSIM = float('-inf')
        self.FID = float('inf')
        self.LPIPS = float('inf')

    def is_save(self,PSNR,SSIM,FID,LPIPS):
        num=0
        if self.PSNR>=PSNR:
            num+=1
        if self.SSIM>=SSIM:
            num+=1
        if FID<=self.FID:
            num+=1
        if LPIPS<=self.LPIPS:
            num+=1
        if num>=self.border:
            self.PSNR=PSNR
            self.SSIM=SSIM
            self.FID=FID
            self.LPIPS=LPIPS
            return True
        else:
            return False

class compute_performance():
    def __init__(self,crop_border=0):
        self.PSNR=PSNR(crop_border=crop_border)
        self.SSIM=SSIM(crop_border=crop_border)
        self.FID=FID()
        self.LPIPS=LPIPS()

    def reset(self):
        self.PSNR.reset()
        self.SSIM.reset()
        self.FID.reset()
        self.LPIPS.reset()

    def update(self,preds,gts):
        preds = paddle.clip((preds + 1) * 127.5 + 0.5, 0, 255).transpose(
            [0, 2, 3, 1]).cast("uint8").cast("float32").numpy()
        gts = paddle.clip((gts + 1) * 127.5 + 0.5, 0, 255).transpose(
            [0, 2, 3, 1]).cast("uint8").cast("float32").numpy()
        self.PSNR.update(preds,gts)
        self.SSIM.update(preds,gts)
        self.FID.update(preds,gts)
        self.LPIPS.update(preds,gts)

    def accumulate(self):
        PSNR=self.PSNR.accumulate()
        SSIM=self.SSIM.accumulate()
        FID=self.FID.accumulate()
        LPIPS=self.LPIPS.accumulate()
        return PSNR,SSIM,FID,LPIPS


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
