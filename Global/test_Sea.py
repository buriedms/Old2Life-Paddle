import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.mapping_model import Pix2PixHDModel_Mapping, InferenceModel
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np

import paddle
import torchvision_paddle.utils as vutils
import datetime
import paddle.distributed as dist
import random

opt = TrainOptions().parse()
visualizer = Visualizer(opt)
opt.IsTrain = False

start_epoch, epoch_iter = 0, 0

opt.start_epoch = start_epoch

### temp for continue train unfixed decoder

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(dataset) * opt.batchSize
print('#training images = %d' % dataset_size)

model = InferenceModel()
model.initialize(opt)

performance = util.compute_performance()

epoch_s_t = datetime.datetime.now()
PSNR_list, SSIM_list, FID_list, LPIPS_list = [], [], [], []
model.eval()
for i, data in enumerate(dataset, start=epoch_iter):
    iter_start_time = time.time()
    epoch_iter += opt.batchSize

    ############## Forward Pass ######################
    # print(pair)
    generated = model(paddle.to_tensor(data['label'], stop_gradient=False),
                      paddle.to_tensor(data['inst'], stop_gradient=False))

    # sum per device losses
    performance.update(generated, data['image'])
    PSNR, SSIM, FID, LPIPS = performance.accumulate()
    performance.reset()
    message = 'Batch: %s || ---PSNR:%.3f ---SSIM:%.3f ---FID:%.3f ---LPIPS:%.3f' % (i, PSNR, SSIM, FID, LPIPS)
    print(message)

    PSNR_list.append(PSNR)
    SSIM_list.append(SSIM)
    FID_list.append(FID)
    LPIPS_list.append(LPIPS)
    ############## Display results and errors ##########

PSNR, SSIM, FID, LPIPS = np.array(PSNR_list).mean(), \
                         np.array(SSIM_list).mean(), \
                         np.array(FID_list).mean(), \
                         np.array(LPIPS_list).mean()
message = 'mean performance || ---PSNR:%.3f ---SSIM:%.3f ---FID:%.3f ---LPIPS:%.3f' % (PSNR, SSIM, FID, LPIPS)
print(message)
# end of epoch
epoch_e_t = datetime.datetime.now()
iter_end_time = time.time()
visualizer.print_log('End ====== :\t Time Taken: %s' % (str(epoch_e_t - epoch_s_t)))
