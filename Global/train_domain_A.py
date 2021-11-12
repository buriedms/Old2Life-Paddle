import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_da_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import paddle
import paddle.distributed as dist
import torchvision_paddle.utils as vutils

opt = TrainOptions().parse()

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataloader = data_loader.load_data()
dataset_size = len(dataloader) * opt.batchSize
print('#training images = %d' % dataset_size)

path = os.path.join(opt.checkpoints_dir, opt.name, 'model.txt')
visualizer = Visualizer(opt)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 0, 0
    if opt.isTrain and len(opt.gpu_ids) > 1:
        visualizer.print_save('Resuming from epoch %d at iteration %d' % (start_epoch - 1, epoch_iter))if dist.get_rank() == 0 else None
    else:
        visualizer.print_save('Resuming from epoch %d at iteration %d' % (start_epoch - 1, epoch_iter))
else:
    start_epoch, epoch_iter = 0, 0

if opt.isTrain and len(opt.gpu_ids) > 1:
    paddle.distributed.init_parallel_env()
# opt.which_epoch=start_epoch-1
model = create_da_model(opt)
fd = open(path, 'w')
fd.write(str(model.module.netG))
fd.write(str(model.module.netD))
fd.close()

total_steps = start_epoch  * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

performance = util.compute_performance()
Save=util.IsSave(border=2)

for epoch in range(start_epoch, opt.niter + opt.niter_decay ):

    # linearly decay learning rate after certain iterations
    if dist.get_rank() == 0:
        if epoch > opt.niter:
            model.module.update_learning_rate()

    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataloader(), start=epoch_iter):

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, generated = model(paddle.to_tensor(data['label'], stop_gradient=False),
                                  paddle.to_tensor(data['inst'], stop_gradient=False),
                                  paddle.to_tensor(data['image'], stop_gradient=False),
                                  paddle.to_tensor(data['feat'], stop_gradient=False), infer=True)

        # sum per device losses
        losses = [paddle.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_featD = (loss_dict['featD_fake'] + loss_dict['featD_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict['G_KL'] + \
                 loss_dict['G_featD']

        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.clear_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.clear_grad()
        loss_D.backward()
        model.module.optimizer_D.step()

        model.module.optimizer_featD.clear_grad()
        loss_featD.backward()
        model.module.optimizer_featD.step()

        # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        # print(f'epoch:{epoch}=====batch_id:{i}=====loss_G:{loss_G.mean().numpy()}=====loss_D:{loss_D.mean().numpy()}')
        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            if opt.isTrain and len(opt.gpu_ids) > 1:
                 visualizer.print_current_errors(epoch, epoch_iter, errors, t, model.module.old_lr) if dist.get_rank()==0 else None
            else:
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, model.module.old_lr)

            # visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            if not os.path.exists(opt.outputs_dir + opt.name):
                os.makedirs(opt.outputs_dir + opt.name)
            imgs_num = 5
            imgs = paddle.concat((data['label'][:imgs_num], generated[:imgs_num], data['image'][:imgs_num]),
                                 0)
            imgs = (imgs + 1.) / 2.0
            image_grid = vutils.save_image(imgs, opt.outputs_dir + opt.name + '/' + str(epoch) + '_' + str(
                total_steps) + '.png', nrow=imgs_num, padding=0, normalize=True)

        if epoch_iter >= dataset_size:
            break
        performance.update(generated[:5], data['image'][:5]) if dist.get_rank()==0 else None
    if dist.get_rank() == 0 :

        PSNR, SSIM, FID, LPIPS = performance.accumulate()
        performance.reset()
        visualizer.print_current_performance(epoch, PSNR, SSIM, FID, LPIPS)
        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('latest')
            # model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        if Save.is_save(PSNR, SSIM, FID, LPIPS):
            model.module.save('best')
            print(f'Epoch:{epoch} || Successfully saved the best model so far.')
            visualizer.print_log(f'Epoch:{epoch} || Successfully saved the best model so far.')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

