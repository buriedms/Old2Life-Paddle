import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
import functools
from util.image_pool import ImagePool
from models.base_model import BaseModel
import math
from models.NonLocal_feature_mapping_model import *


class Mapping_Model(nn.Layer):
    def __init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None):
        super(Mapping_Model, self).__init__()

        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []
        tmp_nc = 64
        n_up = 4

        print("Mapping: You are using the mapping model without global restoration.")

        for i in range(n_up):
            ic = min(tmp_nc * (2 ** i), mc)
            oc = min(tmp_nc * (2 ** (i + 1)), mc)
            model += [nn.Conv2D(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        for i in range(n_blocks):
            model += [
                networks.ResnetBlock(
                    mc,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                    dilation=opt.mapping_net_dilation,
                )
            ]

        for i in range(n_up - 1):
            ic = min(64 * (2 ** (4 - i)), mc)
            oc = min(64 * (2 ** (3 - i)), mc)
            model += [nn.Conv2D(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2D(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        if opt.feat_dim > 0 and opt.feat_dim < 64:
            model += [norm_layer(tmp_nc), activation, nn.Conv2D(tmp_nc, opt.feat_dim, 1, 1)]
        # model += [nn.Conv2D(64, 1, 1, 1, 0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Pix2PixHDModel_Mapping(BaseModel):
    def name(self):
        return "Pix2PixHDModel_Mapping"

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_smooth_l1, stage_1_feat_l2):
        flags = (True, True, use_gan_feat_loss, use_vgg_loss, True, True, use_smooth_l1, stage_1_feat_l2)

        def loss_filter(g_feat_l2, g_gan, g_gan_feat, g_vgg, d_real, d_fake, smooth_l1, stage_1_feat_l2):
            return [
                l
                for (l, f) in zip(
                    (g_feat_l2, g_gan, g_gan_feat, g_vgg, d_real, d_fake, smooth_l1, stage_1_feat_l2), flags
                )
                if f
            ]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # if opt.resize_or_crop != "none" or not opt.isTrain:
        #     torch.backends.cudnn.benchmark = True  #todo
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        self.netG_A = networks.GlobalGenerator_DCDCv2(
            netG_input_nc,
            opt.output_nc,
            opt.ngf,
            opt.k_size,
            opt.n_downsample_global,
            networks.get_norm_layer(norm_type=opt.norm),
            opt=opt,
        )
        self.netG_B = networks.GlobalGenerator_DCDCv2(
            netG_input_nc,
            opt.output_nc,
            opt.ngf,
            opt.k_size,
            opt.n_downsample_global,
            networks.get_norm_layer(norm_type=opt.norm),
            opt=opt,
        )

        if opt.non_local == "Setting_42" or opt.NL_use_mask:
            if opt.mapping_exp == 1:
                self.mapping_net = Mapping_Model_with_mask_2(
                    min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc),
                    opt.map_mc,
                    n_blocks=opt.mapping_n_block,
                    opt=opt,
                )
            else:
                self.mapping_net = Mapping_Model_with_mask(
                    min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc),
                    opt.map_mc,
                    n_blocks=opt.mapping_n_block,
                    opt=opt,
                )
        else:
            self.mapping_net = Mapping_Model(
                min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc),
                opt.map_mc,
                n_blocks=opt.mapping_n_block,
                opt=opt,
            )

        self.mapping_net.apply(networks.weights_init)
        if opt.load_pretrain != "":
            self.load_network(self.mapping_net, "mapping_net", opt.which_epoch, opt.load_pretrain)
        if not opt.no_load_VAE:

            self.load_network(self.netG_A, "G", opt.use_vae_which_epoch, opt.load_pretrainA)
            self.load_network(self.netG_B, "G", opt.use_vae_which_epoch, opt.load_pretrainB)
            for param in self.netG_A.parameters():
                param.stop_gradiant = True
            for param in self.netG_B.parameters():
                param.stop_gradiant = True
            self.netG_A.eval()
            self.netG_B.eval()

        # if opt.isTrain and len(opt.gpu_ids) > 1:
        #     self.netG_A = paddle.DataParallel(self.netG_A)
        #     self.netG_B = paddle.DataParallel(self.netG_B)
        #     self.mapping_net = paddle.DataParallel(self.mapping_net)

        # if opt.gpu_ids:
        #     self.netG_A.cuda(opt.gpu_ids[0])
        #     self.netG_B.cuda(opt.gpu_ids[0])
        #     self.mapping_net.cuda(opt.gpu_ids[0])
        if not self.isTrain or opt.continue_train:
            self.load_network(self.mapping_net, "mapping_net", opt.which_epoch)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.ngf * 2 if opt.feat_gan else input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1

            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, opt.Smooth_L1,
                                                     opt.use_two_stage_mapping)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan)

            self.criterionFeat = paddle.nn.L1Loss()
            self.criterionFeat_feat = paddle.nn.L1Loss() if opt.use_l1_feat else paddle.nn.MSELoss()

            if self.opt.image_L1:
                self.criterionImage = paddle.nn.L1Loss()
            else:
                self.criterionImage = paddle.nn.SmoothL1Loss()

            print(self.criterionFeat_feat)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss_torch(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_Feat_L2', 'G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake',
                                               'Smooth_L1', 'G_Feat_L2_Stage_1')

            # initialize optimizers
            # optimizer G

            if opt.no_TTUR:
                beta1, beta2 = opt.beta1, 0.999
                G_lr, D_lr = opt.lr, opt.lr
            else:
                beta1, beta2 = 0, 0.9
                G_lr, D_lr = opt.lr / 2, opt.lr * 2

            if not opt.no_load_VAE:
                params = list(self.mapping_net.parameters())
                self.optimizer_mapping = paddle.optimizer.Adam(parameters=params, learning_rate=G_lr, beta1=beta1,
                                                               beta2=beta2)

            # optimizer D                        
            params = list(self.netD.parameters())
            self.optimizer_D = paddle.optimizer.Adam(parameters=params, learning_rate=D_lr, beta1=beta1, beta2=beta2)

            print("---------- Optimizers initialized -------------")

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map
        else:
            # create one-hot vector for label map 
            size = label_map.shape
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            # input_label = paddle.to_tensor(paddle.Tensor.size(oneHot_size)).zero_()# todo
            input_label = paddle.ones(oneHot_size)
            input_label = input_label.scatter_(1, label_map.astype(paddle.int64), 1.0)  # todo long
            if self.opt.data_type == 16:
                input_label = input_label.astype(paddle.int16)

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map
            edge_map = self.get_edges(inst_map)
            input_label = paddle.concat((input_label, edge_map), axis=1)
        input_label = paddle.to_tensor(input_label, stop_gradient=False)

        # real images for training
        if real_image is not None:
            real_image = paddle.to_tensor(real_image)

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = paddle.concat((input_label, test_image.detach()), axis=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def save(self, which_epoch):
        self.save_network(self.mapping_net, 'mapping_net', which_epoch, self.gpu_ids)

    def forward(self, label, inst, image, feat, pair=True, infer=False, last_label=None, last_image=None):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)

        # Fake Generation
        input_concat = input_label

        label_feat = self.netG_A.forward(input_concat, flow='enc')
        # print('label:')
        # print(label_feat.min(), label_feat.max(), label_feat.mean())
        # label_feat = label_feat / 16.0

        if self.opt.NL_use_mask:
            label_feat_map = self.mapping_net(label_feat.detach(), inst)
        else:
            label_feat_map = self.mapping_net(label_feat.detach())

        fake_image = self.netG_B.forward(label_feat_map, flow='dec')
        image_feat = self.netG_B.forward(real_image, flow='enc')

        loss_feat_l2_stage_1 = 0
        loss_feat_l2 = self.criterionFeat_feat(label_feat_map, image_feat) * self.opt.l2_feat

        if self.opt.feat_gan:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(label_feat.detach(), label_feat_map, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss        
            pred_real = self.discriminate(label_feat.detach(), image_feat)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake = self.netD.forward(paddle.concat((label_feat.detach(), label_feat_map), axis=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss  
            if pair:
                pred_real = self.discriminate(input_label, real_image)
            else:
                pred_real = self.discriminate(last_label, last_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake = self.netD.forward(paddle.concat((input_label, fake_image), axis=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)

            # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss and pair:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    tmp = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    loss_G_GAN_Feat += D_weights * feat_weights * tmp
        else:
            loss_G_GAN_Feat = paddle.zeros(1).to(label.device)

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat if pair else paddle.zeros(
                1).to(label.device)

        smooth_l1_loss = 0
        if self.opt.Smooth_L1:
            smooth_l1_loss = self.criterionImage(fake_image, real_image) * self.opt.L1_weight

        return [self.loss_filter(loss_feat_l2, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake,
                                 smooth_l1_loss, loss_feat_l2_stage_1), None if not infer else fake_image]

    def encode_features(self, image, inst):
        image = paddle.to_tensor(image)
        feat_num = self.opt.feat_num
        h, w = inst.shape[2], inst.shape[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.shape[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        # edge = torch.cuda.ByteTensor(t.shape).zero_()
        edge = paddle.zeros_like(t, dtype=paddle.int8)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.astype(paddle.int16)
        else:
            return edge.astype(paddle.float32)

    def inference(self, label, inst):

        use_gpu = len(self.opt.gpu_ids) > 0
        # input_concat = label
        input_concat, _, _, _ = self.encode_input(label)
        inst_data = inst

        label_feat = self.netG_A.forward(input_concat, flow="enc")

        if self.opt.NL_use_mask:
            if self.opt.inference_optimize:
                label_feat_map = self.mapping_net.inference_forward(label_feat.detach(), inst_data)
            else:
                label_feat_map = self.mapping_net(label_feat.detach(), inst_data)
        else:
            label_feat_map = self.mapping_net(label_feat.detach())

        fake_image = self.netG_B.forward(label_feat_map, flow="dec")
        return fake_image

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd

        self.optimizer_D.set_lr(lr)
        try:
            self.optimizer_mapping.set_lr(lr)
        except:
            raise NotImplementedError
        # for param_group in self.optimizer_D._parameter_list:
        #     param_group['lr'] = lr
        # for param_group in self.optimizer_G._parameter_list:
        #     param_group['lr'] = lr
        # for param_group in self.optimizer_featD._parameter_list:
        #     param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel_Mapping):
    def forward(self, label, inst):
        return self.inference(label, inst)
