import os
from paddle import Tensor
from collections import OrderedDict
from options.test_options import TestOptions
from models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping
import util.util as util
from PIL import Image
import paddle
import torchvision_paddle.utils as vutils
import paddle.vision.transforms as transforms
import numpy as np

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256
    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img
    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Resize(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    ##

    # if opt.Quality_restore:
    #     opt.name = "mapping_quality"
    #     opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
    #     opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    # if opt.Scratch_and_Quality_restore:
    #     opt.NL_res = True
    #     opt.use_SN = True
    #     opt.correlation_renormalize = True
    #     opt.NL_use_mask = True
    #     opt.NL_fusion_method = "combine"
    #     opt.non_local = "Setting_42"
    #     opt.name = "mapping_scratch"
    #     opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
    #     opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
    #     if opt.HR:
    #         opt.mapping_exp = 1
    #         opt.inference_optimize = True
    #         opt.mask_dilation = 3
    #         opt.name = "mapping_Patch_Attention"


if __name__ == "__main__":

    opt = TestOptions().parse(save=False)
    parameter_set(opt)
    opt.isTrain=False
    model = Pix2PixHDModel_Mapping()
    model.initialize(opt)
    model.eval()

    if not os.path.exists(opt.outputs_dir + "/" + "input_image"):
        os.makedirs(opt.outputs_dir + "/" + "input_image")
    if not os.path.exists(opt.outputs_dir + "/" + "restored_image"):
        os.makedirs(opt.outputs_dir + "/" + "restored_image")
    if not os.path.exists(opt.outputs_dir + "/" + "origin"):
        os.makedirs(opt.outputs_dir + "/" + "origin")

    input_loader = os.listdir(opt.test_input)
    dataset_size = len(input_loader)
    input_loader.sort()

    img_transform = transforms.Compose(
        [transforms.Normalize(mean=(127.5,127.5,127.5),std=(127.5,127.5,127.5),data_format='HWC'),
         transforms.ToTensor()]
    )
    mask_transform = transforms.ToTensor()
    for i in range(dataset_size):

        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % input_name)
            continue
        input = Image.open(input_file).convert("RGB")


        print("Now you are processing %s" % (input_name))

        if opt.test_mode == "Scale":
            input = data_transforms(input, scale=True)
        if opt.test_mode == "Full":
            input = data_transforms(input, scale=False)
        if opt.test_mode == "Crop":
            input = data_transforms_rgb_old(input)
        origin = input
        input = np.array(input).astype('uint8')
        input=img_transform(input)
        input = input.unsqueeze(0)
        mask = paddle.zeros_like(input)
        ### Necessary input

        try:
            with paddle.no_grad():
                generated = model.inference(input, mask)

        except Exception as ex:
            print("Skip %s due to an error:\n%s" % (input_name, str(ex)))
            continue

        if input_name.endswith(".jpg"):
            input_name = input_name[:-4] + ".png"

        image_grid = vutils.save_image(
            (input + 1.0) / 2.0,
            opt.outputs_dir + "/input_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        image_grid = vutils.save_image(
            (generated + 1.0) / 2.0,
            opt.outputs_dir + "/restored_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        origin.save(opt.outputs_dir + "/origin/" + input_name)


