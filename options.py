import os
import argparse
import shutil
import sys
from subprocess import call,run
from PIL import Image
import paddle
import paddle.vision.transforms as transforms

def test():
    file_path='/Users/yangruizhi/Desktop/PR_list/Old2Life/test_images/old/a.png'
    image=Image.open(file_path).convert('RGB')
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )
    image_new=img_transform(image)
    # print(image_new)

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./test_images/old", help="Test images")
    parser.add_argument("--output_folder",type=str,default="./output",help="Restored images, please use the absolute path",)
    parser.add_argument("--GPU", type=str, default="0", help="0,1,2")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint")
    parser.add_argument("--with_scratch", action="store_true")
    parser.add_argument("--HR", action='store_true')
    opts = parser.parse_args()

    gpu1 = opts.GPU

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)



    ## Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    stage_1_input_dir = opts.input_folder
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    ## Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")

    os.makedirs(stage_4_output_dir, exist_ok=True)
    os.makedirs(stage_1_results, exist_ok=True)

    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    return opt

if __name__ == '__main__':
    test()