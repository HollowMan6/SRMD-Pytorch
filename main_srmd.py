#!/usr/bin/python3
# -*- coding:utf-8 -*-
# by 'hollowman6' from Lanzhou University(兰州大学)

import os.path
import sys
import getopt
import traceback

import numpy as np
from scipy.io import loadmat

import torch

from utils import utils_deblur
from utils import utils_image as util
from utils import utils_model


'''
Python 3.8
PyTorch 1.6.0
Windows 10 or Linux

@inproceedings{zhang2018learning,
  title={Learning a single convolutional super-resolution network for multiple degradations},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3262--3271},
  year={2018}
}

'''

"""
# --------------------------------------------
|--model                 # model
   |--srmdnf_x2          # model_name, for noise-free LR image SR
   |--srmdnf_x3 
   |--srmdnf_x4
   |--srmd_x2            # model_name, for noisy LR image
   |--srmd_x3 
   |--srmd_x4
# --------------------------------------------
"""

noise_level_model = 3  # noise level for model
sf = 2  # scale factor
x8 = False                           # default: False, x8 to boost performance
n_channels = 3            # fixed
nc = 128                  # fixed, number of channels
nb = 12                   # fixed, number of conv layers
model_pool = 'model_zoo'  # fixed
srmd_pca_path = os.path.join('kernels', 'srmd_pca_matlab.mat')
sources = "."     # fixed
results = "."       # fixed
picture_format = "png"
using_device = 'cpu'


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    if noise_level_model == -1:
        model_name = 'srmdnf_x' + str(sf)
    else:
        model_name = 'srmd_x' + str(sf)
    model_path = os.path.join(model_pool, model_name+'.pth')
    in_nc = 18 if 'nf' in model_name else 19

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = sources  # L_path, for Low-quality images
    E_path = results   # E_path, for Estimated images
    if not os.path.splitext(E_path)[1]:
        util.mkdir(E_path)

    device = torch.device(using_device)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from utils.network_srmd import SRMD as net
    model = net(in_nc=in_nc, out_nc=n_channels, nc=nc, nb=nb,
                upscale=sf, act_mode='R', upsample_mode='pixelshuffle')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    if os.path.isfile(L_path):
        L_paths = [L_path]
    else:
        L_paths = util.get_image_paths(L_path)

    # ----------------------------------------
    # kernel and PCA reduced feature
    # ----------------------------------------

    # Gaussian kernel, delta kernel 0.01
    kernel = utils_deblur.fspecial('gaussian', 15, 0.01)

    P = loadmat(srmd_pca_path)['P']
    degradation_vector = np.dot(P, np.reshape(kernel, (-1), order="F"))
    if 'nf' not in model_name:  # noise-free SR
        degradation_vector = np.append(
            degradation_vector, noise_level_model/255.)
    degradation_vector = torch.from_numpy(
        degradation_vector).view(1, -1, 1, 1).float()

    for _, img in enumerate(L_paths):
        img_name, _ = os.path.splitext(os.path.basename(img))
        try:
            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_L, alpha = util.imread_uint_alpha(img, n_channels=n_channels)
            # Bicubic to handle alpha channel if the intended picture is supposed to have.
            if not alpha is None and picture_format == "png":
                alpha = util.uint2tensor4(alpha)
                alpha = torch.nn.functional.interpolate(
                    alpha, scale_factor=sf, mode='bicubic', align_corners=False)
                alpha = alpha.to(device)
                alpha = torch.clamp(alpha, 0, 255)
                alpha = util.tensor2uint(alpha) 
            img_L = util.uint2tensor4(img_L)
            degradation_map = degradation_vector.repeat(
                1, 1, img_L.size(-2), img_L.size(-1))
            img_L = torch.cat((img_L, degradation_map), dim=1)
            img_L = img_L.to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------

            if not x8:
                img_E = model(img_L)
            else:
                img_E = utils_model.test_mode(model, img_L, mode=3, sf=sf)

            img_E = util.tensor2uint(img_E)
            if not alpha is None and picture_format == "png":
                alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
                img_E = np.concatenate((img_E, alpha), axis=2)
            elif not alpha is None:
                print("Warning! You lost your alpha channel for this picture!")

            # ------------------------------------
            # save results
            # ------------------------------------
            if os.path.splitext(E_path)[1]:
                util.imsave(img_E, E_path)
            else:
                util.imsave(img_E, os.path.join(
                    E_path, img_name+'.' + picture_format))
            print(os.path.basename(img) + " successfully saved to disk!")
        except Exception:
            traceback.print_exc()
            print(os.path.basename(img) + " failed!")


def showhelp():
    print("Usage: main_srmd.py -i infile -o outfile [options]...")
    print("")
    print("  -h help              show this help")
    print("  -i input-path        input image path (jpg/png/...) or directory (default=running directory)")
    print("  -o output-path       output image path (jpg/png/...) or directory (default=running directory)")
    print("  -n noise-level       denoise level (-1/0/1/2/3/4/5/6/7/8/9/10, default=3)")
    print("  -s scale             upscale ratio (2/3/4, default=2)")
    print("  -m model-path        srmd model path (default='model_zoo')")
    print("  -p srmd-pca-path     srmd blur kernel pca data path (default='kernels/srmd_pca_matlab.mat')")
    print("  -x tta               enable x8 performance mode (default disabled)")
    print("  -c cuda              enable CUDA GPU caculating (default disabled)")
    print("  -f format            output image format (jpg/png/..., default=png)")
    print("")
    print("Note:")
    print("")
    print("  1. Supported image extensions 'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif'.")
    print("  2. This script uses the model trained by the original author for prediction. If necessary, please train the model, ")
    print("define blur kernel and PCA dimension reduction data by yourself.")
    print("  3. The bicubic interpolation method is used to enlarge the alpha channel, which makes up for the defect that the ")
    print("source model does not support transparent (Alpha) channel.")
    print("  4. the X8 performance mode (also known as TTA mode) is to perform seven different kinds of data augment operations ")
    print("such as rotation etc., and then take the average value of each pixel after upscale. So as to improve the image quality,")
    print("it will be 8 times slower than the original mode.")
    print("  5. input-path and output-path accept either file path or directory path")
    print("  6. noise-level = noise level, larger value means stronger denoise effect, -1 = no effect")
    print("  7. scale = scale level, 2 = upscale 2x, 3 = upscale 3x, 4 = upscale 4x")


if __name__ == '__main__':
    picture_format_specified = False
    if len(sys.argv) == 1:
        showhelp()
        sys.exit()
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hxci:o:n:s:m:p:f:", [
                                   "help", "tta", "cuda", "input-path", "output-path", "noise-level", "scale", "model-path", "srmd-pca-path", "format"])
    except getopt.GetoptError:
        print("Unrecognized Parameter exists.")
        showhelp()
        sys.exit()
    for o, a in opts:
        if o in ("-h", "--help"):
            showhelp()
            sys.exit()
        elif o in ("-x", "--tta"):
            x8 = True
        elif o in ("-c", "--cuda"):
            using_device = "cuda"
        elif o in ("-i", "--input-path"):
            sources = a
        elif o in ("-o", "--output-path"):
            results = a
        elif o in ("-n", "--noise-level"):
            try:
                noise_level_model = int(a)
            except Exception:
                print("Incorrect noise level!")
                print("")
                print("denoise level (-1/0/1/2/3/4/5/6/7/8/9/10, default=3)")
                sys.exit()
        elif o in ("-s", "--scale"):
            try:
                sf = int(a)
            except Exception:
                print("Incorrect scale factor!")
                print("")
                print("upscale ratio (2/3/4, default=2)")
                sys.exit()
        elif o in ("-m", "--model-path"):
            model_pool = a
        elif o in ("-p", "--srmd-pca-path"):
            srmd_pca_path = a
        elif o in ("-f", "--format"):
            picture_format = a
            picture_format_specified = True

    if picture_format_specified and os.path.splitext(results)[1]:
        if os.path.splitext(results)[1] != picture_format:
            print("Warning! Since your output file has been specified, the picture format you specified won't take effect.")

    if os.path.splitext(results)[1]:
        picture_format = os.path.splitext(results)[1][1:]

    if os.path.abspath(sources) == os.path.abspath(results):
        print("Warning! You have same input and output, all the original pictures in the folder will be replaced. Press 'Y' to continue!")
        response = input()
        if response != "Y":
            sys.exit()

    if using_device == "cuda":
        if not torch.cuda.is_available():
            print("Warning! Unable to use CUDA, using CPU instead!")
            using_device == "cpu"

    if not os.path.exists(sources):
        print("Input path doesn't exist!")
    elif noise_level_model > 10 or noise_level_model < -1:
        print("Incorrect noise level!")
        print("")
        print("denoise level (-1/0/1/2/3/4/5/6/7/8/9/10, default=3)")
    elif sf > 4 or sf < 2:
        print("Incorrect scale factor!")
        print("")
        print("upscale ratio (2/3/4, default=2)")
    elif not os.path.isdir(model_pool):
        print("Your model path doesn't exists or you didn't specify a directory!")
    elif not os.path.isfile(srmd_pca_path) or not os.path.splitext(srmd_pca_path)[1] == ".mat":
        print("Your SRMD PCA path doesn't exists or you didn't specify a Matlab Data file (.mat)!")
    elif not ("." + picture_format) in util.IMG_EXTENSIONS:
        print("You have Specified the wrong picture format!")
        print("")
        print("Note: Supported image extension 'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif'")
    else:
        main()
