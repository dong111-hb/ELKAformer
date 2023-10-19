#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torchvision.transforms as transforms
import torchvision
import torch.optim
import numpy as np
from PIL import Image
from glob import glob
import scipy.misc as misc
import skimage.metrics
import os
import scipy.misc
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from utils import *
import numpy

process = transforms.ToTensor()
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(0)

import cv2
import numpy as np


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def save_images(rain_image_output, test_label_out, predicted_image_out, filepath):
    cat_image = np.column_stack((test_label_out, predicted_image_out))
    cat_image = np.column_stack((rain_image_output, cat_image))
    # cat_image=np.clip(255*cat_image,0,255)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath, 'png')


def test():
    # # save_path = 'img_real4'
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    # rain_path = 'D:\BaiduNetdiskDownload\Rain_data\RCDNet_SM&Results\derained results\\rain1400\perdict\\'
    # label_path = 'D:\BaiduNetdiskDownload\Rain_data\RCDNet_SM&Results\derained results\\rain1400\label\\'
    rain_path = './img_real42/'
    label_path = 'D:/Rain_Data/Rain100H_/val/label/'

    # rain_list = sorted(os.listdir('./deraing_vit_dim48_Rain100H_P2T_val/derain'))
    # rain_list = sorted(os.listdir(rain_path+'/rain'))
    # label_list = sorted(os.listdir(rain_path+'/label'))

    rain_list = sorted(os.listdir(rain_path))
    label_list = sorted(os.listdir(label_path))

    ssim_sum = 0
    psnr_sum = 0
    for i in range(len(rain_list)):
        print(rain_list[i], "-----", label_list[i])

        predicted_image = skimage.io.imread(rain_path + rain_list[i])
        data_label = skimage.io.imread(label_path + label_list[i])

        # predicted_image = skimage.io.imread(rain_path + '/rain/' + rain_list[i])
        # data_label = skimage.io.imread(rain_path + '/label/' + label_list[i])

        predicted_image = np.clip(predicted_image, 0, 255).astype('uint8')
        clean = np.clip(data_label, 0, 255).astype('uint8')

        # ssim = skimage.metrics.structural_similarity(predicted_image, clean, gaussian_weights=True, sigma=1.5,
        #                                                  use_sample_covariance=False, multichannel=True)
        # psnr = skimage.metrics.peak_signal_noise_ratio(predicted_image, clean)

        clean_ycrcb = cv2.cvtColor(clean, cv2.COLOR_BGR2YCrCb)
        predicted_image_ycrcb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2YCrCb)

        # PSNR = psnr(predicted_image_ycrcb, clean_ycrcb)
        # SSIM = ssim(predicted_image_ycrcb, clean_ycrcb)
        PSNR = psnr(predicted_image, clean)
        SSIM = ssim(predicted_image, clean)
        # img = Image.fromarray(predicted_image.astype('uint8'))

        # img.save('{0}/{1}'.format(save_path, label_list[i]))
        # save_images(clean, predicted_image,
        #                 os.path.join("./img_real4", '%s_%.4f_%.4f.png' % (rain_list[i], psnr, ssim)))
        print("PSNR = %.4f" % PSNR)
        print("SSIM = %.4f" % SSIM)
        ssim_sum += SSIM
        psnr_sum += PSNR

    avg_ssim = ssim_sum / len(rain_list)
    avg_psnr = psnr_sum / len(rain_list)
    print("---- Average SSIM = %.4f----" % avg_ssim)
    print("---- Average PSNR = %.4f----" % avg_psnr)


if __name__ == '__main__':
    test()
    print("perfect,done!")
