# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 05:24:56 2018

@author: henders
"""

import sys
import os
import glob

import numpy as np
import cv2


def color_gradient_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    # Convert to HLS color space and separate the channels.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
#    A = lab[:, :, 1]
    B = lab[:, :, 2]

#    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#    H = hls[:, :, 0]
#    L = hls[:, :, 1]
#    S = hls[:, :, 2]

    # Apply Sobel x to the L channel for HLS, LAB
    # use V channel for HSV
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Appply gradient threshold.
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Apply color channel threshold
    s_binary = np.zeros_like(B)
    s_binary[(B >= s_thresh[0]) & (B <= s_thresh[1])] = 1

    # Stack each channel and return.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary *= 255  # Convert from [0, 1] back to [0, 255]
    return color_binary


def sobel(img, sx_thresh=(20, 100)):
    img = np.copy(img)

    # Convert to HLS color space and separate the channels.
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    # Apply Sobel x to the L channel
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Appply gradient threshold.
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Stack each channel and return.
#    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary))
#    color_binary *= 255  # Convert from [0, 1] back to [0, 255]
    color_binary = sxbinary * 255
    return color_binary



image_dir = './test_frames'
output_dir = './test_frames_output'

images = glob.glob(os.path.join(image_dir, '*.jpg'))
for fname in images:
    print('Processing image {}'.format(fname))
    _, name = os.path.split(fname)
    name, ext = os.path.splitext(name)

    # Read the image.
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])

    # Color/gradient threshold
#    thresh = color_gradient_threshold(img,
#                                      s_thresh=COLOR_THRESHOLD,
#                                      sx_thresh=GRADIENT_THRESHOLD)
#    cv2.imwrite(os.path.join(output_dir, name + '_2_threshold') + ext, thresh)

#    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#    H = hls[:, :, 0]
#    L = hls[:, :, 1]
#    S = hls[:, :, 2] #
#    cv2.imwrite(os.path.join(output_dir, name + '_hls') + ext, hls)
#    cv2.imwrite(os.path.join(output_dir, name + '_hls_h') + ext, H)
#    cv2.imwrite(os.path.join(output_dir, name + '_hls_l') + ext, L)
#    cv2.imwrite(os.path.join(output_dir, name + '_hls_s') + ext, S)

#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    cv2.imwrite(os.path.join(output_dir, name + '_hsv') + ext, hsv)
#    H = hsv[:, :, 0]
#    S = hsv[:, :, 1] #
#    V = hsv[:, :, 2]
#    cv2.imwrite(os.path.join(output_dir, name + '_hsv') + ext, hsv)
#    cv2.imwrite(os.path.join(output_dir, name + '_hsv_h') + ext, H)
#    cv2.imwrite(os.path.join(output_dir, name + '_hsv_s') + ext, S)
#    cv2.imwrite(os.path.join(output_dir, name + '_hsv_v') + ext, V)

#    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#    cv2.imwrite(os.path.join(output_dir, name + '_yuv') + ext, yuv)
#    Y = yuv[:, :, 0]
#    U = yuv[:, :, 1]
#    V = yuv[:, :, 2] #
#    cv2.imwrite(os.path.join(output_dir, name + '_yuv') + ext, yuv)
#    cv2.imwrite(os.path.join(output_dir, name + '_yuv_y') + ext, Y)
#    cv2.imwrite(os.path.join(output_dir, name + '_yuv_u') + ext, U)
#    cv2.imwrite(os.path.join(output_dir, name + '_yuv_v') + ext, V)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imwrite(os.path.join(output_dir, name + '_lab') + ext, lab)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2] #
    cv2.imwrite(os.path.join(output_dir, name + '_lab') + ext, lab)
    cv2.imwrite(os.path.join(output_dir, name + '_lab_l') + ext, L)
    cv2.imwrite(os.path.join(output_dir, name + '_lab_a') + ext, A)
    cv2.imwrite(os.path.join(output_dir, name + '_lab_b') + ext, B)

#    s_binary = np.zeros_like(B)
#    s_binary[(B >= 50) & (B <= 150)] = 1
#    color_binary = s_binary * 255  # Convert from [0, 1] back to [0, 255]
#    cv2.imwrite(os.path.join(output_dir, name + '_lab_b_thresh') + ext, color_binary)

#    thresh = color_gradient_threshold(img,
#                                      s_thresh=(110, 150),
#                                      sx_thresh=(60, 100))
#    cv2.imwrite(os.path.join(output_dir, name + '_thresh') + ext, thresh)
