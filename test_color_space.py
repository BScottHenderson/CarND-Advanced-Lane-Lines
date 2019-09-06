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


# Color and gradient thresholds.
COLOR_THRESHOLD = (220, 255) #(180, 230)    # (170, 250) (150, 230)
GRADIENT_THRESHOLD = (65, 130)  # (65, 100) (65, 130)


def color_gradient_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    Apply a color threshold and a gradient threshold to the given image.

    Args:
        img: apply thresholds to this image
        s_thresh: Color threshold (apply to S channel of HSV)
        sx_thresh: Gradient threshold (apply to x gradient on L channel of HSV)

    Returns:
        new image with thresholds applied
    """

    img = np.copy(img)

    # Convert to HLS color space and separate the channels.
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # H = hls[:, :, 0]
    # L = hls[:, :, 1]
    S = hls[:, :, 2]

    # Convert to LAB color space and separate the channels.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    # A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Apply Sobel x (take the derivative on the x axis) to the HLS L channel.
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal.
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Appply gradient threshold.
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Apply color channel threshold.
    s_thresh = (125, 180)
    S = S * (255 / np.max(S))  # normalize
    S_thresh = np.zeros_like(S)
    S_thresh[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1

    s_thresh = (220, 255)
    B = B * (255 / np.max(B))  # normalize
    B_thresh = np.zeros_like(B)
    B_thresh[(B > s_thresh[0]) & (B <= s_thresh[1])] = 1

    # Combine HLS S and Lab B channel thresholds.
    sb_binary = np.zeros_like(S_thresh)
    sb_binary[(S_thresh == 1) | (B_thresh == 1)] = 1

    # Stack each channel and return.
    #                         B                        G         R
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, sb_binary))
    color_binary *= 255  # Convert from [0, 1] back to [0, 255]
    return np.uint8(color_binary)


def main(name=None):

    print('Name: {}\n'.format(name))

    image_dir = './test_frames'
    output_dir = './test_frames_output'

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    for fname in images:
        print('Processing image {} ...'.format(fname))
        _, name = os.path.split(fname)
        name, ext = os.path.splitext(name)

        # Read the image.
        img = cv2.imread(fname)
        # img_size = (img.shape[1], img.shape[0])

        # Color/gradient threshold
        thresh = color_gradient_threshold(img,
                                          s_thresh=COLOR_THRESHOLD,
                                          sx_thresh=GRADIENT_THRESHOLD)
        cv2.imwrite(os.path.join(output_dir, name + '_color_gradient_threshold') + ext, thresh)

        # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # H = hls[:, :, 0] # left line is clear, dashed line is ok
        # L = hls[:, :, 1]
        # S = hls[:, :, 2] # left line is clear, dashed line is ok
        # cv2.imwrite(os.path.join(output_dir, name + '_hls') + ext, hls)
        # cv2.imwrite(os.path.join(output_dir, name + '_hls_h') + ext, H)
        # cv2.imwrite(os.path.join(output_dir, name + '_hls_l') + ext, L)
        # cv2.imwrite(os.path.join(output_dir, name + '_hls_s') + ext, S)

        # Use HLS, S channel to detect dashed white lane lines.
        # s_thresh = (125, 180)
        # S = S * (255 / np.max(S))  # normalize
        # S_thresh = np.zeros_like(S)
        # S_thresh[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1

        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cv2.imwrite(os.path.join(output_dir, name + '_hsv') + ext, hsv)
        # H = hsv[:, :, 0] # left line is clear, dashed line is ok
        # S = hsv[:, :, 1] # left line is clear, dashed line is gone
        # V = hsv[:, :, 2]
        # cv2.imwrite(os.path.join(output_dir, name + '_hsv') + ext, hsv)
        # cv2.imwrite(os.path.join(output_dir, name + '_hsv_h') + ext, H)
        # cv2.imwrite(os.path.join(output_dir, name + '_hsv_s') + ext, S)
        # cv2.imwrite(os.path.join(output_dir, name + '_hsv_v') + ext, V)

        # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # cv2.imwrite(os.path.join(output_dir, name + '_yuv') + ext, yuv)
        # Y = yuv[:, :, 0]
        # U = yuv[:, :, 1] # left line is clear, dashed line is very dim
        # V = yuv[:, :, 2] # left line is clear, dashed line is gone
        # cv2.imwrite(os.path.join(output_dir, name + '_yuv') + ext, yuv)
        # cv2.imwrite(os.path.join(output_dir, name + '_yuv_y') + ext, Y)
        # cv2.imwrite(os.path.join(output_dir, name + '_yuv_u') + ext, U)
        # cv2.imwrite(os.path.join(output_dir, name + '_yuv_v') + ext, V)

        # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # L = lab[:, :, 0]
        # A = lab[:, :, 1] # nothing
        # B = lab[:, :, 2] # left line is clear, dashed line is very dim
        # cv2.imwrite(os.path.join(output_dir, name + '_lab') + ext, lab)
        # cv2.imwrite(os.path.join(output_dir, name + '_lab_l') + ext, L)
        # cv2.imwrite(os.path.join(output_dir, name + '_lab_a') + ext, A)
        # cv2.imwrite(os.path.join(output_dir, name + '_lab_b') + ext, B)


        # s_thresh = (220, 255)
        # B = B * (255 / np.max(B))  # normalize
        # B_thresh = np.zeros_like(B)
        # B_thresh[((B > s_thresh[0]) & (B <= s_thresh[1]))] = 1

        # sb_binary = np.zeros_like(S_thresh)
        # sb_binary[(S_thresh == 1) | (B_thresh == 1)] = 1

        # #                         B                         G         R
        # color_binary = np.dstack((np.zeros_like(sb_binary), S_thresh, B_thresh))
        # color_binary *= 255  # Convert from [0, 1] back to [0, 255]
        # color_binary = np.uint8(color_binary)
        # cv2.imwrite(os.path.join(output_dir, name + '_color_threshold') + ext, color_binary)


if __name__ == '__main__':
    main(*sys.argv)
