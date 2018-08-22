# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 12:28:54 2018

@author: Scott
"""

import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import cv2


#
# Option Flags
#

PRINT_CAMERA_CALIBRATION_ERROR = True
REFINE_CAMERA_MATRIX_DURING_UNDISTORTION = False
DRAW_WARP_LINES = True


#
# Camera Calibration
#

def calibrate_camera(image_dir):
    """
    Calculate a camera calibration matrix using chessboard images.

    Code taken from:
        OpenCV Python Tutorials
        Camera Calibration
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    Args:
        image_dir: location of calibration images (*.jpg)

    Returns:
        camera calibration matrix
        distortion coefficients
    """

    # Size of calibration chessboard.
    nx = 9  # Number of inside corners in x
    ny = 6  # Number of inside corners in y

    #
    # Get lists of object points and image points
    #

    # Corner refinement termination criteria.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points (after refining them)
        if ret:
            print('Adding points from {}'.format(fname))
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

#            # Draw and display the corners
#            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
#            plt.imshow(img)

#            # Draw and display the corners
#            img = cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
#            cv2.imshow('img', img)
#            cv2.waitKey(500)
        else:
            print('No corners found for {}'.format(fname))
#
#    cv2.destroyAllWindows()

    #
    # Camera calibration
    #

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    """
    objpoints   points on the actual object
    imgpoints   points on the distorted image

    mtx         the camera matrix
    dist        distortion coefficients
    rvecs       rotation vectors
    tvecs       translation vectors

    mtx and dist are used to undistort an image
    rvecs and tvecs give the position of the camera in the real World
    """

    if PRINT_CAMERA_CALIBRATION_ERROR:
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _  = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        total_error /= len(objpoints)
        print('\nTotal camera calibration error: {}\n'.format(total_error))

    return mtx, dist


#
# Undistort
#

def undistort(img, C, D):
    """
    Undistort an image.

    Args:
        img: image to be undistorted
        C: camera calibration matrix
        D: distortion coefficients

    Returns:
        undistorted image
    """
    img = np.copy(img)

    if REFINE_CAMERA_MATRIX_DURING_UNDISTORTION:
        # Refine the camera matrix.
        h, w = img.shape[:2]
        C2, roi = cv2.getOptimalNewCameraMatrix(C, D, (w, h), 1, (w, h))

        # Undistort.
        img = cv2.undistort(img, C, D, None, C2)
        """
        img  source image
        C    camera matrix
        D    distortion coefficients
        None destination
        C2   new camera matrix for distorted image
        """

        # Crop the image.
#        x, y, w, h = roi
#        img = img[y:y + h, x:x + w]
    else:
        # Undistort
        img = cv2.undistort(img, C, D, None, C)

    return img


#
# Apply color and gradient thresholds to an image.
#

def color_gradient_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    Apply a color threshold and a gradient threshold to the given image.

    Args:
        s_thresh: Color threshold (apply to S channel)
        sx_thresh: Gradient threshold (apply to x gradient on L channel)

    Returns:
        new image with thresholds applied
    """

    img = np.copy(img)

    # Convert to HLS color space and separate the channels.
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
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

    # Apply color channel threshold
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1

    # Stack each channel and return.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


#
# Perspective Transformation Matrices
#

def perspective_transform_matrix(img_size):
    """
    Create perspective transformation matrices based on image size.

    Initial implementation of src and dst points obtained from Slack
    post by RPieter based on YouTube video walkthrough. I added some
    modifications for the bottom of the source quadrilateral.

    Args:
        img_size: sample image size (x, y)

    Returns:
        M, Minv,  perspective transformation matrix and inverse
        src, dst: source and destination points for debugging
    """

    # Adjust the bottom slightly to the right
    x_bot_offset = 10

    # Define perspective transformation  area
    bot_width = .60      # percentage of bottom trapezoid width
    mid_width = .06      # percentage of middle trapezoid width
    height_pct = .62     # percentage for trapezoid height
    bottom_trim = .935   # percentage top to bottom to avoid car hood
    src = np.float32([  # clockwise from top-right
        [img_size[0] * (.5 + mid_width / 2),                img_size[1] * height_pct],
        [img_size[0] * (.5 + bot_width / 2) + x_bot_offset, img_size[1] * bottom_trim],
        [img_size[0] * (.5 - bot_width / 2) + x_bot_offset, img_size[1] * bottom_trim],
        [img_size[0] * (.5 - mid_width / 2),                img_size[1] * height_pct]
    ])
    offset = img_size[0] * .25
    dst = np.float32([
        [img_size[0] - offset, 0],
        [img_size[0] - offset, img_size[1]],
        [offset,               img_size[1]],
        [offset,               0]
    ])

    # Get matrices
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv, src, dst


def draw_lines(img, pts):
    """
    Draw lines between a list of points on an image.

    Args:
        img: draw on this image
        pts: list of points

    Returns:
        modified image
    """
    img = np.copy(img)
    first = None
    pt1 = None
    pt2 = None
    color = (0, 0, 255)  # BGR
    thick = 2
    for p in pts:
        if first is None:
            first = p
            pt2 = p
        else:
            pt1 = pt2
            pt2 = p
            cv2.line(img, tuple(pt1), tuple(pt2), color, thick)
    cv2.line(img, tuple(pt2), tuple(first), color, thick)
    return img


#
# LaneLine class
#

class LaneLine():
    """
    Define a class to receive the characteristics of each line detection
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def DrawLaneLine(warped, left_fitx, right_fitx, ploty):
    """
    Draw left and right lane lines on the warped image "warped."
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)


#
# Main
#

def main(name):

    print('Name: {}'.format(name))

    """
    1. Camera calibration
    2. Distortion correction
    3. Color/gradient threshold
    4. Perspective transform
    5. Detect lane lines
    6. Determine the lane curvature
    """

    #
    # Camera Calibration
    #

    # Get the camera matrix (C) and distortion coefficients (D) from
    # the camera calibration images.
    C, D = calibrate_camera('./camera_cal')

    #
    # Perspective Transform Matrix
    #

    M, Minv = None, None  # Calculated below after reading first image.

    #
    # For each image ...
    #

    image_dir = './test_images'
    output_dir = './output_images'
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    for fname in images:
        print('Processing image {}'.format(fname))
        _, name = os.path.split(fname)
        name, ext = os.path.splitext(name)

        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])

        # Distortion correction.
        img = undistort(img, C, D)
        cv2.imwrite(os.path.join(output_dir, name + '_1_undistort') + ext, img)

        # Color/gradient threshold
        thresh = color_gradient_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100))
        cv2.imwrite(os.path.join(output_dir, name + '_2_threshold') + ext, img)

        # Perspective transformation
        if M is None:
            M, Minv, src, dst = perspective_transform_matrix(img_size)
        warped = cv2.warpPerspective(thresh, M, img_size, flags=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, name + '_3_warped') + ext, warped)

        if DRAW_WARP_LINES:
            # Draw lines on undistorted original image (not color threshold output).
            img2 = draw_lines(img, src)
            cv2.imwrite(os.path.join(output_dir, name + '_3_warped1') + ext, img2)
            # Warp the original image (not the color threshold output) and draw lines.
            img3 = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
            img3 = draw_lines(img3, dst)
            cv2.imwrite(os.path.join(output_dir, name + '_3_warped2') + ext, img3)

        # Detect lane lines

if __name__ == '__main__':
    main(*sys.argv)
