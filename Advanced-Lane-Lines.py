# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 12:28:54 2018

@author: Scott
"""

import sys
import os
import glob

import numpy as np
import cv2

from moviepy.editor import VideoFileClip


#
# Option Flags
#

UNDISTORT_REFINE_CAMERA_MATRIX = False  # this is not working
PROCESS_TEST_IMAGES = False
WRITE_OUTPUT_FRAMES = False


#
# Hyperparameters
#

# Color and gradient thresholds.
COLOR_THRESHOLD = (170, 250)
GRADIENT_THRESHOLD = (65, 100)

# Number of sliding windows used to find lane lines.
LANE_LINES_NWINDOWS = 9
# Width of the lane line window +/- margin.
LANE_LINES_MARGIN = 100
# Minimum number of pixels found to recenter lane lines window.
LANE_LINES_MINPIX = 50

# Number of frames to keep for averaging in the LaneLine history.
FRAME_HISTORY_COUNT = 9


#
# Constants
#

# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

# Image text parameters
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_COLOR = (255, 255, 255)
TEXT_THICKNESS = 2
TEXT_LINE_TYPE = cv2.LINE_AA


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
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def AddFrame(self, img, lane_x, lane_y):
        # Fit a second order polynomial to each lane line.
        """
            You're fitting for f(y), rather than f(x), because the lane lines
            in the warped image are near vertical and may have the same x value
            for more than one y value.
        """
        fit = np.polyfit(lane_y, lane_x, 2)
        fit_m = np.polyfit(lane_y * YM_PER_PIX, lane_x * XM_PER_PIX, 2)

        # Generate x and y values for fitting.
        fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
        fitx = fit[0] * fity**2 + fit[1] * fity + fit[2]

        # Define y-value where we want radius of curvature.
        # Use the maximum y-value, corresponding to the bottom of the image.
        y_eval = np.max(fity)

        # Implement the calculation of R_curve (radius of curvature)
        y_eval *= YM_PER_PIX  # convert y value from pixels to meters
        self.radius_of_curvature = np.power((1 + np.square(2 * fit_m[0] * y_eval + fit_m[1])), 1.5) / np.abs(2 * fit_m[0])

        # Calculate offset from image center.
        x = fit_m[0] * y_eval**2 + fit_m[1] * y_eval + fit_m[2]
        img_center = (img.shape[1] / 2.) * XM_PER_PIX
        self.line_base_pos = img_center - x

        self.allx = fitx
        self.ally = fity

        self.current_fit.append(fit)
        # Keep only the most recent N frames.
        if len(self.current_fit) > FRAME_HISTORY_COUNT:
            self.current_fit = self.current_fit[len(self.current_fit) - FRAME_HISTORY_COUNT:]

        self.best_fit = np.average(self.current_fit, axis=0)

        self.diffs = abs(fit - self.best_fit)


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

            corners2 = cv2.cornerSubPix(gray, corners,
                                        (11, 11),  # winSize
                                        (-1, -1),  # zeroZone
                                        criteria)
            # winSize:  half of the size of the search window
            # zeroZone: half of the size of the dead region in the middle of
            # the search zone in which calculations are not done
            # criteria: stop refining after a certain number of iterations or
            # after the corner position moves by less than epsilon
            imgpoints.append(corners2)

#            # Draw and display the corners
#            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
#            plt.imshow(img)

#            # Draw and display the corners
#            img = cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
#            cv2.imshow('img', img)
#            cv2.waitKey(500)
        else:
            # For some of the test images there are only 5 corners in the y
            # direction instead of 6 so findChessboardCorners() does not find
            # corners. I can dynamically modify ny if the first attempt does
            # not work but this causes problems downstream that I am not sure
            # how to deal with. So just ignore these calibration images.
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

    # Print camera calibration error.
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

    if UNDISTORT_REFINE_CAMERA_MATRIX:
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
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]
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

    # Apply color channel threshold
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1

    # Stack each channel and return.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary *= 255  # Convert from [0, 1] back to [0, 255]
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


#
# Find Lane Lines
#

def find_lane_pixels(img):
    """
    Find lane line pixels in a birds-eye image based on histogram data.

    Note that the input image must be grayscale since we're using np.sum()
    to generate the histogram and are then using the histogram frequency
    data as pixel positions. If we have color data the sum could end up
    outside the pixel range (size) of the image.

    Args:
        img: lane line image - must be grayscale!

    Returns:
        leftx, lefty, rightx, righty: Arrays of lane line pixels.
        out_img: Copy of the input image with lane line windows drawn.
    """

    # Take a histogram of the bottom half of the image
    bottom_half = img[img.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on LANE_LINES_NWINDOWS and image shape.
    window_height = np.int(img.shape[0] // LANE_LINES_NWINDOWS)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window.
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices.
    left_lane_inds = []
    right_lane_inds = []

    # For each window ...
    for window in range(LANE_LINES_NWINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] -  window      * window_height

        win_xleft_low   = leftx_current  - LANE_LINES_MARGIN
        win_xleft_high  = leftx_current  + LANE_LINES_MARGIN
        win_xright_low  = rightx_current - LANE_LINES_MARGIN
        win_xright_high = rightx_current + LANE_LINES_MARGIN

        # Draw the windows on the visualization image.
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                               (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window.
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists.
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If we found more than the min. number of pixels, recenter the next
        # window (right or left) based on the mean position of tohse pixels.
        if len(good_left_inds) > LANE_LINES_MINPIX:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > LANE_LINES_MINPIX:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (list of lists of pixels -> array)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions.
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_lane_line_polynomial(img):
    """
    Find lane lines in an image and fit a quadratic polynomial.

    Args:
        img: image containing lane line pixels

    Returns:
        left_fit, right_fit: polynomial coefficients for left and right
                             lane lines in pixels
        left_fit_m, right_fit_m: polynomial coefficients for left and right
                                 lane lines in meters
        out_img: copy of the input image with lane lines drawn
    """

    # Find lane line pixels.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(gray)

    # Fit a second order polynomial to each lane line.
    """
        You're fitting for f(y), rather than f(x), because the lane lines
        in the warped image are near vertical and may have the same x value
        for more than one y value.
    """
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit_m = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit_m = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    # Generate x and y values for plotting.
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Set the color for the left and right lane regions.
    out_img[lefty, leftx] = [255, 0, 0]  # blue
    out_img[righty, rightx] = [0, 0, 255]  # red

    # Plot the left and right curves on the lane lines image.
    lane_color = (0, 255, 255)
    points = [p for p in zip(left_fitx, ploty)]
    out_img = draw_lines(out_img, points, color=lane_color, closed=False)
    points = [p for p in zip(right_fitx, ploty)]
    out_img = draw_lines(out_img, points, color=lane_color, closed=False)

    return left_fit, right_fit, left_fit_m, right_fit_m, out_img


#
# Lane Line Curvature
#

def measure_curvature(img, left_fit, right_fit):
    """
    Calculates the curvature of lane line polynomial functions in meters.

    Args:
        img: lane line image
        left_fit, right_fit: quadratic polynomial coefficients in meters

    Returns:
        radius of curvature for left and right lane lines
    """

    # Define y-value where we want radius of curvature.
    # Use the maximum y-value, corresponding to the bottom of the image.
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)

    # Implement the calculation of R_curve (radius of curvature)
    y_eval *= YM_PER_PIX  # convert y value from pixels to meters
    left_curverad = np.power((1 + np.square(2 * left_fit[0] * y_eval + left_fit[1])), 1.5) / np.abs(2 * left_fit[0])
    right_curverad = np.power((1 + np.square(2 * right_fit[0] * y_eval + right_fit[1])), 1.5) / np.abs(2 * right_fit[0])

    return left_curverad, right_curverad


#
# Draw lines
#

def draw_lines(img, pts, color=(0, 0, 255), thick=2, closed=True):
    """
    Draw lines between a list of points on an image.

    Args:
        img: draw on this image
        pts: list of points
        color: line color (BGR)
        thick: line thickness
        closed: draw a line from last point back to first point?

    Returns:
        copy of input image with lines drawn
    """

    # Make sure points are tuples. This conversion will be harmless in
    # case the points are already tuples. In theory this conversion could
    # be made dependent on the type of pts[0], i.e., the first point, but
    # in practice I could not make this work using either type() or __class__.
    # Also convert to integer - I'm not sure why this is necessary since
    # the image warping source quadrilateral uses floating point coords
    # and that would draw just fine. Still rounding to the nearest integer
    # is close enough for drawing.
    pts = np.copy(pts)
    pts = [tuple([np.int(p[0]), np.int(p[1])]) for p in pts]

    # Do not modify the original image.
    img = np.copy(img)

    first = None
    pt1 = None
    pt2 = None
    for p in pts:
        if first is None:
            first = p
            pt2 = p
        else:
            pt1 = pt2
            pt2 = p
#            print('{} - {}'.format(pt1, pt2))
            cv2.line(img, pt1, pt2, color, thick)
    if closed:
        cv2.line(img, pt2, first, color, thick)

    return img


#
# Find Lane Lines in a video clip
#

class AdvancedLaneLines():
    """
    Define a class to encapsulate video processing for lane lines.

    Lane finding steps:

    1. Camera calibration
    2. Distortion correction
    3. Color/gradient threshold
    4. Perspective transform
    5. Detect lane lines
    6. Determine the lane curvature
    """

    def __init__(self):
        # Camera matrix
        self.C = None
        # Distortion coefficients
        self.D = None
        # Perspective transform matrix
        self.M = None
        # Inverse perspective transform matrix
        self.Minv = None
        # Left lane line
        self.left = LaneLine()
        # Right lane line
        self.right = LaneLine()
        # Current frame number.
        self.current_frame = 0
        # Output dir for modified video frames.
        self.video_dir = None

        # Get the camera matrix (C) and distortion coefficients (D) from
        # the camera calibration images.
        print('Camera calibration ...')
        self.C, self.D = calibrate_camera('./camera_cal')

    def ProcessVideoClip(self, input_file, video_dir=None):
        """
        Apply the FindLaneLines() function to each frame in a given video file.
        Save the results to a new video file in the same location using the
        same filename but with "_lanes" appended.

        Args:
            input_file (str): Process this video file.
            video_dir (str): Optional location for modified video frames.

        Returns:
            none

        To speed up the testing process or for debugging we can use a subclip
        of the video. To do so add

            .subclip(start_second, end_second)

        to the end of the line below, where start_second and end_second are
        integer values representing the start and end of the subclip.
        """
        # Open the video file.
        input_clip = VideoFileClip(input_file)  # .subclip(41, 43)

        # For each frame in the video clip, replace the frame image with the
        # result of applying the 'FindLaneLines' function.
        # NOTE: this function expects color images!!
        output_clip = input_clip.fl_image(self.FindLaneLines)

        # Save the resulting, modified, video clip to a file.
        file_name, ext = os.path.splitext(input_file)
        output_file = file_name + '_lanes' + ext
        output_clip.write_videofile(output_file, audio=False)

        # Cleanup
        input_clip.reader.close()
        input_clip.audio.reader.close_proc()
        del input_clip
        output_clip.reader.close()
        output_clip.audio.reader.close_proc()
        del output_clip

    def FindLaneLines(self, img):
        """
        Given an image (video frame) find lane lines. Draw the lane, along
        with the curvature radius and distance of the car from lane center,
        on a copy of the input image and return the result.

        Args:
            img: input image (video frame)

        Returns:
            copy of the input image with lane lines drawn
        """
        self.current_frame += 1

        img_size = (img.shape[1], img.shape[0])

        # Distortion correction
        img = undistort(img, self.C, self.D)

        # Color/gradient threshold
        thresh = color_gradient_threshold(img,
                                          s_thresh=COLOR_THRESHOLD,
                                          sx_thresh=GRADIENT_THRESHOLD)

        # Perspective transformation
        if self.M is None or UNDISTORT_REFINE_CAMERA_MATRIX:
            self.M, self.Minv, src, dst = perspective_transform_matrix(img_size)
        warped = cv2.warpPerspective(thresh, self.M, img_size, flags=cv2.INTER_LINEAR)

#        # Detect lane lines
#        left_fit, right_fit, left_fit_m, right_fit_m, lane_lines = fit_lane_line_polynomial(warped)
#
#        # Calculate lane line curvature.
#        left_curverad_m, right_curverad_m = measure_curvature(warped, left_fit_m, right_fit_m)
#
#        # Draw filled lane polygon.
#        filled_lane = DrawLaneLine(img, img_size, warped, Minv, left_fit, right_fit)
#
#        # Write the modified video frame to the output file.
#        vidwriter.write(filled_lane)
#
#        # For whatever reason I was not able to get VideoWriter to work.
#        # It produces a file but the file remains empty.
#        # So I will write each frame out to a separate file and then use
#        # the video.py script from the Behavioral Cloning project to create
#        # the video file as a separate step.
#        cv2.imwrite(os.path.join(video_dir, 'frame{:03d}.jpg'.format(frame_count)), filled_lane)

        #
        # Update LaneLine objects
        #

        # Find lane line pixels.
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        leftx, lefty, rightx, righty, _ = find_lane_pixels(gray)

        self.left.AddFrame(warped, leftx, lefty)
        self.right.AddFrame(warped, rightx, righty)

        filled_lane = self.DrawLaneLine(img, img_size, warped, self.left.best_fit, self.right.best_fit)

        # Write left/right lane line curavture radii to image.
        curve_radius = 'Radius: ({} m, {} m)'.format(
                round(self.left.radius_of_curvature, 1),
                round(self.right.radius_of_curvature, 1))
        cv2.putText(filled_lane, curve_radius, (10, 30),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # Calculate average offset from lane center and write to image.
        lane_center = (self.left.line_base_pos + self.right.line_base_pos) / 2.
        offset = 'Offset: {} m'.format(round(lane_center, 1))
        cv2.putText(filled_lane, offset, (10, 60),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # Write the frame number to the image.
        frame = 'Frame: {}'.format(self.current_frame)
        cv2.putText(filled_lane, frame, (1050, 30),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # For whatever reason I was not able to get VideoWriter to work.
        # It produces a file but the file remains empty.
        # So I will write each frame out to a separate file and then use
        # the video.py script from the Behavioral Cloning project to create
        # the video file as a separate step.
        if self.video_dir is not None:
            output_file = os.path.join(self.video_dir,
                                       'frame{:06d}.jpg'.format(self.current_frame))
            cv2.imwrite(output_file, filled_lane)

        # Return the modified image.
        return filled_lane

    def DrawLaneLine(self, img, img_size, warped, left_fit, right_fit):
        """
        Draw left and right lane lines on the given warped image.

        Args:
            img: Original image
            img_size: Original image size
            warped: Warped perspective version of origial image
            left_fit, right_fit: quadratic coefficients for lane lines

        Returns:
            a copy of the input image with the lane filled in
        """

        # Create an image to draw the lines on.
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        # Our input image is already color so this step is not necessary.
#        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Calculate points for each lane line using quadratic equation.
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image.
        cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective
        # transformation matrix (Minv).
        unwarp = cv2.warpPerspective(warp_zero, self.Minv, img_size, flags=cv2.INTER_LINEAR)

        # Combine the lane line image with the original image.
        result = cv2.addWeighted(img, 1, unwarp, 0.3, 0)

        return result

    def ProcessTestImages(self, image_dir, output_dir):
        """
        Process images in the test directory and create output images showing
        the results of various stages of the image processing pipeline.

        Args:
            image_dir: read test images from this directory
            output_dir: write modified images to this directory

        Returns:
            nothing
        """
        images = glob.glob(os.path.join(image_dir, '*.jpg'))
        for fname in images:
            print('Processing image {}'.format(fname))
            _, name = os.path.split(fname)
            name, ext = os.path.splitext(name)

            # Read the image.
            img = cv2.imread(fname)
            img_size = (img.shape[1], img.shape[0])

            # Distortion correction
            img = undistort(img, self.C, self.D)
            cv2.imwrite(os.path.join(output_dir, name + '_1_undistort') + ext, img)

            # Color/gradient threshold
            thresh = color_gradient_threshold(img,
                                              s_thresh=COLOR_THRESHOLD,
                                              sx_thresh=GRADIENT_THRESHOLD)
            cv2.imwrite(os.path.join(output_dir, name + '_2_threshold') + ext, thresh)

            # Perspective transformation
            if self.M is None or UNDISTORT_REFINE_CAMERA_MATRIX:
                self.M, self.Minv, src, dst = perspective_transform_matrix(img_size)
            warped = cv2.warpPerspective(thresh, self.M, img_size, flags=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(output_dir, name + '_3_warped') + ext, warped)

            # Draw lines on undistorted original image (not color threshold output).
            img2 = draw_lines(img, src)
            cv2.imwrite(os.path.join(output_dir, name + '_3_warped1') + ext, img2)

            # Warp the original image (not the color threshold output) and draw lines.
            img3 = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
            img3 = draw_lines(img3, dst)
            cv2.imwrite(os.path.join(output_dir, name + '_3_warped2') + ext, img3)

            # Detect lane lines
            left_fit, right_fit, left_fit_m, right_fit_m, lane_lines = fit_lane_line_polynomial(warped)
            cv2.imwrite(os.path.join(output_dir, name + '_4_lane_lines') + ext, lane_lines)

            # Calculate lane line curvature and write to image.
            left_curverad_m, right_curverad_m = measure_curvature(warped, left_fit_m, right_fit_m)
            curve_radius = 'Radius: ({} m, {} m)'.format(
                    round(left_curverad_m, 1), round(right_curverad_m, 1))
            cv2.putText(img, curve_radius, (10, 30),
                        TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

            # Calculate offset from lane center and write to image.
            y = (img.shape[0] - 1) * YM_PER_PIX
            left_x = left_fit_m[0] * y**2 + left_fit_m[1] * y + left_fit_m[2]
            right_x = right_fit_m[0] * y**2 + right_fit_m[1] * y + right_fit_m[2]
            lane_center = (left_x + right_x) / 2.
            img_center = (img.shape[1] / 2.) * XM_PER_PIX
            offset = 'Offset: {} m'.format(round(img_center - lane_center, 1))
            cv2.putText(img, offset, (10, 60),
                        TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

            # Draw filled lane polygon.
            filled_lane = self.DrawLaneLine(img, img_size, warped, left_fit, right_fit)
            cv2.imwrite(os.path.join(output_dir, name + '_5_filled_lane') + ext, filled_lane)


#
# Main
#

def main(name):

    print('Name: {}'.format(name))
    print()

    proc = AdvancedLaneLines()

    if PROCESS_TEST_IMAGES:
        proc.ProcessTestImages('./test_images', './output_images')
#    proc.ProcessTestImages('./test_frames', './test_frames')
    else:
        video_dir = None
        if WRITE_OUTPUT_FRAMES:
            video_dir = './project_video_lanes'
        proc.ProcessVideoClip('./project_video.mp4', video_dir)


if __name__ == '__main__':
    main(*sys.argv)
