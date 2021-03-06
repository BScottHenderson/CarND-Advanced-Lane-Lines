
# **Advanced Lane Lines** 

## Scott Henderson
## Self Driving Car Nanodegree Term 1 Project 4

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/test1.jpg "Original"
[image2]: ./output_images/test1_1_undistort.jpg "Undistorted"
[image3]: ./output_images/test1_2_threshold.jpg "Threshold"
[image4]: ./output_images/test1_3_warped.jpg "Threshold Warped"
[image5]: ./output_images/test1_3_warped1.jpg "Undistort Warped"
[image6]: ./output_images/test1_3_warped2.jpg "Undistort Warped Birds-Eye"
[image7]: ./output_images/test1_4_lane_lines.jpg "Warped with Lane Lines"
[image8]: ./output_images/test1_5_filled_lane.jpg "Undistorted with Filled Lane"
[video1]: ./project_video.mp4 "Video"
[video2]: ./project_video_lanes.mp4 "Video With Lanes"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

All code for this project is in the Python script file "./Advanced-Lane-Lines.py".  All line numbers below refer to this source file.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration code can be found in the `calibrate_camera()` function on lines 129-238. Using code taken from the OpenCV web site ( https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) I used the images in the "./camera_cal" folder to construct the camera calibration matrix (C) and distortion coefficients (D).

As an initialization step I created an array of "object points" for a single image. These are the (x, y, z) coordinates of chessboard corners in the real world. Assume that the chessboard is on a fixed plane with z=0 for each calibration image. So we can simply copy this initial array for each calibration image.

For each calibration image I convert the image to grayscale and used the `cv2.findChessboardCorners()` function to find corners. For three of the 20 calibration images this was unsuccessful so these three images were skipped. In the cases where the chessboard corners were found I then used the `cv2.CornerSubPix()` function to further refine the chessboard corner points. A copy of the initial "object point" array was added to the overall list of object points and the refined chessboard corner points were added to the image point array.

After all images were processed I used the `cv2.calibrateCamera()` function to construct the camera calibration matrix and distortion coefficients.  As a final step in the `calibrate_camera()` function I calculate the camera calibration error by using the `cv2.projectPoints()` function to project the object points back to image points and then calculating the average distance between these projected points and the original detected chessboard corners.

The resulting total error was less than 0.11

The camera calibration matrix (C) and distortion coefficients (D) were then used in the image processing pipeline with the `cv2.undistort()` function to remove camera distortion from the input images.


### Pipeline (single images)

Original image:
![alt text][image1]

#### 1. Distortion-corrected image.

The first step in processing an image is distortion correction. This is essentially a one-line step that uses the `cv2.undistort()` function to remove camera distortion from the input images. I created a wrapper function called `undistort()`, lines 241-281, that also includes optional camera matrix refinement code I obtained from the OpenCV web site. I was not able to make this code work so the `undistort()` function is really just a call to the `cv2.undistort()` funciton.

Undistorted image:
![alt text][image2]

#### 2. Threshhold image.

The next step is to apply a combination of color and gradient thresholds to the image. The code for this is in the `color_gradient_threshold()` function, lines 284-340. After my initial implementation - using just an HLS image - had a problem with the test video when the car is turning to the right and passing under a tree I changed my color threshold implementation. The new implementation uses the L channel from an HLS version of the image and the B channel from a LAB version of the image. The goal was to come up with a way to detect both yellow and white lines. I believe this was part of the problem in the initial implementation in that the yellow lane line on the left was being lost in the shadow of a tree.

As before I am using the `cv2.Sobel()` function on the L channel of the HLS image to calculate derivatives on the x axis. I apply thresholds to both color images and the Sobel image and combine the results to form the new image.

Image after thresholds applied:
![alt text][image3]

#### 3. Perspective transformaton.

The next step is to apply a perspective transformation on the image to convert it to bird's-eye view. The code for this is in the `perspective_transform_matrix()` function, lines 343-389. Given source and destination points the `cv2.getPerspectiveTransform()` function is used to calculate the perspective transformation matrix. The source points are four points in the source image that form a quadrilateral that roughly outline the lane. The destination points are the target points for the transformed image.

Transformed image:
![alt text][image4]

To verify that the perspective transformation is working as expected I drew the source points on the original (undistorted) color image:
![alt text][image5]

and the destination points on a warped version of that image:
![alt text][image6]

#### 4. Identify lane line pixels and fit with a polynomial

The next step is to find lane line pixels in the warped image. The code for this is in the `find_lane_pixels()` function, lines 392-486. The basic idea is to use a series of windows on a histogram of image pixels to find points with the highest intensity. Since we're looking at histogram data the intensity corresponds to frequency so we get areas of the image with the highest concentration of pixels. The next step is to fit a quadratic polynomial to these lane line points. The code for this is in the `fit_lane_line_polynomial()` function, lines 489-535. This is a simple matter of using the `np.polyfit()` function to fit a polynomial to the lane line pixels found by the `find_lane_pixels()` function.

The results of applying this process to the sample image are:
![alt text][image7]

This image indicates left lane pixels in blue and right lane pixels in red. The windows used to find the lane line pixels are drawn in green. The curves fitted to each collection of points is shown in yellow.

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated using the `measure_curvature()` function, lines 538-564. This function assumes that the coefficients given for the left and right lane polynomials are calculated from coordinates that were converted from pixels to meters. The y value corresponding to the bottom of the image is used - after conversion to meters - in the R curve formula to calculate the radius of curvature for both the left and right lane lines.

The offset from lane center is calculated in the `AdvancedLaneLines.ProcessTestImages()` function (or the `LaneLine.AddFrame()` function on lines 89-126, see Note below), lines 841-914. The idea here is to use the "meters" version of the polynomial coefficients to calculate the x position of each lane line for the bottom of the image and then calculate the distance to the center of the image (assume to be the center of the vehicle). The two distances are averaged and the result becomes the offset from center for the vehicle.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final result using the fitted curve for each lane line is shown in this image:

![alt text][image8]

The radius of curvature for both the left and right lane lines, as well as the offset of the vehicle from the center of the lane, is displayed in the upper left corner.

---
Note:
The code described above was used to process the test images in the "./test_images" folder. There is some duplication of code between the `AdvancedLaneLines.FindLaneLines()` method, lines 704-798, which uses the `LaneLines.AddFrame()` method, lines 89-126, and the `AdvancedLaneLines.ProcessTestImages()` method, lines 841-914. However the code is essentially the same in both cases.
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_lanes.mp4)

---

### Discussion

After my initial project submission there were two areas in the output video that needed addressing. The first was just an initialization issue with the first few frames and was easily resolved. The second was a problem with a few frames around the last curve (to the right) where a shadow from a tree falls across the lane just as the lane is beginning to curve. The lane lines became quite confused for a short bit here in the initial implementation. To resolve this I changed the color threshold code (see description above) and this seems to have resolved the issue for the project video (I have not tried either of the challenge videos due to lack of time).

I have not implemented any sort of sanity checks with respect to lane curvature radii or polynomial fit coefficients or any other potential checks. I believe this is a necessary step for the more difficult videos and, needless to say, real world conditions. For now the smoothing that happens in the `LaneLine.AddFrame()` method will have to suffice.

The values I used for the color and gradient thresholds were obtained via trial and error. A very little bit of trial and error and then only with reference to the very small test of test images in the "./test_images" directory and also a few frames around the problem area discovered in the initial implementation. While the resulting values seemed to have worked well enough for the project video I have not tested the process against either one of the challenge videos and, in general, I feel that more effort could go into refining these values. Perhaps even a neural network of some type.

Similarly with the constants (number of windows, margin size, minimum pixel size) used for finding lane line pixels. The default values seemed to work well enough for the project video but the real world is a complicated place.
