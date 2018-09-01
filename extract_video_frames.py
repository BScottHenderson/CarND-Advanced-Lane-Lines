# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:26:47 2018

@author: henders
"""

import os
import cv2


# Image text parameters
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_COLOR = (255, 255, 255)
TEXT_THICKNESS = 2
TEXT_LINE_TYPE = cv2.LINE_AA


video_file = './project_video.mp4'
output_dir = './project_video_frames'


vidcap = cv2.VideoCapture(video_file)
if not vidcap.isOpened():
    print('Unable to open video file {}'.format(video_file))
else:
    current_frame = 0
    while vidcap.isOpened():
        # Read a frame from the video file.
        ret, img = vidcap.read()
        if ret:
            current_frame += 1
            print('Frame {}'.format(current_frame))

            # Write the frame number to the image.
            frame = 'Frame: {}'.format(current_frame)
            cv2.putText(img, frame, (1050, 30),
                        TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

            # Save the frame image to a file.
            cv2.imwrite(os.path.join(output_dir, 'frame{:06d}.jpg'.format(current_frame)), img)
        else:
            break
