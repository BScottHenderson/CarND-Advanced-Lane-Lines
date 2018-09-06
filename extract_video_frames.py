# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:26:47 2018

@author: henders
"""

import os
import cv2

from moviepy.editor import VideoFileClip


# Image text parameters
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_COLOR = (255, 255, 255)
TEXT_THICKNESS = 2
TEXT_LINE_TYPE = cv2.LINE_AA


class WriteFrame:
    def __init__(self):
        # Current frame number.
        self.current_frame = 0
        # Output dir for modified video frames.
        self.output_dir = None

    def write_frame(self, get_frame, t):
        self.current_frame += 1

        img = get_frame(t)

        # Write the frame number to the image.
        frame = 'Frame: {}'.format(self.current_frame)
        cv2.putText(img, frame, (1050, 30),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # Write the time (parameter t) to the image.
        time = 'Time: {}'.format(int(round(t)))
        cv2.putText(img, time, (1050, 700),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        if self.output_dir is not None:
            output_file = os.path.join(self.output_dir,
                                       'frame{:06d}.jpg'.format(self.current_frame))
            filled_lane_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_file, filled_lane_cv2)

        # Return the modified image.
        return img


wf = WriteFrame()
wf.output_dir = './project_video_frames'

current_frame = 0
input_file = './project_video.mp4'
input_clip = VideoFileClip(input_file)

output_clip = input_clip.fl(wf.write_frame)

# Save the resulting, modified, video clip to a file.
head, ext = os.path.splitext(input_file)
root, file_name = os.path.split(head)
output_file = os.path.join(wf.output_dir, file_name + '_lanes' + ext)
output_clip.write_videofile(output_file, audio=False)

# Cleanup
input_clip.reader.close()
input_clip.audio.reader.close_proc()
del input_clip
output_clip.reader.close()
output_clip.audio.reader.close_proc()
del output_clip
