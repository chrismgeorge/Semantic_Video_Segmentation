import cv2
import numpy as np
import os
from PIL import Image
import random
import sys
import pdb
import imutils

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# vid_file:   is the file path to the video
# new_folder: is the folder path for the images
def main():
    for video in os.listdir('./videos/'):
        if ('mp4' in video): # ignore extra files
            # Set video to cv2
            cap = cv2.VideoCapture('./videos/' + video)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("reading " + str(frame_count) + " frames from " + video)

            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_FPS, fps)

            # Get/ make new folder name
            video_name = video
            new_folder = './videos/' + video[:-4] + '_images/'
            checkDirectory(new_folder)

            # Create directory for segmented images
            new_folder_segmented = './videos/' + video[:-4] + '_color_mask/'
            checkDirectory(new_folder_segmented)

            new_folder_segmented = './videos/' + video[:-4] + '_color_mask_bw/'
            checkDirectory(new_folder_segmented)

            # Create directory for segmented images
            new_folder_segmented = './videos/' + video[:-4] + '_transparent_person/'
            checkDirectory(new_folder_segmented)

            new_folder_segmented = './videos/' + video[:-4] + '_transparent_background/'
            checkDirectory(new_folder_segmented)

            # Iterate and retrieve the frames
            i = 0
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break
                name = new_folder + '{num:0{width}}'.format(num=i, width=6) + '.jpg'
                frame = imutils.resize(frame, width=1280) # decrease size to 720p
                cv2.imwrite(name, frame)
                i += 1
                if i % 100 == 0:
                    print(i, '/', frame_count)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

main()
