import cv2
import numpy as np
import os
from PIL import Image
import random
import sys
import pdb

# vid_file:   is the file path to the video
# new_folder: is the folder path for the images
def main(vid_file, new_folder):
    # Setup for getting video content
    cap = cv2.VideoCapture(vid_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FPS, fps)
    print("reading " + str(frame_count) + " frames from " + vid_file)

    # Iterate and retrieve the frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        name = new_folder + '{num:0{width}}'.format(num=i, width=6) + '.jpg'
        cv2.imwrite(name, frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) is not 3:
        print("Usage: python3 video_2_jpg.py video_destin new_folder")
    else:
        video_name = sys.argv[1]
        folder_destination = sys.argv[2]
        main(video_name, folder_destination)
