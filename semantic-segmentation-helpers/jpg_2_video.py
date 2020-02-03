import os
import sys

from PIL import Image
import cv2
import pdb

def sort_images(image_list):
    # TODO
    pass

# video_name: doesn't include .mp4
def jpg_2_video(video_name):
    og_video_dir = './videos/'
    og_video_path = og_video_dir + video_name + '.mp4' # edit me

    # Get info for new video and release it
    cap = cv2.VideoCapture(og_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Initialize new video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    new_video_name = og_video_dir + video_name + '_segmented.mp4'
    video = cv2.VideoWriter(video_name, fourcc, fps, (image_size, image_size))

    # Get images
    image_path = og_video_dir + video + '_segmented/'
    image_list = os.listdir(image_path)
    images = sort_images(image_list) # TODO

    for image_path in enumerate(images):
        video.write(cv2.imread(image_path))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print('Wrong number of args: python jpg_2_video.py test')
    else:
        # execute only if run as a script
        main(sys.argv[-1])
