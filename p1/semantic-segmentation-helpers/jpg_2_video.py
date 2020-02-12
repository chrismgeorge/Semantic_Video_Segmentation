import os
import sys

from PIL import Image
import cv2
import pdb

# Filter out everything not color_mask and order them
def sort_images(image_list):
    return sorted([x for x in image_list if 'color' in x])

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
    video = cv2.VideoWriter(new_video_name, fourcc, fps, (w, h))

    # Get images
    image_path = og_video_dir + video_name + '_segmented/' + 'color_mask/'
    image_list = os.listdir(image_path)
    images = sort_images(image_list)

    for image_name in enumerate(images):
        cur_image_path = image_path + image_name[1]
        video.write(cv2.imread(cur_image_path))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print('Wrong number of args: python jpg_2_video.py test')
    else:
        # execute only if run as a script
        main(sys.argv[-1])
