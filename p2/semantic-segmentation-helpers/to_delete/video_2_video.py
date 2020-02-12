import os
import sys

from PIL import Image
import cv2
import pdb


# Filter out everything not color_mask and order them
def sort_images(image_list):
    return sorted(image_list)

# video_name: doesn't include .mp4
def jpg_2_video():
    video_name = 'new_vid'
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
    image_path = og_video_dir + video_name + '/'
    image_list = os.listdir(image_path)
    images = sort_images(image_list)

    # Set colors
    r1, g1, b1 = 59, 17, 218 # Original value
    white = [255, 255, 255] # Value that we want to replace it with
    black = [0, 0, 0]

    for image_name in enumerate(images):
        cur_image_path = image_path + image_name[1]
        data = cv2.imread(cur_image_path)

        # Get rgb data
        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]

        # Get masks
        mask = (red == r1) & (green == g1) & (blue == b1)
        not_mask = (red != r1) & (green != g1) & (blue != b1)

        # Apply massk
        data[:,:,:][mask] = white
        data[:,:,:][not_mask] = black

        video.write(data)
        print(image_name)

    cv2.destroyAllWindows()
    video.release()


jpg_2_video()

