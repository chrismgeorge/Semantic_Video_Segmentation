import os
import sys
import time
import argparse
from PIL import Image
import numpy as np
import cv2
import pdb

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

## Globals
COLOR_DICT = {
        'road': [128, 64, 128],
        'sidewalk': [244, 35, 232],
        'building': [70, 70, 70],
        'wall': [102, 102, 156],
        'fence': [190, 153, 153],
        'pole': [153, 153, 153],
        'traffic light': [250, 170, 30],
        'traffic sign': [220, 220, 0],
        'vegetation': [107, 142, 35],
        'terrain': [152, 251, 152],
        'sky': [70, 130, 180],
        'person': [220, 20, 60],
        'rider': [255, 0, 0],
        'car': [0, 0, 142],
        'truck': [0, 0, 70],
        'bus': [0, 60, 100],
        'train': [0, 80, 100],
        'motorcycle': [0, 0, 230],
        'bicycle': [119, 11, 32]
    }

# Doesn't make videos from pngs yetß
# def jpg_2_video(og_video_path, new_video_name, image_dir):
#     cap = cv2.VideoCapture(og_video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     cap.release()

#     # Initialize new video,
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video = cv2.VideoWriter(new_video_name, fourcc, fps, (w, h))

#     # Get images
#     image_list = os.listdir(image_dir)
#     image_list.sort()

#     for image_name in images:
#         if ('.png' in image_name or '.jpg' in image_name):
#             cur_image_path = image_dir + image_name
#             data = cv2.imread(cur_image_path, cv2.IMREAD_UNCHANGED)
#             video.write(data)
#             print(image_name)

def main(args):
    # Get and build net
    args.dataset_cls = cityscapes
    net = network.get_net(args, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    print('Net built.')
    net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
    net.eval()
    print('Net restored.')

    # Initialize colors for color mask
    r1, g1, b1 = COLOR_DICT['person']
    white = [255, 255, 255]
    black = [0, 0, 0]

    # Get and check for data
    data_dir = args.demo_folder
    images = os.listdir(data_dir)
    if len(images) == 0: print('No images in directory %s.' % (data_dir))
    else: print('There are %d images to be processed.' % (len(images)))
    images.sort()

    # Transform images according to datasetup
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Start timer and looping over images
    start_time = time.time()
    for img_id, img_name in enumerate(images):

        # Get image tensor
        img_dir = os.path.join(data_dir, img_name)
        img = Image.open(img_dir).convert('RGB')
        img_tensor = img_transform(img)

        # Run inference on image
        with torch.no_grad():
            pred = net(img_tensor.unsqueeze(0).cuda())
            if img_id % 10 == 0:
                print('%04d/%04d: Inference done.' % (img_id + 1, len(images)))

        # Select image values based on output
        pred = pred.cpu().numpy().squeeze()
        pred = np.argmax(pred, axis=0)

        ### Save colorized results #################################
        colorized = dataset_cls.colorize_mask(pred)
        colorized = colorized.convert('RGB')
        base_dir = args.save_dir.split('_')[0]

        ### Save 1 for regular color mask
        save_1 = base_dir + '_color_mask/' + img_name
        colorized.save(save_1)

        ### Save 2, black and white color mask #################################
        colorized = np.array(colorized)
        data = np.zeros(colorized.shape)

        # Get rgb data
        red, green, blue = colorized[:,:,0], colorized[:,:,1], colorized[:,:,2]

        # Get masks
        mask = (red == r1) & (green == g1) & (blue == b1) # person
        not_mask = (red != r1) & (green != g1) & (blue != b1) # background

        # Apply massk
        data[:,:,:][mask] = white
        data[:,:,:][not_mask] = black

        save_2 = base_dir + '_color_mask_bw/' + img_name
        cv2.imwrite(save_2, data)

        ### Save 3 for person transparency #################################
        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        alpha = (mask * 255).astype(blue.dtype)
        img_RGBA = cv2.merge((red, green, blue, alpha))
        alpha_name = img_name.split('.')[0] + '.png'
        save_3 = base_dir + '_transparent_person/' + alpha_name
        cv2.imwrite(save_3, img_RGBA)

        ### Save 4 for background transparency #################################
        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        alpha = (not_mask * 255).astype(blue.dtype)
        img_RGBA = cv2.merge((red, green, blue, alpha))
        alpha_name = img_name.split('.')[0] + '.png'
        save_4 = base_dir + '_transparent_background/' + alpha_name
        cv2.imwrite(save_4, img_RGBA)

    end_time = time.time()

    print('Results saved.')
    print('Total Inference time %4.2f seconds,' % (end_time - start_time))
    print('which is %4.2f seconds per image.' % ((end_time - start_time)/len(images)))

    # jpg_2_video('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--demo-folder', type=str, default='', help='path to the folder containing demo images', required=True)
    parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
    parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
    parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
    current_args = parser.parse_args()
    assert_and_infer_cfg(current_args, train_mode=False)
    cudnn.benchmark = False
    torch.cuda.empty_cache()
    main(current_args)



