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

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--demo-folder', type=str, default='', help='path to the folder containing demo images', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

# get net
args.dataset_cls = cityscapes
net = network.get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
print('Net restored.')

color_dict = {
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

# get data
data_dir = args.demo_folder
images = os.listdir(data_dir)
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

start_time = time.time()
for img_id, img_name in enumerate(images):
    img_dir = os.path.join(data_dir, img_name)
    img = Image.open(img_dir).convert('RGB')
    img_tensor = img_transform(img)

    # predict
    with torch.no_grad():
        pred = net(img_tensor.unsqueeze(0).cuda())
        print('%04d/%04d: Inference done.' % (img_id + 1, len(images)))

    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)

    # Save colorized results
    color_name = 'color_mask/' + img_name
    colorized = args.dataset_cls.colorize_mask(pred)
    colorized.convert('RGB').save(os.path.join(args.save_dir, color_name))

end_time = time.time()

print('Results saved.')
print('Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results.' % (end_time - start_time, (end_time - start_time)/len(images)))



