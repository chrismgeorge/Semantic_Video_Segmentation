import cv2
import numpy as np
import os
from PIL import Image
import random
import sys
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data = cv2.imread('./videos/new_vid/000415.jpg')

r1, g1, b1 = 59, 17, 218 # Original value
r2, g2, b2 = 255, 255, 255 # Value that we want to replace it with

red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
mask = (red == r1) & (green == g1) & (blue == b1)
data[:,:,:][mask] = [r2, g2, b2]

not_mask = (red != r1) & (green != g1) & (blue != b1)
data[:,:,:][not_mask] = [0, 0, 0]

# r1, g1, b1 = 0, 0, 252 # Original value
# r2, g2, b2 = 255, 255, 255 # Value that we want to replace it with

# red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
# mask = (red == r1) & (green == g1) & (blue == b1)
# data[:,:,:3][mask] = [r2, g2, b2]

plt.imshow(data)
plt.show()

