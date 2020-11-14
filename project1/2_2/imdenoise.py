# Project 1 
# 2 Spatial domain processing
# 2.2 Image denoising
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
from PIL import Image
from mynoisegen import mynoisegen
X = cv2.imread('../lena.512.bmp')

n = mynoisegen('saltpepper', 512, 512, 0.05, 0.05);

n = np.array(n)
Image.fromarray(n.astype(np.uint8), "L").save("noisetest.bmp")