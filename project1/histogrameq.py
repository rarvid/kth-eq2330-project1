# Project 1 
# 2 Spatial domain processing
# 2.1 Histogram equalization
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
from PIL import Image

X = cv2.imread('lena512.bmp', cv2.IMREAD_GRAYSCALE)
# Read in image
hist = cv2.calcHist([X], [0], None, [256], [0,256])
# Calculate histogram of image

plt.plot(hist)
plt.title('Histogram of \'lena512.bmp\'')
plt.xlabel('Gray value of pixel')
plt.ylabel('Number of pixels')
# Plot histrogram
plt.savefig('hist.png')
# Save plotted histogram
plt.clf()
# clears figure
###########################################################################

lena = Image.open('lena512.bmp')
# load image
lena_ar = np.asarray(lena)
# make image into array
a = 0.2
b = 50

lc_array = np.minimum(np.maximum((a * lena_ar + b), 0), 255)
# apply low contrast function
lc_img = Image.fromarray(lc_array.astype(np.uint8), 'L').save('lowc_lena.bmp')
# convert low contrast array to image and save

lc_hist = cv2.calcHist([lc_img], [0], None, [256], [0,256])

plt.plot(lc_hist)
plt.title('Histogram of low contrast lena')
plt.xlabel('Gray value of pixel')
plt.ylabel('Number of pixels')
# Plot histrogram
plt.savefig('lc_hist.png')
# Save plotted histogram
plt.clf()
# clears figure




