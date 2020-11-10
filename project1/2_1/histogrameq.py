# Project 1 
# 2 Spatial domain processing
# 2.1 Histogram equalization
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
from PIL import Image

############################## Histogram #########################################
X = cv2.imread('lena512.bmp', cv2.IMREAD_GRAYSCALE)
# Read in image
hist = cv2.calcHist([X], [0], None, [256], [0,256])
# Calculate histogram of image
histpdf = hist/(512*512)
# calulate pdf of image

plt.plot(histpdf)
plt.title('Histogram of \'lena512.bmp\'')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('histpdf.png')
# Save plotted histogram
plt.clf()
# clears figure
############################### Low-contrast img and histogram ##################

lena = Image.open('lena512.bmp')
# load image
lena_ar = np.asarray(lena)
# make image into array
a = 0.2
b = 50

lc_array = np.minimum(np.maximum((a * lena_ar + b), 0), 255)
# apply low contrast function
Image.fromarray(lc_array.astype(np.uint8), 'L').save('lc_lena.bmp')
# convert low contrast array to image and save

lc_img = cv2.imread('lc_lena.bmp', cv2.IMREAD_GRAYSCALE)
# load low contrast image
lc_hist = cv2.calcHist([lc_img], [0], None, [256], [0,256])
# calculate histogram of low contrast image
lc_histpdf = lc_hist/(512*512)
plt.plot(lc_histpdf)
plt.title('Histogram of low contrast lena')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('lc_histpdf.png')
# Save plotted histogram
plt.clf()
# clears figure

################################# Histogram eqaulization #######################
cmf = np.cumsum(lc_histpdf)
# cmf or trasformation function for histogram eqaulization
plt.plot(cmf)
plt.title('CMF of low contrast lena')
plt.xlabel('Gray value of pixel')
plt.ylabel('s(r)')
# Plot histrogram
plt.savefig('lc_histcmf.png')
# Save plotted histogram
plt.clf()
# clears figure

lc_histcmf = np.round(255 * cmf)
# rounded and scaled cmf for image histogram equalization

lc_img_eq = np.zeros((512,512))
# initialized zero array for new image

for i in range(512):
    for j in range(512):
        g = lc_img[j,i]
        lc_img_eq[j,i] = lc_histcmf[g]
# histogram equalization

Image.fromarray(lc_img_eq.astype(np.uint8), 'L').save('lc_lena_eq.bmp')
# same equalized histogram as image

eq_img = cv2.imread('lc_lena_eq.bmp', cv2.IMREAD_GRAYSCALE)
eq_hist = cv2.calcHist([eq_img], [0], None, [256], [0,256])
# recaululate the equalized histogram

eq_hist = eq_hist/(512*512)
plt.plot(eq_hist)
plt.title('Equalized histogram of low-contrast lena')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('eq_hist.png')
# Save plotted histogram
plt.clf()
# clears figure