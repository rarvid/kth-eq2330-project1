# Project 1 
# 2 Spatial domain processing
# 2.2 Image denoising
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
from PIL import Image
from mynoisegen import mynoisegen
import scipy.signal as sps

im_saltp = cv2.imread('../lena512.bmp', cv2.IMREAD_GRAYSCALE)
im_gaus = cv2.imread('../lena512.bmp', cv2.IMREAD_GRAYSCALE)

# load in Lena
n = mynoisegen('saltpepper', 512, 512, 0.05, 0.05);
# generate s&p noise
Image.fromarray(np.array(255 * n).astype(np.uint8), "L").save("noisetest.png")
# save noise as image (debugging purposes)

gausN = mynoisegen('gaussian', 512, 512, 0, 64)
# generate gaussian noise
cv2.imwrite('gausNoiseTest.png',255 * gausN)
# save noise as image (debugging purposes)

###########################Clean and Noist image creation and histograms####################
cleanHist = cv2.calcHist([im_saltp], [0], None, [256], [0,256])
cleanHistpdf = cleanHist/(512*512)

plt.plot(cleanHistpdf)
plt.title('Histogram of \'lena512.bmp\'')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('cleanHistpdf.png')
# Save plotted histogram
plt.clf()
# clears figure

##############################salt and peppa##############################
im_saltp[n==0] = 0
im_saltp[n==1] = 255
# add s&pnoise
cv2.imwrite("noiseLena.png", im_saltp)
# write to file noisy image

noiseHist = cv2.calcHist([im_saltp], [0], None, [256], [0,256])
# calc noisy image histogram
noiseHistpdf = noiseHist/(512*512)
# calulate is pdf

plt.plot(noiseHistpdf)
plt.title('Histogram of Lena with salt&pepper noise')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('noiseHistpdf.png')
# Save plotted histogram
plt.clf()
# clears figure

##############################gaussian noise#############################
gausnoiseLena = im_gaus + gausN
# Lena with gaussian noise
cv2.imwrite('gausnoiseLena.png', gausnoiseLena)
# write lena with gaussian noise to file

gausNoiseHist = cv2.calcHist([np.uint8(gausnoiseLena)], [0], None, [256], [0,256])
# calc noisy image histogram
gausNoiseHistpdf = gausNoiseHist/(512*512)
# calulate is pdf

plt.plot(gausNoiseHistpdf)
plt.title('Histogram of Lena with Gaussian noise')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('gausNoiseHistpdf.png')
# Save plotted histogram
plt.clf()
# clears figure

#################################Mean filter implementation#############################
####################salt and peppa#####################
meanFilt = np.ones((3,3)) * (1/9)
# generate 3 x 3 mean filter
meanFiltered = sps.convolve2d(im_saltp, meanFilt, mode="full" , boundary="symm")
# apply 3 x 3 mean filter and use symmetrical boundary conditions (mirror boder pixels)
cv2.imwrite('meanFilteredLena.png', meanFiltered)
# save filtered image

meanFilteredHist = cv2.calcHist([np.uint8(meanFiltered)], [0], None, [256], [0,256])
# calc mean filtered image histogram
meanFilteredHistpdf = meanFilteredHist/(512*512)
# calulate hist pdf

plt.plot(meanFilteredHistpdf)
plt.title('Histogram of mean filtered Lena with s&p noise')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('meanFilteredHistpdf.png')
# Save plotted histogram
plt.clf()
# clears figure

##################gauss############################
meanFilteredGaus = sps.convolve2d(gausnoiseLena, meanFilt, mode='full', boundary='symm')\

cv2.imwrite('meanFilteredGausLena.png', meanFilteredGaus)

meanFilteredGausHist = cv2.calcHist([np.uint8(meanFilteredGaus)], [0], None, [256], [0,256])

meanFilteredGausHistpdf = meanFilteredGausHist/(512 * 512)

plt.plot(meanFilteredGausHistpdf)
plt.title('Histogram of mean filtered Lena with gaussian noise')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('meanFilteredGausHistpdf.png')
# Save plotted histogram
plt.clf()
# clears figure

######################################Median filter implementation#######################
######################salt and peppa#######################
medianFiltered = sps.medfilt2d(im_saltp,3)
# apply 3 x 3 median filter to image
cv2.imwrite("medianFilteredLena.png", medianFiltered)
# save median filtered image

medianFilteredHist = cv2.calcHist([np.uint8(medianFiltered)], [0], None, [256], [0,256])
# calc median filtered image histogram
medianFilteredHistpdf = medianFilteredHist/(512*512)
# calulate hist pdf

plt.plot(medianFilteredHistpdf)
plt.title('Histogram of median filtered Lena with s&p noise')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('medianFilteredHistpdf.png')
# Save plotted histogram
plt.clf()
# clears figure

#####################gaussian noise######################
medianFilteredGaus = sps.medfilt2d(gausnoiseLena, 3)

cv2.imwrite('medianFilteredGausLena.png', medianFilteredGaus)\

medianFilteredGausHist = cv2.calcHist([np.uint8(medianFilteredGaus)], [0], None, [256], [0,256])

medianFilteredGausHistpdf = medianFilteredGausHist/(512 * 512)

plt.plot(medianFilteredGausHistpdf)
plt.title('Histogram of median filtered Lena with gaussian noise')
plt.xlabel('Gray value of pixel')
plt.ylabel('p(r)')
# Plot histrogram
plt.savefig('medianFilteredGausHistpdf.png')
# Save plotted histogram
plt.clf()
# clears figure






