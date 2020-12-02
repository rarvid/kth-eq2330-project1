
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
import pywt
import scipy.signal as scs
from PIL import Image
import math

def quantizer(x, step_size):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        for j in range(len(y[0])):
            y[i,j] = ((x[i,j] + step_size/2) // step_size) * step_size
    return y

# z = Image.open('harbour.png')
# step = (256/2**1)
# im = quantizer(np.array(z, dtype=np.float64), step)
# plt.imsave('test1.png', im, cmap=plt.cm.gray)

# step = (256/2**2)
# im = quantizer(np.array(z, dtype=np.float64), step)
# plt.imsave('test2.png', im, cmap=plt.cm.gray)
# plt.imsave("quantized.png", im_q, cmap=plt.cm.gray)

