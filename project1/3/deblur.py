#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nice meme

"""

This script must be in the same folder as its resources file (eg lena512.bmp)
in order to work. I created symbolic link, but the latter are not on the git
repository.

cd /project1/3

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d as conv2
from skimage import io as skio

import myblurgen

fft2=np.fft.fft2
ifft2=np.fft.ifft2

#%%

"""
Functions to plot your amazing images or their Fourier sprectra.
"""

def plot_image(img, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.show()

def plot_spectre(img, title=""):
    """ Take an image, plot a correctly shifted graph its Fourier spectrum."""
    plt.figure()
    plt.title(title)
    plt.imshow(np.log(np.abs(np.fft.fftshift(fft2(img)))+0.000001))#, cmap="gray")
    plt.show()

#%%
# Load the image.
im = skio.imread("man512_outoffocus.bmp")
# remove the colors if it's not a grey image.
if len(im.shape) == 3:
    im = im[:,:,0]
im = im.astype(np.float64)
im = im/255.

# Instanciate the degradation filter.
h = myblurgen.myblurgen("gaussian", 8)

# Compute the degraded image g = h conv f
blurred = conv2(im,h, mode="same", boundary="wrap")

#%%

plot_spectre(im, "Spectrum before degradation")
plot_spectre(blurred, "Spectrum after degradation")
plot_spectre(blurred - im)

#%%

"""
Deblurring / denoising filters.
"""

def weiner(g, h, var):
    """ Who reads documentation? """
    G = fft2(g)
    if h.shape != g.shape:
        hext = np.zeros(G.shape, G.dtype)
        x = len(hext)//2 - len(h)//2
        hext[x:x+len(h), x:x+len(h)] = h
        h = hext
        
    H = fft2(h)
    
    W = np.zeros(G.shape, G.dtype)
    
    W = np.conj(H) / (0.000001+ np.abs(H)*np.abs(H) + var / np.abs(G)*np.abs(G))
    
    return np.fft.fftshift(np.real(ifft2(W*G)))


def nonlinear_denoiser(x, var):
    """doc"""
    
    size = len(x)
    # High frequencies under this limit are considered to belong to the noise.
    threshold = var
    # a value that tells how low the low-pass filter is.
    pass_limit = (len(x) // 5)
    
    # first we define the non-linear function that will tamper will the amplitude
    def amplitude(u, v, F):
        # if the frequency is low, then it passes. It's a low-pass after all.
        #if min(u,v) < pass_limit and max(u,v) > size - pass_limit:
        if (u < pass_limit and v < pass_limit) \
                or (u < pass_limit and v > size-pass_limit)\
                or (u > size-pass_limit and v < pass_limit)\
                or (u > size-pass_limit and v > size-pass_limit):
            return F
        # High frequencies over the threshold are attenuated.
        if np.abs(F) > threshold:
            a = 1/(np.abs(F) / threshold)
            a = a*a
            return a * F
        # Otherwise they pass as well.
        else:
            return F

    X = fft2(x)

    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i,j] = amplitude(i, j, X[i,j])
    
    
    return np.real(ifft2(X))


#%%

b = nonlinear_denoiser (blurred, 0.02)
y = weiner(b, h, 0.0)

plot_image(blurred)
plot_image(y)

#%%
print("=================================")

plot_image(im, "blurred noised image")
plot_spectre(im)

denoised = nonlinear_denoiser (im, 0.0833)
plot_image(denoised, "denoised")
plot_spectre(denoised)

y = weiner(denoised, h, 0.0)
plot_image(y, "restored")
plot_spectre(y)
