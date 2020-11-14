#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nice meme

"""

This script must be in the same folder as its resources file (eg lena512.bmp)
in order to work. I created symbolic link, but the latter are not on the git
repository.

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
    plt.imshow(np.log(np.abs(np.fft.fftshift(fft2(img)))+0.000001), cmap="gray")
    plt.show()

#%%
# Load the image of lena.
im = skio.imread("lena512.bmp")
im = im.astype(np.float64)
im = im/255.

# Instanciate the degradation filter.
h = myblurgen.myblurgen("gaussian", 8)

# Compute the degraded image g = h conv f
blurred = conv2(im,h, mode="same", boundary="wrap")

#%%

plot_spectre(im, "Spectra before degradation")
plot_spectre(blurred, "Spectra after degradation")
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
    





#%%

y = weiner(blurred, h, 0.0)

plot_image(y)

