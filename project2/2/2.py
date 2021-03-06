#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:16:40 2020

@author: jeremy
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio


#%% 2.1

A = np.zeros((8,8))
for i in range(8):
    for k in range(8):
        a = i == 0 and np.sqrt(1/8) or np.sqrt(2/8)
        A[i, k] = a * np.cos((2*k+1) * i * np.pi / 16)

def dct8(x, M=A):
    return np.dot(M, np.dot(x, M.T))

def idct8(y, M=A):
    return np.dot(M.T, np.dot(y, M))

#%% tests
x = np.array([[0.0,0.5,1.,-.5,0.0,.5,1,-.5], \
              [0.0,0.5,1.,-.5,0.0,.5,1,-.5], \
              [0.0,0.5,1.,.5,0.0,.5,1,-.5], \
              [0.0,0.5,1.,.5,0.0,.5,1,-.5], \
              [0.0,0.5,1.,-.5,0.0,-.5,1,-.5], \
              [0.0,0.5,1.,-.5,0.0,-.5,1,-.5], \
              [0.0,0.5,1.,-.5,0.0,.5,1,-.5], \
              [0.0,0.5,1.,-.5,0.0,.5,1,-.5]])
print("x =", x)
print("DCT(x) =", dct8(x))
print("iDCT(y) =", idct8(dct8(x)))

#%% 2.2

def uniform_quantizer(x, step_size=0.5):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        for j in range(len(y[0])):
            y[i,j] = ((x[i,j] + step_size/2) // step_size) * step_size
    return y

#%% Plot a graph of the quantizer function.

xx = np.arange(-5, 5.1, 0.25)
xx = [ [x] for x in xx]
xx = np.array(xx)
yy = uniform_quantizer(xx, 1.5)
plt.figure()
plt.grid()
plt.plot(xx,yy)
plt.show()

#%% 2.3

def distortion(orig, reconstruct):
    d = 0.0
    for i in range(len(orig)):
        for j in range(len(orig[0])):
            d += (orig[i,j] - reconstruct[i,j])**2
    
    return d

def average_distortion(orig, reconstruct):
    return distortion(orig, reconstruct) / len(orig) / len(orig[0])

#%%Compare d with the mean squared error between the original and
#   the quantized DCT coefficients.

"""
So we want to compare the error generated by the quantizer.
What step_size with? Simple: we iterate over a range a values for step_size
and plot the result. If the instructions are not clear, I interpolate them.
"""
step_sizes = [0.01, .05,.075,.1,.15,0.2,.25, .3,.35, 0.4, 0.425, .45,.475,0.5,.525,.55,.575,.6]
distortion_image = []
distortion_coeff = []
image = skio.imread("rsrc/airfield512x512.tif")
image = image[:8, :8]
image = image.astype(np.float64)

for step_size in step_sizes:
    B = uniform_quantizer(A, step_size)
    distortion_coeff.append(average_distortion(A, B))
    distortion_image.append(average_distortion(image, idct8(dct8(image, B), B)))
    #psnr.append(10 * np.log10 (65025.0 / d))

plt.figure()
plt.xlabel("step size")
plt.plot(step_sizes, distortion_image, label="distortion of reconstruction")
plt.plot(step_sizes, distortion_coeff, label="distortion of DCT coefficients")
plt.legend()
plt.yscale("log")
plt.show()

#%% 2.3

def find_vlc_length(blocks):
    """
    Calculate the entropies of each cofficients amoung block to infer the ideal
    VLC codeword length. parameters blocks is a list of 64x64 numpy arrays whose
    coefficent have been quantized.
    
    The function returns the TOTAL LENGTH, or the bit rate.
    It means the sum of entropies. ie the sum of the average lengths of all
    64 VLCs.
    """
    
    block_count = len(blocks)
    total_entropy = 0.0
    
    for i in range(len(blocks[0])):
        for j in range(len(blocks[0][0])):
            
            # Compute the occurences of all the observed of the (i,j)th coef.
            
            nb_occurrences = []
            mapping       = []
            
            for block in blocks:
                coef = block[i,j]
                
                if mapping.count(coef) == 0:
                    mapping.append(coef)
                    nb_occurrences.append(1)
                else:
                    index = mapping.index(coef)
                    nb_occurrences[index] += 1
            
            # Now we know all the probability of occurences: we can
            # calculate the entropy.
            
            entropy = 0.0
            for nb in nb_occurrences:
                p = nb / block_count
                entropy += p * np.log2(1/p)
            
            total_entropy += entropy
        
    return total_entropy

def psnr_bitrate_estimator(step_size):
    """ C'est ici que le sale se produit. """
    """

original pixel  ->  DCT  -> quantized  ->  inverse DCT ->  reconstructed pixel
    |                                                              |
    |                                                              |
     \------------------>  compare  <-----------------------------/
                              |
                              -----> PSNR
    """
    
    
    blocks = []
    total_distortion = 0.0
    
    image_source = ["boats512x512.tif","harbour512x512.tif","peppers512x512.tif"]
    
    for image_name in image_source:
        print("loading ", image_name)
        image = skio.imread("rsrc/"+image_name)
        image = image.astype(np.float64)
        image = image/255.
        
        for i in range(0,512,8):
            for j in range(0,512,8):
                # extract a 8x8 block.
                blk = image[i:i+8, j:j+8]
                # apply DCT + quantization
                blk = dct8(blk)
                blk = uniform_quantizer(blk, step_size)
                
                # perform iDCT and compute the distortion on it.
                total_distortion += distortion(idct8(blk), image[i:i+8, j:j+8])
                
                # store in the big block list, to compute the bit with the VLC later
                blocks.append(blk)
    
    # Now, we need to know the bit-rate, from the VLCs total theoretical length.
    bit_rate = find_vlc_length(blocks)

    total_distortion = total_distortion / (512 * 512 * len(image_source))
    psnr = 10 * np.log10 (65025.0 / total_distortion)

    return psnr, bit_rate


#%% big tests

step_sizes = [0.01, .05,.075,.1,.15,0.2,.25, .3,.35, 0.4, 0.425, .45,.475,0.5,.525,.55,.575,.6]

psnrs = []
bit_rates = []

for step_size in step_sizes:
    print("iteration with step size", step_size)
    result = psnr_bitrate_estimator(step_size)
    psnrs.append(result[0])
    bit_rates.append(result[1])
print("Done!")

#%% plotting the big tests' results

plt.figure()
plt.xlabel("optimal bit rate (bit)")
plt.ylabel("PSNR 'dB)")
plt.plot(bit_rates, psnrs)
plt.show()

#%%

# A little demonstration of my flawless DCT implementation.

image = skio.imread("rsrc/peppers512x512.tif")
image = image.astype(np.float64)
image = image/255.

for i in range(0,512,8):
    for j in range(0,512,8):
        blk = image[i:i+8, j:j+8]
        blk = dct8(blk)
        blk = uniform_quantizer(blk, 0.05)
        image[i:i+8, j:j+8] = blk
plt.figure()
plt.imshow(np.log(np.abs(image[64:196,64:196])+0.000001), cmap="gray")
for i in range(0,512,8):
    for j in range(0,512,8):
        blk = image[i:i+8, j:j+8]
        blk = idct8(blk)
        image[i:i+8, j:j+8] = blk

plt.figure()
plt.imshow(image, cmap="gray")





#%%







if __name__ == "__main__":
    #
    print()
