
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
import pywt
import scipy.signal as scs
from PIL import Image
from quantizer import quantizer
import math

# Global variable for recursion purposes
# Count is used for to obtain a specific depth recursive DWT
# coeff_arr is an array used to save tuples with coefficients after each recursive
# application of the DWT
count = 0
another = 0
coeff_arr =[]

def MSE (im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    err = np.mean(np.square(np.subtract(im1,im2)))
    return err

def MSEC (c1, c2):
    assert(len(c1) == len(c2))
    sum = 0
    for i in range(len(c1)):
        sum += np.mean(np.square(np.subtract(c1[i], c2[i])))
    
    
    return sum

def filter(pic, depth):
    
    # Application of the dwt using 5/3 filter and a periodic copy
    # to address border effects
    coeffs = pywt.dwt2(pic, 'bior2.2', mode='periodization')
    # Indivudal coefficient matrices of Low-Low, High-Low, Low-High, High-High frequencies
    LL, (LH, HL, HH) = coeffs

    # Specify recursion depth and
    # apply the 5/3 filter on the Low-Low frequencies
    global count
    if count < depth:
        count = count + 1
        LL  = filter(LL, depth)    
    
    # Resizing of resulting arrays to get optimal size (512,256,128,64,32)
    trunc_size = len(LL) - (len(LL) % 8)
    LL = LL[:trunc_size, :trunc_size]
    LH = LH[:trunc_size, :trunc_size]
    HL = HL[:trunc_size, :trunc_size]
    HH = HH[:trunc_size, :trunc_size]
    
    # Saving coefficients needed for reconstruction
    coeff_arr.append((LL,(LH, HL, HH)))
    
    # Creating the 2x2 grid to better the display the results of DWT
    horstack_1 = np.hstack((LL, LH))
    horstack_2 = np.hstack((HL, HH))
    four_grid = np.vstack((horstack_1, horstack_2))
    
    return four_grid


def reconstruct(coeff_arr, im):
    global another
    # get saved coeeficients from array
    (LL,(LH, HL, HH)) = coeff_arr[another]
    
    # Applying the reconstruction using the coefficients
    rec_im = pywt.idwt2((im,(LH,HL,HH)), 'bior2.2', mode='periodization')
    
    # Recusrively apply reconstruction    
    if another < 3:
        another = another + 1
        return reconstruct(coeff_arr, rec_im)
    else:
        return rec_im

# load image
x = Image.open("harbour512x512.tiff")
# remove extension from file name
x.filename = x.filename[:-12]

# apply DWT for 4 different depths
four_grid = filter(x,0)
plt.imsave(f'DWT1_{x.filename}.png', four_grid, cmap=plt.cm.gray)
count = 0

four_grid = filter(x,1)
plt.imsave(f'DWT2_{x.filename}.png', four_grid, cmap=plt.cm.gray)
count = 0

four_grid = filter(x,2)
plt.imsave(f'DWT3_{x.filename}.png', four_grid, cmap=plt.cm.gray)
count = 0

four_grid = filter(x,3)
plt.imsave(f'DWT4_{x.filename}.png', four_grid, cmap=plt.cm.gray)

# trim created coeeficient array
coeff_arr = coeff_arr[6:10]
(LL,(LH, HL, HH)) = coeff_arr[0]

# reconstruct original image using save coefficients
recon = reconstruct(coeff_arr, LL)
plt.imsave(f'REC_{x.filename}.png', recon, cmap=plt.cm.gray)
# reset global variable
another = 0

#quantize coefficients

def quantize_coeff(arr,step):
    qt_coeff_arr = arr
    new = [0,0,0,0]
    for iter in range(len(qt_coeff_arr)):
        (LL,(LH, HL, HH)) = qt_coeff_arr[iter]
        # quantize each quadrant
        LLQ = quantizer(np.array(LL, dtype=np.float64), step)
        LHQ = quantizer(np.array(LH, dtype=np.float64), step)
        HLQ = quantizer(np.array(HL, dtype=np.float64), step)
        HHQ = quantizer(np.array(HH, dtype=np.float64), step)
        new[iter] = (LLQ,(LHQ, HLQ, HHQ))
    return new

PSNR_vec = []
entropy_vec = []
mse_savedim = []
mse_ceoffs = []

(LLO,(LHO,HLO,HHO)) = coeff_arr[0]
org_co = []
# array of 13 original subbands
org_co.append(LLO)

    
for co in coeff_arr:
    (_,(LHC,HLC,HHC)) = co
    org_co.append(LHC)
    org_co.append(HLC)
    org_co.append(HHC)

for z in range(10):
    another = 0  
    step = 2**z

    # quanztize coefficients with specified step
    qt_ca = quantize_coeff(coeff_arr,step)

    # reconstruct original image using save quantized coefficients
    (LLQ,(LHQ, HLQ, HHQ)) = qt_ca[0]
    qunatized_recon = reconstruct(qt_ca, LLQ)
    
    # extract coefficeints needed to caculate entropy (13 subbands)
    #  ______________
    # |      |       |
    # |  LL  |  LH   |
    # |______|_______|
    # |      |       |
    # |  HL  |  HH   |
    # |______|_______|
    # 
    
    co_en = []
    # first top left lowband (image)
    co_en.append(LLQ)
    # 12 other spare bands
    for c in qt_ca:
        (_,(LHC,HLC,HHC)) = c
        co_en.append(LHC)
        co_en.append(HLC)
        co_en.append(HHC)
    
            
    # Calculate histogram for each subband
    sen_v = []
    total_entropy = 0
    av_d = 0
    for subband in co_en:
        hist = np.zeros(5000)

        # Calculate histogram of subband
        for r in subband:
            for pix in r:
                pix = int(pix)
                hist[pix] += 1 /(len(subband) * len(subband[0]))
        
        # Calcualte entropy of each pixel and sum for total entropy of subband
        subband_entropy = 0
        for f in hist:
            if f != 0:
                subband_entropy += -1 * f * math.log(f, 2)
        
        # Calculate TOTAL entropy
        total_entropy += subband_entropy
        
        # Calculate MSE of subband
        total_d = MSEC(org_co, co_en)
        # Calcualte average mse
        av_d = total_d / 13
    
    # Add the average entropy of 13 subbands to a value vector for plotting
    entropy_vec.append(total_entropy/13)
        
    # Add average MSE to value vector for plotting
    mse_ceoffs.append(av_d)
    
    # calculate PSNR between base image and quantized images
    PSNR_vec.append(cv2.PSNR(qunatized_recon, np.array(x, dtype=np.float64)))

    # Save quanztized images
    plt.imsave(f'QT_{step}_{x.filename}.png', qunatized_recon, cmap=plt.cm.gray)
    plt.close()
    
    qim = Image.open(f'QT_{step}_{x.filename}.png')
    rec = Image.open(f'REC_{x.filename}.png')
    mse_savedim.append(MSE(qim,rec))
    
print(entropy_vec)
print(mse_ceoffs)
print(mse_savedim)

x_vector = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
plt.plot(entropy_vec, PSNR_vec)
plt.title(f'PSNR/rate of {x.filename} image')
plt.ylabel('PSNR')
plt.xlabel('bit-rate')
# Plot histrogram
plt.savefig(f'PSNR_{x.filename}.png')
# Save plotted histogram
plt.clf()
# clears figure

plt.plot(x_vector, mse_ceoffs, 'r')
plt.title("Distortion between coefficients")
plt.ylabel('MSE')
plt.xlabel('step-size')
# Plot histrogram
plt.savefig(f'MSE_COEF{x.filename}.png')
# Save plotted histogram
plt.clf()
# clears figure

plt.plot(x_vector, mse_savedim, 'g')
plt.title("Distortion between images")
plt.ylabel('MSE')
plt.xlabel('step-size')
# Plot histrogram
plt.savefig(f'MSE_{x.filename}.png')
# Save plotted histogram
plt.clf()
# clears figure
