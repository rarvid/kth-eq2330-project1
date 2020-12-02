
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
import pywt
import scipy.signal as scs
from PIL import Image
from quantizer import quantizer

# Global variable for recursion purposes
# Count is used for to obtain a specific depth recursive DWT
# coeff_arr is an array used to save tuples with coefficients after each recursive
# application of the DWT
count = 0
another = 0
coeff_arr =[]

def len_debug(LL,LH,HL,HH):
    print("LL dims",len(LL), len(LL[0]))
    print("LH dims",len(LH), len(LH[0]))
    print("HL dims",len(HL), len(HL[0]))
    print("HH dims",len(HH), len(HH[0]))
    
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
x = Image.open("boats512x512.tiff")
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
        LLQ = quantizer(np.array(LL, dtype=np.float64), step)
        LHQ = quantizer(np.array(LH, dtype=np.float64), step)
        HLQ = quantizer(np.array(HL, dtype=np.float64), step)
        HHQ = quantizer(np.array(HH, dtype=np.float64), step)
        # len_debug(LL,LH,HL,HH)
        new[iter] = (LLQ,(LHQ, HLQ, HHQ))
    return new

for z in range(10):
    another = 0
    step = (256/2**z)
    # quanztize coefficients with specified step
    qt_ca = quantize_coeff(coeff_arr,step)
    # reconstruct original image using save quantized coefficients
    # len_debug(LLQ, LHQ, HLQ, HHQ)    
    (LLQ,(LHQ, HLQ, HHQ)) = qt_ca[0]
    qunatized_recon = reconstruct(qt_ca, LLQ)
    # Image.fromarray(qunatized_recon.astype(np.uint8), 'L').save(f'QT_{step}_{x.filename}.png')
    plt.imsave(f'QT_{step}_{x.filename}.png', qunatized_recon, cmap=plt.cm.gray)
    plt.close()
    

