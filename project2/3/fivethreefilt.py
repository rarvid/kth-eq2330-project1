
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
import pywt
import scipy.signal as scs
from PIL import Image
'''
Global variable for recursion purposes
Count is used for to obtain a specific depth recursive DWT
coeff_arr is an array used to save tuples with coefficients after each recursive
application of the DWT
'''
count = 0

coeff_arr =[]

def filter(pic):
    
    '''
    Application of the dwt using 5/3 filter and a periodic copy
    to address border effects
    '''
    coeffs = pywt.dwt2(pic, 'bior2.2', mode='periodization')
    # Indivudal coefficient matrices of Low-Low, High-Low, Low-High, High-High frequencies
    LL, (LH, HL, HH) = coeffs

    '''
    Specify recursion depth and
    apply the 5/3 filter on the Low-Low frequencies
    '''
    global count
    if count < 3:
        count = count + 1
        LL  = filter(LL)    
    
    '''
    Resizing of resulting arrays to get optimal size (512,256,128,64,32)
    '''
    trunc_size = len(LL) - (len(LL) % 8)
    LL = LL[:trunc_size, :trunc_size]
    LH = LH[:trunc_size, :trunc_size]
    HL = HL[:trunc_size, :trunc_size]
    HH = HH[:trunc_size, :trunc_size]
    
    '''
    Saving coefficients needed for reconstruction
    '''
    coeff_arr.append((LL,(LH, HL, HH)))
    
    '''
    Creating the 2x2 grid to better the display the results of DWT
    '''
    horstack_1 = np.hstack((LL, LH))
    horstack_2 = np.hstack((HL, HH))
    four_grid = np.vstack((horstack_1, horstack_2))
    
    return four_grid


def reconstruct(coeffs):
    '''
    Extracting the specific frequiency arrays from tuple
    '''
    (LL, (LH, HL, HH)) = coeffs
    
    '''
    Applying the reconstruction using the coefficients
    '''
    rec_im = pywt.idwt2(coeffs, 'bior2.2', mode='periodization')
    plt.imsave('reconstructed.png', rec_im, cmap=plt.cm.gray)

    return rec_im
 
x = Image.open("harbour512x512.tiff")
four_grid = filter(x)
plt.imsave('dwt4.png', four_grid, cmap=plt.cm.gray)


'''
Manual "recursive" application of reconstruction by inserting the reconstructed
image as the Low-Low frequency matrix in each step
'''
first = reconstruct(coeff_arr[0])
(LL1,(LH1, HL1, HH1)) = coeff_arr[1]
second = reconstruct((first,(LH1, HL1, HH1)))
(LL2,(LH2, HL2, HH2)) = coeff_arr[2]
third = reconstruct((second,(LH2, HL2, HH2)))
(LL3,(LH3, HL3, HH3)) = coeff_arr[3]
fourth = reconstruct((third,(LH3, HL3, HH3)))

