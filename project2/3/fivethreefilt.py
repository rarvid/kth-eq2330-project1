
import numpy as np        # math stuff
import cv2          # image processing stuff
from matplotlib import pyplot as plt   # plotting
import pywt
import scipy.signal as scs
from PIL import Image

count = 0

def filter(pic, depth):
    coeffs = pywt.dwt2(pic, 'bior2.2', mode='periodic')
    LL, (LH, HL, HH) = coeffs

    global count
    if count < 4:
        count = count + 1
        LL = filter(LL)
    
    
    
    trunc_size = len(LL) - (len(LL) % 8)
    # LL.resize((trunc_size,trunc_size), refcheck=False)
    # HL.resize((trunc_size,trunc_size), refcheck=False)
    # LH.resize((trunc_size,trunc_size), refcheck=False)
    # HH.resize((trunc_size,trunc_size), refcheck=False)
    LL = LL[:trunc_size, :trunc_size]
    HL = HL[:trunc_size, :trunc_size]
    LH = LH[:trunc_size, :trunc_size]
    HH = HH[:trunc_size, :trunc_size]
    print(f"LL: {len(LL)}  {len(LL[0])}")
    print(f"HL: {len(HL)}  {len(HL[0])}")
    print(f"LH: {len(LH)}  {len(LH[0])}")
    print(f"HH: {len(HH)}  {len(HH[0])}")
    
    horstack_1 = np.hstack((LL, LH))
    horstack_2 = np.hstack((HL, HH))
    four_grid = np.vstack((horstack_1, horstack_2))
    
    return four_grid
 
x = Image.open("harbour512x512.tiff")

four_grid = filter(x)
plt.imshow(four_grid, cmap=plt.cm.gray)
plt.show()    

