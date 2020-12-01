# Two-Band Filter Bank

import numpy as np
import scipy.signal as scs
from coeff import *

def analysis_filter_bank(signal, wavelet_coeff):

    Lo_R = wavelet_coeff / np.linalg.norm(wavelet_coeff)
    Lo_D = np.flip(Lo_R)
    Hi_R = scs.qmf(Lo_R)
    Hi_D = np.flip(Hi_R)

    low = np.convolve(signal, Lo_D)
    high = np.convolve(signal, Hi_D)


    low_d = downscale(low)
    high_d = downscale(high)


    if len(low_d) == 1 or len(high_d) == 1:
        return low_d.tolist() + high_d.tolist()
    else:
        return analysis_filter_bank(low_d, wavelet_coeff) + analysis_filter_bank(high_d, wavelet_coeff)
            

def synthesis_filter_bank(signal, wavelet_coeff):
    if len(signal) > 1:
        for i in range(0, len(signal), 2):
            el = synthesis_filter_bank(signal[i:i + len(signal/2)], wavelet_coeff)
            
        Lo_R = wavelet_coeff / np.linalg.norm(wavelet_coeff)
        Lo_D = np.flip(Lo_R)
        Hi_R = scs.qmf(Lo_R)
        Hi_D = np.flip(Hi_R)

        low_u = upscale(el)

        low = np.convolve(signal, Lo_R)
        high = np.convolve(signal, Hi_R)
    

    
    return 0

def downscale(signal):
    if isinstance(signal, np.ndarray):
        signal = signal.tolist()
        
    # remove -1 element
    signal = signal[1:]
    rem = signal[::2]
    
    return np.array(rem)

def upscale(signal):
    if isinstance(signal, np.ndarray):
        signal = signal.tolist()
        
    i = 1
    while i <= len(signal):
        signal.insert(i, 0)
        i += 2
        
    return np.array(signal)

print(analysis_filter_bank([1,2,3,4], haar()))