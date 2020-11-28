#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:16:40 2020

@author: jeremy
"""

import numpy as np





#%%

A = np.zeros((8,8))
for i in range(8):
    for k in range(8):
        a = i == 0 and np.sqrt(1/8) or np.sqrt(2/8)
        A[i, k] = a * np.cos((2*k+1) * i * np.pi / 16)

def dct8(x):
    return np.dot(A, np.dot(x, A.T))

def idct8(y):
    return np.dot(A.T, np.dot(y, A))

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

#%%


