# Generates the blur functions
# USAGE: h = myblurgen(type, r)
# type:         
# 'outoffocus'  ideal out-of-focus blur
# 'gaussian'    Gaussian blur
import numpy
import math

def myblurgen (type, r):
    type = type.lower()
    if type == 'outoffocus':
        support = r*2+1
        X, Y = numpy.meshgrid(range(1,support + 1), range(1,support + 1))
        distance = numpy.sqrt((X-r-1)**2 + (Y-r-1)**2)
        h = numpy.less_equal(distance, r)
        h = h.astype(numpy.float)
        h = h/numpy.sum(h)
        return numpy.round(h, 4)
    elif type == 'gaussian':
        h = GaussKernel(r*2+1, r/2)
        h = h/numpy.sum(h)
        return numpy.round(h, 4)
    else:
        raise Exception('Unknown blur type.')
        

def GaussKernel(mask_size, sigma2):
    # GAUSSKERNEL 2-D Gaussian kernel 
    #
    # Usage: [f] = GaussKernel(mask_size)
    #
    # Input:  mask_size         size of mask
    #         sigma2            x and y variance
    #
    # Output: f                 mask
    #
    # v1.1 Rudolfs Arvids Kalnins rakal@kth.se 2020 B)
    
    f = numpy.zeros(mask_size)
    mx = (mask_size + 1)/2
    my = mx;
    
    out = []
    for x in range(1,mask_size + 1):
        new = []
        for y in range(1,mask_size + 1):
             z = numpy.exp( -((x-mx)**2 + (y-my)**2)/(2*sigma2))/(2*math.pi*sigma2)
             z = round(z, 4)
             new.append(z)
        
        out.append(new)    
    return out
