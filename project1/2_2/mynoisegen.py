# Generates a noise m-by-n image
# USAGE: noise = mynoisegen(type, m, n, param1, param2)
# 
# type:         
# 'uniform'     uniform noise between param1 and param2
#               default param1 = -1, param2 = 1
# 'gaussian'    gaussian noise of mean param1 and variance param2
#               default param1 = 0, param2 = 1
# 'saltpepper'  salt & pepper noise)
#               p(0) = param1, p(1) = param2
import numpy
import math

def mynoisegen(type, m, n, param1 = None, param2 = None):
    type = type.lower()
    if type == 'uniform' :
        if param1 == None and param2 == None:
            param1 = -1
            param2 = 1
        noise = param1 + (param2 - param1) * numpy.random.rand(m,n)
        return numpy.round(noise, 4)
    elif type == 'gaussian':
        if param1 == None and param2 == None:
            param1 = 0;
            param2 = 1;
        noise = param1 + numpy.sqrt(param2) * numpy.random.randn(m,n)
        return numpy.round(noise, 4)
    elif type == 'saltpepper':
        if param1 == None and param2 == None:
            param1 = 0.5
            param2 = 1
        noise = numpy.ones((m,n)) * 0.5
        
        nn = numpy.random.rand(m,n)
        
        noise[nn <= param1] = 0
        noise[(nn > param1) & (nn <=param1 + param2)] = 1
        
        return noise
    else:
        raise Exception('Unknown noise type')