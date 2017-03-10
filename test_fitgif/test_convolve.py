# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:30:42 2015

@author: andrej
"""
import matplotlib.pyplot as plt
import numpy as np

from fitgif.Filter_ThreeExpos import Filter_ThreeExpos
from fitgif.Filter_Powerlaw import Filter_Powerlaw

inp = np.random.randn(100)*10-3
myfilter = Filter_ThreeExpos(10)
conv = myfilter.convolution_ContinuousSignal_basisfunctions(inp,1)

myfilter_pow = Filter_Powerlaw(10)
conv_pow = myfilter_pow.convolution_ContinuousSignal_basisfunctions(inp,1)


plt.figure()
plt.subplot(311)
plt.plot(inp)
plt.subplot(312)
plt.plot(conv)
plt.subplot(313)
plt.plot(conv_pow)