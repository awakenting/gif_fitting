# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:26:24 2015

@author: andrej
"""

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
plt.show()

testing = io.loadmat('../Larkum1to5/Larkum1/IVTest.mat')

ivs = testing['IVTest']
ivs = ivs[0][0]
# look at variables in ivs
#ivs.dtype.names

# Plot I_s :
plt.figure()
plt.subplot(4,1,1)
plt.plot(ivs[0])
plt.title('Input to soma from "Larkum 1"')
plt.subplot(4,1,2)
plt.plot(ivs[1])
plt.title('Membrane voltage soma from "Larkum 1"')
plt.subplot(4,1,3)
plt.plot(ivs[2])
plt.title('Input to dendrites from "Larkum 1"')
plt.subplot(4,1,4)
plt.plot(ivs[3])
plt.title('Dendritic membrane voltage from "Larkum 1"')