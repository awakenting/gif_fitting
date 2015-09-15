# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:35:19 2015

@author: andrej
"""

#from Experiment_py3 import *
from Experiment_py3 import *
#from GIF import *
from Filter_Rect_LogSpaced_py3 import *

import Tools_py3
import matplotlib.pyplot as plt

PATH = '../Data/'

myExp = Experiment_py3('Experiment 1', 0.1)
myExp.addTrainingSetTrace(PATH + 'Cell3_Ger1Training_ch2_1008.ibw', 1.0, PATH + 'Cell3_Ger1Training_ch3_1008.ibw', 1.0, 120000.0, FILETYPE='Igor')
test_trace = myExp.trainingset_traces[0]
test_trace.detectSpikes()
print('Indices of first 10 spikes : ' + str(test_trace.spks[0:10]))
plt.plot(test_trace.V[0:1000])

# test if detectSpikes and detectSpikes_cython have the same outcome
test_trace.detectSpikes()
spks1 = test_trace.spks
test_trace.detectSpikes_cython()
spks2 = test_trace.spks
print('Number of different detected spikes: ' + str(np.sum(spks1 == spks2)-spks1.size))

# to test the speed up with cython for detectSpikes() you can use the following commands in ipython
'''
%timeit test_trace.detectSpikes()
%timeit test_trace.detectSpikes_cython()
'''



