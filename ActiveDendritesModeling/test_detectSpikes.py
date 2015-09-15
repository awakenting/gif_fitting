# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:35:19 2015

@author: andrej
"""

#from Experiment_py3 import *
from ./GIF_Toolbox/GIF_Toolbox/Code/Experiment_py3 import *
#from GIF import *
from Filter_Rect_LogSpaced_py3 import *

import Tools_py3
import matplotlib.pyplot as plt

PATH = '../Data/'

myExp = Experiment_py3('Experiment 1', 0.1)
myExp.addTrainingSetTrace(PATH + 'Cell3_Ger1Training_ch2_1008.ibw', 1.0, PATH + 'Cell3_Ger1Training_ch3_1008.ibw', 1.0, 120000.0, FILETYPE='Igor')
test_trace = myExp.trainingset_traces[0]
test_trace.detectSpikes()
test_trace.spks[0:10]
plt.plot(test_trace.V[0:1000])