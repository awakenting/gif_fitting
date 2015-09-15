# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:58:42 2015

@author: andrej
"""


import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.facecolor'] = 'white'

############################################################################################################
# OVERVIEW PLOTS
############################################################################################################
for dataset in np.arange(1,6):
    testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum' + str(dataset) + '/IVTest.mat')
    testTraces = testData['IVTest'][0][0]
    testVs = testTraces[1].squeeze()
    testIs = testTraces[0].squeeze()
    testVd = testTraces[3].squeeze()
    testId = testTraces[2].squeeze()
    dt = testTraces[4].squeeze()
    traceLen = testTraces[6].squeeze()
    traceT = traceLen*dt
    traceTime = np.arange(traceLen)*dt
    
    plt.figure(figsize=(20,12))
    plt.subplot(4,1,1)
    plt.plot(traceTime,testIs,'k')
    plt.title('Somatic input current')
    plt.xlabel('Time [ms]')
    plt.ylabel('Current [?]')
    
    plt.subplot(4,1,2)
    plt.plot(traceTime,testVs,'b')
    plt.title('Somatic membrane voltage')
    plt.xlabel('Time [ms]')
    plt.ylabel('Voltage [mV]')
    
    plt.subplot(4,1,3)
    plt.plot(traceTime,testId,'k')
    plt.title('Dendritic input current')
    plt.xlabel('Time [ms]')
    plt.ylabel('Current [?]')
    
    plt.subplot(4,1,4)
    plt.plot(traceTime,testVd,'b')
    plt.title('Dendritic membrane voltage')
    plt.xlabel('Time [ms]')
    plt.ylabel('Voltage [mV]')
    
    plt.tight_layout()
    plt.savefig('/home/andrej/Documents/Code/ActiveDendritesModeling/Figures/overview_larkum'+str(dataset)+'.png')