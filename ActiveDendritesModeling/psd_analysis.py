# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:59:17 2015

@author: andrej
"""
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from Experiment import *

############################################################################################################
# Power spectrum analysis
############################################################################################################
from scipy.signal import welch

def plotPSD (voltage,current,timeWindow,fs=10000,nperseg=4096,maxFrqToPlot=50):
    '''
    Plots the voltage and the powerspectrum of the given voltage in the given 
    time window by using the welch method from scipy.signal and passes nperseg to it.
    The time window will be shaded in the voltage plot.
    Powerspectrum is plotted up to maxFrqToPlot.
    
    timeWindow = 2 element list with beginning and end of the time window
                 (several timeWindows can be passed as a nested list)
    :type = list of 2-element lists, or a single 2-element list
    fs         = sampling frequency
    '''
    plt.figure(figsize=(20,12))
    # check if timeWindow is nested list:
    if any(isinstance(i, list) for i in timeWindow):
        ntwin = len(timeWindow)
        for i,twin in enumerate(timeWindow):
            f, psd = welch(voltage[twin[0]:twin[1]],fs=fs,nperseg=nperseg,scaling='spectrum')
            # plot input current
            plt.subplot(3*ntwin,1,i*3+1)
            plt.plot(current,'k')
            plt.axvspan(twin[0],twin[1], facecolor='k', alpha=0.2)
            plt.title('Input current')
            plt.xlabel('Time [timesteps of 0.1 ms]')
            plt.ylabel('Current [?]')
            
            plt.subplot(3*ntwin,1,i*3+2)
            plt.plot(voltage)
            plt.axvspan(twin[0],twin[1], facecolor='k', alpha=0.2)
            plt.title('Membrane voltage')
            plt.xlabel('Time [timesteps of 0.1 ms]')
            plt.ylabel('Voltage [mV]')
            
            plt.subplot(3*ntwin,1,i*3+3)
            plt.plot(f[f < maxFrqToPlot],psd[f < maxFrqToPlot])
            plt.axvline(5,color='g',label='5 Hz')
            plt.axvline(20,color='r',label='20 Hz')
            plt.title('Power spectrum corresponding to the shaded region')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [mV$^2$]')
            plt.legend()
    else:
        f, psd = welch(voltage[timeWindow[0]:timeWindow[1]],fs=fs,nperseg=nperseg,scaling='spectrum')
        # plot input current
        plt.subplot(311)
        plt.plot(current,'k')
        plt.axvspan(timeWindow[0],timeWindow[1], facecolor='k', alpha=0.2)
        plt.title('Input current')
        plt.xlabel('Time [timesteps of 0.1 ms]')
        plt.ylabel('Current [?]')
        # plot voltage
        plt.subplot(312)
        plt.plot(voltage)
        plt.axvspan(timeWindow[0],timeWindow[1], facecolor='k', alpha=0.2)
        plt.title('Membrane voltage')
        plt.xlabel('Time [timesteps of 0.1 ms]')
        plt.ylabel('Voltage [mV]')
        # plot power spectrum
        plt.subplot(313)
        plt.plot(f[f < maxFrqToPlot],psd[f < maxFrqToPlot])
        plt.title('Power spectrum corresponding to the shaded region')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [mV$^2$]')
        plt.axvline(5,color='g',label='5 Hz')
        plt.axvline(20,color='r',label='20 Hz')
        plt.legend()
        

# visually found time windows for larkum1 data
testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum1/IVTest.mat')
testTraces = testData['IVTest'][0][0]
testVd = testTraces[3].squeeze()
testId = testTraces[2].squeeze()
testVs = testTraces[1].squeeze()
testIs = testTraces[0].squeeze()
dt = testTraces[4].squeeze()
traceLen = testTraces[6].squeeze()
traceT = traceLen*dt
myExp = Experiment('Experiment 1',dt)

# Add training set data
myExp.addTrainingSetTrace(testVs, 10**-3, testIs , 10**-12, traceT, FILETYPE='Array')
myExp.detectSpikes_cython()
spkTrain = myExp.trainingset_traces[0].getSpikeTrain()
f, psd = welch(spkTrain,fs=10000,nperseg=4096,scaling='spectrum')
plt.plot(f[f < 100],psd[f < 100])
plt.title('Power spectrum of somatic spike train')
plt.xlabel('Frequency [Hz]')
plt.axvline(5,color='g',label='5 Hz')
plt.axvline(20,color='r',label='20 Hz')
plt.legend()

'''
twins = [[5000,55000],[55000,105000]]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)

twins = [105000,155000]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)


# visually found time windows for larkum2 data
testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum2/IVTest.mat')
testTraces = testData['IVTest'][0][0]
testVd = testTraces[3].squeeze()
testId = testTraces[2].squeeze()

twins = [[5000,105000],[115000,215000]]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)

twins = [225000,325000]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)

# visually found time windows for larkum3 data
testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum3/IVTest.mat')
testTraces = testData['IVTest'][0][0]
testVd = testTraces[3].squeeze()
testId = testTraces[2].squeeze()

twins = [[0,55000],[55000,255000]]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)

twins = [[265000,315000],[315000,515000]]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)


# visually found time windows for larkum4 data
testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum4/IVTest.mat')
testTraces = testData['IVTest'][0][0]
testVd = testTraces[3].squeeze()
testId = testTraces[2].squeeze()
twins = [0,len(testVd)]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)


# for larkum5
testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum5/IVTest.mat')
testTraces = testData['IVTest'][0][0]
testVd = testTraces[3].squeeze()
testId = testTraces[2].squeeze()
twins = [[0,60000],[60000,len(testVd)]]
twins = [0,len(testVd)]
plotPSD(testVd,testId,twins,maxFrqToPlot=100)
'''
