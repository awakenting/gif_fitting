# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:35:29 2015

@author: andrej
"""
from TwoComp_passive import *
from Experiment import *
from AEC_Badel import *
from TwoComp_passive import *
from Filter_Rect_LogSpaced import *
from Filter_Powerlaw import *
from Filter_ThreeExpos import *
import Tools
import scipy.io as io
from scipy.signal import fftconvolve

import numpy as np
import matplotlib.pylab as plt

plt.style.use('ggplot')

import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.facecolor'] = 'white'


datasetIndex = 5

trainData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum'+str(datasetIndex)+'/IVTrain.mat')
trainTraces = trainData['IVTrain'][0][0]

# recording timestep: 0.1 ms
# trainTraces.dtype.names:
# Out[26]: ('Is', 'Vs', 'Id', 'Vd', 'timestep', 'spks', 'L', 'spktr')
trainVs = trainTraces[1].squeeze()
trainIs = trainTraces[0].squeeze()
trainVd = trainTraces[3].squeeze()
trainId = trainTraces[2].squeeze()
dt = trainTraces[4].squeeze()
traceLen = trainTraces[6].squeeze()
traceT = traceLen*dt
myExp = Experiment('Experiment 1',0.1)

# Add training set data
myExp.addTrainingSetTrace_TwoComp(trainVs, 10**-3, trainIs , 10**-12, trainVd, 10**-3, trainId , 10**-12, len(trainIs)*0.1, FILETYPE='Array')
myExp.detectSpikes_cython()

# Create a new object GIF 
myGIF = TwoComp_passive(0.1)

# Define parameters
myGIF.Tref = 4.0
myGIF.E0 = np.array([-8.91086176926])
filterLength = 1000.0

myGIF.eta_A = Filter_Powerlaw()
myGIF.eta_A.setMetaParameters(length=filterLength, Tconst=5, power=-0.8, powerTime=2000)

myGIF.k_s = Filter_ThreeExpos()
myGIF.k_s.setMetaParameters(length=filterLength, tau_one=1, tau_two=10, tau_three=100)

myGIF.e_ds = Filter_ThreeExpos()
myGIF.e_ds.setMetaParameters(length=filterLength, tau_one=1, tau_two=10, tau_three=100)

# initialize coefficients for filters
powerlaw_coeffs = np.array([-3.788, -56.597])
ks_threeExpos_coeffs = np.array([2.934, -0.312, 0.057])
eds_threeExpos_coeffs = np.array([2.538, -0.023, 0.027])

myGIF.eta_A.setFilter_Coefficients(powerlaw_coeffs)
myGIF.k_s.setFilter_Coefficients(ks_threeExpos_coeffs)
myGIF.e_ds.setFilter_Coefficients(eds_threeExpos_coeffs)




#%% play around with simulations of deterministic spikes
'''

# Generate OU process with temporal correlation 3 ms and mean modulated by a sinusoildal function of 1 Hz
I_OU = Tools.generateOUprocess_sinMean(f=1.0, T=100.0, tau=3.0, mu=0.3, delta_mu=0.5, sigma=0.1, dt=0.1)

current_amplitude = 0.2
trial_length = 10000
I = current_amplitude * np.zeros(trial_length)
I[0:200] = 0
I[250:500] = 0
I[550:-1] = 0
Id = np.zeros(trial_length)
spks = np.array([50])
#spks[500] = 1

(time, eta_a, spks, V, filtered_I, filtered_I_d) = myGIF.simulate_deterministicSpikes(I, Id, spks)

myGIF.plotParameters()
#myGIF.plotParametersWithBasisfunctions()

plt.figure()
plt.subplot(2,1,1)
plt.plot(time,V, label='voltage')
plt.title('Simulated voltage trace')
plt.subplot(2,1,2)
plt.plot(time, I, label='input current')
plt.title('I_OU')
'''

#%% play around with simulations

t_period = 200
trial_length = 300000

somatic_amp = 0.1
dendritic_amp = 0.05

I = somatic_amp * np.sin(np.arange(0,trial_length/t_period, step=1/t_period))
Id = dendritic_amp * np.sin(np.arange(0,trial_length/t_period, step=1/t_period) + np.pi*0.5 )

tr = myExp.trainingset_traces[0]
I = tr.I[0:trial_length]
Id = tr.I_d[0:trial_length]

(time, eta_A_sum, spks, filtered_currents, p_dontspike) = myGIF.simulate(I, Id)
vsim = filtered_currents + eta_A_sum

myGIF.plotParameters()

nsubplots = 4
plt.figure()
plt.subplot(nsubplots,1,1)
plt.plot(time, vsim, label='voltage', hold = True)
vmin = np.min(vsim)
vmax = np.max(vsim)
plt.vlines(spks, vmin, vmax)
plt.title('Simulated voltage trace')

plt.subplot(nsubplots,1,2)
plt.plot(time, tr.V[0:trial_length])
plt.title('Real voltage trace')

plt.subplot(nsubplots,1,3)
plt.plot(time, p_dontspike)
plt.title('Prob. to not spike')

plt.subplot(nsubplots,1,4)
plt.plot(time, I, label='somatic input', hold=True)
plt.plot(time, Id, label='dendritic input')
plt.legend()
plt.title('Input currents')


#%% test convolution

'''
#I = amp * np.sin(np.arange(0,trial_length/t_period,step=1/t_period))
#I = np.ones(trial_length)
I = Tools.generateOUprocess_sinMean(f=1.0, T=1000.0, tau=3.0, mu=0.3, delta_mu=0.5, sigma=0.5, dt=0.1)
(ks_support, ks) = myGIF.k_s.getInterpolatedFilter(0.1)

plt.figure()
plt.plot(ks)

plt.figure()
plt.plot(I, hold = True)

#plt.plot(iconv)
'''
#%%
#plt.figure()
#plt.plot(time, eta_a)
#plt.title('generated eta_a')

#plt.figure()
#plt.plot(time, filtered_I)
#plt.title('I')

#plt.figure()
#plt.plot(time, filtered_I_d)
#plt.title('filtered I_d')