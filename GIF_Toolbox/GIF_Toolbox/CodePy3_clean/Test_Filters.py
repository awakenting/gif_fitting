# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:35:29 2015

@author: andrej
"""
from TwoComp_passive import *
import Tools

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


# Create a new object GIF 
myGIF = TwoComp_passive(0.1)

# Define parameters
myGIF.Tref = 4.0
filterLength = 1000.0

myGIF.eta_A = Filter_Powerlaw()
myGIF.eta_A.setMetaParameters(length=filterLength, Tconst=5, power=-0.8, powerTime=2000)

myGIF.k_s = Filter_ThreeExpos()
myGIF.k_s.setMetaParameters(length=filterLength, tau_one=10, tau_two=100, tau_three=2000)

myGIF.e_ds = Filter_ThreeExpos()
myGIF.e_ds.setMetaParameters(length=filterLength, tau_one=10, tau_two=200, tau_three=1000)

# initialize coefficients for filters
powerlaw_coeffs = np.array([-5, -10])
ks_threeExpos_coeffs = np.array([-10,10,0])
eds_threeExpos_coeffs = np.array([1,1,1])

myGIF.eta_A.setFilter_Coefficients(powerlaw_coeffs)
myGIF.k_s.setFilter_Coefficients(ks_threeExpos_coeffs)
myGIF.e_ds.setFilter_Coefficients(eds_threeExpos_coeffs)

myGIF.plotParameters()

# Generate OU process with temporal correlation 3 ms and mean modulated by a sinusoildal function of 1 Hz
I_OU = Tools.generateOUprocess_sinMean(f=1.0, T=100.0, tau=3.0, mu=0.3, delta_mu=0.5, sigma=0.1, dt=0.1)

current_amplitude = 0.2
I = current_amplitude * np.ones(1000)
Id = np.zeros(1000)
spks = np.array([])
#spks[500] = 1

(time, eta_a, spks, V, filtered_I, filtered_I_d) = myGIF.simulate_deterministicSpikes(I_OU, Id, spks)

plt.figure()
plt.plot(time,V)
plt.title('Simulated voltage trace')

plt.figure()
plt.plot(time, I_OU)
plt.title('I_OU')

#plt.figure()
#plt.plot(time, eta_a)
#plt.title('generated eta_a')

#plt.figure()
#plt.plot(time, filtered_I)
#plt.title('I')

#plt.figure()
#plt.plot(time, filtered_I_d)
#plt.title('filtered I_d')