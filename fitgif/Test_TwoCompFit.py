# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:53:08 2015

@author: andrej
"""

"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib as mpl

from .Experiment import Experiment
from .TwoComp_passive import TwoComp_passive
from .Filter_Powerlaw import Filter_Powerlaw
from .Filter_ThreeExpos import Filter_ThreeExpos

plt.style.use('ggplot')

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.facecolor'] = 'white'
############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
myExp = Experiment('Experiment 1',0.1)

datasetIndex = 4

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


# Add test set data
myExp.addTrainingSetTrace_TwoComp(trainVs, 10**-3, trainIs , 10**-12, trainVd, 10**-3, trainId , 10**-12, len(trainIs)*0.1, FILETYPE='Array')

testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum'+str(datasetIndex)+'/IVTest.mat')
testTraces = testData['IVTest'][0][0]



# recording timestep: 0.1 ms
# testTraces.dtype.names:
# Out[26]: ('Is', 'Vs', 'Id', 'Vd', 'timestep', 'spks', 'L', 'spktr')
testVs = testTraces[1].squeeze()
testIs = testTraces[0].squeeze()
testVd = testTraces[3].squeeze()
testId = testTraces[2].squeeze()
dt = testTraces[4].squeeze()
traceLen = testTraces[6].squeeze()
traceT = traceLen*dt

# add the same test data several times so that a Md* measure can be computed (but it doesn't really make sense)
for i in range(10):
    myExp.addTestSetTrace_TwoComp(testVs, 10**-3, testIs , 10**-12, testVd, 10**-3, testId , 10**-12, len(testIs)*0.1, FILETYPE='Array')


# Plot data
#myExp.plotTrainingSet()
#myExp.plotTestSet()

############################################################################################################
# STEP 2: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object of Two Compartment model 
myGIF = TwoComp_passive(0.1)

# Define parameters
myGIF.Tref = 4.0
filterLength = 1000.0

myGIF.eta_A = Filter_Powerlaw()
myGIF.eta_A.setMetaParameters(length=filterLength, Tconst=5, power=-0.8, powerTime=2000)

myGIF.k_s = Filter_ThreeExpos()
myGIF.k_s.setMetaParameters(length=filterLength, tau_one=1, tau_two=10, tau_three=100)

myGIF.e_ds = Filter_ThreeExpos()
myGIF.e_ds.setMetaParameters(length=filterLength, tau_one=1, tau_two=10, tau_three=100)

# set coefficients for filters for simulation
powerlaw_coeffs = np.array([-0.5, -10])
ks_threeExpos_coeffs = np.array([-5,10,1])
eds_threeExpos_coeffs = np.array([5,-10,1])

myGIF.eta_A.setFilter_Coefficients(powerlaw_coeffs)
myGIF.k_s.setFilter_Coefficients(ks_threeExpos_coeffs)
myGIF.e_ds.setFilter_Coefficients(eds_threeExpos_coeffs)

myGIF.plotParameters()

tr = myExp.trainingset_traces[0]
(time, eta_a, spks, filtered_currents) = myGIF.simulate(tr.I, tr.I_d)

# initialize coefficients for filters for fitting
initial_powerlaw_coeffs = np.array([0.2, 1])
initial_threeExpos_coeffs = np.array([1,1,1])

myGIF.eta_A.setFilter_Coefficients(initial_powerlaw_coeffs)
myGIF.k_s.setFilter_Coefficients(initial_threeExpos_coeffs)
myGIF.e_ds.setFilter_Coefficients(initial_threeExpos_coeffs)

# Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
#myExp.trainingset_traces[0].setROI([[0,100000.0]])

# set Spikes to simulated spike times
tr.spks = np.array(spks/myExp.dt, dtype='int')
tr.spks_flag   = True

# Perform the fit
myGIF.fit(myExp)


# Plot the model parameters
myGIF.plotParameters()

#tr = myExp.trainingset_traces[0]
#(time, eta_a, spks) = myGIF.simulate(tr.I, tr.I_d)


# Save the model
#myGIF.save('./myGIF.pck')

############################################################################################################
# STEP 4: EVALUATE THE GIF MODEL PERFORMANCE (USING MD*)
############################################################################################################

# Use the myGIF model to predict the spiking data of the test data set in myExp
#myPrediction = myExp.predictSpikes(myGIF, nb_rep=100)

# Compute Md* with a temporal precision of +/- 4ms
#Md = myPrediction.computeMD_Kistler(4.0, 0.1)    

# Plot data vs model prediction
#myPrediction.plotRaster(delta=1000.0) 