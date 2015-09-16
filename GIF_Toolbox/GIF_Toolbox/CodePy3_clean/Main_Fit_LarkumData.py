"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""

from Experiment import *
from AEC_Badel import *
from TwoComp_passive import *
from Filter_Rect_LogSpaced import *
from Filter_Powerlaw import *
from Filter_ThreeExpos import *

import Tools
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.facecolor'] = 'white'
############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################

testData = io.loadmat('/home/andrej/Documents/Code/Larkum1to5/Larkum4/IVTest.mat')
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
myExp = Experiment('Experiment 1',dt)

# Add training set data
myExp.addTrainingSetTrace_TwoComp(testVs, 10**-3, testIs , 10**-12, testVd, 10**-3, testId , 10**-12, traceT, FILETYPE='Array')

# Plot data
#myExp.plotTrainingSet()
#myExp.plotTestSet()

############################################################################################################
# STEP 2: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object GIF 
myGIF = TwoComp_passive(0.1)

# Define parameters
myGIF.Tref = 4.0
filterLength = 1000.0

myGIF.eta_A = Filter_Powerlaw()
myGIF.eta_A.setMetaParameters(length=filterLength, Tconst=5, power=-0.8, powerTime=2000)

myGIF.k_s = Filter_ThreeExpos()
myGIF.k_s.setMetaParameters(length=filterLength)

myGIF.e_ds = Filter_ThreeExpos()
myGIF.e_ds.setMetaParameters(length=filterLength)

# initialize coefficients for filters
initial_powerlaw_coeffs = np.array([0.2, 1])
initial_threeExpos_coeffs = np.array([1,1,1])

myGIF.eta_A.setFilter_Coefficients(initial_powerlaw_coeffs)
myGIF.k_s.setFilter_Coefficients(initial_threeExpos_coeffs)
myGIF.e_ds.setFilter_Coefficients(initial_threeExpos_coeffs)

# Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
#myExp.trainingset_traces[0].setROI([[0,100000.0]])

# detect Spikes
myExp.detectSpikes_cython()

# To visualize the training set and the ROI call again
#myExp.plotTrainingSet()

# Perform the fit
myGIF.fit(myExp)


# Plot the model parameters
myGIF.printParameters()
myGIF.plotParameters()

(time, eta_a, spks, filtered_currents) = myGIF.simulate(testIs, testId)

# Save the model
#myGIF.save('./myGIF.pck')

