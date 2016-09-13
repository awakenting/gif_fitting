"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib as mpl

from .Experiment import Experiment
from .GIF import GIF
from .GIF_pow import GIF_pow
from .Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from .Filter_Powerlaw import Filter_Powerlaw


plt.style.use('ggplot')

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
myExp.addTrainingSetTrace(testVs, 10**-3, testIs , 10**-12, traceT, FILETYPE='Array')

# Plot data
#myExp.plotTrainingSet()
#myExp.plotTestSet()

############################################################################################################
# STEP 2: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object GIF 
myGIF_rect = GIF(0.1)

# Define parameters
myGIF_rect.Tref = 4.0  

myGIF_rect.eta = Filter_Rect_LogSpaced()
myGIF_rect.eta.setMetaParameters(length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

myGIF_rect.gamma = Filter_Rect_LogSpaced()
myGIF_rect.gamma.setMetaParameters(length=1000.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

# Create a new object GIF 
myGIF_pow = GIF_pow(0.1)

# Define parameters
myGIF_pow.Tref = 4.0 

myGIF_pow.eta = Filter_Powerlaw()
myGIF_pow.eta.setMetaParameters(length=1000.0, Tconst=5, power=-0.8, powerTime=2000)

myGIF_pow.gamma = Filter_Powerlaw()
myGIF_pow.gamma.setMetaParameters(length=1000.0, Tconst=5, power=-0.8, powerTime=2000)

powerlaw_coeffs = np.array([0.2,1])
myGIF_pow.eta.setFilter_Coefficients(powerlaw_coeffs)
myGIF_pow.gamma.setFilter_Coefficients(powerlaw_coeffs)

# Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
myExp.trainingset_traces[0].setROI([[0,100000.0]])

# detect Spikes
myExp.detectSpikes_cython()

# To visualize the training set and the ROI call again
#myExp.plotTrainingSet()

# Perform the fit
myGIF_rect.fit(myExp, DT_beforeSpike=5.0)
myGIF_pow.fit(myExp, DT_beforeSpike=5.0)


# Plot the model parameters
myGIF_rect.printParameters()
myGIF_pow.printParameters()
myGIF_rect.plotParameters()   
myGIF_pow.plotParameters()   

(time, V, eta_sum, V_T, spks) = myGIF_pow.simulate(testIs, testVs[0])

(time, V, eta_sum, V_T, spks) = myGIF_rect.simulate(testIs, testVs[0])
# Save the model
#myGIF.save('./myGIF.pck')

###############################################################################
# Test hyperparameter of powerlaw filter:
###############################################################################
'''
for pw in [-0.6,-0.7,-0.8,-0.9]:
    myGIF_pow.eta = Filter_Powerlaw()
    myGIF_pow.eta.setMetaParameters(length=1000.0, Tconst=5, power=pw, powerTime=2000)
    
    myGIF_pow.gamma = Filter_Powerlaw()
    myGIF_pow.gamma.setMetaParameters(length=1000.0, Tconst=5, power=pw, powerTime=2000)
    
    powerlaw_coeffs = np.array([0.2,1])
    myGIF_pow.eta.setFilter_Coefficients(powerlaw_coeffs)
    myGIF_pow.gamma.setFilter_Coefficients(powerlaw_coeffs)
    
    # Perform the fit
    myGIF_rect.fit(myExp, DT_beforeSpike=5.0)
    
    print ("\n################################")
    print ("# Fit GIF_pow")
    print ("################################\n")
    
    myGIF_pow.fitVoltageReset(myExp, myGIF_pow.Tref, do_plot=False)
    
    myGIF_pow.fitSubthresholdDynamics(myExp, DT_beforeSpike=5.0)
    
    myGIF_rect.plotParameters()   
    myGIF_pow.plotParameters() 
'''

