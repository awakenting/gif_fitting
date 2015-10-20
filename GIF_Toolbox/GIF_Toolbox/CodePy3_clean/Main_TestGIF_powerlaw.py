"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""

from Experiment import *
from AEC_Badel import *
from GIF import *
from TwoCompGIF import *
from Filter_Rect_LogSpaced import *
from Filter_Powerlaw import *

import Tools
import matplotlib.pyplot as plt

PATH = '../Data/'


############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################

myExp = Experiment('Experiment 1', 0.1)

# Load AEC data
myExp.setAECTrace(PATH + 'Cell3_Ger1Elec_ch2_1007.ibw', 1.0, PATH + 'Cell3_Ger1Elec_ch3_1007.ibw', 1.0, 10000.0, FILETYPE='Igor')

# Load training set data
myExp.addTrainingSetTrace(PATH + 'Cell3_Ger1Training_ch2_1008.ibw', 1.0, PATH + 'Cell3_Ger1Training_ch3_1008.ibw', 1.0, 120000.0, FILETYPE='Igor')

# Load test set data
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1009.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1009.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1010.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1010.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1011.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1011.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1012.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1012.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1013.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1013.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1014.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1014.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1015.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1015.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1016.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1016.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1017.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1017.ibw', 1.0, 20000.0, FILETYPE='Igor')

# Plot data
#myExp.plotTrainingSet()
#myExp.plotTestSet()


############################################################################################################
# STEP 2: ACTIVE ELECTRODE COMPENSATION
############################################################################################################
'''
# Create new object to perform AEC
myAEC = AEC_Badel(myExp.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=myExp.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]  
myAEC.p_nbRep = 15     

# Assign myAEC to myExp and compensate the voltage recordings
myExp.setAEC(myAEC)  
myExp.performAEC()  

# Plot AEC filters (Kopt and Ke)
myAEC.plotKopt()
myAEC.plotKe()

# Plot training and test set
#myExp.plotTrainingSet()
#myExp.plotTestSet()
'''

############################################################################################################
# STEP 3: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object GIF 
myGIF_rect = GIF(0.1)

# Define parameters
myGIF_rect.Tref = 4.0  

myGIF_rect.eta = Filter_Rect_LogSpaced()
myGIF_rect.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

myGIF_rect.gamma = Filter_Rect_LogSpaced()
myGIF_rect.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

# Create a new object GIF 
#myGIF_pow = TwoCompGIF(0.1)
#
## Define parameters
#myGIF_pow.Tref = 4.0 
#
#myGIF_pow.eta = Filter_Powerlaw()
#myGIF_pow.eta.setMetaParameters(length=1000.0, Tconst=5, power=-0.8, powerTime=2000)
#
#myGIF_pow.gamma = Filter_Powerlaw()
#myGIF_pow.gamma.setMetaParameters(length=1000.0, Tconst=5, power=-0.8, powerTime=2000)
#
#powerlaw_coeffs = np.array([0.2,1])
#myGIF_pow.eta.setFilter_Coefficients(powerlaw_coeffs)
#myGIF_pow.gamma.setFilter_Coefficients(powerlaw_coeffs)

# Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
myExp.trainingset_traces[0].setROI([[0,100000.0]])

# detect Spikes
myExp.detectSpikes_cython()

# To visualize the training set and the ROI call again
#myExp.plotTrainingSet()

# Perform the fit
myGIF_rect.fit(myExp, DT_beforeSpike=5.0)
#myGIF_pow.fit(myExp, DT_beforeSpike=5.0)
#myGIF_pow.fitVoltageReset(myExp, myGIF_pow.Tref, do_plot=False)

#myGIF_pow.fitSubthresholdDynamics(myExp, DT_beforeSpike=5.0)
# to test the speed up with cython in ipython:
'''
tr = myExp.trainingset_traces[0]
%timeit (time, V_est, eta_sum_est) = myGIF.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
%timeit (time, V_est, eta_sum_est) = myGIF.simulateDeterministic_forceSpikes_cython(tr.I, tr.V[0], tr.getSpikeTimes())
'''
# Plot the model parameters
#myGIF.printParameters()
myGIF_rect.plotParameters()   
#myGIF_pow.plotParameters()   
tr = myExp.trainingset_traces[0]
(time, V, eta_sum, V_T, spks) = myGIF_rect.simulate(tr.I, tr.V[0])

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
    print ("# Fit TwoCompGIF")
    print ("################################\n")
    
    myGIF_pow.fitVoltageReset(myExp, myGIF_pow.Tref, do_plot=False)
    
    myGIF_pow.fitSubthresholdDynamics(myExp, DT_beforeSpike=5.0)
    
    myGIF_rect.plotParameters()   
    myGIF_pow.plotParameters() 
'''

############################################################################################################
# STEP 3A (OPTIONAL): PLAY A BIT WITH THE FITTED MODEL
############################################################################################################

# Reload the model
#myGIF = GIF.load('./myGIF.pck')

# Generate OU process with temporal correlation 3 ms and mean modulated by a sinusoildal function of 1 Hz
#I_OU = Tools.generateOUprocess_sinMean(f=1.0, T=5000.0, tau=3.0, mu=0.3, delta_mu=0.5, sigma=0.1, dt=0.1)

# Simulate the model with the I_OU current. Use the reversal potential El as initial condition (i.e., V(t=0)=El)
#(time, V, I_a, V_t, S) = myGIF.simulate(I_OU, myGIF.El)
#(time, V, I_a, V_t, S) = myGIF.simulate_cython(I_OU, myGIF.El)

# To test the speed up with cython in ipython
#%timeit (time, V, I_a, V_t, S) = myGIF.simulate(I_OU, myGIF.El)
#%timeit (time, V, I_a, V_t, S) = myGIF.simulate_cython(I_OU, myGIF.El)

'''
# Plot the results of the simulation
plt.figure(figsize=(14,5), facecolor='white')
plt.subplot(2,1,1)
plt.plot(time, I_OU, 'gray')
plt.ylabel('I (nA)')
plt.subplot(2,1,2)
plt.plot(time, V,'black', label='V')
plt.plot(time, V_t,'red', label='V threshold')
plt.ylabel('V (mV)')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()
'''

############################################################################################################
# STEP 4: EVALUATE THE GIF MODEL PERFORMANCE (USING MD*)
############################################################################################################

# Use the myGIF model to predict the spiking data of the test data set in myExp
myPrediction = myExp.predictSpikes(myGIF_rect, nb_rep=500)

# Compute Md* with a temporal precision of +/- 4ms
Md = myPrediction.computeMD_Kistler(4.0, 0.1)    

# Plot data vs model prediction
myPrediction.plotRaster(delta=1000.0) 



