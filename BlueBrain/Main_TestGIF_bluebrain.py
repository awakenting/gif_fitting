"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""
import os
from Experiment_auto_read_T import *
from AEC_Badel import *
from GIF import *
from GIF_pow import *
from Filter_Rect_LogSpaced import *

import Tools
import matplotlib.pyplot as plt

PATH = '../Data/'


####################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
####################################################################################################
#%%  get filenames
animal_number = 5

root_path = '/home/andrej/Documents/Code/BlueBrain/article_4_data/grouped_ephys/'
animal_dirs = sorted(os.listdir(root_path))
PATH = root_path + animal_dirs[animal_number] + '/'
animal_files = sorted(os.listdir(PATH))
unwanted_sessions = ['APThreshold', 'APWaveform']
animals_wanted = []
for filename in animal_files:
    # files end with 'recordingType_recordingNumber.ibw'
    file_split = str.split(filename[0:-4],'_')
    if not file_split[-2] in unwanted_sessions:
        animals_wanted.append(filename)


myExp = Experiment_auto_read_T('Animal: ' + animal_dirs[animal_number])

num_files = int(len(animals_wanted)/2)
for file in animals_wanted[0:int(np.floor(num_files/2))]:
    # get matching voltage and current file
    files = Tools.get_matching_file_pair(animals_wanted, file)
    # Load training set data
    myExp.addTrainingSetTrace(PATH + files[1], 10**-3, PATH + files[0], 10**-12,
                              FILETYPE='Igor')
                                  

                                  
for file in animals_wanted[int(np.floor(num_files/2)):num_files]:
    # get matching voltage and current file
    files = Tools.get_matching_file_pair(animals_wanted, file)
    # Load training set data
    myExp.addTestSetTrace(PATH + files[1], 10**-3, PATH + files[0], 10**-12,
                                  FILETYPE='Igor')

#myExp.plotTrainingSet()

myExp.mergeTrainingTraces()
myExp.mergeTestTraces()
# Plot data
myExp.plotTrainingSet()
myExp.plotTestSet()


#%%
####################################################################################################
# STEP 3: FIT GIF MODEL TO DATA
####################################################################################################

# Create a new object GIF 
myGIF = GIF(myExp.dt)

# Define parameters
myGIF.Tref = 3.0  

myGIF.eta = Filter_Rect_LogSpaced()
myGIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

myGIF.gamma = Filter_Rect_LogSpaced()
myGIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)


# Create a new object GIF 
myGIF_pow = GIF_pow(myExp.dt)

# Define parameters
myGIF_pow.Tref = 3.0 

myGIF_pow.eta = Filter_Powerlaw()
myGIF_pow.eta.setMetaParameters(length=500.0, Tconst=5, power=-0.8, powerTime=2000)

myGIF_pow.gamma = Filter_Powerlaw()
myGIF_pow.gamma.setMetaParameters(length=500.0, Tconst=5, power=-0.8, powerTime=2000)

powerlaw_coeffs = np.array([0.2,1])
myGIF_pow.eta.setFilter_Coefficients(powerlaw_coeffs)
myGIF_pow.gamma.setFilter_Coefficients(powerlaw_coeffs)

# Create a new object GIF 
myGIF_exp = GIF_pow(myExp.dt)

# Define parameters
myGIF_exp.Tref = 3.0 

myGIF_exp.eta = Filter_ThreeExpos()
myGIF_exp.eta.setMetaParameters(length=500.0, tau_one=30, tau_two=100, tau_three=1000)

myGIF_exp.gamma = Filter_ThreeExpos()
myGIF_exp.gamma.setMetaParameters(length=500.0, tau_one=30, tau_two=100, tau_three=1000)

initial_powerlaw_coeffs = np.array([0.2,1])
initial_threeExpos_coeffs = np.array([1,1,1])

myGIF_exp.eta.setFilter_Coefficients(initial_threeExpos_coeffs)
myGIF_exp.gamma.setFilter_Coefficients(initial_threeExpos_coeffs)


# Define the ROI of the training set to be used for the fit (in this example we will use only 
# the first 100 s)
#myExp.trainingset_traces[0].setROI([[0,100000.0]])

# detect Spikes
myExp.detectSpikes_cython()

# To visualize the training set and the ROI call again
#myExp.plotTrainingSet()

# Perform the fit
#myGIF.fit(myExp, DT_beforeSpike=5.0)
myGIF.fit(myExp, DT_beforeSpike=5.0)
myGIF_pow.fit(myExp, DT_beforeSpike=5.0)
myGIF_exp.fit(myExp, DT_beforeSpike=5.0)

testtr = myExp.testset_traces[0]
traintr = myExp.trainingset_traces[0]



(spks_times, V, V_T) = myGIF.simulateVoltageResponse(testtr.I, myGIF.dt)
plt.figure()
plt.plot(V)
plt.ylim(-100,10)
plt.title('test input')
#
#(spks_times, V, V_T) = myGIF.simulateVoltageResponse(traintr.I, myGIF.dt)
#plt.figure()
#plt.plot(V)
#plt.title('training input')


# Plot the model parameters
myGIF.plotParameters()   
myGIF_pow.plotParameters()
myGIF_exp.plotParameters()

# Save the model
#myGIF.save('./myGIF.pck')


####################################################################################################
# STEP 4: EVALUATE THE GIF MODEL PERFORMANCE (USING MD*)
####################################################################################################

# Use the myGIF model to predict the spiking data of the test data set in myExp
#myPrediction = myExp.predictSpikes(myGIF, nb_rep=500)
#myPrediction_pow = myExp.predictSpikes(myGIF_pow, nb_rep=500)
myPrediction_exp = myExp.predictSpikes(myGIF_exp, nb_rep=500)

# Compute Md* with a temporal precision of +/- 4ms
#Md = myPrediction.computeMD_Kistler(4.0, 0.1)
#Md = myPrediction_pow.computeMD_Kistler(4.0, 0.1)
Md = myPrediction_exp.computeMD_Kistler(4.0, 0.1)    

# Plot data vs model prediction
#myPrediction.plotRaster(delta=500.0, dt = myGIF.dt) 
#myPrediction_pow.plotRaster(delta=500.0, dt = myGIF_pow.dt) 
myPrediction_exp.plotRaster(delta=1000.0, dt = myGIF_exp.dt)
plt.suptitle('GIF with three exponentials for gamma')

