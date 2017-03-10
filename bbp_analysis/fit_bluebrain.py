import os

from .Experiment_auto_read_T import Experiment_auto_read_T
from .Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from .GIF import GIF
from . import Tools


def run(animal_number, root_path, unwanted_sessions):
    #%%
    ####################################################################################################
    # STEP 1: LOAD EXPERIMENTAL DATA
    #################################################################################################### 
   
    animal_dirs = sorted(os.listdir(root_path))
    PATH = root_path + animal_dirs[animal_number] + '/'
    animal_files = sorted(os.listdir(PATH))
    
    animals_wanted = []
    for filename in animal_files:
        # files end with 'recordingType_recordingNumber.ibw'
        file_split = str.split(filename[0:-4],'_')
        if not file_split[-2] in unwanted_sessions:
            animals_wanted.append(filename)
    
    
    myExp = Experiment_auto_read_T('Animal_' + animal_dirs[animal_number])
    
    num_files = int(len(animals_wanted)/2)
    for file in animals_wanted[0:num_files]:
        # get matching voltage and current file
        files = Tools.get_matching_file_pair(animals_wanted, file)
        # Load training set data
        myExp.addTrainingSetTrace(PATH + files[1], 10**-3, PATH + files[0], 10**-12,
                                  FILETYPE='Igor')
                                      
                                      
    for file in animals_wanted[0:num_files]:
        # get matching voltage and current file
        files = Tools.get_matching_file_pair(animals_wanted, file)
        # Load training set data
        myExp.addTestSetTrace(PATH + files[1], 10**-3, PATH + files[0], 10**-12,
                                      FILETYPE='Igor')
    
    myExp.mergeTrainingTraces()
    if not len(myExp.trainingset_traces) == 0:
        tracelength = myExp.trainingset_traces[0].T
    else:
        tracelength = 0
    # stop if training trace is smaller than 10 seconds:
    if  tracelength < 10000:
        print('Animal number ' + str(animal_number+1) + ' was not fitted because the merged trace'\
              ' is only ' + str(tracelength) + ' milliseconds long. It needs to be at least 10'\
              ' seconds long')
        return False
        
    myExp.mergeTestTraces()
    
    
    #%%
    ####################################################################################################
    # STEP 2: FIT GIF MODEL TO DATA
    ####################################################################################################
    
    # Create a new object GIF 
    myGIF = GIF(myExp.dt)
    
    # Define parameters
    myGIF.Tref = 4.0  
    
    myGIF.eta = Filter_Rect_LogSpaced()
    myGIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
    
    myGIF.gamma = Filter_Rect_LogSpaced()
    myGIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)
    
    # detect Spikes
    myExp.detectSpikes_cython()
    
    # Perform the fit
    myGIF.fit(myExp, DT_beforeSpike=5.0)
    
    
    #%%
    ####################################################################################################
    # STEP 3: EVALUATE THE GIF MODEL PERFORMANCE (USING MD*)
    ####################################################################################################
    
    # Use the myGIF model to predict the spiking data of the test data set in myExp
    myPrediction = myExp.predictSpikes(myGIF, nb_rep=500)
    
    # Compute Md* with a temporal precision of +/- 4ms
    Md = myPrediction.computeMD_Kistler(4.0, 0.1)
    
    return myGIF, myExp, Md, myPrediction

