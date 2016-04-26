# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:36:33 2015

@author: andrej
"""

import os
from Experiment_auto_read_T import *
from AEC_Badel import *
from GIF import *
from Filter_Rect_LogSpaced import *

import Tools
import matplotlib.pyplot as plt

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
root_path = './article_4_data/grouped_ephys/'

dirs = sorted(os.listdir(root_path))

#%% print file names together with extracted T and dt
for i in range(5):#len(dirs)):
    PATH = root_path + dirs[i] + '/'
    files = sorted(os.listdir(PATH))
    for j in np.arange(len(files),step=2):
        myExp = Experiment_auto_read_T('Experiment 1')
        myExp.addTrainingSetTrace(PATH + files[j], 10**-3, PATH + files[j+1], 10**-12, FILETYPE='Igor')
        tr = myExp.trainingset_traces[0]
        print(files[j] + ': ' + str(len(tr.V)) + ' dt: ' +  str(tr.dt))
        print(files[j+1] + ': ' + str(len(tr.I)))

#%% plot data sets in one figure per animal

figures_path = './results/figures/raw_data'
animalset = np.arange(0,len(dirs))
for i in animalset:
    PATH = root_path + dirs[i] + '/'
    animal_files = sorted(os.listdir(PATH))
    plt.figure()
    plt.subplot(2,1,1)
    for j in np.arange(int(len(animal_files)/2)):
        # files end with 'recordingType_recordingNumber.ibw'
        file_split = str.split(animal_files[j][0:-4],'_')
        file_identifier = file_split[-2] + '_' + file_split[-1] + '.ibw'
        
        # find indeces of matching files in folder (current file always comes first because it's always Ch0)
        file_idc = [i for i, elem in enumerate(animal_files) if file_identifier in elem]
        current_file = animal_files[file_idc[0]]
        voltage_file = animal_files[file_idc[1]]
        
        myExp = Experiment_auto_read_T('Animal: ' + dirs[i] + ' Session: ' +file_identifier)
        myExp.addTrainingSetTrace(PATH + voltage_file, 10**-3, PATH + current_file, 10**-12, FILETYPE='Igor')
        tr = myExp.trainingset_traces[0]
        plt.subplot(2,1,1)
        plt.plot(tr.getTime(),tr.I)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current [pA]')
        
        plt.subplot(2,1,2)
        plt.plot(tr.getTime(),tr.V)
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [mV]')
        
        plt.suptitle('Animal: ' + dirs[i] + ' Cell: ' +file_identifier)
        
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
    
    plt.savefig(figures_path+dirs[i])
    plt.close()
