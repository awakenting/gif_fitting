#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:55:08 2016

@author: andrej
"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle

plt.style.use('ggplot')
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.facecolor'] = 'white'

root_path = './full_dataset/article_4_data/grouped_ephys/'
figures_path = './results/figures/raw_data/full_dataset'
expm_path = './results/experiments/'

dirs = sorted(os.listdir(root_path))
animalset = np.arange(0,len(dirs))

from Experiment import Experiment
from LIF import LIF

def open_filterd_list(filtername):
    with open('/home/andrej/Dropbox/Arbeit/MKP/gif_fitting/BlueBrain/' + filtername + '_infos.pkl', 'rb') as f:
        filtered_list = pickle.load(f)
    return filtered_list

soms = open_filterd_list('som_animals')
vips = open_filterd_list('vip_animals')
pvs = open_filterd_list('pv_animals')

cell_names = list(vips.keys())
expms = []

for i in range(len(vips)):
    current_expm_name = 'Experiment_Cell_' + cell_names[i] + '_merged_idrest_traces.pkl'
    current_expm_path = os.path.join(expm_path,current_expm_name)
    try:
        current_expm = Experiment.load(current_expm_path)
        expms.append(current_expm)
    except:
        pass

my_expm = expms[0]

mylif = LIF(my_expm.dt)

mylif.Tref = 4.0
my_expm.detectSpikes_cython()
mylif.fitVoltageReset(my_expm,4.0)
tr = my_expm.trainingset_traces[0]