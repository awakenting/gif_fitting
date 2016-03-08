# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:01:08 2016

@author: andrej
"""

from GIF_subth_adapt_constrained import GIF_subadapt_constrained
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
plt.style.use('ggplot')
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.facecolor'] = 'white'

root_path = '/home/andrej/Documents/Code/BlueBrain/article_4_data/grouped_ephys/'
model_path = '/home/andrej/Documents/Code/BlueBrain/results/models/subadapt_constrained/'
animal_dirs = sorted(os.listdir(root_path))

#%% load fitted models
gifs =[]
for animalnr in range(len(animal_dirs)):
    if os.path.exists(model_path + 'Animal_' + animal_dirs[animalnr]):
        gifs.append(GIF_subadapt_constrained.load(model_path + 'Animal_' + animal_dirs[animalnr]))
    

#%% fit exponentials to filters
for gifnr,gif in enumerate(gifs):
    b_init = np.array([1,1,1])
    taus_init = np.array([1,1,1])
    
    (t_eta, F_exp_eta) = gif.eta.fitSumOfExponentials(dim=3, bs=b_init, taus=taus_init, dt=gif.dt)
    (t_gamma, F_exp_gamma) = gif.gamma.fitSumOfExponentials(dim=3, bs=b_init, taus=taus_init, dt=gif.dt)
    
    
    
#%% plot histograms of amplitudes
eta_amps = np.zeros((len(gifs),3))
gamma_amps = np.zeros((len(gifs),3))
eta_taus = np.zeros((len(gifs),3))
gamma_taus = np.zeros((len(gifs),3))

for gifnr,gif in enumerate(gifs):
    eta_amps[gifnr,:] = gif.eta.b0
    eta_taus[gifnr,:] = gif.eta.tau0
    
    gamma_amps[gifnr,:] = gif.gamma.b0
    gamma_taus[gifnr,:] = gif.gamma.tau0
    
# Eta
eta_first_amp = np.max(eta_amps,axis=1)
eta_first_amp_ind = np.argmax(eta_amps,axis=1)

plt.figure()
plt.hist(eta_first_amp[eta_first_amp < 100], bins=50)
plt.xlabel('Amplitude [nA]')
plt.ylabel('Count')
plt.title('Eta amplitudes')

plt.figure()
plt.hist(eta_taus[np.arange(eta_amps.shape[0]),eta_first_amp_ind], bins=50)
plt.xlabel('Tau [ms]')
plt.ylabel('Count')
plt.title('Eta time constants')


# Gamma
gamma_first_amp = np.max(gamma_amps,axis=1)
gamma_first_amp_ind = np.argmax(gamma_amps,axis=1)

plt.figure()
plt.hist(gamma_first_amp[gamma_first_amp < 100], bins=50)
plt.xlabel('Amplitude [mV]')
plt.ylabel('Count')
plt.title('Gamma amplitudes')

plt.figure()
plt.hist(gamma_taus[np.arange(gamma_amps.shape[0]),gamma_first_amp_ind], bins=50)
plt.xlabel('Tau [ms]')
plt.ylabel('Count')
plt.title('Gamma time constants')
