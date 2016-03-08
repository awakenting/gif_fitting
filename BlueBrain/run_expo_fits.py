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

root_path = './article_4_data/grouped_ephys/'
model_path = './results/models/subadapt_constrained/'
animal_dirs = sorted(os.listdir(root_path))

#%% load fitted models
gifs =[]
for animalnr in range(len(animal_dirs)):
    if os.path.exists(model_path + 'Animal_' + animal_dirs[animalnr]):
        gifs.append(GIF_subadapt_constrained.load(model_path + 'Animal_' + animal_dirs[animalnr]))
    

#%% fit exponentials to filters
for gifnr,gif in enumerate(gifs):
    print('Fitting exponentials for model ' + str(gifnr) + 'of ' + str(len(gifs)), end='\r')
    if not gif.eta.expfit_falg:
        (t_eta, F_exp_eta) = gif.eta.fit_sumOfExpos_optimize_dim(maxdim=3, dt=gif.dt)
    if not gif.gamma.expfit_falg:
        (t_gamma, F_exp_gamma) = gif.gamma.fit_sumOfExpos_optimize_dim(maxdim=3, dt=gif.dt)
    
#%% save models with expo fit
for gif in gifs:
    gif.save(model_path + 'Animal_' + gif.save_path.split('_')[-1])   
    
#%% plot histograms of amplitudes
eta_amps = np.zeros((len(gifs),3))
gamma_amps = np.zeros((len(gifs),3))
eta_taus = np.zeros((len(gifs),3))
gamma_taus = np.zeros((len(gifs),3))
eta_sses = np.zeros((len(gifs),1))
gamma_sses = np.zeros((len(gifs),1))
eta_dims = np.zeros((len(gifs),1))
gamma_dims = np.zeros((len(gifs),1))

a_ws = np.zeros((len(gifs),1))

for gifnr,gif in enumerate(gifs):
    eta_amps[gifnr,0:gif.eta.expfit_dim] = gif.eta.b0
    eta_taus[gifnr,0:gif.eta.expfit_dim] = gif.eta.tau0
    eta_sses[gifnr]   = gif.eta.get_expfit_sse(dim=gif.eta.expfit_dim,dt=gif.dt)
    eta_dims [gifnr]  = gif.eta.expfit_dim
    
    gamma_amps[gifnr,0:gif.gamma.expfit_dim] = gif.gamma.b0
    gamma_taus[gifnr,0:gif.gamma.expfit_dim] = gif.gamma.tau0
    gamma_sses[gifnr]   = gif.gamma.get_expfit_sse(dim=gif.gamma.expfit_dim,dt=gif.dt)
    gamma_dims [gifnr]  = gif.gamma.expfit_dim
    
    a_ws[gifnr] = gif.a_w

#%% plot histograms    
# Eta
eta_first_amp = np.max(eta_amps,axis=1)
eta_first_amp_ind = np.argmax(eta_amps,axis=1)

plt.figure()
plt.hist(eta_first_amp, bins=50)
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
plt.hist(gamma_first_amp[gamma_first_amp < 1000], bins=50)
plt.xlabel('Amplitude [mV]')
plt.ylabel('Count')
plt.title('Gamma amplitudes')

plt.figure()
plt.hist(gamma_taus[np.arange(gamma_amps.shape[0]),gamma_first_amp_ind], bins=50)
plt.xlabel('Tau [ms]')
plt.ylabel('Count')
plt.title('Gamma time constants')

plt.figure()
plt.hist(eta_dims)
plt.xlabel('Number of exponentials used for fit')
plt.ylabel('# models')
plt.title('Eta')

plt.figure()
plt.hist(gamma_dims)
plt.xlabel('Number of exponentials used for fit')
plt.ylabel('# models')
plt.title('Gamma')

plt.figure()
plt.plot(eta_dims,a_ws,'.')
plt.ylabel('Value of a_ w[nA]')
plt.xlabel('Number of exponentials used to fit eta')
plt.title('# Exponentials vs Value of a_ w')

#%% 
plt.figure()
for gif in gifs:
    plt.subplot(2,1,1)
    (t, F) = gif.eta.getInterpolatedFilter(gif.dt)
    F_fit = gif.eta.multiExpEval(t, gif.eta.b0, gif.eta.tau0)
    plt.plot(t, F_fit, label='Multiexp fit',hold=True)
    
    plt.subplot(2,1,2)
    (t, F) = gif.gamma.getInterpolatedFilter(gif.dt)
    F_fit = gif.gamma.multiExpEval(t, gif.gamma.b0, gif.gamma.tau0)
    plt.plot(t, F_fit, label='Multiexp fit',hold=True)
    
#%% 
for gif in gifs:
    gif.gamma.plot()
    #gif.eta.plot()

#%% plot amplitudes vs taus
for i in np.arange(eta_amps.shape[0]):
    plt.plot(eta_taus[i,:],eta_amps[i,:],'.',MarkerSize=10, hold=True);

plt.xlim((-1,50));
plt.ylim((-3,5));
plt.xlabel('Tau eta [ms]')
plt.ylabel('Amplitude of according time constant')
plt.title('Amplitude vs time constant for Eta')
plt.show();

