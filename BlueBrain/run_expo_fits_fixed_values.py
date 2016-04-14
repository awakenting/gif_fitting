# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:08:43 2016

@author: andrej
"""

from GIF_subth_adapt_constrained import GIF_subadapt_constrained
from Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler
import matplotlib as mpl
plt.style.use('ggplot')
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.facecolor'] = 'white'

root_path = './article_4_data/grouped_ephys/'
model_path = './results/models/subadapt_constrained/'
figure_path = './results/figures/fits/subadapt_constrained/'
animal_dirs = sorted(os.listdir(root_path))

#%% load fitted models
gifs =[]
for animalnr in range(len(animal_dirs)):
    if os.path.exists(model_path + 'Animal_' + animal_dirs[animalnr]):
        gifs.append(GIF_subadapt_constrained.load(model_path + 'Animal_' + animal_dirs[animalnr] + '_tau_w_50ms'))

#%% run fits for fixed values of time constants
eta_taus = [5,50,100]
gamma_taus = [10,50,100]
b_inits = [np.ones((1,1))]

for gifnr,gif in enumerate(gifs):
    print('Fitting exponentials for gif ' + str(gifnr+1) + ' of ' + str(len(gifs)), end='\r', flush=True)
    for filter_name,taus in zip(['gamma', 'eta'],[gamma_taus,eta_taus]):
        # I create the filter again because I sometimes want to update the
        # filter class but not run the model fit again
        filter_obj = getattr(gif,filter_name)
        new_filter = Filter_Rect_LogSpaced()
        new_filter.setMetaParameters(filter_obj.p_length, binsize_lb=filter_obj.p_binsize_lb,
                                    binsize_ub=filter_obj.p_binsize_ub, slope=filter_obj.p_slope)
        new_filter.setFilter_Coefficients(filter_obj.getCoefficients())
        filter_obj = new_filter
        
        # create combinations of tau values
        tau_inits = [np.array([[taus[0]]]),
                     np.array([[taus[0]],[taus[1]]]),
                     np.array([[taus[0]],[taus[2]]]),
                     np.array([[taus[0]],[taus[1]],[taus[2]]])]
        # intialize b with ones
        b_inits = [np.ones((1,1)),np.ones((2,1)),np.ones((2,1)),np.ones((3,1))]
        
        # run expo fit
        (tau_set_opt, opt_dim) = filter_obj.fit_sumOfExpos_optimize_dim(maxdim=len(eta_taus), bs=b_inits,
                                             tausets=tau_inits, ROI=[0,300], dt=gif.dt, fixed_taus=True)
                                             
        setattr(gif,filter_name,filter_obj)
    
    gif.save(gif.save_path)
    
#%% collect values

eta_amps = np.zeros((len(gifs),3))
eta_taus = np.zeros((len(gifs),3))
eta_dims = np.zeros((len(gifs),1))

gamma_amps = np.zeros((len(gifs),3))
gamma_taus = np.zeros((len(gifs),3))
gamma_dims = np.zeros((len(gifs),1))


for gifnr,gif in enumerate(gifs):
    eta_amps[gifnr,0:len(gif.eta.b0)] = np.array(gif.eta.b0).squeeze()
    eta_taus[gifnr,0:len(gif.eta.tau0)] = np.array(gif.eta.tau0).squeeze()
    if type(gif.eta.tau0)== list:
        gif.eta.tau0 = np.reshape(np.array(gif.eta.tau0),(len(gif.eta.tau0),1))
    eta_dims [gifnr]  = gif.eta.expfit_dim
    
    gamma_amps[gifnr,0:len(gif.gamma.b0)] = np.array(gif.gamma.b0).squeeze()
    gamma_taus[gifnr,0:len(gif.gamma.tau0)] = np.array(gif.gamma.tau0).squeeze()
    if type(gif.gamma.tau0)== list:
        gif.gamma.tau0 = np.reshape(np.array(gif.gamma.tau0),(len(gif.gamma.tau0),1))
    gamma_dims [gifnr]  = gif.gamma.expfit_dim
    
#%% plot histograms    
# Eta
eta_first_amp = np.max(eta_amps,axis=1)
eta_first_amp_ind = np.argmax(eta_amps,axis=1)

# Gamma
gamma_first_amp = np.max(gamma_amps,axis=1)
gamma_first_amp_ind = np.argmax(gamma_amps,axis=1)

fig = plt.figure(figsize=(24,20), facecolor='white')
subrows = 4
subcols = 4    
gs = gridspec.GridSpec(subrows, subcols, left=0.1, right=0.90, bottom=0.1, top=0.9)
gs.update(hspace=0.9, wspace=0.9)

ax1 = plt.subplot(gs[0  ,0:2])
ax2 = plt.subplot(gs[1  ,0:2])
ax3 = plt.subplot(gs[2  ,0:2])
ax4 = plt.subplot(gs[3  ,0:2])
ax5 = plt.subplot(gs[0  ,2:4])
ax6 = plt.subplot(gs[1  ,2:4])
ax7 = plt.subplot(gs[2  ,2:4])
ax8 = plt.subplot(gs[3  ,2:4])

#==============================================================================
# Eta plots
#==============================================================================
ax1.hist(eta_first_amp, bins=50)
ax1.set_xlabel('Amplitude [nA]')
ax1.set_ylabel('# models')
ax1.set_title('Eta\n\n  Amplitudes for tau = 5ms')

ax2.hist(eta_taus, bins=50)
ax2.set_xlabel('Tau [ms]')
ax2.set_ylabel('# models')
ax2.set_title('Time constants')

ax3.hist(eta_dims)
ax3.set_xlabel('# exponentials')
ax3.set_ylabel('# models')
ax3.set_xticks([1,2,3])
ax3.set_xticklabels([1,2,3])
ax3.set_title('Number of exponentials used for fit')

ax4.plot(eta_dims,a_ws,'.',MarkerSize=12)
ax4.hold(True)
ax4.plot([0.5,3.5], [0.0,0.0], ':', color='black')
ax4.set_xlim([0.5,3.5])
ax4.set_ylabel('Value of a_ w[nA]')
ax4.set_xlabel('# exponentials')
ax4.set_title('# exponentials vs value of a_ w')

#==============================================================================
# Gamma plots
#==============================================================================
gam_color = next(ax1._get_lines.prop_cycler)['color']
ax5.hist(gamma_first_amp[gamma_first_amp < 1000], bins=50, color=gam_color)
ax5.set_xlabel('Amplitude [mV]')
ax5.set_ylabel('# models')
ax5.set_title('Gamma\n\n Amplitudes')

ax6.hist(gamma_taus, bins=50)#, color=gam_color)
ax6.set_xlabel('Tau [ms]')
ax6.set_ylabel('# models')
ax6.set_title('Time constants')

ax7.hist(gamma_dims, color=gam_color)
ax7.set_xlabel('# exponentials')
ax7.set_ylabel('# models')
ax7.set_xticks([1,2,3])
ax7.set_xticklabels([1,2,3])
ax7.set_title('Number of exponentials used for fit')

ax8.plot(gamma_dims,a_ws,'.',MarkerSize=12, color=gam_color)
ax8.hold(True)
ax8.plot([0.5,3.5], [0.0,0.0], ':', color='black')
ax8.set_xlim([0.5,3.5])
ax8.set_ylabel('Value of a_ w[nA]')
ax8.set_xlabel('# exponentials')
ax8.set_title('# exponentials vs value of a_ w')

plt.suptitle('Summary of exponential fits',fontsize=24)

#==============================================================================
#  Save figure
#==============================================================================
    
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
    
plt.savefig(figure_path + 'expofit_allTaus_TauW_50_stats.png', dpi=120)
plt.close(fig)
    
#%% 
#==============================================================================
#  single gamma fits figure
#==============================================================================
fig = plt.figure(figsize=(24,20), facecolor='white')
subrows = 5
subcols = 5    
gs = gridspec.GridSpec(subrows, subcols, left=0.05, right=0.95, bottom=0.05, top=0.9)
gs.update(hspace=0.3, wspace=0.3)

for gifnr,gif in enumerate(gifs):
    
    plt.subplot(gs[gifnr//5, gifnr%5])
    (t, F) = gif.gamma.getInterpolatedFilter(gif.dt)
    plt.plot([t[0],t[-1]], [0.0,0.0], ':', color='black')
    plt.plot(t, F, 'black', label='Filter')
    F_fit = gif.gamma.multiExpEval(t, gif.gamma.b0, gif.gamma.tau0)
    plt.plot(t, F_fit, label='Multiexp fit',hold=True)
    
    plt.legend()

for i in np.arange(0,21,step=5):
    plt.subplot(gs[i//5, i%5])
    plt.ylabel('Filter')
    
for i in np.arange(19,25,step=1):
    plt.subplot(gs[i//5, i%5])
    plt.xlabel('Time [ms]')
    
plt.suptitle('Exponential fits for gamma', fontsize=24)
    
if not os.path.exists(figure_path):
    os.mkdir(figure_path)
    
plt.savefig(figure_path + 'single_expofits_gamma_fixed_taus.png', dpi=120)
plt.close(fig)


#==============================================================================
#  single eta fits figure
#==============================================================================
fig = plt.figure(figsize=(24,20), facecolor='white')
subrows = 5
subcols = 5    
gs = gridspec.GridSpec(subrows, subcols, left=0.05, right=0.95, bottom=0.05, top=0.9)
gs.update(hspace=0.3, wspace=0.3)

for gifnr,gif in enumerate(gifs):
    plt.subplot(gs[gifnr//5, gifnr%5])
    (t, F) = gif.eta.getInterpolatedFilter(gif.dt)
    plt.plot([t[0],t[-1]], [0.0,0.0], ':', color='black')
    plt.plot(t, F, 'black', label='Filter')
    F_fit = gif.eta.multiExpEval(t, gif.eta.b0, gif.eta.tau0)
    plt.plot(t, F_fit, label='Multiexp fit',hold=True)
    plt.legend()

for i in np.arange(0,22,step=5):
    plt.subplot(gs[i//5, i%5])
    plt.ylabel('Filter')
    
for i in np.arange(19,25,step=1):
    plt.subplot(gs[i//5, i%5])
    plt.xlabel('Time [ms]')
    
plt.suptitle('Exponential fits for eta', fontsize=24)
    
if not os.path.exists(figure_path):
    os.mkdir(figure_path)
    
plt.savefig(figure_path + 'single_expofits_eta_fixed_taus.png', dpi=120)
plt.close(fig)


