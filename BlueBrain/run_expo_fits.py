# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:01:08 2016

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
        gifs.append(GIF_subadapt_constrained.load(model_path + 'Animal_' + animal_dirs[animalnr]))
    

#%% fit exponentials to filters
for gifnr,gif in enumerate(gifs):
    print('Fitting exponentials for model ' + str(gifnr) + 'of ' + str(len(gifs)), end='\r')
    if not gif.eta.expfit_falg:
        (t_eta, F_exp_eta) = gif.eta.fit_sumOfExpos_optimize_dim(maxdim=3, dt=gif.dt)
    if not gif.gamma.expfit_falg:
        (t_gamma, F_exp_gamma) = gif.gamma.fit_sumOfExpos_optimize_dim(maxdim=3, dt=gif.dt)
        
#%% fit exponentials to eta
#eta_taus = [np.array([[5]]),np.array([[5],[30]]),np.array([[5],[75],[30]])]
#
#for gifnr,gif in enumerate(gifs):
#    print('Fitting exponentials for model ' + str(gifnr) + ' of ' + str(len(gifs)), end='\r')
#    (t_eta, F_exp_eta) = gif.eta.fit_sumOfExpos_optimize_dim(maxdim=3, taus=eta_taus, ROI=[0,300], dt=gif.dt)
    
#%% fit exponentials to gamma
#gamma_tau_inits = [np.array([[20]]),np.array([[20],[60]]),np.array([[20],[60],[100]])]
#gamma_bs = [np.array([[150]]),np.array([[150],[-70]]),np.array([[150],[-70],[-10]])]
#
#for gifnr,gif in enumerate(gifs):
#    print('Fitting exponentials for model ' + str(gifnr) + ' of ' + str(len(gifs)), end='\r')
#    (t_gamma, F_exp_gamma) = gif.gamma.fit_sumOfExpos_optimize_dim(maxdim=3, 
#                            bs=gamma_bs, taus=gamma_tau_inits, ROI=[0,300], dt=gif.dt)
                            
                            
#    myGIF.eta = Filter_Rect_LogSpaced()
#    myGIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
#    
#    myGIF.gamma = Filter_Rect_LogSpaced()
#    myGIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)
#%% fit exponentials to gamma
                            
tau1_range = np.arange(3,19,3)
tau2_range = np.arange(30,51,5)
tau3_range = np.arange(10,17,1)

tau_sses = np.zeros((len(tau1_range),len(tau2_range),len(tau3_range)))
print('Gridsearch for '+str(tau_sses.size)+' values on a '+str(tau_sses.shape)+' grid for the '+\
      str(tau_sses.ndim)+' time constants!')
for tau1_idx,tau1 in enumerate(tau1_range):
    for tau2_idx,tau2 in enumerate(tau2_range):
        for tau3_idx,tau3 in enumerate(tau3_range):
            
            gamma_sses = np.zeros((len(gifs),1))
            
            gamma_tau_inits = [np.array([[tau1]]),
                               np.array([[tau1],[tau2]]),
                               np.array([[tau1],[tau2],[tau3]])]
            gamma_bs = [np.array([[150]]),np.array([[150],[-70]]),np.array([[150],[-70],[-10]])]
            
            for gifnr,gif in enumerate(gifs):
                new_gamma = Filter_Rect_LogSpaced()
                new_gamma.setMetaParameters(gif.gamma.p_length, binsize_lb=gif.gamma.p_binsize_lb,
                                            binsize_ub=gif.gamma.p_binsize_ub, slope=gif.gamma.p_slope)
                new_gamma.setFilter_Coefficients(gif.gamma.getCoefficients())
                gif.gamma = new_gamma
                
                (t_gamma, F_exp_gamma) = gif.gamma.fit_sumOfExpos_optimize_dim(maxdim=3, bs=gamma_bs,
                                         taus=gamma_tau_inits, ROI=[0,300], dt=gif.dt, fixed_taus=True)
                                         
                gamma_sses[gifnr]   = gif.gamma.get_expfit_sse(dim=gif.gamma.expfit_dim,dt=gif.dt)
            
            tau_sses[tau1_idx,tau2_idx,tau3_idx] = gamma_sses.sum()

(tau1_idx_opt,tau2_idx_opt,tau3_idx_opt) = np.unravel_index(np.argmin(tau_sses),tau_sses.shape)

tau1_opt = tau1_range[tau1_idx_opt]
tau2_opt = tau2_range[tau2_idx_opt]
tau3_opt = tau3_range[tau3_idx_opt]

print('Optimal set of taus for this set of models was: '+str(tau1_opt)+'ms, '+str(tau2_opt)+'ms and '+str(tau3_opt)+'ms.' )

#%% inspect sse grid
fig = plt.figure(figsize=(24,20), facecolor='white')
subrows = np.ceil(np.sqrt(len(tau3_range)))
subcols = np.ceil(len(tau3_range)/subrows)
gs = gridspec.GridSpec(int(subrows), int(subcols), left=0.1, right=0.90, bottom=0.1, top=0.9)
gs.update(hspace=0.3, wspace=0.3)

for ctau3_idx,ctau3 in enumerate(tau3_range):
    plt.subplot(gs[ctau3_idx])
    img = plt.imshow(tau_sses[:,:,ctau3_idx], aspect='auto', interpolation='none', origin='lower')
    img.axes.set_yticks(np.arange(len(tau1_range)))
    img.axes.set_yticklabels(tau1_range)
    img.axes.set_xticks(np.arange(len(tau2_range)))
    img.axes.set_xticklabels(tau2_range)
    plt.xlabel('Second time constant [ms]',fontsize=12)
    plt.ylabel('First time constant [ms]',fontsize=12)
    plt.title('Total SSE, for tau 3 = '+str(ctau3),fontsize=12)
    plt.colorbar()
    plt.clim(tau_sses.min(),tau_sses.min()*1.5)

#%% use optimal tau values for actual fit

gamma_tau_opt_inits = [np.array([[tau1_opt]]),
                       np.array([[tau1_opt],[tau2_opt]]),
                       np.array([[tau1_opt],[tau2_opt],[tau3_opt]])]
gamma_bs = [np.array([[150]]),np.array([[150],[-70]]),np.array([[150],[-70],[-10]])]
for gifnr,gif in enumerate(gifs):
    print('Fitting exponentials for model ' + str(gifnr) + ' of ' + str(len(gifs)), end='\r')
    (t_gamma, F_exp_gamma) = gif.gamma.fit_sumOfExpos_optimize_dim(maxdim=3, bs=gamma_bs,
                             taus=gamma_tau_opt_inits, ROI=[0,300], dt=gif.dt, fixed_taus=True)

    
#%% save models with expo fit
for gif in gifs:
    gif.save(model_path + 'Animal_' + gif.save_path.split('_')[-1])   
    
#%% plot histograms of amplitudes
eta_amps = np.zeros((len(gifs),3))
gamma_amps = np.zeros((len(gifs),3))
eta_taus = np.zeros((len(gifs),3))
gamma_taus = np.zeros((len(gifs),3))
eta_sses = np.zeros((len(gifs),1))
eta_dims = np.zeros((len(gifs),1))
gamma_dims = np.zeros((len(gifs),1))

a_ws = np.zeros((len(gifs),1))

for gifnr,gif in enumerate(gifs):
    eta_amps[gifnr,0:gif.eta.expfit_dim] = gif.eta.b0
    eta_taus[gifnr,0:gif.eta.expfit_dim] = gif.eta.tau0
    eta_sses[gifnr]   = gif.eta.get_expfit_sse(dim=gif.eta.expfit_dim,dt=gif.dt)
    eta_dims [gifnr]  = gif.eta.expfit_dim
    
    gamma_amps[gifnr,0:gif.gamma.expfit_dim] = gif.gamma.b0.squeeze()
    gamma_taus[gifnr,0:gif.gamma.expfit_dim] = gif.gamma.tau0.squeeze()
    gamma_dims [gifnr]  = gif.gamma.expfit_dim
    
    a_ws[gifnr] = gif.a_w

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
ax1.set_title('Eta\n\n  Amplitudes')

ax2.hist(eta_taus[np.arange(eta_amps.shape[0]),eta_first_amp_ind], bins=50)
ax2.set_xlabel('Tau [ms]')
ax2.set_ylabel('# models')
ax2.set_title('Ttime constants')

ax3.hist(eta_dims)
ax3.set_xlabel('# exponentials')
ax3.set_ylabel('# models')
ax3.set_xticks([1,2,3])
ax3.set_xticklabels([1,2,3])
ax3.set_title('Number of exponentials used for fit')

#==============================================================================
# Eta vs a_w plots
#==============================================================================
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

ax6.hist(gamma_taus[np.arange(gamma_amps.shape[0]),gamma_first_amp_ind], bins=50, color=gam_color)
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
    
plt.savefig(figure_path + '_expofit_stats.png', dpi=120)
#plt.close(fig)

#%% 

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig = plt.figure(figsize=(24,20), facecolor='white')
subrows = 2
subcols = 1    
gs = gridspec.GridSpec(subrows, subcols, left=0.1, right=0.90, bottom=0.1, top=0.9)
gs.update(hspace=0.9, wspace=0.5)

for gifnr,gif in enumerate(gifs):
    ax1 = plt.subplot(gs[0,0])
    (t, F) = gif.eta.getInterpolatedFilter(gif.dt)
    F_fit = gif.eta.multiExpEval(t, gif.eta.b0, gif.eta.tau0)
    plt.plot(t, F_fit, label='Multiexp fit', hold=True)
    plt.xlabel('Time [ms]')
    plt.ylabel('Adaptation current [nA]')
    plt.title('Fitted etas')
        
    ax2 = plt.subplot(gs[1,0])
    (t, F) = gif.gamma.getInterpolatedFilter(gif.dt)
    F_fit = gif.gamma.multiExpEval(t, gif.gamma.b0, gif.gamma.tau0)
    plt.plot(t, F_fit, label='Multiexp fit',hold=True)
    plt.xlabel('Time [ms]')
    plt.ylabel('Change in voltage threshold [mV]')
    plt.title('Fitted gammas')
    

#eta_subax = zoomed_inset_axes(ax1, 2, loc=1) # zoom = 6
eta_subax = inset_axes(ax1,
                   width="70%",  # width = 10% of parent_bbox width
                   height="50%",  # height : 50%
                   loc=1)
                   
gamma_subax = inset_axes(ax2,
               width="70%",  # width = 10% of parent_bbox width
               height="50%",  # height : 50%
               loc=1)

for gifnr,gif in enumerate(gifs):
    (t, F) = gif.eta.getInterpolatedFilter(gif.dt)
    F_fit = gif.eta.multiExpEval(t, gif.eta.b0, gif.eta.tau0)
    eta_subax.hold('on')
    eta_subax.plot(t, F_fit, lw=2)
    eta_subax.plot([t[0], t[-1]], [0,0], ls=':', color='black', lw=2)
    eta_subax.set_xlim(0,20)
    eta_subax.set_ylim(0,1)
    
    (t, F) = gif.gamma.getInterpolatedFilter(gif.dt)
    F_fit = gif.gamma.multiExpEval(t, gif.gamma.b0, gif.gamma.tau0)
    gamma_subax.hold('on')
    gamma_subax.plot(t, F_fit, lw=2)
    gamma_subax.plot([t[0], t[-1]], [0,0], ls=':', color='black', lw=2)
    gamma_subax.set_xlim(0,100)    
    
    
#==============================================================================
#  Save figure
#==============================================================================
    
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
    
plt.savefig(figure_path + 'ensemble_expofits.png', dpi=120)
#plt.close(fig)
    
#%% 
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
    
plt.savefig(figure_path + 'single_expofits_gamma.png', dpi=120)
#plt.close(fig)


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
    
plt.savefig(figure_path + 'single_expofits_eta.png', dpi=120)
plt.close(fig)

