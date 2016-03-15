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
        gifs.append(GIF_subadapt_constrained.load(model_path + 'Animal_' + animal_dirs[animalnr] + '_large_tau_w'))

#%% fit exponentials to filters
for gifnr,gif in enumerate(gifs):
    print('Fitting exponentials for model ' + str(gifnr) + 'of ' + str(len(gifs)), end='\r')
    if not gif.eta.expfit_falg:
        (t_eta, F_exp_eta) = gif.eta.fit_sumOfExpos_optimize_dim(maxdim=3, dt=gif.dt)
    if not gif.gamma.expfit_falg:
        (t_gamma, F_exp_gamma) = gif.gamma.fit_sumOfExpos_optimize_dim(maxdim=3, dt=gif.dt)

#%% do tau grid search for all models
                            
tau1_range = np.arange(2,11,2)
tau2_range = np.arange(35,61,5)
tau3_range = np.arange(100,251,50)
tau_ranges = [tau1_range,tau2_range,tau3_range]
tau_sses = np.zeros(([len(crange) for crange in tau_ranges]))

print('You are about to do a grid search for '+str(tau_sses.size)+
  ' combinations on a '+str(tau_sses.shape)+' grid for '+
  str(tau_sses.ndim)+' time constants and this for '+str(len(gifs))+
  ' models!',flush=True)
  
answer = str(input('\nDo you want to proceed? Enter y for yes or n for no: '))
if answer=='n':
    print('\nNot performing the grid search.')
else:
    for gifnr,gif in enumerate(gifs):
        print('Fitting exponentials for gif ' + str(gifnr+1) + ' of ' + str(len(gifs)), end='\r', flush=True)
        for filter_name in ['gamma', 'eta']:
            # I create the filter again because I sometimes want to update the
            # filter class but not run the model fit again
            filter_obj = getattr(gif,filter_name)
            new_filter = Filter_Rect_LogSpaced()
            new_filter.setMetaParameters(filter_obj.p_length, binsize_lb=filter_obj.p_binsize_lb,
                                        binsize_ub=filter_obj.p_binsize_ub, slope=filter_obj.p_slope)
            new_filter.setFilter_Coefficients(filter_obj.getCoefficients())
            filter_obj = new_filter
            (t_gamma, F_exp_gamma) = filter_obj.expfit_tau_gridsearch(tau_ranges=tau_ranges, ROI=[0,300], dt=gif.dt)
            
            setattr(gif,filter_name,filter_obj)
        
        gif.save(gif.save_path)   
    

#%% inspect sse grid

def plot_gridsearch_result(tau_sses, tau_ranges, tau_opt, ptitle='default title',save=False):
    fig = plt.figure(figsize=(24,20), facecolor='white')
    subrows = np.ceil(np.sqrt(tau_ranges[-1].shape))
    subcols = np.ceil(tau_ranges[-1].shape/subrows)
    gs = gridspec.GridSpec(int(subrows), int(subcols), left=0.1, right=0.90, bottom=0.1, top=0.9)
    gs.update(hspace=0.5, wspace=0.5)
    
    for ctau3_idx,ctau3 in enumerate(tau_ranges[-1]):
        plt.subplot(gs[ctau3_idx])
        img = plt.imshow(tau_sses[:,:,ctau3_idx], aspect='auto', interpolation='none', origin='lower')
        img.axes.set_yticks(np.arange(len(tau_ranges[0])))
        img.axes.set_yticklabels(tau_ranges[0])
        img.axes.set_xticks(np.arange(len(tau_ranges[1]),step=2))
        img.axes.set_xticklabels(np.arange(tau_ranges[1][0],np.max(tau_ranges[1]),step=10))
        plt.xlabel('Second time constant [ms]',fontsize=12)
        plt.ylabel('First time constant [ms]',fontsize=12)
        plt.title('Total SSE, for tau 3 = '+str(ctau3) + 'ms',fontsize=12)
        plt.colorbar()
        plt.clim(tau_sses.min(),tau_sses.min()*1.5)
        
    plt.suptitle(ptitle,fontsize=20)
    
    fig.text(0.8,0.95, 'Optimal set of taus: '+str(tau_opt.squeeze()))
    
    if save==True:
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
            
        plt.savefig(figure_path + ptitle, dpi=120)
        plt.close(fig)
        
#%% plot gridsearch result
plot_gridsearch_result(gifs[0].gamma.tau_sses, gifs[0].gamma.tau_ranges)

#%% use optimal tau values for actual fit

#gamma_tau_opt_inits = [np.array([[tau1_opt]]),
#                       np.array([[tau1_opt],[tau2_opt]]),
#                       np.array([[tau1_opt],[tau2_opt],[tau3_opt]])]
#gamma_bs = [np.array([[150]]),np.array([[150],[-70]]),np.array([[150],[-70],[-10]])]
#for gifnr,gif in enumerate(gifs):
#    print('Fitting exponentials for model ' + str(gifnr) + ' of ' + str(len(gifs)), end='\r')
#    (t_gamma, F_exp_gamma) = gif.gamma.fit_sumOfExpos_optimize_dim(maxdim=3, bs=gamma_bs,
#                             taus=gamma_tau_opt_inits, ROI=[0,300], dt=gif.dt, fixed_taus=True)
 
    
#%% plot histograms of amplitudes
eta_amps = np.zeros((len(gifs),3))
eta_taus = np.zeros((len(gifs),3))
eta_sses = np.zeros((len(gifs),1))
eta_dims = np.zeros((len(gifs),1))

gamma_amps = np.zeros((len(gifs),3))
gamma_taus = np.zeros((len(gifs),3))
gamma_sses = np.zeros((len(gifs),1))
gamma_dims = np.zeros((len(gifs),1))

sses_shape = list(gifs[0].eta.tau_sses.shape)
sses_shape.append(len(gifs))
single_eta_tau_sses = np.zeros(sses_shape)
single_gamma_tau_sses = np.zeros(sses_shape)

for gifnr,gif in enumerate(gifs):
    eta_amps[gifnr,0:len(gif.eta.b0)] = np.array(gif.eta.b0).squeeze()
    eta_taus[gifnr,0:len(gif.eta.tau0)] = np.array(gif.eta.tau0).squeeze()
    if type(gif.eta.tau0)== list:
        gif.eta.tau0 = np.reshape(np.array(gif.eta.tau0),(len(gif.eta.tau0),1))
    eta_sses[gifnr]   = gif.eta.get_expfit_sse(dim=len(gif.eta.b0),dt=gif.dt)
    eta_dims [gifnr]  = gif.eta.expfit_dim
    
    gamma_amps[gifnr,0:len(gif.gamma.b0)] = np.array(gif.gamma.b0).squeeze()
    gamma_taus[gifnr,0:len(gif.gamma.tau0)] = np.array(gif.gamma.tau0).squeeze()
    if type(gif.gamma.tau0)== list:
        gif.gamma.tau0 = np.reshape(np.array(gif.gamma.tau0),(len(gif.gamma.tau0),1))
    gamma_sses[gifnr]   = gif.gamma.get_expfit_sse(dim=len(gif.gamma.b0),dt=gif.dt)
    gamma_dims [gifnr]  = gif.gamma.expfit_dim
    
    single_eta_tau_sses[:,:,:,gifnr] = gif.eta.tau_sses
    single_gamma_tau_sses[:,:,:,gifnr] = gif.gamma.tau_sses
    
#%% tau_w and a_w stats 
a_ws = np.zeros((len(gifs),1))
w_taus = np.zeros((len(gifs),1))
for gifnr,gif in enumerate(gifs):
    a_ws[gifnr] = gif.a_w
    w_taus[gifnr] = gif.tau_w_opt
    
#%% find best set of taus with regards to sum error of all models combined
sum_eta_tau_sses = np.sum(single_eta_tau_sses,axis=3)
sum_gamma_tau_sses = np.sum(single_gamma_tau_sses,axis=3)

sum_eta_tau_opt_idx = np.unravel_index(np.argmin(sum_eta_tau_sses),tau_sses.shape)
sum_eta_tau_opt = [gifs[0].eta.tau_ranges[cdim][sum_eta_tau_opt_idx[cdim]] for cdim in range(len(gifs[0].eta.tau_ranges))]
sum_eta_tau_opt = np.reshape(np.array(sum_eta_tau_opt),(len(sum_eta_tau_opt),1))

sum_gamma_tau_opt_idx = np.unravel_index(np.argmin(sum_gamma_tau_sses),tau_sses.shape)
sum_gamma_tau_opt = [gifs[0].gamma.tau_ranges[cdim][sum_gamma_tau_opt_idx[cdim]] for cdim in range(len(gifs[0].gamma.tau_ranges))]
sum_gamma_tau_opt = np.reshape(np.array(sum_gamma_tau_opt),(len(sum_gamma_tau_opt),1))

plot_gridsearch_result(sum_eta_tau_sses, gifs[0].eta.tau_ranges, sum_eta_tau_opt, ptitle='total sse for all eta_taucomb',save=True)
plot_gridsearch_result(sum_gamma_tau_sses, gifs[0].gamma.tau_ranges, sum_gamma_tau_opt, ptitle='total sse for all gamma_taucomb',save=True)
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
    
plt.savefig(figure_path + 'expofit_allTaus_largeTauW_stats.png', dpi=120)
plt.close(fig)

#%% 

fig = plt.figure(figsize=(24,8), facecolor='white')
subrows = 1
subcols = 3    
gs = gridspec.GridSpec(subrows, subcols, left=0.1, right=0.90, bottom=0.1, top=0.9)
gs.update(hspace=0.5, wspace=0.8)

plt.subplot(gs[0])
plt.hist(w_taus,bins=20)
plt.plot([0.0,0.0],[0,10], ':', color='black')
plt.xlabel('tau_w')
plt.ylabel('# models')
plt.title('tau_w histogram')

plt.subplot(gs[1])
plt.hist(a_ws,bins=10)
plt.plot([0.0,0.0],[0,10], ':', lw=3, color='black')
plt.xlim([-0.02,0.02])
plt.xticks(np.arange(-0.01,0.021,0.01))
plt.xlabel('a_w')
plt.ylabel('# models')
plt.title('a_w histogram')

plt.subplot(gs[2])
plt.plot(w_taus,a_ws,'.')
plt.plot([0,310],[0.0,0.0], ':', lw=3, color='black')
plt.xlim([0,310])
plt.ylim([-0.02,0.02])
plt.xlabel('tau_w [ms]')
plt.ylabel('a_w')
plt.title('a_w vs tau_w')

if not os.path.exists(figure_path):
    os.makedirs(figure_path)
    
plt.savefig(figure_path + 'w_current_stats_large_tau.png', dpi=120)
plt.close(fig)

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
    plt.xlim([0,300])
    plt.xlabel('Time [ms]')
    plt.ylabel('Adaptation current [nA]')
    plt.title('Fitted etas')
        
    ax2 = plt.subplot(gs[1,0])
    (t, F) = gif.gamma.getInterpolatedFilter(gif.dt)
    F_fit = gif.gamma.multiExpEval(t, gif.gamma.b0, gif.gamma.tau0)
    plt.plot(t, F_fit, label='Multiexp fit',hold=True)
    plt.xlim([0,300])
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
    eta_subax.set_xlim(0,50)
    
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
    
plt.savefig(figure_path + 'single_expofits_gamma_large_tau.png', dpi=120)
plt.close(fig)


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
    
plt.savefig(figure_path + 'single_expofits_eta_large_tau.png', dpi=120)
plt.close(fig)

