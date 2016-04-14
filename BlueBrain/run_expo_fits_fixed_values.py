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
model_name = '_tau_w_50ms'
gifs =[]
for animalnr in range(len(animal_dirs)):
    if os.path.exists(model_path + 'Animal_' + animal_dirs[animalnr]):
        gifs.append(GIF_subadapt_constrained.load(model_path + 'Animal_' + animal_dirs[animalnr] + model_name))

#%% function for fits with fixed values of time constants

def generate_fixed_tau_expofits(p_eta_taus, p_gamma_taus, gifs):
    
    for gifnr,gif in enumerate(gifs):
        print('Fitting exponentials for gif ' + str(gifnr+1) + ' of ' + str(len(gifs)), end='\r', flush=True)
        for filter_name,taus in zip(['gamma', 'eta'],[p_gamma_taus,p_eta_taus]):
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
            (tau_set_opt, opt_dim) = filter_obj.fit_sumOfExpos_optimize_dim(maxdim=len(taus), bs=b_inits,
                                                 tausets=tau_inits, ROI=[0,300], dt=gif.dt, fixed_taus=True)
                                                 
            setattr(gif,filter_name,filter_obj)
        
        gif.save(gif.save_path)
    # collect fitted parameter values
    
    eta_amps = np.zeros((len(gifs),3))
    eta_taus = np.zeros((len(gifs),3))
    eta_dims = np.zeros((len(gifs),1))
    
    gamma_amps = np.zeros((len(gifs),3))
    gamma_taus = np.zeros((len(gifs),3))
    gamma_dims = np.zeros((len(gifs),1))
    
    leak_amps = np.zeros((len(gifs),1))
    leak_taus = np.zeros((len(gifs),1))
    
    a_ws = np.zeros((len(gifs),1))
    w_taus = np.zeros((len(gifs),1))
    
    
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
        
        leak_amps[gifnr] = 1/gif.C
        leak_taus[gifnr] = gif.C/gif.gl
    
        a_ws[gifnr] = gif.a_w
        w_taus[gifnr] = gif.tau_w_opt    
        
    # plot histograms    
    
    fig = plt.figure(figsize=(28,20), facecolor='white')
    subrows = 4
    subcols = 4    
    gs = gridspec.GridSpec(subrows, subcols, left=0.1, right=0.90, bottom=0.1, top=0.9)
    gs.update(hspace=0.5, wspace=0.4)
    
    ax1  = plt.subplot(gs[0  ,0])
    ax2  = plt.subplot(gs[1  ,0])
    ax3  = plt.subplot(gs[2  ,0])
    ax4  = plt.subplot(gs[3  ,0])
    ax5  = plt.subplot(gs[0  ,1])
    ax6  = plt.subplot(gs[1  ,1])
    ax7  = plt.subplot(gs[2  ,1])
    ax8  = plt.subplot(gs[3  ,1])
    ax9  = plt.subplot(gs[0  ,2])
    ax10 = plt.subplot(gs[1  ,2])
    ax11 = plt.subplot(gs[0  ,3])
    ax12 = plt.subplot(gs[1  ,3])
    ax13 = plt.subplot(gs[2  ,2])
    
    #==============================================================================
    # Eta plots
    #==============================================================================
    ax1.hist(eta_amps[np.flatnonzero(eta_amps[:,0]),0], bins=20)
    ax1.set_xlabel('Amplitude [nA]')
    ax1.set_ylabel('# models')
    ax1.set_title('Eta\n\n  Amplitudes for tau = ' + str(p_eta_taus[0]) + 'ms')
    ax1.set_xlim([-0.2,3])
    
    ax2.hist(eta_amps[np.flatnonzero(eta_amps[:,1]),1], bins=10)
    ax2.set_xlabel('Amplitude [nA]')
    ax2.set_ylabel('# models')
    ax2.set_title('Amplitudes for tau = ' + str(p_eta_taus[1]) + 'ms')
    ax2.set_xlim([-0.2,3])
    
    ax3.hist(eta_amps[np.flatnonzero(eta_amps[:,2]),2], bins=10)
    ax3.set_xlabel('Amplitude [nA]')
    ax3.set_ylabel('# models')
    ax3.set_title('Amplitudes for tau = ' + str(p_eta_taus[2]) + 'ms')
    ax3.set_xlim([-0.2,3])
    
    ax4.hist(eta_dims)
    ax4.set_xlabel('# exponentials')
    ax4.set_ylabel('# models')
    ax4.set_xticks([1,2,3])
    ax4.set_xticklabels([1,2,3])
    ax4.set_title('Number of exponentials used for fit')
    
    
    #==============================================================================
    # Gamma plots
    #==============================================================================
    gam_color = next(ax1._get_lines.prop_cycler)['color']
    
    ax5.hist(gamma_amps[np.flatnonzero(gamma_amps[:,0]),0], bins=20, color=gam_color)
    ax5.set_xlabel('Amplitude [mV]')
    ax5.set_ylabel('# models')
    ax5.set_title('Gamma\n\n  Amplitudes for tau = ' + str(p_gamma_taus[0]) + 'ms')
    ax5.set_xlim([-100,600])
    
    ax6.hist(gamma_amps[np.flatnonzero(gamma_amps[:,1]),1], bins=10, color=gam_color)
    ax6.set_xlabel('Amplitude [mV]')
    ax6.set_ylabel('# models')
    ax6.set_title('Amplitudes for tau = ' + str(p_gamma_taus[1]) + 'ms')
    ax6.set_xlim([-100,600])
    
    ax7.hist(gamma_amps[np.flatnonzero(gamma_amps[:,2]),2], bins=10, color=gam_color)
    ax7.set_xlabel('Amplitude [mV]')
    ax7.set_ylabel('# models')
    ax7.set_title('Amplitudes for tau = ' + str(p_gamma_taus[2]) + 'ms')
    ax7.set_xlim([-100,600])
    
    ax8.hist(gamma_dims, color=gam_color)
    ax8.set_xlabel('# exponentials')
    ax8.set_ylabel('# models')
    ax8.set_xticks([1,2,3])
    ax8.set_xticklabels([1,2,3])
    ax8.set_title('Number of exponentials used for fit')
    
    #==============================================================================
    # Leak current plots
    #==============================================================================
    leak_color = next(ax1._get_lines.prop_cycler)['color']
    
    ax9.hist(leak_amps, bins=20, color=leak_color)
    ax9.set_xlabel('Amplitude')
    ax9.set_ylabel('# models')
    ax9.set_title('Leak current\n\n  Amplitudes (1/C)')
    
    ax10.hist(leak_taus, bins=20, color=leak_color)
    ax10.set_xlabel('Time [ms]')
    ax10.set_ylabel('# models')
    ax10.set_title('Time constants (C/gl)')
    
    #==============================================================================
    # W current plots
    #==============================================================================
    w_color = next(ax1._get_lines.prop_cycler)['color']
    
    ax11.hist(a_ws, bins=20, color=w_color)
    ax11.plot([0.0,0.0],[0,5], ':', color='black')
    ax11.set_xlim([-0.02,0.02])
    ax11.set_xticks(np.arange(-0.01,0.021,0.01))
    ax11.set_xlabel('Amplitude')
    ax11.set_ylabel('# models')
    ax11.set_title('W current\n\n  Amplitudes')
    
    ax12.hist(w_taus, bins=20, color=w_color)
    ax12.set_xlabel('Time [ms]')
    ax12.set_ylabel('# models')
    ax12.set_title('Time constants')
    
    #==============================================================================
    # Other plots
    #==============================================================================
    n_color = next(ax1._get_lines.prop_cycler)['color']
    n_color = next(ax1._get_lines.prop_cycler)['color']
    
    ax13.plot(w_taus,leak_taus,'.',MarkerSize=12, color=n_color)
    ax13.set_xlabel('tau_w [ms]')
    ax13.set_ylabel('tau_leak [ms]')
    ax13.set_xlim([0, np.max(w_taus)*1.1])
    ax13.set_ylim([0, np.max(leak_taus)*1.1])
    ax13.set_title('tau_w vs. tau_leak')
    
    plt.suptitle('Summary of exponential fits',fontsize=24)
    
    #==============================================================================
    #  Save figure
    #==============================================================================
        
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
        
    plt.savefig(figure_path + model_name + '_expofit_fixed_taus_stats_with' + str(p_gamma_taus) + '_for_gamma.png', dpi=120)
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
        
    plt.savefig(figure_path + model_name + 'single_expofits_gamma_fixed_taus_with' + str(p_gamma_taus) + '_for_gamma.png', dpi=120)
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
        
    plt.savefig(figure_path + model_name + 'single_expofits_eta_fixed_taus_with' + str(p_gamma_taus) + '_for_gamma.png', dpi=120)
    plt.close(fig)

        
        
#%% run fits
my_eta_taus = [5,50,100]
my_gamma_taus = [10,50,150]

generate_fixed_tau_expofits(my_eta_taus,my_gamma_taus,gifs)

