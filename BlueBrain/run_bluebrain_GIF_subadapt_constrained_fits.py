# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:23:11 2016

@author: andrej
"""
import os
import fit_bluebrain_subadapt_constrained as fit_bluebrain_subadapt_constrained
from SpikeTrainComparator import SpikeTrainComparator
from GIF_subth_adapt_constrained import GIF_subadapt_constrained
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.facecolor'] = 'white'

import numpy as np
from scipy.signal import fftconvolve

unwanted_sessions = ['APThreshold', 'APWaveform']
root_path = './article_4_data/grouped_ephys/'
figure_path = './results/figures/fits/subadapt_constrained_tau_w_50ms/'
model_path = './results/models/subadapt_constrained/'
expm_path = './results/experiments/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(expm_path):
    os.makedirs(expm_path)

#%%
num_of_animals = 32
md_values = np.zeros(num_of_animals)
gifs, expms, predicts = [], [], []

do_ensemble_plot = True
do_single_plot = True
save_gifs = True

for nr in range(num_of_animals):
#for nr in [4]:
    fitted = fit_bluebrain_subadapt_constrained.run(nr, root_path, unwanted_sessions)
    if not fitted:
        continue
    else:
        (gif,expm,md,predict) = fitted
        
        expm.save(expm_path)
        gif.expm_file = expm.save_path
        gif.pred = predict
    
    if save_gifs:
        gif.save_path = model_path + str(expm.name) + '_tau_w_50ms'        
        gif.save(gif.save_path)
            
    #md_values[nr] = md
    
    gifs.append(gif)
    expms.append(expm)
    predicts.append(predict)
    
#==============================================================================
#  From here on it's only plotting the results
#==============================================================================

    if do_single_plot:
        tr = expm.trainingset_traces[0]
        
        
        fig = plt.figure(figsize=(28,24), facecolor='white')
        subrows = 6
        subcols = 3    
        gs = gridspec.GridSpec(subrows, subcols, left=0.05, right=0.95, bottom=0.05, top=0.9)
        gs.update(hspace=0.5, wspace=0.3)
    
        ax1 = plt.subplot(gs[0:2, :-1])
        ax2 = plt.subplot(gs[2:4, :-1])
        ax3 = plt.subplot(gs[0:2, -1])
        ax4 = plt.subplot(gs[2:4, -1])
        ax5 = plt.subplot(gs[4:6, -1])
        ax6 = plt.subplot(gs[4, :-1])
        ax7 = plt.subplot(gs[5, :-1])
        
    #==============================================================================
    #    Plot input current 
    #==============================================================================
        
        ax1.plot(tr.getTime(), tr.I, 'gray')
    
        # Plot ROI
        ROI_vector = -10.0*np.ones(int(tr.T/tr.dt)) 
        if tr.useTrace :
            ROI_vector[tr.getROI()] = 10.0
        
        ax1.fill_between(tr.getTime(), ROI_vector, 10.0, color='0.2')
        
        ax1.set_xlim(0,tr.getTime()[-1])
        ax1.set_ylim([min(tr.I), max(tr.I)])
        ax1.set_ylabel("I (nA)")
        ax1.set_xticks([])
        ax1.set_title("Input current (dark region not selected)")
        
    #==============================================================================
    #    Plot membrange potential 
    #==============================================================================   
        ax2.plot(tr.getTime(), tr.V_rec, 'black')    
        
        if tr.AEC_flag :
            ax2.plot(tr.getTime(), tr.V, 'blue')    
            
            
        if tr.spks_flag :
            ax2.plot(tr.getSpikeTimes(), np.zeros(tr.getSpikeNb()), '.', color='red')
        
        # Plot ROI
        ROI_vector = -100.0*np.ones(int(tr.T/tr.dt)) 
        if tr.useTrace :
            ROI_vector[tr.getROI()] = 100.0
        
        ax2.fill_between(tr.getTime(), ROI_vector, 100.0, color='0.2')
        
        ax2.set_xlim(0,tr.getTime()[-1])
        ax2.set_ylim([min(tr.V)-5.0, max(tr.V)+5.0])
        ax2.set_ylabel("Voltage (mV)")   
        ax2.set_title('Membrane potential')
        
    #==============================================================================
    #    Plot fitted kernels 
    #==============================================================================
    
        # Plot kappa
        
        K_support = np.linspace(0,150.0, 300)             
        K = 1./gif.C*np.exp(-K_support/(gif.C/gif.gl)) 
            
        ax3.plot(K_support, K, color='red', lw=2)
        ax3.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        ax3.set_xlim([K_support[0], K_support[-1]])    
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Membrane filter (MOhm/ms)")
        ax3.set_title('Tau for kappa : ' + str(gif.C/gif.gl))
        
        # inset with only the first 100 ms of the kernel 
        
        box = ax3.get_position()
        width = box.width
        height = box.height
        inax_position  = ax3.transAxes.transform([0.4,0.4])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= 0.5
        height *= 0.5  # <= Typo was here
        subax = fig.add_axes([x,y,width,height],axisbg='w')
        
        subax.plot(K_support, K, color='red', lw=2)
        subax.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2)
        subax.set_xlim(0,100)
        
        # Plot eta
        
        (eta_support, eta) = gif.eta.getInterpolatedFilter(gif.dt) 
        
        ax4.plot(eta_support, eta, color='red', lw=2)
        ax4.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        ax4.set_xlim([eta_support[0], eta_support[-1]])    
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("Eta (nA)")
        
        # inset with only the first 100 ms of the kernel 
        
        box = ax4.get_position()
        width = box.width
        height = box.height
        inax_position  = ax4.transAxes.transform([0.4,0.4])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= 0.5
        height *= 0.5  # <= Typo was here
        subax = fig.add_axes([x,y,width,height],axisbg='w')
        
        subax.plot(eta_support, eta, color='red', lw=2)
        subax.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2)
        subax.set_xlim(0,100)
    
        # Plot gamma
        
        (gamma_support, gamma) = gif.gamma.getInterpolatedFilter(gif.dt) 
        
        ax5.plot(gamma_support, gamma, color='red', lw=2)
        ax5.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        ax5.set_xlim([gamma_support[0], gamma_support[-1]])    
        ax5.set_xlabel("Time (ms)")
        ax5.set_ylabel("Gamma (mV)")
        
        # inset with only the first 100 ms of the kernel 
        
        box = ax5.get_position()
        width = box.width
        height = box.height
        inax_position  = ax5.transAxes.transform([0.4,0.4])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= 0.5
        height *= 0.5  # <= Typo was here
        subax = fig.add_axes([x,y,width,height],axisbg='w')
        
        subax.plot(gamma_support, gamma, color='red', lw=2)
        subax.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2)
        subax.set_xlim(0,100)
        
    #==============================================================================
    #    Plot spike raster and psth
    #==============================================================================
        
        # Plot raster
        dt = gif.dt
        delta = 500 # in time steps
        
        nb_rep = min(len(predict.spks_data), len(predict.spks_model) )
        
        cnt = 0
        for spks in predict.spks_data[:nb_rep] :
            cnt -= 1      
            ax6.plot(spks, cnt*np.ones(len(spks)), '|', color='black', ms=5, mew=2)
    
        for spks in predict.spks_model[:nb_rep] :
            cnt -= 1      
            ax6.plot(spks, cnt*np.ones(len(spks)), '|', color='red', ms=5, mew=2)
          
        ax6.set_xlim(0,tr.getTime()[-1])
        ax6.set_yticks([])
        ax6.set_title('Spikes - raster plot, Md* = %0.2f' % (md))
        
        # Plot PSTH
        rect_width  = delta
        rect_size_i = int(float(rect_width)/dt)
        rect_window = np.ones(rect_size_i)/(rect_width/1000.0)
    
        spks_avg_data         = SpikeTrainComparator.getAverageSpikeTrain(predict.spks_data, predict.T, dt)
        spks_avg_data_support = np.arange(len(spks_avg_data))*dt
        spks_avg_data_smooth  = fftconvolve(spks_avg_data, rect_window, mode='same')
           
        spks_avg_model = SpikeTrainComparator.getAverageSpikeTrain(predict.spks_model, predict.T, dt)
        spks_avg_model_support = np.arange(len(spks_avg_data))*dt             
        spks_avg_model_smooth  = fftconvolve(spks_avg_model, rect_window, mode='same')        
        
        ax7.plot(spks_avg_data_support, spks_avg_data_smooth, 'black', label='Data')
        ax7.plot(spks_avg_model_support, spks_avg_model_smooth, 'red', label='Model')
        ax7.set_xlim(0,tr.getTime()[-1])
        #ax7.set_xlim(0,predict.T)
        ax7.legend()
        
        # Compute % of variance explained
        SSE = np.mean( (spks_avg_data_smooth-spks_avg_model_smooth)**2 )
        VAR = np.var(spks_avg_data_smooth)
        pct_variance_explained = (1.0 - SSE/VAR)*100.0  
        
        ax7.set_xlabel("Time (ms)")
        ax7.set_ylabel('PSTH (Hz)')
        ax7.set_title("PSTH (Percentage of variance explained: %0.1f)" % (pct_variance_explained))
        
      
        plt.suptitle(expm.name + ' - fitting result', fontsize = 20)
        plt.show()
        
    #==============================================================================
    #  Save figure
    #==============================================================================
        
        if not os.path.exists(figure_path):
            os.mkdir(figure_path)
            
        plt.savefig(figure_path + str(expm.name) + '_fit_subadapt_constrained.png', dpi=120)
        plt.close(fig)


if do_ensemble_plot:
    
    #==============================================================================
    #  Create ensemble plots with all the fitted kernels of one type in one plot
    #==============================================================================
    plt.style.use('ggplot')
    mpl.rcParams['axes.facecolor'] = 'white'
    
    # kappa 
    
    fig = plt.figure(figsize = (10,10))    
    for gif in gifs:
        K_support = np.linspace(0,150.0, 300)             
        K = 1./gif.C*np.exp(-K_support/(gif.C/gif.gl)) 
            
        plt.plot(K_support, K, lw=2, hold=True)
        
    plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2)
    
    plt.xlim(0,K_support[-1])
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane filter (MOhm/ms)')
    plt.title('Membrane filter kappa for all fits')
    
    plt.savefig(figure_path + 'kappas_fit.png', dpi=120)
    plt.close(fig)
    
    
    # eta
    
    fig = plt.figure(figsize = (10,10))    
    for gif in gifs:
        (eta_support, eta) = gif.eta.getInterpolatedFilter(gif.dt) 
        
        plt.plot(eta_support, eta, lw=2)
        
    plt.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2)
    
    plt.xlim(0,100)
    plt.xlabel('Time [ms]')
    plt.ylabel('Eta [nA]')
    plt.title('Eta for all fits')
    
    plt.savefig(figure_path + 'etas_fit.png', dpi=120)
    plt.close(fig)
    
    
    # gamma
    
    fig = plt.figure(figsize = (10,10))    
    for gif in gifs:
        (gamma_support, gamma) = gif.gamma.getInterpolatedFilter(gif.dt) 
        
        plt.plot(gamma_support, gamma, lw=2)
        
    plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2)
    
    plt.xlim(0,100)
    plt.xlabel('Time [ms]')
    plt.ylabel('Gamma [mV]')
    plt.title('Gamma for all fits')
        
    plt.savefig(figure_path + 'gammas_fit.png', dpi=120)
    plt.close(fig)





























    