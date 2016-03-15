# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:13:55 2016

@author: andrej
"""

from GIF_subth_adapt_constrained import GIF_subadapt_constrained
from GIF import GIF
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

color_cycle = [ccol for ccol in mpl.rcParams['axes.prop_cycle']]
my_colors =[ color_cycle[i]['color'] for i in range(len(color_cycle))]

root_path = './article_4_data/grouped_ephys/'
model_path = './results/models/subadapt_constrained/'
model2_path = './results/models/basic/'
figure_path = './results/figures/fits/subadapt_constrained/'
animal_dirs = sorted(os.listdir(root_path))

#%% load fitted models
gifs1 =[]
gifs2 =[]
for animalnr in range(len(animal_dirs)):
    if os.path.exists(model_path + 'Animal_' + animal_dirs[animalnr]):
        gifs1.append(GIF_subadapt_constrained.load(model_path + 'Animal_' + animal_dirs[animalnr] + '_large_tau_w'))
        gifs2.append(GIF.load(model2_path + 'Animal_' + animal_dirs[animalnr]))

#%% tau_w and a_w stats 
a_ws = np.zeros((len(gifs1),1))
w_taus = np.zeros((len(gifs1),1))
w_tau_scores = np.zeros((len(gifs1),len(gifs1[0].tau_w_scores)))
for gifnr,gif in enumerate(gifs1):
    a_ws[gifnr] = gif.a_w
    w_taus[gifnr] = gif.tau_w_opt
    w_tau_scores[gifnr,:] = gif.tau_w_scores.squeeze()*100

w_tau_low = w_tau_scores[w_taus.squeeze()<50, :]
w_tau_high = w_tau_scores[w_taus.squeeze()>50, :]

w_tau_scores_norm = w_tau_scores.T/np.max(w_tau_scores,axis=1)

w_tau_low_norm = w_tau_scores_norm[:, w_taus.squeeze()<50]
w_tau_high_norm = w_tau_scores_norm[:, w_taus.squeeze()>50]

fig = plt.figure(figsize=(24,20), facecolor='white')
subrows = 2
subcols = 2    
gs = gridspec.GridSpec(subrows, subcols, left=0.1, right=0.90, bottom=0.1, top=0.9)
gs.update(hspace=0.5, wspace=0.5)

x_ticks = np.arange(5,len(gifs1[0].tau_w_values),step=5)
x_ticklabels = gifs1[0].tau_w_values[4:-1:5]

plt.subplot(gs[0,0])
plt.plot(w_tau_low.T,color = my_colors[0], hold=True)
plt.plot(np.argmax(w_tau_low,axis=1),np.max(w_tau_low,axis=1),'*')
plt.xticks(x_ticks,x_ticklabels)
plt.ylim([0,100])
plt.xlabel('value of tau_w [ms]')
plt.ylabel('Variance explained of the fit')
plt.title('Low optimal tau_w values\n\nVariance explained')

plt.subplot(gs[0,1])
plt.plot(w_tau_high.T,color = my_colors[1], hold=True)
plt.plot(np.argmax(w_tau_high,axis=1),np.max(w_tau_high,axis=1),'*')
plt.xticks(x_ticks,x_ticklabels)
plt.ylim([0,100])
plt.xlabel('value of tau_w [ms]')
plt.ylabel('Variance explained of the fit')
plt.title('High optimal tau_w values\n\nVariance explained')

plt.subplot(gs[1,0])
plt.plot(w_tau_low_norm, color = my_colors[0], hold=True)
plt.plot(np.argmax(w_tau_low_norm,axis=0),np.max(w_tau_low_norm,axis=0),'*')
plt.ylim([0.95,1.01])
plt.xlabel('value of tau_w [ms]')
plt.xticks(x_ticks,x_ticklabels)
plt.ylabel('fit performance')
plt.title('Performance relative to best tau_w')

plt.subplot(gs[1,1])
plt.plot(w_tau_high_norm, color = my_colors[1], hold=True)
plt.plot(np.argmax(w_tau_high_norm,axis=0),np.max(w_tau_high_norm,axis=0),'*')
plt.ylim([0.95,1.01])
plt.xlabel('value of tau_w [ms]')
plt.xticks(x_ticks,x_ticklabels)
plt.ylabel('fit performance')
plt.title('Performance relative to best tau_w')

plt.savefig(figure_path + 'varExp_vs_tau_W.png', dpi=120)
plt.close(fig)

#%% 
basic_scores = np.zeros((len(gifs2),1))

for gifnr,gif in enumerate(gifs2):
    basic_scores[gifnr] = gif.var_explained*100
    
basic_scores = basic_scores.squeeze()
w_best_scores = np.max(w_tau_scores,axis=1)
w_improvs = (w_best_scores-basic_scores)/basic_scores*100

fig = plt.figure(figsize=(12,12))
plt.plot(w_taus,w_improvs,'.',MarkerSize=10)
plt.xlim([0,305])
plt.ylim([-1,40])
plt.xlabel('Value of tau_w [ms]')
plt.ylabel('Relative increase in varExplained [%]')
plt.savefig(figure_path + 'varExp_improv.png', dpi=120)
plt.close(fig)

fig = plt.figure(figsize=(24,20))
barwidth=0.2
plt.bar(np.arange(len(gifs1)),basic_scores, barwidth,color=my_colors[0],hold=True)
plt.bar(np.arange(len(gifs1))+barwidth, w_best_scores, barwidth, color=my_colors[1])
my_axes = fig.get_axes()[0]
for i in np.arange(len(w_taus)):
    my_axes.text(i+barwidth,w_best_scores[i]+2,str(w_taus[i]),ha='center',va='center')
plt.legend(['Basic GIF','With w'], loc='upper center')
plt.xlabel('Cell #')
plt.ylabel('Var explained [%]')
plt.savefig(figure_path + 'varExp_basic_vs_w.png', dpi=120)
plt.close(fig)
   
