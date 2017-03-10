"""
Created on Thu Aug  4 13:43:25 2016

@author: andrej
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import ColorConverter as colcon
import xlsxwriter

from fitgif.GIF import GIF
from fitgif.GIF_subth_adapt_constrained import GIF_subadapt_constrained
from .Experiment_auto_read_T import Experiment_auto_read_T

plt.style.use('ggplot')
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.facecolor'] = 'white'
default_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]


unwanted_sessions = ['APThreshold', 'APWaveform']
root_path = './article_4_data/grouped_ephys/'
figure_path = './results/figures/fits/subadapt_constrained_tau_w_50ms/'
subadapt_model_path = './results/models/subadapt_constrained/'
basic_model_path = './results/models/basic/'
expm_path = './results/experiments/'
animal_dirs = sorted(os.listdir(root_path))
    
#%% load fitted models
model_name = '_tau_w_50ms'
subadapt_gifs =[]
basic_gifs = []

for animalnr in range(len(animal_dirs)):
    if os.path.exists(subadapt_model_path + 'Animal_' + animal_dirs[animalnr]):
        subadapt_gifs.append(GIF_subadapt_constrained.load(subadapt_model_path + 'Animal_' + animal_dirs[animalnr] + model_name))
    if os.path.exists(basic_model_path + 'Animal_' + animal_dirs[animalnr]):
        basic_gifs.append(GIF.load(basic_model_path + 'Animal_' + animal_dirs[animalnr]))
        

subadapt_vexp_dV = []
subadapt_vexp_V = []
subadapt_nspks = []
basic_vexp_dV = []
basic_vexp_V = []
basic_nspks = []

expms = []
expm_lengths = []

for basic_gif, subadapt_gif in zip(basic_gifs, subadapt_gifs):
    basic_var_explained_dV, basic_var_explained_V = basic_gif.get_var_explained()
    basic_vexp_dV.append(basic_var_explained_dV)
    basic_vexp_V.append(basic_var_explained_V)
    basic_nspks.append(len(basic_gif.pred.spks_data[0]))
    
    expms.append(Experiment_auto_read_T.load(basic_gif.expm_file))
    expm_lengths.append(expms[-1].trainingset_traces[0].T/1000)
    
    subadapt_var_explained_dV, subadapt_var_explained_V = subadapt_gif.get_var_explained()
    subadapt_vexp_dV.append(subadapt_var_explained_dV)
    subadapt_vexp_V.append(subadapt_var_explained_V)
    subadapt_nspks.append(len(subadapt_gif.pred.spks_data[0]))
    
basic_vexp_dV = np.array(basic_vexp_dV)
basic_vexp_V = np.array(basic_vexp_V)
basic_nspks = np.array(basic_nspks)

subadapt_vexp_dV = np.array(subadapt_vexp_dV)
subadapt_vexp_V = np.array(subadapt_vexp_V)    
subadapt_nspks = np.array(subadapt_nspks)

expm_lengths = np.array(expm_lengths)



#%% plotting

fig = plt.figure(figsize = (18,18))    
subrows = 2
subcols = 2    
gs = gridspec.GridSpec(subrows, subcols, left=0.1, right=0.90, bottom=0.1, top=0.9)
gs.update(hspace=0.5, wspace=0.4)


plt.subplot(gs[0,0])
for i in range(len(basic_vexp_V)):
    plt.plot([0,1],[basic_vexp_V[i]*100,subadapt_vexp_V[i]*100],lw=1,marker='.',color='k')

plt.xlim([-0.1,1.1])
plt.xticks([0,1],['Basic','Extended'])
plt.ylim([0,100])
plt.xlabel('Model')
plt.ylabel('Variance explained in percentage')
plt.title('Fit of V(t)')


plt.subplot(gs[0,1])
for i in range(len(basic_vexp_V)):
    plt.plot([0,1],[basic_vexp_dV[i]*100,subadapt_vexp_dV[i]*100],lw=1,marker='.',color='k')

plt.xlim([-0.1,1.1])
plt.xticks([0,1],['Basic','Extended'])
plt.ylim([0,100])
plt.xlabel('Model')
plt.title('Fit of dV(t)/dt')


plt.subplot(gs[1,0])
plt.plot(basic_nspks,basic_vexp_V*100,'o', label='basic')
plt.plot(basic_nspks,subadapt_vexp_V*100,'o', label='extended')

plt.ylim([0,100])
lgd1 = plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', numpoints=1)
plt.xlabel('Number of spikes in data')
plt.ylabel('Variance explained in percentage')
plt.title('Performance on V(t) vs #spikes')


plt.subplot(gs[1,1])
plt.plot(basic_nspks,basic_vexp_dV*100,'o', label='basic')
plt.plot(basic_nspks,subadapt_vexp_dV*100,'o', label='extended')

plt.ylim([0,100])
lgd2 = plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', numpoints=1)
plt.xlabel('Number of spikes in data')
plt.ylabel('Variance explained in percentage')
plt.title('Performance on dV(t)/dt vs #spikes')

plt.suptitle('Fit performance comparison')

plt.savefig(figure_path + 'varexp_fit_comparison.png',bbox_extra_artists=[lgd1,lgd2], 
            bbox_inches='tight', dpi=120)
plt.close(fig)


#%%
def get_darker_color(color):
    rgb = np.asarray(colcon.to_rgb(colcon,color))
    return rgb*0.6
    
fig = plt.figure(figsize=(24,16))
x = np.arange(len(basic_gifs))
width = 0.125
first_color = default_colors[0]
second_color = default_colors[1]

plt.bar(x - width*2, basic_vexp_V*100, width, color=first_color, label='basic model on V(t)')
plt.bar(x - width, subadapt_vexp_V*100, width, color=get_darker_color(first_color), label='extended model on V(t)')

plt.bar(x, basic_vexp_dV*100, width, color=second_color, label='basic model on dV/dt(t)')
plt.bar(x + width, subadapt_vexp_dV*100, width, color=get_darker_color(second_color), label='extended model on dV/dt(t)')

lgd = plt.legend(bbox_to_anchor=(1, 1.1), loc='upper left')
plt.ylabel('Variance explained [%]')
plt.xticks(x,[expm.name for expm in expms], rotation=60, ha='right')

plt.savefig(figure_path + 'varexp_fit_all.png',bbox_extra_artists=[lgd], 
            bbox_inches='tight', dpi=120)
plt.close(fig)
#%% 

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('./results/model_varExps.xlsx')
worksheet_basic_models = workbook.add_worksheet('Basic models')
worksheet_ext_models = workbook.add_worksheet('Extended models')


# Add a bold format to use to highlight cells.
bold = workbook.add_format({'bold': True})

# Write data headers.
for worksh in [worksheet_basic_models, worksheet_ext_models]:
    worksh.write('A1', 'Cell name', bold)
    worksh.write('B1', 'Var explained on V(t)', bold)
    worksh.write('C1', 'Var explained on dV/dt(t)', bold)

for worksheet, modellist in zip([worksheet_basic_models,worksheet_ext_models],[basic_gifs, subadapt_gifs]):
    for model_idx, model in enumerate(modellist):
        worksheet.write(model_idx+1, 0, model.save_path.split(sep='/')[-1])
        worksheet.write(model_idx+1, 1, model.var_explained_V)
        worksheet.write(model_idx+1, 2, model.var_explained_dV)

workbook.close()
