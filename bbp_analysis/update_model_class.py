# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:53:57 2016

@author: andrej
"""

import os

from fitgif.GIF import GIF
from fitgif.GIF_subth_adapt_constrained import GIF_subadapt_constrained

root_path = './article_4_data/grouped_ephys/'
animal_dirs = sorted(os.listdir(root_path))

subadapt_model_path = './results/models/subadapt_constrained/'
basic_model_path = './results/models/basic/'
expm_path = './results/experiments/'
    
#%% load fitted models
model_name = '_tau_w_50ms'
subadapt_gifs =[]
basic_gifs = []
for animalnr in range(len(animal_dirs)):
    if os.path.exists(subadapt_model_path + 'Animal_' + animal_dirs[animalnr]):
        subadapt_gifs.append(GIF_subadapt_constrained.load(subadapt_model_path + 'Animal_' + animal_dirs[animalnr] + model_name))
    if os.path.exists(basic_model_path + 'Animal_' + animal_dirs[animalnr]):
        basic_gifs.append(GIF.load(basic_model_path + 'Animal_' + animal_dirs[animalnr]))
        
#%% create new objects with updated class and overwrite with parameters from saved model
        
for gif in basic_gifs:
    new_gif = GIF(gif.dt)

    new_gif.gl      = gif.gl
    new_gif.C       = gif.C
    new_gif.El      = gif.El
    new_gif.Vr      = gif.Vr
    new_gif.Tref    = gif.Tref
    
    new_gif.Vt_star = gif.Vt_star
    new_gif.DV      = gif.DV
    new_gif.lambda0 = gif.lambda0
    
    new_gif.eta     = gif.eta
    new_gif.gamma   = gif.gamma
    
    new_gif.avg_spike_shape = gif.avg_spike_shape
    new_gif.avg_spike_shape_support = gif.avg_spike_shape_support

    new_gif.expm_file      = gif.expm_file
    new_gif.pred           = gif.pred
    
    new_gif.var_explained  = gif.var_explained
    new_gif.mean_se = gif.mean_se
    
    new_gif.save_path = gif.save_path
    new_gif.save(gif.save_path)
    
for gif in subadapt_gifs:
    new_gif = GIF_subadapt_constrained(gif.dt)

    new_gif.gl      = gif.gl
    new_gif.C       = gif.C
    new_gif.El      = gif.El
    new_gif.Vr      = gif.Vr
    new_gif.Tref    = gif.Tref
    
    new_gif.Ew      = gif.Ew
    new_gif.a_w     = gif.a_w
    new_gif.tau_w_opt = gif.tau_w_opt
    
    new_gif.tau_w_values   = gif.tau_w_values
    new_gif.tau_w_scores   = gif.tau_w_scores
    
    new_gif.Vt_star = gif.Vt_star
    new_gif.DV      = gif.DV
    new_gif.lambda0 = gif.lambda0
    
    new_gif.eta     = gif.eta
    new_gif.gamma   = gif.gamma
    
    new_gif.avg_spike_shape = gif.avg_spike_shape
    new_gif.avg_spike_shape_support = gif.avg_spike_shape_support

    new_gif.expm_file      = gif.expm_file
    new_gif.pred           = gif.pred
    
    new_gif.save_path = gif.save_path
    new_gif.save(gif.save_path)
    
    
    
    
    