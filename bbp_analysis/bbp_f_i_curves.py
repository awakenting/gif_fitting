
# coding: utf-8

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle

plt.style.use('ggplot')
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.facecolor'] = 'white'

# load data
with open('fi_curve_data.pkl','rb') as f:
    loaded_data = pickle.load(f)


# plot data

plt.figure()
for cell in loaded_data['current_amplitudes'].keys():
    amps = loaded_data['current_amplitudes'][cell]
    rates = loaded_data['firing_rates'][cell]
    plt.plot(amps, rates, '.', ms=12, label=cell)
plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1)
plt.xlabel('Input current [pA]')
plt.ylabel('Firing rate [Hz]')
plt.show()

