from __future__ import print_function, division

import pickle
import matplotlib as plt

def load_fi_data(cell_type):
    with open('fi_curve_data_' + cell_type + '.pkl','rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def plot_all_fi_curves(fi_curve_data):
    i_values = fi_curve_data['current_amplitudes']
    f_values = fi_curve_data['firing_rates']
    cell_names = fi_curve_data['cell_names']
    plt.figure()
    for cell in cell_names:
        if not len(i_values[cell]) == 0:
            plt.plot(i_values[cell], f_values[cell], '.-', ms=12, label=cell)
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1)
    plt.xlabel('Input current [pA]')
    plt.ylabel('Firing rate [Hz]')
    plt.title(fi_curve_data['cell_type'])
    plt.show()
    
def plot_single_cell_fi_curve(fi_curve_data, cell_name):
    i_values = fi_curve_data['current_amplitudes']
    f_values = fi_curve_data['firing_rates']
    plt.figure()
    if not len(i_values[cell_name]) == 0:
        plt.plot(i_values[cell_name], f_values[cell_name], '.-', ms=12, label=cell_name)
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1)
    plt.xlabel('Input current [pA]')
    plt.ylabel('Firing rate [Hz]')
    plt.title(fi_curve_data['cell_type'])
    plt.show()
    
def show_all_cell_names(fi_curve_data):
    for cell in fi_curve_data['cell_names']:
        print(cell)
        
def show_parameters(fi_curve_data):
    print(fi_curve_data['parameters'])
        
# Example usage
'''
loaded_data = load_fi_data('vip')
plot_all_fi_curves(loaded_data)
plot_single_cell_fi_curve(loaded_data, 'Cell_A87_single_traces')
show_all_cell_names(loaded_data)
show_parameters(loaded_data)
'''