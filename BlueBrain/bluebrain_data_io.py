# coding: utf-8

import os
import urllib
import numpy as np
import pickle

from Experiment import Experiment

ROOT_PATH = './full_dataset/article_4_data/grouped_ephys'
ZIPFILE_PATH = './full_dataset/article_4_data'
EXPM_PATH = './results/experiments/'
URL = 'http://microcircuits.epfl.ch/data/released_data/'

if not os.path.exists(EXPM_PATH):
    os.makedirs(EXPM_PATH)

if not os.path.exists(ROOT_PATH):
    print('It seems that the directory of the raw data does not exist. It is expected to be at: ' + ROOT_PATH)

if not os.path.exists(ROOT_PATH):
    print('It seems that the directory with the zip files does not exist. It is expected to be at: ' + ZIPFILE_PATH)


# ==============================================================================
# General io function
# ==============================================================================

def download_info_from_url(url):
    """
    Download content from url and return it.
    """
    r = urllib.request.urlopen(url)
    data = r.read()
    data = data.decode(encoding='UTF-8')

    return data


def get_genetic_cell_infos(filepath):
    """
    Downloads genetic information from cells in the directory at filepath.
    """
    filelist = os.listdir(filepath)
    raw_names = [name[0:-4] for name in filelist]

    cell_names = []
    for name in raw_names:
        # if name.rfind('ET') == -1:
        cell_names.append(name)

    infos = {}
    for cell in cell_names:
        url_complete = URL + cell + '.txt'
        try:
            infos[cell] = download_info_from_url(url_complete)
        except Exception:
            next
    return infos


def save_filtered_cell_infos(filtername, criterion1='SOM:1', criterion2='PV:0', criterion3='VIP:0'):
    """
    Gets genetic information from all cells in ZIPFILE_PATH directory, filters them by the given
    criterions and saves the filtered list with pickle.
    """
    infos = get_genetic_cell_infos(ZIPFILE_PATH)

    desired_cells = {}
    for cell in infos.keys():
        if criterion1 in infos[cell] and criterion2 in infos[cell] and criterion3 in infos[cell]:
            desired_cells[cell] = infos[cell]

    with open(filtername + '_infos.pkl', 'wb') as f:
        pickle.dump(desired_cells, f)


def save_all_cell_infos(filepath):
    """
    Saves genetic information from all cells in ZIPFILE_PATH directory in one list with pickle.
    """
    infos = get_genetic_cell_infos(filepath)
    with open('cell_infos_full.pkl', 'wb') as f:
        pickle.dump(infos, f)


def open_filtered_cell_info_list(filtername):
    """
    Opens the list that was saved with save_filtered_cell_infos with the given filtername.
    """
    with open(filtername + '_infos.pkl', 'rb') as f:
        filtered_list = pickle.load(f)
    return filtered_list


def create_experiments_from_list(cells, cell_type, verbose=True):
    """
    Creates Experiment objects for cells in cells, adds all existing traces and saves them.
    
    Params:
        - cells: List with cell names or dictionairy where the keys are the cell names.
        
    """
    if type(cells) is dict:
        cell_names = list(cells.keys())
    else:
        cell_names = cells
    ncells = len(cell_names)

    for i in range(ncells):
        PATH = os.path.join(ROOT_PATH, cell_names[i])
        animal_files = sorted(os.listdir(PATH))
        ntraces = int(len(animal_files) / 2)

        current_exp = Experiment('Cell_' + cell_names[i] + '_single_traces', cell_type=cell_type)
        exp_merged_traces = Experiment('Cell_' + cell_names[i] + '_merged_idrest_traces', cell_type=cell_type)

        nincluded_idrest_traces = 0
        for j in np.arange(ntraces):
            # files end with 'recordingType_recordingNumber.ibw'
            file_split = str.split(animal_files[j][0:-4], '_')
            file_identifier = file_split[-2] + '_' + file_split[-1] + '.ibw'

            current_recording_type = file_split[-2]

            # find indeces of matching files in folder (current file always comes first because it's always Ch0)
            file_idc = [i for i, elem in enumerate(animal_files) if file_identifier in elem]
            current_file = animal_files[file_idc[0]]
            voltage_file = animal_files[file_idc[1]]

            current_exp.add_trainingset_trace(os.path.join(PATH, voltage_file), 10 ** -3,
                                              os.path.join(PATH, current_file), 10 ** -12, FILETYPE='Igor',
                                              verbose=verbose)
            tr = current_exp.trainingset_traces[j]
            tr.recording_type = current_recording_type
            tr.estimate_input_amp()

            if current_recording_type == 'IDRest':
                exp_merged_traces.add_trainingset_trace(os.path.join(PATH, voltage_file), 10 ** -3,
                                                        os.path.join(PATH, current_file), 10 ** -12, FILETYPE='Igor',
                                                        verbose=verbose)
                tr = current_exp.trainingset_traces[nincluded_idrest_traces]
                tr.recording_type = current_recording_type
                tr.estimate_input_amp()
                nincluded_idrest_traces += 1

        if not len(exp_merged_traces.trainingset_traces) < 3:
            exp_merged_traces.mergeTrainingTraces()
            exp_merged_traces.save(os.path.join(EXPM_PATH), verbose=verbose)

        current_exp.save(os.path.join(EXPM_PATH), verbose=verbose)


def load_merged_traces_experiments_from_list(cells, verbose=True):
    """
    Load experiments where IDRest traces have been merged.
    This function will try to load an experiment with merged IDRest traces for all cells
    in the list and just skip the ones for which it is not found. If no experiments were
    found, None is returned.
    
    Params:
        - cells: List with cell names or dictionairy where the keys are the cell names.
    
    See also:
    load_single_traces_experiments_from_list()
    """
    if type(cells) is dict:
        cell_names = list(cells.keys())
    else:
        cell_names = cells

    expms = []

    for i in range(len(cell_names)):
        current_expm_name = 'Experiment_Cell_' + cell_names[i] + '_merged_idrest_traces.pkl'
        current_expm_path = os.path.join(EXPM_PATH, current_expm_name)
        try:
            current_expm = Experiment.load(current_expm_path, verbose=verbose)
            expms.append(current_expm)
        except:
            pass

    if not len(expms) == 0:
        return expms
    else:
        return None


def load_single_traces_experiments_from_list(cells, verbose=True):
    """
    Load experiments where traces have been added separately.
    
    Params:
        - cells: List with cell names or dictionairy where the keys are the cell names.
        
    See also:
    load_merged_traces_experiments_from_list()
    """
    if type(cells) is dict:
        cell_names = list(cells.keys())
    else:
        cell_names = cells

    expms = []

    for i in range(len(cell_names)):
        current_expm_name = 'Experiment_Cell_' + cell_names[i] + '_single_traces.pkl'
        current_expm_path = os.path.join(EXPM_PATH, current_expm_name)
        try:
            current_expm = Experiment.load(current_expm_path, verbose=verbose)
            expms.append(current_expm)
        except:
            pass

    if not len(expms) == 0:
        return expms
    else:
        return None


# ==============================================================================
# From here on it's interneuron-specific functions
# ==============================================================================

def create_interneuron_specific_experiments(verbose=True):
    """
    Filters cell infos for SOM, PV and VIP neurons, loads them and creates
    Experiment objects.
    """
    # create and save filtered info lists for SOM, PV and VIP neurons
    save_filtered_cell_infos('som_cells', criterion1='SOM:1', criterion2='PV:0', criterion3='VIP:0')
    save_filtered_cell_infos('pv_cells', criterion1='SOM:0', criterion2='PV:1', criterion3='VIP:0')
    save_filtered_cell_infos('vip_cells', criterion1='SOM:0', criterion2='PV:0', criterion3='VIP:1')

    # get saved lists
    som_dict = open_filtered_cell_info_list('som_cells')
    vip_dict = open_filtered_cell_info_list('vip_cells')
    pv_dict = open_filtered_cell_info_list('pv_cells')

    # create experiment objects
    create_experiments_from_list(vip_dict, cell_type='vip', verbose=verbose)
    create_experiments_from_list(som_dict, cell_type='som', verbose=verbose)
    create_experiments_from_list(pv_dict, cell_type='pv', verbose=verbose)


def get_som_expms(merged=False, verbose=True):
    som_dict = open_filtered_cell_info_list('som_cells')
    if merged:
        return load_merged_traces_experiments_from_list(som_dict, verbose=verbose)
    else:
        return load_single_traces_experiments_from_list(som_dict, verbose=verbose)


def get_pv_expms(merged=False, verbose=True):
    pv_dict = open_filtered_cell_info_list('pv_cells')
    if merged:
        return load_merged_traces_experiments_from_list(pv_dict, verbose=verbose)
    else:
        return load_single_traces_experiments_from_list(pv_dict, verbose=verbose)


def get_vip_expms(merged=False, verbose=True):
    vip_dict = open_filtered_cell_info_list('vip_cells')
    if merged:
        return load_merged_traces_experiments_from_list(vip_dict, verbose=verbose)
    else:
        return load_single_traces_experiments_from_list(vip_dict, verbose=verbose)
