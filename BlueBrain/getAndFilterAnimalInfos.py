# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:40:18 2015

@author: andrej
"""

import os
import urllib
import json

files = os.listdir('./full_dataset/article_4_data')

raw_names = [name[0:-4] for name in files]

animal_names = []
for name in raw_names:
    if name.rfind('ET') == -1:
        animal_names.append(name)

def download_from_url(url):
    """Download url and return the content."""
    r = urllib.request.urlopen(url)
    data = r.read()
    data = data.decode(encoding='UTF-8')

    return data
    
#url = 'http://microcircuits.epfl.ch/data/released_data/'
#data = download_from_url(url)
#print(data)

#%% get animal infos
url = 'http://microcircuits.epfl.ch/data/released_data/'
infos = {}
for animal in animal_names:
    url_complete = url + animal + '.txt'
    try:
        infos[animal] = download_from_url(url_complete)
    except Exception:
        next
    
#infos

#%% filter infos and save
desired_animals = {}
criterion1 = 'SOM:1'
criterion2 = 'PV:0'
criterion3 = 'VIP:0'
for animal in infos.keys():
    if criterion1 in infos[animal] and criterion2 in infos[animal] and criterion3 in infos[animal]:
        desired_animals[animal] = infos[animal]

infofile = open('/home/andrej/Dropbox/Arbeit/MKP/gif_fitting/BlueBrain/animal_infos','w')
json.dump(desired_animals,infofile)
infofile.close()

full_infofile = open('/home/andrej/Dropbox/Arbeit/MKP/gif_fitting/BlueBrain/animal_infos_full','w')
json.dump(infos,full_infofile)
full_infofile.close()
#%% open it again with the following command in the terminal

    
# python -m json.tool animal_infos
