# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:22:37 2016

@author: andrej
"""

import os
import zipfile

root_path = './full_dataset/article_4_data/'
data_path = './full_dataset/article_4_data/'

files = sorted(os.listdir(root_path))

for file in files:
    zip_file = zipfile.ZipFile(os.path.join(root_path,file))
    zip_file.extractall(path=data_path)  