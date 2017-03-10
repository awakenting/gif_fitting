# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 22:21:00 2016

@author: andrej
"""
import numpy as np
import matplotlib.pyplot as plt
import ReadIBW


file = '/home/andrej/Dropbox/Arbeit/MKP/bluebrain_data/Volumes/experiment/electrophysiology/'\
        'Shruti_Interneurons/sm100415a1 Folder/x00_FirePattern_ch4_102.ibw'
        
data = ReadIBW.read(file)
plt.plot(data)
plt.ylim([-0.2,0.2])
