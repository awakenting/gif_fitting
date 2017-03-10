# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 20:14:13 2016

@author: andrej
"""

import io
import os
import requests
import zipfile
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

url = 'http://microcircuits.epfl.ch/data/uncurated/sm100415a1_folder.zip'
request = requests.get(url)
file = zipfile.ZipFile(io.BytesIO(request.content))

#%% 
my_dir = './'

if not os.path.exists(my_dir):
    os.makedirs(my_dir)
 
file.extractall(path=my_dir)   

#%%
from time import sleep # this should go at the top of the file
from bs4 import BeautifulSoup
import re
import copy

driver = webdriver.Firefox()
base_url = 'http://microcircuits.epfl.ch/#/uncurated/'
driver.get(base_url)

old_height = 0
new_height = 1
while new_height > old_height:
    old_height = copy.copy(new_height)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep(0.05)
    new_height = driver.execute_script("return document.body.scrollHeight")

html = driver.execute_script("return document.getElementsByTagName('table')[0].innerHTML")
soup = BeautifulSoup(html)

links = [a.attrs['href'] for a in soup.find_all(href=re.compile(".zip$"))]
