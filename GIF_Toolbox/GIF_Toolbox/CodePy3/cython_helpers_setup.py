# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:03:45 2015

@author: andrej
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules=[Extension("cython_helpers",sources=["cython_helpers.pyx"],include_dirs=[np.get_include()])]

setup(
  name = "Cython_helpers",
  ext_modules = cythonize(ext_modules)
)