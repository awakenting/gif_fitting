# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:03:45 2015

@author: andrej
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Hello world app',
  ext_modules = cythonize("cython_test1.pyx"),
)
