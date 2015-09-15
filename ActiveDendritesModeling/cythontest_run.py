# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:09:17 2015

@author: andrej
"""

# pyximport is the easy way to compile the pyx file but if you want to configure
# how it is compiled use the setup.py method
#import pyximport; pyximport.install()
import cython_test1 as cy
import matplotlib.pyplot as plt


#say_hello_to('mirko')

a = cy.integrate_f(0,2,1000)
#b = f(10)
