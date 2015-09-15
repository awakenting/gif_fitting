# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:50:28 2015

@author: andrej
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize, rosen, OptimizeResult

x0 = [1.3, 4]
res = minimize(rosen, x0, method='Nelder-Mead')
print(res.x)