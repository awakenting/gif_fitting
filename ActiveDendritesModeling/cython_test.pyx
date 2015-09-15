# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:01:20 2015

@author: andrej
"""

def say_hello_to(name):
    print("Hello %s!" % name)
    
cdef double f(double x) except? -2:
    return x**2
    
def integrate_f(double a, double b, int N):
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b-a)/N
    for i in range(N):
        s += f(a+i*dx)
    return s * dx