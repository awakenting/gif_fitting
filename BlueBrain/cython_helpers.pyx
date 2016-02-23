# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:33:41 2015

@author: andrej
"""
import numpy as np
import random
import sys
cimport numpy as np
DTYPE = np.double
FTYPE = np.float
ITYPE = np.int
ctypedef np.double_t DTYPE_t
ctypedef np.float_t FTYPE_t
ctypedef np.int_t ITYPE_t
from libc.math cimport exp
        
def c_detectSpikes(int p_T_i, int p_ref_ind, float p_threshold, np.ndarray[DTYPE_t] V):
    cdef int t = 0
    cdef np.ndarray[DTYPE_t] spike_train = np.zeros([p_T_i], dtype=DTYPE)
    
    while t < p_T_i:
        if V[t] >= p_threshold and V[t-1] < p_threshold:
            spike_train[t] = 1.0
            t = t + p_ref_ind
        t = t + 1
    
    return spike_train
    
def c_generateOUprocess(int T_ind, float dt, float tau, float sigma, float mu,\
                        np.ndarray[DTYPE_t] OU_process, np.ndarray[DTYPE_t] white_noise):
    cdef float OU_k1 = dt/tau
    cdef float OU_k2 = np.sqrt(2.0*dt/tau)
    cdef int t
    
    for t in range(T_ind-1):
        OU_process[t+1] = OU_process[t] + (mu - OU_process[t])*OU_k1 + sigma*OU_k2*white_noise[t]
        
    return OU_process

def c_simulate(int p_T, float p_dt, float p_gl, float p_C, float p_El, float p_Vr, float p_Tref,
               float p_Vt_star, float p_DV, float p_lambda0, np.ndarray[DTYPE_t] V,\
               np.ndarray[DTYPE_t] I, np.ndarray[DTYPE_t] p_eta, int p_eta_l, \
               np.ndarray[DTYPE_t] eta_sum, np.ndarray[DTYPE_t] p_gamma,\
               np.ndarray[DTYPE_t] gamma_sum, int p_gamma_l, np.ndarray[DTYPE_t] spks):
    
    cdef int Tref_ind = int(p_Tref/p_dt)
    cdef float p_dontspike, temp_lambda
    cdef int t, j
    
    cdef np.ndarray[FTYPE_t] r= np.random.random_sample(p_T)
#    cdef float maxint = float(sys.maxsize)
    
    t = 0
    while t < (p_T-1):
        
        ## INTEGRATE VOLTAGE
        V[t+1] = V[t] + p_dt/p_C*(-p_gl*(V[t] - p_El) + I[t] - eta_sum[t] )
        
        ## COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
        temp_lambda = p_lambda0 * exp( (V[t+1] - p_Vt_star - gamma_sum[t]) / p_DV )
        # since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0):
        p_dontspike = exp(-temp_lambda*(p_dt/1000.0))                                  
        
        ## PRODUCE SPIKE STOCHASTICALLY
        #r = rand()/rand_max;
        #r = random.randint(0,maxint)/maxint
        if (r[t] > p_dontspike):
                            
            if (t+1 < p_T-1):
                spks[t+1] = 1.0
            
            t = t + Tref_ind
            
            if (t+1 < p_T-1):
                V[t+1] = p_Vr
            
            
            ## UPDATE ADAPTATION PROCESSES     
            for j in range(p_eta_l):
                eta_sum[t+1+j] += p_eta[j]
            
            for j in range(p_gamma_l):
                gamma_sum[t+1+j] += p_gamma[j]
        
        t += 1
    
    return V, eta_sum, gamma_sum, spks
    
def c_simulate_twoComp (int p_T, float p_dt, int p_Tref_ind, float p_lambda0, float p_E0,\
                        int p_eta_A_l, np.ndarray[DTYPE_t] p_eta_A, np.ndarray[DTYPE_t] eta_A_sum,\
                        np.ndarray[DTYPE_t] filtered_currents, np.ndarray[DTYPE_t] spks):
    
    cdef int t
    cdef float p_dontspike, temp_lambda
    
    cdef np.ndarray[FTYPE_t] r= np.random.random_sample(p_T)
    
    t = 0
    while t < (p_T-1):        
        
        temp_lambda = p_lambda0* exp(p_E0 + filtered_currents[t] + eta_A_sum[t])
        p_dontspike = exp(-temp_lambda*(p_dt/1000.0)) 
        
        if (r[t] > p_dontspike):
                            
            if (t+1 < p_T-1):
                spks[t+1] = 1.0
            
            t = t + p_Tref_ind                
            
            ## UPDATE ADAPTATION PROCESS     
            eta_A_sum[t+1 : t+1+p_eta_A_l] +=  p_eta_A
        
        t += 1
        
    return eta_A_sum, spks
    
def c_simulateDeterministic_forceSpikes(int p_T, float p_dt, float p_gl, float p_C, float p_El,
                                        float p_Vr, float p_Tref, np.ndarray[DTYPE_t] V,\
                                        np.ndarray[DTYPE_t] I, np.ndarray[DTYPE_t] eta_sum,\
                                        np.ndarray[ITYPE_t] spks_i):
    
    cdef int t    
    cdef int next_spike = spks_i[0] + int(p_Tref/p_dt)
    cdef int spks_cnt = 0
    cdef int Tref_ind = int(p_Tref/p_dt)
    
    t = 0
    while t < (p_T-1):
        ## INTEGRATE VOLTAGE
        V[t+1] = V[t] + p_dt/p_C*( -p_gl*(V[t] - p_El) + I[t] - eta_sum[t] )       
   
        if ( t == next_spike ):
            spks_cnt = spks_cnt + 1
            if not spks_cnt > (spks_i.size-1):
                next_spike = spks_i[spks_cnt] + Tref_ind
            else: # i.e. there is no more spikes but next_spike should have a different value because otherwise you run into a endless loop
                next_spike = spks_i[spks_cnt-1] + p_T
            V[t-1] = 0
            V[t] = p_Vr
            t= t-1
        
        t += 1
    
    return V, eta_sum
    
def c_integrate_w(int p_T, float p_dt, int p_tau_w, float p_Ew, np.ndarray[DTYPE_t] V, np.ndarray[DTYPE_t] Z):
    
    cdef int t
    
    t = 0
    while t < (p_T-1):
        Z[t+1] = Z[t] + p_dt/p_tau_w * (V[t] - p_Ew - Z[t])
        
        t += 1
        
    return Z
    
def c_simulateDeterministic_forceSpikes_w(int p_T, float p_dt, float p_gl, float p_C, float p_El,
                                          float p_Ew, int p_tau_w, float p_a_w, float p_Vr,\
                                          float p_Tref, np.ndarray[DTYPE_t] V,\
                                          np.ndarray[DTYPE_t] I, np.ndarray[DTYPE_t] W,\
                                          np.ndarray[DTYPE_t] eta_sum, np.ndarray[ITYPE_t] spks_i):
    
    cdef int t    
    cdef int next_spike = spks_i[0] + int(p_Tref/p_dt)
    cdef int spks_cnt = 0
    cdef int Tref_ind = int(p_Tref/p_dt)
    
    t = 0
    while t < (p_T-1):
        # Integrate subthreshold adaptation current
        W[t+1] = W[t] + p_dt/p_tau_w*( p_a_w*(V[t] - p_Ew) - W[t])
        ## INTEGRATE VOLTAGE
        
        V[t+1] = V[t] + p_dt/p_C*( -p_gl*(V[t] - p_El) + I[t] - eta_sum[t] - W[t])       
   
        if ( t == next_spike ):
            spks_cnt = spks_cnt + 1
            if not spks_cnt > (spks_i.size-1):
                next_spike = spks_i[spks_cnt] + Tref_ind
            else: # i.e. there is no more spikes but next_spike should have a different value because otherwise you run into a endless loop
                next_spike = spks_i[spks_cnt-1] + p_T
            V[t-1] = 0
            V[t] = p_Vr
            t= t-1
        
        t += 1
    
    return V, eta_sum
    
    
    
    
    
    
    
    
    
                
        
        