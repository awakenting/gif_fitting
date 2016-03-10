import matplotlib.pyplot as plt
import numpy as np
import abc

import copy

import Tools

from scipy.signal import fftconvolve
from scipy.optimize import leastsq


class Filter :

    """
    Abstract class defining an interface for a filter (ex: spike-triggered current eta or spike-triggered threshold movement).
    
    This class define a function of time expanded using a set of basis functions {f_j(t)}.
    
    A filter h(t) is defined in the form h(t) = sum_j b_j*f_j(t), where b_j is a set 
    of coefficient and f_j(t) is a set of rectangular basis functions.
    
    This class is used to define both the spike-triggered current eta(t) and the spike-triggered
    movement of the firing threshold gamma(t) as well as other filters.
    """
    
    __metaclass__  = abc.ABCMeta
    

    def __init__(self):
        
        self.filter_coeff    = []              # values of coefficients b_j
    
        # Results of multiexponential fit
        self.expfit_falg     = False           # True if the exponential fit has been performed
        self.expfit_dim      = 1
        self.b0              = []                                
        self.tau0            = []
    

    ######################################################################
    # SET METHODS
    ######################################################################
    
    @abc.abstractmethod    
    def setFilter_Coefficients(self, coeff):

        """
        Function to set the coefficients b_j. Implementations of this function should check the consistency between
        number of coefficients and number of basis functions.
        """
    
    @abc.abstractmethod
    def setFilter_Function(self, f):
        
        """
        Given a function of time f(t), the filer is initialized accordingly.
        """    

           
    def setFilter_toZero(self):
       
        """
        Set all parameters b_j to zero.
        """    
        
        nbBasisFunctions = self.getNbOfBasisFunctions()
        self.filter_coeff = np.zeros(nbBasisFunctions)
        
    ######################################################################
    # GET METHODS
    ######################################################################
    
    @abc.abstractmethod
    def getInterpolatedFilter(self, dt) :
        
        """
        Return the interpolated vector f(t). This function must return two arrays:
        time : ms, support of f(t)
        filter : the interpolated filter
        """

    def getInterpolatedFilter_expFit(self, dt) :
  
        if self.expfit_falg :
  
            t = np.arange(int(self.getLength()/dt))*dt
            F_exp = Filter.multiExpEval(t, self.b0, self.tau0)
        
            return (t, F_exp)

        else :        
            print ("Exp filter has not been performed.")
    
    
    def getCoefficients(self) :
        
        """
        Return an array that contains the parameters b_j. 
        """

        return  self.filter_coeff
        

    @abc.abstractmethod
    def getNbOfBasisFunctions(self) :
        
        """
        Return the number of basis functions used to define the filter.
        """

    @abc.abstractmethod
    def getLength(self):
        
        """
        Return the duration (in ms) of the filter
        """
              
    def computeIntegral(self, dt):
        
        """
        Return the duration (in ms) of the filter
        """
        (t, F) = self.getInterpolatedFilter(dt)   
        return sum(F)*dt   
            
    #########################################################
    # Function to perform convolutions
    ######################################################### 
         
    def convolution_ContinuousSignal(self, I, dt):
        
        (F_support, F) = self.getInterpolatedFilter(dt) 
    
        # Compute filtered input      
        I_tmp    = np.array(I,dtype='float64')
        F_star_I = fftconvolve(F, I_tmp, mode='full')*dt
        F_star_I = F_star_I[: int(len(I))]        
        
        F_star_I = F_star_I.astype("double")
        
        return F_star_I     


    def convolution_SpikeTrain(self, spks, T, dt):
        
        """
        spks  : in ms, spike times
        T     : in ms, duration of the experiment
        dt    : in ms, 
        """
        
        spks_i = Tools.timeToIndex(spks, dt)        
        (t,F) = self.getInterpolatedFilter(dt)
        F_length = len(F)
        
        filtered_spks = np.zeros(int(T/dt) + 5*F_length )
        
        for s in spks_i :
            filtered_spks[s : s + F_length] += F
        
        return filtered_spks[:int(T/dt)]




    @abc.abstractmethod                
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):

        """
        Return matrix containing the result of the convolution integral between I and all basis functions that define the filter.
        """      
           
                  
    @abc.abstractmethod
    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        """
        Given a list of spike times (spks, in ms) the function compute the convolution integral between
        the spike train and the filter. T (in ms) denote the length of the experiment.
        If S(t) is the spike train defined by the spike times in spks, the function should return
        a set of N arrays a1, ..., aN with:
        a_i = int_0^t f_i(s)S(t-s)ds
        """


    #########################################################
    # Function to average filters
    #########################################################
            
    @classmethod
    def averageFilters(cls, Fs) :
        
        F_avg = copy.deepcopy(Fs[0])
               
        F_coeff_all = []
        
        all_tau = []
        all_b = []
        
        for F in Fs :
            F_coeff_all.append(F.filter_coeff)
            
            if F.expfit_falg :
                all_tau.append(F.tau0)
                all_b.append(F.b0)
   
   
        F_avg.expfit_falg = False
        F_avg.b0 = []
        F_avg.taus = []
   
        F_avg.filter_coeff = np.mean(F_coeff_all, axis=0)
                
        # If individual filers have been fitted with exponential function, then average the parameters
        if F.expfit_falg :
            F_avg.b0 = np.mean(all_b, axis=0)          
            F_avg.taus = np.mean(all_tau, axis=0)                      
                                 
        return F_avg
    
    
    #########################################################
    # Functions for plotting
    #########################################################
    def plot(self, dt=0.05) :
 
        (t, F) = self.getInterpolatedFilter(dt)
        
        plt.figure(figsize=(5,5), facecolor='white')
        plt.plot([t[0],t[-1]], [0.0,0.0], ':', color='black')
        plt.plot(t, F, 'black', label='Filter')
        
        
        if self.expfit_falg :
            F_fit = Filter.multiExpEval(t, self.b0, self.tau0)
            plt.plot(t, F_fit, 'red', label='Multiexp fit')
        
                
        plt.xlim([t[0],t[-1]])
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Filter")
        plt.show()        
    


    @classmethod
    def plotAverageFilter(cls, Fs, dt=0.05, loglog=False, label_x="Time (ms)", label_y="Filter", plot_expfit=True) :
          
        plt.figure(figsize=(7,6), facecolor='white')
        
        F_interpol_all = []    
        F_exp_all = []

  
        for F in Fs :    
            (t, F_tmp) = F.getInterpolatedFilter(dt) 
            F_interpol_all.append(F_tmp)
            
            if F.expfit_falg :
                (t_exp, F_exp_tmp) = F.getInterpolatedFilter_expFit(dt) 
                F_exp_all.append(F_exp_tmp)
                
            if loglog :
                plt.loglog(t, F_tmp, 'gray')  
            else :
                plt.plot(t, F_tmp, 'gray')    


        F_interpol_mean = np.mean(F_interpol_all, axis=0)      
        
        if loglog :
            plt.loglog(t, F_interpol_mean, 'red', lw=2)   
        else :
            plt.plot(t, F_interpol_mean, 'red', lw=2)     
 
        
        if plot_expfit :
            
            if len(F_exp_all) > 0 :
                
                F_exp_mean = np.mean(F_exp_all, axis=0)  
                plt.plot(t_exp, F_exp_mean, 'black', lw=2, ls='--')         
        
        
        
        plt.plot([t[0],t[-1]], [0.0,0.0], ':', color='black')
        plt.xlim([t[0],t[-1]])
        plt.xlabel(label_x)
        plt.ylabel(label_y)        

 

    #########################################################
    # functions used to perform exp fit
    #########################################################
    def fitSumOfExponentials(self, dim, bs, taus, ROI=None, dt=0.1, fixed_taus=False) :
        
        """
        Fit the interpolated filter with a sum of exponentails: F_fit(t) = sum_j^N b_j exp(-t/tau_j)
        dim : number N of exponentials
        bs  : list with initial conditions for amplitudes b_j
        taus: ms, list with initial conditions for timescales tau_j
        ROI :[lb, ub], in ms (consistent with units of dt). Specify lowerbound and upperbound (in time) where fit is perfomred.
        dt  : the filer is interpolated and fitted using discretization steps defined in dt.
        """

        (t, F) = self.getInterpolatedFilter(dt)
        
        if ROI == None :
            t_fit = t
            F_fit = F
        
        else :
            lb = int(ROI[0]/dt)
            ub = int(ROI[1]/dt)
            
            t_fit = t[ lb : ub ]
            F_fit = F[ lb : ub ]    

        
        if fixed_taus:
            p0 = bs
            
            plsq = leastsq(Filter.multiExpResiduals_fixed_taus, p0, args=(t_fit,F_fit,dim,taus),
                           maxfev=100000,ftol=0.00000001)
            
            bs_opt = plsq[0]
            
            F_exp = Filter.multiExpEval(t, bs_opt, taus)
            
            self.expfit_falg = True
            self.b0          = np.reshape(bs_opt,(dim,1))
            self.tau0        = taus
            
        else:
            p0 = np.concatenate((bs,taus))
            
            plsq = leastsq(Filter.multiExpResiduals, p0, args=(t_fit,F_fit,dim), maxfev=100000,
                           ftol=0.00000001)
            
            p_opt = plsq[0]
            bs_opt = p_opt[:dim]
            taus_opt = p_opt[dim:]
            
            F_exp = Filter.multiExpEval(t, bs_opt, taus_opt)
            
            self.expfit_falg = True
            self.b0          = bs_opt 
            self.tau0        = taus_opt
                
        return (t, F_exp)
        
    def fit_sumOfExpos_optimize_dim(self, maxdim, bs=[], taus=[], ROI=None, dt=0.1,
                                    fixed_taus=False, method='bruteforce') :
        
        """
        Fit the interpolated filter with a sum of exponentails: F_fit(t) = sum_j^N b_j exp(-t/tau_j)
        dim : number N of exponentials
        bs  : list with initial conditions for amplitudes b_j
        taus: ms, list with initial conditions for timescales tau_j
        ROI :[lb, ub], in ms (consistent with units of dt). Specify lowerbound and upperbound (in time) where fit is perfomred.
        dt  : the filer is interpolated and fitted using discretization steps defined in dt.
        """
        
        temp_bs = []
        temp_taus = []
        temp_sse = np.zeros((maxdim,1))
        
        if method == 'bruteforce':
            # check if there are intial values for each value of dim
            if (type(bs) is not list) or (not len(bs) == maxdim):
                bs = [np.ones((dim,1)) for dim in np.arange(1,maxdim+1)]
            if (type(taus) is not list) or (not len(taus) == maxdim):
                taus = [np.ones((dim,1)) for dim in np.arange(1,maxdim+1)]
            
            if fixed_taus:
                for dim in np.arange(1,maxdim+1):
                    self.fitSumOfExponentials(dim,bs[dim-1],taus[dim-1],ROI,dt,fixed_taus=True)
                    
                    temp_bs.append(self.b0)
                    temp_taus.append(self.tau0)
                    temp_sse[dim-1] = self.get_expfit_sse(dim,dt)
            else:
                for dim in np.arange(1,maxdim+1):
                    self.fitSumOfExponentials(dim,bs[dim-1],taus[dim-1],ROI,dt)
                    
                    temp_bs.append(self.b0)
                    temp_taus.append(self.tau0)
                    temp_sse[dim-1] = self.get_expfit_sse(dim,dt)                
                
            opt_ind = np.argmin(temp_sse)
            non_opt = ~(np.arange(len(temp_sse))==opt_ind)
            
            # if a lower dimension than the optimal is only worse by less than
            # occams_tolerance percent take it as optimal solution
            
            occams_tolerance = 1 # in percent of the best SSE
            # relative difference to best SSE
            sse_relative = (temp_sse-temp_sse[opt_ind])/temp_sse[opt_ind]
            # relative differences that are below the tolerance
            sse_occam = sse_relative[non_opt] < occams_tolerance
            # indeces of those below tolerance
            sse_occam_ind = np.flatnonzero(sse_occam)
            if any(sse_occam) and sse_occam_ind.min() < opt_ind:
                opt_ind = sse_occam_ind.min()
                
            self.b0         = temp_bs[opt_ind]
            self.tau0       = temp_taus[opt_ind]
            self.expfit_dim = opt_ind + 1
            
        
        (t, F) = self.getInterpolatedFilter(dt)
        F_exp = self.multiExpEval(t,self.b0,self.tau0)
        
        return (t, F_exp)
                    
    def get_expfit_sse(self, dim, dt=0.1):
        if self.expfit_falg:
            (t, F) = self.getInterpolatedFilter(dt)
            p = np.concatenate((self.b0,self.tau0))
            resids = self.multiExpResiduals(p,t,F,dim)
        else:
            print('Exponential fit has not been performed yet.')
        
        return np.sum(resids**2)
        
    @classmethod
    def multiExpResiduals(cls, p, x, y, d):
        bs = p[0:d]
        taus = p[d:2*d]
            
        return (y - Filter.multiExpEval(x, bs, taus))
        
    @classmethod
    def multiExpResiduals_fixed_taus(cls, p, x, y, d, taus):
        bs = p[0:d]
            
        return (y - Filter.multiExpEval(x, bs, taus)) 
    
    
    @classmethod
    def multiExpEval(cls, x, bs, taus):
        
        result = np.zeros(len(x))
        L = len(bs)
            
        for i in range(L) :
            result = result + bs[i] *np.exp(-x/taus[i])
            
        return result
    

    
    
    

