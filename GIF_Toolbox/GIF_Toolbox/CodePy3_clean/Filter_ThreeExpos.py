import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import fftconvolve
from Filter import *
import Tools


class Filter_ThreeExpos(Filter) :
    
    """
    This class defines a function of time expanded using three exponential functions.
    
    A filter f(t) is defined in the form f(t) = amp_one * exp(-t) + amp_two *exp(-t) + amp_three*exp(-t)
                                                
    where the parameters to be fitted are amp_(one,two,three).
    """

    def __init__(self, length=5000.0, amp_one = -2, amp_two = 10, amp_three = -2) :
        
        Filter.__init__(self)
        
        self.p_length     = length           # ms, filter length
#        self.p_T_one      = T_one            # ms, length in time of the first exponential function
#        self.p_T_two      = T_two            # ms, length in time of the second exponential function        
#        self.p_amp_one    = amp_one          # nA, Amplitude of first exponential function
#        self.p_amp_two    = amp_two          # nA, Amplitude of second exponential function
#        self.p_amp_three  = amp_three        # nA, Amplitude of third exponential function
        
        # Coefficients A_j that define the shape of the filter f(t)
        self.filter_coeff = np.zeros(3)   # values of bins
        
        
    #############################################################################
    # Set functions
    #############################################################################

    def setFilter_Coefficients(self, coeff):
        
        """
        Set the coefficients of the filter (i.e. the values A1 and A2)
        """
        
        if len(coeff) == len(self.filter_coeff) :
            self.filter_coeff = coeff
        else :
            print ("Error, the number of coefficients do not match the number of parameters for this filter!") 
            
    def setFilter_toZero(self):
        
        """
        Set the coefficients of the filter to 0
        """
        
        self.filter_coeff = np.zeros(3) 
       
       
    #############################################################################
    # Get functions
    #############################################################################
    
    def getInterpolatedFilter(self, dt) :
            
        """
        Given a particular dt, the function compute and return the support t and f(t).
        """
        
        length_i = self.p_length/dt        
        
        # filter is the sum of the three exponentials
        filter_interpol =     self.filter_coeff[0]*np.exp(np.arange(length_i)*dt)\
                            + self.filter_coeff[1]*np.exp(np.arange(length_i)*dt)\
                            + self.filter_coeff[2]*np.exp(np.arange(length_i)*dt)


        filter_interpol_support = np.arange(len(filter_interpol))*dt

        return (filter_interpol_support, filter_interpol)

        
        
    def getLength(self):
        
        return self.p_length

        
    #############################################################################
    # Functions to compute convolutions
    #############################################################################

    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        T_i     = int(T/dt)
                       
        spks_i = Tools.timeToIndex(spks, dt)
        length_i = self.p_length/dt
         
        X = np.zeros( (T_i, 3) )
        tmp_X = np.zeros((T_i + length_i,3))
        # Fill matrix            
        for s in spks_i :
            # set all three exponentials
            for basis in np.arange(3):
                tmp_X[s:(s+length_i),basis] += np.exp(np.arange(length_i)*dt)
 
        
        X = tmp_X[:T_i,:]
        
        return X
    
    
    
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):
        
        T_i     = int(len(I))
        length_i = self.p_length/dt
        
#        bins_i  = Tools.timeToIndex(self.bins, dt)                
#        bins_l  = self.getNbOfBasisFunctions()
        
        X = np.zeros( (T_i, 3) )
        I_tmp = np.array(I,dtype='float64')        
        
        # Fill matrix
        for basis in np.arange(3) :
            
            window = np.exp(np.arange(length_i))
            window = np.array(window,dtype='float64')  
        
            F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
            F_star_I = F_star_I[:T_i]        
        
            #F_star_I_shifted = np.concatenate( ( np.zeros( int(bins_i[l]) ), F_star_I) )
            
            X[:,basis] = np.array(F_star_I, dtype='double')
        
        
        return X
    
    
    ########################################################################################
    # AUXILIARY METHODS USED BY THIS PARTICULAR IMPLEMENTATION OF FILTER
    ########################################################################################

    def setMetaParameters(self, length=5000.0):

        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """
        
        self.p_length     = length                  # ms, filter length

        
        
        
        