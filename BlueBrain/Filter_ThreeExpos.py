import numpy as np

from scipy.signal import fftconvolve

from .Filter import Filter
from . import Tools


class Filter_ThreeExpos(Filter) :
    
    """
    This class defines a function of time expanded using three exponential functions.
    
    The filter f(t) is simply the sum of the three exponentials:
    
    f(t) = amp_one * exp(-t/tau_one) + amp_two *exp(-t/tau_two) + amp_three*exp(-t/tau_three)
                                                
    where the parameters to be fitted are amp_(one,two,three).
    """

    def __init__(self, length=5000.0, tau_one=10, tau_two=100, tau_three=500) :
        
        Filter.__init__(self)
        
        self.p_length     = length           # ms, filter length
        self.p_tau_dict   = {'tau0':tau_one, 'tau1':tau_two, 'tau2':tau_three} # the three time constants
        
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
        p_taus_i = [self.p_tau_dict['tau'+str(i)]/dt for i in range(3)]
        
        # filter is the sum of the three exponentials
        filter_interpol =     self.filter_coeff[0]*np.exp(-np.arange(length_i)/p_taus_i[0])\
                            + self.filter_coeff[1]*np.exp(-np.arange(length_i)/p_taus_i[1])\
                            + self.filter_coeff[2]*np.exp(-np.arange(length_i)/p_taus_i[2])


        filter_interpol_support = np.arange(len(filter_interpol))*dt

        return (filter_interpol_support, filter_interpol)
        
    def getInterpolatedBasisfunctions(self, dt) :
            
        """
        Given a particular dt, the function computes and return the support t and the basisfunctions of f(t).
        """
        
        length_i = self.p_length/dt
        p_taus_i = [self.p_tau_dict['tau'+str(i)]/dt for i in range(3)]
        
        # filter is the sum of the three exponentials
        filter_interpol =     [self.filter_coeff[0]*np.exp(-np.arange(length_i)/p_taus_i[0]),\
                               self.filter_coeff[1]*np.exp(-np.arange(length_i)/p_taus_i[1]),\
                               self.filter_coeff[2]*np.exp(-np.arange(length_i)/p_taus_i[2])]


        filter_interpol_support = np.arange(len(filter_interpol[0]))*dt

        return (filter_interpol_support, filter_interpol)

        
        
    def getLength(self):
        
        return self.p_length
        
    def getTimeConstants(self):
        
        return self.p_tau_dict

        
    #############################################################################
    # Functions to compute convolutions
    #############################################################################

    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        T_i     = int(T/dt)
                       
        spks_i = Tools.timeToIndex(spks, dt)
        length_i = self.p_length/dt
        p_taus_i = [self.p_tau_dict['tau'+str(i)]/dt for i in range(3)]
         
        X = np.zeros( (T_i, 3) )
        tmp_X = np.zeros((T_i + length_i,3))
        # Fill matrix            
        for s in spks_i :
            # set all three exponentials
            for basis in np.arange(3):
                tmp_X[s:(s+length_i),basis] += np.exp(-np.arange(length_i)/p_taus_i[basis])
 
        
        X = tmp_X[:T_i,:]
        
        return X
    
    
    
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):
        
        T_i     = int(len(I))
        length_i = self.p_length/dt
        p_taus_i = [self.p_tau_dict['tau'+str(i)]/dt for i in range(3)]
        
#        bins_i  = Tools.timeToIndex(self.bins, dt)                
#        bins_l  = self.getNbOfBasisFunctions()
        
        X = np.zeros( (T_i, 3) )
        I_tmp = np.array(I,dtype='float64')
        
        # Fill matrix
        for basis in np.arange(3) :
            window = np.exp(-np.arange(length_i)/p_taus_i[basis])
            window = np.array(window,dtype='float64')  
        
            F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
            F_star_I = F_star_I[:T_i]        
        
            #F_star_I_shifted = np.concatenate( ( np.zeros( int(bins_i[l]) ), F_star_I) )
            
            X[:,basis] = np.array(F_star_I, dtype='double')
        
        
        return X
    
    
    ########################################################################################
    # AUXILIARY METHODS USED BY THIS PARTICULAR IMPLEMENTATION OF FILTER
    ########################################################################################

    def setMetaParameters(self, length=5000.0, tau_one=1, tau_two=10, tau_three=50):

        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """
        
        self.p_length     = length                  # ms, filter length
        self.p_tau_dict   = {'tau0':tau_one, 'tau1':tau_two, 'tau2':tau_three} # the three time constants
        
        
        
        