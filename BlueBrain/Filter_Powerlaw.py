import numpy as np

from scipy.signal import fftconvolve

from Filter import Filter
import Tools


class Filter_Powerlaw(Filter) :
    
    """
    This class defines a function of time expanded using a truncated power law function.
    
    A filter f(t) is defined in the form f(t) = A1                   for 0 < t < T
                                                A2*(t)^(beta)        for T < t
    where T ~ 5 ms and beta ~ -0.8 are metaparameters and the parameters to be fitted are A1 and A2.
    """

    def __init__(self, length=5000.0, Tconst=5, power=-0.8, powerTime=2000) :
        
        Filter.__init__(self)
        
        self.p_length     = length                  # ms, filter length
        self.p_Tconst     = Tconst                  # ms, initial time period for which the filter remains constant
        self.p_power      = power                   # exponent of power law function
        self.p_powerTime  = powerTime               # length of the power law
        
        # Coefficients A_j that define the shape of the filter f(t)
        self.filter_coeff = np.zeros(2)   # values of bins
        
        
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
        
        self.filter_coeff = np.zeros(2) 
       
       
    #############################################################################
    # Get functions
    #############################################################################
    
    def getInterpolatedFilter(self, dt) :
            
        """
        Given a particular dt, the function compute and return the support t and f(t).
        """
        
        Tconst_i = self.p_Tconst/dt
        #powerTime_i = self.p_powerTime/dt
        length_i = self.p_length/dt
        
        filter_interpol = np.zeros( length_i )
        # set initial constant part to A_1
        filter_interpol[0:Tconst_i] = self.filter_coeff[0]
        # set powerlaw part to A_2
        filter_interpol[Tconst_i:length_i] = self.filter_coeff[1]*np.power(np.arange(Tconst_i,length_i),self.p_power)

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
        Tconst_i = self.p_Tconst/dt
        powerTime_i = self.p_powerTime/dt
         
        X = np.zeros( (T_i, 2) )
        tmp_const = np.zeros( T_i + Tconst_i )
        tmp_pow = np.zeros( T_i + 2*powerTime_i )
        # Fill matrix            
        for s in spks_i :
            tmp_const[s:(s+Tconst_i)] += 1
            tmp_pow[(s + Tconst_i):(s + Tconst_i + powerTime_i)] += np.power(np.arange(Tconst_i,powerTime_i+Tconst_i),self.p_power)
        
        X[:,0] = tmp_const[:T_i]
        X[:,1] = tmp_pow[:T_i]
        
        return X
    
    
    
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):
        
        T_i     = int(len(I))
        
        X = np.zeros( (T_i, 2) )
        I_tmp = np.array(I,dtype='float64') 
        
        Tconst_i = self.p_Tconst/dt
        powerTime_i = self.p_powerTime/dt
        
        windows = { 'win0': np.ones(Tconst_i),\
                    'win1':np.power(np.arange(Tconst_i,powerTime_i+Tconst_i),self.p_power)}
        shifts = {'shift0':0,'shift1':Tconst_i}
        
        # Fill matrix
        for i in np.arange(2) :
            
            window = np.array(windows['win'+str(i)],dtype='float64')
        
            F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
            F_star_I = F_star_I[: T_i]        
        
            F_star_I_shifted = np.concatenate( ( np.zeros( shifts['shift'+str(i)]), F_star_I) )
            
            X[:,i] = np.array(F_star_I_shifted[:T_i], dtype='double')
        
        
        return X
    
    
    ########################################################################################
    # AUXILIARY METHODS USED BY THIS PARTICULAR IMPLEMENTATION OF FILTER
    ########################################################################################

    def setMetaParameters(self, length=5000.0, Tconst=5, power=-0.8, powerTime=2000):

        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """
        
        self.p_length     = length                  # ms, filter length
        self.p_Tconst     = Tconst                  # ms, initial time period for which the filter remains constant
        self.p_power      = power                   # exponent of power law function
        self.p_powerTime  = powerTime               # length of the power law
        
        
        
        