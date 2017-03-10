import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .GIF import GIF
from .Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from . import Tools
from .Tools import reprint
from . import cython_helpers as cyth


class GIF_subadapt_constrained(GIF) :

    """
    Generalized Integrate and Fire model
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
    
    
    where the membrane potential dynamics is given by:
    
    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j) - W
    
    with the subthreshold adaptation current W:
    
    tau_w * dW/dt = a_W(V-El) - W
    
    
    
    
    and the firing threshold V_T is given by:
    
    V_T = Vt_star + sum_j gamma(t-\hat t_j)
    
    
    and \hat t_j denote the spike times.
    """

    def __init__(self, dt=0.1):
        
        super(ThresholdModel, self).__init__()
                   
        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)
  
        # Define model parameters
        
        self.gl      = 1.0/100.0        # nS, leak conductance
        self.C       = 20.0*self.gl     # nF, capacitance
        self.El      = -65.0            # mV, reversal potential
        
        self.Ew      = self.El          # equilibrium potential of subthreshold adaptation current W
        self.a_w     = 1                # nA, parameter for subthreshold adaptation current W
        self.tau_w_opt = 10             # ms, optimal value for tau_w with respect to scores in
                                        # tau_w_scores
        
        self.tau_w_values   = np.array([10,20])     # ms, array of values for the time constant of
                                                    # subthreshold adaptation current W
        self.tau_w_scores   = np.zeros((2,1))       # percentage, array for the values of the
                                                    # percentage explained on dV/dt for the
                                                    # corresponding value in tau_w_values
        
        self.Vr      = -50.0            # mV, voltage reset
        self.Tref    = 4.0              # ms, absolute refractory period
        
        self.Vt_star = -48.0            # mV, steady state voltage threshold VT*
        self.DV      = 0.5              # mV, threshold sharpness
        self.lambda0 = 1.0              # by default this parameter is always set to 1.0 Hz
        
        
        self.eta     = Filter_Rect_LogSpaced()    # nA, spike-triggered current (must be instance of class Filter)
        self.gamma   = Filter_Rect_LogSpaced()    # mV, spike-triggered movement of the firing threshold (must be instance of class Filter)
        
        self.p_trace = [] # temporary parameter to store all the intermediate parameter vectors during the constrained gradient descent
        self.eta_trace = []
        
        
        # Varialbes relatd to fit
        
        self.avg_spike_shape = 0
        self.avg_spike_shape_support = 0
        
        self.expm_file      = []              # filename of the experiment object, with the data that was fitted
        self.pred           = []              # prediction object with simulated traces and spike trains
        
        self.var_explained_dV  = 0
        self.var_explained_V   = 0
        self.mean_se_dV = 0
        self.mean_se_V = 0
        
        
        # Initialize the spike-triggered current eta with an exponential function        
        
        def expfunction_eta(x):
            return 0.2*np.exp(-x/100.0)
        
        #self.eta.setFilter_Function(expfunction_eta)


        # Initialize the spike-triggered current gamma with an exponential function        
        
        def expfunction_gamma(x):
            return 10.0*np.exp(-x/100.0)
        
        #self.gamma.setFilter_Function(expfunction_gamma)        
        
              
            
    ########################################################################################################
    # SET DT FOR NUMERICAL SIMULATIONS (KERNELS ARE REINTERPOLATED EACH TIME DT IS CHANGED)
    ########################################################################################################    
    def setDt(self, dt):

        """
        Define the time step used for numerical simulations. The filters eta and gamma are interpolated accordingly.
        """
        
        self.dt = dt

    def set_tau_w_values(self, values):
        """
        Set the values over which tau_w shall be looped. The score array is initiated accordingly.
        """
        
        self.tau_w_values = values
        self.tau_w_scores = np.zeros((len(values),1))
    
    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################
    def simulateSpikingResponse(self, I, dt):
        
        """
        Simulate the spiking response of the GIF_subadapt model to an input current I (nA) with time step dt.
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El.
        """
        self.setDt(dt)
    
        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)
        
        return spks_times


    def simulateVoltageResponse(self, I, dt) :

        self.setDt(dt)
    
        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)
        
        return (spks_times, V, V_T)


    def simulate(self, I, V0):
 
        """
        Simulate the spiking response of the GIF_subadapt model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times 
        """
 
        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt
        
        # Model parameters
        p_gl        = self.gl
        p_C         = self.C 
        p_El        = self.El
        p_Ew        = self.Ew
        p_tau_w     = self.tau_w_opt
        p_a_w       = self.a_w
        p_Vr        = self.Vr
        p_Tref      = self.Tref
        p_Vt_star   = self.Vt_star
        p_DV        = self.DV
        p_lambda0   = self.lambda0
        
        # Model kernels   
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)   
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)   
        p_gamma     = p_gamma.astype('double')
        p_gamma_l   = len(p_gamma)
      
        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        W = np.array(np.zeros(p_T), dtype="double")
        spks = np.array(np.zeros(p_T), dtype="double")                      
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")            
 
        # Set initial condition
        V[0] = V0
        W[0] = 0
        
        # computational intensive part is calculated by cython function
        V, eta_sum, gamma_sum, spks = cyth.c_simulate_w(p_T, p_dt, p_gl, p_C, p_El, p_Ew, p_tau_w, 
                                                      p_a_w, p_Vr, p_Tref, p_Vt_star, p_DV, 
                                                      p_lambda0, V, I, W, p_eta, p_eta_l, eta_sum, 
                                                      p_gamma, gamma_sum, p_gamma_l, spks)
        
        time = np.arange(p_T)*self.dt
        
        eta_sum   = eta_sum[:p_T]     
        V_T = gamma_sum[:p_T] + p_Vt_star
     
        spks = (np.where(spks==1)[0])*self.dt
    
        return (time, V, eta_sum, V_T, spks)
        
        
    def simulateDeterministic_forceSpikes(self, I, V0, spks):
        
        """
        Simulate the subthresohld response of the GIF_subadapt model to an input current I (nA) with time step dt.
        Adaptation currents are enforced at times specified in the list spks (in ms) given as an argument to the function.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        """
 
        # Input parameters
        p_T          = len(I)
        p_dt         = self.dt
          
          
        # Model parameters
        p_gl        = self.gl
        p_C         = self.C 
        p_El        = self.El
        p_Ew        = self.Ew
        p_tau_w     = self.tau_w_opt
        p_a_w       = self.a_w
        p_Vr        = self.Vr
        p_Tref      = self.Tref
        p_Tref_i    = int(self.Tref/self.dt)
    
    
        # Model kernel      
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)   
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)


        # Define arrays
        V        = np.array(np.zeros(p_T), dtype="double")
        I        = np.array(I, dtype="double")
        W        = np.array(np.zeros(p_T), dtype="double")
        spks     = np.array(spks, dtype="double")
        spks_i   = Tools.timeToIndex(spks, self.dt)


        # Compute adaptation current (sum of eta triggered at spike times in spks) 
        eta_sum  = np.array(np.zeros(p_T + 1.1*p_eta_l + p_Tref_i), dtype="double")   
        
        for s in spks_i :
            eta_sum[s + 1 + p_Tref_i  : s + 1 + p_Tref_i + p_eta_l] += p_eta
        
        eta_sum  = eta_sum[:p_T]  
   
   
        # Set initial condition
        V[0] = V0
        W[0] = 0
        
        # computational intensive part is calculated by cython function
        V, eta_sum = cyth.c_simulateDeterministic_forceSpikes_w(p_T, p_dt, p_gl, p_C, p_El, p_Ew,\
                                                                p_tau_w, p_a_w, p_Vr, p_Tref, V, I,\
                                                                W, eta_sum, spks_i)
        
        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]     

        return (time, V, eta_sum)
        
        
    def fit(self, experiment, DT_beforeSpike = 5.0):
        
        """
        Fit the GIF_subadapt model on experimental data.
        The experimental data are stored in the object experiment.
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """
        
        # Three step procedure used for parameters extraction 
        
        print ("\n################################")
        print ("# Fit GIF_subadapt")
        print ("################################\n")
        
        self.fitVoltageReset(experiment, self.Tref, do_plot=False)
        
        self.initialize_eta()
        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)
        
        self.fitStaticThreshold(experiment)

        self.fitThresholdDynamics(experiment)




    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################        
    def fitSubthresholdDynamics(self, experiment, DT_beforeSpike=5.0):
                    
        print ("\nGIF_subadapt MODEL - Fit subthreshold dynamics..." )
            
        # Expand eta in basis functions
        self.dt = experiment.dt
                
        # constrained gradient descent using scipy.optimize.minimize :
        
        # objective funtion to minimize is the squared error:
        # f(p) = (Y - X*p)^2 , p = [-g_l/C, 1/C, E_l*g_l/C, eta_1/C, ..., eta_M/C, -a_w/C]
        
        # the extra arguments have to be consistent for the function and the constraints, that is
        # why also tau_w is a parameter:
        def func(p, pX, pY, ptau_w = 10):
            error = np.mean((pY - np.dot(pX,p))**2)
            return error
                
        # first boundaries correspond to 0.5 ms to 500 ms for the time constant of kappa tau_m
        # second boundaries correspond to 2 nF to 0.1 pF for the capacitance C
        bonds = ((-2, -0.002),(0.5, 10000),(None, None),
                 (None, None),(None, None),(None, None),
                 (None, None),(None, None),(None, None),
                 (None, None),(None, None),(None, None),
                 (None, None),(None, None),(None, None),
                 (None, None),(None, None),(None, None),
                 (None, None),(None, None),(None, None),
                 (None,None))
                
        
        def get_params(pk):
            new_pk = np.reshape(pk,(len(pk),1))
            self.p_trace.append(new_pk)

        plot_parameter_traces = False
        
        for idx_tau_w, c_tau_w in enumerate(self.tau_w_values):
            
            print('Running value ' + str(idx_tau_w+1) + ' of ' + str(len(self.tau_w_values)) +
                  ' for tau_w...', end='\r')
                  
            tau_w = c_tau_w
            # Build X matrix and Y vector to perform linear regression (use all traces in training set)            
            X = []
            Y = []
        
            cnt = 0
            
            for tr in experiment.trainingset_traces :
            
                if tr.useTrace :
            
                    cnt += 1
                    #reprint( "Compute X matrix for repetition %d \n" % (cnt) )          
                    
                    (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, c_tau_w, DT_beforeSpike=DT_beforeSpike)
         
                    X.append(X_tmp)
                    Y.append(Y_tmp)
        
        
            # Concatenate matrixes associated with different traces to perform a single multilinear regression
            if cnt == 1:
                X = X[0]
                Y = Y[0]
                
            elif cnt > 1:
                X = np.concatenate(X, axis=0)
                Y = np.concatenate(Y, axis=0)
            
            else :
                print ("\nError, at least one training set trace should be selected to perform fit.")
            
        
            # constraints are:
            # 1. C/g_l >= 0  (C/g_l = tau_m)
            # 2. C/g_l = tau_m < tau_w  <=>   tau_w - (C/g_l) >= 0
            cons = ({'type': 'ineq',
                     'fun' : lambda p, pX, pY, ptau_w : np.array([-p[0]**-1]),
                     'args': (X, Y, tau_w)},
                    {'type': 'ineq',
                     'fun' : lambda p, pX, pY, ptau_w : np.array([ptau_w - (-p[0]**-1)]),
                     'args': (X, Y, tau_w)})
                     
            # initial parameter set
            p0 = np.ones([np.shape(X)[1],1])
            # set initial value of -g_l/C to -0.1 so that it's whithin the boundaries, see bonds
            p0[0] = p0[0]*-0.1
                     
            if plot_parameter_traces:
                self.p_trace = []
                res = minimize(func, p0, args=(X, Y, tau_w), constraints=cons, method='SLSQP', 
                   options={'maxiter': 1000}, callback=get_params, bounds = bonds)
                   
                p_matrix = np.concatenate(self.p_trace,axis=1)
                
                plt.figure()    
                for i in range(len(self.p_trace[0])):
                    plt.subplot(4,6,(i+1))
                    plt.plot(p_matrix[i,:])
                    plt.title(str(i))
                    
            else:
                res = minimize(func, p0, args=(X, Y, tau_w), constraints=cons, method='SLSQP', 
                   options={'maxiter': 1000}, bounds = bonds)            
                       
            # optimal parameter set is saved in res.x
            p_opt = res.x
            #print(res.message)
            temp_eta = -p_opt[3:-1]*(1./p_opt[1])
            self.eta_trace.append(np.reshape(temp_eta,(len(temp_eta),1)))
            
            # Compute percentage of variance explained on dV/dt
            self.var_explained_dV = 1.0 - np.mean((Y - np.dot(X,p_opt))**2)/np.var(Y)
            self.mean_se_dV = np.mean((Y - np.dot(X,p_opt))**2)
            # Update model parameters if score is best so far
            
            if self.var_explained_dV > np.max(self.tau_w_scores):                
                self.C  = 1./p_opt[1]
                self.gl = -p_opt[0]*self.C
                self.El = p_opt[2]*self.C/self.gl
                self.eta.setFilter_Coefficients(-p_opt[3:-1]*self.C)
                self.a_w = -p_opt[-1]*self.C
                
            self.tau_w_scores[idx_tau_w] = self.var_explained_dV
                
                
        self.tau_w_opt = self.tau_w_values[np.argmax(self.tau_w_scores)]
        
        print('\nDone')
            

        self.printParameters()   
        
        print ("Percentage of variance explained (on dV/dt): %0.2f" % (np.max(self.tau_w_scores)*100.0))        


        # Compute percentage of variance explained on V
    
        SSE = 0     # sum of squared errors
        MSE = 0     # mean squared error
        VAR = 0     # variance of data
        
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :

                # Simulate subthreshold dynamics 
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
                
                plot_simulated = False
                if plot_simulated:
                    plt.figure()
                    plt.plot(time,V_est,hold=True)
                    plt.plot(time,tr.V)
                    
                    plt.legend(['Estimated voltage','True voltage'])
                    
                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)
                
                SSE += np.sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                MSE += np.mean((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])
                
        self.var_explained_V = 1.0 - SSE / VAR
        self.mean_se_V = MSE

        #print('Sum squared error: %0.2f' % (SSE))
        print ("Percentage of variance explained (on V): %0.2f" % (self.var_explained_V*100.0))
                    

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, c_tau_w, DT_beforeSpike=5.0):
                   
        # Length of the voltage trace       
        Tref_ind = int(self.Tref/trace.dt)
        
        # Select region where to perform linear regression
        selection = trace.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)
        selection_l = len(selection)
        
        # Build X matrix for linear regression
        X = np.zeros( (selection_l, 3) )
        
        # Fill first two columns of X matrix        
        X[:,0] = trace.V[selection]
        X[:,1] = trace.I[selection]
        X[:,2] = np.ones(selection_l) 
        
             
        # Compute and fill the remaining columns associated with the spike-triggered current eta               
        X_eta = self.eta.convolution_Spiketrain_basisfunctions(trace.getSpikeTimes() + self.Tref, trace.T, trace.dt) 
        
        # Integrate tau_w*dZ/dt = V - El - Z, where Z = W/a_w
        
        # Parameters for integration
        p_T          = len(trace.I)
        p_dt         = self.dt
        p_tau_w      = c_tau_w
        p_Ew         = self.Ew
        V            = np.array(trace.V, dtype="double")
        Z            = np.array(np.zeros(p_T), dtype="double")
        
        # initial value        
        Z[0] = 0
        
        # computational intensive part is calculated by cython function
        
        Z = cyth.c_integrate_w(p_T, p_dt, p_tau_w, p_Ew, V, Z)
        Z = np.reshape(Z,(len(Z),1))
        
        # put everything together
        X = np.concatenate( (X, X_eta[selection,:], Z[selection]), axis=1 )


        # Build Y vector (voltage derivative)    
        
        # COULD BE A BETTER SOLUTION IN CASE OF EXPERIMENTAL DATA (NOT CLEAR WHY)
        #Y = np.array( np.concatenate( ([0], np.diff(trace.V)/trace.dt) ) )[selection]
        
        # CORRECT SOLUTION TO FIT ARTIFICIAL DATA
        Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]      

        return (X, Y)
        
 
    ########################################################################################################
    # GET FUNCTIONS FOR FITTING PERFORMANCE
    ########################################################################################################
 
    def get_var_explained(self):
        if self.var_explained_dV is None:
            self.compute_var_explained()
            return self.var_explained_dV, self.var_explained_V
        else:
            return self.var_explained_dV, self.var_explained_V
            
    def compute_var_explained(self, DT_beforeSpike=5.0):
        
        SSE = 0     # sum of squared errors
        MSE = 0     # mean squared error
        VAR = 0     # variance of data
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :

                # Simulate subthreshold dynamics 
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())                    
                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)
                
                SSE += np.sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                MSE += np.mean((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])
                
        self.var_explained_V = 1.0 - SSE / VAR
        self.mean_se_V = MSE
        
        # Build X matrix and Y vector to perform linear regression (use all traces in training set)            
        X = []
        Y = []
    
        cnt = 0
        
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :
        
                cnt += 1
                reprint( "Compute X matrix for repetition %d" % (cnt) )          
                
                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, DT_beforeSpike=DT_beforeSpike)
     
                X.append(X_tmp)
                Y.append(Y_tmp)
    
    
        # Concatenate matrixes associated with different traces to perform a single multilinear regression
        if cnt == 1:
            X = X[0]
            Y = Y[0]
            
        elif cnt > 1:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
        
        else :
            print ("\nError, at least one training set trace should be selected to perform fit.")
   
        eta_coeffs = self.eta.getCoefficients()
        p_opt = np.empty(4+len(eta_coeffs))
            
        p_opt[0] = -self.gl/self.C
        p_opt[1] = 1/self.C
        p_opt[2] = self.gl*self.El/self.C
        p_opt[3:-1] = eta_coeffs
        p_opt[-1] = -self.a_w/self.C
                
        self.var_explained_dV = 1.0 - np.mean((Y - np.dot(X,p_opt))**2)/np.var(Y)
        self.mean_se_dV = np.mean((Y - np.dot(X,p_opt))**2)
        
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     
      
      
    def printParameters(self):

        print ("\n-------------------------")       
        print ("GIF_subadapt model parameters:")
        print ("-------------------------")
        print ("tau_m (ms):\t%0.3f"  % (self.C/self.gl))
        print ("tau_w (ms):\t%0.3f"  % (self.tau_w_opt))
        print ("a_w (nA):\t%0.3f"    % (self.a_w))
        print ("R (MOhm):\t%0.3f"    % (1.0/self.gl))
        print ("C (nF):\t\t%0.3f"    % (self.C))
        print ("gl (nS):\t%0.3f"     % (self.gl))
        print ("El (mV):\t%0.3f"     % (self.El))
        print ("Tref (ms):\t%0.3f"   % (self.Tref))
        print ("Vr (mV):\t%0.3f"     % (self.Vr))  
        print ("Vt* (mV):\t%0.3f"    % (self.Vt_star))
        print ("DV (mV):\t%0.3f"     % (self.DV))    
        print ("-------------------------\n")
                  