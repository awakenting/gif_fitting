import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from numpy.linalg import inv

from ThresholdModel import *
from Filter_ThreeExpos import *
from Filter_Powerlaw import *

from Tools import reprint
import cython_helpers as cyth


class TwoComp_passive(ThresholdModel) :

    """
    Generalized Integrate and Fire model with two compartments
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( k_s*I_s + k_ds*I_d + sum_ t_j_hat eta_A(t-t_j_hat) )
    
    where the somatic membrane potential dynamics is given by:
    
    C dV_s/dt = -gl(V-El) + I_s + k_ds*I_d + sum_j I_A(t-\hat t_j)    
    
    and \hat t_j denote the spike times, I_A denotes the spike triggered adaptation current.
    """

    def __init__(self, dt=0.1):
                   
        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)
  
        # Define model parameters
        
        self.Tref    = 4.0              # ms, absolute refractory period
        
        self.lambda0 = 1.0              # by default this parameter is always set to 1.0 Hz
        
        self.k_s       = Filter_ThreeExpos()  # nA, kernel of somatic membrane filter
        self.e_ds      = Filter_ThreeExpos()  # nA, kernel linking the current injected in the dendrite to the current reaching the soma
        self.eta_A     = Filter_Powerlaw()    # nA, effective spike-triggered adaptation (must be instance of class Filter)                
              
            
    ########################################################################################################
    # SET DT FOR NUMERICAL SIMULATIONS (KERNELS ARE REINTERPOLATED EACH TIME DT IS CHANGED)
    ########################################################################################################    
    def setDt(self, dt):

        """
        Define the time step used for numerical simulations. The filters eta and gamma are interpolated accordingly.
        """
        
        self.dt = dt

    
    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################
    def simulateSpikingResponse(self, I, I_d, dt):
        
        """
        Simulate the spiking response of the TwoComp_passive model to an input current I (nA) with time step dt.
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El.
        """
        self.setDt(dt)
    
        (time, eta_A_sum, spks_times) = self.simulate(I, I_d)
        
        return spks_times



    def simulate(self, I, I_d):
 
        """
        Simulate the spiking response of the TwoComp_passive model to a 
        somatic input current I (nA) and a dendritic input current I_d 
        with time step dt.
        The function returns:
        - time     : ms, support for spks
        - eta_A    : nA, spike triggered adaptive current
        - spks     : ms, list of spike times 
        """
        
        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt
        
        # Model parameters
        p_Tref_ind = int(self.Tref/p_dt)
        p_lambda0   = np.double(self.lambda0)
        
        # Model kernels   
        (p_eta_A_support, p_eta_A) = self.eta_A.getInterpolatedFilter(self.dt)   
        p_eta_A       = p_eta_A.astype('double')
        p_eta_A_l     = len(p_eta_A)
      
        # Define arrays
        spks = np.array(np.zeros(p_T), dtype="double")                      
        eta_A_sum = np.array(np.zeros(p_T + 2*p_eta_A_l), dtype="double")

        # compute convolutions with model kernels
        filtered_I = self.k_s.convolution_ContinuousSignal(I, self.dt)
        filtered_I_d = self.e_ds.convolution_ContinuousSignal(I_d, self.dt)
        
        filtered_currents = filtered_I + filtered_I_d
        
        
        eta_A_sum, spks = cyth.c_simulate_twoComp(p_T, p_dt, p_Tref_ind, p_lambda0, p_eta_A_l, p_eta_A, eta_A_sum, filtered_currents, spks)
        
        # this is a pure python implementation for debugging the code        
        '''
        r = np.random.random_sample(p_T)
        temp_lambda = np.zeros(p_T)
        p_dontspike = np.zeros(p_T)
        
        t = 0
        while t < (p_T-1):
            temp_lambda[t] = p_lambda0 * np.exp(filtered_currents[t] + eta_A_sum[t])
            p_dontspike[t] = np.exp(-temp_lambda[t]*(p_dt/1000.0)) 
            
            if (r[t] > p_dontspike[t]):
                                
                if (t+1 < p_T-1):
                    spks[t+1] = 1.0
                
                t = t + p_Tref_ind                
                
                ## UPDATE ADAPTATION PROCESS     
                eta_A_sum[t+1 : t+1+p_eta_A_l] +=  p_eta_A
            
            t += 1
        '''
        
        time = np.arange(p_T)*self.dt
        
        eta_A_sum   = eta_A_sum[:p_T]     
     
        spks = (np.where(spks==1)[0])*self.dt
        
        
        return (time, eta_A_sum, spks, filtered_currents)
        
    def simulate_deterministicSpikes(self, I, I_d, spks):
        
        
        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt
        
        # Model parameters
        p_Tref_ind = int(self.Tref/p_dt)
        p_lambda0   = np.double(self.lambda0)
        
        # Model kernels   
        (p_eta_A_support, p_eta_A) = self.eta_A.getInterpolatedFilter(self.dt)   
        p_eta_A       = p_eta_A.astype('double')
        p_eta_A_l     = len(p_eta_A)
      
        # Define arrays
        eta_A_sum = np.array(np.zeros(p_T + 2*p_eta_A_l), dtype="double")
        
        spks     = np.array(spks, dtype="double")
        spks_i   = Tools.timeToIndex(spks, self.dt)
        for s in spks_i :
            eta_A_sum[s + 1 + p_Tref_ind  : s + 1 + p_Tref_ind + p_eta_A_l] += p_eta_A
        
        eta_A_sum  = eta_A_sum[:p_T]

        # compute convolutions with model kernels
        filtered_I = self.k_s.convolution_ContinuousSignal(I, self.dt)
        filtered_I_d = self.e_ds.convolution_ContinuousSignal(I_d, self.dt)
        
        filtered_currents = filtered_I + filtered_I_d
        
        V = filtered_currents + eta_A_sum
            
        time = np.arange(p_T)*self.dt
            
        return (time, eta_A_sum, spks, V, filtered_I, filtered_I_d)
        
        
        
    def fit(self, experiment):
        
        """
        Fit the TwoComp_passive model on experimental data.
        The experimental data are stored in the object experiment.
        Only training set traces in experiment are used to perform the fit.
        """
        
        # Three step procedure used for parameters extraction 
        
        print ("\n################################")
        print ("# Fit TwoComp_passive")
        print ("################################\n")

        self.fitModel(experiment)

      
    
    def fitModel(self, experiment):
                        
        self.setDt(experiment.dt)
    
        # Fit a dynamic threshold using a initial condition the result obtained by fitting a static threshold
        
        print ("\nTwoComp_passive MODEL - Fit model...\n")
        
        # Perform fit        
        theta0 = np.concatenate( ( self.k_s.getCoefficients(), self.e_ds.getCoefficients(), self.eta_A.getCoefficients()))
        theta_opt = self.maximizeLikelihood(experiment, theta0, self.buildXmatrix)
        
        # Store result
        self.k_s.setFilter_Coefficients(theta_opt[0:3])
        self.e_ds.setFilter_Coefficients(theta_opt[3:6])
        self.eta_A.setFilter_Coefficients(theta_opt[6:8])

        self.printParameters()
          
        
    def maximizeLikelihood(self, experiment, theta0, buildXmatrix, maxIter=int(1e03), stopCond=1e-06) :
    
        """
        Maximize likelihood. This function can be used to fit any model of the form lambda=exp(Xtheta).
        Here this function is used to fit both:
        - static threshold
        - dynamic threshold
        The difference between the two functions is in the size of theta0 and the returned theta, as well
        as the function buildXmatrix.
        """
        
        # Precompute all the matrices used in the gradient ascent
        all_X        = []
        all_X_spikes = []
        all_sum_X_spikes = []
        
        T_tot = 0.0
        N_spikes_tot = 0.0
        
        traces_nb = 0
        
        for tr in experiment.trainingset_traces:
            
            if tr.useTrace :              
                
                traces_nb += 1
                             
                # Precomputes matrices to perform gradient ascent on log-likelihood
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = buildXmatrix(tr) 
                    
                T_tot        += T
                N_spikes_tot += N_spikes
                    
                all_X.append(X_tmp)
                all_X_spikes.append(X_spikes_tmp)
                all_sum_X_spikes.append(sum_X_spikes_tmp)
        
        logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)

        # Perform gradient ascent

        print ("Maximize log-likelihood (bit/spks)...")
                        
        theta = theta0
        old_L = 1

        for i in range(maxIter) :
            
            learning_rate = 1.0
            
            if i<=10 :                      # be careful in the first iterations (using a small learning rate in the first step makes the fit more stable)
                learning_rate = 0.1
            
            
            L=0; G=0; H=0;  
                
            for trace_i in np.arange(traces_nb):
                (L_tmp,G_tmp,H_tmp) = self.computeLikelihoodGradientHessian(theta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])
                L+=L_tmp; G+=G_tmp; H+=H_tmp;
            
            theta = theta - learning_rate*np.dot(inv(H),G)
                
            if (i>0 and abs((L-old_L)/old_L) < stopCond) :              # If converged
                print ("\nConverged after %d iterations!\n" % (i+1))
                break
            
            old_L = L
            
            # Compute normalized likelihood (for print)
            # The likelihood is normalized with respect to a poisson process and units are in bit/spks
            L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
            print(L_norm, end='\r')
    
        if (i==maxIter - 1) :                                           # If too many iterations
            print ("\nNot converged after %d iterations.\n" % (maxIter))


        return theta
     
        
    def computeLikelihoodGradientHessian(self, theta, X, X_spikes, sum_X_spikes) : 
        
        # IMPORTANT: in general we assume that the lambda_0 = 1 Hz
        # The parameter lambda0 is redundant with Vt_star, so only one of those has to be fitted.
        # We genearlly fix lambda_0 adn fit Vt_star
              
        dt = self.dt/1000.0     # put dt in units of seconds (to be consistent with lambda_0)
        
        X_spikestheta    = np.dot(X_spikes,theta)
        Xtheta           = np.dot(X,theta)
        expXtheta        = np.exp(Xtheta)

        # Compute likelihood (would be nice to improve this to get a normalized likelihood)
        L = np.sum(X_spikestheta) - self.lambda0*dt*np.sum(expXtheta)
                                       
        # Compute gradient
        G = sum_X_spikes - self.lambda0*dt*np.dot(np.transpose(X), expXtheta)
        
        # Compute Hessian
        H = -self.lambda0*dt*np.dot(np.transpose(X)*expXtheta, X)
        
        return (L,G,H)
        
            
    def buildXmatrix(self, tr) :

        """
        Use this function to fit a passive model that fires according to the
        following equation: lambda(t) = lambda0 * exp( k_s*I_s + k_ds*I_d + sum_ t_j_hat eta_A(t-t_j_hat) )
        """
           
        # Get indices be removing absolute refractory periods (-self.dt is to not include the time of spike)       
        selection = tr.getROI_FarFromSpikes(-tr.dt, self.Tref)
        T_l_selection  = len(selection)

            
        # Get spike indices in coordinates of selection   
        spk_train = tr.getSpikeTrain()
        spks_i_afterselection = np.where(spk_train[selection]==1)[0]


        # Compute average firing rate used in the fit   
        T_l = T_l_selection*tr.dt/1000.0                # Total duration of trace used for fit (in s)
        N_spikes = len(spks_i_afterselection)           # Nb of spikes in the trace used for fit
        
        
        # Define X matrix
           
        # Compute and fill columns associated with ...
        # somatic membrane filter k_s (convolution with somatic current)
        X_ks = self.k_s.convolution_ContinuousSignal_basisfunctions(tr.I, tr.dt)
        
        # dendritic current-to-soma filter (convolution with dendritic current)
        X_eds = self.e_ds.convolution_ContinuousSignal_basisfunctions(tr.I_d, tr.dt)
        
        # spike triggered adaptive current
        X_etaA = self.eta_A.convolution_Spiketrain_basisfunctions(tr.getSpikeTimes() + self.Tref, tr.T, tr.dt)
        
        # ... and concatenate them
        X = np.concatenate( (X_ks[selection,:], X_eds[selection,:], X_etaA[selection,:]), axis=1 )
  
        # Precompute other quantities
        X_spikes = X[spks_i_afterselection,:]
        sum_X_spikes = np.sum( X_spikes, axis=0)
        
                
        return (X, X_spikes, sum_X_spikes,  N_spikes, T_l)
 
 
        
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     
  
        
    def plotParameters(self) :
        
        plt.figure(facecolor='white', figsize=(14,4))
        
        # Plot eta_A
        plt.subplot(1,3,1)
        
        (eta_A_support, eta_A) = self.eta_A.getInterpolatedFilter(self.dt) 
        
        plt.plot(eta_A_support, eta_A, color='red', lw=2)
        plt.plot([eta_A_support[0], eta_A_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([eta_A_support[0] - np.size(eta_A)*0.01, eta_A_support[-1] + np.size(eta_A)*0.01])
        plt.ylim([np.min(eta_A), np.max(eta_A)])
        plt.xlabel("Time (ms)")
        plt.ylabel("eta_A (nA)")
        

        # Plot k_s
        plt.subplot(1,3,2)
        
        (k_s_support, k_s) = self.k_s.getInterpolatedFilter(self.dt) 
        
        plt.plot(k_s_support, k_s, color='red', lw=2)
        plt.plot([k_s_support[0], k_s_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([k_s_support[0] - np.size(k_s)*0.01, k_s_support[-1] + np.size(k_s)*0.01])
        plt.ylim([np.min(k_s)*1.1, np.max(k_s)*1.1])
        plt.xlabel("Time (ms)")
        plt.ylabel("k_s (mV)")
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=0.35, hspace=0.25)
        
        # Plot e_ds
        plt.subplot(1,3,3)
        
        (e_ds_support, e_ds) = self.e_ds.getInterpolatedFilter(self.dt) 
        
        plt.plot(e_ds_support, e_ds, color='red', lw=2)
        plt.plot([e_ds_support[0], e_ds_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([e_ds_support[0] - np.size(e_ds)*0.01, e_ds_support[-1] + np.size(e_ds)*0.01])
        plt.ylim([np.min(e_ds)*1.1, np.max(e_ds)*1.1])
        plt.xlabel("Time (ms)")
        plt.ylabel("e_ds (mV)")
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=0.35, hspace=0.25)

        plt.show()
        
    def plotParametersWithBasisfunctions(self) :
        
        plt.figure(facecolor='white', figsize=(14,4))
        
        # Plot eta_A
        plt.subplot(1,3,1)
        
        (eta_A_support, eta_A) = self.eta_A.getInterpolatedFilter(self.dt) 
        
        plt.plot(eta_A_support, eta_A, color='red', lw=2)
        plt.plot([eta_A_support[0], eta_A_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([eta_A_support[0] - np.size(eta_A)*0.01, eta_A_support[-1] + np.size(eta_A)*0.01])
        plt.ylim([np.min(eta_A), np.max(eta_A)])
        plt.xlabel("Time (ms)")
        plt.ylabel("eta_A (nA)")
        

        # Plot k_s
        plt.subplot(1,3,2)
        
        (k_s_support, k_s) = self.k_s.getInterpolatedBasisfunctions(self.dt) 
        
        for basis in k_s:
            plt.plot(k_s_support, basis, lw=2)
        plt.plot([k_s_support[0], k_s_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([k_s_support[0] - np.size(k_s)*0.01, k_s_support[-1] + np.size(k_s)*0.01])
        plt.ylim([np.min(k_s)*1.1, np.max(k_s)*1.1])
        plt.xlabel("Time (ms)")
        plt.ylabel("Basis of k_s (mV)")
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=0.35, hspace=0.25)
        
        # Plot e_ds
        plt.subplot(1,3,3)
        
        (e_ds_support, e_ds) = self.e_ds.getInterpolatedBasisfunctions(self.dt) 
        
        for basis in e_ds:
            plt.plot(e_ds_support, basis, lw=1.5)
        plt.plot([e_ds_support[0], e_ds_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([e_ds_support[0] - np.size(e_ds)*0.01, e_ds_support[-1] + np.size(e_ds)*0.01])
        plt.ylim([np.min(e_ds)*1.1, np.max(e_ds)*1.1])
        plt.xlabel("Time (ms)")
        plt.ylabel("Basis of e_ds (mV)")
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=0.35, hspace=0.25)

        plt.show()
      
      
    def printParameters(self):

        print ("\n-------------------------")       
        print ("TwoComp_passive model parameters:")
        print ("-------------------------")
        print ("Tref (ms):\t%0.3f"   % (self.Tref))
        print ("K_s, Amplitude of short time constant("+str(self.k_s.getTimeConstants()['tau0'])+" ms):\t%0.3f" %(self.k_s.getCoefficients()[0]))
        print ("K_s, Amplitude of medium time constant("+str(self.k_s.getTimeConstants()['tau1'])+" ms):\t%0.3f" %(self.k_s.getCoefficients()[1]))
        print ("K_s, Amplitude of long time constant("+str(self.k_s.getTimeConstants()['tau2'])+" ms):\t%0.3f" %(self.k_s.getCoefficients()[2]))
        print ("E_ds, Amplitude of short time constant("+str(self.e_ds.getTimeConstants()['tau0'])+" ms):\t%0.3f" %(self.e_ds.getCoefficients()[0]))
        print ("E_ds, Amplitude of medium time constant("+str(self.e_ds.getTimeConstants()['tau1'])+" ms):\t%0.3f" %(self.e_ds.getCoefficients()[1]))
        print ("E_ds, Amplitude of long time constant("+str(self.e_ds.getTimeConstants()['tau2'])+" ms):\t%0.3f" %(self.e_ds.getCoefficients()[2]))
        print ("Eta_A, constant :\t%0.3f" %(self.eta_A.getCoefficients()[0]))
        print ("Eta_A, power amplitude:\t%0.3f" %(self.eta_A.getCoefficients()[1]))
        print ("-------------------------\n")
                  

    @classmethod
    def compareModels(cls, TwoComp_passives, labels=None):

        """
        Given a list of TwoComp_passive models, TwoComp_passives, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS        

        print ("\n#####################################")
        print ("TwoComp_passive model comparison")
        print ("#####################################\n")
        
        cnt = 0
        for TwoComp_passive in TwoComp_passives :
            
            print ("Model: " + labels[cnt])          
            TwoComp_passive.printParameters()
            cnt+=1

        print ("#####################################\n")                
                
        # PLOT PARAMETERS
        plt.figure(facecolor='white', figsize=(9,8)) 
               
        colors = plt.cm.jet( np.linspace(0.7, 1.0, len(TwoComp_passives) ) )   
        
        # Membrane filter
        plt.subplot(2,2,1)
            
        cnt = 0
        for TwoComp_passive in TwoComp_passives :
            
            K_support = np.linspace(0,150.0, 1500)             
            K = 1./TwoComp_passive.C*np.exp(-K_support/(TwoComp_passive.C/TwoComp_passive.gl))     
            plt.plot(K_support, K, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  


        # Spike triggered current
        plt.subplot(2,2,2)
            
        cnt = 0
        for TwoComp_passive in TwoComp_passives :
            
            if labels == None :
                label_tmp =""
            else :
                label_tmp = labels[cnt]
            
            (eta_support, eta) = TwoComp_passive.eta.getInterpolatedFilter(0.1)         
            plt.plot(eta_support, eta, color=colors[cnt], lw=2, label=label_tmp)
            cnt += 1
            
        plt.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
        
        if labels != None :
            plt.legend()       
            
        
        plt.xlim([eta_support[0], eta_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Eta (nA)')        
        

        # Escape rate
        plt.subplot(2,2,3)
            
        cnt = 0
        for TwoComp_passive in TwoComp_passives :
            
            V_support = np.linspace(TwoComp_passive.Vt_star-5*TwoComp_passive.DV,TwoComp_passive.Vt_star+10*TwoComp_passive.DV, 1000) 
            escape_rate = TwoComp_passive.lambda0*np.exp((V_support-TwoComp_passive.Vt_star)/TwoComp_passive.DV)                
            plt.plot(V_support, escape_rate, color=colors[cnt], lw=2)
            cnt += 1
          
        plt.ylim([0, 100])    
        plt.plot([V_support[0], V_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
    
        plt.xlim([V_support[0], V_support[-1]])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Escape rate (Hz)')  


        # Spike triggered threshold movememnt
        plt.subplot(2,2,4)
            
        cnt = 0
        for TwoComp_passive in TwoComp_passives :
            
            (gamma_support, gamma) = TwoComp_passive.gamma.getInterpolatedFilter(0.1)         
            plt.plot(gamma_support, gamma, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
      
        plt.xlim([gamma_support[0]+0.1, gamma_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Gamma (mV)')   

        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
        
        plt.show()
    

       
    @classmethod
    def plotAverageModel(cls, TwoComp_passives):

        """
        Average model parameters and plot summary data.
        """
                   
        #######################################################################################################
        # PLOT PARAMETERS
        #######################################################################################################        
        
        fig = plt.figure(facecolor='white', figsize=(16,7))  
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.90, wspace=0.35, hspace=0.5)   
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
       
       
        # MEMBRANE FILTER
        #######################################################################################################
        
        plt.subplot(2,4,1)
                    
        K_all = []
        
        for TwoComp_passive in TwoComp_passives :
                      
            K_support = np.linspace(0,150.0, 300)             
            K = 1./TwoComp_passive.C*np.exp(-K_support/(TwoComp_passive.C/TwoComp_passive.gl))     
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  

       
        # SPIKE-TRIGGERED CURRENT
        #######################################################################################################
    
        plt.subplot(2,4,2)
                    
        K_all = []
        
        for TwoComp_passive in TwoComp_passives :
                
            (K_support, K) = TwoComp_passive.eta.getInterpolatedFilter(0.1)      
   
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]/10.0])
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike-triggered current (nA)')  
 
 
        # SPIKE-TRIGGERED MOVEMENT OF THE FIRING THRESHOLD
        #######################################################################################################
    
        plt.subplot(2,4,3)
                    
        K_all = []
        
        for TwoComp_passive in TwoComp_passives :
                
            (K_support, K) = TwoComp_passive.gamma.getInterpolatedFilter(0.1)      
   
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)   
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        plt.xlim([K_support[0], K_support[-1]])
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike-triggered threshold (mV)')  
 
      
         # R
        #######################################################################################################
    
        plt.subplot(4,6,12+1)
 
        p_all = []
        for TwoComp_passive in TwoComp_passives :
                
            p = 1./TwoComp_passive.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('R (MOhm)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])        
        
        
        # tau_m
        #######################################################################################################
    
        plt.subplot(4,6,18+1)
 
        p_all = []
        for TwoComp_passive in TwoComp_passives :
                
            p = TwoComp_passive.C/TwoComp_passive.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau_m (ms)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
   
        # El
        #######################################################################################################
    
        plt.subplot(4,6,12+2)
 
        p_all = []
        for TwoComp_passive in TwoComp_passives :
                
            p = TwoComp_passive.El
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('El (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
          
        # V reset
        #######################################################################################################
    
        plt.subplot(4,6,18+2)
 
        p_all = []
        for TwoComp_passive in TwoComp_passives :
                
            p = TwoComp_passive.Vr
            p_all.append(p)
        
        print ("Mean Vr (mV): %0.1f" % (np.mean(p_all)))
        
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vr (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
        
        
        # Vt*
        #######################################################################################################
    
        plt.subplot(4,6,12+3)
 
        p_all = []
        for TwoComp_passive in TwoComp_passives :
                
            p = TwoComp_passive.Vt_star
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vt_star (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])    
        
        # Vt*
        #######################################################################################################
    
        plt.subplot(4,6,18+3)
 
        p_all = []
        for TwoComp_passive in TwoComp_passives :
                
            p = TwoComp_passive.DV
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('DV (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])    
