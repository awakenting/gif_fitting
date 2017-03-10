import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from numpy.linalg import inv

from .GIF import GIF
from .Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from . import Tools

plt.style.use('ggplot')
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.facecolor'] = 'white'

class GIF_with_animation(GIF) :

    """
    Generalized Integrate and Fire model
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
    
    
    where the membrane potential dynamics is given by:
    
    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)
    
    
    and the firing threshold V_T is given by:
    
    V_T = Vt_star + sum_j gamma(t-\hat t_j)
    
    
    and \hat t_j denote the spike times.
    """

    def __init__(self, dt=0.1):
                   
        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)
  
        # Define model parameters
        
        self.gl      = 1.0/100.0        # nS, leak conductance
        self.C       = 20.0*self.gl     # nF, capacitance
        self.El      = -65.0            # mV, reversal potential
        
        self.Vr      = -50.0            # mV, voltage reset
        self.Tref    = 4.0              # ms, absolute refractory period
        
        self.Vt_star = -48.0            # mV, steady state voltage threshold VT*
        self.DV      = 0.5              # mV, threshold sharpness
        self.lambda0 = 1.0              # by default this parameter is always set to 1.0 Hz
        
        
        self.eta     = Filter_Rect_LogSpaced()    # nA, spike-triggered current (must be instance of class Filter)
        self.gamma   = Filter_Rect_LogSpaced()    # mV, spike-triggered movement of the firing threshold (must be instance of class Filter)
        self.gamma_temp   = Filter_Rect_LogSpaced()
        
        # Varialbes relatd to fit
        
        self.avg_spike_shape = 0
        self.avg_spike_shape_support = 0
        
        
        # Initialize the spike-triggered current eta with an exponential function        
        
        def expfunction_eta(x):
            return 0.2*np.exp(-x/100.0)
        
        #self.eta.setFilter_Function(expfunction_eta)


        # Initialize the spike-triggered current gamma with an exponential function        
        
        def expfunction_gamma(x):
            return 10.0*np.exp(-x/100.0)
        
        #self.gamma.setFilter_Function(expfunction_gamma)        
        #self.gamma_temp.setFilter_Function(expfunction_gamma)
              
            

          
        
    def maximizeLikelihood(self, experiment, beta0, buildXmatrix, do_animation, maxIter=10**3, stopCond=10**-6) :
    
        """
        Maximize likelihood. This function can be used to fit any model of the form lambda=exp(Xbeta).
        Here this function is used to fit both:
        - static threshold
        - dynamic threshold
        The difference between the two functions is in the size of beta0 and the returned beta, as well
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
                
                # Simulate subthreshold dynamics 
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
                             
                # Precomputes matrices to perform gradient ascent on log-likelihood
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = buildXmatrix(tr, V_est) 
                    
                T_tot        += T
                N_spikes_tot += N_spikes
                    
                all_X.append(X_tmp)
                all_X_spikes.append(X_spikes_tmp)
                all_sum_X_spikes.append(sum_X_spikes_tmp)
        
        logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)

        # Perform gradient ascent

        print ("Maximize log-likelihood (bit/spks)...")
                        
        beta = beta0
        old_L = 1
        
        beta_history = []

        for i in range(maxIter) :
            
            learning_rate = 1.0
            
            if i<=10 :                      # be careful in the first iterations (using a small learning rate in the first step makes the fit more stable)
                learning_rate = 0.1
            
            
            L=0; G=0; H=0;  
                
            for trace_i in np.arange(traces_nb):
                (L_tmp,G_tmp,H_tmp) = self.computeLikelihoodGradientHessian(beta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])
                L+=L_tmp; G+=G_tmp; H+=H_tmp;
            
            beta = beta - learning_rate*np.dot(inv(H),G)
            
            if do_animation:
                # save figures for temporary shape of gamma
                beta_history.append(beta)
                
            if (i>0 and abs((L-old_L)/old_L) < stopCond) :              # If converged
                print ("\nConverged after %d iterations!\n" % (i+1))
                break
            
            old_L = L
            
            # Compute normalized likelihood (for print)
            # The likelihood is normalized with respect to a poisson process and units are in bit/spks
            L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
            # the keyword end replaces the default end character ('\n') by another one,
            # '\r' brings the cursor to the first position of the line
            print(L_norm, end='\r') 
            
        if do_animation:
            self.save_temp_figs(beta_history,experiment.trainingset_traces[0], force_spikes=False)
            self.save_temp_figs(beta_history,experiment.trainingset_traces[0], force_spikes=True)
    
        if (i==maxIter - 1) :                                           # If too many iterations
            print("\nNot converged after %d iterations.\n" % (maxIter))


        return beta
    
 
 
        
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     

        
    def save_temp_figs(self, beta_history, trace, force_spikes=False):
        '''
        This is a helper function, only used for the animation of the fitting procedure.
        It saves figures of the temporary shape of gamma, the simulated voltage
        trace and lambda.
        '''
        print('Generate figures for animation ...')
        trace_T = len(trace.I)
        real_spks_i = trace.getSpikeIndices()
        
        # time window in which the trace shall be plotted (in timesteps)
        t_start, t_end = 0, 5000
        
        # find limits for y-axis:
        ymax, ymin, DV_ymax, lambda_ymax = 0, 0, 0, []
        for i, beta_temp in enumerate(beta_history):
            DV_temp      = 1.0/beta_temp[0]
            Vt_star_temp = -beta_temp[1]*DV_temp
            self.gamma_temp.setFilter_Coefficients(-beta_temp[2:]*DV_temp)
            DV_ymax = np.max([DV_ymax,DV_temp])
            
            # compute current filter and update max and min values            
            (gamma_support, gamma) = self.gamma_temp.getInterpolatedFilter(self.dt)
            ymax = np.max(np.concatenate(([ymax],gamma)))
            ymin = np.min(np.concatenate(([ymin],gamma)))
            
            if force_spikes:
                if i==0: # since the subthreshold dynamics don't change this computation is only needed once
                    (time, V, eta_sum) = self.simulateDeterministic_forceSpikes(trace.I, trace.V[0], trace.getSpikeTimes())
                Tref_i    = int(self.Tref/self.dt)
                gamma_l     = len(gamma)
                gamma_sum  = np.zeros(trace_T + 1.1*gamma_l + Tref_i)
                for s in real_spks_i :
                    gamma_sum[s + 1 + Tref_i  : s + 1 + Tref_i + gamma_l] += gamma                
                gamma_sum  = gamma_sum[:trace_T]
                V_T = gamma_sum + Vt_star_temp
            else:
                self.gamma.setFilter_Coefficients(-beta_temp[2:]*DV_temp)
                (time, V, eta_sum, V_T, spks) = self.simulate(trace.I, trace.V[0])
                
            # cumpute lambda and update max value
            
            lambda_ = np.exp( (V - V_T) / DV_temp )
            if force_spikes:
                lambda_ymax.append(np.mean(lambda_[t_start:t_end]))
            else:
                lambda_ymax.append(np.max(lambda_[t_start:t_end]))
        
        if force_spikes:
            lambda_ymax = np.min(np.array(lambda_ymax))
        else:
            lambda_ymax = np.mean(np.array(lambda_ymax))
            
            
        
        # generate figures
        previous_beta = beta_history[0]
        for i, beta_temp in enumerate(beta_history):
            print('Figure ' + str(i+1) + ' of ' + str(len(beta_history)),end='\r')
            
            fig = plt.figure(figsize=(16,12))
            
            ### Plot current shape of gamma in comparison to previous one ###
            
            
            plt.subplot2grid((3,3), (0, 0), colspan=2)
            # plot previous gamma in gray
            DV_temp      = 1.0/previous_beta[0]
            self.gamma_temp.setFilter_Coefficients(-previous_beta[2:]*DV_temp)
            (gamma_support, gamma) = self.gamma_temp.getInterpolatedFilter(self.dt) 
            
            plt.plot(gamma_support, gamma, color='0.5', lw=2) # 0.5 defines a shade of 50% of gray
            
            # plot current gamma in red
            DV_temp      = 1.0/beta_temp[0]
            self.gamma_temp.setFilter_Coefficients(-beta_temp[2:]*DV_temp)
            (gamma_support, gamma) = self.gamma_temp.getInterpolatedFilter(self.dt) 
            
            plt.plot(gamma_support, gamma, color='red', lw=2)
            
            
            plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2)                
            plt.xlim([gamma_support[0], gamma_support[-1]])
            plt.ylim((ymin, ymax))
            plt.xlabel("Time (ms)")
            plt.ylabel("Gamma (mV)")
            plt.title('$\gamma (t)$')
            
            
            ### Plot value of DV ###
            
            plt.subplot2grid((3,3), (0, 2))
            plt.bar(0.25,DV_temp,width=0.5)
            plt.xlim(0,1)
            plt.ylim(0,DV_ymax)
            plt.xticks([0.5],['DV'])
            plt.ylabel('DV (mV)')
            plt.title('Value of DV')
            
            
            
            ### Plot simulated voltage according to current fit of gamma ###
            plt.subplot2grid((3,3), (1, 0), colspan=3)
            Vt_star_temp = -beta_temp[1]*DV_temp
            self.gamma.setFilter_Coefficients(-beta_temp[2:]*DV_temp)
            
            if force_spikes:
                if i==0: # since the subthreshold dynamics don't change this computation is only needed once
                    (time, V, eta_sum) = self.simulateDeterministic_forceSpikes(trace.I, trace.V[0], trace.getSpikeTimes())
                Tref_i    = int(self.Tref/self.dt)
                gamma_l     = len(gamma)
                gamma_sum  = np.zeros(trace_T + 1.1*gamma_l + Tref_i)
                for s in real_spks_i :
                    gamma_sum[s + 1 + Tref_i  : s + 1 + Tref_i + gamma_l] += gamma                
                gamma_sum  = gamma_sum[:trace_T]
                V_T = gamma_sum + Vt_star_temp
            else:
                (time, V, eta_sum, V_T, spks) = self.simulate(trace.I, trace.V[0])
                spks_i = Tools.timeToIndex(spks, self.dt)

            # plot voltage for the first 500 ms:
            
            v_ymin = np.min(trace.V)
            v_ymax = np.max(trace.V)
            # plot real and simulated voltage trace
            plt.plot(time[t_start:t_end], V[t_start:t_end], color='r', label='simulated V(t)')
            plt.plot(time[t_start:t_end], trace.V[t_start:t_end], color='k', label='true V(t)')
            plt.ylim((v_ymin,v_ymax))            
            
            # plot firing threshold trace
            plt.plot(time[t_start:t_end], V_T[t_start:t_end], color='g', label='V_T(t)')
            
            # plot real (and simulated spikes if not forced)
            if not force_spikes:
                plt.vlines(time[spks_i[(spks_i > t_start) & (spks_i < t_end)]], ymin= v_ymin, ymax= v_ymax, colors='r')
            
            plt.vlines(time[real_spks_i[(real_spks_i > t_start) & (real_spks_i < t_end)]], ymin= v_ymin, ymax= v_ymax, colors='k') 
            
            #plt.xlabel("Time (ms)")
            plt.ylabel("Membrane voltage (mV)")
            plt.legend()
            plt.title('Membrange voltage')

            
            
            ### Plot lambda(t), i.e. the firing intensity in Hz ###
            plt.subplot2grid((3,3), (2, 0), colspan=3)
            
            
            lambda_ = np.exp( (V - V_T) / DV_temp )
            plt.plot(time[t_start:t_end], lambda_[t_start:t_end], color='k', label='lambda(t)')
            plt.vlines(time[real_spks_i[(real_spks_i > t_start) & (real_spks_i < t_end)]], ymin= 0, ymax= lambda_ymax, linestyles='dotted', colors='k', label='true spikes')
            plt.xlabel("Time (ms)")
            plt.ylabel("Firing intensity (Hz)")
            plt.ylim((0,lambda_ymax))
            plt.legend()
            plt.title('$\lambda (t)$')
            
            
            ### Save figure ###
            plt.tight_layout()
            
            if force_spikes:
                plt.savefig('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/images/fit_animation_forcedSpikes_'+str(int((t_end-t_start)*self.dt))+'ms_'+str(i)+'.png')
            else:
                plt.savefig('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/images/fit_animation_stochasticSpikes_'+str(int((t_end-t_start)*self.dt))+'ms_'+str(i)+'.png')
            plt.close(fig)
            
            
            previous_beta = beta_temp
        
