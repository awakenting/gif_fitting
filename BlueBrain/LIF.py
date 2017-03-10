#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:06:21 2016

@author: andrej
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy.linalg import inv

from ThresholdModel import ThresholdModel
from Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
import Tools
from Tools import reprint
import cython_helpers as cyth


class LIF(ThresholdModel) :

    """
    Leaky Integrate and Fire model
    Spike are produced deterministically :

    t_spike = t, V(t)=V_threshold


    where the membrane potential dynamics is given by:

    C dV/dt = -gl(V-El) + I

    """

    def __init__(self, dt=0.1):

        super(ThresholdModel, self).__init__()

        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)

        # Define model parameters

        self.gl      = 1.0/100.0        # nS, leak conductance
        self.C       = 20.0*self.gl     # nF, capacitance
        self.El      = -65.0            # mV, reversal potential

        self.Vr      = self.El          # mV, voltage reset
        self.Vt      = None              # mV, spiking threshold
        self.Tref    = 4.0              # ms, absolute refractory period

        # Varialbes relatd to fit

        self.fitted = False
        self.avg_spike_shape = 0
        self.avg_spike_shape_support = 0

        self.expm_file      = []              # filename of the experiment object, with the data that was fitted
        self.pred           = []              # prediction object with simulated traces and spike trains

        self.var_explained_dV  = None
        self.var_explained_V   = None
        self.mean_se_dV = 0
        self.mean_se_V = 0


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
    def simulateSpikingResponse(self, I, dt):

        """
        Simulate the spiking response of the LIF model to an input current I (nA) with time step dt.
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El.
        """
        self.setDt(dt)

        (time, V, spks_times) = self.simulate(I, self.El)

        return spks_times


    def simulateVoltageResponse(self, I, dt) :

        self.setDt(dt)

        (time, V, spks_times) = self.simulate(I, self.El)

        return (spks_times, V)


    def simulate(self, I, V0):

        """
        Simulate the spiking response of the LIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
            - time     : ms, support for V, eta_sum, V_T, spks
            - V        : mV, membrane potential
            - spks     : ms, list of spike times
        """

        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt

        # Model parameters
        p_gl        = self.gl
        p_C         = self.C
        p_El        = self.El
        p_Vr        = self.Vr
        p_Tref      = self.Tref
        p_Vt       = self.Vt

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        spks = np.array(np.zeros(p_T), dtype="double")

        # Set initial condition
        V[0] = V0

        # computational intensive part is calculated by cython function
        V, spks = cyth.c_simulate_lif(p_T, p_dt, p_gl, p_C, p_El, p_Vr, p_Tref, p_Vt, V, I, spks)

        time = np.arange(p_T)*self.dt

        spks = (np.where(spks==1)[0])*self.dt

        return (time, V, spks)


    def fit(self, experiment, DT_beforeSpike = 5.0):

        """
        Fit the LIF model on experimental data.
        The experimental data are stored in the object experiment.
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """

        self.fitVoltageReset(experiment, self.Tref, do_plot=False)

        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike, do_plot=False)



    ########################################################################################################
    # FIT VOLTAGE RESET GIVEN ABSOLUTE REFRACOTORY PERIOD (step 1)
    ########################################################################################################
    def fitVoltageReset(self, experiment, Tref, do_plot=False):

        """
        Tref: ms, absolute refractory period.
        The voltage reset is estimated by computing the spike-triggered average of the voltage.
        """

        # Fix absolute refractory period
        self.dt = experiment.dt
        self.Tref = Tref

        all_spike_average = []
        all_spike_thr = []
        all_spike_nb = 0
        for tr in experiment.trainingset_traces :

            if tr.useTrace :
                if len(tr.spks) > 0 :
                    (support, spike_average, spike_nb, avg_thr) = tr.compute_average_spike_features()
                    all_spike_average.append(spike_average)
                    all_spike_thr.append(avg_thr)
                    all_spike_nb += spike_nb
                else:
                    self.fitted = False
                    return

        spike_average = np.mean(all_spike_average, axis=0)

        # Estimate voltage reset
        Tref_ind = np.where(support >= self.Tref)[0][0]
        self.Vr = spike_average[Tref_ind]

        # Estimate spiking threshold
        self.Vt = np.mean(all_spike_thr)

        # Save average spike shape
        self.avg_spike_shape = spike_average
        self.avg_spike_shape_support = support
        
        self.fitted = True

        if do_plot :
            plt.figure()
            plt.plot(support, spike_average, 'black')
            plt.plot([support[Tref_ind]], [self.Vr], '.', color='red')
            plt.show()



    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################


    def fitSubthresholdDynamics(self, experiment, DT_beforeSpike=5.0, do_plot=False):
        if not self.fitted:
            return
        self.dt = experiment.dt

        # Build X matrix and Y vector to perform linear regression (use all traces in training set)
        X = []
        Y = []

        cnt = 0

        for tr in experiment.trainingset_traces :

            if tr.useTrace :

                cnt += 1

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


        # Linear Regression
        XTX     = np.dot(np.transpose(X), X)
        XTX_inv = inv(XTX)
        XTY     = np.dot(np.transpose(X), Y)
        b       = np.dot(XTX_inv, XTY)
        b       = b.flatten()


        # Update and print model parameters
        self.C  = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl

        # Compute percentage of variance explained on dV/dt
        mse_dV = np.mean((Y - np.dot(X,b))**2)
        var_explained_dV = 1.0 - mse_dV/np.var(Y)
        self.mean_se_dV = mse_dV
        self.var_explained_dV = var_explained_dV
        
        # Compute percentage of variance explained on V

        SSE = 0     # sum of squared errors
        MSE = 0     # mean squared error
        VAR = 0     # variance of data

        for tr in experiment.trainingset_traces :

            if tr.useTrace :

                # Simulate subthreshold dynamics
                (time, V_est, spks_est) = self.simulate(tr.I, tr.V[0])

                if do_plot:
                    plt.figure()
                    plt.plot(time[0:],tr.V,hold=True)
                    plt.plot(time[0:],V_est)

                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)

                SSE += np.sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                MSE += np.mean((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])

        self.var_explained_V = 1.0 - SSE / VAR
        self.mean_se_V = MSE

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, DT_beforeSpike=5.0):

        # Select region where to perform linear regression
        selection = trace.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)
        selection_l = len(selection)
        self.subthreshold_time = selection_l*self.dt

        # Build X matrix for linear regression
        X = np.zeros( (selection_l, 3) )

        # Fill first two columns of X matrix
        X[:,0] = trace.V[selection]
        X[:,1] = trace.I[selection]
        X[:,2] = np.ones(selection_l)

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

    def compute_var_explained(self, experiment, DT_beforeSpike=5.0):

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
        b = np.empty(3+len(eta_coeffs))

        b[0] = -self.gl/self.C
        b[1] = 1/self.C
        b[2] = self.gl*self.El/self.C
        b[3:] = eta_coeffs

        # Compute percentage of variance explained on dV/dt

        self.var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
        self.mean_se_dV = np.mean((Y - np.dot(X,b))**2)


    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################


    def plotParameters(self) :

        plt.figure(facecolor='white', figsize=(14,4))

        # Plot kappa

        K_support = np.linspace(0,150.0, 300)
        K = 1./self.C*np.exp(-K_support/(self.C/self.gl))

        plt.plot(K_support, K, color='red', lw=2)
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2)

        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane filter (MOhm/ms)")

        plt.show()


    def printParameters(self):

        print ("\n-------------------------")
        print ("LIF model parameters:")
        print ("-------------------------")
        print ("tau_m (ms):\t%0.3f"  % (self.C/self.gl))
        print ("R (MOhm):\t%0.3f"    % (1.0/self.gl))
        print ("C (nF):\t\t%0.3f"    % (self.C))
        print ("gl (nS):\t%0.3f"     % (self.gl))
        print ("El (mV):\t%0.3f"     % (self.El))
        print ("Tref (ms):\t%0.3f"   % (self.Tref))
        print ("Vr (mV):\t%0.3f"     % (self.Vr))
        print ("-------------------------\n")


    @classmethod
    def compareModels(cls, LIFs, labels=None):

        """
        Given a list of LIF models, LIFs, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS

        print ("\n#####################################")
        print ("LIF model comparison")
        print ("#####################################\n")

        cnt = 0
        for LIF in LIFs :

            print ("Model: " + labels[cnt])
            LIF.printParameters()
            cnt+=1

        print ("#####################################\n")

        # PLOT PARAMETERS
        plt.figure(facecolor='white', figsize=(9,8))

        colors = plt.cm.jet( np.linspace(0.7, 1.0, len(LIFs) ) )

        # Membrane filter

        cnt = 0
        for LIF in LIFs :

            K_support = np.linspace(0,150.0, 1500)
            K = 1./LIF.C*np.exp(-K_support/(LIF.C/LIF.gl))
            plt.plot(K_support, K, color=colors[cnt], lw=2)
            cnt += 1

        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)

        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')



    @classmethod
    def plotAverageModel(cls, LIFs):

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

        plt.subplot(2,3,1)

        K_all = []

        for LIF in LIFs :

            K_support = np.linspace(0,150.0, 300)
            K = 1./LIF.C*np.exp(-K_support/(LIF.C/LIF.gl))
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


         # R
        #######################################################################################################

        plt.subplot(2,3,1)

        p_all = []
        for LIF in LIFs :

            p = 1./LIF.gl
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('R (MOhm)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])


        # tau_m
        #######################################################################################################

        plt.subplot(2,3,1)

        p_all = []
        for LIF in LIFs :

            p = LIF.C/LIF.gl
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau_m (ms)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])


        # El
        #######################################################################################################

        plt.subplot(2,3,1)

        p_all = []
        for LIF in LIFs :

            p = LIF.El
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('El (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])

