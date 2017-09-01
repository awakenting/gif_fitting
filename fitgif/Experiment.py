import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from .SpikeTrainComparator import SpikeTrainComparator
from .TwoComp_passive import TwoComp_passive
from .Trace import Trace


class Experiment :

    """
    Objects of this class contains the experimental data and an "AEC object" that takes care of Active Electrode Compensation
    """


    def __init__(self, name, cell_type='none', dt=0.1):

        #print ("Create new Experiment")

        self.name               = name          # Experiment name

        self.cell_type          = cell_type     # Cell type as by genetic information (e.g. 'vip','som' or 'pv')

        self.save_path          = 'default_expm'# Relative path to the file storing this experiment


        self.dt                 = dt            # Sampling (all traces in same experiment must have same sampling)

        self.dt_was_set         = False         # True if dt was set already by a previously added trace

        self.AEC_trace          = 0             # Trace object containing voltage and input current used for AEC

        self.trainingset_traces = []            # Traces for training

        self.trainingset_traces_twoComp = []    # Traces for training, that also contain dendritic current

        self.testset_traces     = []            # Traces of test set (typically multiple experiments conducted with frozen noise)

        self.AEC                = 0             # Object that knows how to perform AEC


        self.spikeDetection_threshold    = 0.0  # mV, voltage threshold used to detect spikes

        self.spikeDetection_ref          = 3.0  # ms, absolute refractory period used for spike detection to avoid double counting of spikes



    ############################################################################################
    # FUNCTIONS TO ADD TRACES TO THE EXPERIMENT
    ############################################################################################


    def set_AEC_trace(self, V, V_units, I, I_units, FILETYPE='Igor', verbose=True):
        print ("Set AEC trace...")
        trace_tmp = Trace.get_dt_and_T_from_file(V, V_units, I, I_units, FILETYPE=FILETYPE)
        self.AEC_trace = trace_tmp
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp

    def set_AEC_trace_with_time_units(self, V, V_units, I, I_units, T, dt, FILETYPE='Igor', verbose=True):
        print ("Set AEC trace...")
        trace_tmp = Trace(V, V_units, I, I_units, T, dt, FILETYPE=FILETYPE)
        self.AEC_trace = trace_tmp
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp


    def add_trainingset_trace(self, V, V_units, I, I_units, FILETYPE='Igor', verbose=True):
        #print ("Add Training Set trace...")
        trace_tmp = Trace.get_dt_and_T_from_file( V, V_units, I, I_units, FILETYPE=FILETYPE)
        self.trainingset_traces.append(trace_tmp)
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp

    def add_trainingset_trace_with_time_units(self, V, V_units, I, I_units, T, dt, FILETYPE='Igor', verbose=True):
        #print ("Add Training Set trace...")
        trace_tmp = Trace(V, V_units, I, I_units, T, dt, FILETYPE=FILETYPE)
        self.trainingset_traces.append(trace_tmp)
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp

    def add_trainingset_trace_for_two_comp (self, V, V_units, I, I_units, V_d, V_d_units, I_d,
                                            I_d_units, T, FILETYPE='Igor', verbose=True):
        #print ("Add Two Compartments Training Set trace ...")
        trace_tmp = Trace( V, V_units, I, I_units, T, self.dt, FILETYPE=FILETYPE, V_d = V_d,
                          V_d_units = V_d_units, I_d = I_d, I_d_units = I_d_units)
        self.trainingset_traces.append(trace_tmp)
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp


    def add_testset_trace(self, V, V_units, I, I_units, FILETYPE='Igor', verbose=True):
        #print ("Add Test Set trace...")
        trace_tmp = Trace.get_dt_and_T_from_file(V, V_units, I, I_units, FILETYPE=FILETYPE)
        self.testset_traces.append(trace_tmp)
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp

    def add_testset_trace_with_time_units(self, V, V_units, I, I_units, T, dt, FILETYPE='Igor', verbose=True):
        #print ("Add Test Set trace...")
        trace_tmp = Trace(V, V_units, I, I_units, T, dt, FILETYPE=FILETYPE)
        self.testset_traces.append(trace_tmp)
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp

    def add_testset_trace_for_two_comp (self, V, V_units, I, I_units, V_d, V_d_units, I_d,
                                        I_d_units, T, FILETYPE='Igor', verbose=True):
        #print ("Add Two Compartments Test Set trace ...")
        trace_tmp = Trace( V, V_units, I, I_units, T, self.dt, FILETYPE=FILETYPE, V_d = V_d,
                          V_d_units = V_d_units, I_d = I_d, I_d_units = I_d_units)
        self.testset_traces.append(trace_tmp)
        self.check_dt_value_consistency(trace_tmp, verbose=verbose)

        return trace_tmp

    def mergeTrainingTraces(self):
        if not len(self.trainingset_traces) == 0:
            mergedV = np.zeros(0)
            mergedI = np.zeros(0)
            ROIs = []
            transition_length = 10
            for tr in self.trainingset_traces :
                tempV = tr.V
                tempI = tr.I

                # store time intervals of single traces to set appropriate ROI later
                tempV_T = tr.T
                tempV_dt = tr.dt
                next_ROI_start = (len(mergedV) + transition_length)*self.dt
                ROIs.append([next_ROI_start, next_ROI_start + tempV_T])

                # merge traces
                mergedV = np.concatenate((mergedV, np.ones(transition_length)*tempV[0],tempV))
                mergedI = np.concatenate((mergedI, np.ones(transition_length)*tempI[0],tempI))

            self.trainingset_traces = []
            self.add_trainingset_trace_with_time_units(mergedV, 10**-3, mergedI, 10**-9,
                                                       T = int(len(mergedV)*tempV_dt), dt = tempV_dt,
                                                       FILETYPE = 'Array')

            self.trainingset_traces[0].setROI(ROIs)

    def mergeTestTraces(self):
        if not len(self.testset_traces) == 0:
            mergedV = np.zeros(0)
            mergedI = np.zeros(0)
            ROIs = []
            transition_length = 10
            for tr in self.testset_traces :
                tempV = tr.V
                tempI = tr.I

                # store time intervals of single traces to set appropriate ROI later
                tempV_T = tr.T
                tempV_dt = tr.dt
                next_ROI_start = (len(mergedV) + transition_length)*self.dt
                ROIs.append([next_ROI_start, next_ROI_start + tempV_T])

                # merge traces
                mergedV = np.concatenate((mergedV, np.ones(transition_length)*tempV[0],tempV))
                mergedI = np.concatenate((mergedI, np.ones(transition_length)*tempI[0],tempI))

            self.testset_traces = []

    #===============================================================================================
    # ATTENTION !! I add the SAME trace three times because the calculation of MD* requires at least
    #                two test traces
    #===============================================================================================

            print('Adding the SAME test trace three times...')
            n_repeat = 3
            for i in range(n_repeat):
                self.add_testset_trace_with_time_units(mergedV, 10**-3, mergedI, 10**-9,
                                                       T = int(len(mergedV)*tempV_dt), dt = tempV_dt,
                                                       FILETYPE = 'Array')
                self.testset_traces[i].setROI(ROIs)



    ############################################################################################
    # FUNCTIONS ASSOCIATED WITH ACTIVE ELECTRODE COMPENSATION
    ############################################################################################
    def setAEC(self, AEC):

        self.AEC = AEC


    def getAEC(self):

        return self.AEC


    def performAEC(self):

        self.AEC.performAEC(self)


    ############################################################################################
    # FUNCTIONS FOR SAVING AND LOADING AN EXPERIMENT
    ############################################################################################
    def save(self, path, verbose=True):
        """
        Save Experiment object with pickle.

        Parameters:
        - path: The path to the folder where you want to save the file. Resulting full path of
                the file will be path/Experiment_Experimentname.pkl where Experimentname
                is the .name attribute of the Experiment object. Load the file with
                expm = Experiment.load('path/Experiment_Experimentname.pkl')
        """

        filename = path + "Experiment_" + self.name + '.pkl'
        self.save_path = filename
        if verbose:
            print ("Saving: " + filename + "...")
        f = open(filename,'wb')
        pkl.dump(self, f)
        if verbose:
            print ("Done!")


    @classmethod
    def load(cls, filename, verbose=True):
        if verbose:
            print ("Load experiment: " + filename + "...")

        f = open(filename,'rb')
        expr = pkl.load(f)

        if verbose:
            print ("Done!" )

        return expr


    ############################################################################################
    # EVALUATE PERFORMANCES OF A MODEL
    ############################################################################################
    def predictSpikes(self, spiking_model, nb_rep=500):

        # Collect spike times in test set

        all_spks_times_testset = []

        for tr in self.testset_traces:

            if tr.useTrace :

                spks_times = tr.getSpikeTimes()
                all_spks_times_testset.append(spks_times)


        # Predict spike times using model
        T_test = self.testset_traces[0].T
        I_test = self.testset_traces[0].I

        if isinstance(spiking_model, TwoComp_passive):
            I_d_test = self.testset_traces[0].I_d

        all_spks_times_prediction = []

        print ("Predict spike times...")

        for rep in np.arange(nb_rep) :
            print( "Progress: %2.1f %% \r" % (100*(rep+1)/nb_rep), end='\r'),
            if isinstance(spiking_model, TwoComp_passive):
                spks_times = spiking_model.simulateSpikingResponse(I_test, I_d_test, self.dt)
            else:
                spks_times = spiking_model.simulateSpikingResponse(I_test, self.dt)

            all_spks_times_prediction.append(spks_times)

        #print

        prediction = SpikeTrainComparator(T_test, all_spks_times_testset, all_spks_times_prediction)

        return prediction



    ############################################################################################
    # AUXILIARY FUNCTIONS
    ############################################################################################
    def detectSpikes_python(self, threshold=0.0, ref=3.0, verbose=True):

        # implement here a function that detects all the spkes in all the traces...
        # parameters of spike detection should be set in this class and not in trace
        
        if verbose:
            print( "Detect spikes!")

        self.spikeDetection_threshold = threshold
        self.spikeDetection_ref = ref

        if self.AEC_trace != 0 :
            self.AEC_trace.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)

        for tr in self.trainingset_traces :
            tr.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)

        for tr in self.testset_traces :
            tr.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)

        if verbose:
            print ("Done!")


    def detectSpikes_cython(self, threshold=0.0, ref=3.0, verbose=True):

        # implement here a function that detects all the spkes in all the traces...
        # parameters of spike detection should be set in this class and not in trace
        if verbose:
            print ("Detect spikes!")

        self.spikeDetection_threshold = threshold
        self.spikeDetection_ref = ref

        if self.AEC_trace != 0 :
            self.AEC_trace.detectSpikes_cython(self.spikeDetection_threshold, self.spikeDetection_ref)

        for tr in self.trainingset_traces :
            tr.detectSpikes_cython(self.spikeDetection_threshold, self.spikeDetection_ref)

        for tr in self.testset_traces :
            tr.detectSpikes_cython(self.spikeDetection_threshold, self.spikeDetection_ref)
        
        if verbose:
            print( "Done!")


    def getTrainingSetNb(self):

        return len(self.trainingset_traces)

    def check_dt_value_consistency(self, trace_tmp, verbose=True):
        """
        Check if a dt value was already set for this Experiment and if the dt value of the given
        trace is consistent with it.
        """
        if self.dt_was_set:
            if self.dt != trace_tmp.dt:
                if verbose:
                    print('Warning: You added a training trace that has a different dt [' + str(trace_tmp.dt) + \
                          '] than already existing traces [' + str(self.dt) + '] .')
        else:
            self.dt = trace_tmp.dt
            self.dt_was_set = True

    def get_estimated_input_amps(self):
        """
        Get the estimated current amplitudes of the input currents in IDRest trainingset traces.
        """
        amps = []
        for tr in self.trainingset_traces:
            if not tr.estimated_input_amp is None:
                amps.append(tr.estimated_input_amp)
        return amps

    def get_recording_types(self):
        """
        Get the recording types of trainingset traces.
        """
        rtypes = []
        for tr in self.trainingset_traces:
            rtypes.append(tr.recording_type)
        return rtypes

    def compute_fi_curve(self, by='spike_count', initial_cut=100):
        '''
        Compute the F-I curve of this cell by calculating F-I points from IDRest traningset traces.
        '''
        idrest_traces = [tr for tr in self.trainingset_traces if tr.recording_type=='IDRest' and tr.T > 2000]
        i_values = []
        f_values = []
        if not len(idrest_traces)==0:
            for tr in idrest_traces:
                f_values.append(tr.getFiringRate_inROI(by=by, initial_cut=initial_cut))
                i_values.append(tr.get_estimated_input_amp())

        return i_values, f_values


    ############################################################################################
    # FUNCTIONS FOR PLOTTING
    ############################################################################################
    def plotTrainingSet(self):

        plt.figure(figsize=(12,8), facecolor='white')

        cnt = 0

        for tr in self.trainingset_traces :

            # Plot input current
            plt.subplot(2*self.getTrainingSetNb(),1,cnt*2+1)
            plt.plot(tr.getTime(), tr.I, 'gray')

            # Plot ROI
            ROI_vector = -10.0*np.ones(int(tr.T/tr.dt))
            if tr.useTrace :
                ROI_vector[tr.getROI()] = 10.0

            plt.fill_between(tr.getTime(), ROI_vector, 10.0, color='0.2')

            plt.ylim([min(tr.I)-0.5, max(tr.I)+0.5])
            plt.ylabel("I (nA)")
            plt.xticks([])

            # Plot membrane potential
            plt.subplot(2*self.getTrainingSetNb(),1,cnt*2+2)
            plt.plot(tr.getTime(), tr.V_rec, 'black')

            if tr.AEC_flag :
                plt.plot(tr.getTime(), tr.V, 'blue')


            if tr.spks_flag :
                plt.plot(tr.getSpikeTimes(), np.zeros(tr.getSpikeNb()), '.', color='red')

            # Plot ROI
            ROI_vector = -100.0*np.ones(int(tr.T/tr.dt))
            if tr.useTrace :
                ROI_vector[tr.getROI()] = 100.0

            plt.fill_between(tr.getTime(), ROI_vector, 100.0, color='0.2')

            plt.ylim([min(tr.V)-5.0, max(tr.V)+5.0])
            plt.ylabel("Voltage (mV)")

            cnt +=1

        plt.xlabel("Time (ms)")

        plt.subplot(2*self.getTrainingSetNb(),1,1)
        plt.title('Experiment ' + self.name + " - Training Set (dark region not selected)")
        plt.subplots_adjust(left=0.10, bottom=0.07, right=0.95, top=0.92, wspace=0.25, hspace=0.25)

        plt.show()


    def plotTestSet(self):

        plt.figure(figsize=(12,6), facecolor='white')

        # Plot  test set currents
        plt.subplot(3,1,1)

        for tr in self.testset_traces :
            plt.plot(tr.getTime(), tr.I, 'gray')
        plt.ylabel("I (nA)")
        plt.title('Experiment ' + self.name + " - Test Set")
        # Plot  test set voltage
        plt.subplot(3,1,2)
        for tr in self.testset_traces :
            plt.plot(tr.getTime(), tr.V, 'black')
        plt.ylabel("Voltage (mV)")

        # Plot test set raster
        plt.subplot(3,1,3)

        cnt = 0
        for tr in self.testset_traces :
            cnt += 1
            if tr.spks_flag :
                plt.plot(tr.getSpikeTimes(), cnt*np.ones(tr.getSpikeNb()), '|', color='black', ms=5, mew=2)

        plt.yticks([])
        plt.ylim([0, cnt+1])
        plt.xlabel("Time (ms)")

        plt.subplots_adjust(left=0.10, bottom=0.07, right=0.95, top=0.92, wspace=0.25, hspace=0.25)

        plt.show()