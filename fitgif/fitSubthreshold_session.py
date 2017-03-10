(X_tmp, Y_tmp) = myGIF_pow.fitSubthresholdDynamics_Build_Xmatrix_Yvector(myExp.trainingset_traces[0], DT_beforeSpike=5.0)
tr = myExp.trainingset_traces[0]
spks = tr.getSpikeTimes()
plt.plot(X_tmp[0:1000,4])
'''
xtx = np.dot(X_tmp.T,X_tmp)
xtx
xty     = np.dot(np.transpose(X_tmp), Y_tmp)
xty
xtx_inv = inv(xtx)
b = np.dot(xtx_inv,xty)
s = np.dot(X_tmp,b)
ydiff = Y_tmp-s
np.sum(ydiff==np.nan)
diffmean = np.mean(ydiff)
diffmean**2
np.var(Y_tmp)
ydiffsq = ydiff**2
diffmean = np.mean(ydiffsq)
1-diffmean/np.var(Y_tmp)
'''

tr = myExp.trainingset_traces[0]
(time, V_est, eta_sum_est) = myGIF_pow.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
indices_tmp = tr.getROI_FarFromSpikes(0.0, myGIF_pow.Tref)
SSE = 0     # sum of squared errors
VAR = 0     # variance of data
SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
vdiff = V_est[indices_tmp] - tr.V[indices_tmp]
vdiffsq = vdiff**2
se = sum(vdiffsq)
se = np.sum(vdiffsq)


plt.figure()
plt.subplot(2,1,1)
timeWindow = np.arange(5000,6000)
plt.plot(time[timeWindow],tr.V[timeWindow],label='true voltage')
plt.plot(time[timeWindow],V_est[timeWindow],label='simulated voltage')
plt.plot(time[timeWindow],vdiff[timeWindow],label='error')
plt.title('Power law')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time[timeWindow],eta_sum_est[timeWindow],label='eta_sum')
plt.title('Power law')
plt.legend()


tr = myExp.trainingset_traces[0]
(time, V_est, eta_sum_est) = myGIF_rect.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
indices_tmp = tr.getROI_FarFromSpikes(0.0, myGIF_pow.Tref)
SSE = 0     # sum of squared errors
VAR = 0     # variance of data
SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
vdiff = V_est[indices_tmp] - tr.V[indices_tmp]
vdiffsq = vdiff**2
se = sum(vdiffsq)
se = np.sum(vdiffsq)


plt.figure()
plt.subplot(2,1,1)
plt.plot(time[timeWindow],tr.V[timeWindow],label='true voltage')
plt.plot(time[timeWindow],V_est[timeWindow],label='simulated voltage')
plt.plot(time[timeWindow],vdiff[timeWindow],label='error')
plt.title('Rectangular basis')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time[timeWindow],eta_sum_est[timeWindow],label='eta_sum')
plt.title('Rectangular basis')
plt.legend()










get_ipython().magic('save dotProductProblem_session 0-126')














