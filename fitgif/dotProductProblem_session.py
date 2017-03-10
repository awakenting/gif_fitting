# coding: utf-8

runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Filter_Rect_LogSpaced.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
eta = Filter_Rect_LogSpaced()
eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
eta
bins = eta.bins
import Tools
bins_i = Tools.timeToIndex(bins,0.1)
bins_i2 = Tools.timeToIndex(bins,0.5)
filtint = eta.getInterpolatedFilter(0.1)
support = filtint[0]
support[0:10]
eta.plot()
def expfunction_eta(x):
    return 0.2*np.exp(-x/100.0)

self.eta.setFilter_Function(expfunction_eta)
def expfunction_eta(x):
    return 0.2*np.exp(-x/100.0)

eta.setFilter_Function(expfunction_eta)
eta.plot()
filtint2 = eta.getInterpolatedFilter(0.5)
support2 = filtint2[0]
filtint[1].shape
filtint2[1].shape
plt.plot(support,filtint[1])
plt.plot(support2,filtint2[1])
support2[0:10]
filtint2[0][0:10]
filtint2[1][0:10]
filtint2[1].shape
filtint2[0].shape
filtint1 = eta.getInterpolatedFilter(0.1)
plt.plot(filtint1[0],filtint1[1])
filtint1[0].shape
support[-1]
eta.p_length
support[-2]
support[-3]
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
myGIF.printParameters()
#myGIF.plotParameters()   
myGIF.plotParameters()   
myGIF.plotParameters()   
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
debugfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
myGIF.fit(myExp, DT_beforeSpike=5.0)
myGIF_rect.fit(myExp, DT_beforeSpike=5.0)
myGIF_pow.fit(myExp, DT_beforeSpike=5.0)
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
myGIF_pow.fit(myExp, DT_beforeSpike=5.0)
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
debugfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
runfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
(X_tmp, Y_tmp) = myGIF_pow.fitSubthresholdDynamics_Build_Xmatrix_Yvector(myExp.trainingset_traces[0], DT_beforeSpike=5.0)
X_tmp.shape
plt.plot(X[0:1000,3])
plt.plot(X_tmp[0:1000,3])
plt.plot(X_tmp[0:1000,4])
tr = myExp.trainingset_traces[0]
spks = tr.getSpikeTimes()
plt.plot(X_tmp[350:450,3])
plt.plot(X_tmp[350:450,4])
plt.plot(X_tmp[375:425,3])
plt.plot(X_tmp[375:435,3])
plt.plot(X_tmp[375:445,3])
debugfile('/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean/Main_TestGIF_powerlaw.py', wdir='/home/andrej/Documents/Code/GIF_Toolbox/GIF_Toolbox/CodePy3_clean')
XTX = np.dot(X.T,X)
XTX = np.dot(X_tmp.T,X_tmp)
xtx = np.dot(X_tmp.T,X_tmp)
a2max = np.max(X_tmp[:,4])
get_ipython().magic('save dotProductProblem_session 0-72')
