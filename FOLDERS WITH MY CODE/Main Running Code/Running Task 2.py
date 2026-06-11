#This is just going to be a file to use the funcitons. Good coding habits i think
from functionsForTask1 import trajectoriesPlot,histogramPlot,spikeRasterPlot
from functionsForTask2 import *

SEED = 2
np.random.seed(SEED)

T=1000
K=100
Ntrials = 100

Rh = 50
x0 = 0.7
beta = 1.0
sigma = 0
m = 500
r = 2     # only positive integers


#cdfBinned = initialProbRamp(1,0.1,1000,100,cdfBin=True)
#estimate = initialProbRamp(1,0.1,1000,100,cdfBin=False)

#print(len(cdfBinned))

#tMatrix=transitionMatrixRampFast(beta,sigma,T,K)
#inProb = initialProbRamp(sigma,x0,T,K)
#print(tMatrixPlease[1])
#print('break')
#print(tMatrixPlease)

#print(transitionProbBinRamp(9,10,1.1,0.5,1000,10))

#print(np.linspace(0,19,20) )

rampS_tSeq = simSeqRampFast(Ntrials ,beta,sigma,x0,T,K, returnS = True)
rampX_tSeq = rampS_tSeq / (K - 1)
spikeStep = spikeTrainsFromXt(rampX_tSeq,T,Rh)

#print(rampX_tSeq.shape)
#spikeRasterPlot(spikeStep)
#print(rampS_tSeq)
#rampR_tSeq = rampX_tSeq * Rh 

#trajectoriesPlot(rampX_tSeq , title= 'Discrete x_t trajectories over time - Ramp (β = 1,σ = 0)')
#compareRtTrajectoriesPlot(beta,sigma,x0,T,K,Ntrials,Rh, title = 'Discrete and Continous Ramp rates against time (β = 0,σ = 1.0)')

#logMat = transitionMatrixRamp(beta,sigma,T,K,safe=True)
#print(logMat)

#logProb = initialProbRamp(sigma,x0,T,K,safe=True)
#print(np.exp(logProb))

#stepSim = simTwoStateHomogeneousStep(m,r,T,Ntrials,x0)



#jumptimes = jTFromSequence(stepSim,x0)

#print(jumptimes)
#trajectoriesPlot(stepSim)

#histogramPlot(jumptimes,binWidth =5, T=T,Rh=Rh,model = 'S',title= 'TBD',showMedian=False)

#stepSim2 = simR1StateHomogeneousStep(m,r,T,Ntrials,x0,fast=True , exact = True)

#trajectoriesPlot(stepSim2,title = 'Corrected Markov Chain Simulated Step x_t Trajectories (m = 500 , r = 400)')
#jumptimes = jTFromSequence(stepSim2,x0)

#histogramPlot(jumptimes,binWidth =5, T=T,Rh=Rh,model = 'S',showMedian=False, title= 'Histogram of times jump occurs in markov model (m = 500, r = 10)  ')
#spikesStep = spikeTrainsFromXt(stepSim2,T,Rh)
#spikeRasterPlot(spikes,T)

#print(logEmissionMat(spikes,T,Rh, r= r, x0 = x0,model ='S'))
#posteriorStep = hmmExpectedStatesWrapper(rampX_tSeq,T,Rh,x0,kValue=K,beta=beta,sigma=sigma,model = 'R')
#posterior = hmmExpectedStatesWrapper(stepSim2,T,Rh,x0,m=m,r=r,model='S')
#print(posterior.shape)
#print(posteriorStep[:3])

#_,probJumpMat,_ = gettingExpectationMat(rampX_tSeq,T,Rh,x0,kValue=K,beta=beta,sigma=sigma,model='R')
#trajectoriesPlot(probJumpMat)

#expectationSmooth=gettingExpectationMat(rampX_tSeq,T,Rh,x0,kValue=K,beta=beta,sigma=sigma,model='R',plot = False,filter = False, title='Worst Parameters for Expectation vs True for x_t - Ramp (β = 0.5,σ = 1.4)')
#expectationFilter=gettingExpectationMat(rampX_tSeq,T,Rh,x0,kValue=K,beta=beta,sigma=sigma,model='R',plot = False,filter = True)

"""
# Comaparison between smooth + filtering
trial = 0

time = np.arange(T) / T

smooth_mae = maeDistance(rampX_tSeq[trial], expectationSmooth[trial])
filter_mae = maeDistance(rampX_tSeq[trial], expectationFilter[trial])

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))

axes[0].plot(time, rampX_tSeq[trial], color="black", linestyle="--",linewidth=1, label="Ground truth")
axes[0].plot(time, expectationSmooth[trial], color="C0", label="Smoothed posterior")
axes[0].set_ylabel(r"$x_t$")
axes[0].set_title(f"Smoothed posterior, MAE={smooth_mae:.2g}")
axes[0].legend()

axes[1].plot(time, rampX_tSeq[trial], color="black",linestyle="--", linewidth=1, label="Ground truth")
axes[1].plot(time, expectationFilter[trial], color="C1",  label="Filtered posterior")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel(r"$x_t$")
axes[1].set_title(f"Filtered posterior, MAE={filter_mae:.2g}")
axes[1].legend()

fig.suptitle("Filtered vs smoothed posterior expectation")
plt.tight_layout()
plt.show()"""



testSigma = np.linspace(0.05,5,25)
testBeta = np.linspace(0,10,25)
inferenceContourPlotterRamp(x0,Rh,Ntrials,T,K,sigmaValues=testSigma,betaValues=testBeta, model = 'R')

#xtFromStep = simR1StateHomogeneousStep(m,r,T,Ntrials,x0,fast=True)
#jumpsSteps = jTFromSequence(xtFromStep,x0)
#ProbJumpMat,_ = gettingExpectationMat(xtFromStep,T,Rh,x0,m=m, r=r,model='S',plot = True ,jumps=jumpsSteps,filter=False , title = 'Probability vs Ground Truth - Step (m=500 , r=2)')
#testM = np.linspace(0,1000,100)
#testR = np.linspace(1,25,25)    # dont go higher than 20 as then it runs the same values of r mutliplpe times since its rounded
#print(testR)
#inferenceContourPlotterRamp(x0,Rh,Ntrials,T,K,rValues=testR,mValues=testM,model = 'S',filter=True)

#trajectoriesPlot(probJumpMat)
