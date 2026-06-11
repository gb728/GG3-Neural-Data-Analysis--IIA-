#Running Task 4
# %%   Imports
%reload_ext autoreload
%autoreload 2

import functionsForTask4 as t4
import functionsForTask3 as t3
import functionsForTask2 as t2
import functionsForTask1 as t1
import numba
import importlib
import pickle
import numpy as np


importlib.reload(t4)
# %%  Parameters & Constants

M = 10              # use 10<= M <= 100
K = int(100)
T = 100
Ntrials = 100
x0 = 0.5
Rh = 25

numba.set_num_threads(20)


# %% Task 4.1



gammaShapeValues = [1,2,3,4,5,6]
alldata = []
for gamma in gammaShapeValues:
    alldata.append(t4.findingRampLikePSTH(M,T,x0,Ntrials,Rh,K,gammaShape=gamma))

# %% Plotting PSTH's
t4.plotAllGammaPSTHs(alldata,T,binWidth=5, smoothWindow=3)
#t4.plotAllGammaPSTHsWithMean(alldata,T,binWidth=5, smoothWindow=3)
# %%

resultsByGamma = {}

for gammaData in alldata:
    gammaShape = list(gammaData.keys())[0]
    rows = gammaData[gammaShape]

    print(f"\nRunning gamma shape = {gammaShape}")

    confusion, rampErrorRate, stepErrorRate, overallAccuracy, logBFs = t4.confusionFromRows(rows,M,T,x0,Rh,kValue=K,)

    resultsByGamma[gammaShape] = {
        "confusion": confusion,
        "rampErrorRate": rampErrorRate,
        "stepErrorRate": stepErrorRate,
        "overallAccuracy": overallAccuracy,
        "logBFs": logBFs
    }

# %%
for gammaShape, result in resultsByGamma.items():
    print("\nGamma shape:", gammaShape)
    print(result["confusion"])
    print("Ramp error:", result["rampErrorRate"])
    print("Step error:", result["stepErrorRate"])
    print("Accuracy:", result["overallAccuracy"])

# %%
t4.plotGammaResultsBarChart(resultsByGamma = resultsByGamma)

# %%
alldata2 = []
for gamma in gammaShapeValues:
    alldata2.append(t4.findingRampLikePSTH(M,T,x0,Ntrials,Rh,K,gammaShape=gamma))

resultsByGamma2 = {}

for gammaData in alldata2:
    gammaShape = list(gammaData.keys())[0]
    rows = gammaData[gammaShape]

    print(f"\nRunning gamma shape = {gammaShape}")

    confusion, rampErrorRate, stepErrorRate, overallAccuracy, logBFs = t4.confusionFromRows(rows,M,T,x0,Rh,kValue=K,)

    resultsByGamma2[gammaShape] = {
        "confusion": confusion,
        "rampErrorRate": rampErrorRate,
        "stepErrorRate": stepErrorRate,
        "overallAccuracy": overallAccuracy,
        "logBFs": logBFs
    }


# %%
t4.plotGammaResultsBarChart(resultsByGamma = resultsByGamma2)

# %% Saving it for later just in case
"""
with open("task4_alldata.pkl", "wb") as f:
    pickle.dump(alldata, f)"""
# %%
"""
NtrialValues = np.unique(
    np.round(np.logspace(0, np.log10(400), 15)).astype(int)
)

numberOfDatasets = 20

rows = []

for N in NtrialValues:
    print(f"\nRunning model selection for Ntrials = {N}")

    confusion, rampErrorRate, stepErrorRate, overallAccuracy, logBFs = t3.confusionMatrixWrapper(
        numberOfDatasets=numberOfDatasets,
        Ntrials=N,
        M=M,
        T=T,
        x0=x0,
        Rh=Rh,
        kValue=K,
        SDfraction=0.125,
        UniformPriorNotGaussianForMarginal=False,
        UniformSample= True,
        useGammaEmission= True,
        gammaShape=2
    )

    rows.append([
        N,
        confusion[0, 0],
        confusion[0, 1],
        confusion[1, 0],
        confusion[1, 1],
        rampErrorRate,
        stepErrorRate,
        overallAccuracy,
        np.mean(logBFs),
        np.std(logBFs, ddof=1)
    ])

rows = np.array(rows, dtype=float)

header = (
    "Ntrials,rampCorrect,rampAsStep,stepAsRamp,stepCorrect,"
    "rampErrorRate,stepErrorRate,overallAccuracy,meanLogBF,sdLogBF"
)

np.savetxt(
    "modelSelection_errRate_mismatch_4.1.2_m=10_dataset=20.csv",
    rows,
    delimiter=",",
    header=header,
    comments=""
)"""
# %%
mismatchFilePath = r"modelSelection_errRate_mismatch_4.1.2_m=10_dataset=20.csv"


t3.plotModelSelectionErrorRates(
    mismatchFilePath,
    title="4.1.2: Gaussian prior SD = 1/8, True Emmision Gamma shape = 2"
)
# %% 4.2
beta = 1.1
sigma =0.3
m = 1
r = 1
Ntrials = 10000


rampX_T = t3.simulateX_tWrapper(T,x0,Ntrials,beta=beta,sigma=sigma,kValue=K,model='R')
rampSpikes = t3.spikeTrainsFromXt(rampX_T,T,Rh,useGammaEmission=True,gammaShape=1)
#jumpsramp = t2.jTFromSequence(rampX_T,x0)

stepX_T = t3.simulateX_tWrapper(T,x0,Ntrials,m=m,r=r,model='S')
stepSpikes = t3.spikeTrainsFromXt(stepX_T,T,Rh,useGammaEmission=True,gammaShape=1)
#t1.spikeRasterPlot(spikes, T=T,title = 'Spike Raster')
#t1.histogramPlot(rampX_T*Rh,model='R',T=T,Rh=Rh,binWidth=1)
t1.psthPlot(stepSpikes, pad = True, windowSize = 3)
# %%

t4.fourGammaRasterPlot(T, x0, Ntrials, Rh, gammaShapes=(1, 2, 3, 4),
                    model="S",m=m,r=r, maxTrials=20)#,savePath=r'Figure for Latex/Week 4/Spike rasters for different gamma Step.pdf')
# %%
t4.fourGammaPSTHPlot(T=T, x0=x0, Ntrials=Ntrials, Rh=Rh, gammaShapes=(1,2,3,4),
                  model="S", m=m, r=r,  binWidth=3, smoothWindow=3)#,savePath=r'Figure for Latex/Week 4/PSTH from different gamma Step.pdf')
# %%
t1.fanoPlot(t1.fanoValue(rampSpikes,binWidth= 5),T = T, binWidth= 5,  title = 'Fano value against time of ')#step model (m = 500,  r = 10)' )
# %%
t4.fourGammaFanoPlot(T=T,x0=x0,Ntrials=Ntrials,Rh=Rh,gammaShapes=(1, 2, 3, 4),model="S",m=m,r=r,binWidth=5)#,savePath=r'Figure for Latex/Week 4/Fano from different gamma Step.pdf')
# %%
