# %% Imports
%reload_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import numba
import importlib

import functionsForTask2 as t2
import functionsForTask3 as t3
import DoingTask14 as t14

importlib.reload(t3)

# %% Parameters
# T and K are hard coded
T = 100
K = int(100)
Rh=50 
x0 = 0.2
Ntrials = 100

M = 10


"""#priors
ranBeta = np.random.uniform(0,4)
logRanSisgma = np.random.uniform(np.log(0.04),np.log(4))
ranR = int(np.random.uniform(1,6))
ranM = np.random.uniform(0 , 0.75 * T)
ranx0 = np.random.uniform(0,0.5)"""

numba.set_num_threads(8)
#print(numba.get_num_threads())

# %% General

beta = 1.1
sigma = 0.3

rampX_t = t3.simulateX_tWrapper(T,x0,Ntrials,beta = beta, sigma= sigma,kValue = K, model = 'R')
rampSpikes = t2.spikeTrainsFromXt(rampX_t,T,Rh)

m = 0.4 * T
r = int(3)

stepX_t = t3.simulateX_tWrapper(T,x0,Ntrials, m =m, r=r, model = 'S')
stepSpikes = t2.spikeTrainsFromXt(stepX_t,T,Rh)


# %% 3.1.1 Ramp Model - priting the plots but also printing out the stuff for 3.1.2


expectationBeta, expectationLogSigma, standardDeviationBeta, standardDeviationLogSigma,estimationErrorBeta, estimationErrorSigma, = t3.mapMaker(rampSpikes,M,T,x0,Rh,betaTrue=beta,sigmaTrue=sigma,kValue=K,model = 'R', disablePlots=False)

print(f"Expectation of beta {expectationBeta:.3g}\n"
      f"Standard deviation of beta {standardDeviationBeta:.3g} \n"
      f"Error in beta {estimationErrorBeta:.3g}\n")


print(f"Expectation of sigma {np.exp(expectationLogSigma):.3g}\n"
      f"Standard deviation of sigma {standardDeviationLogSigma:.3g} \n"
      f"Error in sigma {estimationErrorSigma:.3g}\n")

# %% 3.1.1 Step Model
expectationM,expectationR, standardDeviationM, standardDeviationR, estimationErrorM, estimationErrorR = t3.mapMaker(stepSpikes,M,T,x0,Rh,mTrue=m,rTrue=r,model='S',disablePlots=True)

print(f"Expectation of m {expectationM:.3g}\n"
      f"Standard deviation of m {standardDeviationM:.3g} \n"
      f"Error in m {estimationErrorM:.3g}\n")



print(f"Expectation of r {expectationR:.3g}\n"
      f"Standard deviation of r {standardDeviationR:.3g} \n"
      f"Error in r {estimationErrorR:.3g}\n") 



# %% 3.1.2 running several trials and saving all the info so I can plot later


#NtrialToTry = np.unique(np.round(np.logspace(0, np.log10(400), 66)).astype(int))


"""
betaExpectations = []
logSigmaExpectations = []

betaSDs = []
logSigmaSDs = []

betaErrors = []
logSigmaErrors = []

# Generate one large dataset once, then use subsets.
# This makes the effect of increasing Ntrials easier to see.
rampSpikesLarge = rampSpikes[:400]

for n in NtrialToTry:
    print(f"Running Ntrials = {n}")

    spikesN = rampSpikesLarge[:n]

    (
        expectationBeta,
        expectationLogSigma,
        standardDeviationBeta,
        standardDeviationLogSigma,
        estimationErrorBeta,
        estimationErrorLogSigma,
    ) = t3.mapMaker(
        spikesN,
        M,
        T,
        x0,
        Rh,
        betaTrue=beta,
        sigmaTrue=sigma,
        kValue=K,
        model='R',
        disablePlots=True
    )

    betaExpectations.append(expectationBeta)
    logSigmaExpectations.append(expectationLogSigma)

    betaSDs.append(standardDeviationBeta)
    logSigmaSDs.append(standardDeviationLogSigma)

    betaErrors.append(abs(estimationErrorBeta))
    logSigmaErrors.append(abs(estimationErrorLogSigma))

betaExpectations = np.array(betaExpectations)
logSigmaExpectations = np.array(logSigmaExpectations)

betaSDs = np.array(betaSDs)
logSigmaSDs = np.array(logSigmaSDs)

betaErrors = np.array(betaErrors)
logSigmaErrors = np.array(logSigmaErrors)

results = np.column_stack([
    NtrialToTry,
    betaExpectations,
    betaSDs,
    betaErrors,
    logSigmaExpectations,
    logSigmaSDs,
    logSigmaErrors
])

header = (
    "Ntrials,betaExpectation,betaSD,betaError,"
    "logSigmaExpectation,logSigmaSD,logSigmaError"
)

np.savetxt(
    "ifgoingtouserenamehere!.csv",
    results,
    delimiter=",",
    header=header,
    comments=""
)"""

# %% Making the plots for the above info.

rampResultsFilePath = r"ramp_trial_results.csv"
rampResultsFilePathBig = r"ramp_trial_results_50times.csv"
stepResultsFilePath = r"step_trial_results.csv" 
stepResultsFilePathBig = r"step_trial_results_50times.csv" 

# use varparamTrue = np.log(sigma)
t3.readAndPlotPosteriorResults(rampResultsFilePathBig,meanparamTrue=beta,varparamTrue=np.log(sigma),model='R')

# %%
"""
betaValuesToTest = np.linspace(0.05,4,6)
sigmaValuesToTest = np.exp(np.linspace(np.log(0.04), np.log(4.0), 6))

t3.runParameterSweepandSave(betaValuesToTest, sigmaValuesToTest, Ntrials=Ntrials,
                              M=M, T=T, x0=x0, Rh=Rh, K=K, model='R',
                              filename="RENAMEramp_parameter_sweep_v3.csv",numberOfDatasets=10)


mValuesToTest = np.linspace(0.05*T,0.75*T,20)
rValuesToTest = np.linspace(1,6,6)

t3.runParameterSweepandSave(mValuesToTest, rValuesToTest, Ntrials=Ntrials,
                              M=M, T=T, x0=x0, Rh=Rh, model='S',
                              filename="RENAMEstep_parameter_sweep_v3.csv",numberOfDatasets=10)"""


# %%

rampSweepFile = r"ramp_parameter_sweep.csv" 
rampSweepFile2 = r"ramp_parameter_sweep_v2.csv"
rampSweepFile3 = r"ramp_parameter_sweep_v3.csv"
stepSweepFile = r"step_parameter_sweep.csv"
stepSweepFile2 = r"step_parameter_sweep_v2.csv"
stepSweepFile3 = r"step_parameter_sweep_v3.csv"


t3.plotParameterSweepHeatmap(rampSweepFile3,model='R', metric="rangeCombinedError")
t3.plotParameterSweepHeatmap(stepSweepFile3,model='S',metric="rangeCombinedError", T=T)
# %% 3.2.1
"""
NtrialValues = np.unique(
    np.round(np.logspace(0, np.log10(400), 25)).astype(int)
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
        SDfraction=0.25,
        UniformPriorNotGaussianForMarginal=True
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
    "modelSelection_errRate_uniform_3.2.1_m=10_dataset=20.csv",
    rows,
    delimiter=",",
    header=header,
    comments=""
)


#===== 3.2.2
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
        SDfraction=0.25,
        UniformPriorNotGaussianForMarginal=False
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
    "modelSelection_errRate_3.2.2_m=10_dataset=20.csv",
    rows,
    delimiter=",",
    header=header,
    comments=""
)

#======== 3.2.3
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
        SDfraction=0.25,
        UniformPriorNotGaussianForMarginal=False,
        UniformSample=False
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
    "modelSelection_errRate_3.2.3_m=10_dataset=20.csv",
    rows,
    delimiter=",",
    header=header,
    comments=""
)"""

# %% 

csv321filepath = r"modelSelection_errRate_uniform_3.2.1_m=10_dataset=20.csv"
csv322filepath = r"modelSelection_errRate_3.2.2_m=10_dataset=20.csv"
csv323filepath = r"modelSelection_errRate_3.2.3_m=10_dataset=20.csv"

t3.plotModelSelectionErrorRates(
    csv321filepath,
    title="3.2.1: Uniform sampling, uniform prior"
)

t3.plotModelSelectionErrorRates(
    csv322filepath,
    title="3.2.2: Uniform sampling, Gaussian prior"
)

t3.plotModelSelectionErrorRates(
    csv323filepath,
    title="3.2.3: Gaussian sampling, Gaussian prior"
)

t3.plotModelSelectionAccuracyComparison(
    [
        csv321filepath,
        csv322filepath,
        csv323filepath
    ],
    [
        "Uniform sample + uniform prior",
        "Uniform sample + Gaussian prior",
        "Gaussian sample + Gaussian prior"
    ]
)
# %% Comparing different SD values for 3.2.2

SDValues = [1/16, 1/8, 1/4, 1/2, 1]
numberOfDatasets = 50

# Ntrials  = 25.

"""
t3.runSDSweepSameDatasetsAndSave(
    SDValues=SDValues,
    Ntrials=25,
    numberOfDatasets=numberOfDatasets,
    M=M,
    T=T,
    x0=x0,
    Rh=Rh,
    kValue=K,
    filename="modelSelection_for_more_SD_3.2.2_m=10_dataset=20_Ntrial=25.csv"
)

# Ntrials = 100.

t3.runSDSweepSameDatasetsAndSave(
    SDValues=SDValues,
    Ntrials=100,
    numberOfDatasets=numberOfDatasets,
    M=M,
    T=T,
    x0=x0,
    Rh=Rh,
    kValue=K,
    filename="modelSelection_for_more_SD_3.2.2_m=10_dataset=20_Ntrial=100.csv"
)"""

# %%

SDValues = [1/16, 1/8, 1/4, 1/2, 1]

t3.plotSDFractionErrorRates(
    "modelSelection_for_more_SD_3.2.2_m=10_dataset=20_Ntrial=25.csv",
    SDValues=SDValues,
    Ntrials=25,
    plotTitle="3.2.2: Gaussian prior width, Ntrials = 25"
)

t3.plotSDFractionErrorRates(
    "modelSelection_for_more_SD_3.2.2_m=10_dataset=20_Ntrial=100.csv",
    SDValues=SDValues,
    Ntrials=100,
    plotTitle="3.2.2: Gaussian prior width, Ntrials = 100"
)


t3.plotSDFractionBarComparison(
    "modelSelection_for_more_SD_3.2.2_m=10_dataset=20_Ntrial=25.csv",
    "modelSelection_for_more_SD_3.2.2_m=10_dataset=20_Ntrial=100.csv",
    SDValues,
    plotTitlePrefix="3.2.2 Gaussian prior width"
)



# %%
results, rows = t3.evaluation_of_ad_hoc(
    numberOfDatasets=10000,
    Ntrials=100,
    T=T,
    x0=x0,
    Rh=Rh,
    kValue=K,
    SDfraction=0.25,
    UniformSample=False
)
# %%
for name, stats in results.items():
    print("\n", name)
    print(stats["confusion"])
    print(f"Ramp error rate: {stats['rampErrorRate']:.3f}")
    print(f"Step error rate: {stats['stepErrorRate']:.3f}")
    print(f"Overall accuracy: {stats['overallAccuracy']:.3f}")
# %%
t14.plotFeatureResultsBarChart(results)
# %%
featureOrder = [
    "PSTH area",
    "Max adjacent PSTH increase",
    "Top two Fano adjacent change",
    "Max Fano factor",
    "Mean Fano factor",
    
]

featureNames = []
rampErrorRates = []
stepErrorRates = []

for featureName in featureOrder:

    if featureName not in results:
        continue

    if  featureName == "Majority vote":
        continue

    stats = results[featureName]

    featureNames.append(featureName)
    rampErrorRates.append(100 * stats["rampErrorRate"])
    stepErrorRates.append(100 * stats["stepErrorRate"])

x = np.arange(len(featureNames))
width = 0.35

plt.figure(figsize=(10, 5))

bars1 = plt.bar(
    x - width / 2,
    rampErrorRates,
    width=width,
    edgecolor="black",
    label="Ramp classified as step"
)

bars2 = plt.bar(
    x + width / 2,
    stepErrorRates,
    width=width,
    edgecolor="black",
    label="Step classified as ramp"
)

plt.bar_label(bars1, fmt="%.1f", padding=3, rotation=45)
plt.bar_label(bars2, fmt="%.1f", padding=3, rotation=45)

plt.xticks(x, featureNames, rotation=25, ha="right")
plt.ylabel("Error rate (%)")
plt.ylim(0, max(max(rampErrorRates), max(stepErrorRates)) * 1.2 + 1)
plt.title("Ad-hoc feature classification error rates")
plt.legend()
plt.tight_layout()
plt.show()
# %%
#plt.gcf().savefig(r"Figure for Latex/Week 3/FigureName.pdf", bbox_inches="tight")
# %%
