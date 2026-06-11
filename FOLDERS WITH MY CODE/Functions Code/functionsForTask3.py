from inference import hmm_normalizer, batch_hmm_normalizer
import numpy as np
from functionsForTask2 import simR1StateHomogeneousStep, simSeqRampFast ,initialProbRamp,transitionMatrixRamp,logEmissionMat,transitionMatrixR1StateStep, spikeTrainsFromXt
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.stats import norm
from DoingTask14 import psthAreaFeature,maxPSTHAdjacentIncreaseFeature,fanoMeanFeature,fanoMaxFeature,topTwoFanoAdjacentChangeFeature



# ---- 3.1

def simulateX_tWrapper(T,x0,Ntrials, beta = None, sigma = None, m = None, r = None , kValue = None, model = 'R'):
    model = model.upper()
    if model == 'R':
        x_t = simSeqRampFast(Ntrials,beta,sigma,x0,T,kValue,returnS= False)
    
    elif model == 'S':
        x_t = simR1StateHomogeneousStep(m,r,T,Ntrials,x0,exact=True)
    else:
        raise ValueError("Model is either 'R' or 'S'")
    
    return x_t

def hmmNormalizerWrapper(spikes,T,Rh,x0,beta = -1,sigma = -1,kValue = -1,m= -1,r=-1, model = 'None'):
    if model.upper() != 'R' and  model.upper() != 'S':
        raise ValueError("Model type must be selected as either Ramp 'R' or Step 'S' ")    

    if model.upper() == 'R':
        if beta == -1 or sigma == -1 or kValue == -1:
            raise ValueError("Supply a Value of: K or beta or sigma")
        initialProb = initialProbRamp(sigma,x0,T,kValue, safe = False)  
        tMat = transitionMatrixRamp(beta,sigma,T,kValue,safe=False, fast= True)
        ll = logEmissionMat(spikes,T,Rh,kValue = kValue, model = model)       

    else:
        if m == -1 or r == -1:
            raise ValueError("Supply a Value of: m or r")
        r=int(r)
        initialProb =  np.zeros(r + 1) 
        initialProb[0] = 1        
        tMat = transitionMatrixR1StateStep(m,r,safe=False)    
        ll = logEmissionMat(spikes,T,Rh,x0=x0,r=r,model = model)   
        initialProb = initialProb @ np.linalg.matrix_power(tMat,r) # as we are using the shifted distribution

    model_log_likelihoods = []

    if tMat.ndim == 2:
        tMat = tMat[None, :, :]

    return batch_hmm_normalizer(initialProb, tMat, ll)

def mapMaker(spikes,M,T,x0,Rh, betaTrue = None, sigmaTrue = None, mTrue = None, rTrue = None , kValue = None, model = 'R',disablePlots = False):
    model = model.upper()

    if model == 'R':
        logsigmaValues = np.linspace(np.log(0.04),np.log(4),M)
        betaValues = np.linspace(0 , 4, M)

        loglikelihoodAll = []
        for i, logsigmaI in enumerate(logsigmaValues):
            loglikelihoodI = []
            for j, betaJ in enumerate(betaValues):
                loglikelihoodJ = hmmNormalizerWrapper(spikes,T,Rh,x0,beta = betaJ,sigma = np.exp(logsigmaI),kValue = kValue,model = 'R')
                loglikelihoodI.append(loglikelihoodJ)

            loglikelihoodAll.append(loglikelihoodI)

        loglikelihoodAll = np.array(loglikelihoodAll)
        normalisedPosterior = np.exp(loglikelihoodAll - logsumexp(loglikelihoodAll))    #i want to output a matrxi in the same form as the logliklihoodAll here. Need to fix

        if not disablePlots:
            #then plot the normalised posterior on the grid. Thats it. Yeah thats actually it. Just makesure its in the same format, and you just plot this on a grid. I think ill look into what is a good way to plot it.
            plt.figure(figsize=(9, 6))

            plt.imshow(
                normalisedPosterior,
                origin="lower",
                aspect="auto",
                extent=[betaValues[0], betaValues[-1], logsigmaValues[0], logsigmaValues[-1]]
            )

            plt.colorbar(label="Posterior probability")

            """
            plt.contour(
                betaValues,
                logsigmaValues,
                normalisedPosterior,
                colors="white",
                linewidths=0.8
            )"""

            if betaTrue is not None and sigmaTrue is not None:
                plt.scatter(betaTrue, np.log(sigmaTrue), marker="x",color = 'red', s=80, label="True parameter")

            plt.xlabel(r"$\beta$")
            plt.ylabel(r"$\log \sigma$")
            plt.title("Approximate posterior over ramp parameters")
            plt.legend()
            plt.show()

        if betaTrue is not None and sigmaTrue is not None:
            # compute expectation , variance , error ---> Need to do each parameter seperately
            betaMarginal = normalisedPosterior.sum(axis=0)
            logsigmaMarginal = normalisedPosterior.sum(axis=1)

            expectationBeta = np.sum(betaValues * betaMarginal)
            expectationLogSigma = np.sum(logsigmaValues * logsigmaMarginal)

            varianceBeta = np.sum((betaValues - expectationBeta)**2 * betaMarginal)
            varianceLogSigma = np.sum((logsigmaValues - expectationLogSigma)**2 * logsigmaMarginal)

            standardDeviationBeta = np.sqrt(varianceBeta)
            standardDeviationLogSigma = np.sqrt(varianceLogSigma)

            estimationErrorBeta = abs(betaTrue - expectationBeta)
            estimationErrorSigma = abs(sigmaTrue - np.exp(expectationLogSigma))
    
            return expectationBeta, expectationLogSigma, standardDeviationBeta, standardDeviationLogSigma,estimationErrorBeta, estimationErrorSigma,
        else:   #for 3.2
            return loglikelihoodAll


    elif model == 'S':
        mValues = np.linspace(0,3/4 * T,M)
        rValues = np.linspace(1 , 6, min(M,6),dtype=int)

        loglikelihoodAll = []
        for i, rI in enumerate(rValues):
            loglikelihoodI = []
            for j, mJ in enumerate(mValues):
                loglikelihoodJ = hmmNormalizerWrapper(spikes,T,Rh,x0,m = mJ,r = rI,model = 'S')
                loglikelihoodI.append(loglikelihoodJ)

            loglikelihoodAll.append(loglikelihoodI)

        loglikelihoodAll = np.array(loglikelihoodAll)
        normalisedPosterior = np.exp(loglikelihoodAll - logsumexp(loglikelihoodAll)) 

        if not disablePlots:
            plt.figure(figsize=(9, 6))

            plt.imshow(
                normalisedPosterior,
                origin="lower",
                aspect="auto",
                extent=[mValues[0], mValues[-1], rValues[0], rValues[-1]]
            )

            plt.colorbar(label="Posterior probability")

            """
            plt.contour(
                betaValues,
                logsigmaValues,
                normalisedPosterior,
                colors="white",
                linewidths=0.8
            )"""

            if mTrue is not None and rTrue is not None:
                plt.scatter(mTrue, int(rTrue), marker="x", s=80,color = 'red', label="True parameter")

            plt.ylabel(r"$r$")
            plt.xlabel(r"$m$")
            plt.title("Approximate posterior over step parameters")
            plt.legend()
            plt.show()
        
        if mTrue is not None and rTrue is not None:
            mMarginal = normalisedPosterior.sum(axis=0)
            rMarginal = normalisedPosterior.sum(axis=1)

            expectationM = np.sum(mValues * mMarginal)
            expectationR = np.sum(rValues * rMarginal)

            varianceM = np.sum((mValues - expectationM)**2 * mMarginal)
            varianceR = np.sum((rValues - expectationR)**2 * rMarginal)

            standardDeviationM = np.sqrt(varianceM)
            standardDeviationR = np.sqrt(varianceR)

            estimationErrorM = abs(mTrue - expectationM)
            estimationErrorR = abs(rTrue - expectationR)

            
            return expectationM,expectationR, standardDeviationM, standardDeviationR, estimationErrorM, estimationErrorR

        else:   #for 3.2
            return loglikelihoodAll

    else:
        raise ValueError('functionsForTask3:Input a valid model. either = \'S\' or\'R\'')
    

def readAndPlotPosteriorResults(filepath, meanparamTrue=None, varparamTrue=None, model='R'):
    model = model.upper()

    (
        NtrialToTry,
        meanparamExpectations,
        meanparamSDs,
        meanparamErrors,
        varparamExpectations,
        varparamSDs,
        varparamErrors
    ) = np.loadtxt(
        filepath,
        delimiter=",",
        skiprows=1,
        unpack=True
    )

    if model == 'R':
        meanLabel = r"$\beta$"
        varLabel = r"$\log\sigma$"
        meanTitle = r"Posterior estimate of $\beta$ as trial count increases"
        varTitle = r"Posterior estimate of $\log\sigma$ as trial count increases"
        errorTitle = r"Ramp parameter estimation error as trial count increases"

    elif model == 'S':
        meanLabel = r"$m$"
        varLabel = r"$r$"
        meanTitle = r"Posterior estimate of $m$ as trial count increases"
        varTitle = r"Posterior estimate of $r$ as trial count increases"
        errorTitle = r"Step parameter estimation error as trial count increases"

    else:
        raise ValueError("model must be either 'R' or 'S'")

    # Plot first parameter expectation
    plt.figure(figsize=(7, 4))

    plt.errorbar(
        NtrialToTry,
        meanparamExpectations,
        yerr=meanparamSDs,
        marker="o",
        capsize=4,
        label=r"Posterior mean $\pm$ SD"
    )

    if meanparamTrue is not None:
        plt.axhline(meanparamTrue, linestyle="--", label=f"True {meanLabel}")

    plt.xscale("log")
    plt.xlabel("Number of trials")
    plt.ylabel(meanLabel)
    plt.title(meanTitle)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot second parameter expectation
    plt.figure(figsize=(7, 4))

    plt.errorbar(
        NtrialToTry,
        varparamExpectations,
        yerr=varparamSDs,
        marker="o",
        capsize=4,
        color = 'orange',
        label=r"Posterior mean $\pm$ SD"
    )

    if varparamTrue is not None:
        plt.axhline(varparamTrue, linestyle="--", label=f"True {varLabel}")

    plt.xscale("log")
    plt.xlabel("Number of trials")
    plt.ylabel(varLabel)
    plt.title(varTitle)
    plt.legend()
    plt.tight_layout()
    plt.show()

    meanPercentErrors = 100 * np.abs(meanparamExpectations - meanparamTrue) / abs(meanparamTrue)
    varPercentErrors = 100 * np.abs(varparamExpectations - varparamTrue) / abs(varparamTrue)

    # Plot estimation errors
    plt.figure(figsize=(7, 4))

    plt.plot(
        NtrialToTry,
        meanPercentErrors,
        marker="o",
        label=f"{meanLabel} Percentage error"
    )

    plt.plot(
        NtrialToTry,
        varPercentErrors,
        marker="o",
        color = 'orange',
        label=f"{varLabel} Percentage error"
    )

    plt.xscale("log")
    plt.xlabel("Number of trials")
    plt.ylabel("Percentage estimation error")
    plt.title(errorTitle)
    plt.legend()
    plt.tight_layout()
    plt.show()

def percentError(estimate, trueValue):
    if trueValue == 0:
        return np.nan
    return 100 * abs(estimate - trueValue) / abs(trueValue)

def runParameterSweepandSave(meanValues, varValues, Ntrials, M, T, x0, Rh, K=None, model='R',filename="parameter_sweep_results.csv", numberOfDatasets=1):
    model = model.upper()
    rows = []

    for meanTrue in meanValues:
        for varTrue in varValues:
            print(f"Running true mean-param = {meanTrue:.3g}, true var-param = {varTrue:.3g}")

            datasetRows = []

            for datasetIndex in range(numberOfDatasets):
                print(f"  Dataset {datasetIndex + 1}/{numberOfDatasets}")

                if model == 'R':
                    betaTrue, sigmaTrue = meanTrue, varTrue

                    x_t = simulateX_tWrapper(T, x0, Ntrials, beta=betaTrue,
                                             sigma=sigmaTrue, kValue=K, model='R')
                    spikes = spikeTrainsFromXt(x_t, T, Rh)

                    expectationMean, expectationVar, sdMean, sdVar, _, _ = mapMaker(
                        spikes, M, T, x0, Rh, betaTrue=betaTrue, sigmaTrue=sigmaTrue,
                        kValue=K, model='R', disablePlots=True
                    )

                    meanAbsError = abs(expectationMean - betaTrue)
                    varAbsError = abs(expectationVar - np.log(sigmaTrue))

                    meanPercentError = percentError(expectationMean, betaTrue)
                    varPercentError = percentError(np.exp(expectationVar), sigmaTrue)

                elif model == 'S':
                    mTrue, rTrue = meanTrue, int(varTrue)

                    x_t = simulateX_tWrapper(T, x0, Ntrials, m=mTrue, r=rTrue, model='S')
                    spikes = spikeTrainsFromXt(x_t, T, Rh)

                    expectationMean, expectationVar, sdMean, sdVar, _, _ = mapMaker(
                        spikes, M, T, x0, Rh, mTrue=mTrue, rTrue=rTrue,
                        model='S', disablePlots=True
                    )

                    meanAbsError = abs(expectationMean - mTrue)
                    varAbsError = abs(expectationVar - rTrue)

                    meanPercentError = percentError(expectationMean, mTrue)
                    varPercentError = percentError(expectationVar, rTrue)

                else:
                    raise ValueError("model must be either 'R' or 'S'")

                combinedPercentError = np.nanmean([meanPercentError, varPercentError])

                datasetRows.append([expectationMean, expectationVar, sdMean, sdVar,
                                    meanAbsError, varAbsError, meanPercentError,
                                    varPercentError, combinedPercentError])

            datasetRows = np.array(datasetRows, dtype=float)

            averagedValues = np.nanmean(datasetRows, axis=0)
            errorSDsAcrossDatasets = np.nanstd(datasetRows[:, 4:], axis=0, ddof=1)

            rows.append([meanTrue, varTrue, numberOfDatasets, *averagedValues,
                         *errorSDsAcrossDatasets])

    rows = np.array(rows, dtype=float)

    header = ("trueMean,trueVar,numberOfDatasets,"
              "meanExpectation,varExpectation,meanSD,varSD,"
              "meanAbsError,varAbsError,meanPercentError,varPercentError,"
              "combinedPercentError,"
              "meanAbsErrorDatasetSD,varAbsErrorDatasetSD,"
              "meanPercentErrorDatasetSD,varPercentErrorDatasetSD,"
              "combinedPercentErrorDatasetSD")

    np.savetxt(filename, rows, delimiter=",", header=header, comments="")
    
def plotParameterSweepHeatmap(filepath, model='R', metric="rangeCombinedError",
                              T=None, annotate="auto", figsize=None):
    model = model.upper()
    data = np.genfromtxt(filepath, delimiter=",", names=True)

    meanVals = np.unique(data["trueMean"])
    varVals = np.unique(data["trueVar"])
    Z = np.full((len(varVals), len(meanVals)), np.nan)

    for row in data:
        meanIndex = np.argmin(np.abs(meanVals - row["trueMean"]))
        varIndex = np.argmin(np.abs(varVals - row["trueVar"]))

        if metric == "rangeCombinedError":
            if model == 'R':
                meanScaled = 100 * row["meanAbsError"] / 4
                varScaled = 100 * row["varAbsError"] / (np.log(4) - np.log(0.04))
            elif model == 'S':
                if T is None:
                    meanRange = meanVals.max() - meanVals.min()
                else:
                    meanRange = 0.75 * T
                meanScaled = 100 * row["meanAbsError"] / meanRange
                varScaled = 100 * row["varAbsError"] / (6 - 1)
            Z[varIndex, meanIndex] = 0.5 * (meanScaled + varScaled)
        else:
            Z[varIndex, meanIndex] = row[metric]

    if model == 'R':
        xLabel, yLabel = r"True $\beta$", r"True $\sigma$"
        title = "Ramp parameter recovery error"
    elif model == 'S':
        xLabel, yLabel = r"True $m$", r"True $r$"
        title = "Step parameter recovery error"
    else:
        raise ValueError("model must be either 'R' or 'S'")

    if metric == "rangeCombinedError":
        cbarLabel = "Mean error (% of parameter range)"
    elif "Percent" in metric:
        cbarLabel = "Percentage error (%)"
    elif "SD" in metric:
        cbarLabel = "Standard deviation across datasets"
    else:
        cbarLabel = metric

    if figsize is None:
        figsize = (max(7, 0.55 * len(meanVals)), max(4.5, 0.55 * len(varVals)))

    plt.figure(figsize=figsize)
    im = plt.imshow(Z, origin="lower", aspect="auto", interpolation="nearest")
    plt.colorbar(im, label=cbarLabel)

    plt.xticks(np.arange(len(meanVals)), [f"{v:.3g}" for v in meanVals],
               rotation=45 if len(meanVals) > 8 else 0, ha="right" if len(meanVals) > 8 else "center")
    plt.yticks(np.arange(len(varVals)), [f"{v:.3g}" for v in varVals])

    nCells = len(meanVals) * len(varVals)
    doAnnotate = annotate is True or (annotate == "auto" and nCells <= 80)

    if doAnnotate:
        fontSize = 9 if nCells <= 60 else 7
        for i in range(len(varVals)):
            for j in range(len(meanVals)):
                if not np.isnan(Z[i, j]):
                    plt.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center",
                             fontsize=fontSize)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---- 3.2

# If want a function to output R
# So calculates both marginal liklihoods
# A function that gets marginal liklihoods for either model
# That funciton needs to find the liklihood so you can 'integrate it' which is found through hmm normaliser / forward pass   < --- Map maker now also returns the loglikihood matrix of all the params for either model

def logBayesFactor(spikes,M,T,x0,Rh,kValue,SDfraction=1/4,UniformPriorNotGaussianForMarginal = True):


    loglikilhoodRamp = mapMaker(spikes,M,T,x0,Rh,kValue = kValue,disablePlots=True, model = 'R')
    loglikilhoodStep = mapMaker(spikes,M,T,x0,Rh,disablePlots=True , model = 'S')

    loglikilhoodRamp = np.asarray(loglikilhoodRamp, dtype=float)
    loglikilhoodStep = np.asarray(loglikilhoodStep, dtype=float)

    if UniformPriorNotGaussianForMarginal:
        logMarginalRamp = logsumexp(loglikilhoodRamp) - np.log(loglikilhoodRamp.size)
        logMarginalStep = logsumexp(loglikilhoodStep) - np.log(loglikilhoodStep.size)

    else:
        logMarginalRamp = logMarginalLikelihoodWithGaussianProbabilities(T,loglikilhoodRamp,SDfraction,model = 'R')
        logMarginalStep = logMarginalLikelihoodWithGaussianProbabilities(T,loglikilhoodStep,SDfraction,model = 'S')

    return logMarginalRamp - logMarginalStep

# not using this code as recommended
def gaussianGridWeightsCDF(values, mean, sd):           #applying a gaussian like in the ramp model HMM but across the parameters ... we are told to calculate this by doing guassian PDF and normalising ... so need to change it.
    values = np.asarray(values)

    edges = np.zeros(len(values) + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = -np.inf
    edges[-1] = np.inf

    weights = norm.cdf(edges[1:], loc=mean, scale=sd) - norm.cdf(edges[:-1], loc=mean, scale=sd)
    weights = weights / weights.sum()

    return weights

def gaussianWeightsPDF(values, mean, sd):
    weights = norm.pdf(values, loc=mean, scale=sd)
    weights = weights / weights.sum()
    return weights


def logMarginalLikelihoodWithGaussianProbabilities(T,loglikelihood,SDfraction,model ='S'):
    model = model.upper()
    loglikelihood = np.asarray(loglikelihood, dtype=float)

    if model == 'S':
        mMean = 0.5 * (0.75 * T)
        mSD = (0.75 * T) * SDfraction
        rMean = 1
        rSD = (6 - 1) * SDfraction

        numRValues = loglikelihood.shape[0]
        numMValues = loglikelihood.shape[1]

        mValues = np.linspace(0, 0.75 * T, numMValues)
        rValues = np.arange(1, numRValues + 1)

        marginallikelihoodSum = 0

        mWeights = gaussianWeightsPDF(mValues, mMean, mSD)
        rWeights = gaussianWeightsPDF(rValues, rMean, rSD)

        logPrior = np.log(rWeights[:, None]) + np.log(mWeights[None, :])        


    elif model == 'R':
        betaMean = 2 
        betaSD = 4 * SDfraction
        logSigmaMean = 0.5 * (np.log(0.04) + np.log(4))
        logSigmaSD = (np.log(4) - np.log(0.04)) * SDfraction

        numLogSigmaValues = loglikelihood.shape[0]
        numBetaValues = loglikelihood.shape[1]

        betaValues = np.linspace(0,4,numBetaValues)
        logSigmaValues = np.linspace(np.log(0.04), np.log(4), numLogSigmaValues)

        betaWeights = gaussianWeightsPDF(betaValues, betaMean, betaSD)
        logSigmaWeights = gaussianWeightsPDF(logSigmaValues, logSigmaMean, logSigmaSD)

        logPrior = np.log(logSigmaWeights[:, None]) + np.log(betaWeights[None, :])
    
    else:
        raise ValueError("model must be 'R' or 'S'")


    return logsumexp(loglikelihood + logPrior)
                
def choosingModel(logBayesFactor):

    chosenModel = 'R' if logBayesFactor > 0 else 'S'

    return chosenModel

def confusionMatrixWrapper(numberOfDatasets, Ntrials, M, T, x0, Rh, kValue,SDfraction=0.25,UniformPriorNotGaussianForMarginal=True,UniformSample = True,useGammaEmission = False,gammaShape = None):
    rng = np.random.default_rng(100)
    confusion = np.zeros((2, 2), dtype=int)
    # rows = true model: 0 ramp, 1 step
    # cols = chosen model: 0 ramp, 1 step


    logBFs = []

    # always testing both so for large number of datasets its effectively the same
    for trueModel in ['R', 'S']:
        for datasetIndex in range(numberOfDatasets):

            if trueModel == 'R':
                if UniformSample:
                    beta = np.random.uniform(0, 4)
                    logsigma = np.random.uniform(np.log(0.04), np.log(4))
                else:
                    betaMean = 2 
                    betaSD = 4 * SDfraction
                    logSigmaMean = 0.5 * (np.log(0.04) + np.log(4))
                    logSigmaSD = (np.log(4) - np.log(0.04)) * SDfraction

                    beta = rng.normal(loc = betaMean, scale = betaSD)
                    beta = np.clip(beta,0,4)
                    logsigma = rng.normal(loc = logSigmaMean, scale = logSigmaSD)
                    logsigma = np.clip(logsigma,np.log(0.04),np.log(4))

                sigma = np.exp(logsigma)

                x_t = simulateX_tWrapper(T, x0, Ntrials,beta=beta,sigma=sigma,kValue=kValue,model='R')

            else:
                if UniformSample:
                    m = np.random.uniform(0, 0.75 * T)
                    r = np.random.randint(1, 7)
                else:
                    mMean = 0.5 * (0.75 * T)
                    mSD = (0.75 * T) * SDfraction
                    rMean = 1
                    rSD = (6 - 1) * SDfraction

                    m = rng.normal(loc = mMean, scale = mSD)
                    m = np.clip(m,0,0.75*T)
                    rValues = np.arange(1, 7)
                    rWeights = gaussianWeightsPDF(rValues, rMean, rSD)
                    r = rng.choice(rValues, p=rWeights) #as it is discrete


                x_t = simulateX_tWrapper(T, x0, Ntrials,m=m,r=r,model='S')
            if useGammaEmission:
                spikes = spikeTrainsFromXt(x_t, T, Rh,useGammaEmission=True, gammaShape=gammaShape)
            else:
                spikes = spikeTrainsFromXt(x_t,T,Rh)

            logBF = logBayesFactor(spikes, M, T, x0, Rh, kValue = kValue,SDfraction=SDfraction,UniformPriorNotGaussianForMarginal=UniformPriorNotGaussianForMarginal)

            chosenModel = choosingModel(logBF)

            trueIndex = 0 if trueModel == 'R' else 1
            chosenIndex = 0 if chosenModel == 'R' else 1

            confusion[trueIndex, chosenIndex] += 1
            logBFs.append(logBF)

            print(
                f"True {trueModel}, dataset {datasetIndex+1}/{numberOfDatasets}: "
                f"chosen {chosenModel}, logBF = {logBF:.3g}"
            )

    rampErrorRate = confusion[0, 1] / confusion[0].sum()
    stepErrorRate = confusion[1, 0] / confusion[1].sum()
    overallAccuracy = (confusion[0, 0] + confusion[1, 1]) / confusion.sum()

    return confusion, rampErrorRate, stepErrorRate, overallAccuracy, np.array(logBFs)

def runSDSweepSameDatasetsAndSave(SDValues, Ntrials, numberOfDatasets,
                                  M, T, x0, Rh, kValue, filename,
                                  seed=100):

    rng = np.random.default_rng(seed)

    datasets = []

    # Generate datasets once
    for trueModel in ['R', 'S']:
        for datasetIndex in range(numberOfDatasets):

            if trueModel == 'R':
                beta = rng.uniform(0, 4)
                logsigma = rng.uniform(np.log(0.04), np.log(4))
                sigma = np.exp(logsigma)

                x_t = simulateX_tWrapper(
                    T, x0, Ntrials,
                    beta=beta,
                    sigma=sigma,
                    kValue=kValue,
                    model='R'
                )

            else:
                m = rng.uniform(0, 0.75 * T)
                r = rng.integers(1, 7)

                x_t = simulateX_tWrapper(
                    T, x0, Ntrials,
                    m=m,
                    r=r,
                    model='S'
                )

            spikes = spikeTrainsFromXt(x_t, T, Rh)
            datasets.append((trueModel, spikes))

    # Compute likelihood grids once for each dataset
    likelihoods = []

    for datasetIndex, (trueModel, spikes) in enumerate(datasets):
        print(f"Computing likelihood grids for dataset {datasetIndex + 1}/{len(datasets)}")

        loglikelihoodRamp = mapMaker(
            spikes, M, T, x0, Rh,
            kValue=kValue,
            model='R',
            disablePlots=True
        )

        loglikelihoodStep = mapMaker(
            spikes, M, T, x0, Rh,
            model='S',
            disablePlots=True
        )

        likelihoods.append((trueModel, loglikelihoodRamp, loglikelihoodStep))

    rows = []

    # Now test each SD fraction on the exact same datasets
    for fraction in SDValues:
        print(f"\nEvaluating SD fraction = {fraction}")

        confusion = np.zeros((2, 2), dtype=int)
        logBFs = []

        for trueModel, loglikelihoodRamp, loglikelihoodStep in likelihoods:

            logMarginalRamp = logMarginalLikelihoodWithGaussianProbabilities(
                T,
                loglikelihoodRamp,
                fraction,
                model='R'
            )

            logMarginalStep = logMarginalLikelihoodWithGaussianProbabilities(
                T,
                loglikelihoodStep,
                fraction,
                model='S'
            )

            logBF = logMarginalRamp - logMarginalStep
            chosenModel = choosingModel(logBF)

            trueIndex = 0 if trueModel == 'R' else 1
            chosenIndex = 0 if chosenModel == 'R' else 1

            confusion[trueIndex, chosenIndex] += 1
            logBFs.append(logBF)

        rampErrorRate = confusion[0, 1] / confusion[0].sum()
        stepErrorRate = confusion[1, 0] / confusion[1].sum()
        overallAccuracy = (confusion[0, 0] + confusion[1, 1]) / confusion.sum()

        rows.append([
            fraction,
            Ntrials,
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
        "SDfraction,Ntrials,rampCorrect,rampAsStep,stepAsRamp,stepCorrect,"
        "rampErrorRate,stepErrorRate,overallAccuracy,meanLogBF,sdLogBF"
    )

    np.savetxt(
        filename,
        rows,
        delimiter=",",
        header=header,
        comments=""
    )

    return rows

def plotModelSelectionErrorRates(filepath, title=None):
    data = np.genfromtxt(filepath, delimiter=",", names=True)

    Ntrials = data["Ntrials"]
    rampError = 100 * data["rampErrorRate"]
    stepError = 100 * data["stepErrorRate"]

    plt.figure(figsize=(7, 4))

    plt.plot(Ntrials, rampError, marker="o", label="Ramp classified as step")
    plt.plot(Ntrials, stepError, marker="o", label="Step classified as ramp")

    plt.xscale("log")
    plt.xlabel("Number of trials")
    plt.ylabel("Error rate (%)")

    if title is None:
        title = "Model selection error rate against number of trials"

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(r"Figure for Latex/Week 4/Rename3.pdf")
    plt.show()

def plotModelSelectionAccuracyComparison(filepaths, labels):
    plt.figure(figsize=(7, 4))

    for filepath, label in zip(filepaths, labels):
        data = np.genfromtxt(filepath, delimiter=",", names=True)

        plt.plot(
            data["Ntrials"],
            100 * data["overallAccuracy"],
            marker="o",
            label=label
        )

    plt.xscale("log")
    plt.xlabel("Number of trials")
    plt.ylabel("Overall accuracy (%)")
    plt.title("Model selection accuracy against number of trials")
    plt.legend()
    plt.tight_layout()
    plt.show()


# running code for varying values of SDfraction. So do one for 1/2 1/4 and 1/8. Then do it for 25 trials and 100 trials. So just 2 sets. This is for 3.2.2 <--- Chill, doing CSV now, just need to plot it afterwards

#So for the gauss gauss, we want to find the error rates for the adhocs methods in week one.
#Ad hocs were psth area , top two fano adjacent change?? , fano max, fano mean.

def evaluation_of_ad_hoc(numberOfDatasets, Ntrials, T, x0, Rh, kValue,SDfraction=0.25,UniformSample = True):
    rng = np.random.default_rng(100)
    featureNames = [
        "PSTH area",
        "Max adjacent PSTH increase",
        "Mean Fano factor",
        "Max Fano factor",
        "Top two Fano adjacent change",
        "Majority vote"
    ]

    confusions = {
        name: np.zeros((2, 2), dtype=int)
        for name in featureNames
    }

    # rows = individual dataset results
    rows = []

    for trueModel in ['R', 'S']:
        trueIndex = 0 if trueModel == 'R' else 1

        for datasetIndex in range(numberOfDatasets):

            if trueModel == 'R':
                if UniformSample:
                    beta = rng.uniform(0, 4)
                    logsigma = rng.uniform(np.log(0.04), np.log(4))
                else:
                    betaMean = 2
                    betaSD = 4 * SDfraction

                    logSigmaMean = 0.5 * (np.log(0.04) + np.log(4))
                    logSigmaSD = (np.log(4) - np.log(0.04)) * SDfraction

                    beta = rng.normal(loc=betaMean, scale=betaSD)
                    beta = np.clip(beta, 0, 4)

                    logsigma = rng.normal(loc=logSigmaMean, scale=logSigmaSD)
                    logsigma = np.clip(logsigma, np.log(0.04), np.log(4))

                sigma = np.exp(logsigma)

                x_t = simulateX_tWrapper(
                    T, x0, Ntrials,
                    beta=beta,
                    sigma=sigma,
                    kValue=kValue,
                    model='R'
                )

            else:
                if UniformSample:
                    m = rng.uniform(0, 0.75 * T)
                    r = rng.integers(1, 7)
                else:
                    mMean = 0.5 * (0.75 * T)
                    mSD = (0.75 * T) * SDfraction

                    rMean = 1
                    rSD = (6 - 1) * SDfraction

                    m = rng.normal(loc=mMean, scale=mSD)
                    m = np.clip(m, 0, 0.75 * T)

                    rValues = np.arange(1, 7)
                    rWeights = gaussianWeightsPDF(rValues, rMean, rSD)
                    r = rng.choice(rValues, p=rWeights)

                x_t = simulateX_tWrapper(
                    T, x0, Ntrials,
                    m=m,
                    r=r,
                    model='S'
                )

            spikes = spikeTrainsFromXt(x_t, T, Rh)

            f1 = psthAreaFeature(spikes)
            f2 = maxPSTHAdjacentIncreaseFeature(spikes)
            f3 = fanoMeanFeature(spikes)
            f4 = fanoMaxFeature(spikes)
            f5 = topTwoFanoAdjacentChangeFeature(spikes)

            # prediction convention:
            # 0 = ramp, 1 = step
            predictions = np.array([
                0 if f1 > 39.5 else 1,
                0 if f2 < 2.33 else 1,
                0 if f3 < 1.25 else 1,
                0 if f4 < 0.14 else 1,
                0 if f5 < 1.09 else 1
            ], dtype=int)

            # majority vote across the 5 features
            majorityPrediction = 1 if predictions.sum() >= 3 else 0

            allPredictions = np.append(predictions, majorityPrediction)

            for name, pred in zip(featureNames, allPredictions):
                confusions[name][trueIndex, pred] += 1

            rows.append([
                trueIndex,
                *predictions,
                majorityPrediction,
                f1, f2, f3, f4, f5
            ])

            print(
                f"True {trueModel}, dataset {datasetIndex + 1}/{numberOfDatasets}, "
                f"predictions = {predictions}, majority = {majorityPrediction}"
            )

    results = {}

    for name, confusion in confusions.items():
        rampErrorRate = confusion[0, 1] / confusion[0].sum()
        stepErrorRate = confusion[1, 0] / confusion[1].sum()
        overallAccuracy = (confusion[0, 0] + confusion[1, 1]) / confusion.sum()

        results[name] = {
            "confusion": confusion,
            "rampErrorRate": rampErrorRate,
            "stepErrorRate": stepErrorRate,
            "overallAccuracy": overallAccuracy
        }

    rows = np.array(rows, dtype=float)

    return results, rows

def plotSDFractionErrorRates(filepath, SDValues, Ntrials=None, plotTitle=None):
    data = np.genfromtxt(filepath, delimiter=",", names=True)

    SDValues = np.array(SDValues, dtype=float)

    rampError = 100 * data["rampErrorRate"]
    stepError = 100 * data["stepErrorRate"]

    plt.figure(figsize=(7, 4))

    plt.plot(SDValues, rampError, marker="o", label="Ramp classified as step")
    plt.plot(SDValues, stepError, marker="o", label="Step classified as ramp")

    plt.xlabel("Gaussian prior SD fraction")
    plt.ylabel("Error rate (%)")

    # Optional: log scale can make 1/8, 1/4, 1/2 spacing look nicer
    plt.xscale("log", base=2)
    plt.xticks(SDValues, [f"{v:.3g}" for v in SDValues])

    if plotTitle is None:
        if Ntrials is not None:
            plotTitle = f"Effect of Gaussian prior width, Ntrials = {Ntrials}"
        else:
            plotTitle = "Effect of Gaussian prior width"

    plt.title(plotTitle)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotSDFractionBarComparison(filepath25, filepath100, SDValues, plotTitlePrefix="3.2.2"):
    data25 = np.genfromtxt(filepath25, delimiter=",", names=True)
    data100 = np.genfromtxt(filepath100, delimiter=",", names=True)

    SDValues = np.array(SDValues, dtype=float)
    labels = [f"{v:.3g}" for v in SDValues]

    x = np.arange(len(SDValues))
    width = 0.35

    # -------- Ramp classified as step --------
    rampError25 = 100 * data25["rampErrorRate"]
    rampError100 = 100 * data100["rampErrorRate"]

    plt.figure(figsize=(7, 4))

    bars1 = plt.bar(x - width / 2, rampError25, width,
                    edgecolor="black", label="Ntrials = 25")
    bars2 = plt.bar(x + width / 2, rampError100, width,
                    edgecolor="black", label="Ntrials = 100")

    plt.bar_label(bars1, fmt="%.1f", padding=3)
    plt.bar_label(bars2, fmt="%.1f", padding=3)

    plt.xticks(x, labels)
    plt.xlabel("Gaussian prior SD fraction")
    plt.ylabel("Error rate (%)")
    plt.title(f"{plotTitlePrefix}: Ramp classified as step")
    plt.legend()
    plt.tight_layout()
    #plt.savefig(r"Figure for Latex/Week 3/FigureName2.pdf", bbox_inches="tight")
    plt.show()

    # -------- Step classified as ramp --------
    stepError25 = 100 * data25["stepErrorRate"]
    stepError100 = 100 * data100["stepErrorRate"]

    plt.figure(figsize=(7, 4))

    bars1 = plt.bar(x - width / 2, stepError25, width,
                    edgecolor="black", label="Ntrials = 25")
    bars2 = plt.bar(x + width / 2, stepError100, width,
                    edgecolor="black", label="Ntrials = 100")

    plt.bar_label(bars1, fmt="%.1f", padding=3)
    plt.bar_label(bars2, fmt="%.1f", padding=3)

    plt.xticks(x, labels)
    plt.xlabel("Gaussian prior SD fraction")
    plt.ylabel("Error rate (%)")
    plt.title(f"{plotTitlePrefix}: Step classified as ramp")
    plt.legend()
    plt.tight_layout()
    #plt.savefig(r"Figure for Latex/Week 3/FigureName1.pdf", bbox_inches="tight")
    plt.show()