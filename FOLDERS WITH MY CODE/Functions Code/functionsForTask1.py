from models import *
import matplotlib.pyplot as plt
from scipy import signal
import random

SEED = 2

np.random.seed(SEED)
random.seed(SEED)

def compareFanoTwoParameters( #plot 2 Fano curves while varying parameters.
    modelType,
    param1Name,
    param1Values,
    param2Name,
    param2Values,
    baseParams=None,
    Ntrials=5000,
    T=1000,
    fanoBinWidth=50,
    title=None
):


    modelType = modelType.lower()

    if baseParams is None:
        if modelType == "ramp":
            baseParams = {
                "beta": 1.0,
                "sigma": 0.5,
                "x0": 0.2,
                "Rh": 50
            }
        elif modelType == "step":
            baseParams = {
                "m": T / 2,
                "r": 3,
                "x0": 0.2,
                "Rh": 50
            }
        else:
            raise ValueError("modelType must be 'ramp' or 'step'")

    nPlots = len(param1Values)
    nCols = min(3, nPlots)
    nRows = np.ceil(nPlots / nCols)

    fig, axes = plt.subplots(
        nRows,
        nCols,
        figsize=(5 * nCols, 4 * nRows),
        sharex=True,
        sharey=True
    )

    axes = np.array(axes).reshape(-1)

    for axIndex, param1Value in enumerate(param1Values):
        ax = axes[axIndex]

        for param2Value in param2Values:
            params = baseParams.copy()
            params[param1Name] = param1Value
            params[param2Name] = param2Value

            if modelType == "ramp":
                model = RampModel(**params)
            elif modelType == "step":
                model = StepModel(**params)
            else:
                raise ValueError("modelType must be 'ramp' or 'step'")

            spikes, _, _ = model.simulate(Ntrials=Ntrials, T=T)

            fano = fanoValue(spikes, binWidth=fanoBinWidth)

            nBins = fano.shape[0]
            time = (np.arange(nBins) + 0.5) * fanoBinWidth / T

            ax.plot(
                time,
                fano,
                marker="o",
                label=f"{param2Name} = {param2Value}"
            )

        ax.axhline(1, linestyle="--", color="black", label="Poisson baseline")
        ax.set_title(f"{param1Name} = {param1Value}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Fano factor")
        ax.set_xlim(0, 1)
        ax.legend()

    # Hide unused subplots
    for j in range(len(param1Values), len(axes)):
        axes[j].axis("off")

    if title is None:
        title = f"{modelType.capitalize()} model Fano comparison: varying {param1Name} and {param2Name}"

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def psthVector(spikeData, binWidth=50, smoothWindow=3):
    T = spikeData.shape[1]
    Ntrials = spikeData.shape[0]

    T_trimmed = (T // binWidth) * binWidth
    spikeData = spikeData[:, :T_trimmed]

    binned = spikeData.reshape(Ntrials, -1, binWidth)
    meanCounts = binned.sum(axis=2).mean(axis=0)

    dt = 1 / T
    binDuration = binWidth * dt

    psth = meanCounts / binDuration

    if smoothWindow > 1:
        window = np.ones(smoothWindow) / smoothWindow
        psth = np.convolve(psth, window, mode="same")

    return psth

def rmseDistance(traj1, traj2):  #root mean square error
    return np.sqrt(np.mean((traj1 - traj2) ** 2))

def stepMatchContourForFixedRamp(beta=1.1,sigma=0.5,x0=0.2,Rh=50,mValues=None,rValues=None,Ntrials=5000,T=1000,binWidth=50,smoothWindow=3,):   # plots contour map of rmse when finding matching step model for ramp
    
    if mValues is None:
        mValues = np.linspace(T / 4, 3 * T / 4, 15)

    if rValues is None:
        rValues = np.linspace(0.5, 6, 15)

    # Simulate fixed ramp model
    ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)
    rampSpikes, _, _ = ramp.simulate(Ntrials=Ntrials, T=T)
    rampPSTH = psthVector(rampSpikes, binWidth=binWidth, smoothWindow=smoothWindow)

    distances = np.zeros((len(rValues), len(mValues)))

    bestDistance = np.inf
    bestM = None
    bestR = None
    bestStepPSTH = None

    for i, r in enumerate(rValues):
        for j, m in enumerate(mValues):
            step = StepModel(m=m, r=r, x0=x0, Rh=Rh)
            stepSpikes, _, _ = step.simulate(Ntrials=Ntrials, T=T)
            stepPSTH = psthVector(stepSpikes, binWidth=binWidth, smoothWindow=smoothWindow)

            distance = rmseDistance(rampPSTH, stepPSTH)
            distances[i, j] = distance

            if distance < bestDistance:
                bestDistance = distance
                bestM = m
                bestR = r
                bestStepPSTH = stepPSTH

    # Plot contour map
    M, R = np.meshgrid(mValues, rValues)

    plt.figure(figsize=(8, 5))

    contour = plt.contourf(M, R, distances, levels=20)
    plt.colorbar(contour, label="PSTH distance / RMSE")

    plt.scatter(bestM, bestR, marker="x", s=100, color="black", label="Best match")

    plt.xlabel("Step mean jump time m")
    plt.ylabel("Step jump variability parameter r")
    plt.title(
        f"Step PSTH distance from fixed ramp PSTH\n"
        f"Ramp: beta={beta}, sigma={sigma}"
    )
    plt.legend()
    plt.show()

    # Plot best PSTH overlay
    nBins = len(rampPSTH)
    time = (np.arange(nBins) + 0.5) * binWidth / T

    plt.figure(figsize=(7, 4))
    plt.plot(time, rampPSTH, label="Fixed ramp PSTH")
    plt.plot(time, bestStepPSTH, linestyle="--", label="Best matching step PSTH")

    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate, Hz")
    plt.title(
        f"Best step match: m={bestM:.1f}, r={bestR:.2f}, distance={bestDistance:.3f}"
    )
    plt.xlim(0, 1)
    plt.legend()
    plt.show()

    return distances, mValues, rValues, bestM, bestR, bestDistance

def rampMatchContourForFixedStep(m=469,r=3,x0=0.2,Rh=50,betaValues=None,sigmaValues=None,Ntrials=5000,T=1000,binWidth=50,smoothWindow=3):
    # Plots contour map of RMSE when finding matching ramp model for fixed step model

    if betaValues is None:
        betaValues = np.linspace(0.1, 3, 15)

    if sigmaValues is None:
        sigmaValues = np.linspace(0.05, 2, 15)

    # Simulate fixed step model
    step = StepModel(m=m, r=r, x0=x0, Rh=Rh)
    stepSpikes, _, _ = step.simulate(Ntrials=Ntrials, T=T)
    stepPSTH = psthVector(stepSpikes, binWidth=binWidth, smoothWindow=smoothWindow)

    distances = np.zeros((len(sigmaValues), len(betaValues)))

    bestDistance = np.inf
    bestBeta = None
    bestSigma = None
    bestRampPSTH = None

    for i, sigma in enumerate(sigmaValues):
        for j, beta in enumerate(betaValues):
            ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)
            rampSpikes, _, _ = ramp.simulate(Ntrials=Ntrials, T=T)
            rampPSTH = psthVector(rampSpikes, binWidth=binWidth, smoothWindow=smoothWindow)

            distance = rmseDistance(stepPSTH, rampPSTH)
            distances[i, j] = distance

            if distance < bestDistance:
                bestDistance = distance
                bestBeta = beta
                bestSigma = sigma
                bestRampPSTH = rampPSTH

    # Plot contour map
    B, S = np.meshgrid(betaValues, sigmaValues)

    plt.figure(figsize=(8, 5))

    contour = plt.contourf(B, S, distances, levels=20)
    plt.colorbar(contour, label="PSTH distance / RMSE")

    plt.scatter(bestBeta,bestSigma,marker="x",s=100,color="black",label="Best match")

    plt.xlabel("Ramp drift parameter beta")
    plt.ylabel("Ramp noise parameter sigma")
    plt.title(
        f"Ramp PSTH distance from fixed step PSTH\n"
        f"Step: m={m}, r={r}"
    )
    plt.legend()
    plt.show()

    # Plot best PSTH overlay
    nBins = len(stepPSTH)
    time = (np.arange(nBins) + 0.5) * binWidth / T

    plt.figure(figsize=(7, 4))
    plt.plot(time, stepPSTH, label="Fixed step PSTH")
    plt.plot(time, bestRampPSTH, linestyle="--", label="Best matching ramp PSTH")

    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate, Hz")
    plt.title(
        f"Best ramp match: beta={bestBeta:.2f}, sigma={bestSigma:.2f}, "
        f"distance={bestDistance:.3f}"
    )
    plt.xlim(0, 1)
    plt.legend()
    plt.show()

    return distances, betaValues, sigmaValues, bestBeta, bestSigma, bestDistance



def spikeRasterPlot(spikes, T=None, jumps=None, title="Spike raster plot"):
    if T is None:
        T = spikes.shape[1]

    Ntrials = spikes.shape[0]

    trial_idx, time_idx = np.nonzero(spikes)

    counts = spikes[trial_idx, time_idx]

    trial_raster = np.repeat(trial_idx + 1, counts)
    time_raster = np.repeat(time_idx, counts)

    plt.figure(figsize=(8, 4))

    # Plot spikes
    plt.scatter(time_raster, trial_raster, s=8, marker="|")

    # Plot jumps only if they are provided
    if jumps is not None:
        jumps = np.asarray(jumps)

        # Only plot jumps inside the trial
        valid = jumps < T

        plt.scatter(
            jumps[valid],
            (np.arange(Ntrials) + 1)[valid],
            s=20,
            marker="x"
        )

    plt.xlabel("Time bin")
    plt.ylabel("Trial")
    plt.title(title)

    plt.ylim(0.5, Ntrials + 0.5)
    plt.xlim(0, T)

    plt.show()



def histogramPlot(data, binWidth=5, model="S", T=None, Rh=None , title= None, showMedian=True):               

    data = np.asarray(data)
    model = model.upper()

    if model == "S":
        # Step model: data is already jump times
        times = data

        if T is None:
            T = int(np.max(times)) + 1

        # Remove jumps outside the trial
        #stimes = times[times < T]

        xlabel = "Jump time bin"
        if title == None:
            title = "Histogram of Step Model jump times"

    elif model == "R":
        # Ramp model: data should be xs or rates, shape (Ntrials, T)
        if T is None:
            T = data.shape[1]

        if Rh is None:
            # Assume data is xs
            threshold = 1
            reached = data >= threshold
            xlabel = "Time bin when x first reaches 1"
            if title == None:
                title = "Histogram of RampModel threshold times"
        else:
            # Assume data is rates
            threshold = Rh
            reached = data >= threshold
            xlabel = "Time bin when rate first reaches Rh"
            if title == None:
                title = "Histogram of Ramp Model rate-saturation times"

        # Which trials reached the threshold?
        hit = np.any(reached, axis=1)

        # First time each valid trial reached threshold
        times = np.argmax(reached[hit], axis=1)

    else:
        raise ValueError("model must be 'S' or 'R'")

    # Safety: remove NaNs if any
    times = np.asarray(times)
    times = times[~np.isnan(times)]

    # Make bin edges
    bins = np.arange(0, T + binWidth, binWidth)

    # Use lo_histogram instead of plt.hist
    counts, _ = lo_histogram(times, bins)

    plt.figure(figsize=(7, 4))

    plt.bar(
        bins[:-1],
        counts,
        width=np.diff(bins),
        align="edge",
        edgecolor="black"
    )
    if showMedian and len(times) > 0:
        medianTime = np.median(times)

        plt.axvline(
            medianTime,
            linestyle="--",
            linewidth=2,
            label=f"Median = {medianTime:.1f}"
        )

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.show()


def trajectoriesPlot(data , title = "Trajectories over time"):
    fig,ax = plt.subplots()

    T = data.shape[1]
    time = np.arange(T) / T
    ax.plot(time,data.T)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trajectory value")
    ax.set_title(title)
    plt.show()

def psthPlot(spikeData, binWidth=5,pad=True,windowSize = 3):
    T = spikeData.shape[1]
    Ntrials = spikeData.shape[0]

    T_trimmed = (T // binWidth) * binWidth
    spikeData = spikeData[:, :T_trimmed]

    binned = spikeData.reshape(Ntrials, -1, binWidth)

    averageCounts = binned.sum(axis=2).mean(axis=0)

    dt = 1 / T
    binDuration = binWidth * dt

    averageRate = averageCounts / binDuration
    setMode = 'same'

    if pad == True:                                         # this just prevents the first few and last few of values being smaller due to the rectangular window and how the convolution function is coded. (Assumes that its a value of 0 outside the signal )
        padAmount = windowSize // 2
        averageRate = np.pad(averageRate, pad_width=padAmount, mode="edge")
        setMode = 'valid'
    #print(averageRate)
    #print(averageRate.shape)

    window = signal.windows.boxcar(windowSize)
    window = window / window.sum()   # to normalise 
    smoothedRate = signal.convolve(averageRate,window,mode=setMode)

    #print(smoothedRate)
    #print(smoothedRate.shape)

    bins = np.arange(0, T_trimmed + binWidth, binWidth)
    time = bins[:-1] / T

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        time,
        smoothedRate
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate, spikes/s")
    ax.set_title("PSTH")

    plt.show()

def fanoValue(spikeData, binWidth = 5):
    T = spikeData.shape[1]
    Ntrials = spikeData.shape[0]

    T_trimmed = (T // binWidth) * binWidth
    spikeData = spikeData[:, :T_trimmed]

    binned = spikeData.reshape(Ntrials, -1, binWidth)

    binnedCounts = binned.sum(axis=2)

    meanCounts = binnedCounts.mean(axis=0)
  
    varianceCounts = binnedCounts.var(axis=0,ddof=1)
    """sampleVariance = []                          
    for i in range(meanCounts.shape[0]):
        #print(i)
        sampleVariance_i = []
        for j in range(Ntrials):
            sampleVariance_j = (binnedCounts[j,i] - meanCounts[i]) ** 2
            sampleVariance_i.append(sampleVariance_j)
        
        sampleVariance.append(sum(sampleVariance_i)/(Ntrials-1))

    fanoFactor = sampleVariance / meanCounts"""

    fanoFactor = np.where(meanCounts > 0, varianceCounts / meanCounts, np.nan)

    return fanoFactor

def fanoPlot(
    fanoFactorList,
    T,
    binWidth=5,
    title="Fano values against time",
    eventTimes=None,
    eventLabel="Median event time"
):
    nBins = fanoFactorList.shape[0]

    # Bin centres in seconds
    time = (np.arange(nBins) + 0.5) * binWidth / T

    plt.figure(figsize=(7, 4))

    plt.plot(time, fanoFactorList, label="Fano factor")
    plt.axhline(1, linestyle="--", color="black", label="Poisson baseline")

    # Optional vertical median line
    if eventTimes is not None:
        eventTimes = np.asarray(eventTimes)

        # Keep only valid event times inside the trial
        eventTimes = eventTimes[(eventTimes >= 0) & (eventTimes < T)]

        if len(eventTimes) > 0:
            medianTime = np.median(eventTimes)
            medianTimeSeconds = medianTime / T

            plt.axvline(
                medianTimeSeconds,
                linestyle="--",
                linewidth=2,
                label=f"{eventLabel} = {medianTimeSeconds:.3f}s"
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Fano factor")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
    
def getRampHitTimesFromRates(rates, Rh=50):
    reached = rates >= Rh
    hit = np.any(reached, axis=1)

    hitTimes = np.full(rates.shape[0], np.nan)
    hitTimes[hit] = np.argmax(reached[hit], axis=1)

    return hitTimes

def rateMeanVariancePlot(rates,T,Rh=50,title="Mean and variance of rate",showMedian=True):
    rates = np.asarray(rates)

    time = np.arange(rates.shape[1]) / T

    meanRate = rates.mean(axis=0)
    varianceRate = rates.var(axis=0, ddof=1)

    plt.figure(figsize=(7, 4))

    plt.plot(time, meanRate, label="Mean rate")
    plt.plot(time, varianceRate, linestyle="--", label="Rate variance")

    if showMedian:
        # Find first time each trial hits max rate
        reached = rates >= Rh
        hit = np.any(reached, axis=1)

        if np.any(hit):
            hitTimes = np.full(rates.shape[0], T)
            hitTimes[hit] = np.argmax(reached[hit], axis=1)
            medianTime = np.median(hitTimes)
            medianTimeSeconds = medianTime / T

            plt.axvline(
                medianTimeSeconds,
                linestyle="--",
                linewidth=2,
                label=f"Median hit time = {medianTimeSeconds:.3f}s"
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()
    plt.show()

def psthValues(spikeData, binWidth=5, pad=True, windowSize=3):
    T = spikeData.shape[1]
    Ntrials = spikeData.shape[0]

    T_trimmed = (T // binWidth) * binWidth
    spikeData = spikeData[:, :T_trimmed]

    binned = spikeData.reshape(Ntrials, -1, binWidth)

    averageCounts = binned.sum(axis=2).mean(axis=0)

    dt = 1 / T
    binDuration = binWidth * dt

    averageRate = averageCounts / binDuration
    setMode = "same"

    if pad:
        padAmount = windowSize // 2
        averageRate = np.pad(averageRate, pad_width=padAmount, mode="edge")
        setMode = "valid"

    window = signal.windows.boxcar(windowSize)
    window = window / window.sum()

    smoothedRate = signal.convolve(averageRate, window, mode=setMode)

    bins = np.arange(0, T_trimmed + binWidth, binWidth)
    time = (bins[:-1] + binWidth / 2) / T   # bin centres in seconds

    return time, smoothedRate

def comparePSTHPlot(
    spikeData1,
    spikeData2,
    binWidth=5,
    pad=True,
    windowSize=3,
    label1="PSTH 1",
    label2="PSTH 2",
    title="Comparison of PSTHs"
):
    time1, psth1 = psthValues(spikeData1, binWidth=binWidth, pad=pad, windowSize=windowSize)
    time2, psth2 = psthValues(spikeData2, binWidth=binWidth, pad=pad, windowSize=windowSize)

    plt.figure(figsize=(7, 4))

    plt.plot(time1, psth1, label=label1)
    plt.plot(time2, psth2, linestyle="--", label=label2)

    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()
    plt.show()


def randomSpikeSample(timeA,trialA = 400):      # for task 1.4

    ranX_0 = np.random.uniform(0,0.5)

    choice = ['ramp','step']
    ranChoice = random.choice(choice)

    if ranChoice == 'ramp':
        ranBeta = np.random.uniform(0,4)
        logRanSigma = np.random.uniform(np.log(0.04),np.log(4))
        ranSigma = np.exp(logRanSigma)
        randomModel = RampModel(beta=ranBeta,sigma=ranSigma,x0 = ranX_0)

    else:
        ranM = np.random.uniform(0.25 * timeA , 0.75 * timeA)
        ranR = np.random.uniform(0.5,6)
        randomModel = StepModel(m=ranM,r=ranR, x0=ranX_0)
    
    randomSpikes, _, _ = randomModel.simulate(Ntrials=trialA, T=timeA)

    return randomSpikes , ranChoice

#=============================================
#=============================================

"""
timeLength = 1000

trialsDone = 5000

step = StepModel(m=500 ,r=400)  #, isi_gamma_shape = 1
stepspikes, stepjumps, steprates = step.simulate(Ntrials=trialsDone, T=timeLength)

ramp = RampModel(beta=1.1, sigma = 0.5)  #isi_gamma_shape = 0.01
rampspikes,rampxs,ramprates = ramp.simulate(Ntrials=trialsDone, T=timeLength)

step2 = StepModel(m=479 , r=1.43)
step2spikes, step2jumps, step2rates = step2.simulate(Ntrials=trialsDone, T=timeLength)

ramp2 = RampModel(beta=1,sigma=2)
ramp2spikes,ramp2xs,ramp2rates = ramp2.simulate(Ntrials=10, T=timeLength)
"""
#ranSpikes , trueLabel = randomSpikeSample(timeA = timeLength ,trialA = 400)



#spikeRasterPlot(rampspikes ,timeLength, jumps = None , title= "Ramp Model Spike Raster (β = 2,  σ = 0.8)")  #Step Model Spike Raster (m = 20, r = 100)
    #histogramPlot(steprampspikesjumps, model='S', T=timeLength , title = "Histogram of latent jump times (m = 50, r = 10)   " )    # Histogram of latent jump times (m = 20, r = 100)       
#histogramPlot(stepjumps, model='S', T=timeLength, title = 'Histogram of times jump occurs (m = 500, r = 10)  ', showMedian= False)
#trajectoriesPlot(steprates, title = "Trajectories of step model latent variable (m = 500,  r = 400)    ") # Trajectories of step model rates (m = 20, r = 100)    
    #
#psthPlot(rampspikes, binWidth=5, pad = True, windowSize = 3)    #To show it quantitatively, run multiple simulated datasets for different trial counts and compute the standard deviation of the PSTH across repeated datasets.
#psthPlot(step2spikes, pad = True, windowSize = 3)
#fanoPlot(fanoValue(rampspikes, binWidth= 50),T = timeLength, binWidth=50, title = 'Fano value against time') #title = 'Fano value against time for step model (m=500 , r=500)'
#hitTimes = getRampHitTimesFromRates(ramprates, Rh=50)
#fanoPlot(fanoValue(stepspikes,binWidth= 50),T = timeLength, binWidth= 50,  title = 'Fano value against time of step model (m = 500,  r = 10)' ,eventTimes=stepjumps, eventLabel="Median histogram time")
#histogramPlot(stepjumps, model='S', T=timeLength, title = 'Histogram of times when jumps occur (m = 1000,  r = 0.2)  ', showMedian=True)
#trajectoriesPlot(ramp2rates)
#rateMeanVariancePlot(steprates, T=timeLength , title= 'Mean and Variance of step rates (m = 1000 , r = 0.2)',showMedian=True)
#comparePSTHPlot(rampspikes,step2spikes,binWidth=5, windowSize=3,label1="Ramp PSTH",label2="Step PSTH",title=" Ramp vs Distance minimised step PSTH")


"""
compareFanoTwoParameters(
    modelType="ramp",
    param1Name="beta",
    param1Values=[0.5, 1.0, 2.5],
    param2Name="sigma",
    param2Values=[0.1, 0.5, 1.0, 2.0 ,5],
    baseParams={
        "x0": 0.2,
        "Rh": 50
    },
    Ntrials=5000,
    T=1000,
    fanoBinWidth=50
) """

"""
compareFanoTwoParameters(
    modelType="step",
    param1Name="m",
    param1Values=[200, 500, 1000],
    param2Name="r",
    param2Values=[0.2, 2, 5, 10],
    baseParams={
        "x0": 0.2,
        "Rh": 50
    },
    Ntrials=5000,
    T=1000,
    fanoBinWidth=50
)"""

"""
distances, mValues, rValues, bestM, bestR, bestDistance = stepMatchContourForFixedRamp(
    beta=1.1,
    sigma=0.5,
    x0=0.2,
    Rh=50,
    mValues=np.linspace(100, 1000, 20),
    rValues=np.linspace(0.2, 8, 20),
    Ntrials=5000,
    T=1000,
    binWidth=5,
    smoothWindow=3
)"""
