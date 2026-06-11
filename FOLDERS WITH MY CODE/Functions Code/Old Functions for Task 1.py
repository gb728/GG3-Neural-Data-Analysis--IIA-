from models import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def spikeRasterPlot(spikes, T=None, jumps=None, title="Spike raster plot"):
    """
    Makes a spike raster plot.

    Parameters
    ----------
    spikes : array, shape (Ntrials, T)
        Spike counts for each trial and time bin.

    T : int, optional
        Number of time bins. If not given, inferred from spikes.

    jumps : array, optional
        Jump times for StepModel. If None, no jump markers are plotted.

    title : str
        Plot title.
    """

    if T is None:
        T = spikes.shape[1]

    Ntrials = spikes.shape[0]

    trial_idx, time_idx = np.nonzero(spikes)

    counts = spikes[trial_idx, time_idx]

    trial_raster = np.repeat(trial_idx, counts)
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
            np.arange(Ntrials)[valid],
            s=20,
            marker="x"
        )

    plt.xlabel("Time bin")
    plt.ylabel("Trial")
    plt.title(title)

    plt.ylim(-0.5, Ntrials - 0.5)
    plt.xlim(0, T)

    plt.show()



def histogramPlot(data, binWidth=5, model="S", T=None, Rh=None):                # icl i need to fix this function since its kinda trashy (dont even use lo_histogram)
    """
    Histogram for StepModel jump times or RampModel threshold-crossing times.

    Parameters
    ----------
    data : array
        For model="S": jump times, shape (Ntrials,)
        For model="R": xs or rates, shape (Ntrials, T)

    binWidth : int
        Width of histogram bins in time bins.

    model : str
        "S" for StepModel
        "R" for RampModel

    T : int or None
        Number of time bins. If None, inferred where possible.

    Rh : float or None
        Only needed if data is rates from RampModel instead of xs.
        If Rh is None, assumes data is xs and threshold is 1.
        If Rh is given, assumes data is rates and threshold is Rh.
    """

    data = np.asarray(data)
    model = model.upper()

    if model == "S":
        # Step model: data is already jump times
        times = data

        if T is None:
            T = int(np.max(times)) + 1

        # Remove jumps outside the trial
        times = times[times < T]

        xlabel = "Jump time bin"
        title = "Histogram of StepModel jump times"

    elif model == "R":
        # Ramp model: data should be xs or rates, shape (Ntrials, T)
        if T is None:
            T = data.shape[1]

        if Rh is None:
            # Assume data is xs
            threshold = 1
            reached = data >= threshold
            xlabel = "Time bin when x first reaches 1"
            title = "Histogram of RampModel threshold times"
        else:
            # Assume data is rates
            threshold = Rh
            reached = data >= threshold
            xlabel = "Time bin when rate first reaches Rh"
            title = "Histogram of RampModel rate-saturation times"

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

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.show()


def trajectoriesPlot(data):
    fig,ax = plt.subplots()

    T = data.shape[1]
    time = np.arange(T) / T
    ax.plot(time,data.T)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trajectory value")
    ax.set_title("Trajectories over time")
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

    ax.bar(
        time,
        smoothedRate,
        width=binDuration,
        align="edge",
        edgecolor="black"
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
    """sampleVariance = []                          #just use  .var()
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

def fanoPlot(fanoFactorList, T, binWidth = 5):
    nBins = fanoFactorList.shape[0]

    time = (np.arange(nBins) * binWidth) / T
    width = binWidth / T

    plt.bar(time,fanoFactorList,width = width, align = "edge", edgecolor="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Fano factor")
    plt.show()
    


trialsDone = 5000
timeLength = 1000
step = StepModel(m=timeLength/2,r=timeLength/750)  #, isi_gamma_shape = 1
stepspikes, stepjumps, steprates = step.simulate(Ntrials=trialsDone, T=timeLength)

ramp = RampModel(beta=1.1,sigma=0.5, isi_gamma_shape = 0.01) 
rampspikes,rampxs,ramprates = ramp.simulate(Ntrials=trialsDone, T=timeLength)

step2 = StepModel(m=timeLength/2 , r=timeLength/750)
step2spikes, step2jumps, step2rates = step2.simulate(Ntrials=trialsDone, T=timeLength)

ramp2 = RampModel(beta=1.1,sigma=0.5)
ramp2spikes,ramp2xs,ramp2rates = ramp2.simulate(Ntrials=trialsDone, T=timeLength)

#spikeRasterPlot(rampspikes ,timeLength)
#histogramPlot(rampxs, model='R', T=timeLength )           
#histogramPlot(ramprates, model='R', T=timeLength,Rh=50)
#trajectoriesPlot(rampxs)
#trajectoriesPlot(ramprates)
#psthPlot(stepspikes, pad = True, windowSize = 3)
#psthPlot(rampspikes, pad = True, windowSize = 3)
fanoPlot(fanoValue(rampspikes),T = timeLength)
fanoPlot(fanoValue(ramp2spikes),T = timeLength)