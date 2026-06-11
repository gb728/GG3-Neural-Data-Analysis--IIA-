#Functions for Task 4.
from functionsForTask3 import simulateX_tWrapper,spikeTrainsFromXt,logBayesFactor,choosingModel
from functionsForTask2 import jTFromSequence
from functionsForTask1 import psthVector,fanoValue
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


#the human decides if the psth is ramp like omfg   # I am now the monkey being tested on.


# randomly generat. I have not got time for this :)
def findingRampLikePSTH(M,T,x0,Ntrials,Rh,K,gammaShape):
    rng = np.random.default_rng(111)
    final = {}
    rows = []
    
    betaValues = np.linspace(0,4, M)
    logSigmaValues = np.linspace(np.log(0.04), np.log(4), M)
    mValues = np.linspace(0, 0.75 * T, M)
    rValues = np.arange(1, min(M, 6) + 1)

    betaValues = betaValues[(betaValues >= 0.3) & (betaValues <= 2.5)]
    logSigmaValues = logSigmaValues[(logSigmaValues>=np.log(0.04))&(logSigmaValues<=np.log(2.5))]
    mValues = mValues[(mValues >= 35) & (mValues <= 60)]
    rValues = rValues[(rValues >= 1) & (rValues <= 4)]

    attempts = 0
    while len(rows) < M:
        attempts += 1

        trueModel = rng.choice(["R", "S"], p=[0.5, 0.5])
        if trueModel == "R":
            beta = rng.choice(betaValues)
            logSigma = rng.choice(logSigmaValues)
            sigma = np.exp(logSigma)

            x_t = simulateX_tWrapper(T,x0,Ntrials,beta=beta,sigma=sigma,kValue=K,model='R')

            params = {"beta": beta,"sigma": sigma,"m": np.nan,"r": np.nan}

        else:
            m = rng.choice(mValues)
            r = int(rng.choice(rValues))

            x_t = simulateX_tWrapper(T,x0,Ntrials,m=m,r=r,model="S")

            params = {"beta": np.nan,"sigma": np.nan,"m": m,"r": r}

        spikes = spikeTrainsFromXt(x_t,T,Rh,useGammaEmission=True,gammaShape=gammaShape)



        rows.append({
            "trueModel": trueModel,
            "beta": params["beta"],
            "sigma": params["sigma"],
            "m": params["m"],
            "r": params["r"],
            "gammaShape": gammaShape,
            "spikes": spikes,
            })

    final[gammaShape] = rows

    return final

def plotAllGammaPSTHs(alldata, T, binWidth=50, smoothWindow=3):
    plt.figure(figsize=(9, 5))

    for gammaData in alldata:
        # gammaData is like {gammaShape: rows}
        gammaShape = list(gammaData.keys())[0]
        rows = gammaData[gammaShape]

        for i, row in enumerate(rows):
            spikes = row["spikes"]

            psth = psthVector(
                spikes,
                binWidth=binWidth,
                smoothWindow=smoothWindow
            )

            time = (np.arange(len(psth)) + 0.5) * binWidth / T

            plt.plot(
                time,
                psth,
                color=f"C{gammaShape - 1}",
                alpha=0.35,
                label=f"Gamma shape = {gammaShape}" if i == 0 else None
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate")
    plt.title("Ramp-like PSTHs for each gamma emission shape")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(r"Figure for Latex/Week 4/Rename2.pdf")
    plt.show()


def plotAllGammaPSTHsWithMean(alldata, T, binWidth=50, smoothWindow=3):
    fig, ax = plt.subplots(figsize=(9, 5))

    for gammaData in alldata:
        gammaShape = list(gammaData.keys())[0]
        rows = gammaData[gammaShape]

        psths = []
        colour = f"C{gammaShape - 1}"

        for row in rows:
            spikes = row["spikes"]

            psth = psthVector(
                spikes,
                binWidth=binWidth,
                smoothWindow=smoothWindow
            )
            psths.append(psth)

            time = (np.arange(len(psth)) + 0.5) * binWidth / T

            # thin individual PSTHs
            ax.plot(
                time,
                psth,
                color=colour,
                alpha=0.4,
                linewidth=1
            )

        psths = np.array(psths)
        meanPSTH = psths.mean(axis=0)

        # bold mean PSTH
        ax.plot(
            time,
            meanPSTH,
            color=colour,
            linewidth=3,
            label=f"Gamma shape = {gammaShape}"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate")
    ax.set_title("Ramp-like PSTHs for each gamma emission shape")
    ax.set_xlim(0, 1)

    # main legend for gamma colours
    legend1 = ax.legend(title="Mean PSTH by gamma shape", loc="upper left")

    # extra legend explaining line meaning
    thinHandle = mlines.Line2D([], [], color="black", alpha=0.25, linewidth=1,
                               label="Individual PSTHs")
    thickHandle = mlines.Line2D([], [], color="black", linewidth=3,
                                label="Mean PSTH")

    legend2 = ax.legend(
        handles=[thinHandle, thickHandle],
        title="Line meaning",
        loc="lower right"
    )

    ax.add_artist(legend1)

    plt.tight_layout()
    plt.show()

def confusionFromRows(rows, M, T, x0, Rh, kValue):
    confusion = np.zeros((2, 2), dtype=int)
    logBFs = []

    for i, row in enumerate(rows):
        spikes = row["spikes"]
        trueModel = row["trueModel"]

        logBF = logBayesFactor(spikes,M,T,x0,Rh,kValue=kValue,UniformPriorNotGaussianForMarginal=True)

        chosenModel = choosingModel(logBF)

        trueIndex = 0 if trueModel == "R" else 1
        chosenIndex = 0 if chosenModel == "R" else 1

        confusion[trueIndex, chosenIndex] += 1
        logBFs.append(logBF)

        print(
            f"True {trueModel}, dataset {i+1}/{len(rows)}: "
            f"chosen {chosenModel}, logBF = {logBF:.3g}"
        )

    rampTotal = confusion[0].sum()
    stepTotal = confusion[1].sum()

    rampErrorRate = confusion[0, 1] / rampTotal if rampTotal > 0 else np.nan
    stepErrorRate = confusion[1, 0] / stepTotal if stepTotal > 0 else np.nan
    overallAccuracy = (confusion[0, 0] + confusion[1, 1]) / confusion.sum()

    return confusion, rampErrorRate, stepErrorRate, overallAccuracy, np.array(logBFs)

def plotGammaResultsBarChart(resultsByGamma):
    gammaShapes = sorted(resultsByGamma.keys())

    rampErrors = [100 * resultsByGamma[g]["rampErrorRate"] for g in gammaShapes]
    stepErrors = [100 * resultsByGamma[g]["stepErrorRate"] for g in gammaShapes]
    accuracies = [100 * resultsByGamma[g]["overallAccuracy"] for g in gammaShapes]

    x = np.arange(len(gammaShapes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars1 = ax.bar(x - width, rampErrors, width, label="Ramp error rate")
    bars2 = ax.bar(x, stepErrors, width, label="Step error rate")
    bars3 = ax.bar(x + width, accuracies, width, label="Overall accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gammaShapes])

    ax.set_xlabel("Gamma shape")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Model classification performance for each gamma shape, Ntrials = 100", pad=18)

    ax.set_ylim(0, 115)

    ax.legend(loc="upper left")

    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars3, fmt="%.1f", padding=3, fontsize=9)

    fig.tight_layout(pad=2.0)
    #plt.savefig(r"Figure for Latex/Week 4/Rename5.pdf")
    plt.show()

 # 4.2 , so basically remaking the functions for task one just more spacee conscious   

    
def rasterOnAx(ax, spikes, T, jumps=None, maxTrials=None, title=""):
    if maxTrials is not None:
        spikes = spikes[:maxTrials]
        if jumps is not None: jumps = jumps[:maxTrials]

    N = spikes.shape[0]
    trial_idx, time_idx = np.nonzero(spikes)
    counts = spikes[trial_idx, time_idx]

    ax.scatter(np.repeat(time_idx, counts), np.repeat(trial_idx + 1, counts), s=8, marker="|")

    if jumps is not None:
        valid = jumps < T
        ax.scatter(jumps[valid], (np.arange(N) + 1)[valid], s=18, marker="x", color="orange")

    ax.set(title=title, xlim=(0, T), ylim=(0.5, N + 0.5), xlabel="Time bin", ylabel="Trial")


def fourGammaRasterPlot(T, x0, Ntrials, Rh, gammaShapes=(1, 2, 3, 4), model="R",K=None, beta=None, sigma=None, m=None, r=None,maxTrials=50, savePath=None):

    model = model.upper()

    if model == "R":
        x_t = simulateX_tWrapper(T, x0, Ntrials, beta=beta, sigma=sigma, kValue=K, model="R")
        jumps = None
        mainTitle = rf"Ramp model: $\beta={beta}$, $\sigma={sigma}$"

    elif model == "S":
        x_t = simulateX_tWrapper(T, x0, Ntrials, m=m, r=r, model="S")
        jumps = jTFromSequence(x_t, x0)
        mainTitle = rf"Step model: $m={m}$, $r={r}$"

    else:
        raise ValueError("model must be 'R' or 'S'")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, gammaShape in zip(axes, gammaShapes):
        spikes = spikeTrainsFromXt(x_t, T, Rh, useGammaEmission=True, gammaShape=gammaShape)
        rasterOnAx(ax, spikes, T, jumps=jumps, maxTrials=maxTrials, title=f"Gamma shape = {gammaShape}")

    fig.suptitle(mainTitle, fontsize=14)
    fig.tight_layout()

    if savePath is not None:
        fig.savefig(savePath, bbox_inches="tight")

    plt.show()
    return x_t

def fourGammaPSTHPlot(T, x0, Ntrials, Rh, gammaShapes=(1, 2, 3, 4), model="R",K=None, beta=None, sigma=None, m=None, r=None,binWidth=50, smoothWindow=3, savePath=None):
    model = model.upper()

    if model == "R":
        x_t = simulateX_tWrapper(T, x0, Ntrials, beta=beta, sigma=sigma, kValue=K, model="R")
        title = rf"PSTH for ramp model: $\beta={beta}$, $\sigma={sigma}$"

    elif model == "S":
        x_t = simulateX_tWrapper(T, x0, Ntrials, m=m, r=r, model="S")
        title = rf"PSTH for step model: $m={m}$, $r={r}$"

    else:
        raise ValueError("model must be 'R' or 'S'")

    plt.figure(figsize=(8, 5))

    for gammaShape in gammaShapes:
        spikes = spikeTrainsFromXt(x_t, T, Rh, useGammaEmission=True, gammaShape=gammaShape)

        psth = psthVector(spikes, binWidth=binWidth, smoothWindow=smoothWindow)
        time = (np.arange(len(psth)) + 0.5) * binWidth / T

        plt.plot(time[1:], psth[1:], linewidth=2, label=f"Gamma shape = {gammaShape}")

    plt.xlabel("Time (s)")
    plt.xlim(time[1], 1)
    plt.ylabel("Firing rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()

    if savePath is not None:
        plt.savefig(savePath, bbox_inches="tight")

    plt.show()

    return x_t

def fourGammaFanoPlot(T, x0, Ntrials, Rh, gammaShapes=(1, 2, 3, 4), model="R",K=None, beta=None, sigma=None, m=None, r=None,binWidth=50, savePath=None):
    model = model.upper()

    if model == "R":
        x_t = simulateX_tWrapper(T, x0, Ntrials,beta=beta, sigma=sigma, kValue=K, model="R")
        title = rf"Fano factor for ramp model: $\beta={beta}$, $\sigma={sigma}$"

    elif model == "S":
        x_t = simulateX_tWrapper(T, x0, Ntrials,m=m, r=r, model="S")
        title = rf"Fano factor for step model: $m={m}$, $r={r}$"

    else:
        raise ValueError("model must be 'R' or 'S'")

    plt.figure(figsize=(8, 5))

    for gammaShape in gammaShapes:
        spikes = spikeTrainsFromXt(x_t, T, Rh,useGammaEmission=True,gammaShape=gammaShape)

        fano = fanoValue(spikes, binWidth=binWidth)
        time = (np.arange(len(fano)) + 0.5) * binWidth / T

        plt.plot(time,fano,marker="o",label=f"Gamma shape = {gammaShape}")

    plt.axhline(1, linestyle="--", color="black", label="Poisson baseline")
    plt.xlabel("Time (s)")
    plt.ylabel("Fano factor")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()

    if savePath is not None:
        plt.savefig(savePath, bbox_inches="tight")

    plt.show()

    return x_t