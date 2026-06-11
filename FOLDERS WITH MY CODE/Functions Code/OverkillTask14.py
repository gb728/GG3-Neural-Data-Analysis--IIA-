
from functionsForTask1 import *


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

from joblib import Parallel, delayed                #this might be overkill....





#=========new redo of actually investigating the

def smoothArray(x, windowSize=3):
    if windowSize <= 1:
        return np.asarray(x)

    padAmount = windowSize // 2
    xPadded = np.pad(x, pad_width=padAmount, mode="edge")

    window = np.ones(windowSize) / windowSize
    return np.convolve(xPadded, window, mode="valid")


def getPSTHAndFano(spikeData, binWidth=50, smoothWindow=3):
    T = spikeData.shape[1]
    Ntrials = spikeData.shape[0]

    T_trimmed = (T // binWidth) * binWidth
    spikeData = spikeData[:, :T_trimmed]

    binned = spikeData.reshape(Ntrials, -1, binWidth)
    binnedCounts = binned.sum(axis=2)

    meanCounts = binnedCounts.mean(axis=0)
    varianceCounts = binnedCounts.var(axis=0, ddof=1)

    dt = 1 / T
    binDuration = binWidth * dt

    psth = meanCounts / binDuration

    fano = np.where(meanCounts > 0, varianceCounts / meanCounts, np.nan)
    fano = np.nan_to_num(fano, nan=0.0)

    psth = smoothArray(psth, smoothWindow)
    fano = smoothArray(fano, smoothWindow)

    return psth, fano, binDuration

# general optimisation of a feature:

def optimiseThresholdForFeature(
    featureFunction,
    featureName,
    rampIf,
    timeA=1000,
    trialA=200,
    nTrain=300,
    nTest=100,
    binWidth=50,
    smoothWindow=3
):

    trainFeatures = []
    trainLabels = []

    for _ in range(nTrain):
        spikes, trueLabel = randomSpikeSample(timeA=timeA, trialA=trialA)

        featureValue = featureFunction(
            spikes,
            binWidth=binWidth,
            smoothWindow=smoothWindow
        )

        trainFeatures.append(featureValue)
        trainLabels.append(1 if trueLabel == "ramp" else 0)

    trainFeatures = np.array(trainFeatures)
    trainLabels = np.array(trainLabels)

    uniqueValues = np.sort(np.unique(trainFeatures))

    if len(uniqueValues) == 1:
        candidateThresholds = uniqueValues
    else:
        candidateThresholds = (uniqueValues[:-1] + uniqueValues[1:]) / 2

    bestAccuracy = -1
    bestThreshold = None

    for threshold in candidateThresholds:
        if rampIf == "low":
            predictions = (trainFeatures < threshold).astype(int)
        elif rampIf == "high":
            predictions = (trainFeatures > threshold).astype(int)
        else:
            raise ValueError("rampIf must be 'low' or 'high'")

        accuracy = np.mean(predictions == trainLabels)

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestThreshold = threshold

    testFeatures = []
    testLabels = []

    for _ in range(nTest):
        spikes, trueLabel = randomSpikeSample(timeA=timeA, trialA=trialA)

        featureValue = featureFunction(
            spikes,
            binWidth=binWidth,
            smoothWindow=smoothWindow
        )

        testFeatures.append(featureValue)
        testLabels.append(1 if trueLabel == "ramp" else 0)

    testFeatures = np.array(testFeatures)
    testLabels = np.array(testLabels)

    if rampIf == "low":
        testPredictions = (testFeatures < bestThreshold).astype(int)
        ruleText = f"if {featureName} < {bestThreshold:.4f}, predict ramp"
    else:
        testPredictions = (testFeatures > bestThreshold).astype(int)
        ruleText = f"if {featureName} > {bestThreshold:.4f}, predict ramp"

    testAccuracy = np.mean(testPredictions == testLabels)

    stepMask = testLabels == 0
    rampMask = testLabels == 1

    stepAccuracy = np.mean(testPredictions[stepMask] == 0) if np.any(stepMask) else np.nan
    rampAccuracy = np.mean(testPredictions[rampMask] == 1) if np.any(rampMask) else np.nan

    print(featureName)
    print("-" * len(featureName))
    print("Rule:", ruleText)
    print(f"Training accuracy: {bestAccuracy:.3f}")
    print(f"Test accuracy:     {testAccuracy:.3f}")
    print(f"Step accuracy:     {stepAccuracy:.3f}")
    print(f"Ramp accuracy:     {rampAccuracy:.3f}")

    return {
        "feature_name": featureName,
        "threshold": bestThreshold,
        "ramp_if": rampIf,
        "train_accuracy": bestAccuracy,
        "test_accuracy": testAccuracy,
        "step_accuracy": stepAccuracy,
        "ramp_accuracy": rampAccuracy,
        "test_features": testFeatures,
        "test_labels": testLabels,
        "test_predictions": testPredictions
    }


#different features of interest
def psthAreaFeature(spikeData, binWidth=50, smoothWindow=3):
    psth, fano, binDuration = getPSTHAndFano(
        spikeData,
        binWidth=binWidth,
        smoothWindow=smoothWindow
    )

    return np.sum(psth) * binDuration

def maxPSTHAdjacentIncreaseFeature(spikeData, binWidth=50, smoothWindow=3):
    psth, fano, binDuration = getPSTHAndFano(
        spikeData,
        binWidth=binWidth,
        smoothWindow=smoothWindow
    )

    psthDiff = np.diff(psth)

    if len(psthDiff) == 0:
        return 0.0

    return np.max(psthDiff)

def fanoMeanFeature(spikeData, binWidth=50, smoothWindow=3):
    psth, fano, binDuration = getPSTHAndFano(
        spikeData,
        binWidth=binWidth,
        smoothWindow=smoothWindow
    )

    return np.mean(fano)

def fanoMaxFeature(spikeData, binWidth=50, smoothWindow=3):
    psth, fano, binDuration = getPSTHAndFano(
        spikeData,
        binWidth=binWidth,
        smoothWindow=smoothWindow
    )

    return np.max(fano)

def topTwoFanoAdjacentChangeFeature(spikeData, binWidth=50, smoothWindow=3):
    psth, fano, binDuration = getPSTHAndFano(
        spikeData,
        binWidth=binWidth,
        smoothWindow=smoothWindow
    )

    fanoDiff = np.abs(np.diff(fano))

    if len(fanoDiff) == 0:
        return 0.0

    sortedDiffs = np.sort(fanoDiff)[::-1]

    largest = sortedDiffs[0]

    if len(sortedDiffs) > 1:
        secondLargest = sortedDiffs[1]
    else:
        secondLargest = 0.0

    return (largest + secondLargest) / 2

#resultpsthArea = optimiseThresholdForFeature(psthAreaFeature, "psth_area",rampIf="high",timeA=1000,trialA=200,nTrain=2000,nTest=750,binWidth=50,smoothWindow=3)

#resultpsthAdjacent = optimiseThresholdForFeature(maxPSTHAdjacentIncreaseFeature,"max_psth_adjacent_increase",rampIf="low",timeA=1000,trialA=200,nTrain=2000,nTest=750,binWidth=50,smoothWindow=3)

#resultfanoMax = optimiseThresholdForFeature(fanoMaxFeature,"fano_max",rampIf="low",timeA=1000,trialA=200,nTrain=2000,nTest=750,binWidth=50,smoothWindow=3 )

#resulttwogreatestadjacent = optimiseThresholdForFeature(topTwoFanoAdjacentChangeFeature,"top_two_fano_adjacent_change",rampIf="low",timeA=1000,trialA=200,nTrain=2000,nTest=750,binWidth=50,smoothWindow=3)

#resultFanoMean = optimiseThresholdForFeature(fanoMeanFeature,"fano_mean",rampIf="low",timeA=1000,trialA=200,nTrain=2000,nTest=750,binWidth=50,smoothWindow=3)


#allResults = [
    resultpsthArea,
    resultpsthAdjacent,
    resultfanoMax,
    resulttwogreatestadjacent,
    resultFanoMean
#]


def saveThresholdResultsNPZ(results, filename="threshold_results.npz"):
    featureNames = []
    rampIfs = []
    thresholds = []
    trainAccuracies = []
    testAccuracies = []
    stepAccuracies = []
    rampAccuracies = []

    saveDict = {}

    for i, result in enumerate(results):
        featureNames.append(result["feature_name"])
        rampIfs.append(result["ramp_if"])
        thresholds.append(result["threshold"])
        trainAccuracies.append(result["train_accuracy"])
        testAccuracies.append(result["test_accuracy"])
        stepAccuracies.append(result["step_accuracy"])
        rampAccuracies.append(result["ramp_accuracy"])

        saveDict[f"test_features_{i}"] = result["test_features"]
        saveDict[f"test_labels_{i}"] = result["test_labels"]
        saveDict[f"test_predictions_{i}"] = result["test_predictions"]

    np.savez_compressed(
        filename,
        feature_names=np.array(featureNames),
        ramp_ifs=np.array(rampIfs),
        thresholds=np.array(thresholds),
        train_accuracies=np.array(trainAccuracies),
        test_accuracies=np.array(testAccuracies),
        step_accuracies=np.array(stepAccuracies),
        ramp_accuracies=np.array(rampAccuracies),
        n_results=np.array(len(results)),
        **saveDict
    )

    print(f"Saved results to {filename}")

#saveThresholdResultsNPZ(allResults, "threshold_results.npz")


def loadThresholdResultsNPZ(filename="threshold_results.npz"):
    data = np.load(filename, allow_pickle=True)

    nResults = int(data["n_results"])

    results = []

    for i in range(nResults):
        result = {
            "feature_name": str(data["feature_names"][i]),
            "ramp_if": str(data["ramp_ifs"][i]),
            "threshold": float(data["thresholds"][i]),
            "train_accuracy": float(data["train_accuracies"][i]),
            "test_accuracy": float(data["test_accuracies"][i]),
            "step_accuracy": float(data["step_accuracies"][i]),
            "ramp_accuracy": float(data["ramp_accuracies"][i]),
            "test_features": data[f"test_features_{i}"],
            "test_labels": data[f"test_labels_{i}"],
            "test_predictions": data[f"test_predictions_{i}"]
        }

        results.append(result)

    print(f"Loaded results from {filename}")

    return results

def plotFeatureResultsBarChart(results):
    featureNames = [r["feature_name"] for r in results]
    testAccuracies = [r["test_accuracy"] for r in results]
    stepAccuracies = [r["step_accuracy"] for r in results]
    rampAccuracies = [r["ramp_accuracy"] for r in results]

    x = np.arange(len(featureNames))
    width = 0.25

    plt.figure(figsize=(10, 5))

    bars1 = plt.bar(x - width, testAccuracies, width=width,edgecolor="black", label="Test accuracy")
    bars2 = plt.bar(x, stepAccuracies, width=width,edgecolor="black", label="Step accuracy")
    bars3 = plt.bar(x + width, rampAccuracies, width=width,edgecolor="black", label="Ramp accuracy")
    plt.bar_label(bars1, fmt="%.2f", padding=3, rotation=45)
    plt.bar_label(bars2, fmt="%.2f", padding=3, rotation=45)
    plt.bar_label(bars3, fmt="%.2f", padding=3, rotation=45)

    plt.axhline(0.5, linestyle="--", color="red", label="Accuracy = 0.5")

    plt.xticks(x, featureNames, rotation=25, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Single-feature accuracies of classifiying")
    plt.legend()
    plt.tight_layout()
    plt.show()

allResults = loadThresholdResultsNPZ("threshold_results.npz")

# Swap positions 2 and 3
allResults[2], allResults[3] = allResults[3], allResults[2]

plotFeatureResultsBarChart(allResults)

plotFeatureResultsBarChart(allResults)


#=== legacy classification stuff (but not relevant as they dont want something that outputs the right answer they just want understanding about feature to show knowledge)


def featureFinder(spikeData , binWidths = 5):   #Worse than featureFinder2 as less general
    
    T = spikeData.shape[1]
    Ntrials = spikeData.shape[0]

    T_trimmed = (T // binWidths) * binWidths
    spikeData = spikeData[:, :T_trimmed]

    binned = spikeData.reshape(Ntrials, -1, binWidths)
    binnedCounts = binned.sum(axis=2)
    meanCounts = binnedCounts.mean(axis=0)

    dt = 1 / T
    binDuration = binWidths * dt

    averageRate = meanCounts / binDuration


    #fanoValueArray = fanoValue(spikeData, binWidth=binWidths)          #just recomputing fano as its quicker as we only bin the data once instead of twice if using this function.

    varianceCounts = binnedCounts.var(axis=0, ddof=1)

    fano = np.full_like(meanCounts, np.nan, dtype=float)
    valid = meanCounts > 0
    fano[valid] = varianceCounts[valid] / meanCounts[valid]
    fanoValueArray = np.nan_to_num(fano, nan=0.0)

    #print(averageRate)
    #print(averageRate.shape)
    #print(fanoValueArray)
    #print(fanoValueArray.shape)

    features = np.concatenate([averageRate,fanoValueArray])

    return features     #outputs the entire average rate for all datasets used and the fano value.

def featureFinder2(spikeData, binWidths = 5):
    T = spikeData.shape[1]
    Ntrials = spikeData.shape[0]

    T_trimmed = (T // binWidths) * binWidths
    spikeData = spikeData[:, :T_trimmed]

    binned = spikeData.reshape(Ntrials, -1, binWidths)
    binnedCounts = binned.sum(axis=2)
    meanCounts = binnedCounts.mean(axis=0)

    dt = 1 / T
    binDuration = binWidths * dt

    averageRate = meanCounts / binDuration

    varianceCounts = binnedCounts.var(axis=0, ddof=1)

    fano = np.full_like(meanCounts, np.nan, dtype=float)
    valid = meanCounts > 0
    fano[valid] = varianceCounts[valid] / meanCounts[valid]
    fanoValueArray = np.nan_to_num(fano, nan=0.0)


    lowestAverageRate = averageRate.min()
    highestAverageRate = averageRate.max()
    meanAverageRate = averageRate.mean()
    lowestFanoValue = fanoValueArray.min()
    meanFanoValue = fanoValueArray.mean()
    highestFanoValue = fanoValueArray.max()

    slope = np.diff(averageRate)
    maxSlopeIndex = np.argmax(slope)
    maxSlope = slope[maxSlopeIndex]

    timeofMaxSlope = maxSlopeIndex / len(slope)

    #fanoSlope = np.diff(fanoValueArray)                     #need to go through and check this, and see if can get the slope or smth, this is wrong.
    #maxFanoSlopeIndex = np.argmax(fanoSlope)
    #maxFanoSlope = slope[maxFanoSlopeIndex]

    timeofMaxSlope = maxSlopeIndex / len(slope)


    #print(meanAverageRate)

    features2 = np.array([lowestAverageRate,highestAverageRate,meanAverageRate,maxSlope,timeofMaxSlope,lowestFanoValue,meanFanoValue,highestFanoValue]) 

    return features2

def makeDataset(nSamples, timeLength, numberOfTrials=400, binWidth=5):
    X = []
    y = []

    for _ in range(nSamples):
        spikes, label = randomSpikeSample(timeA = timeLength, trialA=numberOfTrials)

        features = featureFinder(spikes, binWidths=binWidth)

        X.append(features)
        y.append(1 if label == "ramp" else 0)

    X = np.array(X)
    y = np.array(y)

    return X, y


#using all the cores in my cpu so its faster!

def makeOneSampleFast(timeLength, numberOfTrials, binWidth):
    spikes, label = randomSpikeSample(timeA=timeLength, trialA=numberOfTrials)
    features = featureFinder2(spikes, binWidths=binWidth)
    y = 1 if label == "ramp" else 0
    return features, y


def makeDatasetFast(nSamples=1000, timeLength=100, numberOfTrials=400, binWidth=5):
    results = Parallel(n_jobs=1)(
        delayed(makeOneSampleFast)(timeLength, numberOfTrials, binWidth)
        for _ in range(nSamples)
    )

    X, y = zip(*results)

    return np.array(X), np.array(y)


"""
timeLengthUsed = 1000

ranSpikes , trueLabel = randomSpikeSample(timeA = timeLength ,trialA = 400)
#print(ranSpikes)
#print(ranSpikes.shape)
#print('wtf')
#featureFinder2(ranSpikes , binWidths= 5)


X, y = makeDatasetFast(nSamples=400, timeLength=timeLengthUsed, numberOfTrials=1, binWidth=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=100,
    random_state=42,
    eval_metric="logloss",
    n_jobs=1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

model.save_model("smallclassification-400.json")"""
