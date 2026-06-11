from OverkillTask14 import *
import matplotlib.pyplot as plt


SEED = 13

np.random.seed(SEED)
random.seed(SEED)

loaded_modelbig = xgb.XGBClassifier()
loaded_modelbig.load_model("step_vs_ramp_xgb1000000.json")

loaded_modelsmall = xgb.XGBClassifier()
loaded_modelsmall.load_model("UsingfeatureFinder2-10000.json")

loaded_modelfinally = xgb.XGBClassifier()
loaded_modelfinally.load_model("UsingfeatureFinder2withmorefeatures-10000.json")

timeLengthUsed = 1000



def featureFinderOld(spikeData, binWidths = 5):
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

    #print(meanAverageRate)

    features2 = np.array([lowestAverageRate,highestAverageRate,meanAverageRate,timeofMaxSlope,lowestFanoValue,meanFanoValue,highestFanoValue])

    return features2


newSpikes, trueLabel = randomSpikeSample(timeA=timeLengthUsed, trialA=1)
newFeatures = featureFinder(newSpikes, binWidths=5).reshape(1, -1)
newFeatures2 = featureFinder2(newSpikes, binWidths=5).reshape(1, -1)
newFeaturesold =featureFinderOld(newSpikes, binWidths=5).reshape(1, -1)

pred = loaded_modelbig.predict(newFeatures)[0]
probs = loaded_modelbig.predict_proba(newFeatures)[0]

predS = loaded_modelsmall.predict(newFeaturesold)[0]
probsS = loaded_modelsmall.predict_proba(newFeaturesold)[0]

predF = loaded_modelfinally.predict(newFeatures2)[0]
probsF = loaded_modelfinally.predict_proba(newFeatures2)[0]

print("True:", trueLabel)
print("Predicted:", "ramp" if pred == 1 else "step")
print("P(step), P(ramp):", probs)
print('below is for small:')
print("Predicted:", "ramp" if predS == 1 else "step")
print("P(step), P(ramp):", probsS)
print("Predicted:", "ramp" if predF == 1 else "step")
print("P(step), P(ramp):", probsF)

xgb.plot_importance(loaded_modelsmall, max_num_features=20)
plt.show()

#trialsDone = 400
#step = StepModel(m=timeLength/2,r=timeLength/750)  #, isi_gamma_shape = 1
#stepspikes, stepjumps, steprates = step.simulate(Ntrials=trialsDone, T=timeLength)

#spikeRasterPlot(newSpikes, T= timeLengthUsed)