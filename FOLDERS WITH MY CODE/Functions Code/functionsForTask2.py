#First question, sure you can use fwdbwd algo if its discrete hmm ... you can just use kalman filtering to get continuous hmms.

#Use log probabilities !!!! You will run into underflow if not and some other junk . LOOK at the note at the end of task 2.1!!!! --- ermm no!

from models import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr, logsumexp
from inference import poisson_logpdf, hmm_expected_states

SEED = 2
np.random.seed(SEED)

#==== For Task 2.1

#function for intial where you input the index and it outputs the probability for a given ramp model

def initialStatePointRamp(indexOfInterest,sigma,x0,T,kValue ):    #using point estimates of discretising (need renormalisation)
    stepSize = 1/T
    mean = (x0 * (kValue-1))
    var = stepSize*((kValue-1)*sigma)**2
    logcontinuousProb =  -0.5 * (np.log(2*np.pi*var) +((indexOfInterest-mean)**2 )/(var))  #putting gaussian and doing the log of it for stability
    return np.exp(logcontinuousProb)

def initalStateBinRamp(indexOfInterest,sigma,x0,T,kValue ): #using cdf bins to discretise (should be approx if variance is low -think about it) (no normalisation)
    stepSize = 1/T
    mean = (x0 * (kValue-1))
    std = ((kValue-1)*sigma)*(stepSize**0.5)

    if indexOfInterest == 0:
        binprob = norm.cdf(indexOfInterest+0.5,loc=mean,scale=std)
    elif indexOfInterest == kValue-1:
        binprob = 1 - norm.cdf(indexOfInterest-0.5,loc=mean,scale=std)
    else:
        upperCdf = norm.cdf(indexOfInterest+0.5,loc=mean,scale=std)
        lowerCdf = norm.cdf(indexOfInterest-0.5,loc=mean,scale=std)
        binprob = upperCdf - lowerCdf
    
    return binprob

def logDifferenceCDF(logUpperCdf,logLowerCdf):          # same as log( log cdfupper -log cdf lower)
    return (logUpperCdf) + np.log(1- np.exp(logLowerCdf -logUpperCdf))

def initialStateBinRampSafe(indexOfInterest,sigma,x0,T,kValue ): # Using log_ndtr to get .cdf safely
    stepSize = 1/T
    mean = (x0 * (kValue-1))
    std = ((kValue-1)*sigma)*(stepSize**0.5)

    #oh you code it lol
    codedIndexOfInterestHigh = (indexOfInterest + 0.5 - mean)/std
    codedIndexOfInterestLow = (indexOfInterest - 0.5 - mean)/std

    logUpperCdf = log_ndtr(codedIndexOfInterestHigh)
    logLowerCdf = log_ndtr(codedIndexOfInterestLow)
    if indexOfInterest == 0:
        logbinprob = log_ndtr(codedIndexOfInterestHigh)
    elif indexOfInterest == kValue-1:
        logbinprob = log_ndtr(-codedIndexOfInterestLow) #property of sigmoidal function
    else:
        logbinprob = logDifferenceCDF(logUpperCdf,logLowerCdf)
    
    return logbinprob


def initialProbRamp(sigma,x0,T,kValue,cdfBin = True,safe = True): # when comparing its like at k=50 they differ by 10^-3 for k=100 by 10^-4 at worst so not much of a difference
    x=[]
    if safe == True:
        for i in range(kValue):
            probabilityIndexI = initialStateBinRampSafe(i,sigma,x0,T,kValue)
            x.append(probabilityIndexI)
        x=np.array(x)
        x = x - logsumexp(x)

        
    else:

        if cdfBin == False:
            for i in range(kValue):
                probabilityIndexI = initialStatePointRamp(i,sigma,x0,T,kValue)
                x.append(probabilityIndexI)

        elif cdfBin == True:
            for i in range(kValue):
                probabilityIndexI = initalStateBinRamp(i,sigma,x0,T,kValue)
                x.append(probabilityIndexI)

        x = np.array(x)
        x = x / x.sum()

    return x

def transitionProbBinRamp(nextIndex,currentIndex,beta,sigma,T,kValue):

    #when at the maxvalue it stays there
    if currentIndex == kValue-1:        # as s can only be from 0 to kvalue - 1 !!!
        if nextIndex == currentIndex:
            binprob = 1
        else:
            binprob = 0
        
    else:
        stepSize = 1/T
        mean = currentIndex + beta * stepSize * (kValue-1)
        std = ((kValue-1)*sigma)*(stepSize**0.5)

        if nextIndex == 0:
            binprob = norm.cdf(nextIndex+0.5,loc=mean,scale=std)
        elif nextIndex == kValue-1:
            binprob = 1 - norm.cdf(nextIndex-0.5,loc=mean,scale=std)
        else:
            upperCdf = norm.cdf(nextIndex+0.5,loc=mean,scale=std)
            lowerCdf = norm.cdf(nextIndex-0.5,loc=mean,scale=std)
            binprob = upperCdf - lowerCdf
    
    return binprob

def transitionProbBinRampSafe(nextIndex,beta,sigma,T,kValue): # i want to input current index 

    #k long row vector of next indexes for current index
    columnOfTransition= np.arange(kValue) 
    stepSize = 1/T
    mean = columnOfTransition + beta * stepSize * (kValue-1)
    std = ((kValue-1)*sigma)*(stepSize**0.5)

    codedColumnsHigh = (nextIndex + 0.5 - mean)/std
    codedColumnsLow = (nextIndex - 0.5 - mean)/std

    logUpperCdf = log_ndtr(codedColumnsHigh)
    logLowerCdf = log_ndtr(codedColumnsLow)

    if nextIndex == 0:
        binProbRowVector = logUpperCdf
    elif nextIndex == kValue-1:
        binProbRowVector = log_ndtr(-codedColumnsLow)
    else:
        binProbRowVector = logDifferenceCDF(logUpperCdf,logLowerCdf)

    if nextIndex == kValue - 1:
        binProbRowVector[-1] = 0          #log(1) = 0
    else:
        binProbRowVector[-1] = -np.inf          #log(0) = -inf
    return binProbRowVector 

def transitionMatrixRampFast(beta, sigma, T, kValue):
    dt = 1 / T

    states = np.arange(kValue)

    currentStates = states[:, None]   # column shape: (K, 1)
    nextStates = states[None, :]      # row shape: (1, K)

    mean = currentStates + beta * dt * (kValue - 1)
    std = (kValue - 1) * sigma * np.sqrt(dt)

    upper = nextStates + 0.5
    lower = nextStates - 0.5

    tMat = norm.cdf(upper, loc=mean, scale=std) - norm.cdf(lower, loc=mean, scale=std)  # do it as a vector

    # first state collects everything below 0.5
    tMat[:, 0] = norm.cdf(0.5, loc=mean[:, 0], scale=std)

    # last state collects everything above K-1.5
    tMat[:, -1] = 1 - norm.cdf(kValue - 1.5, loc=mean[:, 0], scale=std)

    # absorbing final state
    tMat[-1, :] = 0
    tMat[-1, -1] = 1

    # just to remove tiny floating point errors
    tMat = tMat / tMat.sum(axis=1, keepdims=True)

    return tMat

def transitionMatrixRamp(beta,sigma,T,kValue,safe=True,fast = False):
    if safe == True:
        columns = []
        for j in range(kValue):
            column = transitionProbBinRampSafe(j,beta,sigma,T,kValue)   #outputs row vectors!!
            columns.append(column)

        logT =np.array(columns).T
        logT =logT- logsumexp(logT,axis=1, keepdims=True)

        return logT

    else:
        if fast:
            return transitionMatrixRampFast(beta,sigma,T,kValue)
        else:
            tMat =[]
            for i in range(kValue):
                tRow = []
                for j in range(kValue):
                    tInd = transitionProbBinRamp(j,i,beta,sigma,T,kValue)
                    tRow.append(tInd)

                tRow=np.array(tRow)
                tMat.append(tRow)

        return np.array(tMat)


#i want to output the the just sequence of sampled s values. Simple okay

def latentSequenceRamp(beta,sigma,x0,T,kValue,cdfBin = False ,tMat = None, initialProb = None, rng=None, safe= False):

    if initialProb is None:
        initialProb = initialProbRamp(sigma,x0,T,kValue,cdfBin = cdfBin,safe=safe)
    if rng is None:
        rng = np.random.default_rng()

    sSeq = []
    possibleKValues = np.arange(kValue) 
    sInitial = rng.choice(possibleKValues,p=initialProb)
    sSeq.append(sInitial)

    if T == 1: 
        return np.array(sSeq)

    elif T <= 0: 
        raise ValueError('sequence length has to be more than 0')
    
    else:
        if tMat is None:
            tMat = transitionMatrixRampFast(beta,sigma,T,kValue)
            print('Feeding in tMat is a lot quicker for bigger trials than not doing it')

        for i in range(T-1):
            si = rng.choice(possibleKValues, p = tMat[sSeq[-1]])     
            sSeq.append(si)
        
        return np.array(sSeq)

def simSeqRamp(Ntrials,beta,sigma,x0,T,kValue, returnS = True,cdfBin = False,safe = False):
    
    rng = np.random.default_rng()
    initialProb = initialProbRamp(sigma,x0,T,kValue,cdfBin = cdfBin, safe = False) 
    tMat = transitionMatrixRampFast(beta,sigma,T,kValue)
    severalSequences=[]
    for i in range(Ntrials):
        seqi = latentSequenceRamp(beta,sigma,x0,T,kValue,tMat = tMat, initialProb = initialProb ,cdfBin = cdfBin,rng = rng)
        severalSequences.append(seqi)

    severalSequences = np.array(severalSequences)
    if returnS == True:
        return severalSequences
    else:
        return severalSequences / (kValue - 1)
    
#just sample random walks instead of gaussians so no need for .choice or cdfs ...damn
def simSeqRampFast(Ntrials,beta,sigma,x0,T,kValue, returnS = True,):
    
    rng = np.random.default_rng()
    stepSize = 1/T

    # omfg i can just use a vector of size of the number of trials and just do that many random walks at once using vector addition fuckkkk thats smart smhh
    sSeq = []
    s0 = (kValue-1)*x0 + (kValue-1)*sigma*(stepSize**0.5)*rng.normal(size = Ntrials)
    s0Integer = np.rint(s0).astype(int)
    s0Clipped = np.clip(s0Integer, 0, kValue - 1)
    sSeq.append(s0Clipped)
    for i in range(T-1):
        si = sSeq[-1] + beta * stepSize * (kValue-1) + (kValue-1)*sigma*(stepSize**0.5)*rng.normal(size = Ntrials)
        siInteger = np.rint(si).astype(int)
        siClipped = np.clip(siInteger, 0, kValue - 1)

        #enforces that it stays at the maximum value
        siMaxState = sSeq[-1] == kValue - 1
        siClipped[siMaxState] = kValue - 1

        sSeq.append(siClipped)
        
    sSeq = np.array(sSeq).T # transpose as to make the sequence the right way around
    if returnS == True:
        return sSeq
    else:
        return sSeq / (kValue - 1)

def compareRtTrajectoriesPlot(beta,sigma,x0,T,K,Ntrials,Rh, title = 'This is the title'):
    discX_tSeq = simSeqRampFast(Ntrials ,beta,sigma,x0,T,K, returnS = False)
    discR_tSeq = discX_tSeq * Rh

    ramp = RampModel(beta=beta, sigma = sigma, x0=x0, Rh= Rh) 
    _,_,rampRates = ramp.simulate(Ntrials=Ntrials, T=T)

    fig,ax = plt.subplots()

    time = np.arange(T) / T
    for i in range(Ntrials):
        ax.plot(time,discR_tSeq[i],  color='tab:blue' ,label = 'Discrete'if i ==0 else None)
        ax.plot(time,rampRates[i], color='tab:orange' , linestyle='--',label = 'Continuous' if i ==0 else None)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trajectory value")
    ax.legend()
    ax.set_title(title)
    plt.show()

#=== For Task 2.2

# assume it starts in low state so no need for the 
def simTwoStateHomogeneousStep(m,r,T,Ntrials,x0,fast=True):
    #transition:
    probabilityOfJump = r/(m+r)
    if not fast:    # using the transition matrix and choice
        tMat = np.array([[1-probabilityOfJump,probabilityOfJump],[0,1]])
        sMat = [] 
        for i in range(Ntrials):
            sSeq = [0] # 0=low , 1=high for x_t
            
            if T > 1:           
                    rng = np.random.default_rng()
                    for j in range(T-1):
                        si = rng.choice([0,1], p = tMat[sSeq[-1]])  
                        sSeq.append(si)
            sMat.append(sSeq)
        sMat = np.array(sMat)

    else:   #can do like rejection sampling, get a uniform sample and if it is smaller than the probability reject, if its greater accept (and jump!)
        rng = np.random.default_rng()
        sMat = np.zeros((Ntrials, T), dtype=int)    # this is quicker as we just use vectors instead of another loop for each ntrials ... its always comes back to vectors
        for i in range(1,T):
            previous = sMat[:,i-1]
            
            boolJumpState = rng.random(Ntrials) < probabilityOfJump

            jumpState = boolJumpState.astype(int)

            sMat[:, i] = np.where(previous == 1, 1, jumpState) 

    xMat = np.clip(sMat,x0,1)
    return xMat


def jTFromSequence(sequence,x0): # trying to find the jump times from the sequence
    jumptimes = []          
    for i in range(sequence.shape[0]):  # i dont think this works if sequence is just 1, but i mean the output of sim2state is a matrix either way so it should be supported
        for j in range (sequence.shape[1]-1):
            difference = sequence[i,j+1] - sequence[i,j]
            if difference == (1-x0):
                jumptimes.append(j+1)
                break     
        if len(jumptimes) != i+1:
            jumptimes.append(sequence.shape[1] +1 )
            
    return np.array(jumptimes)

def transitionMatrixR1StateStep(m,r,better=True,safe=False):            # this works ... not to efficiently but works, i was thinking smth to do with the diagonal since omg you can use the identity.
    #jump prob:
    r = int(r)              # we  have an integer number of states so r must be integer so we force it here to be so!!. --->>>>> It must be integer for this model to be exact <---- TRuE!!!
    probabilityOfJump = r/(m+r)
    if safe:
        probabilityOfJump = np.log(probabilityOfJump)   # so it is log probability and a log matrix so safer
    if better:      #just use identity matrices cmon!!
        tMat = np.eye(r+1)*(1-probabilityOfJump) + np.eye(r+1,k=1)*(probabilityOfJump)
        tMat[-1, :] = 0
        tMat[-1, -1] = 1

    else:
        tMatRow = np.zeros(r+1)  #r long from 0 to r
        tMat = []
        tMatRow[0] = 1-probabilityOfJump
        tMatRow[1] = probabilityOfJump
        tMat.append(tMatRow)
        for i in range(r-1):
            tMatRow.pop(i+2)
            tMatRow.insert(0,0)
        #for the bottom row
        tMatFinalRow = np.zeros(r)
        tMatFinalRow.append(1)
        tMat.append(tMatFinalRow)
        tMat = np.array(tMat)

    return tMat

def simR1StateHomogeneousStep(m,r,T,Ntrials,x0,fast = True,exact = False):
    #jump prob:
    r= int(r)   # to ensure that its an integer!
    if not fast:
        tMat = transitionMatrixR1StateStep(m,r)
        sMat = []   # this is like the hmm we are doing its on s   (dont need s here!)
        #xMat = []   # this is the state, and this is 1 if sMat is at its last state or its 0 elsewhere.
        r = int(r)
        choice = np.arange(r + 1)

        loopamount = T-1
        if exact: #here is the difference between this markov chain and numpy in models.py. In models it outputs the number of failures that occured, whereas here in the else: we just do total number of trials. Effectively in our version we dont take into account the fact that it pysically cannot jump the first r steps. 
            loopamount = T+r-1 #so here i am implmenting a 'fix' for this difference in negative binomial outputs, by jsut running it r times longer and cutting off the first r samples where it is incapable of jumping.  

        for i in range(Ntrials):
            sSeq = [0] # 0=low , 1=high for x_t
            
            if T > 1:           
                    rng = np.random.default_rng()
                    for j in range(loopamount):
                        si = rng.choice(choice, p = tMat[int(sSeq[-1])])  
                        sSeq.append(si)
            sMat.append(sSeq)
        sMat = np.array(sMat)

    else:
        rng = np.random.default_rng()
        probabilityOfJump = r/(r+m)
        upperloop = T
        if exact:
            upperloop = T +r
        sMat = np.zeros((Ntrials, upperloop), dtype=int)    # this is quicker as we just use vectors instead of another loop for each ntrials ... its always comes back to vectors

        for i in range(1,upperloop):
            previous = sMat[:,i-1]
            
            boolJumpState = rng.random(Ntrials) < probabilityOfJump

            jumpState = boolJumpState.astype(int)

            sMat[:, i] = np.minimum(previous + jumpState, r)  #so this ensures it doesnt jump forever and gets stuck at the final state (as it thinks its [0,0,0,0,0,1-p,p] but if it jumps up we push it back to stay in the sequence)

    if exact:
        sMat = sMat[:,r:] # chopping off the extra initial values of 0

    maskMat = np.ones((Ntrials,T)) * (r-1)
    xMat = sMat - maskMat   # highest value is either  1 rest are 0 or lower!.
    xMat = np.clip(xMat,x0,1)  

    return xMat

#====== Task 2.3

#consturct the array of the emmision for each of the models. -> Its the same no?
#For ramp ->n_t|r_t poisson (r_tdt) yeah and if s_t is known x_t is known and then r_t is known. So yeah
#For step -> poisson(r_o dt) or poiss(r_h dt) depending on if low state or high state simple.

def logEmissionMat( spikes, T, Rh,kValue = -1, x0 = -1, r = -1, model = 'None'): # model is either 'S' or 'R'
    if model.upper() != 'R' and  model.upper() != 'S':
        raise ValueError("Model type must be selected as either Ramp 'R' or Step 'S' ")

    dt = 1/T

    if model.upper() == 'R':
        if kValue == -1:
            raise ValueError("Supply a Value of K")
        
        s_states = np.arange(kValue)
        lambdas = s_states * dt * Rh * 1/(kValue-1)

    if model.upper() == 'S':            
        r= int(r)   # this will come up often as a safety for anything involved with the step model.
        if x0 == -1:
            raise ValueError("Supply a Value of x0")
        if r == -1:
            raise ValueError("Supply a Value of r , to know the number of states")
        
        s_states = np.ones(r + 1) * x0
        s_states[-1] = 1
        lambdas = s_states * dt * Rh

    lls = poisson_logpdf(spikes, lambdas.copy())

    return lls

def spikeTrainsFromXt(x_t,T,Rh,useGammaEmission = False ,gammaShape = None,rng = None):
    dt = 1/T

    if useGammaEmission:
        if gammaShape is None:
            raise ValueError("t2.spikeTrainsFromXt: when using gamma emmision, apply value to parameter gammaShape  = ... ")
        spikes = np.array([gamma_isi_point_process(Rh * x_t[i] * dt, gammaShape) for i in range(x_t.shape[0])])
        return spikes
    if rng is None:
        rng = np.random.default_rng()
    spikes = rng.poisson(Rh * x_t * dt)

    return spikes # this is already an integeer as poisson should only give out integer values

# Note that the posterior is for the states given the counts not the x_t
def hmmExpectedStatesWrapper(x_t,T,Rh,x0,beta = -1,sigma = -1,kValue = -1,m= -1,r=-1,filter = False, model = 'None'):
    if model.upper() != 'R' and  model.upper() != 'S':
        raise ValueError("Model type must be selected as either Ramp 'R' or Step 'S' ")
    
    spikes = spikeTrainsFromXt(x_t,T,Rh)        # maybe i input spikes isntead of x_t, that might be a bit easier. / fix later if need be.

    if model.upper() == 'R':
        if beta == -1 or sigma == -1 or kValue == -1:
            raise ValueError("Supply a Value of: K or beta or sigma")
        initialProb = initialProbRamp(sigma,x0,T,kValue, safe = False)  #they dont want log values ??? What was the point of implementing it before??
        tMat = transitionMatrixRamp(beta,sigma,T,kValue,safe=False, fast= True)
        ll = logEmissionMat(spikes,T,Rh,kValue = kValue, model = model)       

    else:
        if m == -1 or r == -1:
            raise ValueError("Supply a Value of: m or r")
        r=int(r)
        initialProb =  np.zeros(r + 1) 
        initialProb[0] = 1         # intial probability is just the first state, as we start with 0 succesess for the nb when starting - thats it.
        tMat = transitionMatrixR1StateStep(m,r,safe=False)    
        ll = logEmissionMat(spikes,T,Rh,x0=x0,r=r,model = model)   

    posteriors = []
    normalizers = []

    #if there are several trials of the spikes then it is a matrix not an array, so poisson log pdf produces the wrong size to insert into the hmm_expected_state function so need to loop through
    for i in range(spikes.shape[0]):
        expected_states, normalizer = hmm_expected_states(initialProb,tMat,ll[i],filter = filter)


        posteriors.append(expected_states)
        normalizers.append(normalizer)
    
    return np.array(posteriors)


    #ultimately i want to return the output of the expected states from hmm_expected_state

# This is the posterior expectation in terms of the x_T given the count not s_t like the posterior probability
def x_tExpectationPosterior(s_tPosteriors, timeOfInterest,trialOfInterest,x0= -1, model = 'None'):     # as we want the expectations for time t. But 
    model = model.upper()
    if model != 'R' and  model != 'S':
        raise ValueError("Model type must be selected as either Ramp 'R' or Step 'S' ")
    
    Ntrials = s_tPosteriors.shape[0]
    T = s_tPosteriors.shape[1]

    if trialOfInterest >= Ntrials or timeOfInterest >= T:
        raise ValueError("Time of Trial of interest is outside possible values. ")

    if model == 'R':
        expectation = 0
        kValue = s_tPosteriors.shape[2]
        for i in range(s_tPosteriors.shape[2]):
            expectation += ( i * s_tPosteriors[trialOfInterest][timeOfInterest][i] )/(kValue - 1)            # wait problem is that the posterior is like a huge 3 dimension matrix ...
        
        return expectation
    
    if model == 'S':
        if x0 == -1:
            raise ValueError("Supply a Value for x0")
        r = s_tPosteriors.shape[2] - 1
        posteriorR = s_tPosteriors[trialOfInterest ][timeOfInterest][r]             # this is the probability of being in the upper rate level as they assked for.
        expectation = x0 * (1-posteriorR) + posteriorR          # they dont care about this ....
        
        return expectation, posteriorR


#function kinda breaks for lower T, if T is not 1000
#i want this funciton to take in x_t sequence then  calculate all the e[x_t]'s then plot them + return the vlaues
def gettingExpectationMat(x_t,T,Rh,x0,beta = -1,sigma = -1,kValue = -1,m= -1,r=-1,filter = False, model = 'None',plot=True, jumps = None,fast = True,title='Comparison in latent variable between expectations and ground truth'):
    # if plot is true it also plots the comparison

    model = model.upper()
    posterior = hmmExpectedStatesWrapper(x_t,T,Rh,x0,beta = beta,sigma = sigma,kValue = kValue,m= m,r=r,filter = filter, model = model)
    exceptionMat = []
    if model == 'S':
        probJumpMat = []

    T = x_t.shape[1]
    Ntrials =  x_t.shape[0]

    if model == 'S':
        x_t =  (x_t > x0).astype(int)           # so when we compare with the probability, its at 0 in lower and 1 in upper (as the true proabbility of being at the upper state while at the low state is just 0 or vice versa , instead of the lower bound being x_0 its 0.)

    if fast:                    # just because the contour plot does take a while ...
        if model == 'R':
            kValue = posterior.shape[2]
            x_states = np.arange(kValue) / (kValue - 1)
            exceptionMat = posterior @ x_states

        if model == 'S':
            probJumpMat = posterior[:, :, -1]
            jumptimeMat = np.where(probJumpMat>0.5,1,0)
            jumptimeMat = np.diff(jumptimeMat,axis=1,prepend=0)  # a matrix at 1 at the jump time and zeros elsewhere
            hasJump = np.any(jumptimeMat == 1, axis=1)
            estimatedJumpTimes = np.argmax(jumptimeMat == 1,axis = 1)
            estimatedJumpTimes[~hasJump] = T


    else:
        for i in range(Ntrials):
            eRow = []
            for j in range(T):
                if model == 'R':
                    ei = x_tExpectationPosterior(posterior,j,i,model = model)
                else:
                    ei, pji = x_tExpectationPosterior(posterior,j,i,x0=x0,model = model)
                    probJumpMat.append(pji)
                
                eRow.append(ei)
            exceptionMat.append(eRow)

        exceptionMat = np.array(exceptionMat)       # so have the full expectation, now need to plot it.

        if model == 'S':
            probJumpMat = np.array(probJumpMat)
            probJumpMat = probJumpMat.reshape(Ntrials, T)  # i can do this instead of double append but computationally it should still take the same amount of time ygm

            jumptimeMat = np.where(probJumpMat>0.5,1,0)
            jumptimeMat = np.diff(jumptimeMat,axis=1,prepend=0)  # a matrix at 1 at the jump time and zeros elsewhere
            hasJump = np.any(jumptimeMat == 1, axis=1)
            estimatedJumpTimes = np.argmax(jumptimeMat == 1,axis = 1)
            estimatedJumpTimes[~hasJump] = T

    if plot:
        fig,ax = plt.subplots()
        time = np.arange(T) / T
        for i in range(Ntrials):
            colour = f"C{i}"

            if model =='S':
                ax.plot(time,probJumpMat[i],  color=colour ,label = 'Posterior Probability of upper state' if i == 0 else None)
                if jumps is not None and jumps[i] < T :
                    ax.scatter(jumps[i]/T,0.5,s=30,marker="o", color = colour , label = 'True Jump time' if i == 0 else None)
                if estimatedJumpTimes[i] < T:
                    ax.scatter(estimatedJumpTimes[i]/T,probJumpMat[i][estimatedJumpTimes[i]],s=40,marker="x", color = colour , label = 'Estimated jump times' if i == 0 else None)
        
            else:
                ax.plot(time,exceptionMat[i],  color=colour ,label = 'Expectations' if i == 0 else None)    
            ax.plot(time,x_t[i], color=colour , linestyle='--',label = 'Ground-Truth' if i == 0 else None )
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trajectory value")
        ax.legend()
        ax.set_title(title)
        plt.show()


        #return exceptionMat, probJumpMat, estimatedJumpTimes        # ??? Um we only care about the probability of jumping not the expectation? Awesome
    if model == 'S':
        return probJumpMat,estimatedJumpTimes
    return exceptionMat

#Obvs you do sm sort of distance between them but what clacultation of distance is the best? (Im just going to do  RMSE as i did that before .... -- Maybe its possible to look into it.)
# the only possible cost functions are ones, which are just distant. I think the only two worth considering are minimising error or root square error. As that is first and second norms (1-norm and 2-norm measures of distance) . Anything else it too strongly favoured, as in too much penalty as you go higher like cubic and quartic and so on

# mean absolute error
# They want 'average error' ... is that MAE ? or is that literally mean error? it says average error so you are getting mean error - No its MAE as that is dumb as hell im sorry. Because abs is about how far the line is from each other, mean error is just that its above and below the line the same amount instead of being in line with it...
def maeDistance(x1 ,x2):    # its MAE as we want the inference to be as accurate to reality as possible!
    return np.mean(np.abs(x1-x2))       # this is the sum of absolutes right?

def inferenceContourPlotterRamp(x0,Rh,Ntrials,T,kValue,sigmaValues=None,betaValues=None,mValues=None,rValues=None, fast = True , model = None,filter = False):
    if model is None:
        raise ValueError("Declare what model is being used")
    model = model.upper()

    #literal logic puzzle to solve
    if (sigmaValues is None and betaValues is None) == (rValues is None and mValues is None):
        raise ValueError("Search values must be given")
    
    if (sigmaValues is None) ==  (rValues is None):
        raise ValueError("Search values must be given")
    if rValues is not None:
        rValues = np.array(rValues,dtype=int)      #have to be an integer!
        sigmaValues = rValues       # so only using one in the code but its compatible for both models

    if (betaValues is None)  == (mValues is None):       # want this to be nor because if they are both not none thats a problem
        raise ValueError("Search values must be given")
    
    if model == 'S':
        sigmaValues = rValues   # so only using one in the code but its compatible for both models
        betaValues = mValues      


    
    distances = np.zeros((len(sigmaValues), len(betaValues)))

    bestDistance = np.inf
    bestParamVar = None
    bestParmaMean = None

    for i, ParaV in enumerate(sigmaValues):
        for j, ParaM in enumerate(betaValues):
            if model == 'R':
                x_t = simSeqRampFast(Ntrials,ParaM,ParaV,x0,T,kValue, returnS = False)
                expectedx_t = gettingExpectationMat(x_t,T,Rh,x0,kValue=kValue,beta=ParaM,sigma=ParaV,model=model,plot=False, fast = fast , filter = filter) # for the step model it outputs the probability of being in the upper state not expected mean
            if model == 'S':
                x_t = simR1StateHomogeneousStep(ParaM,ParaV,T,Ntrials,x0,fast=fast)    
                expectedx_t,_ = gettingExpectationMat(x_t,T,Rh,x0,m=ParaM, r=ParaV,model='S',plot = False ,fast=fast, filter=filter)
                x_t =  (x_t > x0).astype(int)          # same reasoning as done in the gettegExpectationMat function


            distance = maeDistance(x_t, expectedx_t)
            distances[i, j] = distance

            if distance < bestDistance:
                bestDistance = distance
                bestParamVar = ParaV
                bestParmaMean = ParaM

    # Plot contour map
    B, S = np.meshgrid(betaValues, sigmaValues)

    plt.figure(figsize=(8, 5))

    contour = plt.contourf(B, S, distances, levels=20)
    cbar = plt.colorbar(contour, label="Distance between posterior expectation and ground truth / MAE")
    for tick in cbar.ax.get_yticklabels():
        tick.set_rotation(45)

    plt.scatter(bestParmaMean, bestParamVar, marker="x", s=100, color="black", label="Best match")
    if model == 'R':
        plt.xlabel("Beta")
        plt.ylabel("Sigma")
        plt.title(
            f"Posterior Expectation distance from ground truth latent variable x_t\n"
            f"Best: beta={bestParmaMean:.3g}, sigma={bestParamVar:.3g}, distance={bestDistance:.3g}\n"
            f"Current: x0 ={x0:.3g} , Rh = {Rh:.3g}"
        )
    else:
        plt.xlabel("M")
        plt.ylabel("R")
        plt.title(
            f"Posterior probability of upper state distance from ground truth state\n"
            f"Best: m={bestParmaMean:.3g}, r={bestParamVar:.3g}, distance={bestDistance:.3g}\n"
            f"Current: x0 ={x0:.3g} , Rh = {Rh:.3g}"
        )

    plt.legend()
    plt.show()

#def theTrajPlotForTask23(expectaionMat,):