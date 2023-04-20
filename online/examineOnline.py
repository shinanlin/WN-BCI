import sys
sys.path.append('.')

import numpy as np
from tqdm import tqdm
import pickle
import os
import pandas as pd
from compare.spatialFilters import TDCA
from compare.utils import ITR


# %% define parameters

winLENs = [0.1,0.2,0.3]
expName = 'online'

# input 16 character 
charN = 16 
targetN = 40
# %% load dataset

dir = 'data/datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    DATA = pickle.load(fp)

# %% load dataset

for subDATA in tqdm(DATA):

    CUED,FREE = [],[]
    subName = subDATA['name']

    trainX, trainy = subDATA['train']['X'],subDATA['train']['y']
    testX, testy = subDATA['cued']['X'], subDATA['cued']['y']
    freeX, freey = subDATA['free']['X'],subDATA['free']['y']

    # reshape 
    trainX,testX,freeX = np.concatenate(trainX),np.concatenate(testX),np.concatenate(freeX)
    
    trainy,testy,freey = np.concatenate(trainy),np.concatenate(testy),np.concatenate(freey)

    trialN, chnN, T = freeX.shape

    # reshape free spelling data to block format
    freeX = freeX.reshape((trialN//charN, charN, chnN, T))
    freey = freey.reshape((trialN//charN, charN))

    testX = np.stack([testX[testy==i] for i in np.unique(testy)])
    testy = np.stack([testy[testy==i] for i in np.unique(testy)])
    testX = np.transpose(testX,axes=(1,0,-2,-1))
    testy = testy.T

    for winLEN in winLENs:

        model = TDCA(winLEN=winLEN)
        model.fit(trainX, trainy)

        for bINX, (thisBlockX, thisBlocky) in enumerate(zip(testX, testy)):
            acc = model.score(thisBlockX, thisBlocky)
            itr = ITR(targetN,acc,winLEN)

            f = pd.DataFrame({
                'accuracy':[acc],
                'ITR':[itr],
                'winLEN':[winLEN],
                'block':[bINX],
                'subject':[subName]
            })

            CUED.append(f)

        for bINX, (thisBlockX, thisBlocky) in enumerate(zip(freeX, freey)):

            acc = model.score(thisBlockX, thisBlocky)
            # freespelling has extra 0.5 interval
            itr = ITR(targetN, acc, winLEN+0.5)

            f = pd.DataFrame({
                'accuracy': [acc],
                'ITR': [itr],
                'winLEN': [winLEN],
                'block': [bINX],
                'subject': [subName]
            })

            FREE.append(f)

    CUED = pd.concat(CUED,axis=0,ignore_index=True)
    FREE = pd.concat(FREE,axis=0,ignore_index=True)
    add = 'results/%s/%s' % (expName, subName)

    if not os.path.exists(add):
        os.makedirs(add)
    CUED.to_csv(add+os.sep+'cuedPerformance.csv')
    FREE.to_csv(add+os.sep+'freePerformance.csv')
