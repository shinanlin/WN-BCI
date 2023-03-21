import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from scipy import rand
import seaborn as sns
from mne.decoding import ReceptiveField
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
from scipy.stats import stats
import compare.utils as utils
from compare.spatialFilters import *
import random

# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
chnNames = ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3',
            'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2']
seedNUM = int(1)
n_band=5
targetNUM = 40
codespace = 40
saveFILE = 'best.csv'
winLENs = [0.1,0.2]
lag = 0.14
best = ['yangyuxing','fanzixiao','libowen', 'zhouyuqing','yaofeifan']


# %%

dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)
# %%  
_classes = np.unique(wholeset[0]['WN']['y']).tolist()
# plant seeds:一定要记住randoms直接返回的是抽取的标签，不是索引
pickedSet = []

for seed in range(seedNUM):
    random.seed(seed)
    pickedSet.append({
        'seed': seed,
        'tag': 'random',
        'code': random.sample(_classes, targetNUM)
    })

# %%  
for sub in tqdm(wholeset):
    
    frames = []

    
    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    add = 'results/%s/%s' % (expName,subName)        
    if subName not in best:
        pass
    else:
        for tag in ['WN','SSVEP']:

            X = sub[tag]['X'][:, chnINX]
            y = sub[tag]['y'][:]

            X = np.stack([X[y == i] for i in np.unique(y)])
            y = np.stack([y[y == i] for i in np.unique(y)])

            # classification
            X = np.transpose(X, axes=(1, 0, -2, -1))
            y = np.transpose(y, axes=(-1, 0))

            loo = LeaveOneOut()
            loo.get_n_splits(X)

            for cv, (train_index, test_index) in enumerate(loo.split(X, y)):

                X_train, X_test = np.concatenate(
                    X[train_index]), np.concatenate(X[test_index])
                y_train, y_test = np.concatenate(
                    y[train_index]), np.concatenate(y[test_index])

                # predict
                for winLEN in winLENs:

                    model = TDCA(winLEN=winLEN,lag=lag,srate=srate,montage=codespace,n_band=n_band)
                    model.fit(X_train,y_train)
                    score = model.score(X_test,y_test)
                    
                    coefMatrix = model.rho
                    labels = np.arange(targetNUM)

                    for codeset in tqdm(pickedSet, desc="Sub:%s,T:%ss,cv:%sth"%(subName,winLEN,cv)):
                        picked = [_classes.index(i) for i in codeset['code']]
                        picked_coef = coefMatrix[picked, :][:, picked]
                        accuracy = accuracy_score(
                            labels, np.argmax(picked_coef, axis=0))

                        f = pd.DataFrame({
                            'accuracy': [accuracy],
                            'winLEN': [winLEN],
                            'ITR':[utils.ITR(targetNUM,accuracy,winLEN-0.5)],
                            'method': [tag],
                            'cv': [cv],
                            'seed': [codeset['seed']],
                            'subject': [subName],
                            'score':[score]
                        })

                        frames.append(f)

                df = pd.concat(frames,axis=0,ignore_index=True)
                add = 'results/%s/%s' % (expName,subName)
                if not os.path.exists(add):
                    os.makedirs(add)
                df.to_csv(add+os.sep+saveFILE)
