import sys
sys.path.append('.')

import os
from scipy import rand
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import stats
from spatialFilters import *
import argparse
import random

# %% argument parse


parser = argparse.ArgumentParser()
parser.add_argument('--expName', type=str, default='sweep')
parser.add_argument('--seedNUM', type=int, default=100)
parser.add_argument('--targetNUM', type=int, default=40)
parser.add_argument('--saveFILE', type=str, default='methodOpt.csv')


srate = 250
expName = 'sweep'
seedNUM = 1
targetNUM = 40
poolNUM = 100
saveFILE = 'methodOpt.csv'
winLENs = [0.3]

# %% parameters to be optmized

# latency 
lags = [0.14]

# n_bands

n_bands = [1,2,3,4,5]

# filter banks
fbs = [[(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
        [(14, 90), (10, 100)],
        [(22, 90), (16, 100)],
        [(30, 90), (24, 100)],
        [(38, 90), (32, 100)],]

# filterband coef
# As  = np.arange(0.25,2,step=0.25)
# Bs = np.arange(0.25,1.25,step=0.25)

As  = [1]
Bs = [0.25]

# n_components
n_components = [1,2,3]

# train block
train_size = [2,3,4,5,6]

# montage 
chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2']

# %%

dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

# %%
for sub in tqdm(wholeset):

    frames = []

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    add = 'results/%s/%s' % (expName, subName)
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        for lag in lags:

            lag = round(lag*srate)

            y = sub['wn']['y']
            X = sub['wn']['X'][-len(y):, chnINX]

            for seed in tqdm(np.arange(seedNUM)):

                random.seed(seed)

                picked = random.sample(np.unique(y).tolist(), targetNUM)

                X_picked = np.concatenate([X[y == i] for i in picked])
                y_picked = np.concatenate([y[y == i] for i in picked])

                stratSplit = StratifiedShuffleSplit(
                    n_splits=6, test_size=1/6, random_state=42)

                for cv, (train_index, test_index) in enumerate(stratSplit.split(X_picked, y_picked)):

                    X_train, X_test = X_picked[train_index], X_picked[test_index]
                    y_train, y_test = y_picked[train_index], y_picked[test_index]

                    # predict
                    for winLEN in winLENs:

                        for a in As:
                            for b in Bs:

                                model = gridSearch(fbPara=fbs,winLEN=winLEN, lag=lag,
                                        srate=srate, montage=targetNUM,a=a,b=b)

                                model.fit(X_train, y_train)
                                score = model.score(X_test, y_test)

                                frame = pd.DataFrame({
                                    'score': [score],
                                    'winLEN': [winLEN],
                                    'cv': [cv],
                                    'seed': [seed],
                                    'subject': [subName],
                                    'a':a,
                                    'b':b,
                                })

                                frames.append(frame)

                    df = pd.concat(frames, axis=0, ignore_index=True)
                    add = 'results/%s/%s' % (expName, subName)
                    if not os.path.exists(add):
                        os.makedirs(add)
                    df.to_csv(add+os.sep+saveFILE)
