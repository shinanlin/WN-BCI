import sys
sys.path.append('.')

import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
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
n_band = 5
targetNUM = 40
codespace = 40
saveFILE = 'latency.csv'
latencies = np.arange(0.02,0.4,0.04)
winLENs = [0.1]
# %%

dir = './data/datasets/%s.pickle' % expName
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
add = 'results'+os.sep+expName

for fnames in os.listdir(add):
    f = add+os.sep+fnames+os.sep+saveFILE
    if os.path.exists(f):
        os.remove(f)

# %%
for sub in tqdm(wholeset):

    frames = []

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    add = 'results/%s/%s' % (expName, subName)
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        for tag in ['WN','SSVEP']:

            X = sub[tag]['X'][:, chnINX]
            y = sub[tag]['y']

            X = np.stack([X[y == i] for i in np.unique(y)])
            y = np.stack([y[y == i] for i in np.unique(y)])

            # classification
            X = np.transpose(X, axes=(1, 0, -2, -1))
            y = np.transpose(y, axes=(-1, 0))

            loo = LeaveOneOut()
            loo.get_n_splits(X)

            for lag in tqdm(latencies):

                for cv, (train_index, test_index) in enumerate(loo.split(X, y)):

                    X_train, X_test = np.concatenate(
                        X[train_index]), np.concatenate(X[test_index])
                    y_train, y_test = np.concatenate(
                        y[train_index]), np.concatenate(y[test_index])

                    # predict
                    for winLEN in winLENs:

                        model = TDCA(winLEN=winLEN, lag=lag, srate=srate,
                                    montage=codespace, n_band=n_band)
                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)

                        coefMatrix = model.rho
                        labels = np.arange(targetNUM)

                        for codeset in tqdm(pickedSet, desc="Sub:%s,T:%ss,cv:%sth" % (subName, winLEN, cv)):
                            picked = [_classes.index(i) for i in codeset['code']]
                            picked_coef = coefMatrix[picked, :][:, picked]
                            accuracy = accuracy_score(
                                labels, np.argmax(picked_coef, axis=0))

                            f = pd.DataFrame({
                                'accuracy': [score],
                                'winLEN': [winLEN],
                                'ITR': [utils.ITR(targetNUM, score, winLEN)],
                                'method': [codeset['tag']],
                                'cv': [cv],
                                'latency':[lag],
                                'seed': [codeset['seed']],
                                'tag':[tag],
                                'subject': [subName],
                            })

                            frames.append(f)

                    df = pd.concat(frames, axis=0, ignore_index=True)
                    add = 'results/%s/%s' % (expName, subName)
                    if not os.path.exists(add):
                        os.makedirs(add)
                    df.to_csv(add+os.sep+saveFILE)
