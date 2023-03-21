import sys
sys.path.append('.')
from scipy.io import savemat, loadmat
import copy
from compare.spatialFilters import *
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedShuffleSplit
import os
import pandas as pd
import random
from tqdm import tqdm
import pickle
import numpy as np
# %%
# this script is for computing the performance of optimized codeset

# %%
srate = 250
expName = 'sweep'
chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
codespace = 160
targetNUM = 40

n_band = 5
seedNUM = int(1e3)
winLENs = [0.3]
lag = 0.14

# group optimized of personalized?
optMode = 'group'
# based on simulate or real response
optMethods = ['simulate','real']

saveFILE = 'groupOpt.csv'
# %%
dir = 'datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

# %%
_classes = np.unique(wholeset[0]['wn']['y']).tolist()
# plant seeds:一定要记住randoms直接返回的是抽取的标签，不是索引
pickedSet = []
for seed in range(seedNUM):
    random.seed(seed)
    pickedSet.append({
        'seed':seed,
        'tag': 'random',
        'code':random.sample(_classes, targetNUM)
    })
# %% load optimized sequence

# optSet = loadmat('seqOpt/mat/optimized_group.mat')
optSet = loadmat('seqOpt/mat/opt_group_large.mat')
# pick optimized seq

# %%
for sub in tqdm(wholeset):

    frames = []
    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]
    X = sub['wn']['X'][:, chnINX]
    y = sub['wn']['y']

    X = np.stack([X[y == i] for i in np.unique(y)])
    y = np.stack([y[y == i] for i in np.unique(y)])

    # classification
    X = np.transpose(X,axes=(1,0,-2,-1))
    y = np.transpose(y,axes=(-1,0))

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    seq = optSet[optMode] if optMode == 'group' else optSet[subName]

    for optMethod in optMethods:
        # 一定要注意这里拿到的是索引，不是标签
        # pickINX = (np.array([s[optMethod][0]['code'][0, 0].T[0].tolist() for s in seq])).tolist()[0]
        pickIndices = [[s[optMethod][i]['code'][0, 0].T[0].tolist() for i in range(len(s))] for s in seq][0]
        for s,pickINX in enumerate(pickIndices):
            pickedSet.append({
                'seed':s,
                'tag': optMethod,
                'code': [_classes[i] for i in pickINX]
            })

    for cv, (train_index, test_index) in enumerate(loo.split(X, y)):
        X_train, X_test = np.concatenate(
                X[train_index]), np.concatenate(X[test_index])
        y_train, y_test = np.concatenate(
            y[train_index]), np.concatenate(y[test_index])


        for winLEN in winLENs:

            model = TDCA(winLEN=winLEN, lag=lag, srate=srate,
                         montage=codespace, n_band=n_band)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            
            # 这里可能还需要给混淆矩阵重新排序下
            coefMatrix = model.rho

            labels = np.arange(targetNUM)
            # 搞定分布
            S = []
            tags = []

            for codeset in tqdm(pickedSet):
                # 这里的picked是索引
                picked = [_classes.index(i) for i in  codeset['code']]
                picked_coef = coefMatrix[picked, :][:, picked]
                accuracy = accuracy_score(labels, np.argmax(picked_coef, axis=0))

                f = pd.DataFrame({
                    'accuracy': [accuracy],
                    'winLEN':[winLEN],
                    'method': [codeset['tag']],
                    'cv':[cv],
                    'seed':[codeset['seed']],
                    'subject':[subName],
                })

                frames.append(f)

            df = pd.concat(frames, axis=0, ignore_index=True)
            add = 'results/%s/%s' % (expName, subName)
            if not os.path.exists(add):
                os.makedirs(add)
            df.to_csv(add+os.sep+saveFILE)
