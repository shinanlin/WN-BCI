
import sys
sys.path.append('.')
sys.path.append('./compare')


from scipy import rand
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import compare.utils as utils
from compare.spatialFilters import *
import random


# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'sweep'
seedNUM = int(1e3)
n_band = 5
targetNUM = 40
codespace = 160
saveFILE = 'LN.csv'
tag = 'wn'
modes = ['linear','nonlinear','full']
winLENs = [0.86]
lag = 0.14

# %%
dir = './data/datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)
# %%
_classes = np.unique(wholeset[0]['wn']['y']).tolist()
# plant seeds:一定要记住randoms直接返回的是抽取的标签，不是索引
pickedSet = []

for seed in range(seedNUM):
    random.seed(seed)
    pickedSet.append({
        'seed': seed,
        'tag': 'random',
        'code': random.sample(_classes, targetNUM)
    })

#%% refresh

add = 'results'+os.sep+expName

for fnames in os.listdir(add):
    f = add+os.sep+fnames+os.sep+saveFILE
    if os.path.exists(f):
        os.remove(f)

# %%
for sub in tqdm(wholeset):

    frames = []
    cMs = []

    subName = sub['name']

    add = 'results/%s/%s' % (expName, subName)
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        for mode in modes:

            X = sub[tag]['X']
            y = sub[tag]['y'][:]
            S = sub[tag]['STI']

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

                    model = LN(S=(S,_classes),winLEN=winLEN, lag=lag, srate=srate,
                                 montage=codespace, n_band=n_band,mode=mode)
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
                            'accuracy': [accuracy],
                            'winLEN': [winLEN],
                            'ITR': [utils.ITR(targetNUM, accuracy, winLEN)],
                            'method': [tag],
                            'cv': [cv],
                            'seed': [codeset['seed']],
                            'subject': [subName],
                            'score': [score],
                            'mode':[mode]
                        })

                        frames.append(f)

                    cM = pd.DataFrame(index=model._classes,
                                 columns=model._classes, data=model.rho)
                    cM.reset_index(level=0, inplace=True)
                    cM = cM.melt(id_vars='index', value_name='rho',
                               var_name='i')
                    cM = cM.rename(columns={'index': 'j'})

                    cM['subject'] = subName
                    cM['method'] = tag
                    cM['mode'] = mode
                    cM['winLEN'] = winLEN
                    cM['cv'] = cv

                    cMs.append(cM)

                df1 = pd.concat(frames, axis=0, ignore_index=True)
                df2 = pd.concat(cMs, axis=0, ignore_index=True)

                add = 'results/%s/%s' % (expName, subName)
                if not os.path.exists(add):
                    os.makedirs(add)
                df1.to_csv(add+os.sep+saveFILE)
                df2.to_csv(add+os.sep+'LNcM.csv')
