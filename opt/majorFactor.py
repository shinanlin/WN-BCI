import sys
sys.path.append('.')
from scipy.io import savemat, loadmat
import copy
from spatialFilters import TRCA
from sklearn.model_selection import StratifiedShuffleSplit
import os
import pandas as pd
from utils import codeDistance
import random
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split



# %%
srate = 240
expName = 'sweep'
chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4',
            'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2']
seedNUM = 1
targetNUM = 40
poolNUM = int(1e2)
saveFILE = 'factors.csv'
winLEN = 0.3

# %%

expName = 'sweep'
dir = 'datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

# %%
_classes = np.unique(wholeset[0]['wn']['y']).tolist()
# plant seed

randoms = []
for seed in range(poolNUM):
    random.seed(seed)
    randoms.append(random.sample(_classes, targetNUM))


# %%

for sub in tqdm(wholeset):

    frames = []
    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]
    X = sub['wn']['X'][:, chnINX]
    y = sub['wn']['y']

    for seedINX, pick in enumerate(randoms):

        X_picked = np.concatenate([X[y == i] for i in pick])
        y_picked = np.concatenate([y[y == i] for i in pick])

        stratSplit = StratifiedShuffleSplit(
            n_splits=6, test_size=1/6, random_state=42)

        for cv, (train_index, test_index) in enumerate(stratSplit.split(X_picked, y_picked)):

            X_train, X_test = X_picked[train_index], X_picked[test_index]
            y_train, y_test = y_picked[train_index], y_picked[test_index]

            # predict
            model = TRCA(winLEN=winLEN, lag=33, srate=srate,
                         montage=targetNUM, n_band=1)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            _classes,order = np.unique(y_test,return_index=True)
            coef = model.rho[order]
            p = model.confidence[order]
            predicted = model.predicted[order]

            f = pd.DataFrame(data=coef, index=np.arange(
                targetNUM), columns=np.arange(targetNUM))
            f.reset_index(level=0, inplace=True)
            f = f.melt(id_vars='index', value_name='coef',var_name='class-2')
            f = f.rename(columns={'index': 'class-1'})

            f['label-1'] = np.tile(_classes,targetNUM)
            f['label-2'] = np.repeat(_classes,targetNUM)
            f['predicted'] = np.tile(predicted, targetNUM)
            f['correct'] = np.tile(predicted == _classes,targetNUM)

            f['p'] = np.tile(p,targetNUM)
            f['seed'] = seedINX
            f['cv'] = cv
            f['accuracy'] = score
            f['subject'] = subName

            frames.append(f)

        df = pd.concat(frames, axis=0, ignore_index=True)
        add = 'results/%s/%s' % (expName, subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)

