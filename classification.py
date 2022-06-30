import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import ReceptiveField
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from NRC import NRC, recordModule, RegularizedRF
from scipy.stats import zscore
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import stats
from spatialFilters import TRCA,fbCCA,tTRCA


srate = 240
tmin, tmax = -.2, 0.5
expName = 'exp-2'
chnNames = ['PZ', 'POZ', 'OZ', 'P1', 'P2', 'P3',
            'P4', 'P7', 'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2']

dir = './datasets/%s.pickle' % expName

winLENs = np.arange(0.2,1,step=.1)

with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

frames = []
for sub in tqdm(wholeset):

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    all_y = sub['y']
    labels = np.arange(1,60+1,step=1)

    R = sub['X'][:, chnINX]
    S = sub['stimulus'][:].astype('float64')

    for tag in ['wn','mseq','ssvep']:

        this_labels = labels[sub['tags'] == tag]
        
        # take labels out
        this_y = [i if np.any(i == this_labels) else None for i in all_y]
        this_y = np.array(list(filter(None,this_y)))

        # take data out
        this_X = np.concatenate([R[all_y == i] for i in np.unique(this_y)])
        this_S = np.concatenate([S[all_y == i] for i in np.unique(this_y)])

        stratSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        for cv,(train_index, test_index) in enumerate(stratSplit.split(this_X, this_y)):
            X_train, X_test = this_X[train_index], this_X[test_index]
            y_train, y_test = this_y[train_index], this_y[test_index]
            S_train, S_test = this_S[train_index], this_S[test_index]

            # predict
            for winLEN in winLENs:

                model = TRCA(winLEN=winLEN,lag=10)
                # model = fbCCA(winLEN=winLEN, srate=240)
                model.fit(X_train,y_train)
                score = model.score(X_test,y_test)

                frame = pd.DataFrame({
                    'score': [score],
                    'winLEN':[winLEN],
                    'tag':[tag],
                    'cv':[cv],
                    'subject':[subName]
                })

                frames.append(frame)

    df = pd.concat(frames,axis=0,ignore_index=True)
    df.to_csv('results/%s/%s/classification.csv' % (expName, subName))






