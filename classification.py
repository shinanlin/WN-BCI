import matplotlib.pyplot as plt
from pyrsistent import s
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
from spatialFilters import TRCA, Matching,fbCCA


srate = 500
expName = 'offline'
chnNames = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ']

dir = './datasets/%s.pickle' % expName

winLENs = np.arange(0.2,1,step=.2)

with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

frames = []
for sub in tqdm(wholeset):

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    for tag in ['wn','ssvep']:

        X = sub[tag]['X'][:,chnINX]
        y = sub[tag]['y']
        # S = np.stack([sub[tag]['STI'][i-1] for i in y])
        

        stratSplit = StratifiedShuffleSplit(n_splits=6, test_size=1/6, random_state=42)

        for cv,(train_index, test_index) in enumerate(stratSplit.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            # S_train, S_test = S[train_index], S[test_index]
            y_train, y_test = y[train_index], y[test_index]


            # predict
            for winLEN in winLENs:

                model = TRCA(winLEN=winLEN,lag=75,srate=srate)
                model.fit(X_train,y_train)
                score = model.score(X_test,y_test)
                
                # model = fbCCA(winLEN=winLEN, srate=250)
                # model = Matching(winLEN=winLEN,lag=35)
                # model.fit(S_train,y_train)
                # score = model.score(X_train,y_train)
                
                
                frame = pd.DataFrame({
                    'score': [score],
                    'winLEN':[winLEN],
                    'tag':[tag],
                    'cv':[cv],
                    'subject':[subName]
                })

                frames.append(frame)

        df = pd.concat(frames,axis=0,ignore_index=True)
        df.to_csv('results/%s/classification.csv' % (expName))






