import matplotlib.pyplot as plt
from pyrsistent import s
from scipy import rand
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
import random

srate = 250
expName = 'sweep'
chnNames = ['PZ','PO5', 'POZ', 'PO3','PO4', 'PO6', 'O1', 'OZ','O2']


dir = './datasets/%s.pickle' % expName

# winLENs = np.arange(0.2,1,step=.1)
winLENs = [0.3]

with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

frames = []
for sub in tqdm(wholeset):

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    for tag in ['wn']:

        X = sub[tag]['X'][:,chnINX]
        y = sub[tag]['y']
        # S = np.stack([sub[tag]['STI'][i-1] for i in y])
        
        for seed in tqdm(np.arange(1e3)):

            random.seed(seed)
            picked = random.sample(range(160),40)
            X_picked = np.concatenate([X[y == i+1] for i in picked])
            y_picked = np.concatenate([y[y == i+1] for i in picked])


            stratSplit = StratifiedShuffleSplit(n_splits=6, test_size=1/6, random_state=42)

            for cv, (train_index, test_index) in enumerate(stratSplit.split(X_picked, y_picked)):
                X_train, X_test = X_picked[train_index], X_picked[test_index]
                # S_train, S_test = S[train_index], S[test_index]
                y_train, y_test = y_picked[train_index], y_picked[test_index]


                # predict
                for winLEN in winLENs:

                    model = TRCA(winLEN=winLEN,lag=35,srate=srate)
                    model.fit(X_train,y_train)
                    score = model.score(X_test,y_test)
                    
                    
                    frame = pd.DataFrame({
                        'score': [score],
                        'winLEN':[winLEN],
                        'tag':[tag],
                        'cv':[cv],
                        'seed':[seed],
                        'subject':[subName]
                    })

                    frames.append(frame)

            df = pd.concat(frames,axis=0,ignore_index=True)
            add = 'results/%s/%s' % (expName,subName)
            if not os.path.exists(add):
                os.makedirs(add)
            df.to_csv(add+os.sep+'classification.csv')






