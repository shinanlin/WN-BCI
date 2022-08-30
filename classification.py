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

# parameters
srate = 250
expName = 'confirm'
chnNames = ['PZ','PO5', 'POZ', 'PO3','PO4', 'PO6','PO7','PO8', 'O1', 'OZ','O2']
seedNUM = 1
targetNUM = 40
poolNUM = 100
saveFILE = 'classification.csv'
# winLENs = [0.3]
winLENs = np.arange(0.2,1,step=.1)


dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

for sub in tqdm(wholeset):
    
    frames = []

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    add = 'results/%s/%s' % (expName,subName)        
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        for tag in ['wn', 'ssvep','high']:

            y = sub[tag]['y']
            X = sub[tag]['X'][-len(y):,chnINX]
            # S = np.array(sub[tag]['STI'])
            
            for seed in tqdm(np.arange(seedNUM)):

                random.seed(seed)
                picked = random.sample(np.unique(y).tolist(), targetNUM)

                X_picked = np.concatenate([X[y == i] for i in picked])
                y_picked = np.concatenate([y[y == i] for i in picked])
                # S_picked = np.concatenate([S[S == i] for i in picked])


                stratSplit = StratifiedShuffleSplit(n_splits=6, test_size=1/6, random_state=42)

                for cv, (train_index, test_index) in enumerate(stratSplit.split(X_picked, y_picked)):
                    X_train, X_test = X_picked[train_index], X_picked[test_index]
                    # S_train, S_test = S[train_index], S[test_index]
                    y_train, y_test = y_picked[train_index], y_picked[test_index]

                    # predict
                    for winLEN in winLENs:

                        model = TRCA(winLEN=winLEN,lag=35,srate=srate,montage=targetNUM)
                        model.fit(X_train,y_train)
                        score = model.score(X_test,y_test)
                        
                        
                        frame = pd.DataFrame({
                            'score': [score],
                            'winLEN':[winLEN],
                            'tag':[tag],
                            'cv':[cv],
                            'seed':[seed],
                            'subject':[subName],
                        })

                        frames.append(frame)

                    df = pd.concat(frames,axis=0,ignore_index=True)
                    add = 'results/%s/%s' % (expName,subName)
                    if not os.path.exists(add):
                        os.makedirs(add)
                    df.to_csv(add+os.sep+saveFILE)






