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

# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
n_band = 5
targetNUM = 40
saveFILE = 'montage.csv'
winLENs = [0.1, 0.2, 0.3, 0.4, 0.5]
lag = 0.14

# montages
montages = [

    ['O1', 'OZ', 'O2'],
    ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3',
        'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2'],
    ['CPZ','CP1','CP2','CP3','CP4','CP5','CP6','TP7','TP8','PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3',
        'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2'],
    ['ALL']

]

montageNames = ['Central occipital',
                'classical', 'Occipital', 'Parieto-occipital','All']
# %%

dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

# %%
# add = 'results'+os.sep+expName

# for fnames in os.listdir(add):
#     f = add+os.sep+fnames+os.sep+saveFILE
#     if os.path.exists(f):
#         os.remove(f)

# %%
for sub in tqdm(wholeset):

    frames = []

    subName = sub['name']

    add = 'results/%s/%s' % (expName, subName)
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        for tag in ['WN','SSVEP']:

            X = sub[tag]['X']
            y = sub[tag]['y']

            X = np.stack([X[y == i] for i in np.unique(y)])
            y = np.stack([y[y == i] for i in np.unique(y)])

            # classification
            X = np.transpose(X, axes=(1, 0, -2, -1))
            y = np.transpose(y, axes=(-1, 0))
            
            

            for (monatge,monatgeName) in zip(montages,montageNames):

                loo = LeaveOneOut()
                loo.get_n_splits(X)

                if monatgeName=='All':
                    monatge = sub['channel']

                chnINX = [sub['channel'].index(i) for i in monatge]
                X_ = X[:,:,chnINX]

                for cv, (train_index, test_index) in enumerate(loo.split(X_, y)):

                    X_train, X_test = np.concatenate(
                        X_[train_index]), np.concatenate(X_[test_index])
                    y_train, y_test = np.concatenate(
                        y[train_index]), np.concatenate(y[test_index])

                    # predict
                    for winLEN in winLENs:

                        model = TDCA(winLEN=winLEN, lag=lag, srate=srate,
                                    montage=targetNUM, n_band=n_band)
                        model.fit(X_train, y_train)
                        accuracy = model.score(X_test, y_test)

                        f = pd.DataFrame({
                            'accuracy': [accuracy],
                            'winLEN': [winLEN],
                            'ITR': [utils.ITR(targetNUM, accuracy, winLEN)],
                            'method': [tag],
                            'cv': [cv],
                            'subject': [subName],
                            'monatage':[monatgeName]
                        })

                        frames.append(f)

                    df = pd.concat(frames, axis=0, ignore_index=True)
                    add = 'results/%s/%s' % (expName, subName)
                    if not os.path.exists(add):
                        os.makedirs(add)
                    df.to_csv(add+os.sep+saveFILE)