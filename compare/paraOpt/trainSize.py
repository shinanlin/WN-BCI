import sys
sys.path.append('.')

from scipy import rand
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut,LeavePOut,ShuffleSplit
import compare.utils as utils
from compare.spatialFilters import *

# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
n_band = 5
targetNUM = 40
saveFILE = 'trainSize.csv'
winLENs = [0.1, 0.2, 0.3, 0.4, 0.5]
lag = 0.14
trainSizes = np.arange(2,7)
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

            for train_size in trainSizes:

                loo = ShuffleSplit(n_splits=8, test_size=1, train_size=train_size)

                for cv, (train_index, test_index) in enumerate(loo.split(X, y)):

                    X_train, X_test = np.concatenate(
                        X[train_index]), np.concatenate(X[test_index])
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
                            'trainsize':[train_size]
                        })

                        frames.append(f)

                    df = pd.concat(frames, axis=0, ignore_index=True)
                    add = 'results/%s/%s' % (expName, subName)
                    if not os.path.exists(add):
                        os.makedirs(add)
                    df.to_csv(add+os.sep+saveFILE)