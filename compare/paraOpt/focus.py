import sys
sys.path.append('.')

import matplotlib.pyplot as plt
from scipy import rand
import seaborn as sns
from mne.decoding import ReceptiveField
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from scipy.stats import stats
import compare.utils as utils
from compare.spatialFilters import *
import random

# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
chnNames = ['PZ','PO5', 'POZ', 'PO3','PO4', 'PO6', 'O1', 'OZ','O2']
seedNUM = int(1)
n_band=5
targetNUM = 40
saveFILE = 'focus.csv'
winLENs = [0.2]
lag = 0.14
colN,rowN = 8,5

# %%

dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)


# %% delete csv in patch
add = 'results'+os.sep+expName

# for fnames in os.listdir(add):
#     f = add+os.sep+fnames+os.sep+saveFILE
#     if os.path.exists(f):
#         os.remove(f)

# %%  
for sub in tqdm(wholeset):
    
    frames = []

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    add = 'results/%s/%s' % (expName,subName)        
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        for tag in ['WN','SSVEP']:

            X = sub[tag]['X'][:, chnINX]
            y = sub[tag]['y'][:]

            _class = np.unique(y)
            X = np.stack([X[y == i] for i in _class])
            y = np.stack([y[y == i] for i in _class])

            # classification
            X = np.transpose(X, axes=(1, 0, -2, -1))
            y = np.transpose(y, axes=(-1, 0))

            loo = LeaveOneOut()
            loo.get_n_splits(X)

            layout = np.unique(y).reshape(colN,rowN).T

            for cv, (train_index, test_index) in enumerate(loo.split(X, y)):

                X_train, X_test = np.concatenate(
                    X[train_index]), np.concatenate(X[test_index])
                y_train, y_test = np.concatenate(
                    y[train_index]), np.concatenate(y[test_index])

                # predict
                for winLEN in winLENs:

                    model = TDCA(winLEN=winLEN,lag=lag,srate=srate,montage=targetNUM,n_band=n_band)
                    model.fit(X_train,y_train)

                    accuracy = model.score(X_test,y_test)

                    rho = model.rho

                    H = utils.rho2Confidence(rho)
                    cordNames = ['center','up','left','right','down','upRight','upLeft','downRight','downLeft']
                    for i in y_test:
                        
                        k_values = np.zeros(np.shape(cordNames))

                        this_epoch = H[model._classes==i][0]
                        center = this_epoch[model._classes == i]
                        k_values[0] = center
                        rowINX,colINX = np.argwhere(layout==i)[0]
                        
                        upCord = (rowINX-1,colINX-1)
                        leftCord = (rowINX,colINX-1)
                        rightCord = (rowINX,colINX+1)
                        downCord = (rowINX+1, colINX)

                        upRightCord = (rowINX-1,colINX+1)
                        upLeftCord = (rowINX-1,colINX-1)
                        downRightCord = (rowINX+1,colINX+1)
                        downLeftCord = (rowINX+1,colINX-1)

                        for (keyINX,key) in enumerate([upCord, leftCord, rightCord, downCord, upRightCord, upLeftCord, downRightCord,downLeftCord]):
                            if np.any(np.array(key)<0):
                                k_values[keyINX+1] = np.nan
                            else:
                                k_values[keyINX+1] = H[key[0], key[1]]
                            
                        f = pd.DataFrame(
                            index=cordNames, columns=[i], data=k_values)
                        f.reset_index(level=0, inplace=True)
                        f = f.melt(id_vars='index',
                                value_name='rho', var_name='class')
                        f = f.rename(columns={'index': 'cord'})

                        f['subject'] = subName
                        f['tag'] = tag
                        f['winLEN'] = winLEN
                        f['cv'] = cv
                        f['accuracy'] = accuracy
                        frames.append(f)

                    df = pd.concat(frames,axis=0,ignore_index=True)
                    add = 'results/%s/%s' % (expName,subName)
                    if not os.path.exists(add):
                        os.makedirs(add)
                    df.to_csv(add+os.sep+saveFILE)
