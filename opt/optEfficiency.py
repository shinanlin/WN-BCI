import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from scipy.io import savemat,loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import random
from utils import codeDistance
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from spatialFilters import TRCA
import copy

# %%
srate = 240
expName = 'sweep'
chnNames = ['PZ','PO5', 'POZ', 'PO3','PO4', 'PO6','PO7','PO8', 'O1', 'OZ','O2']
seedNUM = 1
targetNUM = 40
poolNUM = int(1e3)
saveFILE = 'reheat.csv'

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
    randoms.append(random.sample(_classes,targetNUM))

optSequence = loadmat('opt/optimized.mat')

# %%

for sub in tqdm(wholeset):

    frames = []
    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]
    X = sub['wn']['X'][:, chnINX]
    y = sub['wn']['y']
    

    winLENs,reheats,codes,tags,types = [],[],[],[],[]

    # load optimized sequence and parameters
    for name in ['group',subName]:

        seq = np.squeeze(optSequence[name])

        tag = 'group' if name=='group' else 'customized'
        
        for key in ['real','simulate']:

            # type
            types = types+[key for _ in range(len(seq))]
            # tag
            tags = tags + [tag for _ in range(len(seq))]
            # code :optimized sequence code
            codes = codes + (np.array([s[key]['code'][0,0].T[0].tolist() for s in seq])+1).tolist()
            # winLEN corresponds to each code
            winLENs = winLENs + [s['winLEN'][0, 0] for s in seq]
            # reheat parameter corresponds to each code
            reheats = reheats + [s['reheat'][0, 0] for s in seq]


    for seedINX,(pick,winLEN) in enumerate(zip(codes,winLENs)):

        X_picked = np.concatenate([X[y == i] for i in pick])
        y_picked = np.concatenate([y[y == i] for i in pick])

        stratSplit = StratifiedShuffleSplit(n_splits=6, test_size=1/6, random_state=42)

        for cv, (train_index, test_index) in enumerate(stratSplit.split(X_picked, y_picked)):

            X_train, X_test = X_picked[train_index], X_picked[test_index]
            y_train, y_test = y_picked[train_index], y_picked[test_index]

            # predict
            model = TRCA(winLEN=winLEN,lag=33,srate=srate,montage=targetNUM,n_band=1)
            model.fit(X_train,y_train)
            score = model.score(X_test,y_test)
                    
            frame = pd.DataFrame({
                'score': [score],
                'winLEN': [winLEN],
                'cv': [cv],
                'subject': [subName],
                'opt':[types[seedINX]],
                'reheats': [reheats[seedINX]],
                'tag':[tags[seedINX]]
            })

            frames.append(frame)

        df = pd.concat(frames, axis=0, ignore_index=True)
        add = 'results/%s/%s' % (expName, subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)

