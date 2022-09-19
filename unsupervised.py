from itertools import accumulate
from random import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from modeling import Code2EEG, Match
import os
import pandas as pd
# %%
# define parameters
srate = 240
refreshrate = 60
poolSize = 0
testSize = 40
trainSizes = np.arange(10,100,20)

tmin,tmax = 0,0.8
n_band =3
p=0.96
winLENs = np.arange(0.2, 1, step=.1)

chnNames = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']
saveFILE = 'unsupervised.csv'
# %%
# load data
expName = 'sweep'
dir = 'datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

# %%
for sub in tqdm(wholeset):

    frames = []

    chnINX = [sub['channel'].index(i) for i in chnNames]
    X = sub['wn']['X'][:,chnINX]
    y = sub['wn']['y']
    S = sub['wn']['STI']
    subName = sub['name']

    # %%

    for trainSize in trainSizes: 
        # reshape to class*block*chn*N
        X_ = np.stack(X[y == i] for i in np.unique(y))
        y_ = np.stack(y[y == i] for i in np.unique(y))

        # split conditions
        X_train, X_test, y_train, y_test = train_test_split(X_, y_,test_size=testSize,train_size=trainSize,random_state=253)

        X_train,X_test = np.concatenate(X_train, axis=0), np.concatenate(X_test, axis=0)
        y_train,y_test = np.concatenate(y_train, axis=0), np.concatenate(y_test, axis=0)
        S_train,S_test = np.stack([S[i-1] for i in y_train]),np.stack([S[i-1] for i in y_test])


        # %%
        # train forward model
        code2EEG = Code2EEG(srate=srate,winLEN=1,tmin=tmin,tmax=tmax,S=(S,np.unique(y)),estimator=p,padding=True,n_band=n_band,component=1)
        code2EEG.fit(X_train,y_train)

        # X_: subset of data for classification
    
        R_ = code2EEG.predict(S_test)
        X_ = code2EEG.enhancer.transform(X_test)

        for winLEN in tqdm(winLENs):


            templates = code2EEG.predict(S=S_test)

            # inifinte large of the tenplates but just sample small portion of it for validation
            model = Match(winLEN=winLEN,srate=srate,lag=0.14)
            model.fit(R_,y_test)
            acc = model.score(X_,y_test)

            f = pd.DataFrame({
                'subject': [subName],
                'accuracy':[acc],
                'winLEN': [winLEN],
                'train':[trainSize],
                'targetN':[testSize]
            })

            frames.append(f)

        df = pd.concat(frames, axis=0, ignore_index=True)
        add = 'results/%s/%s' % (expName, subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)
