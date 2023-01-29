import sys
sys.path.append('.')


from scipy.signal import resample
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from modeling import Code2EEG, Match
import os
import pandas as pd
import utils
# %%
# define parameters
srate = 250
refreshrate = 60
poolSize = 1e4

testSize = 40  #select n classes of data as test data
tmin,tmax = 0,0.5
n_band = 5
p=0.96
blockNUM=6
winLENs = [1]
winLEN = 1
lag = 0
pools = np.logspace(1, 4, num=20).astype(int)

chnNames = ['PZ','PO5', 'POZ', 'PO3','PO4', 'PO6', 'O1', 'OZ','O2']
saveFILE = 'largeset.csv'

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

    X = X-np.mean(X,axis=-1,keepdims=True)

    add = 'results/%s/%s' % (expName,subName)        
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:

        # %%
        # reshape to class*block*chn*N
        X_ = np.stack(X[y == i] for i in np.unique(y))
        y_ = np.stack(y[y == i] for i in np.unique(y))

        # split conditions
        X_train, X_test, y_train, y_test = train_test_split(X_, y_,test_size=testSize,random_state=253)

        X_train,X_test = np.concatenate(X_train, axis=0), np.concatenate(X_test, axis=0)
        y_train,y_test = np.concatenate(y_train, axis=0), np.concatenate(y_test, axis=0)
        S_train,S_test = np.stack([S[i-1] for i in y_train]),np.stack([S[i-1] for i in y_test])


        # %%
        np.random.seed(253)
        # re-create massive loads of STI patterns
        buildSTI = np.random.uniform(0, 1, (int(poolSize), int(winLEN*refreshrate)))
        # upsample
        buildSTI = resample(buildSTI,num=int(winLEN*srate),axis=-1)

        # %%
        # train forward model
        code2EEG = Code2EEG(srate=srate,winLEN=winLEN,tmin=tmin,tmax=tmax,S=(S,np.unique(y)),estimator=p,padding=True,n_band=n_band,component=1)
        code2EEG.fit(X_train,y_train)

        # X_: subset of data for classification
        X_ = code2EEG.enhancer.transform(X_test)

        _class_subset = np.unique(y_test)

        S_test = np.stack(S_test[y_test == i].mean(axis=0) for i in _class_subset)
        X_ = np.concatenate([X_[y_test==i] for i in _class_subset])

        n = len(np.unique(y_test))
        for size in tqdm(pools):

            adhere = buildSTI[:size]

            templates = np.concatenate((S_test,adhere),axis=0)
            _classes = np.arange(start=0,stop=len(templates),step=1,dtype='int')
            
            templates = code2EEG.predict(S=templates)
            _classes = np.arange(0,len(templates),dtype=int)
            y_test = np.repeat(_classes[:n],repeats=blockNUM)
            # inifinte large of the tenplates but just sample small portion of it for validation

            for t in winLENs:

                model = Match(srate=srate,winLEN=t,lag=lag)
                model.fit(templates,_classes)
                accuracy = model.score(X_,y_test)

                f = pd.DataFrame({
                    'subject': [subName],
                    'accuracy':[accuracy],
                    'N':[len(_classes)],
                    'sample':[n],
                    'ITR':[utils.ITR(len(_classes),accuracy,winLEN)],
                    'winLEN': [t]
                })

                frames.append(f)

            df = pd.concat(frames, axis=0, ignore_index=True)
            add = 'results/%s/%s' % (expName, subName)
            if not os.path.exists(add):
                os.makedirs(add)
            df.to_csv(add+os.sep+saveFILE)
