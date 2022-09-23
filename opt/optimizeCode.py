from sklearn.model_selection import train_test_split

import numpy as np
import pickle
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import random
from modeling import Code2EEG,Match
from utils import codeDistance
import pandas as pd
import os

# %%
# define parameters
srate = 240
winLEN = 1
tmin, tmax = 0, .8
n_band = 1
# penalty
p = 0.9
targetNUM = 40
seedNUM = 100
saveFILE = 'optimize-coef.csv'

# %%

expName = 'sweep'
dir = 'datasets/%s.pickle' % expName
winLENs = np.arange(0.2, 1, step=.2)
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

X_ = []
y_ = []
S_ = []


for sub in wholeset:

    chnNames = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    X_.append(sub['wn']['X'][:,chnINX])
    y_.append(sub['wn']['y'])


X_ = np.concatenate(X_)
y_ = np.concatenate(y_)


S = sub['wn']['STI']
_class = np.unique(y_)

# modeling response
code2eeg = Code2EEG(srate=srate,winLEN=winLEN,tmin=tmin,tmax=tmax,S=(S,_class),estimator=p,padding=True,n_band=n_band)
code2eeg.fit(X_,y_)

# response: reconstructed and real evoked potentials
# build a relationship bwteen these: code(STI), code response(real_X) and modelled code response(fake_X)
STI = np.stack(S[i-1] for i in y_)

fake_X = code2eeg.predict(STI)
real_X = code2eeg.enhancer.transform(X_)

# s = np.stack([real_X[y_==i].mean(axis=0) for i in _class])
# s = np.transpose(s,axes=(1,-1,0))
# savemat("response.mat" , data)


frames = []
# %%
for seed in tqdm(np.arange(seedNUM)):    
    
    random.seed(seed)
    picked = random.sample(_class.tolist(), targetNUM)

    response_R = np.concatenate([real_X[y_ == i] for i in picked])
    response_F= np.concatenate([fake_X[y_ == i] for i in picked])
    code_label = np.concatenate([y_[y_ == i] for i in picked])
    code = np.concatenate([STI[y_ == i] for i in picked])

    code_ = np.stack([code[code_label == i].mean(axis=0) for i in np.unique(code_label)])
    real_code_response = np.stack([response_R[code_label == i].mean(axis=0) for i in np.unique(code_label)])
    fake_code_response = np.stack([response_F[code_label == i].mean(axis=0) for i in np.unique(code_label)])


    _,code_dist = codeDistance(code_)
    _,f_response_dist = codeDistance(fake_code_response)
    _,r_response_dist = codeDistance(real_code_response)
    
    # calculate decoding accuracy
    model = Match(winLEN=winLEN,srate=srate,lag=0)
    model.fit(response_F, code_label)
    score = model.score(response_R,code_label)

    f = pd.DataFrame({
        'distance' : [r_response_dist,f_response_dist,code_dist],
        'type' : ['response','reconstruct','code']
    })
    f['accuracy'] = score
    f['seed'] = seed


    # for (dists,dist_name) in zip([code_dist,f_response_dist,r_response_dist],['code','reconstruct','response']):
    #     f = pd.DataFrame(data=dists, index=np.arange(
    #         targetNUM), columns=np.arange(targetNUM))
    #     f.reset_index(level=0, inplace=True)
    #     f = f.melt(id_vars='index', value_name='coef',
    #                         var_name='code-2')
    #     f = f.rename(columns={'index': 'code-1'})

    #     f['type'] = dist_name
    #     f['seed'] = seed
    #     f['accuracy'] = score

    frames.append(f)

    df = pd.concat(frames,axis=0,ignore_index=True)

    add = 'results/%s' % (expName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)
