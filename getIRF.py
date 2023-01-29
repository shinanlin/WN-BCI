import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import ReceptiveField
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from NRC import NRC, recordModule, RegularizedRF
from utils import returnFFT, returnPSD, returnSpec
from scipy import stats
import random

srate = 240
tmin, tmax = -0.1, .4
expName = 'sweep'
chnNames = ['PZ', 'POZ', 'OZ']

random.seed(253)

dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

tags = ['wn']
for sub in tqdm(wholeset):

    for tag in tags:
    
        chnINX = [sub['channel'].index(i) for i in chnNames]

        y = sub[tag]['y']
        X = sub[tag]['X']
        X = X[:,chnINX]
        S = np.stack([sub[tag]['STI'][i-1] for i in y])

        # RF using merely xorrs
        decoder = NRC(srate=srate, tmin=tmin, tmax=tmax, alpha=0.9)

        decoder.fit(R=X, S=S)
        csr = decoder.Csr[:,:,np.newaxis,:]
        recoder = recordModule(srate=srate,sub=sub['name'],chn=chnNames,exp=expName)
        recoder.recordKernel(csr, y, 'locked', tmin, tmax)

        # shuffle
        # permute = np.arange(len(S))
        # random.shuffle(permute)
        # shuffledS = S[permute]
        # decoder.fit(R=X, S=shuffledS[:, :X.shape[-1]])
        # csr = decoder.Csr[:,:,np.newaxis,:]
        # recoder = recordModule(srate=srate,sub=sub['name'],chn=chnNames,exp=expName)
        # recoder.recordKernel(csr, y, 'shuffled', tmin, tmax)

    # recoder.recordEEG(X,y) 

    



