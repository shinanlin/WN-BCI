
import sys
sys.path.append('.')

import numpy as np
import pickle
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm
from modeling  import Code2EEG,EEG2Code
from utils import returnFFT,returnPSD
import os

#%%
# this script is for generating codespace for optimization

#%%
# parameters
srate = 250
winLEN = 1 
n_band = 1
n_component = 1
p = 0.95
refreshrate = 60

tmin = -0.5
tmax = 0
expName = 'sweep'
tag = 'wn'
saveFILE = 'info.csv'


#%% load dataset

dir = 'datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

chnNames = ['PZ', 'PO3','PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']
chnINX = [wholeset[0]['channel'].index(i) for i in chnNames]

#%% refresh

add = 'results'+os.sep+expName

for fnames in os.listdir(add):
    f = add+os.sep+fnames+os.sep+saveFILE
    if os.path.exists(f):
        os.remove(f)
#%%

for i,sub in tqdm(enumerate(wholeset)):

    frames = []

    X = sub[tag]['X'][:, chnINX]
    y = sub[tag]['y']
    S = sub[tag]['STI']
    subName = sub['name']

    _class = np.unique(y)
    STI = np.concatenate([S[_class == i] for i in y])

    #%% build backward model

    model = EEG2Code(srate=srate,winLEN=winLEN,tmin=tmin,tmax=tmax,S=(S,np.unique(y)),estimator=p,padding=True,n_band=n_band,component=n_component)
  
    model.fit(X,y)
    #%% compute the lower bound

    # single trial reconstrcution of STI
    sEst = model.predict(X)

    # noise of stimulus
    sN = STI-sEst 
    freqz, STI_F = returnFFT(STI)
    freqz, sEst_F = returnFFT(sEst)
    freqz, sN_F = returnFFT(sN)


    # calculate the complecx conjugate
    conj_sEst, conj_sN = np.conjugate(sEst_F), np.conjugate(sN_F)

    covF_SS = sEst_F*conj_sEst
    cov_NN = sN_F*conj_sN
    # <trial average>
    lbSNR = covF_SS.mean(axis=0)/cov_NN.mean(axis=0)
    lbSNR[freqz > refreshrate//2] = 0
    lbINFOrate = np.cumsum(np.log2(1+lbSNR))
    lbINFO = np.sum(np.log2(1+lbSNR))

    #%% compute the upper bound

    aveEvoked = np.squeeze(model.enhanced)
    aveEvoked = np.concatenate([aveEvoked[_class == i] for i in y])

    xX = model.enhancer.transform(X)
    xX = np.squeeze(xX)
    
    xNoise = xX - aveEvoked

    freqz, rF = returnFFT(np.squeeze(model.enhanced))
    freqz, S_F = returnFFT(model.STI)
    rConjugate = np.conjugate(rF)
    # Hf: frequency TRF
    Hf = (rConjugate*S_F).mean(axis=0)/(rConjugate*rF).mean(axis=0)

    freqz, ss = returnFFT(aveEvoked)
    freqz, nn = returnFFT(xNoise)
    freqz, xx = returnFFT(xX)

    STIPower = 1/(srate*winLEN) * (np.abs(STI_F)**2)

    xPower = 1/(srate*winLEN) * (np.abs(xx)**2)
    sPower = 1/(srate*winLEN) * (np.abs(ss)**2)
    nPower = 1/(srate*winLEN) * (np.abs(nn)**2)
    
    ubSNR = sPower.mean(axis=0)/nPower.mean(axis=0)
    ubINFOrate = np.cumsum(np.log2(1+ubSNR))
    ubINFO = np.sum(np.log2(1+ubSNR))

    #%% record

    f = pd.DataFrame({

        'f': freqz,
        'lbSNR': np.abs(lbSNR),
        'lbrate': np.abs(lbINFOrate),
        'ubSNR': np.abs(ubSNR),
        'ubrate': np.abs(ubINFOrate),
        'SS': np.abs(covF_SS).mean(axis=0),
        'NN': np.abs(cov_NN).mean(axis=0),
        'sPower':np.abs(sPower).mean(axis=0),
        'nPower':np.abs(nPower).mean(axis=0),
        'Hf': np.abs(Hf),
        'X':np.abs(xPower).mean(axis=0),
        'STI':np.abs(STIPower).mean(axis=0)
        
    })

    f['subject'] = subName
    f['ubINFO'] = ubINFO
    f['lbINFO'] = lbINFO

    frames.append(f)

    df = pd.concat(frames, axis=0, ignore_index=True)
    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)
