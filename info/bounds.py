
import sys
sys.path.append('.')
sys.path.append('./compare')

import numpy as np
import pickle
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm
from compare.modeling  import Code2EEG,EEG2Code
from compare.utils import returnFFT
from scipy.integrate import simpson
import os

#%%
# this script is for generating codespace for optimization

#%%
# parameters
srate = 250
winLEN = 0.5 
n_band = 1
n_component = 1
p = 0.95
refreshrate = 60

tmin = -0.3
tmax = 0
expName = 'compare'
tag = 'WN'
saveFILE = 'info.csv'

#%% load dataset

dir = 'data/datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

chnNames  = ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3',
                       'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2']
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
    N = X.shape[-1]
    STI = np.concatenate([S[_class == i] for i in y])

    #%% build backward model

    model = EEG2Code(srate=srate,winLEN=winLEN,tmin=tmin,tmax=tmax,S=(S,np.unique(y)),estimator=p,padding=True,n_band=n_band,component=n_component)
  
    model.fit(X,y)
    #%% compute the lower bound

    # single trial reconstrcution of STI
    sEst = model.predict(X)

    # noise of stimulus
    sN = STI - sEst 
    freqz, STI_F = returnFFT(STI,srate=srate)
    freqz, sEst_F = returnFFT(sEst,srate=srate)
    freqz, sN_F = returnFFT(sN,srate=srate) 

    # calculate the complecx conjugate
    conj_sEst, conj_sN = np.conjugate(sEst_F), np.conjugate(sN_F)

    covF_SS = sEst_F*conj_sEst
    cov_NN = sN_F*conj_sN
    # <trial average>
    lbSNR = [covF_SS[y == i].mean(axis=0)/cov_NN[y == i].mean(axis=0) for i in _class]
    logSNR = [np.log2(1+snr) for snr in lbSNR]
    logSNR = np.mean(logSNR,axis=0)
    logSNR[freqz >= refreshrate//2] = 0
    lbINFOrate = [simpson(logSNR[:n], freqz[:n])
                  for n in np.arange(1, len(freqz), 1)]
    lbINFOrate.insert(0,0)
    lbINFO = lbINFOrate[-1]
    lbSNR = np.mean(lbSNR,axis=0)
    lbSNR[freqz >= refreshrate//2] = 0

    #%% compute the upper bound

    aveEvoked = np.squeeze(model.enhanced)
    aveEvoked = np.concatenate([aveEvoked[_class == i] for i in y])

    xX = model.enhancer.transform(X)
    xX = np.squeeze(xX)
    
    xNoise = xX - aveEvoked

    freqz, rF = returnFFT(np.squeeze(model.enhanced),srate=srate)
    freqz, S_F = returnFFT(model.STI,srate=srate)
    rConjugate = np.conjugate(rF)
    # Hf: frequency TRF
    Hf = (rConjugate*S_F).mean(axis=0)/(rConjugate*rF).mean(axis=0)

    freqz, ss = returnFFT(aveEvoked,srate=srate)
    freqz, nn = returnFFT(xNoise,srate=srate)
    freqz, xx = returnFFT(xX,srate=srate)

    STIPower = 1/(srate*N) * (np.abs(STI_F)**2)

    xPower = 1/(srate*N) * (np.abs(xx)**2)
    sPower = 1/(srate*N) * (np.abs(ss)**2)
    nPower = 1/(srate*N) * (np.abs(nn)**2)
    
    # ubSNR = sPower.mean(axis=0)/nPower.mean(axis=0)
    K=6
    ubSNR = [sPower[y==i].mean(axis=0)/nPower[y==i].mean(axis=0) for i in _class]
    logSNR = [np.log2(1+((K-1)/K*snr-1/K)) for snr in ubSNR]
    logSNR = np.mean(logSNR,axis=0)
    # logSNR = np.log2(1+ubSNR)
    ubINFOrate = [simpson(logSNR[:n], freqz[:n])
                  for n in np.arange(1, len(freqz), 1)]
    ubINFOrate.insert(0, 0)
    ubINFO = ubINFOrate[-1]
    ubSNR = np.mean(ubSNR,axis=0)

    #%% record

    f = pd.DataFrame({

        'f': freqz,
        'lbSNR': np.abs(lbSNR),
        'lbrate': np.abs(lbINFOrate),
        'ubSNR': np.abs(ubSNR),
        'ubrate': np.abs(ubINFOrate),
        'SS': np.abs(covF_SS).mean(axis=0),
        'NN': np.abs(cov_NN).mean(axis=0),
        'sPower':sPower.mean(axis=0),
        'nPower':nPower.mean(axis=0),
        'Hf': np.abs(Hf),
        'X':xPower.mean(axis=0),
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
