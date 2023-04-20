
import sys
sys.path.append('.')
sys.path.append('./compare')

import numpy as np
import pickle
import pandas as pd
from compare.spatialFilters import TDCA,vanilla
from tqdm import tqdm
from compare.modeling  import Code2EEG,EEG2Code
from compare.utils import returnFFT,returnPSD
import os


#%%
# parameters
srate = 250
winLEN = 0.5
n_band = 1
n_component = 1
_t = np.arange(0,winLEN,1/srate)

expNames = ['gamma']

tags = ['WN', 'SSVEP']
saveFILE = 'psd.csv'


#%% load dataset

DATASETs = []

for exp in expNames:
        
    dir = 'data/datasets/%s.pickle' % exp
    with open(dir, "rb") as fp:
        set = pickle.load(fp)
    DATASETs.append(set)


#%% refresh
for (expName, dataset) in zip(expNames, DATASETs):

    add = 'results'+os.sep+expName
    if os.path.exists(add):
        for fnames in os.listdir(add):
            f = add+os.sep+fnames+os.sep+saveFILE
            if os.path.exists(f):
                os.remove(f)
    else:
        os.mkdir(add)
#%%
    for i,sub in tqdm(enumerate(dataset)):
        
        frames = []
        for tag in tags:
            
            if tag in sub.keys():

                chnINX = [78, 75, 71, 72, 77, 76, 87, 90]

                X = sub[tag]['X'][:, chnINX]
                y = sub[tag]['y']
                
                subName = sub['name']

                _class = np.unique(y)
                N = X.shape[-1]
                winLEN = N/srate


                #%% 
                
                freqz, ss = returnFFT(X,srate=srate)
                ss = ss.mean(axis=1)
                # spectral power
                xPower = (1/(srate*N)) * (np.abs(ss)**2)
                # average across trial
                xPower = np.stack([xPower[y == k] for k in np.unique(y)])

                xPower = np.mean(xPower, axis=1)

                #%% record
                
                for name,sig in zip(['X'],[xPower]):
                    f = pd.DataFrame(columns=freqz,index=_class,data=sig)
                    f.reset_index(level=0, inplace=True)
                    f = f.melt(id_vars='index', value_name='psd',
                            var_name='f')
                    f = f.rename(columns={'index': 'label'})

                    f['component'] = name
                    f['tag'] = tag
                    frames.append(f)

        df = pd.concat(frames, axis=0, ignore_index=True)
        df['subject'] = subName
        df['exp'] = expName

        #%% save
        add = 'results/%s/%s' % (expName, subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)
