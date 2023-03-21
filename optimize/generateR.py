
import sys
sys.path.append('.')

import numpy as np
import pickle
from scipy.signal import resample
from tqdm import tqdm
import random
from compare.modeling  import Code2EEG
import os
from compare.utils import recordModule
#%%
# this script is for generating codespace for optimization

#%%
# parameters
srate = 250
winLEN = 1 
classNUM = 160
tmin, tmax = 0, .8
n_band = 1
n_component = 1
p = 0.9
refreshrate = 60
poolSize = int(1e4)

expName = 'sweep'
#%% load dataset

dir = 'datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)
    
rep  = wholeset[0]

chnNames = ['PZ', 'PO3','PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']
chnINX = [wholeset[0]['channel'].index(i) for i in chnNames]
S = rep['wn']['STI']
_class = np.unique(rep['wn']['y'])
X_sub = np.concatenate([sub['wn']['X'] for sub in wholeset])
y_sub = np.concatenate([sub['wn']['y'] for sub in wholeset])

group = dict()
group['wn'] = dict(
    X=X_sub,
    y=y_sub,
)
group['name'] = 'group'
wholeset.append(group)

#%%
# generate random sequence
np.random.seed(253)
# re-create massive loads of STI patterns
buildSTI = np.random.uniform(0, 1, (int(poolSize), int(winLEN*refreshrate)))
# upsample
upsamSTI  = resample(buildSTI,num=int(winLEN*srate),axis=-1)

#%%
preliminary = dict()
codeSpace = dict()

for i,sub in tqdm(enumerate(wholeset)):

    X_sub = sub['wn']['X'][:, chnINX]
    y_sub = sub['wn']['y']
    name = sub['name']
    
    code2EEG = Code2EEG(srate=srate, winLEN=winLEN, tmin=tmin, tmax=tmax, S=(S, _class), estimator=p, padding=True, n_band=n_band, component=n_component)
    code2EEG.fit(X_sub, y_sub)

    simulate = code2EEG.predict(S)

    # enhanced pattern
    pattern = code2EEG.enhanced

    s = S.T

    sub = dict()
    sub['name'] = name
    sub['pattern'] = np.transpose(pattern, (1, -1, 0))
    sub['simulate'] = np.transpose(simulate,(1,-1,0))
    sub['stimulus'] = s
    preliminary['S%s' % i] = sub

    
    recorder = recordModule(sub=name,exp=expName,srate=srate)
    recorder.recordEEG(pattern,code2EEG._classes)

    if name == 'group':
        extra = code2EEG.predict(upsamSTI)
        sub_codespace = dict()
        sub_codespace['name'] = name
        sub_codespace['simulate'] = np.transpose(extra,(1,-1,0))
        sub_codespace['stimulus'] = buildSTI
        codeSpace = sub_codespace


from scipy.io import savemat
savemat("seqOpt/mat/preliminary_polorized.mat" , preliminary)

