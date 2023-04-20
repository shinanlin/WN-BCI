import sys
sys.path.append('.')

from scipy import rand
import pickle
import pandas as pd
import os
from tqdm import tqdm
from compare.spatialFilters import *


# %%
srate = 250
expName = 'compare'
n_band=1
targetNUM = 40
codespace = 40
saveFILE = 'pattern.csv'
winLENs = [0.02,0.1,0.2,0.3,0.4,0.5]
lag = 0.14

# %%
# this script is for computing the performance in general
# parameters

expName = 'compare'
dir = './data/datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)


chNames = wholeset[0]['channel']
chnINX = [chNames.index(i) for i in chNames if i not in ['M1','M2','CB1','CB2']]

for i in ['M1','M2','CB1','CB2']:
    chNames.remove(i)
OZ = [chNames.index(i) for i in chNames if i in ['OZ']]

# %%
for sub in tqdm(wholeset):
    
    frames = []

    subName = sub['name']

    add = 'results/%s/%s' % (expName,subName)        
    for tag in ['WN','SSVEP']:

        X = sub[tag]['X'][:, chnINX]
        y = sub[tag]['y'][:]

        for winLEN in winLENs:

            model = TDCA(winLEN=winLEN, lag=lag, srate=srate,
                            montage=codespace, n_band=n_band)
            model.fit(X, y)

            pattern = model.pattern
            m = np.argmax(np.abs(np.squeeze(pattern)))
            pattern = pattern / pattern[m]
            
            f = pd.DataFrame(data=pattern,index=chNames)
            f.reset_index(level=0, inplace=True)
            f = f.melt(id_vars='index', value_name='w',var_name='band')
            f = f.rename(columns={'index': 'channel'})

            f['winLEN'] = winLEN
            f['method'] = tag
            f['subject'] = subName

            frames.append(f)
    
        df = pd.concat(frames,axis=0,ignore_index=True)

        add = 'results/%s/%s' % (expName,subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)

