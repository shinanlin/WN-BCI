import numpy as np
from mne.decoding import ReceptiveField,TimeDelayingRidge
import pickle
from spatialFilters import TDCA
from NRC import NRC

class Code2EEG():

    def __init__(self, S,srate=250, winLEN=1, tmin=0, tmax=0.5, estimator=0.98, scoring='corrcoef',padding=True) -> None:

        self.srate=srate
        self.tmin=tmin
        self.tmax=tmax
        self.winLEN = int(srate*winLEN)
        self.estimator=estimator
        self.scoring = scoring
        self.padding = padding

        self.padLEN = int(0.2*srate) if self.padding else 0
            
        self._loadSTI(S)

        pass


    def _loadSTI(self,*S):

        from scipy.stats import zscore
        
        # load STI as a class attibute
        STI,y = S[0]
        self.montage=np.unique(y)
        self.STI = np.stack([STI[y==i].mean(axis=0) for i in self.montage])
        self.STI = zscore(self.STI,axis=-1)
        
        self.STI = self.STI[:,:self.winLEN]

        return


    def fit(self,X,y):
        
        self._classes = np.unique(y)

        # trim
        X = X[:,:,:self.winLEN]
        N = np.shape(X)[-1]
        # TDCA 
        enhancer = TDCA(srate=self.srate, winLEN=N /
                        self.srate, montage=len(self._classes))
        enhanced = enhancer.fit_transform(X,y)

        STI = np.concatenate([self.STI[self.montage==i] for i in self._classes])

        regressor = NRC(srate=self.srate,tmin=self.tmin,tmax=self.tmax,alpha=self.estimator)
        regressor.fit(R=enhanced,S=STI)

        self.regressor = regressor
        self.enhancer = enhancer
        self.enhanced = enhanced
        self.trf = regressor.trf
        
        pass

    def predict(self,S):
        
        
        pad = np.zeros((S.shape[0],self.padLEN))
        S = np.concatenate((pad,S),axis=-1)
        R_ = self.regressor.predict(S)

        # discard padding
        return R_[:,:,self.padLEN:]
    
    def score(self, S, R):
        from mne.decoding.receptive_field import _SCORERS
        # Create our scoring object
        scorer_ = _SCORERS[self.scoring]

        r_pred = self.predict(S)

        R_ = self.enhancer.transform(R)
        scores = []
        for (r_,r) in zip(r_pred,R_):
            
            score = scorer_(r.T, r_.T, multioutput='raw_values')
            scores.append(score)

        scores = np.stack(scores)
            
        return scores


class ForwardModel(ReceptiveField):

    def __init__(self, tmin, tmax, sfreq, feature_names=None, estimator=None, fit_intercept=None, scoring='corrcoef', patterns=False, n_jobs=1, edge_correction=True, verbose=None):
        self.regressors = []
        super().__init__(tmin, tmax, sfreq, feature_names, estimator, fit_intercept, scoring, patterns, n_jobs, edge_correction, verbose)

    def fit(self, S, R, y):
        
        self._class = np.unique(y)

        kernels = []
        for condition in np.unique(y):
            this_R, this_S = R[y == condition], S[y == condition]
            
            this_R = this_R.transpose((-1, 0, 1))
            this_S = this_S.T[:, :, np.newaxis]
            regressor = TimeDelayingRidge(
                tmin=self.tmin, tmax=self.tmax, sfreq=self.sfreq, alpha=self.estimator)
            # project stimulus to response
            regressor.fit(X=this_S,y=this_R)
            kernels.append(regressor.coef_)
            self.regressors.append(regressor)
        self.kernels = np.stack(kernels)
        self.trf = self.kernels.mean(axis=(0,1))

        return self

    def predict(self,S):
        # predict R according to S
        recon = []
        S = S.T[:, :, np.newaxis]
        for predictor in self.regressors:
            r_ = predictor.predict(S)
            recon.append(r_.transpose((1,-1,0)))


        recon = np.stack(recon)
        self.recon = recon
        # recon = stats.zscore(recon,axis=-1)
        
        return recon.mean(axis=0)

    def score(self, S, R):
        from scipy.stats import zscore
        from mne.decoding.receptive_field import _SCORERS
        # Create our scoring object
        scorer_ = _SCORERS[self.scoring]

        r_pred = self.predict(S)
    
        scores = []
        for (r_,r) in zip(r_pred,R):
            r_,r = zscore(r_.T,axis=0),zscore(r.T,axis=0)
            score = scorer_(r, r_, multioutput='raw_values')
            scores.append(score)
        scores = np.stack(scores)
            
        return scores
      
if __name__=='__main__':
    
    # X = np.random.random((10,9,240))
    # y = np.arange(0,10,1)
    # S = np.random.random((10, 240))

    srate = 240
    expName = 'sweep'

    dir = 'datasets/%s.pickle' % expName
    winLENs = np.arange(0.2, 1, step=.2)
    with open(dir, "rb") as fp:
        wholeset = pickle.load(fp)

    sub = wholeset[3]

    chnNames = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ']
    chnINX = [sub['channel'].index(i) for i in chnNames]
    X = sub['wn']['X'][:, chnINX]
    y = sub['wn']['y']
    # S = np.stack([sub['wn']['STI'][i-1] for i in y])
    S = sub['wn']['STI']
    testINX = np.arange(50)
    X, y, S = X[testINX], y[testINX], S[testINX]
    
    model = Code2EEG(srate=240,tmin=0,tmax=0.5,S=(S,np.unique(y)))
    model.fit(X,y)
    model.score(S,X)

