from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from forwardModeling import Code2EEG
from spatialFilters import TDCA


class EEG2Code(Code2EEG):

    def __init__(self, S, srate=250, winLEN=1, tmin=0, tmax=0.5, estimator=0.98, scoring='corrcoef', padding=True, n_band=5, component=3) -> None:
        super().__init__(S, srate, winLEN, tmin, tmax, estimator, scoring, padding, n_band, component)


    def fit(self, X, y):

        # input:EEG output:stimulus sequence(code)
        
        super().fit(X, y)

        self._classes = np.setdiff1d(self.montage,self._classes)
        return self


    def predict(self,X):

        n_epoch = X.shape[0]

        fb_coefs = (np.arange(1, self.band+1)**-1.25+0.25)[:,np.newaxis]

        # get_template
        reconEvoked = super().predict(self.STI) 
        X_evoked = self.enhancer.transform(X)

        reconEvoked = np.reshape(reconEvoked,(len(self.montage),self.component,self.band,-1),order='F')
        X_evoked = np.reshape(X_evoked,(n_epoch,self.component,self.band,-1),order='F')

        rhos = np.zeros((X_evoked.shape[0], len(self.montage), self.component))

        for epochINX,this_x in enumerate(X_evoked):

            for evokedINX,(condition,this_r) in enumerate(zip(self.montage,reconEvoked)):
                
                if condition in self._classes:
                    for componentINX,(fb_x,fb_r) in enumerate(zip(this_x,this_r)):
                        
                        coef = 0

                        for fbINX,(x,r) in enumerate(zip(fb_x,fb_r)):
                            
                                rho = np.corrcoef(r,x)[0,1]
                                coef += rho*fb_coefs[fbINX]

                        rhos[epochINX,evokedINX,componentINX] = coef
                else:
                    rhos[epochINX,evokedINX,:] = np.nan

        # rhos = rhos.sum(axis=-1)
        targets = np.nanargmax(rhos,axis=1)
        
        return np.squeeze(self.montage[targets])

    def score(self, X, y):

        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))



class Match():

    def __init__(self, srate=250, winLEN=1, lag=0.14) -> None:
        
        self.srate=srate
        self.winLEN =round(self.srate*winLEN)
        self.lag = round(self.srate*lag)

        pass

    def fit(self, X, y):

        # X: X should be reconstruct response,X: 160*3*5
        
        X = X[:,:,self.lag:self.lag+self.winLEN]
        _,chn,T = X.shape
        self._classes = np.unique(y)

        self.evokeds = np.zeros((len(self._classes),chn,T))
        for i,label in enumerate(self._classes):
            self.evokeds[i] = X[y==label].mean(axis=0)
        
        return self

    def predict(self, X):

        from sklearn.cross_decomposition import CCA

        fb_coef = np.arange(1, 5+1)**-1.25+0.25
        # predict input should be filtered response!!!
        X = X[:, :, self.lag:self.lag+self.winLEN]
        n_epoch = X.shape[0]
        coef = np.zeros((n_epoch,len(self._classes)))
        for i,epoch in enumerate(X):
            for j,evoked in enumerate(self.evokeds):
                r1 = 0
                for fbINX,(fb_epoch,fb_evoked) in enumerate(zip(epoch,evoked)):
                    r = np.corrcoef(fb_epoch,fb_evoked)[0,1]
                    r1 +=  r*fb_coef[fbINX]
                coef[i,j] = r1

                    # u,v = CCA(n_components=1).fit_transform(epoch.T,evoked.T)
                    # coef[i,j] = np.corrcoef(u.T, v.T)[0,1]

        index = np.argmax(coef,axis=1)
        y = self._classes[index]

        return y

    def score(self,X,y):

        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


if __name__=='__main__':

    srate = 240
    expName = 'sweep'

    dir = 'datasets/%s.pickle' % expName
    winLENs = np.arange(0.2, 1, step=.2)
    with open(dir, "rb") as fp:
        wholeset = pickle.load(fp)

    sub = wholeset[2]

    chnNames = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ']
    chnINX = [sub['channel'].index(i) for i in chnNames]
    X = sub['wn']['X'][:, chnINX]
    y = sub['wn']['y']
    S = sub['wn']['STI']

    # reshape to class*block*chn*N
    X_ = np.stack(X[y == i] for i in np.unique(y))
    y_ = np.stack(y[y == i] for i in np.unique(y))

    # split conditions
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y_, test_size=0.5, random_state=253)

    X_train, X_test = np.concatenate(
        X_train, axis=0), np.concatenate(X_test, axis=0)
    y_train, y_test = np.concatenate(
        y_train, axis=0), np.concatenate(y_test, axis=0)
    S_train, S_test = np.stack([S[i-1] for i in y_train]),np.stack([S[i-1] for i in y_test])

    
    model = EEG2Code(srate=240,tmin=0,tmax=0.9,S=(S,np.unique(y)),component=1,estimator=0.8)
    model.fit(X_train,y_train)
    s = model.score(X_test,y_test)
    print(s)
