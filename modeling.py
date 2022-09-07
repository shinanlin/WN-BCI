import numpy as np
from spatialFilters import TDCA
from NRC import NRC
from sklearn.model_selection import train_test_split


class Code2EEG():

    def __init__(self, S, srate=250, winLEN=1, tmin=0, tmax=0.5, estimator=0.98, scoring='corrcoef', padding=True, n_band=5, component=3) -> None:

        self.srate = srate
        self.tmin = tmin
        self.tmax = tmax
        self.winLEN = int(srate*winLEN)
        self.estimator = estimator
        self.scoring = scoring
        self.band = n_band
        self.component = component
        self.padding = padding
        self.padLEN = int(0.2*srate) if self.padding else 0

        self._loadSTI(S)

        pass

    def _loadSTI(self, *S):

        from scipy.stats import zscore

        # load STI as a class attibute
        STI, y = S[0]
        self.montage = np.unique(y)
        self.STI = np.stack([STI[y == i].mean(axis=0) for i in self.montage])
        self.STI = zscore(self.STI, axis=-1)
        self.STI = self.STI[:, :self.winLEN]

        return

    def fit(self, X, y):

        self._classes = np.unique(y)

        # trim
        X = X[:, :, :self.winLEN]
        N = np.shape(X)[-1]
        # TDCA
        enhancer = TDCA(srate=self.srate, winLEN=N /
                        self.srate, montage=len(self._classes), n_band=self.band, n_components=self.component)

        # input: enhanced response and the respective STI
        enhanced = enhancer.fit_transform(X, y)
        # reshaped enhance to (fb * components)
        STI = np.concatenate([self.STI[self.montage == i]
                              for i in self._classes])

        regressor = NRC(srate=self.srate, tmin=self.tmin,
                        tmax=self.tmax, alpha=self.estimator)
        regressor.fit(R=enhanced, S=STI)

        self.regressor = regressor
        self.enhancer = enhancer
        self.enhanced = enhanced
        self.trf = regressor.trf

        return self

    def predict(self, S):

        # padding for VEP onset
        pad = np.zeros((S.shape[0], self.padLEN))
        S = np.concatenate((pad, S), axis=-1)
        R_ = self.regressor.predict(S)

        # discard padding
        return R_[:, :, self.padLEN:]

    def score(self, S, R):

        from mne.decoding.receptive_field import _SCORERS
        # Create our scoring object
        scorer_ = _SCORERS[self.scoring]

        r_pred = self.predict(S)

        R_ = self.enhancer.transform(R)
        scores = []
        for (r_, r) in zip(r_pred, R_):

            score = scorer_(r.T, r_.T, multioutput='raw_values')
            scores.append(score)

        scores = np.stack(scores)

        return scores


class Match():

    def __init__(self, srate=250, winLEN=1, lag=0.14) -> None:

        self.srate = srate
        self.winLEN = round(self.srate*winLEN)
        self.lag = round(self.srate*lag)

        pass

    def fit(self, X, y):

        # X: X should be reconstruct response,X: 160*3*5

        X = X[:, :, self.lag:self.lag+self.winLEN]
        _, chn, T = X.shape
        self._classes = np.unique(y)

        self.evokeds = np.zeros((len(self._classes), chn, T))
        for i, label in enumerate(self._classes):
            self.evokeds[i] = X[y == label].mean(axis=0)

        return self

    def predict(self, X):

        from sklearn.cross_decomposition import CCA

        fb_coef = np.arange(1, 5+1)**-1.25+0.25
        # predict input should be filtered response!!!
        X = X[:, :, self.lag:self.lag+self.winLEN]
        n_epoch = X.shape[0]
        coef = np.zeros((n_epoch, len(self._classes)))
        for i, epoch in enumerate(X):
            for j, evoked in enumerate(self.evokeds):
                r1 = 0
                for fbINX, (fb_epoch, fb_evoked) in enumerate(zip(epoch, evoked)):
                    r = np.corrcoef(fb_epoch, fb_evoked)[0, 1]
                    r1 += r*fb_coef[fbINX]
                coef[i, j] = r1

                # u,v = CCA(n_components=1).fit_transform(epoch.T,evoked.T)
                # coef[i,j] = np.corrcoef(u.T, v.T)[0,1]

        index = np.argmax(coef, axis=1)
        y = self._classes[index]

        return y

    def score(self, X, y):

        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


if __name__ == '__main__':

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
    S_train, S_test = np.stack(
        [S[i-1] for i in y_train]), np.stack([S[i-1] for i in y_test])

    model = EEG2Code(srate=240, tmin=0, tmax=0.9, S=(
        S, np.unique(y)), component=1, estimator=0.8)
    model.fit(X_train, y_train)
    s = model.score(X_test, y_test)
    print(s)
