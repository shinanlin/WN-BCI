from statistics import mode
import numpy as np
import os
from numpy.lib.function_base import append, diff, select
from numpy.lib.utils import source
from numpy.linalg.linalg import qr
import scipy.linalg as la

from scipy.signal.filter_design import freqs
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.metrics import accuracy_score
from scipy import stats, signal
import math
import copy
from NRC import NRC
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind

class TRCA():

    def __init__(self, n_components=1, n_band=1, montage=40, winLEN=2, lag=35,srate=250):

        self.n_components = n_components
        self.n_band = n_band
        self.montage = montage
        self.frequncy = np.linspace(8, 15.8, num=montage)
        self.phase = np.tile(np.arange(0, 2, 0.5)*math.pi, 10)
        self.srate = srate
        self.winLEN = round(self.srate*winLEN)
        self.lag = lag

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of TRCA
            Returns the modified instance.
        """

        X = X[:, :, self.lag:self.lag+self.winLEN]

        self._classes = np.unique(y)

        self.filter = []
        self.evokeds = []
        for fbINX in range(self.n_band):
            # 每种信号都有不同的模版

            filter = []
            evokeds = []

            for classINX in self._classes:

                this_class_data = X[y == classINX]
                this_band_data = self.filterbank(
                    this_class_data, self.srate, freqInx=fbINX)
                    
                evoked = np.mean(this_band_data, axis=0)
                evokeds.append(evoked)
                weight = self.computer_trca_weight(this_band_data)
                filter.append(weight[:, :self.n_components])

            self.filter.append(np.concatenate(filter, axis=-1))
            self.evokeds.append(np.stack(evokeds))

        self.filter = np.stack(self.filter)
        self.evokeds = np.stack(self.evokeds).transpose((1, 0, 2, 3))

        return self

    def transform(self, X, y):
        from scipy.stats import zscore
        """
        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            shape is (n_epochs, n_sources, n_times).
        """

        enhanced = np.zeros((self.n_band, self.montage, self.winLEN))

        for classINX, classEvoked in enumerate(self.evokeds):
            for fbINX, (filter, fb) in enumerate(zip(self.filter, classEvoked)):
                enhance = np.dot(fb.T, filter[:, classINX])
                enhanced[fbINX, classINX] = enhance
        
        
        enhanced = zscore(enhanced,axis=-1)

        return enhanced

    def fit_transform(self, X, y):

        return self.fit(X, y).transform(X, y)

    def IRF(self, X, y):

        X = X[:, :, :self.winLEN]
        N = X.shape[-1]
        # extract template
        labels = np.unique(y)
        conNUM = len(labels)
        ave = np.zeros((conNUM, N))
        for i, l in enumerate(labels):
            ave[i] = X[y == l].mean(axis=(0, 1))

        # 生成和数据等长的正弦刺激
        sti = self.sine(N)[labels]

        score = np.zeros((conNUM, N))
        for i, (t1, t2) in enumerate(zip(sti, ave)):
            s = np.correlate(t2-t2.mean(), t1-t1.mean(), mode='full')
            s = s / (N * t2.std() * t1.std())
            score[i] = s[N-1:]

        ave_score = score.mean(axis=0)
        self.lag = np.argmax(ave_score)

        return score, self.lag

    def sine(self, winLEN):
        sine = []
        win = winLEN
        t = np.arange(0, (win/self.srate), 1/self.srate)
        for f, p in zip(self.frequncy, self.phase):
            sine.append([np.sin(2*np.pi*i*f+p) for i in t])
        self.sti = np.stack(sine)
        return self.sti

    def predict(self, X):

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)

        # crop test data according to predefined latency

        X = X[:, :, self.lag:self.lag+self.winLEN]

        result = []

        
        H = np.zeros(X.shape[0])
        fb_coefs = np.expand_dims(
            np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)

        self.rho = np.zeros((X.shape[0], self.montage))

        for epochINX, epoch in enumerate(X):

            r = np.zeros((self.n_band, self.montage))

            for fbINX in range(self.n_band):

                epoch_band = np.squeeze(self.filterbank(epoch, self.srate, fbINX))

                for (classINX, evoked) in zip(np.arange(self.montage), self.evokeds):

                    template = np.squeeze(evoked[fbINX, :, :])
                    w = np.squeeze(self.filter[fbINX, :])
                    rtemp = np.corrcoef(
                        np.dot(epoch_band.T, w).reshape(-1), np.dot(template.T, w).reshape(-1))
                    r[fbINX, classINX] = rtemp[0, 1]

            rho = np.dot(fb_coefs, r)

            self.rho[epochINX] = rho
            # hypothesis testing
            target = np.nanargmax(rho)
            rhoNoise = np.delete(rho, target)
            rhoNoise = np.delete(rhoNoise, np.isnan(rhoNoise))
            _, H[epochINX], _ = ttest_ind(rhoNoise, [rho[0, target]])

            result.append(self._classes[target])

        self.confidence = H

        return np.stack(result)

    def residual(self, x, result):

        evoked = self.evokeds[int(result)]

        for fbINX in range(self.n_band):

            epoch_band = np.squeeze(self.filterbank(x, self.srate, fbINX))
            template = np.squeeze(evoked[fbINX, :, :])
            w = np.squeeze(self.filter[fbINX, :])
            t = np.dot(template.T, w).reshape(-1)
            s = np.dot(epoch_band.T, w).reshape(-1)
            residual = t-s
            stats.ttest_1samp(residual, 0.0)

        pass

    def score(self, X, y):

        return accuracy_score(y, self.predict(X))

    def dyStopping(self, X, former_win):

        # 判断要设置什么窗
        srate = self.srate
        p_val = 0.0025

        dyStopping = np.arange(0.4, former_win+0.1, step=0.2)

        for ds in dyStopping:

            ds = int(ds*srate)
            self.predict(X[:, :, :ds+self.lag])

            score = self.confidence < p_val
            pesudo_acc = np.sum(score != 0)/len(score)
            print('mean confidence:', self.confidence.mean())
            print('pesudo_acc {pesudo_acc} %'.format(pesudo_acc=pesudo_acc))

            # if pesudo_acc >= 0.85:
            #     boostWin = ds
            #     break

            if self.confidence.mean() < p_val:
                boostWin = ds
                break

        # 难分的epoch下一次继续
        n = np.argsort(self.confidence)
        difficult = X[n[-5:]]

        if not 'boostWin' in locals().keys():
            boostWin = int(dyStopping[-1]*srate)

        return (boostWin/srate), difficult

    def recordCoff(self, X, y, subINX, adINX, frames):
        # 判断要设置什么窗
        epochNUM = len(y)
        srate = self.srate
        former_win = 500
        dyStopping = np.arange(0.4, (former_win/srate)+0.1, step=0.2)

        for ds in dyStopping:
            ds = int(ds*srate)
            labels = self.predict(X[:, :, :ds])

            coff = self.confidence.squeeze()
            personCoff = self.confidencePer.squeeze()
            refCoff = self.confidenceRef.squeeze()

            coff = np.concatenate([coff, personCoff, refCoff])
            tags = np.repeat(['combined', 'personal', 'ref'], epochNUM)
            trueLabel = np.tile(y, 3)
            predictedLabel = np.tile(labels, 3)
            correctness = np.tile(y == labels, 3)

            frame = pd.DataFrame({
                'coff': coff,
                'tags': tags,
                'trueLabel': trueLabel,
                'predictedLabel': predictedLabel,
                'correctness': correctness
            })
            frame['personID'] = 'S %s' % subINX
            frame['adINX'] = adINX
            frame['winLEN'] = ds

            frames.append(frame)

        # df = pd.concat(frames, axis=0)
        # df.to_csv('dynamics/S{inx}.csv'.format(inx=subINX))

        return frames

    def filterbank(self, x, srate, freqInx):

        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

        srate = srate/2

        Wp = [passband[freqInx]/srate, 90/srate]
        Ws = [stopband[freqInx]/srate, 100/srate]
        [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
        [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')

        filtered_signal = np.zeros(np.shape(x))
        if len(np.shape(x)) == 2:
            for channelINX in range(np.shape(x)[0]):
                filtered_signal[channelINX, :] = signal.filtfilt(
                    B, A, x[channelINX, :])
            filtered_signal = np.expand_dims(filtered_signal, axis=-1)
        else:
            for epochINX, epoch in enumerate(x):
                for channelINX in range(np.shape(epoch)[0]):
                    filtered_signal[epochINX, channelINX, :] = signal.filtfilt(
                        B, A, epoch[channelINX, :])

        return filtered_signal

    def computer_trca_weight(self, eeg):
        """
        Input:
            eeg : Input eeg data (# of targets, # of channels, Data length [sample])
        Output:
            W : Weight coefficients for electrodes which can be used as a spatial filter.
        """
        epochNUM, self.channelNUM, _ = eeg.shape

        S = np.zeros((self.channelNUM, self.channelNUM))

        for epoch_i in range(epochNUM):

            x1 = np.squeeze(eeg[epoch_i, :, :])
            x1 = x1 - np.mean(x1, axis=1, keepdims=True)

            for epoch_j in range(epoch_i+1, epochNUM):
                x2 = np.squeeze(eeg[epoch_j, :, :])
                x2 = x2 - np.mean(x2, axis=1, keepdims=True)
                S = S + np.dot(x1, x2.T) + np.dot(x2, x1.T)

        UX = np.stack([eeg[:, i, :].ravel() for i in range(self.channelNUM)])
        UX = UX - np.mean(UX, axis=1, keepdims=True)
        Q = np.dot(UX, UX.T)

        C = np.linalg.inv(Q).dot(S)
        _, W = np.linalg.eig(C)

        return W


class fbCCA():
    def __init__(self, n_components=1, n_band=5, srate=240, conditionNUM=20, lag=35, winLEN=2):
        self.n_components = n_components
        self.n_band = n_band
        self.srate = srate
        self.conditionNUM = conditionNUM
        self.montage = np.linspace(0, conditionNUM-1, conditionNUM).astype('int64')
        self.frequncy_info = np.linspace(8, 17.5, num=self.conditionNUM)

        self.lag = lag
        self.winLEN = int(self.srate*winLEN)


    def fit(self, X=[], y=[]):

        epochLEN = self.winLEN

        self._classes = np.unique(y)
        sineRef = self.get_reference(
            self.srate, self.frequncy_info, n_harmonics=5, data_len=epochLEN)

        self.evokeds = sineRef

        return self

    def predict(self, X):

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)

        X = X[:, :, self.lag:self.lag+self.winLEN]

        result = []

        H = np.zeros(X.shape[0])
        corr = np.zeros(X.shape[0])

        fb_coefs = np.expand_dims(
            np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)

        for epochINX, epoch in enumerate(X):

            r = np.zeros((self.n_band, self.conditionNUM))
            cca = CCA(n_components=1)
            for fbINX in range(self.n_band):
                epoch_band = np.squeeze(self.filterbank(epoch, self.srate, fbINX))
                for (classINX, evoked) in zip(self.montage, self.evokeds):
                    u, v = cca.fit_transform(evoked.T, epoch_band.T)
                    rtemp = np.corrcoef(u.T, v.T)
                    r[fbINX, classINX] = rtemp[0, 1]
            rho = np.dot(fb_coefs, r)

            # snr = np.power(rho,2)/(1-np.power(rho,2)) * featureNUM
            target = np.nanargmax(rho)
            # snrNoise = np.delete(snr,target)
            rhoNoise = np.delete(rho, target)

            _, H[epochINX], _ = ttest_ind(rhoNoise, [rho[0, target]])
            corr[epochINX] = rho[0, target]

            result.append(self._classes[target])

        self.confidence = H
        self.corr = corr

        return np.stack(result)

    def filterbank(self, x, srate, freqInx):

        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

        srate = srate/2
        Wp = [passband[freqInx]/srate, 90/srate]
        Ws = [stopband[freqInx]/srate, 100/srate]
        [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
        [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')

        filtered_signal = np.zeros(np.shape(x))
        if len(np.shape(x)) == 2:
            for channelINX in range(np.shape(x)[0]):
                filtered_signal[channelINX, :] = signal.filtfilt(
                    B, A, x[channelINX, :])
            filtered_signal = np.expand_dims(filtered_signal, axis=-1)
        else:
            for epochINX, epoch in enumerate(x):
                for channelINX in range(np.shape(epoch)[0]):
                    filtered_signal[epochINX, channelINX, :] = signal.filtfilt(
                        B, A, epoch[channelINX, :])

        return filtered_signal

    def get_reference(self, srate, frequncy_info, n_harmonics, data_len):

        t = np.arange(0, (data_len/srate), 1/srate)
        reference = []

        for j in range(n_harmonics):
            harmonic = [np.array([np.sin(2*np.pi*i*frequncy_info*(j+1)) for i in t]),
                        np.array([np.cos(2*np.pi*i*frequncy_info*(j+1)) for i in t])]
            reference.append(harmonic)
        reference = np.stack(reference[i] for i in range(len(reference)))
        reference = np.reshape(
            reference, (2*n_harmonics, data_len, len(frequncy_info)))
        reference = np.transpose(reference, (-1, 0, 1))

        return reference

    def score(self, X, y):

        return accuracy_score(y, self.predict(X))


class TDCA(TRCA):

    def __init__(self, n_components=1, n_band=5, montage=40, winLEN=2, lag=35, srate=250):
        super().__init__(n_components, n_band, montage, winLEN, lag, srate)
        
    def fit(self, X, y):
        # X: 160*9*750
        self._classes = np.unique(y)
        self.filters = []
        self.evokeds = []
        self.epochNUM, self.channelNUM, N = X.shape

        # 先子带滤波，否则会引入很大计算量
        X = self.augmentation(X)

        X = np.transpose(X, axes=(1, 0, -2, -1))

        augumentX = []

        for fbX in X:

            augumentClass = []

            for classINX in self._classes:
                this_class_data = fbX[y == classINX]

                augumentEpoch = []

                for epoch in this_class_data:

                    augumentEpoch.append(epoch)

                augumentClass.append(np.stack(augumentEpoch))

            augumentX.append(augumentClass)

        # augumentX:
        # 如果每个condition的数目不一样，就不能stack
        # augumentX = np.stack(augumentX)

        self.computer_tdca_weight(augumentX)

        return self
    
    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            shape is (n_epochs, n_sources, n_times).
        """

        if y is None:
            y = np.arange(X.shape[0])

        enhanced = np.zeros((self.montage, self.n_band, self.winLEN))

        fbed_X = self.augmentation(X)
        for conditionINX,condition in enumerate(np.unique(y)):
            classEvoked = fbed_X[y==condition].mean(axis=0)
            for fbINX,(fb,filter) in enumerate(zip(classEvoked,self.filters)):
                enhance = fb.T.dot(filter)
                enhanced[conditionINX,fbINX] = enhance.T
            
        return enhanced


    def predict(self, X):

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)

        result = []
        
        H = np.zeros(X.shape[0])
        fb_coefs = np.expand_dims(
            np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)

        X = self.augmentation(X)

        for epochINX, epoch in enumerate(X):

            r = np.zeros((self.n_band, len(self.montage)))

            for (classINX, evoked) in zip(self._classes, self.evokeds):

                for fbINX, (fbEvoked, fbEpoch, filter) in enumerate(zip(evoked, epoch, self.filters)):

                    rtemp = np.corrcoef((
                        np.dot(fbEpoch.T, filter).reshape(-1),
                        np.dot(fbEvoked.T, filter).reshape(-1)
                    ))
                    r[fbINX, classINX] = rtemp[0, 1]

            rho = np.dot(fb_coefs, r)
            missing = np.setdiff1d(
                self.montage, self._classes)  # missing filter
            rho[:, missing] = None

            target = np.nanargmax(rho)
            rhoNoise = np.delete(rho, target)
            rhoNoise = np.delete(rhoNoise, np.isnan(rhoNoise))
            _, H[epochINX], _ = ttest_ind(rhoNoise, [rho[0, target]])
            result.append(target)

        self.confidence = H

        return np.stack(result)


    def computer_tdca_weight(self, augmentX):

        augmentEvoked = []
        for fbs in augmentX:
            augmentEvoked.append([con.mean(axis=0) for con in fbs])
        # augmentEvoked: 5*40*9*478
        augmentEvoked = np.stack(augmentEvoked)

        for (fbEvoked, fbEpochs) in zip(augmentEvoked, augmentX):
            # norm
            # fbEvoked:40*9*478; fbEpochs:40*9*478
            fbEvoked = fbEvoked-np.mean(fbEvoked, axis=-1, keepdims=True)
            fbEvokedFeature = -np.mean(fbEvoked, axis=0, keepdims=True)
            betwClass = fbEvoked-fbEvokedFeature
            betwClass = np.concatenate(betwClass, axis=1)
            # norm
            fbEpochs = [
                this_class-np.mean(this_class, axis=-1, keepdims=True) for this_class in fbEpochs
            ]
            allClassEvoked = [
                this_class-np.mean(this_class, axis=0, keepdims=True) for this_class in fbEpochs
            ]

            allClassEvoked = [np.transpose(this_class, axes=(
                1, 2, 0)) for this_class in allClassEvoked]
            allClassEvoked = [np.reshape(
                this_class, (self.channelNUM, -1), order='F') for this_class in allClassEvoked]
            allClassEvoked = np.hstack(allClassEvoked)

            Hb = betwClass/math.sqrt(self.montage)
            Hw = allClassEvoked/math.sqrt(self.epochNUM)
            Sb = np.dot(Hb, Hb.T)
            Sw = np.dot(Hw, Hw.T)+0.001*np.eye(Hw.shape[0])

            # inv(Sw)*B
            C = np.linalg.inv(Sw).dot(Sb)
            _, W = np.linalg.eig(C)
            _, W = la.eig(C)

            # tmd又是反的？
            self.filters.append(W[:, :self.n_components])

        self.evokeds = np.transpose(augmentEvoked, axes=(1, 0, -2, -1))

        return


    def augmentation(self, X):

        augmentX = []
        for epoch in X:
            fbedEpoch = []
            for fbINX in range(self.n_band):
                fbEpoch = self.filterbank(epoch, self.srate, fbINX)
                fbedEpoch.append(fbEpoch)
            fbedEpoch = np.concatenate(fbedEpoch, axis=-1)
            augmentX.append(fbedEpoch)

        augmentX = np.stack(augmentX)
        augmentX = np.transpose(augmentX, axes=(0, -1, 1, 2))

        return augmentX


class Matching(fbCCA):

    def __init__(self, n_components=1, n_band=5, srate=250, conditionNUM=40, lag=35, winLEN=2,tmin=0,tmax=0.5):
        self.tmin = tmin
        self.tmax = tmax
        super().__init__(n_components, n_band, srate, conditionNUM, lag, winLEN)

    def fit(self, S, y):

        # load stimulus waves

        self._classes = np.unique(y)
        
        S = np.stack([S[y==i].mean(axis=0) for i in self._classes])
    
        X_  = self.subBand(X)

        rf = NRC(srate=self.srate,tmin=self.tmin,tmax=self.tmax)
        rf.fit()
        

        NRC

        return self



    def augment(self,S):

        ampNUM = self.ampNUM
        self.amp = np.arange(0, ampNUM+1)

        ampX = []

        for drift in self.amp:
            ampX.append(S[:,np.newaxis,drift:drift+self.winLEN])
        S_ = np.concatenate(ampX,axis=1)
        
        return S_

    def subBand(self,X):
        
        Bands = []

        for epoch in X:

            filtered = []

            for fb in range(self.n_band):
            
                filtered.append(np.squeeze(self.filterbank(epoch,self.srate,fb)))

            Bands.append(np.stack(filtered))

        Bands = np.stack(Bands)

        return Bands



if __name__ == '__main__':


    X = np.random.random((40,9,240))
    y = np.arange(1,41,1)
    S = np.random.random((40, 240))

    model = TDCA(winLEN=1,srate=240)
    # model.fit(X,y)
    model.fit_transform(X,y)
