import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import seaborn as sns
import math
from scipy.fft import fft, fftfreq
import os
from scipy import signal
from scipy.stats import zscore
import pandas as pd
from tensorpac.utils import ITC


def codeDistance(codeset):
    
    n_code = codeset.shape[0]

    D = np.zeros((n_code,n_code))
    for i,s1 in enumerate(codeset):
        for j,s2 in enumerate(codeset):
            D[i,j] = np.linalg.norm(s1-s2)
    
    row,col = np.diag_indices_from(D)
    D[row,col] = np.nan
    return D

def ITR(N, P, winBIN):

    winBIN = winBIN+0.5
    
    if P == 1:
        ITR = math.log2(N)*60/winBIN
    elif P == 0:
        ITR = (math.log2(N) + 0 + (1-P)*math.log2((1-P)/(N-1)))*60/winBIN
        ITR = 0
    else:
        ITR = (math.log2(N)+P*math.log2(P)+(1-P)
               * math.log2((1-P)/(N-1)))*60/winBIN

    return ITR

def returnFFT(s,srate=250):

    s = zscore(s,axis=-1)
    N = s.shape[-1]
    T = 1/srate
    x = np.linspace(0, N*T, N, endpoint=False)
    # performFFT
    fftAmp = fft(s,axis=-1)
    # take only half
    freqz = fftfreq(N, T)[:N//2]
    # norm
    fftAmp = fftAmp.T[:N//2]
    fftAmp = 1.0/N * np.abs(fftAmp)

    return freqz, fftAmp.T


def returnPSD(s,srate=250):

    freqz, fftAmp = signal.welch(
    x=s, axis=-1, fs=srate, nperseg=s.shape[-1]//2, nfft=s.shape[-1], scaling='spectrum', window='boxcar',)


    lowFre = np.argwhere((1<freqz)&(freqz<80)).squeeze()
    freqz = freqz[lowFre]
    fftAmp = fftAmp.T
    fftAmp = fftAmp[lowFre].T

    return freqz,fftAmp


def returnSpec(s,srate=250):

    f, t, Sxx = signal.spectrogram(s, srate, nperseg=18, nfft=1000,
                       noverlap=16, mode='psd', scaling='density',axis=-1)
    
    return f, t, Sxx


def lowFilter(s,srate=250,band=[3,15]):
    # # band pass
    fs = srate
    b, a = signal.butter(N=3, Wn=band, fs=fs, btype='bandpass')
    fS = signal.filtfilt(b, a, s,axis=-1)
    return fS


def returnStimulus(y,winLEN,srate=250):

    frequncy = np.linspace(8, 15.8, num=40)
    phase = np.tile(np.arange(0, 2, 0.5)*math.pi, 10)

    t = np.arange(0, winLEN, 1/srate)
    S_set = []

    for f, p in zip(frequncy, phase):
        S_set.append([np.sin(2*np.pi*i*f+p) for i in t])

    S_set = np.stack(S_set)
    S = np.stack([S_set[i] for i in y])

    return S


def returnHilbert(y,srate=250):

    complex = hilbert(y,axis=-1)
    phase = np.unwrap(np.angle(complex))
    
    return phase

def returnITC(X,srate=250):

    # X in shape [epoch,chnnal,T]
    f_min,f_max = 3,70

    X_ = np.transpose(X,(1,0,-1))

    itc = [ITC(x, srate, f_pha=(f_min, f_max, 1, 0.2),cycle=20).itc for x in X_]
    
    freqz = ITC(X[0], srate, f_pha=(f_min, f_max, 1, 0.2), cycle=20)._xvec

    itc = np.stack(itc).mean(axis=-1)

    return freqz,itc


def returnSNR(X,y,srate=250):

    epochN,chnN,N = X.shape
    n_harmonic = 6
    fs = srate

    frequency = np.linspace(8, 15.8, num=9)
    SNRs = np.zeros((epochN,chnN))

    for epochINX,(epoch,label) in enumerate(zip(X,y)):

        f = frequency[label]

        for chnINX,chn in enumerate(epoch):

            allPower = np.square(abs(fft(chn)/N))
            fpower = 0
            for nh in np.arange(1, n_harmonic+1):

                n = np.arange(N)
                omega = 2*(f*nh)/fs*np.pi
                temp = -1j*omega*(n-1)
                component = np.sum(chn.dot(np.exp(temp)))
                fc = np.square(np.abs(component/N))

                fpower = fpower + fc

            baseline = np.sum(allPower[1+N//2:])
            snrdata = fpower / (baseline - fpower)
            snrdata = 10*np.log10(snrdata)

            SNRs[epochINX, chnINX] = snrdata

    return SNRs


def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )

    return psd / mean_noise


def narrow_snr(X,srate=1000):

    freq,psd = returnPSD(X,srate=srate)
    
    snrs = snr_spectrum(psd,noise_n_neighbor_freqs=5,noise_skip_neighbor_freqs=1)
    
    return snrs,freq


class recordModule():

    def __init__(self,sub='wsub1',recordAdd='results',exp='exp-1',srate=240,chn=np.linspace(0,63,64).astype('int64')):

        self.subName = str(sub)
        self.recordAdd=recordAdd
        self.exp = exp
        self.chnMontage = chn
        self.srate=srate
        self.checkFolder()


    def checkFolder(self):

        folder = os.path.join(self.recordAdd, self.exp,self.subName)
        if os.path.exists(folder) is False:
            os.makedirs(folder)
        self.recordAdd = folder
        return

    def returnSNR(self,*X):
        freqz,SNRs,labels = X
        epochNUM,chnNUM,_ = SNRs.shape
        frames = []

        for epoch,label in zip(SNRs,labels):
            
            frame = pd.DataFrame(data=epoch,index=self.chnMontage[:chnNUM],columns=freqz)
            frame.reset_index(level=0, inplace=True)
            frame = frame.melt(id_vars='index', value_name='amplitude',
                               var_name='frequency')
            frame = frame.rename(columns={'index': 'channel'})
            frame['condition'] = label
            
            frames.append(frame)

        filePath = self.recordAdd+os.sep+'SNR.csv'
        df = pd.concat(frames,ignore_index=True,axis=0)
        df['subject'] = self.subName

        df.to_csv(filePath)

        return


    def recordEEG(self,*X):

        if len(X) ==2:
            EEG,labels = X
        else:
            raise Exception('Input variables must contain data and labels')

        _,chnNUM,T = EEG.shape
        t = np.arange(0,T/self.srate,1/self.srate)

        frames = []
        for epoch,label in zip(EEG,labels):
            
            frame = pd.DataFrame(data=epoch,index=self.chnMontage[:chnNUM],columns=t)
            frame.reset_index(level=0, inplace=True)
            frame = frame.melt(id_vars='index', value_name='EEG',
                               var_name='time')
            frame = frame.rename(columns={'index': 'channel'})
            frame['condition'] = label
            
            frames.append(frame)

        filePath = self.recordAdd+os.sep+'EEG.csv'
        df = pd.concat(frames,ignore_index=True,axis=0)
        df['subject'] = self.subName
        df.to_csv(filePath)

        return 

    def recordStimulus(self,*X):

        S,y = X
        # labels = np.arange(1,len(S)+1,step=1)
        T = S.shape[-1]
        t = np.arange(0, T/self.srate, 1/self.srate)
        frame = pd.DataFrame(data=S,index=y, columns=t)
        frame.reset_index(level=0, inplace=True)

        frame = frame.melt(id_vars='index', value_name='stimulus',
                           var_name='time')
        frame = frame.rename(columns={'index': 'condition'})

        df = frame
        filePath = self.recordAdd+os.sep+'stimulus.csv'

        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df, exsited])
            df.drop_duplicates()
        df.to_csv(filePath)

        return

    def recordKernel(self,*ensemble):
        
        # kernel epoch*chn*T
        if len(ensemble) == 6:
            kernel, labels, type,config,tmin,tmax = ensemble
        elif len(ensemble) == 5:
            kernel, labels, type,tmin,tmax = ensemble
        elif len(ensemble) == 4:
            kernel, labels, tmin,tmax = ensemble
        elif len(ensemble) == 3:
            kernel, labels, type = ensemble
            tmin,tmax = 0,kernel.shape[-1]/self.srate
        elif len(ensemble) == 2:
            kernel,labels ==  ensemble
            type = 'regularized'
            tmin, tmax = 0, kernel.shape[-1]/self.srate
        else:
            raise Exception('Input variables must contain data and labels')

        _,chnNUM,featureNUM,T = kernel.shape
        lags = np.arange(tmin, tmax+(1/self.srate), 1/self.srate)
        frames = []

        for epoch,label in zip(kernel,labels):

            epoch = np.transpose(epoch,axes=(1,0,-1))

            for featureINX,feature in enumerate(epoch):
                frame = pd.DataFrame(data=feature,index=self.chnMontage[:chnNUM], columns=lags)
                frame.reset_index(level=0, inplace=True)
                frame = frame.melt(id_vars='index', value_name='trf',var_name='lags')
                frame = frame.rename(columns={'index': 'channel'})
                frame['condition'] = label
                frame['feauture'] = featureINX

                frames.append(frame)
        
        df = pd.concat(frames, ignore_index=True)
        df['subject'] = self.subName
        df['type'] = type
        df['config'] = config

        filePath = self.recordAdd+os.sep+'TRF.csv'
        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df,exsited])
            df.drop_duplicates()

        df.to_csv(filePath)
        return

    def recordPearson(self,corr,config):

        frames = []
        for i,r in enumerate(corr):
            frame = pd.DataFrame(data=r, index=['attendModel','ignoreModel'],columns=['attendS','ignoreS'])
            frame.reset_index(level=0, inplace=True)
            frame = frame.rename(columns={'index': 'model'})
            frame['cv'] = i
            frames.append(frame)

        df = pd.concat(frames,ignore_index=True)
        df['subject'] = self.subName
        df['config'] = config

        filePath = self.recordAdd+os.sep+'corr.csv'
        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df, exsited])
            df.drop_duplicates()
        df.to_csv(filePath)

        pass

    def recordPhase(self,*X):

        EEG, labels = X
        
        _, chnNUM, T = EEG.shape
        t = np.arange(0, T/self.srate, 1/self.srate)
        frames = []
        for epoch, label in zip(EEG, labels):

            frame = pd.DataFrame(
                data=epoch, index=self.chnMontage[:chnNUM], columns=t)
            frame.reset_index(level=0, inplace=True)
            frame = frame.melt(id_vars='index', value_name='Theta',
                               var_name='time')
            frame = frame.rename(columns={'index': 'channel'})
            frame['condition'] = label

            frames.append(frame)

        filePath = self.recordAdd+os.sep+'Phase.csv'
        df = pd.concat(frames, ignore_index=True, axis=0)
        df['subject'] = self.subName
        df.to_csv(filePath)

        return
 
    def recordSpectral(self,*X):
        
        if len(X) ==2:
            EEG,labels = X
        elif len(X) == 3:
            freqz,EEG,labels = X
        else:
            raise Exception('Input variables must contain data and labels')

        _,chnNUM,T = EEG.shape

        frames = []
        for epoch,label in zip(EEG,labels):
            
            frame = pd.DataFrame(data=epoch,index=self.chnMontage[:chnNUM],columns=freqz)
            frame.reset_index(level=0, inplace=True)
            frame = frame.melt(id_vars='index', value_name='amplitude',
                               var_name='frequency')
            frame = frame.rename(columns={'index': 'channel'})
            frame['condition'] = label
            
            frames.append(frame)

        filePath = self.recordAdd+os.sep+'spectral.csv'
        df = pd.concat(frames,ignore_index=True,axis=0)
        df['subject'] = self.subName

        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df,exsited],ignore_index=True,axis=0)
            df.dropna(how='all')
            df.drop_duplicates()

        df.to_csv(filePath)

        return

    def recordITC(self,*X):

        freqz,ITC,labels = X
        chnNUM,_ = ITC.shape

        frame = pd.DataFrame(data=ITC,index=self.chnMontage[:chnNUM],columns=freqz)
        frame.reset_index(level=0, inplace=True)
        frame = frame.melt(id_vars='index', value_name='ITC',
                            var_name='frequency')
        frame = frame.rename(columns={'index': 'channel'})

        filePath = self.recordAdd+os.sep+'ITC.csv'
        frame['subject'] = self.subName

        frame.to_csv(filePath)

        return

    

if __name__ == '__main__':

    # R = np.random.random((10, 64, 30))
    R = np.random.random((10,64,720))
    y = np.arange(1,10,1)
    S = np.random.random((10, 720))


    # recoder = recordModule()
    # recoder.recordKernel(rf.kernel,y,'r',)
