
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.stats import zscore
from sympy import re


def returnFFT(s,srate=240):

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


def returnPSD(s,srate=240):

    freqz, fftAmp = signal.welch(x=s, axis=-1, fs=srate, nperseg=100,nfft=1000,scaling='spectrum')

    lowFre = np.argwhere((0<freqz)&(freqz<45)).squeeze()
    freqz = freqz[lowFre]
    fftAmp = fftAmp.T
    fftAmp = fftAmp[lowFre].T

    return freqz,fftAmp


def returnSpec(s,srate=240):

    f, t, Sxx = signal.spectrogram(s, srate, nperseg=18, nfft=1000,
                       noverlap=16, mode='psd', scaling='density',axis=-1)
    
    return f, t, Sxx


def lowFilter(s,srate=240):
    # # band pass
    fs = srate
    b, a = signal.butter(N=5, Wn=[5, 20], fs=fs, btype='bandpass')
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



