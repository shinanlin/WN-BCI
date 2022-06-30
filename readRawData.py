import numpy as np
import mne
import os
import pickle
import sys
from scipy import signal
import scipy.io as scio
from tqdm import tqdm

class cntReader():
    """
    should be in the heriachy as :
    -exp
        - subject
            - date
                - session
    """

    def __init__(self, fileAdd, stiLen,srate=250) -> None:
        # file address
        self.fileAdd = fileAdd
        self.subjects = []
        self.sessions = []
        self.srate=srate

        # parameters
        self.stiLen = stiLen
        self.cwd = sys.path[0]

        exp = self.fileAdd.split(os.sep)[-1]
        self.pickleAdd = os.path.join('curry', exp)
        if os.path.exists(self.pickleAdd) is False:
            os.makedirs(self.pickleAdd)
        self.alreadyHave = os.listdir(self.pickleAdd)

    def readRaw(self):
        # get subject name
        self.getSubject()

        for subINX, sub in enumerate(self.subList):
            # for subject level
            subName = sub.split('/')[-1]

            folderList = os.listdir(sub)
            if '.DS_Store' in folderList:
                folderList.remove('.DS_Store')

            for datesRecording in folderList:
                # for dates
                folder = sub+os.sep+datesRecording
                fileList = os.listdir(folder)

                for sessionName in fileList:

                    if os.path.splitext(sessionName)[-1] == '.cnt':
                        sessionName = os.path.join(folder, sessionName)
                        # read continous raw data
                        raw = self._getSession(sessionName)

                        # split continous data into epoch
                        epoch = self.epochSplit(raw)

                        self.sessions.append(epoch)

            if self.sessions != []:
                
                X = np.concatenate([session['X']
                                    for session in self.sessions], axis=0)
                y = np.concatenate([session['y']
                                    for session in self.sessions], axis=0)

                restX = np.concatenate([session['restX']
                                    for session in self.sessions], axis=0)
                                    
                channel = self.sessions[0]['channel']

                sessions = dict(
                    X=X,
                    y=y,
                    restX=restX,
                    channel=channel
                )

                with open('%s/%s.pickle' % (self.pickleAdd, subName), "wb+") as fp:
                    pickle.dump(sessions, fp, protocol=pickle.HIGHEST_PROTOCOL)

                self.sessions = []

        return

    def _getSession(self, sessionName):

        raw = mne.io.read_raw_cnt(
            input_fname=sessionName,
            data_format='auto',
            preload=True,
            date_format='mm/dd/yy')

        return raw

    def epochSplit(self, raw):

        task_events, task,nontask_events,nontask = self.defineEvent(raw)

        taskEpoch = mne.Epochs(raw, task_events, event_id=task,
                           tmin=0, tmax=self.stiLen, preload=True, baseline=None)
        # downsample
 
        taskEpoch.resample(self.srate)
        taskEpoch.filter(4,80)

        nonTaskEpoch = mne.Epochs(raw, nontask_events, event_id=nontask,
                           tmin=0, tmax=10, preload=True, baseline=None)

        nonTaskEpoch = nonTaskEpoch.resample(self.srate)
        nonTaskEpoch.filter(4,80)
        nonTaskEpoch = nonTaskEpoch.get_data()

        y = []
        X = taskEpoch.get_data()
        newd = {v: k for k, v in task.items()}
        for event in task_events[:, -1]:
            y.append(int(newd.get(event)))
        y = np.array(y)

        validChn = taskEpoch.ch_names[:-1]
        data = dict(
            X=X[:,:-1],
            y=y,
            restX=nonTaskEpoch[:,:-1],
            channel=validChn
        )
        return data

    def correctEvent(self, raw):

        events, event_dict = mne.events_from_annotations(raw)
        x = np.squeeze(raw['Trigger'][0])
        onset = np.squeeze(np.argwhere(np.diff(x) > 0))
        if onset != []:
            events[:, 0] = onset[:len(events)]

        # 255是结束trigger,去掉结束trigger
        valid_dict = {k: v for k, v in event_dict.items() if int(k) != 255}
        valid_event = np.stack([e for e in events if e[-1] in [*valid_dict.values()]])

        return valid_dict, valid_event

    def defineEvent(self, raw):

        event_dict, events = self.correctEvent(raw)
        # 这一步是为了把符合条件的event取出来
        task_dict = {k: v for k, v in event_dict.items() if int(k) < 100}
        nontask_dict ={k: v for k, v in event_dict.items() if 90<=int(k) <= 150}

        # keys = [int(key) for key in task.keys()]
        # order = sorted(range(len(keys)), key=lambda k: keys[k])
        # task = task[order]

        task_events = []
        for e in events:
            if e[-1] in task_dict.values():
                task_events.append(e)
        task_events = np.stack(task_events)

        nontask_events = []
        for e in events:
            if e[-1] in nontask_dict.values():
                nontask_events.append(e)
        nontask_events = np.stack(nontask_events)

        return task_events, task_dict,nontask_events,nontask_dict

    def getSubject(self):

        subList = os.listdir(self.fileAdd)

        for ah in self.alreadyHave:
            subList.remove(ah.split('.')[0])

        if '.DS_Store' in subList:
            subList.remove('.DS_Store')

        self.subList = [os.path.join(self.fileAdd, subName)
                        for subName in subList]

class datasetMaker():
    def __init__(self, winLEN=4, afterCue=0.5, visualDelay=0.0, Scansys=0.0, BPsys=0.00, testSize=0) -> None:
        self.srate = 250
        self.winLEN = winLEN

        self.afterCue = afterCue
        self.visualDelay = visualDelay

        # scan具有的系统延迟是0.048s-12个点
        self.ScanDelay = (self.visualDelay+Scansys)
        # bp具有的系统延迟是0.056s-14点
        self.BPDelay = (self.visualDelay+BPsys)

        self.channel = []
        self.testSize = testSize

    def initiation(self, para):

        self.datadir = para.data_dir
        self.savedir = para.dataset_dir
        self.srate = para.down_frequency_sample

        return self

    def ensembleData(self):
        data_list = os.listdir(path=self.datadir)
        if '.DS_Store' in data_list:
            data_list.remove('.DS_Store')
        WholeSet = []
        data_list = sorted(data_list)

        for filename in tqdm(data_list):
            if filename.split('.')[-1] == 'mat' and filename.split('.')[0] != 'Freq_Phase':

                path = os.path.join(self.datadir, filename)
                data = scio.loadmat(path)['data'][0][0]
                raw_data = data['data']
                freqs = data['freqs']
                _, index = np.unique(freqs, return_index=True)
                raw_data = raw_data[:, :, index, :]
                _,_,classNUM,blockNUM = raw_data.shape
                datasetName = filename.split('-')[0]
                labels = np.repeat(np.arange(0,classNUM),blockNUM)
                raw_data = raw_data.transpose((-2,-1,0,1))
                raw_data = np.concatenate(raw_data,axis=0)
                # 记录来自哪个数据集
                if datasetName == 'Alpha':
                    self.channel = [47, 53, 54, 55, 56, 57, 60, 61, 62]
                    # ['PZ','PO5','PO3','POz','PO4','PO6','O1','OZ','O2']
                    epochs = self.extractEpoch(
                        data=raw_data,label=labels, delay=self.ScanDelay)
                elif datasetName == 'Theta':
                    self.channel = [18, 58, 44, 62, 45, 59, 8, 63, 9]
                    epochs = self.extractEpoch(
                        data=raw_data, label=labels,delay=self.BPDelay)
                WholeSet.append(epochs)

        self.splitDataset(WholeSet)

        return

    def splitDataset(self,wholeSet):

        testSize = self.testSize
        dataset = dict(
            trainX = [],
            trainy = [],
            testX = [],
            testy = []
            )

        for subData in wholeSet:

            X = subData['X']
            y = subData['y']

            _class = np.unique(y)

            sub = dict(
                trainX = [],
                trainy = [],
                testX = [],
                testy = []
            )

            for c in _class:

                this_class_X = X[y==c]
                this_class_y = y[y==c]
                epochNUM = len(this_class_X)

                # trainINX = np.arange(0,epochNUM-testSize)
                # testINX = np.arange(epochNUM-testSize,epochNUM)
                
                testINX = np.arange(0, testSize)
                trainINX = np.arange(testSize, epochNUM)


                sub['trainX'].append(this_class_X[trainINX])
                sub['trainy'].append(this_class_y[trainINX])
                
                sub['testX'].append(this_class_X[testINX])
                sub['testy'].append(this_class_y[testINX])

            dataset['trainX'].append(np.concatenate(sub['trainX'],axis=0))
            dataset['trainy'].append(np.concatenate(sub['trainy'],axis=0))

            dataset['testX'].append(np.concatenate(sub['testX'],axis=0))
            dataset['testy'].append(np.concatenate(sub['testy'],axis=0))     


        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        datasetName = os.path.join(self.savedir, 'WholeSet.pickle')
        with open(datasetName, "wb+") as fp:
            pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def extractEpoch(self,data,label,delay):
        
        X_ = []
        Y_ = []

        ls = np.unique(label)
        for l in ls:
            X_.append(data[label == l])
            blockNUM = np.sum(label == l)
            Y_.append([l for _ in range(blockNUM)])

        # 每一个block的训练数目不一样
        X = np.concatenate(X_,axis=0)
        Y = np.concatenate(Y_,axis=0)

        srate = self.srate

        gazeLen = round((self.winLEN)*srate)

        delay = round((self.afterCue+delay)*srate)

        segment_data = np.arange(delay,
                                 delay+gazeLen)

        data = X[:,self.channel,:]
        data = data[:,:,segment_data]

        filtered = np.zeros_like(data)

        for i,epoch in enumerate(data):
            filtered[i] = self._filterEpoch(epoch)

        wholeset = dict(
            X = filtered,
            y = Y
        )

        return wholeset

    def _filterEpoch(self, epoch):

        # # band pass
        fs = 250
        b, a = signal.butter(N=5, Wn=[1, 100], fs=fs, btype='bandpass')

        filtered = np.zeros(epoch.shape)

        for chINX, chn in enumerate(epoch):
            # 去除基线
            chn = chn - chn.mean()
            # 滤波 [1,100]
            chn = signal.filtfilt(b, a, chn)
            # 归一化
            filtered[chINX] = self.minMax(chn,[-1, 1])

        return filtered

    def minMax(self, x, scale):
        nmin, nmax = scale
        x = (x-min(x))/(max(x)-min(x))*(nmax-nmin)+nmin
        return x

class curryDataset(datasetMaker):
    
    def __init__(self, exp='exp-1',winLEN=2, afterCue=0, visualDelay=0, srate=240,curryAdd = 'curry') -> None:
        
        cwd = sys.path[0]
        self.exp = exp
        self.savedir = os.path.join(cwd, 'datasets')
        self.curryAdd = os.path.join(cwd, curryAdd,self.exp)
        self.stiAdd = os.path.join(cwd,'stimulation',self.exp)
        super().__init__(winLEN=winLEN, afterCue=afterCue, visualDelay=visualDelay)
        
        self.srate=srate
        self.sampleLEN = round(self.srate*self.winLEN)

    def readCurry(self):
        rawCurry = 'raw'+os.sep+self.exp
        loader = cntReader(rawCurry,stiLen=self.winLEN,srate=self.srate)
        loader.readRaw()

    def readStimulation(self,subName):

        path = self.stiAdd + os.sep + subName
        stiFile = os.listdir(path)
        stiFile = sorted(stiFile)

        INFO = []
        for sti in stiFile:

            if sti.split('.')[-1]=='mat':

                sti = os.path.join(path, sti)
                info = scio.loadmat(sti)

                INFO.append(dict(
                    record = info['record'],
                    stimulus=info['stimulus'],
                    stiOrder=info['index_code']
                ))
            
        return INFO

    def ensembleData(self):
        # readData
        self.readCurry()

        data_list = os.listdir(path=self.curryAdd)
        if '.DS_Store' in data_list:
            data_list.remove('.DS_Store')
        WholeSet = []
        data_list = sorted(data_list)

        for filename in tqdm(data_list):
            
            subName = filename.split('.')[0]

            # read stimulation
            stimulation = self.readStimulation(subName)

            path = os.path.join(self.curryAdd, filename)

            with open(path, "rb") as fp:
                data = pickle.load(fp)
            
            # reorder
            order = np.argsort(data['y'])
            y = data['y'][order]
            X = data['X'][order]
            s = stimulation[0]['stimulus'][:self.sampleLEN].T
            stimulus = np.stack([s[i-1] for i in y])
            sub = dict(
                X=X,
                y = y,
                restX = data['restX'],
                stimulus=stimulus,
                dropFrame = stimulation[0]['record'],
                tags = np.repeat(['wn','mseq','ssvep'],20),
                channel=data['channel'],
                name = subName,
            )
            # 记录来自哪个数据集
            WholeSet.append(sub)

        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        datasetName = os.path.join(self.savedir, '%s.pickle'%self.exp)
        with open(datasetName, "wb+") as fp:
            pickle.dump(WholeSet, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return 

    def discardBad(self,X,y):
        picked_X = []
        picked_y = []

        old_label = None

        for epochINX,(epoch,label) in enumerate(zip(X,y)):
                
            while (old_label==label):
                break
            else:
                if (epochINX == 0):
                    pass
                else:
                    picked_X.append(old_epoch)
                    picked_y.append(old_label)

            old_label = label
            old_epoch = epoch

        picked_X = np.stack(picked_X)
        picked_y = np.stack(picked_y)
        return picked_X,picked_y

if __name__ == '__main__':

    exp = 'exp-2'
    winLEN = 3
    srate = 240
    curryMaker = curryDataset(exp=exp,winLEN=winLEN,srate=srate)
    curryMaker.ensembleData()
