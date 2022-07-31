import numpy as np
import mne
import os
import pickle
import sys
from scipy import signal
import scipy.io as scio
from tqdm import tqdm
from autoreject import AutoReject
import matplotlib.pyplot as plt

class Config():
    def __init__(self, exp='exp-1', srate=250, tstart=0, winLEN=1, event_dict={'wn': 1, 'ssvep': 2, 'rest': 3}) -> None:
        self.exp = exp
        self.srate = srate
        self.tstart = tstart
        self.winLEN = winLEN
        self.event_dict = event_dict
        pass
class cntReader():
    """
    cntReader has one permission:load .cnt file into pkls
    should be in the heriachy as :
    -exp
        - subject
            - date
                - session
    """

    def __init__(self, fileAdd, config) -> None:
        # file address
        self.fileAdd = fileAdd
        self.subjects = []
        self.sessions = []
        self.srate=config.srate
        self.event_dict = config.event_dict

        # parameters
        self.stiLEN = config.winLEN
        self.tstart = config.tstart
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
            self.subName = sub.split('/')[-1]

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
                sessions = np.concatenate(self.sessions)
                with open('%s/%s.pickle' % (self.pickleAdd, self.subName), "wb+") as fp:
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

        #从这里把区分开三种情况区分开
        Events = self.defineEvent(raw)
        DATA = []

        # notch and filter
        raw.notch_filter((50,100))
        
        for this_exp_events in  Events:

            tag, events, events_dict = this_exp_events
            # 睁闭眼30s
            winLEN = self.stiLEN
            taskEpoch = mne.Epochs(raw, events, event_id=events_dict,
                                   tmin=self.tstart, tmax=winLEN+self.tstart, preload=True, baseline=None)
            taskEpoch.drop_channels(['Trigger'])
            # downsample
            taskEpoch.resample(self.srate)
            taskEpoch._filename = raw.filenames[0]+os.sep+tag

            X = taskEpoch.get_data()
            chn = taskEpoch.ch_names

            y= []
            newd = {v: k for k, v in events_dict.items()}
            for event in events:
                y.append(int(newd.get(event[-1])))
            y = np.array(y)
            DATA.append((tag,X,y,chn))

        return DATA

    def discardBad(self,epoch):
        
        picks = ['R1', 'R2', 'R3', 'R4', 'R5',
                'L1', 'L2', 'L3', 'L4', 'L5']

        epoch_ = epoch.copy()
        picked = epoch_.pick(picks)

        print('start auto rejection')

        # maximum number of bad sensors in a non-rejected trial
        n_interpolates = np.array([1,2,3])
        # ρ the maximum number of sensors that can be interpolated
        consensus_percs = np.linspace(0.2, 0.8, 5)
        
        ar = AutoReject(n_interpolates, consensus_percs,picks=picks,cv=8,
                thresh_method='bayesian_optimization', random_state=42)
        ar.fit(picked)

        clean, reject_log = ar.transform(picked, return_log=True)
        reject_plot = reject_log.plot(orientation='horizontal',show=False)

        subFolder = self.dropFolder+os.sep+self.subName
        if os.path.exists(subFolder) is False:
            os.makedirs(subFolder)

        this_drop_file = picked._filename.split('/')
        this_drop_file = this_drop_file[-2].split('.')[0]+'_'+this_drop_file[-1]

        plt.title(this_drop_file)
        reject_plot.savefig('%s/%s.png'%(subFolder,this_drop_file), dpi=400, format='png')

        return clean,reject_log

    def correctEvent(self,raw):

        events, event_dict = mne.events_from_annotations(raw)
        # discard events
        # 255是结束trigger,去掉结束trigger

        x = raw.get_data()[-1]
        onset = np.squeeze(np.argwhere(np.diff(x)>0))
        events[:, 0] = onset[:len(events)]

        return event_dict, events

    def defineEvent(self, raw):
        event_dict, events = self.correctEvent(raw)

        # events = events[20:,]
        # 把符合条件的event取出来
        Events = []

        for key,value in self.event_dict.items():
            
            task_dict = {k: v for k, v in event_dict.items() if int(k) in value}
            task_events = [e for e in events if e[-1] in [*task_dict.values()]]
            Events.append((key, task_events, task_dict))

        return Events

    def getSubject(self):

        subList = os.listdir(self.fileAdd)

        for ah in self.alreadyHave:
            subList.remove(ah.split('.')[0])

        if '.DS_Store' in subList:
            subList.remove('.DS_Store')

        self.subList = [os.path.join(self.fileAdd, subName)
                        for subName in subList]

class datasetMaker():
    
    def __init__(self, config):

        self.config = config
        self.exp = config.exp
        self.srate=config.srate
        self.winLEN = config.winLEN
        self.sampleLEN = round(self.srate*self.winLEN)
        self.tstart = config.tstart
        cwd = sys.path[0]
        self.rawAdd = os.path.join(cwd, 'raw',self.exp)
        self.curryAdd = os.path.join(cwd, 'curry',self.exp)
        self.saveAdd = os.path.join(cwd, 'datasets')
        self.stiAdd = os.path.join(cwd,'stimulation',self.exp)

    def readCurry(self):
        loader = cntReader(self.rawAdd,self.config)
        loader.readRaw()

    def readSTI(self,fileName):

        path = self.stiAdd + os.sep + fileName
        S = scio.loadmat(path)
        STI = S['WN']
        label = S['label']

        STI  = np.repeat(STI,4,axis=-1)
        return STI,label[0]


    def ensemble(self):
        # readData
        self.readCurry()

        dataList = os.listdir(path=self.curryAdd)
        if '.DS_Store' in dataList:
            dataList.remove('.DS_Store')
        wholeSet = []
        dataList = sorted(dataList)


        for filename in tqdm(dataList):
            
            subName = filename.split('.')[0]

            # read stimulation
            path = os.path.join(self.curryAdd, filename)
            with open(path, "rb") as fp:
                sessions = pickle.load(fp)

            STI, label = self.readSTI('STI.mat')
            labels = np.tile(label,6)
            sub = dict()
            for session in sessions:
                tag,X,y,chnNames = session
                if tag == 'wn':
                    s, y = STI, labels[labels > 160]
                else:
                    s, y = STI, labels[labels <= 160]
                exp = dict(
                    X = X,
                    y = y,
                    STI = s
                )
                sub[tag] = exp
            sub['channel'] = chnNames
            sub['name'] = subName

            # # 记录来自哪个数据集
            wholeSet.append(sub)

        if not os.path.exists(self.saveAdd):
            os.makedirs(self.saveAdd)
        datasetName = os.path.join(self.saveAdd, '%s.pickle'%self.exp)
        with open(datasetName, "wb+") as fp:
            pickle.dump(wholeSet, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return 

if __name__ == '__main__':

    event_dict = dict(
        wn=[2],
        ssvep=[1]
    )
    config = Config(exp='dense',srate=240,winLEN=0.6,event_dict=event_dict)
    curryMaker = datasetMaker(config)
    curryMaker.ensemble()
