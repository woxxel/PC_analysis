''' contains various useful program snippets for neuron analysis:

  get_nFolder   get number of folders in path
  pathcat       attach strings to create proper paths
  _hsm          half sampling mode to obtain baseline


'''

import os, pickle, cmath, time, cv2, h5py
import scipy as sp
import scipy.stats as sstats
from scipy import signal, cluster
import numpy as np
import matplotlib.pyplot as plt
from fastcluster import linkage
from scipy.spatial.distance import squareform

from .utils import gauss_smooth


def get_performance(paths,rw_pos=[50,70],rw_delay=0,f=15,plot_bool=False,plot_ax=None):

    nSes = len(paths)
    L = 120         ## length of track in cm
    nbin = 100
    if len(rw_pos) <= 2:
        rw_pos = np.full((len(paths),2),rw_pos)
    if np.isscalar(rw_delay):
        rw_delay = np.full(len(paths),rw_delay)
    range_approach = [-2,4]      ## in secs
    ra = range_approach[1]-range_approach[0]
    vel_arr = np.linspace(0.5,30,51)

    dataStore = {}
    ### can only get performance from mouse behavior: stopping / velocity
    for s,path in enumerate(paths):

        # pathSession = os.path.join('Session%02d'%s)

        dataBH = prepare_behavior(path)
        if dataBH is None:
            continue

        dataBH['velocity'] *= L/nbin
        try:
            hist = gauss_smooth(np.histogram(dataBH['velocity'],vel_arr)[0],2,mode='nearest')

            vel_run_idx = signal.find_peaks(hist,distance=10,prominence=10)[0][-1]
            vel_run = vel_arr[vel_run_idx]
            vel_min_idx = signal.find_peaks(-hist,distance=5)[0]
            vel_min_idx = vel_min_idx[vel_min_idx<vel_run_idx][-1]
            vel_thr = vel_arr[vel_min_idx]
        except:
            vel_thr = dataBH['velocity'].mean()

        dataStore[s] = {}
        dataStore[s]['trials'] = {}
        dataStore[s]['trials']['RW_reception'] = np.zeros(dataBH['trials']['ct'],'bool')
        dataStore[s]['trials']['RW_frame'] = np.zeros(dataBH['trials']['ct'],'int')-100
        dataStore[s]['trials']['slowDown'] = np.zeros(dataBH['trials']['ct'],'bool')
        dataStore[s]['trials']['frame_slowDown'] = np.zeros(dataBH['trials']['ct'],'int')
        dataStore[s]['trials']['pos_slowDown'] = np.zeros(dataBH['trials']['ct'])*np.NaN
        dataStore[s]['trials']['t_slowDown_beforeRW'] = np.zeros(dataBH['trials']['ct'])*np.NaN

        dataBH['RW_approach_time'] = np.zeros((dataBH['trials']['ct'],ra*f))
        dataBH['RW_approach_space'] = np.zeros((dataBH['trials']['ct'],nbin))*np.NaN
        for t in range(dataBH['trials']['ct']):

            pos_trial = dataBH['binpos'][dataBH['trials']['start'][t]:dataBH['trials']['start'][t+1]].astype('int')
            vel_trial = dataBH['velocity'][dataBH['trials']['start'][t]:dataBH['trials']['start'][t+1]]
            time_trial = dataBH['time'][dataBH['trials']['start'][t]:dataBH['trials']['start'][t+1]]
            for j,p in enumerate(range(nbin)):
                dataBH['RW_approach_space'][t,j] = vel_trial[pos_trial==p].mean()

            # try:        ## fails, when last trial is cut off, due to measure end
            idx_enterRW = np.where(pos_trial>rw_pos[s,0])[0][0]         ## find, where first frame within rw position is
            idx_RW_reception = int(idx_enterRW + rw_delay[s]*f)

            if pos_trial[idx_RW_reception]<rw_pos[s,1]:
                dataStore[s]['trials']['RW_frame'][t] = dataBH['trials']['start'][t] + idx_RW_reception
                dataStore[s]['trials']['RW_reception'][t] = True
                idx_trough_tmp = signal.find_peaks(-vel_trial,prominence=2,height=-vel_thr,distance=f)[0]
                # print(idx_enterRW)
                idx_trough_tmp = idx_trough_tmp[idx_trough_tmp>idx_enterRW]
                if len(idx_trough_tmp)>0:
                    idx_trough = idx_enterRW + idx_trough_tmp[0]
                    ### slowing down should occur before this - defined by drop below threshold velocity
                    slow_down = np.where((vel_trial[:idx_trough]>vel_thr) & (pos_trial[:idx_trough]<=rw_pos[s,1]) & (pos_trial[:idx_trough]>5))[0]# & (pos_trial[:idx_trough]<rw_pos[i,1]))[0]
                    if len(slow_down) > 0:
                        slow_down = slow_down[-1]
                        # print(pos_trial[slow_down])
                        if vel_trial[slow_down+1] < vel_thr :#vel_trial[slow_down+1]<vel_thr:
                            dataStore[s]['trials']['slowDown'][t] = True
                            dataStore[s]['trials']['frame_slowDown'][t] = dataBH['trials']['start'][t] + slow_down
                            dataStore[s]['trials']['pos_slowDown'][t] = pos_trial[slow_down]
                            dataStore[s]['trials']['t_slowDown_beforeRW'][t] = time_trial[idx_RW_reception] - time_trial[slow_down]
            # except:
            #     continue

            idx_enterRW = int(dataBH['trials']['start'][t]+np.where(pos_trial>rw_pos[s,0])[0][0] + rw_delay[s]*f)     ## find, where first frame within rw position is

            dataBH['RW_approach_time'][t,:ra*f+np.minimum(0,len(dataBH['velocity'])-(idx_enterRW+f*range_approach[1]))] = dataBH['velocity'][idx_enterRW+f*range_approach[0]:idx_enterRW+f*range_approach[1]]

        # plot_fig = False
        if not (plot_ax is None):

            plot_ax.plot(np.linspace(0,nbin-1,nbin),gauss_smooth(dataBH['RW_approach_space'],(0,1)).T,color=[0.5,0.5,0.5],alpha=0.5,linewidth=0.3)
            plot_ax.plot(np.linspace(0,nbin-1,nbin),np.nanmean(dataBH['RW_approach_space'],0),color='k')
            # plot_ax.plot(dataStore[s]['trials']['pos_slowDown'][dataStore[s]['trials']['slowDown']],dataBH['velocity'][dataStore[s]['trials']['frame_slowDown'][dataStore[s]['trials']['slowDown'][:]]],'rx')
            plot_ax.plot([0,nbin],[vel_thr,vel_thr],'k--',linewidth=0.5)


        if plot_bool:
            plt.figure()
            plt.subplot(221)
            plt.plot(np.linspace(range_approach[0],range_approach[1],f*ra),dataBH['RW_approach_time'].T,color=[0.5,0.5,0.5],alpha=0.5)
            plt.plot(np.linspace(range_approach[0],range_approach[1],f*ra),dataBH['RW_approach_time'].mean(0),color='k')
            plt.plot(-dataStore[s]['trials']['t_slowDown_beforeRW'][dataStore[s]['trials']['slowDown'][:]],dataBH['velocity'][dataStore[s]['trials']['frame_slowDown'][dataStore[s]['trials']['slowDown'][:]]],'rx')
            plt.plot(range_approach,[vel_thr,vel_thr],'k--',linewidth=0.5)
            plt.xlim(range_approach)
            plt.subplot(222)
            plt.plot(np.linspace(0,nbin-1,nbin),dataBH['RW_approach_space'].T,color=[0.5,0.5,0.5],alpha=0.5)
            plt.plot(np.linspace(0,nbin-1,nbin),np.nanmean(dataBH['RW_approach_space'],0),color='k')
            plt.plot(dataStore[s]['trials']['pos_slowDown'][dataStore[s]['trials']['slowDown']],dataBH['velocity'][dataStore[s]['trials']['frame_slowDown'][dataStore[s]['trials']['slowDown'][:]]],'rx')
            plt.plot([0,nbin],[vel_thr,vel_thr],'k--',linewidth=0.5)
            ax = plt.subplot(223)
            ax.hist(dataBH['velocity'],np.linspace(0.5,30,51))
            ax.plot(np.linspace(0.5,30,50),hist)
            ax.plot([vel_thr,vel_thr],[0,ax.get_ylim()[-1]],'k--')
            plt.show(block=False)

    return dataStore


def prepare_behavior(pathBehavior,nbin=100,nbin_coarse=20,f=15.,T=None):
    '''
        loads behavior from specified path and processes it for analysis of active periods
        Requires file to contain a dictionary with values for each frame, aligned to imaging data:
            * time      - time in seconds
            * position  - mouse position
            * active    - boolean array defining active frames (included in analysis)

    '''
    ### load data
    with open(pathBehavior,'rb') as f_open:
        loadData = pickle.load(f_open)

    if T is None:
        T = loadData['time'].shape[0]
    print(loadData.keys())
    ## first, handing over some general data
    data = {}
    for key in ['active','time','velocity']:
        data[key] = loadData[key]
    # data['active'] = sp.ndimage.binary_closing(data['active'],np.ones(30),border_value=True)
    position = loadData['position']

    data['RW_position'] = loadData['reward_location']
    # print('actives:',data['active'].sum())
    
    ## apply binning
    min_val,max_val = np.nanpercentile(position,(0.1,99.9)) # this could/should be done in data aligning
    environment_length = max_val - min_val

    binpos = np.minimum((position - min_val) / environment_length * nbin,nbin-1).astype('int')
    bin_array = np.linspace(0,nbin,nbin+1)-0.5

    ## define dwelltimes
    data['dwelltime'] = np.histogram(
            binpos[data['active']],
            bin_array
        )[0]/f


    ## define trials
    data['trials'] = {}
    trial_start = np.hstack([0,np.where(np.diff(position)<(-environment_length/2))[0] + 1])

    ## remove partial trials from data (if fraction of length < partial_threshold)
    partial_threshold = 0.6
    if not (binpos[0] < nbin*(1-partial_threshold)):
        # print('remove partial first trial @',trial_start[0])
        data['active'][:max(0,trial_start[0])] = False

    if not (binpos[-1] >= nbin*partial_threshold):
        # print('remove partial last trial @',trial_start[-1])
        data['active'][trial_start[-1]:] = False

    data['nFrames'] = np.count_nonzero(data['active'])

    ### preparing data for active periods, only

    ## defining arrays of active time periods
    data['binpos'] = binpos[data['active']]
    data['time'] = data['time'][data['active']]
    data['velocity'] = data['velocity'][data['active']]

    binpos_coarse = np.minimum((position-min_val) / environment_length * nbin_coarse,nbin_coarse-1).astype('int')
    data['dwelltime_coarse'] = np.histogram(
            binpos_coarse[data['active']],
            np.linspace(0,nbin_coarse,nbin_coarse+1)-0.5
        )[0]/f
    data['binpos_coarse'] = binpos_coarse[data['active']]

    ## define start points
    data['trials']['start'] = np.hstack([0,np.where(np.diff(data['binpos'])<(-nbin/2))[0] + 1,data['active'].sum()])
    data['trials']['start_t'] = data['time'][data['trials']['start'][:-1]]
    data['trials']['ct'] = len(data['trials']['start']) - 1


    ## getting trial-specific behavior data
    data['trials']['dwelltime'] = np.zeros((data['trials']['ct'],nbin))
    data['trials']['nFrames'] = np.zeros(data['trials']['ct'],'int')#.astype('int')

    data['trials']['binpos'] = {}
    for t in range(data['trials']['ct']):
        data['trials']['binpos'][t] = data['binpos'][data['trials']['start'][t]:data['trials']['start'][t+1]]
        data['trials']['dwelltime'][t,:] = np.histogram(data['trials']['binpos'][t],bin_array)[0]/f
        data['trials']['nFrames'][t] = len(data['trials']['binpos'][t])
    
    return data

    # if plot_bool:
    #     plt.figure(dpi=300)
    #     plt.plot(data['time'],data['position'],'r.',markersize=1,markeredgecolor='none')
    #     plt.plot(data['time'][data['active']],data['position'][data['active']],'k.',markersize=2,markeredgecolor='none')
    #     plt.show(block=False)
    # return data