import os, time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import pickle as pkl
import mat73

import scipy.io as sio
from scipy.ndimage import binary_closing, gaussian_filter1d as gauss_filter
from scipy.signal import resample

from .align_helper import *

# def align_mouse(mouse,dataset='AlzheimerMice_Hayashi',ssh_alias=None):
#
    # server_path = "/usr/users/cidbn1/neurodyn"
    # dirs = os.listdir(os.path.join(serverpath,dataset,mouse))
    # dirs.sort()
    # for dir in dirs:
    #     if dir.startswith('Session'):
    #         s = int(dir[-2:])
    #         print('processing Session',s)
    #         align_data(server_path,dataset,mouse,s,ssh_alias)


def align_data_on_hpc(datapath_in,datapath_out,dataset,mouse,session,
                    ssh_alias=None,
                    T=8989,rw_delay=0,min_stretch=0.9,keep_plot=False):
    
    '''
        TODO:
            - in some rare cases, the very last fractional trial is not aligned properly/shows weird behavior
            - maybe put this into a class?
            - automatically detect whether alignment was successful, or contains "squished" data
            - calculate probability of reward delivery, including fractional trials at beginning / end (avoid >1 probs)
    '''

    session_path = os.path.join(datapath_in,dataset,mouse,session)
    data_path, _ = get_file_path(ssh_alias,session_path)

    if ssh_alias:
        # print(data_path)
        path = f"./tmp"

        figure_path = f"{path}/aligned_m={mouse}_s={session[-2:]}.png"
        results_path = f"{path}/aligned_behavior.pkl"

        # figure_path = f"{datapath_out}/{dataset}/{mouse}/behavior_alignment"
        # figure_path += f"/aligned_m={mouse}_s={session[-2:]}.png"

        # results_path = f"{datapath_out}/{dataset}/{mouse}/{session}/aligned_behavior.pkl"

    else:
        # storage_path = f"/scratch/users/{os.environ['USER']}/data/"
        # storage_path = f"/usr/users/cidbn1/placefields/"

        figure_path = f"{datapath_out}/{dataset}/{mouse}/behavior_alignment"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        figure_path += f"/aligned_m={mouse}_s={session[-2:]}.png"

        results_path = f"{datapath_out}/{dataset}/{mouse}/{session}/aligned_behavior.pkl"
    
    align_data(data_path,results_path,figure_path,T=T,
        rw_delay=rw_delay,min_stretch=min_stretch)
    
    if not keep_plot:
        plt.close()

    if ssh_alias:
        cmd = f"scp {figure_path} {results_path} {ssh_alias}:{datapath_out}/{dataset}/{mouse}/{session}/"
        os.system(cmd)

    # return data_resampled

    # return data,data_align,data_resampled,rw_col,rw_loc
    #time.sleep(3)

def align_data(data_path,results_path,figure_path,T=8989,
            rw_delay=0,min_stretch=0.9):

    data, rw_col = load_behavior_data(data_path,speed_gauss_sd=3)

    data_align, rw_loc, rw_prob = align_behavior_data(data,min_stretch)
    data_resampled = resample_behavior_data(data_align,T)

    
    fig,ax = plt.subplots(3,1,sharex=True,figsize=(10,4))

    plot_mouse_location(ax[0],data,rw_loc)
    plt.setp(ax[0],ylabel='location')

    min_val,max_val = np.nanpercentile(data['position'],(0.1,99.9))
    environment_length = max_val - min_val

    

    plot_mouse_location(ax[1],data_resampled,rw_loc)
    plt.setp(ax[1],ylabel='bin (aligned)')

    ax[2].plot(data_resampled['time'],data_resampled['velocity'],'k-',lw=0.5)
    velocity = gauss_filter(np.maximum(0,np.diff(data_resampled['position'],prepend=data_resampled['position'][0])),2)
    ax[2].plot(data_resampled['time'],velocity,'r-',lw=0.5)
    plt.setp(ax[2],ylabel='velocity',xlabel='time [s]')

    rw_loc = (rw_loc-min_val) / environment_length
    rw_loc = round(rw_loc*20)/20    ## round to next 5
    print(f'reward @{rw_loc} with prob {rw_prob}')
    
    data_resampled['reward_location'] = rw_loc
    data_resampled['reward_prob'] = rw_prob

    with open(results_path, "wb") as output_file:
        pkl.dump(data_resampled, output_file)

    # fileName = f"aligned_m={mouse}_s={session[-2:]}__reward_col={rw_col}_loc={rw_loc}.png"

    plt.tight_layout()

    plt.savefig(figure_path,dpi=150)
    plt.show(block=False)


def load_behavior_data(data_path,
                       rw_col=None,mic_col=None,
                       speed_gauss_sd=5):

    """
        this function loads behavioral data either from .txt-, or from .mat-file
        Files are loaded according to file-structure, with notable differences in some mice:
            - mic_col: [None or 8] (default: None)
                the column that contains data on whether recording
                was active or not during frame, sometimes not present
            - rw_col: [None, 3 or 8] (default: None)
                the column that contains data on whether a reward was
                delivered during frame. If 'None', column is inferred from data
    """
    rw_col_candidates = [3,8]

    _,ext = os.path.splitext(data_path)
    if ext=='.txt':

        data_tmp = pd.read_csv(data_path,sep='\t')

        ## in some of the .txt files, column header is f**cked up
        if not ('Microscope' in data_tmp.keys()):
            data_tmp.reset_index(inplace=True)
            data_tmp.drop('level_0',axis=1,inplace=True)
            cols = list(data_tmp.columns[1:-1])
            cols.extend(['Microscope','Licking'])

            col_dict = {}
            for key_ref,key in zip(data_tmp.columns,cols):
                col_dict[key_ref] = key
            data_tmp.rename(columns=col_dict,inplace=True)
        cols = data_tmp.columns

        data_tmp = data_tmp.dropna()    ## remove NaNs

        ## find fitting reward column
        if not rw_col:
            for col in rw_col_candidates:
                looks_like_it = is_rw_col(np.array(data_tmp[cols[col-1]]),np.array(data_tmp['Time']))
                if looks_like_it:
                    rw_col = col
                    break
        if not rw_col:
            raise RuntimeError('No reward column could be found')


        cols = data_tmp.columns

        data = {
            'time': np.array(data_tmp['Time']),
            'velocity': np.array(data_tmp['Speed']),
            'reward': np.array(data_tmp[cols[rw_col-1]])>0.5,
            'frame': np.array(data_tmp['Frame#']).astype('int'),
            'position': np.array(data_tmp['omegaY'])
        }
        if mic_col:
            data['recording'] = binary_closing(data_tmp[cols[mic_col]]<1,np.ones(5))
        else:
            idx_start = np.where(data['frame']==1)[0][0]
            print(data['frame'])
            idx_end = np.where(data['frame']==8989)[0][0]+3 if 8989 in data['frame'] else len(data['frame'])
            data['recording'] = np.zeros_like(data['frame'],'bool')
            data['recording'][idx_start:idx_end+1] = True

    else:
        try:
            data_tmp = sio.loadmat(data_path)
        except:
            print('matlab file version 7.3 detected, loading with mat73...')
            data_tmp = mat73.loadmat(data_path)
        data_tmp = data_tmp['crop_bhdata']

        ## find fitting reward column
        if not rw_col:
            for col in rw_col_candidates:
                looks_like_it = is_rw_col(data_tmp[:,col],data_tmp[:,1])
                if looks_like_it:
                    rw_col = col
                    break
        if not rw_col:
            raise RuntimeError('No reward column could be found')

        data = {
            'time': data_tmp[:,1],
            'velocity': data_tmp[:,2],
            'reward': data_tmp[:,rw_col]>0.5,
            'frame': data_tmp[:,4].astype('int'),
            'position': data_tmp[:,6]
        }
        if mic_col:
            data['recording'] = binary_closing(data_tmp[:,mic_col]<1,np.ones(5))
        else:
            idx_start = np.where(data['frame']==1)[0][0]
            
            if 8989 in data['frame']:
                idx_end = min(np.where(data['frame']==8989)[0][0]+4,np.where(data['frame']==8989)[0][-1])
            else:
                idx_end = len(data['frame'])
            data['recording'] = np.zeros_like(data['frame'],'bool')
            data['recording'][idx_start:idx_end+1] = True

    data['velocity'] = gauss_filter(data['velocity'],speed_gauss_sd)

    return data, rw_col


def align_behavior_data(data,
        min_stretch=0.9,
        rw_delay=0,
        align_tolerance=5,rw_tolerance=5):

    loc_buffer=2
    rw_tmp = binary_closing(data['reward'],np.ones(100))

    ## get positions at which reward is delivered, and where mouse passes through rw location (should be aligned)
    idxs_reward_delivery = np.where(np.diff(rw_tmp.astype('int'))==1)[0]
    
    # print(np.where(np.diff(rw_tmp.astype('int'))==1))
    # low, high = np.percentile(data['position'],(0.1,99.9))
    # pos = data['position'][idxs_reward_delivery]
    # print('rw_loc:',pos,(pos-low)/(high-low))
    
    ## identify location of reward
    # rw_loc_candidates = [-150,0,150]
    rw_loc = np.mean(data['position'][idxs_reward_delivery[:3]])
    # idx_rw_loc = np.argmin(np.abs(rw_loc_candidates - rw_loc))
    # rw_loc = rw_loc_candidates[idx_rw_loc]

    idxs_reward_passthrough = np.where(np.diff((data['position']>rw_loc).astype('int'))==1)[0] + 1
    rw_prob = len(idxs_reward_delivery)/len(idxs_reward_passthrough)    # probability of reward delivery

    loc_aligned = np.zeros_like(data['position'])

    idx_rwd_prev = 0
    idx_rwpt_prev = 0

    # loc_dist = np.max(data['position']) - np.min(data['position']) + loc_buffer

    data['time'] -= data['time'][data['recording']][0]
    data_aligned = data.copy()

    ## instead of assigning each pass to each reward, find proper fitting index
    n_passthrough = 0
    end_reached = False
    for idx_rwd in idxs_reward_delivery:

        ## find fitting passthrough index
        idx_match = False
        while not idx_match:
            if n_passthrough==len(idxs_reward_passthrough):
                end_reached = True
                break
            else:
                idx_rwpt = idxs_reward_passthrough[n_passthrough]
                if (not idx_rwd_prev==0) and (idx_rwpt-idx_rwpt_prev)<min_stretch*(idx_rwd-idx_rwd_prev):
                    n_passthrough += 1
                else:
                    idx_match = True
        if end_reached:
            break

        ### now, find loc of this trial, where mouse passes through rw-zone
        if (abs(idx_rwpt-idx_rwd) > align_tolerance) & \
            ((rw_loc-rw_tolerance > data['position'][idx_rwd]) | (rw_loc+rw_tolerance < data['position'][idx_rwd])):

            ## align location
            loc_aligned[idx_rwd_prev:idx_rwd] = apply_to_stretched_out(
                lambda x: resample(x,idx_rwd-idx_rwd_prev),
                data['position'][idx_rwpt_prev:idx_rwpt],
                loc_buffer=loc_buffer
            )

            idx_rwpt_prev = idx_rwpt
            idx_rwd_prev = idx_rwd
        else:
            ## merely copy over raw data
            if (idx_rwpt_prev==idx_rwd_prev):
                loc_aligned[idx_rwd_prev:idx_rwd] = data['position'][idx_rwpt_prev:idx_rwd]
            else:
                loc_aligned[idx_rwd_prev:idx_rwd] = resample(data['position'][idx_rwpt_prev:idx_rwpt],idx_rwd-idx_rwd_prev)

            idx_rwpt_prev = idx_rwd
            idx_rwd_prev = idx_rwd

    # when last index is reached, attach remaining location data
    dT_end = min(len(data['position'])-idx_rwpt,len(loc_aligned) - idx_rwd_prev)
    loc_aligned[idx_rwd_prev:idx_rwd_prev+dT_end] = data['position'][idx_rwpt_prev:idx_rwpt_prev+dT_end]
    # loc_aligned[idx_rwd_prev+dT_end:] = np.nan

    min_val,max_val = np.nanpercentile(loc_aligned,(0.1,99.9))

    ## remove outliers (sometimes come in through resampling or other weird stuff)
    loc_aligned[loc_aligned<min_val] = min_val
    loc_aligned[loc_aligned>max_val] = max_val

    data_aligned['position'] = loc_aligned

    return data_aligned, rw_loc, rw_prob


def resample_behavior_data(data,
        T=8989):

    """
        Function to resample the data to T frames
        This function also creates binned location
    """


    loc_buffer=2    ## small non-zero value required to distinguish between start and end when wrapping
    
    data = data.copy()
    min_val,max_val = np.nanpercentile(data['position'],(0.1,99.9))

    loc_dist = max_val - min_val + loc_buffer

    for key in ['time','position','velocity','frame','reward']:
        data[key] = data[key][data['recording']]

    pos_tmp = data['position'].copy() - min_val
    trial_idx = np.where(np.diff(pos_tmp)<(-loc_dist/2))[0]
    for idx in trial_idx:
        pos_tmp[idx+1:] = pos_tmp[idx+1:] + loc_dist

    data_resampled = {
        'frame': np.linspace(1,T,T).astype('int'),
        'time': np.zeros(T),
        'position': np.zeros(T),
        'velocity': np.zeros(T),
        'reward': np.zeros(T,dtype='bool'),
        'reward_location': None,
        'reward_prob': None,
    }
    
    fs = np.unique(data['frame'])
    for f in range(T+1):
        if f in fs:
            data_resampled['position'][f-1] = np.median(pos_tmp[data['frame']==f])
            data_resampled['time'][f-1] = np.median(data['time'][data['frame']==f])
            data_resampled['reward'][f-1] = np.any(data['reward'][data['frame']==f])
            data_resampled['velocity'][f-1] = np.mean(data['velocity'][data['frame']==f])
        else:
            ## sometimes, single frames are not covered
            ## in this case, merely copy values from last frame
            data_resampled['position'][f-1] = np.median(pos_tmp[data['frame']==f-1])
            data_resampled['time'][f-1] = np.median(data['time'][data['frame']==f-1])
            data_resampled['reward'][f-1] = np.any(data['reward'][data['frame']==f-1])
            data_resampled['velocity'][f-1] = np.mean(data['velocity'][data['frame']==f-1])

    data_resampled['position'] = np.mod(data_resampled['position']+loc_buffer/2,loc_dist)
    # data_resampled['bin_position'] = (data_resampled['position'] / (max_val - min_val) * nbins).astype('int')

    data_resampled['position'] += min_val


    return data_resampled


def plot_mouse_location(ax,data,rw_loc=0):

    # loc = data['bin_position'] if 'bin_position' in data.keys() else data['position']
    loc = data['position']
    min_val = np.nanmin(loc)
    max_val = np.nanmax(loc)

    loc_dist = max_val - min_val

    # identify location of trial-starts
    trial_idx = np.where(np.diff(loc)<(-loc_dist/2))[0]
    for idx in trial_idx:
        ax.axvline(data['time'][idx],color='b',lw=0.5)

    ax.axhline(rw_loc,color='k',ls='--',lw=0.5)
    ax.plot(data['time'],loc,'k.',ms=0.5,label='active')
    # idxs_reward_delivery = np.where(np.diff(data['reward'].astype('int'))==1)[0]
    # ax.plot(data['time'][idxs_reward_delivery],data['position'][idxs_reward_delivery],'b.',ms=5)

    if 'active' in data.keys():
        ax.plot(data['time'][~data['active']],loc[~data['active']],'r.',ms=.5,label='inactive')
    ax.plot(data['time'][data['reward']],loc[data['reward']],'b.',ms=5,label='reward')
