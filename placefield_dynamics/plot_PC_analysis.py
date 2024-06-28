from multiprocessing import get_context

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, rc
# from matplotlib.cm import get_cmap
from matplotlib_scalebar.scalebar import ScaleBar
# from matplotlib.patches import Arc
# from matplotlib.ticker import AutoLocator, MultipleLocator, AutoMinorLocator, LogLocator, ScalarFormatter, MaxNLocator, NullFormatter
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import matplotlib.image as mpimg
# from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
# import scipy.ndimage as spim
import scipy.stats as sstats

from scipy.io import loadmat
# from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
# from scipy.cluster import hierarchy

from collections import Counter
from tqdm import *
import os, time, cv2, itertools, sys
# import multiprocessing as mp
import warnings

from caiman.utils.utils import load_dict_from_hdf5

from .utils import periodic_distr_distance, bootstrap_data, gauss_smooth, get_reliability, com, get_status_arr, get_CI, add_number

# get_firingrate, compute_serial_matrix, 
from .utils import pickleData
# from .utils_analysis import get_performance, define_active

# sys.path.append('/home/wollex/Data/Science/PhD/Programs/PC_modelling/')
# from model_turnover import *

warnings.filterwarnings("ignore")

def plot_PC_analysis(cluster,plot_arr=[0,1],N_bs=10,n_processes=0,reprocess=False,sv=False,sv_ext='png',PC=None,active=None):#,N_bs,s_offset,sv,sv_suffix,sv_ext,arrays,occupancy,ROI_recurr,N_pairs,N_norm,pop_overlap)#,ROI_rec2,ROI_tot2)#pathBase,mouse)

    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    # cluster.para = set_para(*os.path.split(cluster.para['pathMouse']),1)

    nSes = cluster.data['nSes']
    nC = cluster.data['nC']

    nbin = cluster.data['nbin']
    t_ses = np.linspace(1,nSes,nSes)

    L_track = 100
    ### think about: doing all the plots in an UI, able to change parameters such as gate & reward location, sessions considered, etc
    ### -> which plots are needed, which ones are not that important?


    #plt.figure()
    #plt.plot(F_shifts(np.linspace(-50,50,101),0.2,0.4,10,0))
    #plt.show(block=False)

    #exp_dist = @(a,x_data) a(2)*exp(-x_data/a(1));
    #log_exp_dist = @(a,x_data) log(a(2)) - x_data/a(1);

    ### ----------------- plotting options ---------------- ###
    pl_dat = plot_dat(cluster.data['mouse'],cluster.paths['figures'],nSes,cluster.data,sv_ext=sv_ext)

    plot_fig = np.zeros(1000).astype('bool')

    for p in plot_arr:
        plot_fig[p] = True

    ## 0:     basic cluster stats (sessions active / cluster, cluster score)
    ## 1:     neuron numbers (active, silent, PCs (NRNG,GT,RW))
    ## 2:     (long) population drift / neuron tracking accuracy
    ## 3:     number of fields
    ## 4:     PC coverage (old:9)
    ## 5:     PC drift, displayed in sessions
    ## 6:     stability analysis
    ## 7:     ?? choosing criterion
    ## 8:     ?? example draw stuff
    ## 9:     matching score statistics
    ## 10:    session alignment
    ## 11:    ROI positions
    ## 12:    neuron distances
    ## 13:    3d-neuron movement
    ## 14:    matching performance vs manual (only works for m879, m884)


    nSes_real = cluster.status['sessions'].sum()
    # print(np.where(cluster.status['sessions'])[0])

    t_start = time.time()

#### ------------ get information about populations ------------- ###

  #pop_thr = 3;
  #nPop = sum(nOcc>=pop_thr);

  #idx_pure = sum(nOcc(:,3:5)>=pop_thr,2)==1;
  #idx_none = sum(nOcc(:,3:5)>=pop_thr,2)==0;
  #idx_mixed = sum(nOcc(:,3:5)>=pop_thr,2)>1;

  #idx_pop = nOcc>=pop_thr;
##    idx_pop_pure = idx_pop & idx_pure;

  #nPC_only = idx_none & idx_pop(:,2);
  #NRNG_only = idx_pure & idx_pop(:,3);
  #GT_only = idx_pure & idx_pop(:,4);
  #RW_only = idx_pure & idx_pop(:,5);

  #nPop_nPC = sum(nPC_only);
  #nPop_GT = sum(GT_only);
  #nPop_RW = sum(RW_only);
  #nPop_NRNG = sum(NRNG_only);
  #nPop_multi = sum(idx_mixed);

  #pTotal = sum(nOcc) / (nC*nSes)






    if plot_fig[32]:

        sig_theta = cluster.stability['all']['mean'][0,2]

        s_range = 10
        ds = 1
        s1_shifts,s2_shifts,f1,f2 = np.unravel_index(cluster.compare['pointer'].col,(nSes,nSes,cluster.params['field_count_max'],cluster.params['field_count_max']))
        c_shifts = cluster.compare['pointer'].row

        Ds = s2_shifts-s1_shifts
        idx_ds = np.where(Ds==ds)

        N_data = len(idx_ds)

        idx_shifts = cluster.compare['pointer'].data[idx_ds].astype('int')-1
        shifts_ds = cluster.compare['shifts'][idx_shifts]
        s1_shifts_ds = s1_shifts[idx_ds]
        s2_shifts_ds = s2_shifts[idx_ds]
        c_shifts_ds = c_shifts[idx_ds]
        f1_ds = f1[idx_ds]
        stab_loc = np.zeros((50,2))
        print(np.abs(shifts_ds))
        for i in range(50):
            idx = ((np.abs(shifts_ds)>i) & (np.abs(shifts_ds) < (i+1)))
            print('------ i = %d: %d -----'%(i,idx.sum()))
            # loc_ref = cluster.fields['location'][c_shifts_ds[idx],s1_shifts_ds[idx],f1_ds[idx],0]

            for j in np.where(idx)[0]:
                s = s1_shifts_ds[j]
                c = c_shifts_ds[j]
                idx_c_shifts = (s1_shifts==s) & ((s+1)<s2_shifts) & (s2_shifts<(s+s_range)) & (c_shifts==c)
                idx_shifts = cluster.compare['pointer'].data[idx_c_shifts].astype('int')-1
                c_shifts_now = cluster.compare['shifts'][idx_shifts]

                stab_loc[i,0] += (np.abs(c_shifts_now) < sig_theta).sum()
                stab_loc[i,1] += len(c_shifts_now)
                # print(c,s)
                # print(shifts_ds)

                # loc = cluster.fields['location'][c_shifts[j],s2_shifts[j]+1:s2_shifts[j]+s_range,:]
        # print(stab_loc)
        plt.figure()
        plt.plot(stab_loc[:,0]/stab_loc[:,1])
        plt.show(block=False)




    if plot_fig[50]:
        print('### plot behavior stuff ###')

        f = 50
        rw_delay = 1

        for s in range(15,16):
        # s = 12

            ### load data from text-file
            # find txt file:
            pathSession = cluster.paths['sessions'][s+1]
            for file in os.listdir(pathSession):
              if file.endswith(".txt"):
                  pathBH = os.path.join(pathSession, file)

            bh = np.loadtxt(pathBH,skiprows=1)

            ### crop data to actual recording
            # print(bh[-1000:,3])
            start_idx = np.where(bh[:,3]==1)[0][0]
            end_idx = np.where(bh[:,3]>=8989)[0][0]+4
            # print(start_idx,end_idx)

            pos = bh[start_idx:end_idx,5]
            pos -= pos.min()
            pos *= 100/pos.max()

            idx_teleport = np.where(np.diff(pos)<-10)[0]+1
            trial_start = np.hstack([idx_teleport])
            ntrials = len(trial_start) -1
            times = bh[start_idx:end_idx,0]
            times -= times[0]

            reward = bh[start_idx:end_idx,2]>0.5
            lick = bh[start_idx:end_idx,7]>0.5
            # print(bh[start_idx:end_idx,7])

            time_reward = times[reward]
            time_lick = times[lick]

            plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
            plt.subplot(121)
            plt.plot(times,pos,'k-')
            plt.plot(times[reward],pos[reward],'go',markersize=2)
            plt.plot(times[lick],pos[lick],'ro',markersize=2)
            plt.title('session %d'%s)

            rw_reception = np.zeros(ntrials,'bool')
            rw_lick = np.zeros(ntrials,'bool')
            for t in range(ntrials):
                t_start = trial_start[t]
                t_end = trial_start[t+1]
                t_pos = pos[t_start:t_end]

                t_RW = reward[t_start:t_end]
                t_lick = lick[t_start:t_end]
                t_time = times[t_start:t_end]
                if np.any(t_RW):
                    enter_RW = np.where(t_RW)[0][0]-rw_delay*f
                    plt.plot(t_time[enter_RW],t_pos[enter_RW],'yo',markersize=2)

                    rw_reception[t] = True

                    for idx in np.where(t_RW)[0]:
                        if np.any(t_lick[idx:idx+2*f]):
                            rw_lick[t] = True

            plt.subplot(122)
            plt.hist(np.diff(np.where(lick)[0]),np.linspace(0,200,101))
            plt.xlabel('inter-lick-interval')

            plt.show(block=False)

            if sv:
                pl_dat.save_fig('lick_Example')



    if plot_fig[51]:

        print('### identification of a steady state ###')
        # rw_ct = cluster.sessions['give_reward'][cluster.status['sessions']].sum(1)
        # rw2_ct = cluster.sessions['got_reward'][cluster.status['sessions']].sum(1)
        # PC_ct = cluster.status['activity'][:,cluster.status['sessions'],5].sum(0)
        # RW_ct = cluster.status['activity'][:,cluster.status['sessions'],4].sum(0)
        # t_act = cluster.sessions['time_active'][cluster.status['sessions']]

        # rw2_ct = cluster.sessions['got_reward'].sum(1)
        PC_ct = cluster.status['activity'][...,2].sum(0)
        RW_ct = cluster.status['activity'][...,4].sum(0)
        t_act = cluster.behavior['time_active']

        s_bool = np.zeros(nSes,'bool')
        s_bool[17:87] = True
        # s_bool[:] = True
        s_bool[~cluster.status['sessions']] = False
        s=52

        pathSession = cluster.paths['sessions'][s]
        data = define_active(pathSession)
        dataPerf = get_performance(cluster.params['pathMouse'],[s],cluster.session_data['RW_pos'][s,:],cluster.session_data['delay'][s],plot_bool=False)

        RW = np.zeros((nSes,2))*np.NaN
        for s in np.where(cluster.status['sessions'])[0]:
            RW[s,0] = cluster.performance[s]['trials']['RW_reception'].sum()
            RW[s,1] = RW[s,0]/cluster.behavior['trial_ct'][s]

            # t_active = data['active'].sum()/15
            # print(t_active,t_act[s])

        fig = plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])

        ax = plt.axes([0.1,0.7,0.325,0.235])
        ax.plot(data['time'],data['position']*1.2,'k.',markersize=0.8,markeredgecolor='none')#linewidth=0.5)
        ax.plot(data['time'][dataPerf[0]['trials']['RW_frame']],1.2*data['position'][dataPerf[0]['trials']['RW_frame']],'ro',markersize=1.5)
        ax.plot([0,600],[60,60],'--',color='tab:red')
        # ax.plot(times[lick],pos[lick],'ro',markersize=1)
        pl_dat.remove_frame(ax,['top','right'])
        ax.set_xticks([])
        ax.yaxis.set_label_coords(-0.175,0.6)
        ax.set_ylabel('position [cm]',fontsize=10)
        pl_dat.add_number(fig,ax,order=1)
        # print('plot running trace in here')
        ax = plt.axes([0.1,0.55,0.325,0.15])
        ax.plot(data['time'],data['velocity'],'k-',linewidth=0.5)
        ax.bar(data['time'],data['active']*2,width=np.diff(data['time'])[0],color='tab:blue')
        pl_dat.remove_frame(ax,['top','right'])
        ax.set_xlabel('time [s]')
        ax.set_ylabel('velocity $[\\frac{cm}{s}]$',fontsize=10)
        ax.yaxis.set_label_coords(-0.175,0.4)
        # ax.plot(data['time'],pos[reward],'ro',markersize=2)

        s_arr = np.arange(nSes)

        ax = plt.axes([0.625,0.775,0.275,0.16])
        pl_dat.add_number(fig,ax,order=2)
        ax.set_ylim([0,400])
        ax.plot(s_arr[cluster.status['sessions']],cluster.sessions['time_active'][cluster.status['sessions']],color='k',alpha=0.3)
        ax.plot(s_arr[s_bool],cluster.sessions['time_active'][s_bool],color='k')
        ax.set_xticklabels([])
        tw = [0,1,1.5,2]
        try:
            for j,s in enumerate([0,11,31,53]):
                ax.annotate(xy=(s,cluster.sessions['time_active'][s]+75),xytext=(s,1.2*(cluster.sessions['time_active'][s]+125)),s='$%1.2g$'%tw[j],ha='center',arrowprops=dict(arrowstyle='->'),fontsize=6)

            for j,s in enumerate([87,97]):
                ax.annotate(xy=(s,cluster.sessions['time_active'][s]+75),xytext=(s,1.2*(cluster.sessions['time_active'][s]+125)),s='rw',ha='center',arrowprops=dict(arrowstyle='->'),fontsize=6)
        except:
            pass
        # ax.annotate(xy=(31,300),xytext=(31,420),s='$t_w = 1$',ha='center',arrowprops=dict(arrowstyle='->'),fontsize=8)
        # ax.annotate(x=31,y=300,s='$t_w = 1$',arrowprops=dict(arrowstyle='->'))
        pl_dat.remove_frame(ax,['top'])
        ax.set_ylabel('$t_{active}$ [s]',ha='center')
        ax.yaxis.set_label_coords(-0.2,0.5)
        ax2 = ax.twinx()
        ax2.plot(s_arr[cluster.status['sessions']],RW[cluster.status['sessions'],0],color='tab:red',alpha=0.3)
        ax2.plot(s_arr[s_bool],RW[s_bool,0],color='tab:red')
        # ax2.plot(s_arr[cluster.status['sessions']],cluster.sessions['got_reward'][cluster.status['sessions']].sum(1),color='tab:red',alpha=0.3)
        # ax2.plot(s_arr[s_bool],cluster.sessions['got_reward'][s_bool].sum(1),color='tab:red')
        ax2.set_ylim([0,40])
        pl_dat.remove_frame(ax2,['top'])
        ax2.set_xticklabels([])
        ax2.set_ylabel('# rewards')

        ax = plt.axes([0.625,0.55,0.275,0.16])
        ax.plot(s_arr[cluster.status['sessions']],cluster.sessions['speed'][cluster.status['sessions']],color='k',alpha=0.3)
        ax.plot(s_arr[s_bool],cluster.sessions['speed'][s_bool],color='k')
        ax.set_ylim([0,20])
        pl_dat.remove_frame(ax,['top','right'])
        ax.set_xticklabels([])
        ax.set_ylabel('velocity',ha='center')
        ax.yaxis.set_label_coords(-0.2,0.5)

        # var = RW[:,1]
        var = t_act
        ax = plt.axes([0.625,0.325,0.275,0.16])
        PC_rate = PC_ct/var
        RW_rate = RW_ct/var
        ax.plot(s_arr[cluster.status['sessions']],PC_rate[cluster.status['sessions']],'tab:blue',alpha=0.3)
        # ax.plot(s_arr[cluster.status['sessions']],RW_rate[cluster.status['sessions']],'tab:red',alpha=0.3)
        ax.plot(s_arr[s_bool],PC_rate[s_bool],'tab:blue')
        # ax.plot(s_arr[s_bool],RW_rate[s_bool],'tab:red')
        pl_dat.remove_frame(ax,['top'])
        # ax.set_xlabel('session')
        ax.set_ylabel('$\\frac{\# PC}{t_{active}}$',ha='center')
        ax.set_ylim([0,3])
        ax.set_xticklabels([])
        ax.yaxis.set_label_coords(-0.2,0.5)

        ax2 = ax.twinx()
        ax2.plot(s_arr[cluster.status['sessions']],PC_ct[cluster.status['sessions']],'.',color='k',markersize=1,alpha=0.3)
        # ax2.plot(s_arr[cluster.status['sessions']],RW_ct[cluster.status['sessions']],'.',color='tab:red',markersize=1,alpha=0.3)
        ax2.plot(s_arr[s_bool],PC_ct[s_bool],'.',color='k',markersize=1)
        # ax2.plot(s_arr[s_bool],RW_ct[s_bool],'.',color='tab:red',markersize=1)
        pl_dat.remove_frame(ax2,['top'])
        ax2.set_ylabel('# PC')
        ax2.set_ylim([0,500])
        ax.set_xticklabels([])

        # plot firing rates
        nu = np.zeros((nSes,3,3))*np.NaN
        MI = np.zeros((nSes,3,3))*np.NaN
        rel = np.zeros((nSes,2,3))*np.NaN
        pval = np.zeros(nSes)*np.NaN
        nu_tmp = []
        # nu_RW_tmp = []
        MI_tmp = []
        for s in np.where(cluster.status['sessions'])[0]:
            nu_nPC = cluster.stats['firingrate'][cluster.status['activity'][:,s,1]&(~cluster.status['activity'][:,s,2]),s]

            idx_PC = np.where(cluster.status_fields[:,s,:])
            nu_PC = cluster.fields['baseline'][idx_PC[0],s,idx_PC[1],0]
            nu_PC = cluster.fields['amplitude'][idx_PC[0],s,idx_PC[1],0]/cluster.fields['baseline'][idx_PC[0],s,idx_PC[1],0]
            # nu_RW = cluster.stats['oof_firingrate_adapt'][cluster.status['activity'][:,s,4],s]

            # nu_if_PC = cluster.stats['if_firingrate_adapt'][cluster.status['activity'][:,s,5],s]
            # nu_if_RW = cluster.stats['if_firingrate_adapt'][cluster.status['activity'][:,s,4],s]

            # idx_PC = np.where(cluster.fields['status'][:,s,:]>=3)
            # idx_RW = np.where(cluster.fields['status'][:,s,:]==4)

            rel_PC = cluster.fields['reliability'][idx_PC[0],s,idx_PC[1]]
            # rel_RW = cluster.fields['reliability'][idx_RW[0],s,idx_RW[1]]

            if len(nu_PC)>0:
                nu[s,0,0] = nu_nPC.mean()
                nu[s,0,1:] = np.percentile(nu_nPC,[15.8,84.1])#nu_nPC.std()

                nu[s,1,0] = nu_PC.mean()
                nu[s,1,1:] = np.percentile(nu_PC,[15.8,84.1])#nu_PC.std()

                rel[s,0,0] = rel_PC.mean()
                rel[s,0,1:] = np.percentile(rel_PC,[15.8,84.1])#nu_nPC.std()

                # if len(nu_RW)>0:
                    # nu[s,2,0] = nu_RW.mean()
                    # nu[s,2,1:] = np.percentile(nu_RW,[15.8,84.1])#nu_RW.std()
                    # rel[s,1,0] = rel_RW.mean()
                    # rel[s,1,1:] = np.percentile(rel_RW,[15.8,84.1])#nu_PC.std()
                    # nu[s,2,1:] = np.percentile(nu_RW,[15.8,84.1])#nu_RW.std()


                res = sstats.ttest_ind(nu_PC,nu_nPC)
                pval[s] = res.pvalue

                MI_PC = cluster.stats['MI_value'][cluster.status['activity'][:,s,2],s]
                MI_nPC = cluster.stats['MI_value'][cluster.status['activity'][:,s,1]&(~cluster.status['activity'][:,s,2]),s]

                if np.any(np.isnan(nu_PC)) | (len(nu_PC)<=1):
                    print('beedoo beeddooooo')
                    print('session %d not enough data'%s)
                elif s_bool[s]:
                    nu_tmp.append(nu_PC)
                    # nu_RW_tmp.append(nu_RW)
                    MI_tmp.append(MI_PC)
                MI[s,0,0] = MI_nPC.mean()
                MI[s,0,1] = MI_nPC.std()
                MI[s,1,0] = MI_PC.mean()
                MI[s,1,1] = MI_PC.mean()
        res = sstats.f_oneway(*nu_tmp)
        # res = sstats.kruskal(*nu_tmp)
        print('firing rate PC')
        print(res)
        # res = sstats.f_oneway(*nu_RW_tmp)
        # res = sstats.kruskal(*nu_RW_tmp)
        # print('firing rate RW')
        # print(res)
        # res = sstats.kruskal(*MI_tmp)
        # print('MI')
        # print(res)

        ax = plt.axes([0.625,0.1,0.275,0.16])
        # ax.plot(s_arr[cluster.status['sessions']],nu[cluster.status['sessions'],2,0],'tab:red',alpha=0.3)
        # ax.plot(s_arr[cluster.status['sessions']],nu[cluster.status['sessions'],0,0],'k',alpha=0.3)
        ax.plot(s_arr[cluster.status['sessions']],nu[cluster.status['sessions'],1,0],'tab:blue',alpha=0.3)
        # pl_dat.plot_with_confidence(ax,s_arr[s_bool],nu[s_bool,0,0],nu[s_bool,0,1:].T,col='k',label='nPC')
        pl_dat.plot_with_confidence(ax,s_arr[s_bool],nu[s_bool,1,0],nu[s_bool,1,1:].T,col='tab:blue',label='PC')
        # pl_dat.plot_with_confidence(ax,s_arr[s_bool],nu[s_bool,2,0],nu[s_bool,2,1:].T,col='tab:red',label='RW')
        pl_dat.remove_frame(ax,['top','right'])
        ax.set_ylabel('$\\nu^+ / \\nu^-$',ha='center')
        ax.yaxis.set_label_coords(-0.2,0.5)
        # ax.set_ylim([0.,0.15])
        ax.set_xlabel('session')
        ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.2,1.1])

        X = np.vstack([var[s_bool],np.ones(len(var[s_bool]))]).T
        res = np.linalg.lstsq(X,PC_ct[s_bool])[0]
        arr = np.linspace(var[s_bool].min(),var[s_bool].max(),2)
        # res2 = np.linalg.lstsq(X,RW_ct[s_bool])[0]
        # arr2 = np.linspace(rw2_ct[s_bool].min(),rw2_ct[s_bool].max(),2)
        arr2 = np.linspace(var[s_bool].min(),var[s_bool].max(),2)


        ax_speed1 = plt.axes([0.1,0.25,0.3,0.1])
        ax_speed2 = plt.axes([0.1,0.1,0.3,0.1])
        pl_dat.add_number(fig,ax_speed1,order=3)

        get_performance(cluster.para['pathMouse'],[15],rw_delay=0,plot_ax=ax_speed1)
        get_performance(cluster.para['pathMouse'],[53],rw_delay=1.5,plot_ax=ax_speed2)

        for axx in [ax_speed1,ax_speed2]:
            pl_dat.remove_frame(axx,['top','right'])
            axx.set_ylabel('v [$\\frac{cm}{s}$]')
            axx.set_xlim([0,100])
            axx.set_ylim([0,22])

        ax_speed2.set_xlabel('position')





        # ax = plt.axes([0.1,0.1,0.3,0.3])
        # pl_dat.add_number(fig,ax,order=3)
        # PC_fit = res[1]+arr*res[0]
        # RW_fit = res2[1]+arr2*res2[0]
        # # PC_corr = np.corrcoef(rw2_ct[s_bool],PC_ct[s_bool])[0,1]
        # # RW_corr = np.corrcoef(rw2_ct[s_bool],RW_ct[s_bool])[0,1]
        # # print(PC_corr)
        # # print(RW_corr)
        # PC_corr = np.corrcoef(var[s_bool],PC_ct[s_bool])[0,1]
        # RW_corr = np.corrcoef(var[s_bool],RW_ct[s_bool])[0,1]
        # ax.plot(arr,PC_fit,'k-')
        # ax.plot(arr2,RW_fit,'k-')
        # ax.plot(var[cluster.status['sessions']],PC_ct[cluster.status['sessions']],'o',color='tab:blue',markersize=3,alpha=0.3,markeredgecolor='None')
        # ax.plot(var[cluster.status['sessions']],RW_ct[cluster.status['sessions']],'o',color='tab:red',markersize=3,alpha=0.3,markeredgecolor='None')
        # ax.plot(var[s_bool],PC_ct[s_bool],'o',color='tab:blue',markersize=3,markeredgecolor='None')
        # ax.plot(var[s_bool],RW_ct[s_bool],'o',color='tab:red',markersize=3,markeredgecolor='None')
        # ax.text(arr2[-1]+10,PC_fit[-1]+10,'%.2g'%PC_corr,fontsize=8)
        # ax.text(arr2[-1]+10,RW_fit[-1]+10,'%.2g'%RW_corr,fontsize=8)
        # pl_dat.remove_frame(ax,['top','right'])
        # ax.set_xlabel('$t_{active}$ [s]')
        # ax.set_ylabel('# PC')

        plt.tight_layout()
        plt.show(block=False)

        # s_bool = np.copy(cluster.status['sessions'])
        # s_bool[88:] = False
        # PC_ct = PC_ct[s_bool]
        # RW_ct = RW_ct[s_bool]
        # rw2_ct = rw2_ct[s_bool]
        # t_act = t_act[s_bool]
        # print(np.cov(rw2_ct,PC_ct))
        # nsteps = 80
        # corr = {'rw_PC':np.zeros(nsteps),
        #         'rw_RW':np.zeros(nsteps),
        #         't_PC':np.zeros(nsteps),
        #         't_RW':np.zeros(nsteps)}
        # for ds in range(nsteps):
        #     corr['rw_PC'][ds] = np.corrcoef(rw2_ct[ds:],PC_ct[ds:])[0,1]
        #     corr['rw_RW'][ds] = np.corrcoef(rw2_ct[ds:],RW_ct[ds:])[0,1]
        #     corr['t_PC'][ds] = np.corrcoef(t_act[ds:],PC_ct[ds:])[0,1]
        #     corr['t_RW'][ds] = np.corrcoef(t_act[ds:],RW_ct[ds:])[0,1]
        # # print(np.corrcoef(rw_ct,PC_ct))
        # print(np.corrcoef(rw2_ct,PC_ct))
        # # print(np.corrcoef(rw_ct,RW_ct))
        # print(np.corrcoef(rw2_ct,RW_ct))
        # print(np.corrcoef(t_act,PC_ct))
        # print(np.corrcoef(t_act,RW_ct))
        # # print(np.corrcoef(rw_ct,PC_ct))
        # plt.figure()
        # plt.plot(corr['rw_PC'],'k-')
        # plt.plot(corr['rw_RW'],'r-')
        # plt.plot(corr['t_PC'],'k--')
        # plt.plot(corr['t_RW'],'r--')
        # plt.ylim([0,1])
        # plt.show(block=False)

        if sv:
            pl_dat.save_fig('steady_state')
        # ds = 1
        # session_bool = np.pad(cluster.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(cluster.status['sessions'][:],(0,0),constant_values=False)
        # for s in np.where(session_bool)[0]:
        #
        #     loc_s1 = cluster.fields['location'][cluster.status['activity'][:,s,2],s,:,0]
        #     loc_s2 = cluster.fields['location'][cluster.status['activity'][:,s,2],s+ds,:,0]
        #
        #     print('find number of stable place fields (RW/others)')
        #     print('find rates of recruitment / stabilization -> saturation somewhere?')
        #

    if plot_fig[52]:


        plt.figure()

        loc_vel = np.zeros((nSes,nbin,2))*np.NaN
        for s in range(nSes):
            try:
                pathSession = os.path.join(cluster.params['pathMouse'],'Session%02d'%(s+1))
                for file in os.listdir(pathSession):
                  if file.endswith(".txt"):
                      pathBH = os.path.join(pathSession, file)

                bh = np.loadtxt(pathBH,skiprows=1)

                ### crop data to actual recording
                start_idx = np.where(bh[:,3]==1)[0][0]
                end_idx = np.where(bh[:,3]>=8989)[0][0]+4

                pos = bh[start_idx:end_idx,5]
                pos -= pos.min()
                pos *= 100/pos.max()

                velocity = np.diff(np.append(pos[0],pos))*50
                velocity[velocity<0] = 0
                velocity = sp.ndimage.gaussian_filter(velocity,10)
                active = sp.ndimage.binary_opening(velocity>2,structure=np.ones(int(15/2)))
                active = sp.ndimage.binary_closing(velocity>2,structure=np.ones(int(15/2)))

                for i in range(nbin-1):
                    loc_vel[s,i,0] = velocity[(pos>i) & (pos<(i+1))].mean()
                    loc_vel[s,i,1] = velocity[(pos>i) & (pos<(i+1)) & active].mean()

                if s<9:
                    plt.subplot(3,3,s+1)
                    plt.plot(loc_vel[s,:,1])
                    plt.ylim([0,20])
            except:
                pass
        plt.show(block=False)

        plt.figure()
        plt.subplot(121)
        plt.imshow(loc_vel[...,0],clim=[5,15])
        plt.subplot(122)
        plt.imshow(loc_vel[...,1],clim=[5,15])
        plt.show(block=False)

            # return


    if plot_fig[40]:

        print('### plot within session correlated pairs ###')

        print('neurons whos place field is activated conjointly')
        print('kinda difficult - cant find anything obvious on first sight')

        plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        ## find trials

        cluster.fields['trial_act']

        high_corr = np.zeros(nC)
        for i,s in enumerate(range(0,20),1):
            idx = np.where(cluster.status_fields)

            idx_s = idx[1] == s
            c = idx[0][idx_s]
            f = idx[2][idx_s]
            trials = cluster.fields['trial_act'][c,s,f,:cluster.behavior['trial_ct'][s]]

            trial_corr = np.corrcoef(trials)
            trial_corr[np.tril_indices_from(trial_corr)] = np.NaN

            # idx[0][idx_s]# np.fill_diagonal(trial_corr,np.NaN)
            # print(trial_corr)
            # print(trial_corr.shape)
            idx_high_corr = np.where(trial_corr > 0.5)
            # print(cluster.stats.keys())
            for c1,c2 in zip(idx_high_corr[0],idx_high_corr[1]):
                # print('c: %d, %d'%(c1,c2))
                # print(cluster.matching['com'][c1,s,:])
                # print(cluster.matching['com'][c2,s,:])
                # print(np.linalg.norm(cluster.matching['com'][c1,s,:]-cluster.matching['com'][c2,s,:]))
                # if np.linalg.norm(cluster.matching['com'][c1,s,:]-cluster.matching['com'][c2,s,:])>10:
                high_corr[c1] += 1
                high_corr[c2] += 1

            # plt.subplot(5,4,i)
            # plt.hist(trial_corr.flat,np.linspace(-1,1,51))
        high_corr[high_corr==0] = np.NaN
        plt.hist(high_corr,np.linspace(0,400,51))
        plt.show(block=False)



    if plot_fig[41]:

        print('### find neurons which are conjointly activated / coding ###')
        print('also, cant find anything super telling right now...')
        # print(cluster.status)
        # status_act = cluster.status['activity'][cluster.status['clusters'],:,1]
        # status_act = cluster.stats['act_stability_temp'][cluster.status['clusters'],:,1]
        status_act = cluster.stats['field_stability_temp'][cluster.status['clusters'],:]
        # print(status_act)
        # print(status_act.shape)
        status_act = gauss_smooth(status_act[:,cluster.status['sessions']],(0,1))
        status_act[np.isnan(status_act)] = 0
        # print(status_act[0,:])
        # print(status_act[1,:])
        act_corr = np.corrcoef(status_act)
        act_corr[np.tril_indices_from(act_corr)] = np.NaN
        # print(act_corr)
        # return
        status_code = cluster.status['activity'][cluster.status['clusters'],:,2]
        status_code = status_code[:,cluster.status['sessions']]

        code_corr = np.corrcoef(status_code)
        code_corr[np.tril_indices_from(code_corr)] = np.NaN

        # plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        # plt.hist(act_corr.flat,np.linspace(-1,1,101),alpha=0.5,color='k')
        # plt.hist(code_corr.flat,np.linspace(-1,1,101),alpha=0.5,color='tab:blue')
        # plt.show(block=False)

        idx = np.where(code_corr>0.9)

        c_IDs = np.where(cluster.status['clusters'])[0]
        s_IDs = np.where(cluster.status['sessions'])[0]

        # plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        # c1 = np.unique(c_IDs[idx[0]])
        # c2 = np.unique(c_IDs[idx[1]])
        #
        # print(len(c1))
        # plt.hist(cluster.fields['location'][c1,...,0].flat,np.linspace(0,100,101))
        # plt.show(block=False)

        idx_c = np.where(cluster.status['clusters'])[0]

        # status_corr = code_corr
        status_corr = act_corr

        idx_out = (~np.any(status_corr>0.2,1)) | (status_code.sum(1)<5)
        status_corr = status_corr[~idx_out,:]
        status_corr = status_corr[:,~idx_out]

        idx_c = idx_c[~idx_out]
        status_corr[np.isnan(status_corr)] = -9
        # print(status_corr.shape)



        ordered_corr,res_order,res_linkage = compute_serial_matrix(1-status_corr,'average')
        cluster_idx = sp.cluster.hierarchy.cut_tree(res_linkage,height=0.5)

        # Z = hierarchy.linkage(1-code_corr,method='average')

        plt.figure()
        # plt.pcolormesh(code_corr[res_order,:][:,res_order],cmap='jet')
        plt.pcolormesh(status_corr[res_order,:][:,res_order],cmap='jet')
        plt.clim([0,1])
        plt.colorbar()
        plt.show(block=False)

        nClusters = len(np.unique(cluster_idx))
        neuron_clusters = {}
        neuron_clusters['N'] = np.zeros(nClusters,'int')
        for c in np.unique(cluster_idx):
            idxes  = idx_c[np.where(cluster_idx==c)[0]]
            neuron_clusters['N'][c] = len(idxes)

            neuron_clusters[c] = {}
            neuron_clusters[c]['members'] = idxes
            neuron_clusters[c]['overlap'] = np.zeros((neuron_clusters['N'][c],neuron_clusters['N'][c],2))

            status_test1 = cluster.status[idxes,:,1]
            status_test2 = cluster.status[idxes,:,2]
            for i,n in enumerate(idxes):
                neuron_clusters[c]['overlap'][i,:,0] = status_test1[:,cluster.status[n,:,1]].sum(1)
                neuron_clusters[c]['overlap'][i,:,1] = status_test2[:,cluster.status[n,:,2]].sum(1)

        for c in np.where(neuron_clusters['N']>10)[0]:
            plt.figure(figsize=(7,5),dpi=300)
            for i,n in enumerate(neuron_clusters[c]['members']):
                ax = plt.axes([0.1+0.075*i,0.1,0.05,0.9])
                firingmap = cluster.stats['firingmap'][n,cluster.status['sessions'],:]
                firingmap[~cluster.status[n,cluster.status['sessions'],1],:] = np.NaN
                firingmap = gauss_smooth(firingmap,(0,2))
                firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
                im = ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,5])
                ax.barh(range(cluster.status['sessions'].sum()),-(cluster.status[n,cluster.status['sessions'],2]*10.),left=-5,facecolor='r')
                ax.set_xlim([-10,nbin])
                ax.set_ylim([cluster.status['sessions'].sum(),-0.5])
                ax.set_yticklabels([])
            plt.show(block=False)
        return neuron_clusters



    if plot_fig[42]:

        print('### display neurons with extraordinary activation / coding probability ###')

        s_bool = np.zeros(nSes,'bool')
        s_bool[17:87] = True
        # s_bool[:] = True
        s_bool[~cluster.status['sessions']] = False

        ds = 2
        if reprocess:
            cluster.stats['field_stability'] = get_field_stability(cluster,SD=1.96,s_bool=s_bool)
        if reprocess:
            cluster.stats['field_stability_temp'] = get_field_stability_temp(cluster,SD=1.96,ds=ds)
        if reprocess:
            cluster.stats['act_stability'] = get_act_stability(cluster,s_bool=s_bool)
        if reprocess:
            cluster.stats['act_stability_temp'] = get_act_stability_temp(cluster,ds=ds)

        act_clusters = np.any(cluster.status['activity'][:,s_bool,1],1)
        r_stab = cluster.stats['field_stability'][act_clusters]
        r_stab_temp = gauss_smooth(cluster.stats['field_stability_temp'][act_clusters,:],(0,0))
        act_stab_temp = gauss_smooth(cluster.stats['act_stability_temp'][act_clusters,:,1],(0,1))

        # r_stab = np.nanmax(r_stab_temp[act_clusters,:][:,s_bool],1)
        nC_good = len(r_stab)

        # print(np.histogram(np.nanmax(r_stab_temp[:,s_bool],1),np.linspace(0,1,21)))

        # r_stab_temp_masked = np.ma.masked_array(r_stab_temp,mask=~cluster.status['activity'][...,1])
        # r_stab = np.ma.mean(r_stab_temp_masked,1)

        act_stab = cluster.stats['act_stability'][act_clusters,1]
        # act_stab = cluster.stats['p_post_c']['act']['act'][:,1,0]

        fig = plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        print(pl_dat.sv_opt['dpi'])
        ax = plt.axes([0.05,0.65,0.4,0.3])
        pl_dat.add_number(fig,ax,order=1,offset=[-50,25])
        pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/individual/r_stab_example.png'
        ax.axis('off')
        if os.path.exists(pic_path):
            im = mpimg.imread(pic_path)
            ax.imshow(im)
            ax.set_xlim([0,im.shape[1]])

        ax = plt.axes([0.575,0.7,0.15,0.225])
        pl_dat.add_number(fig,ax,order=2)
        # ax.hist(r_stab,np.linspace(0,0.6,51),color='tab:blue',alpha=0.5)
        # ax.set_yticks([])
        # ax.set_xticklabels([])
        # pl_dat.remove_frame(ax,['top','right'])



        # ax2 = ax.twinx()
        ax.plot(r_stab+0.02*np.random.rand(nC_good),cluster.status['activity'][act_clusters,:,1][:,s_bool].sum(1)+0.5*np.random.rand(nC_good),'k.',markersize=1,markeredgecolor='none')
        # ax.yaxis.set_label_position("left")
        # ax.yaxis.set_ticks_position("left")
        ax.set_xlabel('$r_{\\gamma^+}^{\infty}$')
        ax.set_ylabel('$N_{\\alpha^+}$')
        ax.set_ylim([0,70])
        ax.set_xticks(np.linspace(0,0.75,4))
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.8,0.7,0.15,0.225])
        Nb_arr = np.arange(80)
        ax.plot(r_stab+0.02*np.random.rand(nC_good),cluster.status['activity'][act_clusters,:][:,s_bool,2].sum(1)+0.5*np.random.rand(nC_good),'k.',markersize=1,markeredgecolor='none')
        SD = 1.96
        sig_theta = cluster.stability['all']['mean'][0,2]
        # ax.plot(((2*SD*sig_theta/100 * Nb_arr**2)/(Nb_arr-1))/s_bool.sum(),Nb_arr,'r--',linewidth=0.75)
        exp_random = ((2*SD*sig_theta/100 * Nb_arr**2)/(Nb_arr-1))/s_bool.sum()
        ax.plot(exp_random,Nb_arr,'r--',linewidth=0.75)
        ax.set_ylabel('$N_{\\beta^+}$')
        ax.set_xlabel('$r_{\\gamma^+}^{\infty}$')
        ax.set_ylim([0,70])
        ax.set_xticks(np.linspace(0,0.75,4))
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.89,0.75,0.075,0.05])
        Nb = cluster.status['activity'][act_clusters,:,2]
        Nb = Nb[:,s_bool].sum(1)
        nb_diff = np.zeros(nC_good)*np.NaN
        for nb in range(1,80):
            idx_c = Nb==nb
            nb_diff[idx_c] = r_stab[idx_c] - exp_random[nb]
        ax.hist(nb_diff,np.linspace(-0.05,0.15,51),color='k')
        ax.set_yticks([])
        ax.set_xlabel('$\Delta r^{\infty}_{\\gamma}$',fontsize=8)
        ax.xaxis.set_label_coords(1.15,-0.2)
        pl_dat.remove_frame(ax,['top','right'])



        # ax = plt.axes([0.725,0.7,0.225,0.225])
        # pl_dat.add_number(fig,ax,order=3)
        # # idx_codeoften = cluster.status['activity'][...,2].sum(1)>5
        # # nOften = idx_codeoften.sum()
        # # ax.plot(0,np.NaN,'r.',alpha=0.5,label='$N_{\\beta} \leq 5$')
        # # ax.plot(0,np.NaN,'k.',label='$N_{\\beta} > 5$')
        #
        # ax.scatter(r_stab+0.02*np.random.rand(nC_good),act_stab+0.02*np.random.rand(nC_good),s=cluster.status[act_clusters,:,2][:,s_bool].sum(1)/10,c='k',edgecolor='none')
        # # ax.scatter(r_stab[~idx_codeoften]+0.02*np.random.rand(nC-nOften),act_stab[~idx_codeoften]+0.02*np.random.rand(nC-nOften),s=cluster.status[~idx_codeoften,:,2].sum(1)/10,c='r',edgecolor='none',alpha=0.5)
        # ax.set_xlabel('$r_{\\gamma^+}^{\infty}$')
        # ax.set_ylabel('$p(\\alpha_{s+1}^+|\\alpha_s^+)$')
        # pl_dat.remove_frame(ax,['top','right'])
        # # ax.legend(fontsize=10)

        field_stab_range = [0,0.15,0.3,0.4,0.6,1]

        for j in range(len(field_stab_range)-1):

            # try:
            idx_c = np.where((r_stab > field_stab_range[j]) & (r_stab < field_stab_range[j+1]) & (cluster.status['activity'][act_clusters,:,1][:,s_bool].sum(1)>0.2*nSes))[0]
            # print(idx_c)
            # idx_c = np.where(act_clusters)[0][idx_c0]
            # print(idx_c)
            # print(field_stab_range[j])
            # print(idx_c)
            c_arr = np.random.choice(idx_c,1,replace=False)
            for i,c0 in enumerate(c_arr):
                ax = plt.axes([0.1+0.1*i+0.175*j,0.08,0.075,0.425])

                c = np.where(act_clusters)[0][c0]
                firingmap = cluster.stats['firingmap'][c,cluster.status['sessions'],:]
                firingmap[~cluster.status['activity'][c,cluster.status['sessions'],1],:] = np.NaN
                firingmap = gauss_smooth(firingmap,(0,4))
                firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
                im = ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,5])
                ax.barh(range(cluster.status['sessions'].sum()),-(cluster.status['activity'][c,cluster.status['sessions'],2]*10.),left=-5,facecolor='r')
                ax.set_xlim([-10,nbin])
                ax.set_ylim([cluster.status['sessions'].sum(),-0.5])
                if (j==0) & (i==0):
                    pl_dat.add_number(fig,ax,order=3)#,offset=[-100,10]
                    ax.set_ylabel('Session')
                else:
                    ax.set_yticklabels([])
                if (i==0):
                    ax.set_xlabel('position',fontsize=8)
                    # ax.xaxis.set_label_coords(1.25,-0.1)
                #r_{\\gamma^+}^{\infty} \\approx
                ax.set_title('$\\approx %.2f$'%r_stab[c0],fontsize=6)
                pl_dat.remove_frame(ax,['top','left','right'])

                ax = plt.axes([0.1+0.075+0.175*j,0.08,0.05,0.425])
                ax.plot([1,1],[nSes,0],'--',color=[0.6,0.6,0.6],linewidth=0.5)
                ax.plot([0.5,0.5],[nSes,0],':',color=[0.6,0.6,0.6],linewidth=0.3)
                ax.plot(r_stab_temp[c0,cluster.status['sessions']],np.arange(0,cluster.status['sessions'].sum()),'-',color=[0.6,0.6,0.6],linewidth=0.3)
                ax.plot(gauss_smooth(r_stab_temp[c0,cluster.status['sessions']],1),np.arange(0,cluster.status['sessions'].sum()),'k-',linewidth=0.75)

                ax.plot(act_stab_temp[c0,cluster.status['sessions']],np.arange(0,cluster.status['sessions'].sum()),'r-')

                ax.set_xlim([0,1.2])
                ax.set_ylim([cluster.status['sessions'].sum(),-0.5])
                ax.set_yticklabels([])
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.set_xlabel('$r_{\gamma}^{%d}$'%ds,fontsize=8)
                pl_dat.remove_frame(ax,['bottom','right'])
            # except:
                # pass


        cbaxes = plt.axes([0.945,0.355,0.005,0.15])
        cb = fig.colorbar(im,cax = cbaxes,orientation='vertical')
        cb.set_label('firing rate [Hz]',fontsize=8)
        # cb.axis.set_label_coords(1,0.5)
        # cb.set_ticks([0,3])
        # cb.set_ticklabels(['$\\nu_{min}$','$\\nu_{max}$'])

        # plt.plot(gauss_smooth(np.nansum(r_stab>0.5,0),1))
        # plt.xlim([0,90])
        # plt.ylim([0,100])
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('individual_neurons')
        return



        ds=10
        if reprocess:
            cluster.stats['field_stability_temp'] = get_field_stability_temp(cluster,SD=1.96,ds=ds)
        r_stab_temp = cluster.stats['field_stability_temp']
        plt.figure(figsize=(7,5),dpi=300)

        field_stab_range = [0,0.4,0.6,0.8,1]

        # for j in range(len(field_stab_range)-1):

            # idx_c = np.unique(np.where((r_stab_temp > field_stab_range[j]) & (r_stab_temp < field_stab_range[j+1]))[0])
        idx_c = np.unique(np.where((r_stab_temp > 0.5) & (cluster.status['activity'][...,1].sum(1)[:,np.newaxis]>20))[0])
            # print(field_stab_range[j])
            # print(idx_c)
        c_arr = np.random.choice(idx_c,4,replace=False)
        for j,c in enumerate(c_arr):
            ax = plt.axes([0.1+0.225*j,0.08,0.075,0.5])

            firingmap = cluster.stats['firingmap'][c,cluster.status['sessions'],:]
            firingmap[~cluster.status['activity'][c,cluster.status['sessions'],1],:] = np.NaN
            firingmap = gauss_smooth(firingmap,(0,2))
            firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
            im = ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,5])
            ax.barh(range(cluster.status['sessions'].sum()),-(cluster.status['activity'][c,cluster.status['sessions'],2]*10.),left=-5,facecolor='r')
            ax.set_xlim([-10,nbin])
            ax.set_ylim([cluster.status['sessions'].sum(),-0.5])
            if (j==0):
                pl_dat.add_number(fig,ax,order=4)#,offset=[-100,25]
                ax.set_ylabel('Session')
            else:
                ax.set_yticklabels([])
            # if (i==0):
                # ax.set_xlabel('Location [bins]',fontsize=8)
                # ax.xaxis.set_label_coords(1.25,-0.1)

            # ax.set_title('$r_{\\gamma^+}^* = %.2g$'%r_stab_temp[c],fontsize=8)
            pl_dat.remove_frame(ax)

            ax = plt.axes([0.1+0.1+0.225*j,0.08,0.075,0.5])
            ax.plot(r_stab_temp[c,cluster.status['sessions']],np.arange(0,cluster.status['sessions'].sum()),'k-')
            print(act_stab_temp[c,cluster.status['sessions']])
            ax.plot(act_stab_temp[c,cluster.status['sessions']],np.arange(0,cluster.status['sessions'].sum()),'r-')
            ax.set_xlim([-0.05,1.2])
            ax.set_ylim([cluster.status['sessions'].sum(),-0.5])

        plt.show(block=False)


    





    if plot_fig[44]:

        print('get transition probabilities')

        # if not 'transition' in cluster.stats.keys():
        cluster.get_locTransition_prob()

        SD = 1
        p_rec_loc = np.zeros((nSes,nbin))*np.NaN
        # for ds in range(1,min(nSes,41)):
        ds = 1
        session_bool = np.where(np.pad(cluster.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(cluster.status['sessions'][:],(0,0),constant_values=False))[0]

        s1_shifts,s2_shifts,f1,f2 = np.unravel_index(cluster.compare['pointer'].col,(nSes,nSes,cluster.params['field_count_max'],cluster.params['field_count_max']))
        c_shifts = cluster.compare['pointer'].row
        sig = 6
        di = 3

        Ds = s2_shifts-s1_shifts
        idx = np.where(Ds==ds)[0]
        idx_shifts = cluster.compare['pointer'].data[idx].astype('int')-1
        shifts = cluster.compare['shifts'][idx_shifts]

        reloc_dist = np.zeros((nbin,2))
        s = s1_shifts[idx]
        f = f1[idx]
        c = c_shifts[idx]
        loc_shifts = np.round(cluster.fields['location'][c,s,f,0]).astype('int')
        for i in range(nbin):
            i_min = max(0,i-di)
            i_max = min(nbin,i+di)
            for s0 in np.where(cluster.status['sessions'])[0]:
                idx_loc = (loc_shifts>=i_min) & (loc_shifts<i_max) & (s==s0)

                shifts_loc = shifts[idx_loc]
                N_data = len(shifts_loc)
                N_stable = (np.abs(shifts_loc)<(SD*sig)).sum()

                p_rec_loc[s0,i] = N_stable/N_data

            idx_loc = (loc_shifts>=i_min) & (loc_shifts<i_max)
            shifts_loc = shifts[idx_loc]

            reloc_dist[i,:] = [np.abs(shifts_loc).mean(),np.abs(shifts_loc).std()]

        ds = 1
        session_bool = np.where(np.pad(cluster.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(cluster.status['sessions'][:],(0,0),constant_values=False))[0]
        # print(session_bool)
        loc_stab = np.zeros((nSes,nbin+2,nbin+2))
        loc_stab_p = np.zeros((nSes,nbin+2,nbin+2))
        for s in session_bool:#range(nSes):#np.where(cluster.status['sessions'])[0]:
            ### assign bin-specific remapping to rows, active nPC (nbins+1) & silent (nbins+2)
            for c in np.where(cluster.status['activity'][:,s,2])[0]:
                ## find belonging fields
                if cluster.status['activity'][c,s+ds,2]:
                    d = np.abs(np.mod(cluster.fields['location'][c,s,:,0][:,np.newaxis] - cluster.fields['location'][c,s+ds,:,0]+nbin/2,nbin)-nbin/2)
                    d[np.isnan(d)] = nbin
                    f1,f2 = sp.optimize.linear_sum_assignment(d)
                    for f in zip(f1,f2):
                        if d[f] < nbin:
                            loc_stab[s,int(round(cluster.fields['location'][c,s,f[0],0])),int(round(cluster.fields['location'][c,s+ds,f[1],0]))] += 1
                            loc_stab_p[s,int(round(cluster.fields['location'][c,s,f[0],0])),:nbin] += cluster.fields['p_x'][c,s+ds,f[1],:]

        loc_stab = loc_stab[:,:nbin,:nbin]
        loc_stab_p = loc_stab_p[:,:nbin,:nbin]

        fields = np.zeros((nSes,nbin))
        for i,s in enumerate(np.where(cluster.status['sessions'])[0]):
            idx_PC = np.where(cluster.status_fields[:,s,:])
            fields[s,:] = np.nansum(cluster.fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
        print(fields.shape)
        print(fields.sum(1))
        fields /= fields.sum(1)[:,np.newaxis]


        plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])

        ## find location specific stabilization
        ax = plt.axes([0.1,0.8,0.2,0.15])
        RW_stab = np.zeros(nSes)*np.NaN
        GT_stab = np.zeros(nSes)*np.NaN
        nRnG_stab = np.zeros(nSes)*np.NaN
        RW_dense = np.zeros(nSes)*np.NaN
        GT_dense = np.zeros(nSes)*np.NaN
        nRnG_dense = np.zeros(nSes)*np.NaN
        for s in np.where(cluster.status['sessions'])[0]:
            RW_pos = cluster.session_data['RW_pos'][s,:].astype('int')
            GT_pos = cluster.session_data['GT_pos'][s,:].astype('int')
            # print(RW_pos)
            # print(GT_pos)
            zone_nRnG = np.ones(nbin,'bool')
            zone_nRnG[RW_pos[0]:RW_pos[1]] = False
            zone_nRnG[GT_pos[0]:GT_pos[1]] = False
            # print(s)
            # print(np.nanmean(p_rec_loc[s,RW_pos[0]:RW_pos[1]]))
            # RW_stab[s] = np.nanmean(p_rec_loc[s,RW_pos[0]:RW_pos[1]])
            # GT_stab[s] = np.nanmean(p_rec_loc[s,GT_pos[0]:GT_pos[1]])
            # nRnG_stab[s] = np.nanmean(p_rec_loc[s,zone_nRnG])
            RW_dense[s] = np.nanmean(fields[s,RW_pos[0]:RW_pos[1]])
            GT_dense[s] = np.nanmean(fields[s,GT_pos[0]:GT_pos[1]])
            nRnG_dense[s] = np.nanmean(fields[s,zone_nRnG])

        # ax.plot(gauss_smooth(RW_stab,1,mode='constant'),color='tab:red')
        # ax.plot(gauss_smooth(GT_stab,1,mode='constant'),color='tab:green')
        # ax.plot(gauss_smooth(nRnG_stab,1,mode='constant'),color='tab:blue')
        ax.plot(gauss_smooth(RW_dense,1,mode='constant'),color='tab:red')
        ax.plot(gauss_smooth(GT_dense,1,mode='constant'),color='tab:green')
        ax.plot(gauss_smooth(nRnG_dense,1,mode='constant'),color='tab:blue')
        # non_start = np.copy(cluster.params['zone_mask']['others'])
        # non_start[:13] = False
        # nRnG_stab = np.nanmean(p_rec_loc[:,non_start],1)
        # START_stab = np.nanmean(p_rec_loc[:,15:35],1)
        ax.set_ylim([0,0.03])

        ax = plt.axes([0.4,0.8,0.1,0.15])
        pl_dat.plot_with_confidence(ax,range(nbin),reloc_dist[:,0],reloc_dist[:,1],col='k')


        s_arr = np.array([0,5,10,15])
        # s_arr += np.where(cluster.status['sessions'])[0][0]
        print(s_arr)
        # s_arr = np.array([0,10,21])
        n_int = len(s_arr)-1

        gate = np.any(cluster.params['zone_mask']['gate'])
        if gate:
            ax_GT = plt.axes([0.3,0.1,0.15,0.175])
            ax_GT_dist = plt.axes([0.4,0.2,0.05,0.05])
            ax_GT.bar(range(nbin),1000.*cluster.params['zone_mask']['gate'],width=1,facecolor='tab:green',alpha=0.3)
            ax_GT.set_ylim([0,0.1])
            pl_dat.remove_frame(ax_GT,['top','right'])

        ax_RW = plt.axes([0.1,0.1,0.15,0.175])
        ax_RW_dist = plt.axes([0.125,0.2,0.05,0.05])
        ax_RW.bar(range(nbin),1000.*cluster.params['zone_mask']['reward'],width=1,facecolor='tab:red',alpha=0.3)
        ax_RW.set_ylim([0,0.1])
        pl_dat.remove_frame(ax_RW,['top','right'])
        ax_RW.set_xlabel('position [bins]')

        ax_nRnG = plt.axes([0.5,0.1,0.15,0.175])
        ax_nRnG_dist = plt.axes([0.6,0.2,0.05,0.05])
        ax_nRnG.bar(range(nbin),1000.*cluster.params['zone_mask']['others'],width=1,facecolor='tab:blue',alpha=0.3)
        ax_nRnG.set_ylim([0,0.1])
        pl_dat.remove_frame(ax_nRnG,['top','right'])
        ax_nRnG.set_xlabel('position [bins]')

        for j in range(n_int):
            col = [1,0.2*j,0.2*j]
            occ = loc_stab_p[s_arr[j]:s_arr[j+1],cluster.params['zone_mask']['reward'],:].sum(0).sum(0)
            occ /= occ.sum()
            ax_RW.plot(range(nbin),occ,'-',color=col,label='Sessions %d-%d'%(s_arr[j]+1,s_arr[j+1]),linewidth=0.5)
            # ax.bar(range(nbin),loc_stab[:20,cluster.params['zone_mask']['reward'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)

            col = [0.2*j,0.2*j,1]
            occ = loc_stab_p[s_arr[j]:s_arr[j+1],cluster.params['zone_mask']['others'],:].sum(0).sum(0)
            occ /= occ.sum()
            ax_nRnG.plot(range(nbin),occ,'-',color=col,linewidth=0.5)
            # ax.bar(range(nbin),loc_stab[:20,cluster.params['zone_mask']['others'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)

            if gate:
                col = [0.2*j,0.8,0.2*j]
                occ = loc_stab_p[s_arr[j]:s_arr[j+1],cluster.params['zone_mask']['gate'],:].sum(0).sum(0)
                occ /= occ.sum()
                ax_GT.plot(range(nbin),occ,'-',color=col,linewidth=0.5)
                # ax.bar(range(nbin),loc_stab[:20,cluster.params['zone_mask']['others'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)
        ax_RW.legend(fontsize=6,loc='upper left',bbox_to_anchor=[0.05,1.1])
        props = dict(boxstyle='round', facecolor='w', alpha=0.8)


        s_bool = np.zeros(nSes,'bool')
        # s_bool[17:87] = True
        s_bool[32:63] = True
        s_bool[~cluster.status['sessions']] = False

        ls_arr = ['-','--',':']
        label_arr = ['silent','active','coding']

        mask_others = np.copy(cluster.params['zone_mask']['others'])
        mask_others[:15] = False
        smooth = 1

        ax = plt.axes([0.1,0.4,0.35,0.2])

        ax.plot([0,nbin],[0,0],'--',color=[0.5,0.5,0.5],linewidth=0.5)
        for j in range(3):

            ax.plot(range(nbin),gauss_smooth(np.nanmean(cluster.stats['transition']['recruitment'][s_bool,:,j],0),smooth),color='k',linestyle=ls_arr[j],label='recruiting' if j==0 else None)

            ax.plot(range(nbin),-gauss_smooth(np.nanmean(cluster.stats['transition']['dismissal'][s_bool,:,j],0),smooth),color='k',linestyle=ls_arr[j],label='dismissing' if j==0 else None)

            ax.plot(range(nbin),gauss_smooth(np.nanmean(cluster.stats['transition']['recruitment'][s_bool,:,j]-cluster.stats['transition']['dismissal'][s_bool,:,j],0),smooth),color='tab:red',linestyle=ls_arr[j],linewidth=0.5)

        ax.plot(range(nbin),gauss_smooth(np.nanmean(cluster.stats['transition']['stabilization'][s_bool,:],0),smooth),color='tab:green',label='stabilizing')
        ax.set_ylim([-3,3])
        ax.set_ylabel('$\\frac{\# PC}{bin}$')
        pl_dat.remove_frame(ax,['top','right'])
        ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.1,1.1])


        # ax_abs = plt.axes([0.1,0.1,0.35,0.2])



        ax_recruit = plt.axes([0.7,0.8,0.25,0.15])
        # ax_dismiss = plt.axes([0.6,0.6,0.25,0.15])
        ax_diff = plt.axes([0.7,0.6,0.25,0.15])
        ax_stable = plt.axes([0.7,0.4,0.25,0.15])

        # ax_abs.plot([0,nSes],[0,0],color=[0.5,0.5,0.5],linestyle='--',linewidth=0.3)
        # ax_abs.plot(cluster.stats['transition']['stabilization'].sum(1),'-',color='k')
        for j in range(3):
            # ax_abs.plot(cluster.stats['transition']['recruitment'][:,:,j].sum(1),linestyle=ls_arr[j],color='k',linewidth=0.5)
            # ax_abs.plot(-cluster.stats['transition']['dismissal'][:,:,j].sum(1),linestyle=ls_arr[j],color='k',linewidth=0.5)
            # ax_abs.plot(cluster.stats['transition']['recruitment'][:,:,j].sum(1)-cluster.stats['transition']['dismissal'][:,:,j].sum(1),linestyle=ls_arr[j],color='r',linewidth=0.5)

            ax_recruit.plot(range(nSes),gauss_smooth(np.nanmean(cluster.stats['transition']['recruitment'][:,cluster.params['zone_mask']['reward'],j],1),smooth),linestyle=ls_arr[j],color='tab:red',label=label_arr[j])
            ax_recruit.plot(range(nSes),gauss_smooth(np.nanmean(cluster.stats['transition']['recruitment'][:,mask_others,0],1),smooth),linestyle=ls_arr[j],color='tab:blue')

            ax_recruit.plot(range(nSes),-gauss_smooth(np.nanmean(cluster.stats['transition']['dismissal'][:,cluster.params['zone_mask']['reward'],j],1),smooth),linestyle=ls_arr[j],color='tab:red')
            ax_recruit.plot(range(nSes),-gauss_smooth(np.nanmean(cluster.stats['transition']['dismissal'][:,mask_others,j],1),smooth),linestyle=ls_arr[j],color='tab:blue')

            ax_diff.plot(range(nSes),gauss_smooth(np.nanmean(cluster.stats['transition']['recruitment'][:,cluster.params['zone_mask']['reward'],j],1),smooth)-gauss_smooth(np.nanmean(cluster.stats['transition']['dismissal'][:,cluster.params['zone_mask']['reward'],j],1),smooth),linestyle=ls_arr[j],color='tab:red')

            ax_diff.plot(range(nSes),gauss_smooth(np.nanmean(cluster.stats['transition']['recruitment'][:,mask_others,0],1),smooth)-gauss_smooth(np.nanmean(cluster.stats['transition']['dismissal'][:,mask_others,j],1),smooth),linestyle=ls_arr[j],color='tab:blue')

        ax_stable.plot(range(nSes),gauss_smooth(np.nanmean(cluster.stats['transition']['stabilization'][:,cluster.params['zone_mask']['reward']],1),smooth),linestyle='-',color='tab:red')
        ax_stable.plot(range(nSes),gauss_smooth(np.nanmean(cluster.stats['transition']['stabilization'][:,mask_others],1),smooth),linestyle='-',color='tab:blue')

        ax_recruit.set_ylim([-3,3])
        ax_recruit.set_xlim([0,np.where(cluster.status['sessions'])[0][-1]])
        ax_recruit.legend(fontsize=8)
        pl_dat.remove_frame(ax_recruit,['top','right'])
        ax_recruit.set_ylabel('recruit')
        # ax_dismiss.set_ylim([0,3])
        # ax_dismiss.set_xlim([0,np.where(cluster.status['sessions'])[0][-1]])
        # pl_dat.remove_frame(ax_dismiss,['top','right'])
        # ax_dismiss.set_ylabel('dismiss')
        ax_stable.set_ylim([0,3])
        ax_stable.set_xlim([0,np.where(cluster.status['sessions'])[0][-1]])
        ax_diff.set_ylabel('diff')
        ax_diff.set_xlim([0,np.where(cluster.status['sessions'])[0][-1]])
        pl_dat.remove_frame(ax_diff,['top','right'])
        pl_dat.remove_frame(ax_stable,['top','right'])
        ax_stable.set_ylabel('stable')

        plt.show(block=False)

        if sv:
            pl_dat.save_fig('timedep_remapping')


    if plot_fig[45]:

        print('### plot, how long-range probs are influenced by salience / reward ###')

        s_arr = [1,5,10,15,20]
        # s_arr = [1,5,17,30,87]
        plt.figure(figsize=(7,5),dpi=300)


        key = 'code'
        ### during s5-10
        for j in range(len(s_arr)-1):
            ax1 = plt.axes([0.1,0.75-j*0.225,0.25,0.15])
            ax1.plot(np.nanmean(cluster.stats['p_post_s']['act']['act'][s_arr[j]:s_arr[j+1],:,0],0),color=[0.6,0.6,0.6],linewidth=0.5)
            ax1.plot(np.nanmean(cluster.stats['p_post_RW_s'][key]['act'][s_arr[j]:s_arr[j+1],:,0],0),color='r')
            ax1.plot(np.nanmean(cluster.stats['p_post_GT_s'][key]['act'][s_arr[j]:s_arr[j+1],:,0],0),color='g')
            ax1.plot(np.nanmean(cluster.stats['p_post_nRnG_s'][key]['act'][s_arr[j]:s_arr[j+1],:,0],0),color='b')
            ax1.set_ylim([0,1])

            ax2 = plt.axes([0.4,0.75-j*0.225,0.25,0.15])
            ax2.plot(np.nanmean(cluster.stats['p_post_s']['act']['code'][s_arr[j]:s_arr[j+1],:,0],0),color=[0.6,0.6,0.6],linewidth=0.5)
            ax2.plot(np.nanmean(cluster.stats['p_post_RW_s'][key]['code'][s_arr[j]:s_arr[j+1],:,0],0),color='r')
            ax2.plot(np.nanmean(cluster.stats['p_post_GT_s'][key]['code'][s_arr[j]:s_arr[j+1],:,0],0),color='g')
            ax2.plot(np.nanmean(cluster.stats['p_post_nRnG_s'][key]['code'][s_arr[j]:s_arr[j+1],:,0],0),color='b')
            ax2.set_ylim([0,1])

            ax3 = plt.axes([0.7,0.75-j*0.225,0.25,0.15])
            ax3.plot(np.nanmean(cluster.stats['p_post_s']['act']['stable'][s_arr[j]:s_arr[j+1],:,0],0),color=[0.6,0.6,0.6],linewidth=0.5)
            ax3.plot(np.nanmean(cluster.stats['p_post_RW_s'][key]['stable'][s_arr[j]:s_arr[j+1],:,0],0),color='r')
            ax3.plot(np.nanmean(cluster.stats['p_post_GT_s'][key]['stable'][s_arr[j]:s_arr[j+1],:,0],0),color='g')
            ax3.plot(np.nanmean(cluster.stats['p_post_nRnG_s'][key]['stable'][s_arr[j]:s_arr[j+1],:,0],0),color='b')
            ax3.set_ylim([0,1])
        plt.show(block=False)



    if plot_fig[46]:

        print('### role of individual neurons pt.2 ###')
        s_bool = np.zeros(nSes,'bool')
        s_bool[17:87] = True
        s_bool[~cluster.status['sessions']] = False

        ds = 3
        plt.figure(figsize=(7,4),dpi=300)

        if (not ('act_stability_temp' in cluster.stats.keys())) | reprocess:
            cluster.stats['act_stability_temp'] = get_act_stability_temp(cluster,ds=ds)
        if (not ('field_stability_temp' in cluster.stats.keys())) | reprocess:
            cluster.stats['field_stability_temp'] = get_field_stability_temp(cluster,SD=1.96,ds=ds)
        if (not ('field_stability' in cluster.stats.keys())):
            cluster.stats['field_stability'] = get_field_stability(cluster,SD=1.96,s_bool=s_bool)
        if (not ('act_stability' in cluster.stats.keys())):
            cluster.stats['act_stability'] = get_act_stability(cluster,s_bool=s_bool)

        r_stab = cluster.stats['field_stability_temp']
        act_stab = cluster.stats['act_stability_temp'][...,1]

        act_stab_thr = [0.1,0.9]
        r_stab_thr = [0.1,0.5]

        cluster_high = np.nanmax(r_stab,1)>r_stab_thr[1]
        cluster_low = np.nanmax(r_stab,1)<r_stab_thr[0]

        c_high,s_high,f_high = np.where((r_stab>r_stab_thr[1])[...,np.newaxis] & s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)
        c_low,s_low,f_low = np.where((r_stab<r_stab_thr[0])[...,np.newaxis] & s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)
        c_med,s_med,f_med = np.where(((r_stab>=r_stab_thr[0]) & (r_stab<=r_stab_thr[1]))[...,np.newaxis] & s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)
        c_high_low,s_high_low,f_high_low = np.where((r_stab<r_stab_thr[0])[...,np.newaxis] & cluster_high[:,np.newaxis,np.newaxis] & s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)
        c_low_low,s_low_low,f_low_low = np.where((r_stab<r_stab_thr[0])[...,np.newaxis] & cluster_low[:,np.newaxis,np.newaxis] & s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)

        c_all,s_all,f_all = np.where(s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)
        c_all_low,s_all_low,f_all_low = np.where((np.nanmax(r_stab,1)<r_stab_thr[0])[:,np.newaxis,np.newaxis] & s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)

        # loc_all = cluster.fields['location'][c_all,s_all,f_all,0]
        # loc_high = cluster.fields['location'][c_high,s_high,f_high,0]
        # loc_low = cluster.fields['location'][c_low,s_low,f_low,0]

        loc_all = cluster.fields['p_x'][c_all,s_all,f_all,:].sum(0)
        loc_high = cluster.fields['p_x'][c_high,s_high,f_high,:].sum(0)
        loc_low = cluster.fields['p_x'][c_low,s_low,f_low,:].sum(0)
        loc_all_low = cluster.fields['p_x'][c_all_low,s_all_low,f_all_low,:].sum(0)

        # print(loc_all.shape)
        for loc in [loc_all,loc_high,loc_low,loc_all_low]:
            loc /= loc.sum()


        ax = plt.axes([0.1,0.825,0.35,0.1])
        # plt.hist(loc_all,np.linspace(0,100,101),density=True,alpha=0.5)
        ax.bar(np.linspace(0,nbin-1,nbin),loc_high,width=1,alpha=0.5,color='tab:red')
        ax.bar(np.linspace(0,nbin-1,nbin),loc_low,width=1,alpha=0.5,color='tab:blue')
        # ax.bar(np.linspace(0,nbin-1,nbin),loc_all_low,width=1,alpha=0.5,color='k')
        # ax.bar(np.linspace(0,100,101),loc_high,alpha=0.5,color='tab:red')
        pl_dat.remove_frame(ax,['top','right'])
        ax.set_ylabel('density')
        ax.set_xlabel('position')
        ax.set_xticklabels([])

        # ax = plt.axes([0.1,0.7,0.35,0.1])
        # c_all_low,s_all_low,f_all_low = np.where(cluster_high[:,np.newaxis,np.newaxis] & (r_stab<r_stab_thr[1])[...,np.newaxis] & s_bool[np.newaxis,:,np.newaxis] & cluster.status_fields)
        # loc_high_low = cluster.fields['p_x'][c_all_low,s_all_low,f_all_low,:].sum(0)
        # ax.bar(np.linspace(0,nbin-1,nbin),loc_high_low,width=1,alpha=0.5,color='tab:red')
        # pl_dat.remove_frame(ax,['top','right'])
        # ax.set_xlabel('position')
        # ax.hist(loc_high,np.linspace(0,100,101),density=True,alpha=0.5,color='tab:red')
        # ax.hist(loc_low,np.linspace(0,100,101),density=True,alpha=0.5,color='tab:blue')
        # ax.hist(loc_all_low,np.linspace(0,100,101),density=True,alpha=0.5,color='k')
        # ax.hist()

        ax = plt.axes([0.6,0.825,0.35,0.1])
        ax.hist(cluster.stats['act_stability'][cluster_high,1],np.linspace(0.001,1,21),density=True,alpha=0.5,color='tab:red')
        ax.hist(cluster.stats['act_stability'][cluster_low,1],np.linspace(0.001,1,21),density=True,alpha=0.5,color='tab:blue')
        ax.set_xlabel('$r^{\infty}_{\\alpha^+}$')
        pl_dat.remove_frame(ax,['top','right'])

        c_low_low0,s_low_low0 = np.unique([c_low_low,s_low_low],axis=1)
        c_high_low0,s_high_low0 = np.unique([c_high_low,s_high_low],axis=1)
        c_high0,s_high0 = np.unique([c_high,s_high],axis=1)

        # print(c_low0.shape)
        ax = plt.axes([0.1,0.5,0.075,0.15])
        # MI_low = cluster.stats['MI_value'][c_low0,s_low0]
        # MI_med = cluster.stats['MI_value'][c_med,s_med]
        MI_low_low = cluster.stats['MI_value'][c_low_low0,s_low_low0]
        MI_high_low = cluster.stats['MI_value'][c_high_low0,s_high_low0]
        MI_high = cluster.stats['MI_value'][c_high0,s_high0]
        res = sstats.kruskal(MI_low_low,MI_high)
        print(res)
        res = sstats.f_oneway(MI_low_low,MI_high_low)
        print(res)
        # MI_low = cluster.stats['MI_value'][c_low,s_low]
        # MI_low = cluster.stats['Isec_value'][c_low,s_low]
        # ax.boxplot(MI_low,positions=[0],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.boxplot(MI_low_low,positions=[0],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.boxplot(MI_high_low,positions=[1],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

        # ax.boxplot(MI_med,positions=[3],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        # MI_high = cluster.stats['Isec_value'][c_high,s_high]
        ax.boxplot(MI_high,positions=[2],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.set_ylabel('MI')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['low $r_{\\gamma^+}^%d$'%ds,'low$^* r_{\\gamma^+}^%d$'%ds,'high $r_{\\gamma^+}^%d$'%ds],rotation=60,fontsize=8)
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.275,0.5,0.075,0.15])
        nu_low = cluster.stats['if_firingrate_adapt'][c_low,s_low,f_low]
        ax.boxplot(nu_low[np.isfinite(nu_low)],positions=[0],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        nu_high = cluster.stats['if_firingrate_adapt'][c_high,s_high,f_high]
        ax.boxplot(nu_high[np.isfinite(nu_high)],positions=[1],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.set_ylabel('$\\nu^+$')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['low $r_{\\gamma^+}^%d$'%ds,'low$^* r_{\\gamma^+}^%d$'%ds,'high $r_{\\gamma^+}^%d$'%ds],rotation=60,fontsize=8)
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.45,0.5,0.075,0.15])
        sig_low = cluster.fields['width'][c_low_low,s_low_low,f_low_low,0]
        sig_high_low = cluster.fields['width'][c_high_low,s_high_low,f_high_low,0]
        sig_high = cluster.fields['width'][c_high,s_high,f_high,0]
        ax.boxplot(sig_low,positions=[0],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.boxplot(sig_high_low,positions=[1],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.boxplot(sig_high,positions=[2],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.set_ylabel('$\\sigma_{\\theta}$')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['low $r_{\\gamma^+}^%d$'%ds,'low$^* r_{\\gamma^+}^%d$'%ds,'high $r_{\\gamma^+}^%d$'%ds],rotation=60,fontsize=8)
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.625,0.5,0.075,0.15])
        rel_low = cluster.fields['reliability'][c_low_low,s_low_low,f_low_low]
        rel_high_low = cluster.fields['reliability'][c_high_low,s_high_low,f_high_low]
        rel_high = cluster.fields['reliability'][c_high,s_high,f_high]
        ax.boxplot(rel_low,positions=[0],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.boxplot(rel_high_low,positions=[1],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.boxplot(rel_high,positions=[2],widths=[0.5],flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax.set_ylabel('$a$')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['low $r_{\\gamma^+}^%d$'%ds,'low$^* r_{\\gamma^+}^%d$'%ds,'high $r_{\\gamma^+}^%d$'%ds],rotation=60,fontsize=8)
        pl_dat.remove_frame(ax,['top','right'])


        ax = plt.axes([0.8,0.5,0.075,0.15])
        nPF_low = cluster.fields['nModes'][c_low_low0,s_low_low0]
        nPF_high_low = cluster.fields['nModes'][c_high_low0,s_high_low0]
        nPF_high = cluster.fields['nModes'][c_high_low0,s_high_low0]
        ax.bar([0.8,1.8],[(nPF_low==1).sum(),(nPF_low>=2).sum()],width=0.2)
        ax.bar([1.,2.],[(nPF_high_low==1).sum(),(nPF_high_low>=2).sum()],width=0.2)
        ax.bar([1.2,2.2],[(nPF_high==1).sum(),(nPF_high>=2).sum()],width=0.2)
        # ax.set_xticks([0,1,2])
        # ax.set_xticklabels(['low $r_{\\gamma^+}^%d$'%ds,'low^* $r_{\\gamma^+}^%d$'%ds,'high $r_{\\gamma^+}^%d$'%ds],rotation=60,fontsize=8)
        pl_dat.remove_frame(ax,['top','right'])


        # ax.plot(r_stab[c_all,s_all],cluster.stats['MI_value'][c_all,s_all],'k.',markeredgewidth=0,markersize=1)
        act_clusters = np.any(cluster.status['activity'][:,s_bool,1],1)

        # print(r_stab < r_stab_thr[1])
        hist_r_stab = np.zeros(120)
        hist_act_stab = np.zeros(120)
        for c in np.where(act_clusters)[0]:
            ct_r = Counter(np.diff(np.where(r_stab[c,s_bool] < r_stab_thr[1])[0])-1)
            ct_act = Counter(np.diff(np.where(act_stab[c,s_bool] < act_stab_thr[1])[0])-1)
            for key,i in ct_r.items():
                hist_r_stab[key] += i
            for key,i in ct_act.items():
                hist_act_stab[key] += i
        # ax = plt.axes([0.1,0.1,0.15,0.1])
        # ax.hist((r_stab > r_stab_thr[1]).sum(1),np.linspace(1,50,50))
        # print(hist_r_stab)

        hist_r_stab[:1] = 0
        hist_act_stab[:1] = 0

        ax = plt.axes([0.55,0.1,0.15,0.15])
        ax.bar(np.linspace(0,50-1,50),hist_r_stab[:50],width=1,color='tab:red')
        ax.set_ylim([0,500])
        pl_dat.remove_frame(ax,['top','right'])
        ax = plt.axes([0.8,0.1,0.15,0.15])
        ax.bar(np.linspace(0,50-1,50),hist_act_stab[:50],width=1,color='tab:red')
        # ax.hist((act_stab > act_stab_thr[1]).sum(1),np.linspace(1,50,50))
        ax.set_ylim([0,500])
        pl_dat.remove_frame(ax,['top','right'])

        p_r = hist_r_stab/hist_r_stab.sum()
        p_act = hist_act_stab/hist_act_stab.sum()
        print(np.dot(hist_r_stab/hist_r_stab.sum(),np.arange(len(hist_r_stab))))
        print(np.dot(hist_act_stab/hist_act_stab.sum(),np.arange(len(hist_act_stab))))

        # print(p_r)
        # print(np.arange(len(hist_r_stab))[np.where(p_r>0.95)[0]])
        # print(np.arange(len(hist_act_stab))[np.where(p_act>0.95)[0]])

        r_high_ct = (r_stab > r_stab_thr[1]).sum(1)
        act_high_ct = (act_stab > act_stab_thr[1]).sum(1)

        status_Lr = np.zeros((nC,nSes))
        for c in range(nC):
            s0_act = 0
            inAct = False
            for s in np.where(cluster.status['sessions'])[0]:
                if inAct:
                    if (r_stab[c,s] <= r_stab_thr[1]):
                        Lr = cluster.status['sessions'][s0_act:s].sum()
                        status_Lr[c,s0_act:s] = Lr
                        inAct=False
                else:
                    if (r_stab[c,s] > r_stab_thr[1]):
                        s0_act = s
                        inAct = True
            if inAct:
                Lr = cluster.status['sessions'][s0_act:s+1].sum()
                status_Lr[c,s0_act:s+1] = Lr

        mean_Lr = np.zeros((nSes,2))
        for s in range(nSes):
            # print('session: %d'%s)
            # print(status_Lr[status_Lr[:,s]>0,s])
            mean_Lr[s,0] = status_Lr[status_Lr[:,s]>0,s].mean()
            mean_Lr[s,1] = status_Lr[status_Lr[:,s]>0,s].std()

        ax = plt.axes([0.1,0.1,0.2,0.1])
        pl_dat.plot_with_confidence(ax,np.arange(nSes),mean_Lr[:,0],mean_Lr[:,1],col='k')


        ax = plt.axes([0.3,0.1,0.2,0.1])
        ax.hist(np.where(status_Lr>=1)[1],np.linspace(0,120,121))



        plt.show(block=False)

        return status_Lr



    if plot_fig[47]:


        s_bool = np.zeros(nSes,'bool')
        s_bool[17:87] = True
        # s_bool[:] = True
        s_bool[~cluster.status['sessions']] = False

        plt.figure(figsize=(7,3),dpi=300)

        ax_height=0.175
        ### plot distributions of p_post_c
        cluster_active = np.any(cluster.status['activity'][:,s_bool,1],1)
        ax = plt.axes([0.05,0.75,0.075,ax_height])
        ax.hist(cluster.stats['p_post_c']['code']['act'][cluster_active,1,1],np.linspace(0,1,21),density=True,orientation='horizontal')
        ax.set_ylabel('$p(\\alpha_{s+1}^+|\\alpha^+)$',fontsize=10)
        ax.set_xticks([])
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.set_xlim(np.flipud(ax.get_xlim()))
        pl_dat.remove_frame(ax,['top','left','bottom'])


        ax = plt.axes([0.05,0.45,0.075,ax_height])
        ax.hist(cluster.stats['p_post_c']['stable']['code'][cluster_active,1,1],np.linspace(0,1,21),density=True,orientation='horizontal')
        ax.set_ylabel('$p(\\beta_{s+1}^+|\\beta^+)$',fontsize=10)
        ax.set_xticks([])
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.set_xlim(np.flipud(ax.get_xlim()))
        pl_dat.remove_frame(ax,['top','left','bottom'])

        # print(np.isfinite(cluster.stats['p_post_c']['code']['act'][cluster_active,1,1]).sum())
        # print(np.isfinite(cluster.stats['p_post_c']['stable']['code'][cluster_active,1,1]).sum())
        # print(np.isfinite(cluster.stats['p_post_c']['code']['stable'][cluster_active,1,0]).sum())
        ax = plt.axes([0.05,0.15,0.075,ax_height])
        ax.hist(cluster.stats['p_post_c']['stable']['stable'][cluster_active,1,1],np.linspace(0,1,21),density=True,orientation='horizontal')
        ax.set_ylabel('$p(\\gamma_{s+1}^+|\\gamma^+)$',fontsize=10)
        ax.set_xticks([])
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.set_xlim(np.flipud(ax.get_xlim()))
        pl_dat.remove_frame(ax,['top','left','bottom'])


        ax = plt.axes([0.25,0.75,0.2,ax_height])
        ax.plot(gauss_smooth(cluster.stats['p_post_s']['code']['act'][:,1,1],1,mode='constant'))
        ax.set_ylim([0,1])
        # ax.set_xlabel('session')
        # ax.set_ylabel('prob.')
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.25,0.45,0.2,ax_height])
        ax.plot(gauss_smooth(cluster.stats['p_post_s']['stable']['code'][:,1,1],1,mode='constant'))
        ax.set_ylim([0,1])
        # ax.set_xlabel('session')
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.25,0.15,0.2,ax_height])
        ax.plot(gauss_smooth(cluster.stats['p_post_s']['stable']['stable'][:,1,1],1,mode='constant'))
        ax.set_ylim([0,1])
        ax.set_xlabel('session')
        pl_dat.remove_frame(ax,['top','right'])






        ### plot model, fed with according probabilities
        state_label = 'beta'
        status_act,status_PC = get_stats_from_cluster(cluster)
        nC_good,nSes_good = status_act.shape
        status_test = get_model_from_paras(nC_good,nSes_good,cluster=cluster,model='drawing')

        status_act_test = status_test['act']
        status_PC_test = status_test['code']
        status_stable_test = status_test['stable']
        print(status_act_test.sum(0))

        # plt.figure()
        ax = plt.axes([0.55,0.8,0.25,0.15])
        ax.plot(status_act_test.sum(0),'k.',markersize=2)
        ax.plot(status_PC_test.sum(0),'b.',markersize=2)
        ax.plot(status_stable_test.sum(0),'r.',markersize=2)
        ax.set_ylim([0,1500])
        pl_dat.remove_frame(ax,['top','right'])

        recurr = {'act':    {},
                'PC':   {}}

        labels = ['alpha','beta']
        for j,mode in enumerate(['act','PC']):
            state_label = labels[j]
            status = status_act if mode=='act' else status_PC
            status_dep = np.ones_like(status_act) if mode=='act' else status_act
            recurr[mode]['data'] = get_recurr(status,status_dep)

            # print(recurr)

            # status_act_test,status_PC_test = get_model_from_cluster(cluster)
            status_test = status_act_test if mode=='act' else status_PC_test
            status_dep_test = np.ones_like(status_act_test) if mode=='act' else status_act_test
            recurr[mode]['test'] = get_recurr(status_test,status_dep_test)


            rec_mean = np.nanmean(np.nanmean(recurr[mode]['data'],0))
            rec_var = np.sqrt(np.nansum(np.nanvar(recurr[mode]['data'],0))/(recurr[mode]['data'].shape[1]-1))


            ax = plt.axes([0.55,0.5-0.35*j,0.175,0.175])
            ax.hist(status.sum(1),np.linspace(0,nSes_good,nSes_good+1),alpha=0.5,color='k')
            ax.hist(status_test.sum(1),np.linspace(0,nSes_good,nSes_good+1),alpha=0.5,color='r')
            pl_dat.remove_frame(ax,['top','right'])
            ax.set_xlabel('$N_{\\%s}$'%state_label)
            # ax.set_ylim([0,500])
            # ax = plt.subplot(222)
            # p = status.sum()/(nSes_good*nC_good)

            # ax.plot([0,nSes],[p,p],'k--')
            ax = plt.axes([0.825,0.5-0.35*j,0.15,0.2])
            # ax.text(10,p_base+0.04,'$p^{(0)}_{\\%s^+}$'%(state_label),fontsize=8)
            SD = 1
            ax.plot([1,nSes_good],[rec_mean,rec_mean],'k--',linewidth=0.5)

            pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good-1,nSes_good),np.nanmean(recurr[mode]['data'],0),SD*np.nanstd(recurr[mode]['data'],0),col='k',ls='-',label='emp. data')
            pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good-1,nSes_good),np.nanmean(recurr[mode]['test'],0),SD*np.nanstd(recurr[mode]['test'],0),col='tab:red',ls='-',label='model')
            ax.set_ylim([0,1])
            ax.set_ylabel('$p(\\%s^+_{s+\Delta s} | \\%s^+_s)$'%(state_label,state_label),fontsize=8)#'p(recurr.)')
            ax.set_xlim([0,40])#nSes_good])
            pl_dat.remove_frame(ax,['top','right'])

            if j == 0:
                ax.legend(fontsize=8,handlelength=1,loc='upper right',bbox_to_anchor=[1.1,1.7])
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('$\Delta$ sessions')

        plt.show(block=False)

        if sv:
            pl_dat.save_fig('popDyn_individual')



   



    if plot_fig[61]:

        print('### check within session difference in activity / coding / place field position ###')

        # fr_thr = 0.1
        s_bool = np.zeros(nSes,'bool')
        # s_bool[17:87] = True
        s_bool[:] = True
        s_bool[~cluster.status['sessions']] = False

        t_ct_max = max(cluster.behavior['trial_ct'])

        nsteps = 4
        perc_act = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_coding = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_act_overlap = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_coding_overlap = np.zeros((nsteps,nSes,t_ct_max))*np.NaN

        coding_overlap = np.zeros((nSes,2,2))*np.NaN

        field_var = np.zeros((nSes,2))*np.NaN

        coding_fit = np.zeros((nsteps,nSes,2,3))*np.NaN
        coding_overlap_fit = np.zeros((nsteps,nSes,2,3))*np.NaN

        fig = plt.figure(figsize=(7,5),dpi=300)

        ax = plt.axes([0.1,0.775,0.2,0.15])
        pl_dat.add_number(fig,ax,order=1)
        s = 40
        pathMouse = '/media/wollex/Analyze_AS1/linstop/246'
        pathSession = os.path.join(pathMouse,'Session%02d'%(s+1))
        data = define_active(pathSession)
        pathLoad = os.path.join(pathSession,'results_redetect.mat')
        ld = loadmat(pathLoad)
        n = np.random.choice(np.where(cluster.status['activity'][:,s,2][...,np.newaxis] & (cluster.fields['reliability'][:,s,:]>0.4) & (cluster.fields['Bayes_factor'][:,s,:]>100))[0])
        n = 459
        S = ld['S'][n,:]
        baseline = np.median(S[S>0])
        S = np.floor(S/(baseline+1*np.std(S[S<baseline]-baseline)))
        spikes = S[S>0]
        spikes -= spikes.min()
        spikes /= spikes.max()
        spikes *= 5
        ax.plot(data['time'],data['position'],'k.',markersize=1,markeredgecolor='none')
        ax.scatter(data['time'][np.where(S>0)[0]],data['position'][np.where(S>0)[0]],s=spikes,c='r')#,markeredgecolor='none')
        ax.set_xlim([0,600])
        ax.set_xlabel('time')
        ax.set_ylabel('position')
        pl_dat.remove_frame(ax,['top','right'])
        # plt.show(block=False)
        # return
        ax_fit_overlap = plt.axes([0.675,0.125,0.135,0.2])
        pl_dat.add_number(fig,ax_fit_overlap,order=9)
        ax_fit_coact = plt.axes([0.675,0.425,0.135,0.2])
        pl_dat.add_number(fig,ax_fit_coact,order=6)
        j=0
        color_t = plt.cm.rainbow(np.linspace(0,1,nsteps))
        for s in tqdm(np.where(s_bool)[0]):

            t_ct = cluster.behavior['trial_ct'][s]
            trial_coding = np.zeros((t_ct,nC,5),'bool')
            nPF = cluster.status_fields[:,s,:].sum()

            fields = np.zeros((t_ct,nbin))
            for t in range(t_ct):
                trial_coding[t,:,:] = cluster.fields['trial_act'][:,s,:,t] & cluster.status_fields[:,s,:]
                fields[t,:] = cluster.fields['p_x'][np.where(trial_coding[t,...])[0],s,np.where(trial_coding[t,...])[1],:].sum(0)

            fields /= fields.sum(1)[:,np.newaxis]
            fields_cs = np.cumsum(fields,1)
            fields_cs_mean = fields_cs.mean(0)

            field_diff = np.max(fields_cs-fields_cs_mean[np.newaxis,:],1)
            field_var[s,0] = np.mean(field_diff)
            field_var[s,1] = np.std(field_diff)

            dt = 5
            if ((s+1)<nSes):
                if s_bool[s+1]:
                    t_start = min(cluster.behavior['trial_ct'][s],dt)
                    t_end = max(0,cluster.behavior['trial_ct'][s]-dt)
                    coding_s1_start = np.any(cluster.fields['trial_act'][:,s,:,:t_start],-1) & cluster.status_fields[:,s,:]
                    coding_s1_end = np.any(cluster.fields['trial_act'][:,s,:,t_end:],-1) & cluster.status_fields[:,s,:]

                    ### get first dt trials and last dt trials
                    t_start = min(cluster.behavior['trial_ct'][s+1],dt)
                    t_end = max(0,cluster.behavior['trial_ct'][s+1]-dt)
                    coding_s2_start = np.any(cluster.fields['trial_act'][:,s+1,:,:t_start],-1) & cluster.status_fields[:,s+1,:]
                    coding_s2_end = np.any(cluster.fields['trial_act'][:,s+1,:,t_end:],-1) & cluster.status_fields[:,s+1,:]

                    coding_overlap[s,0,0] = coding_s2_start[coding_s1_start].sum()/coding_s1_start.sum()
                    coding_overlap[s,0,1] = coding_s2_end[coding_s1_start].sum()/coding_s1_start.sum()
                    coding_overlap[s,1,0] = coding_s2_start[coding_s1_end].sum()/coding_s1_end.sum()
                    coding_overlap[s,1,1] = coding_s2_end[coding_s1_end].sum()/coding_s1_end.sum()

            if (s in [19,64]):
                ax = plt.axes([0.425+0.125*j,0.775,0.075,0.15])
                im = ax.imshow(fields,clim=[0,0.03],aspect='auto',cmap='hot')
                ax.set_xlim([0,100])
                ax.set_xlabel('position')
                if j==0:
                    ax.set_ylabel('trial')
                    pl_dat.add_number(fig,ax,order=2)
                else:
                    cbaxes = plt.axes([0.63,0.775,0.01,0.15])
                    h_cb = plt.colorbar(im,cax=cbaxes)
                    h_cb.set_label('PF density',fontsize=8)
                    h_cb.set_ticks([])
                j+=1
                ax.set_title('Session %d'%(s+1),fontsize=8)
            if s==50:
                ax = plt.axes([0.8,0.775,0.18,0.125])
                pl_dat.add_number(fig,ax,order=3)
                ax.plot(fields_cs.T,color=[0.6,0.6,0.6],linewidth=0.5)
                ax.plot(fields_cs_mean,color='k',linewidth=1)
                pl_dat.remove_frame(ax,['top','right'])
                ax.set_xlabel('position')
                ax.set_ylabel('cdf(density)',fontsize=8)

            for dt in range(nsteps):
                col = color_t[dt]

                # nAct = cluster.status['activity'][:,s,1].sum(0)
                # if (dt==0) & (s in [10,50]):
                trial_coding = np.zeros((t_ct,nC,5),'bool')
                for t in range(t_ct):
                    t_min = max(0,t-dt)
                    t_max = min(t_ct,t+dt)+1
                    trial_coding[t,:,:] = np.any(cluster.fields['trial_act'][:,s,:,t_min:t_max],-1) & cluster.status_fields[:,s,:]


                # perc_act[s,:t_ct] = trial_act.sum(1)/nAct
                perc_coding[dt,s,:t_ct-dt] = (trial_coding.sum(axis=(1,2))/nPF)[:t_ct-dt]

                # dt_act_overlap = np.zeros((t_ct,t_ct))*np.NaN
                dt_coding_overlap = np.zeros((t_ct,t_ct))*np.NaN
                for t in range(t_ct):
                    # dt_act_overlap[t,:t_ct-t] = trial_act[t:,trial_act[t,:]].sum(1)/trial_act[t:,:].sum(1)
                    dt_coding_overlap[t,:t_ct-t] = trial_coding[t:,trial_coding[t,:,:]].sum(-1)/trial_coding[t:,...].sum(axis=(1,2))
                # perc_act_overlap[s,:t_ct] = np.nanmean(dt_act_overlap[:t_ct-dt,:],0)
                perc_coding_overlap[dt,s,:t_ct] = np.nanmean(dt_coding_overlap[:t_ct-dt,:],0)

                if t_ct > (dt+2):
                    X = np.vstack([np.arange(dt+1,t_ct-dt),np.ones(t_ct-(2*dt+1))]).T

                    Y = perc_coding[dt,s,dt+1:t_ct-dt]
                    coding_fit[dt,s,:,0] = np.linalg.lstsq(X,Y)[0]
                    CI = get_CI(coding_fit[dt,s,:,0],X,Y)
                    coding_fit[dt,s,:,1] = coding_fit[dt,s,:,0]-CI
                    coding_fit[dt,s,:,2] = coding_fit[dt,s,:,0]+CI

                    Y = perc_coding_overlap[dt,s,dt+1:t_ct-dt]
                    coding_overlap_fit[dt,s,:,0] = np.linalg.lstsq(X,Y)[0]
                    CI = get_CI(coding_overlap_fit[dt,s,:,0],X,Y)
                    coding_overlap_fit[dt,s,:,1] = coding_overlap_fit[dt,s,:,0]-CI
                    coding_overlap_fit[dt,s,:,2] = coding_overlap_fit[dt,s,:,0]+CI
                # print(sstats.linregress(np.arange(dt+1,t_ct),perc_coding[dt,s,dt+1:t_ct]))
                # coding_fit[s,0],coding_fit[s,1],_,_,coding_fits[s,2] = sstats.linregress(np.arange(dt+1,t_ct),perc_coding[dt,s,dt+1:t_ct])
                # coding_overlap_fit[s,:] = sstats.linregress(np.arange(dt+1,t_ct),perc_coding_overlap[dt,s,dt+1:t_ct])[0]

        ax = plt.axes([0.84,0.92,0.1,0.05])
        pl_dat.plot_with_confidence(ax,np.arange(nSes),field_var[:,0],field_var[:,1],col='k',lw=0.5)
        pl_dat.remove_frame(ax,['top','right'])
        ax.set_ylabel('$D_{KS}$',fontsize=8)
        ax.set_xlabel('session',fontsize=8)
        ax.xaxis.set_label_coords(0.4,-0.2)
        ax.set_ylim([0.,0.25])

        for dt in [0,3]:
            col = color_t[dt]
            ax_nPC = plt.axes([0.1,0.425,0.175,0.2])
            pl_dat.add_number(fig,ax_nPC,order=4)
            ax_nPC.plot(perc_coding[dt,...].T,linewidth=0.1,color=[0.6,0.6,0.6])
            pl_dat.plot_with_confidence(ax_nPC,range(t_ct_max),np.nanmean(perc_coding[dt,...],0),np.nanstd(perc_coding[dt,...],0),col=col)
            # ax_nPC.plot([dt,dt],[0,1],'k--')
            ax_nPC.set_ylim([0,1])
            ax_nPC.set_ylabel('% PC active')
            ax_nPC.set_xlabel('trial')

            ax_overlap = plt.axes([0.1,0.125,0.175,0.2])
            pl_dat.add_number(fig,ax_overlap,order=7)
            ax_overlap.plot(perc_coding_overlap[dt,...].T,linewidth=0.1,color=[0.6,0.6,0.6])
            pl_dat.plot_with_confidence(ax_overlap,range(t_ct_max),np.nanmean(perc_coding_overlap[dt,...],0),np.nanstd(perc_coding_overlap[dt,...],0),col=col)
            ax_overlap.set_ylim([0,1])
            ax_overlap.set_ylabel('overlap')


        dt = 3
        col = color_t[dt]
        ax = plt.axes([0.4,0.425,0.125,0.075])
        ax.plot([0,nSes],[0,0],'k--',linewidth=0.5)
        pl_dat.plot_with_confidence(ax,range(nSes),coding_fit[dt,:,0,0],coding_fit[dt,:,0,1:].T,col='k')
        # ax.plot(gauss_smooth(coding_fit[:,0],1,mode='constant'),'k')
        ax.set_xlim([0,nSes])
        ax.set_ylim([-0.05,0.01])
        ax.set_ylabel('m')
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.4,0.55,0.125,0.075])
        pl_dat.add_number(fig,ax,order=5)
        ax.plot([0,nSes],[1,1],'k--',linewidth=0.5)
        pl_dat.plot_with_confidence(ax,range(nSes),coding_fit[dt,:,1,0],coding_fit[dt,:,1,1:].T,col='k')
        # ax.plot(gauss_smooth(coding_fit[:,1],1,mode='constant'),'r')
        ax.set_xlim([0,nSes])
        ax.set_ylim([0.5,1.1])
        ax.set_xticklabels([])
        ax.set_ylabel('b')
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.4,0.125,0.125,0.075])
        ax.plot([0,nSes],[0,0],'k--',linewidth=0.5)
        pl_dat.plot_with_confidence(ax,range(nSes),coding_overlap_fit[dt,:,0,0],coding_overlap_fit[dt,:,0,1:].T,col='k')
        # ax.plot(gauss_smooth(coding_overlap_fit[:,0],1,mode='constant'),'k')
        ax.set_xlim([0,nSes])
        ax.set_ylim([-0.05,0.01])
        ax.set_ylabel('m')
        ax.set_xlabel('session')
        pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.4,0.25,0.125,0.075])
        pl_dat.add_number(fig,ax,order=8)
        ax.plot([0,nSes],[1,1],'k--',linewidth=0.5)
        pl_dat.plot_with_confidence(ax,range(nSes),coding_overlap_fit[dt,:,1,0],coding_overlap_fit[dt,:,1,1:].T,col='k')
        # ax.plot(gauss_smooth(coding_overlap_fit[:,1],1,mode='constant'),'r')
        ax.set_xlim([0,nSes])
        ax.set_ylim([0.5,1.1])
        ax.set_xticklabels([])
        ax.set_ylabel('b')
        pl_dat.remove_frame(ax,['top','right'])

        for dt in range(nsteps):
            col = color_t[dt]
            # ax_fit_coact.errorbar(dt-0.2+0.01*np.random.rand(nSes),coding_fit[dt,:,0,0],np.abs(coding_fit[dt,:,0,0][:,np.newaxis]-coding_fit[dt,:,0,1:]).T,fmt='k.',markersize=0.5,linewidth=0.3)
            ax_fit_coact.errorbar(coding_fit[dt,:,0,0],coding_fit[dt,:,1,0],xerr=np.abs(coding_fit[dt,:,0,0][:,np.newaxis]-coding_fit[dt,:,0,1:]).T,yerr=np.abs(coding_fit[dt,:,1,0][:,np.newaxis]-coding_fit[dt,:,1,1:]).T,fmt='.',color=col,ecolor=[0.6,0.6,0.6],markersize=2,linewidth=0.1,markeredgewidth=0)

            ax_fit_overlap.plot(0,np.NaN,'.',color=col,markersize=2,label='$\Delta t = %d$'%dt)
            ax_fit_overlap.errorbar(coding_overlap_fit[dt,:,0,0],coding_overlap_fit[dt,:,1,0],xerr=np.abs(coding_overlap_fit[dt,:,0,0][:,np.newaxis]-coding_overlap_fit[dt,:,0,1:]).T,yerr=np.abs(coding_overlap_fit[dt,:,1,0][:,np.newaxis]-coding_overlap_fit[dt,:,1,1:]).T,fmt='.',color=col,ecolor=[0.6,0.6,0.6],markersize=2,linewidth=0.1,markeredgewidth=0)

        for axx in [ax_fit_overlap,ax_fit_coact]:
            axx.plot([0,0],[0,1.2],'k--',linewidth=0.5,zorder=0)
            axx.set_xlim([-0.03,0.02])
            axx.set_ylim([0,1.2])
            pl_dat.remove_frame(axx,['top','right'])
            axx.set_ylabel('intercept b')
        ax_fit_overlap.legend(fontsize=6,loc='upper right',bbox_to_anchor=[1.4,1.25])
        ax_fit_overlap.set_xlabel('slope m')
        ax_overlap.set_xlabel('$\Delta$ trial')
        pl_dat.remove_frame(ax_overlap,['top','right'])
        pl_dat.remove_frame(ax_nPC,['top','right'])


        ax = plt.axes([0.925,0.55,0.05,0.075])
        # pl_dat.add_number(fig,ax,order=1,offset=[-150,0])
        pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/others/startEnd.png'

        ax.axis('off')
        im = mpimg.imread(pic_path)
        ax.imshow(im)
        ax.set_xlim([0,im.shape[1]])

        ax_overlap = plt.axes([0.925,0.45,0.05,0.08])
        pl_dat.add_number(fig,ax_overlap,order=10,offset=[-150,125])
        width=0.5
        ax_overlap.plot(-0.25+np.random.rand(nSes)*0.5,coding_overlap[:,0,0],'.',color=[0.6,0.6,0.6],markersize=2,markeredgecolor='none')
        mask = np.isfinite(coding_overlap[:,0,0])
        # ax_overlap.errorbar(0.25,np.nanmean(coding_overlap[:,0,0]),np.nanstd(coding_overlap[:,0,0]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.boxplot(coding_overlap[mask,0,0],positions=[0],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        ax_overlap.plot(0.75+np.random.rand(nSes)*0.5,coding_overlap[:,1,0],'.',color=[0.6,0.6,0.6],markersize=2,markeredgecolor='none')
        # ax_overlap.errorbar(1.25,np.nanmean(coding_overlap[:,1,0]),np.nanstd(coding_overlap[:,1,0]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.boxplot(coding_overlap[mask,1,0],positions=[1],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

        ax = plt.axes([0.925,0.225,0.05,0.075])
        # pl_dat.add_number(fig,ax,order=1,offset=[-150,0])
        pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/others/endStart.png'

        ax.axis('off')
        im = mpimg.imread(pic_path)
        ax.imshow(im)
        ax.set_xlim([0,im.shape[1]])

        mask = np.isfinite(coding_overlap[:,0,1])
        ax_overlap2 = plt.axes([0.925,0.125,0.05,0.08])
        ax_overlap2.plot(-0.25+np.random.rand(nSes)*0.5,coding_overlap[:,1,0],'.',color=[0.6,0.6,0.6],markersize=2,markeredgecolor='none')
        ax_overlap2.boxplot(coding_overlap[mask,1,0],positions=[0],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        # ax_overlap2.errorbar(0.25,np.nanmean(coding_overlap[:,0,1]),np.nanstd(coding_overlap[:,0,1]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap2.plot(0.75+np.random.rand(nSes)*0.5,coding_overlap[:,1,1],'.',color=[0.6,0.6,0.6],markersize=2,markeredgecolor='none')
        ax_overlap2.boxplot(coding_overlap[mask,1,1],positions=[1],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        # ax_overlap2.errorbar(1.25,np.nanmean(coding_overlap[:,1,1]),np.nanstd(coding_overlap[:,1,1]),fmt='r.',markersize=5,linestyle='none')

        res = sstats.kruskal(coding_overlap[:,0,0],coding_overlap[:,1,0],nan_policy='omit')
        print(res)

        res = sstats.kruskal(coding_overlap[:,1,0],coding_overlap[:,1,1],nan_policy='omit')
        print(res)

        for axx in [ax_overlap,ax_overlap2]:
            axx.set_xlim([-0.5,1.5])
            axx.set_ylim([0,0.5])
            pl_dat.remove_frame(axx,['top','right'])
            axx.set_ylabel('overlap')
        ax_overlap.set_xticks([0,1])
        ax_overlap.set_xticklabels(['start-start','end-start'],rotation=60)
        ax_overlap2.set_xticks([0,1])
        ax_overlap2.set_xticklabels(['end-start','end-end'],rotation=60)


        plt.show(block=False)
        # np.where(trial_active)

        if sv:
            pl_dat.save_fig('trial_correlation')

    if plot_fig[62]:

        s_bool = np.zeros(nSes,'bool')
        # s_bool[17:87] = True
        s_bool[:] = True
        s_bool[~cluster.status['sessions']] = False

        t_ct_max = max(cluster.behavior['trial_ct'])

        nsteps = 4
        perc_act = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_coding = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_act_overlap = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_coding_overlap = np.zeros((nsteps,nSes,t_ct_max))*np.NaN

        field_var = np.zeros((nSes,2))*np.NaN

        coding_fit = np.zeros((nsteps,nSes,2,3))*np.NaN
        coding_overlap_fit = np.zeros((nsteps,nSes,2,3))*np.NaN
        coding_overlap = np.zeros((nSes,2,2))*np.NaN

        fig = plt.figure(figsize=(7,5),dpi=300)

        # plt.show(block=False)
        # return
        ax_overlap = plt.axes([0.1,0.125,0.2,0.5])
        # pl_dat.add_number(fig,ax_fit_overlap,order=9)
        # ax_fit_coact = plt.axes([0.725,0.425,0.225,0.2])
        # pl_dat.add_number(fig,ax_fit_coact,order=6)
        j=0
        # color_t = plt.cm.rainbow(np.linspace(0,1,nsteps))

        dt = 5


        for s in tqdm(np.where(s_bool)[0][:-1]):

            if s_bool[s+1]:
                t_start = min(cluster.behavior['trial_ct'][s],dt)
                t_end = max(0,cluster.behavior['trial_ct'][s]-dt)
                coding_s1_start = np.any(cluster.fields['trial_act'][:,s,:,:t_start],-1) & cluster.status_fields[:,s,:]
                coding_s1_end = np.any(cluster.fields['trial_act'][:,s,:,t_end:],-1) & cluster.status_fields[:,s,:]

                ### get first dt trials and last dt trials
                t_start = cluster.behavior['trial_ct'][s+1]#min(cluster.behavior['trial_ct'][s+1],dt)
                t_end = 0#max(0,cluster.behavior['trial_ct'][s+1]-dt)
                coding_s2_start = np.any(cluster.fields['trial_act'][:,s+1,:,:t_start],-1) & cluster.status_fields[:,s+1,:]
                coding_s2_end = np.any(cluster.fields['trial_act'][:,s+1,:,t_end:],-1) & cluster.status_fields[:,s+1,:]

                coding_overlap[s,0,0] = coding_s2_start[coding_s1_start].sum()/coding_s1_start.sum()
                coding_overlap[s,0,1] = coding_s2_end[coding_s1_start].sum()/coding_s1_start.sum()
                coding_overlap[s,1,0] = coding_s2_start[coding_s1_end].sum()/coding_s1_end.sum()
                coding_overlap[s,1,1] = coding_s2_end[coding_s1_end].sum()/coding_s1_end.sum()


        ax_overlap.plot(np.random.rand(nSes)*0.5,coding_overlap[:,0,0],'k.',markersize=1)
        ax_overlap.errorbar(0.25,np.nanmean(coding_overlap[:,0,0]),np.nanstd(coding_overlap[:,0,0]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.plot(1+np.random.rand(nSes)*0.5,coding_overlap[:,1,0],'k.',markersize=1)
        ax_overlap.errorbar(1.25,np.nanmean(coding_overlap[:,1,0]),np.nanstd(coding_overlap[:,1,0]),fmt='r.',markersize=5,linestyle='none')

        ax_overlap.plot(2+np.random.rand(nSes)*0.5,coding_overlap[:,0,1],'k.',markersize=1)
        ax_overlap.errorbar(2.25,np.nanmean(coding_overlap[:,0,1]),np.nanstd(coding_overlap[:,0,1]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.plot(3+np.random.rand(nSes)*0.5,coding_overlap[:,1,1],'k.',markersize=1)
        ax_overlap.errorbar(3.25,np.nanmean(coding_overlap[:,1,1]),np.nanstd(coding_overlap[:,1,1]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.set_ylim([0,0.5])

        res = sstats.kruskal(coding_overlap[:,0,0],coding_overlap[:,1,0],nan_policy='omit')
        print(res)

        res = sstats.kruskal(coding_overlap[:,1,0],coding_overlap[:,1,1],nan_policy='omit')
        print(res)

        plt.show(block=False)


def get_code_stability(cluster,s_bool):

    # nC = cluster.data['nC']
    # nSes = cluster.data['nSes']
    # nbin = cluster.data['nbin']

    # nC,nSes = cluster.status['activity'].shape[:2]
    code_stability = np.zeros((cluster.data['nC'],3))*np.NaN

    for c in np.where(cluster.status['clusters'])[0]:#[:10]

        count_code = cluster.status['activity'][c,s_bool,2].sum()
        count_code_possible = cluster.status['activity'][c,s_bool,1].sum()#s_bool.sum()
        count_code_recurr = 0
        count_code_recurr_possible = 0

        for s in np.where(s_bool)[0][:-1]:

            if cluster.status['sessions'][s+1]:
                count_code_recurr_possible += 1
                if cluster.status['activity'][c,s,2]:
                    count_code_recurr += cluster.status['activity'][c,s+1,2]

        code_stability[c,0] = count_code/count_code_possible if count_code_possible>0 else 0
        code_stability[c,1] = count_code_recurr/count_code_recurr_possible if count_code_recurr_possible>0 else np.NaN
        code_stability[c,2] = code_stability[c,1] - (count_code)/count_code_possible

        # print('--- neuron %d : ---'%c)
        # print(act_stability[c,:])
        # print('counts: %d/%d'%(count_act,count_act_possible))
        # print('counts (recurr): %d/%d'%(count_act_recurr,count_act_recurr_possible))
        # print(cluster.status['activity'][c,s_bool,1])
        # print(cluster.status['sessions'][s_min:s_max])
    return code_stability

def get_code_stability_temp(cluster,status_PC=None,ds=3):

    # nC = cluster.data['nC']
    # nSes = cluster.data['nSes']
    # nbin = cluster.data['nbin']

    if status_PC is None:
        status_PC = cluster.status['activity'][...,2]
    nC,nSes = cluster.status['activity'].shape[:2]
    code_stability = np.zeros((cluster.data['nC'],cluster.data['nSes'],2))*np.NaN

    for c in np.where(cluster.status['clusters'])[0]:#[:10]

        for s in np.where(cluster.status['sessions'])[0][:-1]:
            s_min = max(0,s-ds)
            s_max = min(cluster.data['nSes']-1,s+ds+1)

            count_code = status_PC[c,s_min:s_max].sum()
            count_code_possible = cluster.status['activity'][c,s_min:s_max,1].sum()#cluster.status['sessions'][s_min:s_max].sum()
            count_code_recurr = 0
            count_code_recurr_possible = 0

            for s2 in range(s_min,s_max):
                if cluster.status['sessions'][s2]:
                    if cluster.status['sessions'][s2+1]:
                        count_code_recurr_possible += 1
                        if status_PC[c,s2]:
                            count_code_recurr += status_PC[c,s2+1]

            # if status_PC[c,s]:
            code_stability[c,s,0] = count_code/count_code_possible if count_code_possible>0 else 0
            code_stability[c,s,1] = count_code_recurr/count_code_recurr_possible if count_code_recurr_possible>0 else np.NaN
            # else:
                # code_stability[c,s,:] = 0
            # print('--- neuron %d @ s%d: ---'%(c,s))
            # print(act_stability[c,s,:])
            # print('counts: %d/%d'%(count_act,count_act_possible))
            # print(cluster.status['activity'][c,s_min:s_max,1])
            # print(cluster.status['sessions'][s_min:s_max])
    return code_stability



def get_field_stability_fmap(cluster,smooth=2):

    cluster.data['nC'],nSes = cluster.status['activity'].shape[:2]

    field_stability = np.zeros(cluster.data['nC'])*np.NaN

    for c in np.where(cluster.status['clusters'])[0]:#[:10]

        fmap_corr = np.corrcoef(gauss_smooth(cluster.stats['firingmap'][c,:,:],(0,smooth)))
        fmap_corr = fmap_corr[cluster.status['activity'][c,:,1],:]
        fmap_corr = fmap_corr[:,cluster.status['activity'][c,:,1]]
        np.fill_diagonal(fmap_corr,np.NaN)
        field_stability[c] = np.nanmean(fmap_corr)

    return field_stability


#def set_margins(ax,x_margin,y_margin)
  #pos = get(ax, 'Position');
  #pos(1) = pos(1)-x_margin;
  #pos(3) = pos(3)+2*x_margin;
  #pos(2) = pos(2)-y_margin;
  #pos(4) = pos(4)+2*y_margin;
  #set(ax, 'Position',pos)
#end


#function [p_out] = nchoosek_sum(n,k,p)

  #p_out = 0;
  #for i=k:n
    #p_out = p_out + nchoosek(n,i)*p^i*(1-p)^i;
  #end
#end


#function [p_tmp] = expfit_jackknife(X,Y,W)
  ### jackknifing an exponential fit
  #if nargin < 3
    #W = ones(size(Y));
  #end
  #mask = Y>0;
  #Y = log(Y);

  #N_data = sum(mask);
  #p_tmp = zeros(N_data,2);
  #j=1;
  #for i=find(mask)
    #mask_ampl = mask;
    #mask_ampl(i) = false;

    #p_tmp(j,:) = lscov(X(mask_ampl,:),Y(mask_ampl),W(mask_ampl));
    #j=j+1;
  #end

#end


#function data_out = hsm(data)
  #### adapted from python version of caiman
  #### Robust estimator of the mode of a data set using the half-sample mode.
  #### versionadded: 1.0.3

  #### Create the function that we can use for the half-sample mode
  #### needs input of sorted data

  #Ndat = length(data);
  #if Ndat == 1
      #data_out = data(1);
  #elseif Ndat == 2
      #data_out = mean(data);
  #elseif Ndat == 3
      #i1 = data(2) - data(1);
      #i2 = data(3) - data(2);
      #if i1 < i2
          #data_out = mean(data(1:2));
      #elseif i2 > i1
          #data_out = mean(data(2:end));
      #else
          #data_out = data(2);
      #end
  #else

      #wMin = inf;
      #N = floor(Ndat/2) + mod(Ndat,2);
      #for i = 1:N
          #w = data(i+N-1) - data(i);
          #if w < wMin
              #wMin = w;
              #j = i;
          #end
      #end
      #data_out = hsm(data(j:j+N-1));
  #end
#end


#function [spikeNr,md,sd_r] = get_spikeNr(data,time)
  #md = hsm(sort(data));       # Find the mode

  ## only consider values under the mode to determine the noise standard deviation
  #ff1 = data - md;
  #ff1 = -ff1 .* (ff1 < 0);

  ## compute 25 percentile
  #ff1 = sort(ff1);
  #ff1(ff1==0) = NaN;
  #Ns = round(sum(ff1>0) * .5);

  ## approximate standard deviation as iqr/1.349
  #iqr_h = ff1(end-Ns);
  #sd_r = 2 * iqr_h / 1.349;
  #data_thr = md+2*sd_r;
  #spikeNr = sum(floor(data/data_thr));

#end



#function plot_CI_as_fill(mean,CI,x_arr,ax,color,plt_label)

  #hold(ax,'on')
  ### make sure, arrays are properly sized
  #if size(mean,1) > size(mean,2)
    #mean = mean.T;
  #end
  #if size(CI,1) > size(CI,2)
    #CI = CI.T;
  #end
  #if size(x_arr,1) > 1
    #x_arr = x_arr.T;
  #end

  #if size(CI,1) == 1  ## CI provided as symmetric value (e.g. variance)
    #CI = [mean-CI;mean+CI];
  #end
  #mask = ~isnan(mean);
  #x = [x_arr(mask) fliplr(x_arr(mask))];
  #inBetween = [CI(1,mask), fliplr(CI(2,mask))];

  #fill(ax,x,inBetween,color{2},'FaceAlpha',0.5,'EdgeColor','None','HandleVisibility','off');
  #if isempty(plt_label)
    #plot(ax,x_arr(mask),mean(mask),'-','Color',color{1},'LineWidth',2,'HandleVisibility','off')
  #else
    #plot(ax,x_arr(mask),mean(mask),'-','Color',color{1},'LineWidth',2,'DisplayName',plt_label)
  #end
#end


#function [bs_median, bs_CI] = bootstrapping(data,N_bs,mode)

  #### data should have entries:
  ### 1st dim: data to be bootstrapped
  ### 2nd dim: # independent points to be bootstrapped

  #N_samples = size(data,1);
  #N_data = size(data,2);
  #bs_stats = zeros(N_data,N_bs);  ## mean
  #for ds = 1:N_data
    #N_s = N_samples-ds;
    #bs_samples = randi(N_s,N_s,N_bs);
    #for l=1:N_bs
      #dat_bs = data(bs_samples(:,l),ds);
      #if mode=='mean'
        #bs_stats(ds,l) = nanmean(dat_bs);
      #end
    #end
  #end

  #bs_median = median(bs_stats,2,'omitnan');
  #bs_CI = prctile(bs_stats,[2.5,97.5],2);
#end





# def bootstrap_shifts(fun,shifts,N_bs,nbin):
#
#   L_track = 100
#   N_data = len(shifts)
#   if N_data == 0:
#     return np.zeros(4)*np.NaN,np.zeros(4)*np.NaN,np.zeros((2,nbin))*np.NaN
#
#   samples = np.random.randint(0,N_data,(N_bs,N_data))
#   sample_randval = np.random.rand(N_bs,N_data)
#   shift_distr_bs = np.zeros((N_bs,nbin))
#   par = np.zeros((N_bs,4))*np.NaN
#   for i in range(N_bs):
#     x1 = np.argmin(abs(shifts[samples[i,:],:]-sample_randval[i,:,np.newaxis]),1)-L_track/2
#     shift_distr_bs[i,:] = np.histogram(x1,np.linspace(-(L_track/2+0.5),L_track/2+0.5,nbin+1),density=True)[0]
#     par[i,:],p_cov = fun(x1)
#   #print(par)
#   p = np.nanmean(par,0)
#   p_std = np.nanstd(par,0)
#
#   return p,p_std,shift_distr_bs


def bootstrap_shifts(fun,shifts,N_bs,nbin):

  L_track = 100
  N_data = len(shifts)
  if N_data == 0:
    return np.zeros(4)*np.NaN,np.zeros((2,4))*np.NaN,np.zeros(4)*np.NaN,np.zeros((2,nbin))*np.NaN

  samples = np.random.randint(0,N_data,(N_bs,N_data))
  # sample_randval = np.random.rand(N_bs,N_data)
  shift_distr_bs = np.zeros((N_bs,nbin))
  par = np.zeros((N_bs,4))*np.NaN
  for i in range(N_bs):
    shift_distr_bs[i,:] = shifts[samples[i,:],:].sum(0)
    shift_distr_bs[i,:] /= shift_distr_bs[i,:].sum()
    par[i,:],p_cov = fun(shift_distr_bs[i,:])
  p = np.nanmean(par,0)
  p_CI = np.percentile(par,[2.5,97.5],0)
  p_std = np.nanstd(par,0)

  return p, p_CI, p_std, shift_distr_bs


def get_shift_distr(ds,compare,para):

  nSes,nbin,N_bs,idx_celltype = para
  L_track=100
  p = {'all':{},
        'cont':{},
        'mix':{},
        'discont':{},
        'silent_mix':{},
        'silent':{}}

  s1_shifts,s2_shifts,f1,f2 = np.unravel_index(compare['pointer'].col,(nSes,nSes,5,5))
  #print(idx_celltype)
  Ds = s2_shifts-s1_shifts
  idx_ds = np.where((Ds==ds) & idx_celltype)[0]
  N_data = len(idx_ds)

  idx_shifts = compare['pointer'].data[idx_ds].astype('int')-1
  shifts = compare['shifts'][idx_shifts]
  shifts_distr = compare['shifts_distr'][idx_shifts,:].toarray()
  
  for pop in p.keys():
    if pop == 'all':
      idxes = np.ones(N_data,'bool')
    elif pop=='cont':
      idxes = compare['inter_coding'][idx_ds,1]==1
    elif pop=='mix':
      idxes = ((compare['inter_coding'][idx_ds,1]>0) & (compare['inter_coding'][idx_ds,1]<1)) & (compare['inter_active'][idx_ds,1]==1)
    elif pop=='discont':
      idxes = (compare['inter_coding'][idx_ds,1]==0) & (compare['inter_active'][idx_ds,1]==1)
    elif pop=='silent_mix':
      idxes =(compare['inter_active'][idx_ds,1]>0) & (compare['inter_active'][idx_ds,1]<1)
    elif pop=='silent':
      idxes = compare['inter_active'][idx_ds,1]==0

    # p[pop]['mean'], p[pop]['std'], _ = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[idxes,:],N_bs,nbin)
    p[pop]['mean'], p[pop]['CI'], p[pop]['std'], _ = bootstrap_shifts(fit_shift_model,shifts_distr[idxes,:],N_bs,nbin)
  return p

## fitting functions and options
F_shifts = lambda x,A0,A,sig,theta : A/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-theta)**2/(2*sig**2)) + A0/len(x)     ## gaussian + linear offset
def fit_shift_model(data):
  p_bounds = ([0,0,0,-10],[1,1,50,10])
  # shift_hist = np.histogram(data,np.linspace(-50,50,101),density=True)[0]
  # shift_hist[0] = shift_hist[1]
  # shift_hist /= shift_hist.sum()
  try:
    # return curve_fit(F_shifts,np.linspace(-49.5,49.5,100),shift_hist,bounds=p_bounds)
    return curve_fit(F_shifts,np.linspace(-49.5,49.5,100),data,bounds=p_bounds)
  except:
    return np.zeros(4)*np.NaN, np.NaN

