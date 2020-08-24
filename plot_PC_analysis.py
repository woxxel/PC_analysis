from multiprocessing import get_context

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, rc
from matplotlib.cm import get_cmap
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Arc
from matplotlib.ticker import AutoLocator, MultipleLocator, AutoMinorLocator, LogLocator, ScalarFormatter, MaxNLocator
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.ndimage as spim
import scipy.stats as sstats
from scipy import signal
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit

from collections import Counter
from tqdm import *
import os, time, math, h5py, pickle, random, cv2, itertools
import multiprocessing as mp
import warnings

from utils import get_nPaths, pathcat, periodic_distr_distance, bootstrap_data, get_average, pickleData, z_from_point_normal_plane, KS_test, E_stat_test, gauss_smooth, calculate_img_correlation, get_shift_and_flow, get_reliability, com
from utils_data import set_para

warnings.filterwarnings("ignore")

def plot_PC_analysis(cluster,plot_arr=[0,1],N_bs=10,n_processes=0,reprocess=False,sv=False,sv_ext='png',PC=None,active=None):#,N_bs,s_offset,sv,sv_suffix,sv_ext,arrays,occupancy,ROI_recurr,N_pairs,N_norm,pop_overlap)#,ROI_rec2,ROI_tot2)#pathBase,mouse)

    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    nSes = cluster.meta['nSes']
    nC = cluster.meta['nC']

    nbin = cluster.para['nbin']
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
    pl_dat = plot_dat(cluster.meta['mouse'],pathcat([cluster.meta['pathMouse'],'Figures']),nSes,cluster.para,sv_ext=sv_ext)

    plot_fig = np.zeros(100).astype('bool')

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


    nSes_real = cluster.sessions['bool'].sum()
    # print(np.where(cluster.sessions['bool'])[0])

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


#### ------------------------- plot basic cluster stats ----------------------- ###

    if plot_fig[0]:
        print('### plot cell activity statistics ###')
        s = 25

        # session_bool = np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False)

        mode = 'PC'
        state_label = 'alpha' if (mode=='act') else 'beta'
        status_act = cluster.status[cluster.stats['cluster_bool'],:,1]
        status_act = status_act[:,cluster.sessions['bool']]
        # status_act = status_act[:,session_bool]
        status_PC = cluster.status[cluster.stats['cluster_bool'],:,2]
        status_PC = status_PC[:,cluster.sessions['bool']]
        nC_good,nSes_good = status_act.shape
        nSes_max = np.where(cluster.sessions['bool'])[0][-1]

        p_act = np.count_nonzero(status_act)/(nC_good*nSes_good)
        p_PC = np.count_nonzero(status_PC)/np.count_nonzero(status_act)

        rnd_var_act = np.random.random(status_act.shape)
        rnd_var_PC = np.random.random(status_PC.shape)
        status_act_test = np.zeros(status_act.shape,'bool')
        status_act_test_rnd = np.zeros(status_act.shape,'bool')
        status_PC_test = np.zeros(status_PC.shape,'bool')
        status_PC_test_rnd = np.zeros(status_PC.shape,'bool')
        for c in range(nC_good):
            # status_act_test[c,:] = rnd_var_act[c,:] < (np.count_nonzero(status_act[c,:])/nSes_good)
            nC_act = status_act[c,:].sum()
            status_act_test[c,np.random.choice(nSes_good,nC_act,replace=False)] = True
            status_act_test_rnd[c,:] = rnd_var_act[c,:] < p_act
            # status_PC_test[c,status_act_test[c,:]] = rnd_var_PC[c,status_act_test[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
            # status_PC_test[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
            status_PC_test[c,np.where(status_act[c,:])[0][np.random.choice(nC_act,status_PC[c,:].sum(),replace=False)]] = True
            status_PC_test_rnd[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < p_PC


        status = status_act if mode=='act' else status_PC
        status_test = status_act_test if mode=='act' else status_PC_test

        fig = plt.figure(figsize=(7,5),dpi=300)


        if sv:   ## enable, when saving
            ### plot contours of two adjacent sessions
            # load data from both sessions
            pathLoad = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%s])
            ld = loadmat(pathLoad,variable_names=['A'])
            A1 = ld['A']#.toarray().reshape(cluster.meta['dims'][0],cluster.meta['dims'][1],-1)
            Cn = A1.sum(1).reshape(cluster.meta['dims'])
            Cn -= Cn.min()
            Cn /= Cn.max()

            pathLoad = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
            ld = loadmat(pathLoad,variable_names=['A'])
            A2 = ld['A']

            # adjust to same reference frame
            x_grid, y_grid = np.meshgrid(np.arange(0., cluster.meta['dims'][0]).astype(np.float32), np.arange(0., cluster.meta['dims'][1]).astype(np.float32))
            x_remap = (x_grid - \
            cluster.sessions['shift'][s+1,0] + cluster.sessions['shift'][s,0] + \
            cluster.sessions['flow_field'][s+1,:,:,0] - cluster.sessions['flow_field'][s,:,:,0]).astype('float32')
            y_remap = (y_grid - \
            cluster.sessions['shift'][s+1,1] + cluster.sessions['shift'][s,1] + \
            cluster.sessions['flow_field'][s+1,:,:,1] - cluster.sessions['flow_field'][s,:,:,1]).astype('float32')

            ax_ROI = plt.axes([0.04,0.56,0.275,0.375])
            pl_dat.add_number(fig,ax_ROI,order=1,offset=[-50,50])
            # plot background, based on first sessions
            ax_ROI.imshow(Cn,origin='lower',clim=[0,1],cmap='viridis')

            # plot contours occuring in first and in second session, only, and...
            # plot contours occuring in both sessions (taken from first session)

            twilight = plt.get_cmap('hsv')
            cNorm = colors.Normalize(vmin=0,vmax=100)
            scalarMap = plt.cm.ScalarMappable(norm=cNorm,cmap=twilight)

            if mode=='act':
                idx_s1 = cluster.status[:,s,1] & (~cluster.status[:,s+1,1])
                idx_s2 = cluster.status[:,s+1,1] & (~cluster.status[:,s,1])
                idx_s12 = cluster.status[:,s+1,1] & (cluster.status[:,s,1])
            elif mode=='PC':
                idx_s1 = cluster.status[:,s,2] & (~cluster.status[:,s+1,2])
                idx_s2 = cluster.status[:,s+1,2] & (~cluster.status[:,s,2])
                idx_s12 = cluster.status[:,s+1,2] & (cluster.status[:,s,2])

            n_s1 = cluster.IDs['neuronID'][idx_s1,s,1].astype('int')
            n_s2 = cluster.IDs['neuronID'][idx_s2,s+1,1].astype('int')
            n_s12 = cluster.IDs['neuronID'][idx_s12,s,1].astype('int')

            A_tmp = sp.sparse.hstack([sp.sparse.csc_matrix(cv2.remap(img.reshape(cluster.meta['dims']), x_remap,y_remap, cv2.INTER_CUBIC).reshape(-1,1)) for img in A2[:,n_s2].toarray().T])

            if mode=='act':
                [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['dashed']) for a in A1[:,n_s1].T]
                [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['solid']) for a in A1[:,n_s12].T]
                [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['dotted']) for a in A_tmp.T]
            elif mode=='PC':
                # print(np.where(idx_s1)[0])
                # print(n_s1)
                for c,n in zip(np.where(idx_s1)[0],n_s1):
                    a = A1[:,n]
                    f = np.where(cluster.fields['status'][c,s,:]>2)[0][0]
                    colVal = scalarMap.to_rgba(cluster.fields['location'][c,s,f,0])
                    ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['dashed'])
                # print(np.where(idx_s2)[0])
                # print(n_s2)
                for i,(c,n) in enumerate(zip(np.where(idx_s2)[0],n_s2)):
                    a = A_tmp[:,i]
                    f = np.where(cluster.fields['status'][c,s+1,:]>2)[0][0]
                    colVal = scalarMap.to_rgba(cluster.fields['location'][c,s+1,f,0])
                    ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['dotted'])
                for c,n in zip(np.where(idx_s12)[0],n_s12):
                    a = A1[:,n]
                    f = np.where(cluster.fields['status'][c,s,:]>2)[0][0]
                    colVal = scalarMap.to_rgba(cluster.fields['location'][c,s,f,0])
                    ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['solid'])

            if mode=='PC':
                # cbaxes = plt.axes([0.285,0.75,0.01,0.225])
                cbaxes = plt.axes([0.05,0.535,0.15,0.0125])
                cb = fig.colorbar(scalarMap,cax = cbaxes,orientation='horizontal')
                # cb.set_label('location')
                cb.set_label('location',fontsize=8,rotation='horizontal',ha='left',va='center')#,labelpad=0,y=-0.5)
                # print(cb.ax.__dict__.keys())
                cbaxes.xaxis.set_label_coords(1.075,0.4)

            ax_ROI.plot(np.NaN,np.NaN,'k-',label='$\\alpha_{s_1} \cap \\alpha_{s_2}$')
            ax_ROI.plot(np.NaN,np.NaN,'k--',label='$\\alpha_{s_1}$')
            ax_ROI.plot(np.NaN,np.NaN,'k:',label='$\\alpha_{s_2}$')
            # ax_ROI.legend(fontsize=10,bbox_to_anchor=[1.2,1.1],loc='upper right',handlelength=1)

            sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
            ax_ROI.add_artist(sbar)
            ax_ROI.set_xticks([])
            ax_ROI.set_yticks([])
        # plt.show()
        # return
        ### plot distribution of active sessions/neuron
        ax = plt.axes([0.45,0.65,0.175,0.275])
        pl_dat.add_number(fig,ax,order=2)

        # plt.hist(cluster.status[...,1:3].sum(1),pl_dat.h_edges,color=[[0.6,0.6,0.6],'k'],width=0.4,label=['# sessions active','# sessions coding']);
        ax.plot([nSes_real,nSes_real],[0,100],'r--',zorder=0)
        ax.hist(status.sum(1),pl_dat.h_edges,color='k',width=0.8,label='emp. data');
        # ax.hist(status_test.sum(1),pl_dat.h_edges,color='tab:red',alpha=0.7,width=0.8,label='rnd. data');
        if mode=='act':
            ax.hist((status_act_test_rnd).sum(1),pl_dat.h_edges,color=[0.5,0.5,0.5],alpha=0.7,width=0.8)
        elif mode=='PC':
            ax.hist((status_PC_test_rnd).sum(1),pl_dat.h_edges,color=[0.5,0.5,0.5],alpha=0.7,width=0.8)

        ax.set_xlabel('$N_{\\%s^+}$'%state_label)
        ax.set_ylabel('# neurons')
        # ax.legend(fontsize=10,loc='upper right')
        ax.set_xlim([-0.5,nSes_good+0.5])
        if mode=='act':
            ax.set_ylim([0,200])
        elif mode =='PC':
            ax.set_ylim([0,500])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        ### obtain inter-coding intervals
        ICI = np.zeros((nSes_good,2))    # inter-coding-interval
        IPI = np.zeros((nSes_good,2))    # inter-pause-interval (= coding duration)

        # print(status.shape)
        # return

        t_start = time.time()
        ICI[:,0] = get_ICPI(status,mode='ICI')
        print('time taken: %.2f'%(time.time()-t_start))
        t_start = time.time()
        ICI[:,1] = get_ICPI(status_test,mode='ICI')

        IPI[:,0] = get_ICPI(~status,mode='IPI')
        IPI[:,1] = get_ICPI(~status_test,mode='IPI')

        IPI_bs = np.zeros((nSes_good,N_bs))
        IPI_bs_test = np.zeros((nSes_good,N_bs))
        ICI_bs = np.zeros((nSes_good,N_bs))
        ICI_bs_test = np.zeros((nSes_good,N_bs))
        for i in range(N_bs):
            IPI_bs[:,i] = get_ICPI(~status[np.random.randint(0,nC_good,nC_good),:],mode='IPI')
            IPI_bs_test[:,i] = get_ICPI(~status_test[np.random.randint(0,nC_good,nC_good),:],mode='IPI')

            ICI_bs[:,i] = get_ICPI(status[np.random.randint(0,nC_good,nC_good),:],mode='ICI')
            ICI_bs_test[:,i] = get_ICPI(status_test[np.random.randint(0,nC_good,nC_good),:],mode='ICI')

        pval_IPI = np.zeros(nSes_good)*np.NaN
        pval_ICI = np.zeros(nSes_good)*np.NaN
        for s in range(nSes_good):
            # print('ttest (s=%d)'%(s+1))
            # print(np.nanmean(IPI_bs[s,:]),np.nanstd(IPI_bs[s,:]))
            # print(np.nanmean(IPI_bs_test[s,:]),np.nanstd(IPI_bs_test[s,:]))
            # res = sstats.ttest_ind(IPI_bs[s,:],IPI_bs_test[s,:])
            # print(res)
            res = sstats.ttest_ind_from_stats(np.nanmean(IPI_bs[s,:]),np.nanstd(IPI_bs[s,:]),N_bs,np.nanmean(IPI_bs_test[s,:]),np.nanstd(IPI_bs_test[s,:]),N_bs,equal_var=True)
            pval_IPI[s] = res.pvalue

            res = sstats.ttest_ind_from_stats(np.nanmean(ICI_bs[s,:]),np.nanstd(ICI_bs[s,:]),N_bs,np.nanmean(ICI_bs_test[s,:]),np.nanstd(ICI_bs_test[s,:]),N_bs,equal_var=True)
            pval_ICI[s] = res.pvalue

        print('time taken: %.2f'%(time.time()-t_start))
        IPI_bs[IPI_bs==0] = np.NaN
        # print(IPI_bs)

        ICI[ICI==0] = np.NaN
        IPI[IPI==0] = np.NaN

        ax = plt.axes([0.75,0.11,0.225,0.3])
        pl_dat.add_number(fig,ax,order=6)
        # ax.loglog(IPI[:,0],'k-',label='IPI')
        pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(IPI_bs,1),np.nanstd(IPI_bs,1),col='k',lw=0.5,label='$I_{\\%s^+}$'%state_label)
        pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(IPI_bs_test,1),np.nanstd(IPI_bs_test,1),col='tab:red',lw=0.5)
        pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(ICI_bs,1),np.nanstd(ICI_bs,1),col='k',ls=':',lw=0.5,label='$I_{\\%s^-}$'%state_label)
        pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(ICI_bs_test,1),np.nanstd(ICI_bs_test,1),col='tab:red',ls=':',lw=0.5)
        # ax.loglog(np.nanmean(IPI_bs,1)-np.nanstd(IPI_bs,1),'-',color=[0.1,0.5,0.5])
        # ax.loglog(np.nanmean(IPI_bs,1)+np.nanstd(IPI_bs,1),'-',color=[0.5,0.1,0.5])
        # ax.loglog(np.nanmean(IPI_bs,1),'k-',label='IPI')
        # ax.loglog(IPI.mean(0)+IPI.std(0),'-',color=[0.5,0.5,0.5])
        # ax.loglog(IPI[:,1],'tab:red')
        # ax.loglog(ICI[:,0],'k:',label='ICI')
        # ax.loglog(ICI[:,1],color='tab:red',linestyle=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.9,np.maximum(105,nSes_max)])
        ax.set_ylim([1,10**5])#ax.get_ylim()[1]])
        ax.set_ylabel('# occurence')
        ax.set_xlabel('$\mathcal{L}_{\\%s^+}$ / $\mathcal{L}_{\\%s^-}$ [sessions]'%(state_label,state_label))
        ax.legend(fontsize=10,loc='lower left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # print(np.log10(pval))
        # ax = plt.axes([0.21,0.325,0.1,0.075])
        # ax.plot(np.log10(pval_IPI),'k',linewidth=0.5)
        # ax.plot(np.log10(pval_ICI),'k:',linewidth=0.5)
        # ax.plot([0,nSes_good],[-10,-10],'k--',linewidth=0.3)
        # ax.set_xscale('log')
        # ax.set_xlim([0.9,np.maximum(105,nSes_max)])
        # ax.set_xticks(np.logspace(0,2,3))
        # # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        # ax.set_ylim([-300,0])
        # ax.set_ylabel('$\log_{10}(p_{val})$',fontsize=7,rotation='horizontal',labelpad=-15,y=1.15)#,ha='center',va='center')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        ICI_summed =  ICI*np.arange(nSes_good)[:,np.newaxis]
        IPI_summed =  IPI*np.arange(nSes_good)[:,np.newaxis]
        # ax = plt.axes([0.45,0.325,0.15,0.15])
        # pl_dat.add_number(fig,ax,order=5)
        # ax.plot(IPI_summed[:,0]/np.nansum(IPI_summed[:,0]),'k-',label='$I_{\\%s^+}$'%(state_label))
        # ax.plot(IPI_summed[:,1]/np.nansum(IPI_summed[:,1]),'-',color='tab:red')
        # ax.plot(ICI_summed[:,0]/np.nansum(ICI_summed[:,0]),'k:',label='$I_{\\%s^-}$'%(state_label))
        # ax.plot(ICI_summed[:,1]/np.nansum(ICI_summed[:,1]),':',color='tab:red')
        # ax.set_xscale('log')
        # ax.set_xticklabels([])
        # ax.set_ylabel('$p_{\in \mathcal{L}_{\\%s^+} / \mathcal{L}_{\\%s^-}}$'%(state_label,state_label))
        # # ax.legend(fontsize=10)
        # ax.set_xlim([0.8,np.maximum(105,nSes_max)])
        # ax.set_xticks(np.logspace(0,2,3))
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.plot(IPI_summed[:,1]/np.nansum(IPI_summed[:,1]),'-',color='tab:red')
        # ax.plot(ICI_summed[:,1]/np.nansum(ICI_summed[:,1]),':',color='tab:red')
        # ax.set_yscale('log')

        ax = plt.axes([0.875,0.35,0.1,0.1])
        # ax.plot(IPI*np.arange(nSes)/cluster.status[...,1].sum())

        ax.plot(range(1,nSes_good),np.nancumsum(IPI_summed[1:,0]/np.nansum(IPI_summed[1:,0])),'k-')
        ax.plot(range(1,nSes_good),np.nancumsum(IPI_summed[1:,1]/np.nansum(IPI_summed[1:,1])),'-',color='tab:red')
        ax.plot(range(1,nSes_good),np.nancumsum(ICI_summed[1:,0]/np.nansum(ICI_summed[1:,0])),'k:')
        ax.plot(range(1,nSes_good),np.nancumsum(ICI_summed[1:,1]/np.nansum(ICI_summed[1:,1])),':',color='tab:red')

        # ax.legend(fontsize=10)
        ax.set_xscale('log')
        ax.set_xlim([0.8,np.maximum(105,nSes_max)])
        ax.set_xticks(np.logspace(0,2,3))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(9))
        # print(LogLocator().tick_values(1,100))
        # ax.set_xlabel('$\mathcal{L}_{\\%s^+}$ / $\mathcal{L}_{\\%s^-}$'%(state_label,state_label))
        ax.set_ylabel('$cdf_{\in \mathcal{L}_{\\%s^+} / \mathcal{L}_{\\%s^-}}$'%(state_label,state_label),fontsize=8,rotation='horizontal',labelpad=-15,y=1.15)#,ha='center',va='center')
        # ax.set_xlabel('cont.coding [ses.]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax = plt.subplot(4,4,4)
        # ax.plot(status.sum(1),IPI_stats[:,2],'k.',markersize=0.5)


        print('count here only, if active')

        status_act = cluster.status[cluster.stats['cluster_bool'],:,1]
        status_PC = cluster.status[cluster.stats['cluster_bool'],:,2]
        status = status_act if mode=='act' else status_PC

        status_dep = None if mode=='act' else status_act

        ds = 1
        dp_pos,p_pos = get_dp(status,status_act,status_dep=status_dep,status_session=cluster.sessions['bool'],ds=ds,mode=mode)
        dp_neg,p_neg = get_dp(~status,status_act,status_dep=status_dep,status_session=cluster.sessions['bool'],ds=ds,mode=mode)

        status_dep = None if mode=='act' else status_act_test
        dp_pos_test,p_pos_test = get_dp(status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)
        dp_neg_test,p_neg_test = get_dp(~status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)


        ax = plt.axes([0.1,0.11,0.16,0.225])
        pl_dat.add_number(fig,ax,order=4)
        ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),p_pos+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0,alpha=0.6,label='$\\%s^+_s$'%(state_label))
        ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),p_pos_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0,zorder=1)
        ax.set_yticks(np.linspace(0,1,3))
        ax.set_xlabel('$N_{\\%s^+}$'%(state_label))
        ax.set_ylabel('$\Delta p (\\%s^+_{s+1} | \\%s^+_s)$'%(state_label,state_label))
        pl_dat.remove_frame(ax,['top','right'])



        res = sstats.ks_2samp(dp_pos,dp_pos_test)
        print('IPI')
        print(res)
        print(np.nanmean(dp_pos),np.nanstd(dp_pos))
        print(np.nanpercentile(dp_pos,[2.5,97.5]))
        print(np.nanmean(dp_pos_test),np.nanstd(dp_pos_test))

        res = sstats.ks_2samp(dp_neg,dp_neg_test)
        print('IAI')
        print(res)
        print(np.nanmean(dp_neg),np.nanstd(dp_neg))
        print(np.nanmean(dp_neg_test),np.nanstd(dp_neg_test))

        width=0.75
        ax = plt.axes([0.41,0.35,0.075,0.125])
        pl_dat.add_number(fig,ax,order=5)
        ax.plot([-0.5,1.5],[0,0],'--',color=[0.6,0.6,0.6],linewidth=0.5)
        bp = ax.boxplot(dp_pos[np.isfinite(dp_pos)],positions=[0],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        bp_test = ax.boxplot(dp_pos_test[np.isfinite(dp_pos_test)],positions=[1],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        for element in ['boxes','whiskers','means','medians','caps']:
            plt.setp(bp[element], color='k')
            plt.setp(bp_test[element], color='tab:red')
        # ax.bar(1,np.nanmean(dp_pos_test),facecolor='tab:red')
        # ax.errorbar(1,np.nanmean(dp_pos_test),np.abs(np.nanmean(dp_pos_test)-np.nanpercentile(dp_pos_test,[2.5,97.5]))[:,np.newaxis],ecolor='r')
        pl_dat.remove_frame(ax,['top','right','bottom'])
        ax.set_xticks([])
        ax.set_ylabel('$\left\langle \Delta p \\right \\rangle$')
        ax.set_ylim([-0.25,0.75])

        ax2 = plt.axes([0.41,0.11,0.075,0.225])
        ax2.hist(dp_pos,np.linspace(-1,1,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.hist(dp_pos_test,np.linspace(-1,1,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_pos,np.linspace(0,2,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_pos_test,np.linspace(0,2,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.set_xticks([])# status_dilate = sp.ndimage.morphology.binary_dilation(status,np.ones((1,3),'bool'))
        ax2.set_xlim([0,ax2.get_xlim()[1]*2])
        ax2.set_ylim([-0.5,1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax = ax2.twiny()
        ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),dp_pos+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0,alpha=0.6,label='$\\%s^+_s$'%(state_label))
        ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),dp_pos_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0,zorder=1)

        ax.set_yticks(np.linspace(-1,1,5))
        ax.set_xlim([-20,nSes_good])
        ax.set_ylim([-0.5,1])
        # ax.set_ylim([0,4])

        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_xlabel('$N_{\\%s^+}$'%(state_label),x=1,y=-0.1)
        ax2.set_ylabel('$\Delta p (\\%s^+_{s+1} | \\%s^+_s)$'%(state_label,state_label))#'$p_{\\alpha}^{\pm1}$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(fontsize=10,loc='upper left')

        ax = plt.axes([0.5,0.35,0.075,0.125])
        ax.plot([-0.5,1.5],[0,0],'--',color=[0.6,0.6,0.6],linewidth=0.5)
        bp = ax.boxplot(dp_neg[np.isfinite(dp_neg)],positions=[0],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        bp_test = ax.boxplot(dp_neg_test[np.isfinite(dp_neg_test)],positions=[1],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        for element in ['boxes','whiskers','means','medians','caps']:
            plt.setp(bp[element], color='k')
            plt.setp(bp_test[element], color='tab:red')

        pl_dat.remove_frame(ax,['top','right','bottom'])
        ax.set_xticks([])
        ax.set_ylim([-0.25,0.75])
        ax.set_yticklabels([])

        ax2 = plt.axes([0.5,0.11,0.075,0.225])
        ax2.invert_xaxis()
        ax2.hist(dp_neg,np.linspace(-1,1,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.hist(dp_neg_test,np.linspace(-1,1,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp,np.linspace(0,2,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_test,np.linspace(0,2,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.set_xticks([])
        ax2.set_xlim([ax2.get_xlim()[0]*2,0])
        ax2.set_ylim([-0.5,1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax = ax2.twiny()
        ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),dp_neg+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0,alpha=0.6,label='$\\beta_s$')
        ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),dp_neg_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0,zorder=1)
        # ax.set_ylim([0,1])
        ax.set_xlim([0,nSes_good+20])
        ax.set_yticks([])
        # ax.set_yticks(np.linspace(0,1,3))
        ax.set_ylim([-0.5,1])
        # ax.set_ylim([0,4])
        ax.xaxis.tick_bottom()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('$\Delta p (\\%s^-_{s+1} | \\%s^-_s)$'%(state_label,state_label))
        # ax.set_xlabel('\t # sessions')
        # ax.set_ylabel('$p (\\alpha_{s\pm1} | \\alpha_s)$')#'$p_{\\alpha}^{\pm1}$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)




        # ax.legend(fontsize=10,loc='lower left')


        # ax = plt.axes([0.75,0.625,0.125,0.325])
        # status_dilate = sp.ndimage.morphology.binary_dilation(~status,np.ones((1,3),'bool'))
        # cont_score = 1-(status_dilate&(~status)).sum(1)/(2*status.sum(1))
        # ax = plt.axes([0.75,0.625,0.225,0.325])
        # ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),cont_score+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0)
        #
        # status_dilate_test = sp.ndimage.morphology.binary_dilation(status_test,np.ones((1,3),'bool'))
        # cont_score_test = 1-(status_dilate_test&(~status_test)).sum(1)/(2*status_test.sum(1))
        # ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),cont_score_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0)
        # ax.set_ylim([0,1])
        # ax.set_xlim([0,nSes_max])
        # ax.set_xlabel('# sessions active')
        # ax.set_ylabel('$p (\\alpha_{s\pm1} | \\alpha_s)$')#'$p_{\\alpha}^{\pm1}$')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)






        # ax.plot(ICI_stats[:,0]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,0]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,0],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,0],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,8)
        # ax.plot(ICI_stats[:,1]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,1]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,1],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,1],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,12)
        # ax.plot(ICI_stats[:,2]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,2]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,2],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,2],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,16)
        # ax.plot(ICI_stats[:,3]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,3]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,3],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,3],np.linspace(0,nSes,nSes+1))

        # print(ICI_stats)
        # print(IPI_stats)
        # print(IPI*np.arange(nSes))
        # print((IPI*np.arange(nSes)).sum())
        status_act = status_act[:,cluster.sessions['bool']]
        status_PC = status_PC[:,cluster.sessions['bool']]
        status = status[:,cluster.sessions['bool']]
        recurr = np.zeros((nSes_good,nSes_good))*np.NaN
        N_active = status_act.sum(0)
        # session_bool = np.pad(cluster.sessions['bool'][1:],(0,1),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False)

        for s in range(nSes_good):#np.where(cluster.sessions['bool'])[0]:
            overlap = status[status[:,s],:].sum(0).astype('float')
            N_ref = N_active if mode=='act' else status_act[status_PC[:,s],:].sum(0)
            recurr[s,1:nSes_good-s] = (overlap/N_ref)[s+1:]


        recurr_test = np.zeros((nSes_good,nSes_good))*np.NaN
        N_active_test = status_test.sum(0)
        for s in range(nSes_good):
            # overlap_act_test = status_test[status_test[:,s],:].sum(0).astype('float')
            overlap_test = status_test[status_test[:,s],:].sum(0).astype('float')
            N_ref = N_active_test if mode=='act' else status_act_test[status_PC_test[:,s],:].sum(0)
            recurr_test[s,1:nSes_good-s] = (overlap_test/N_ref)[s+1:]

        # ax = plt.subplot(2,4,8)
        ax = plt.axes([0.775,0.65,0.2,0.275])
        pl_dat.add_number(fig,ax,order=3)

        p = status.sum()/(nSes_good*nC_good)
        ax.plot([0,nSes],[p,p],'k--')
        ax.text(10,p+0.04,'$p^{(0)}_{\\%s^+}$'%(state_label),fontsize=8)
        SD = 1
        pl_dat.plot_with_confidence(ax,np.linspace(1,nSes_good,nSes_good),np.nanmean(recurr,0),SD*np.nanstd(recurr,0),col='k',ls='-',label='emp. data')
        pl_dat.plot_with_confidence(ax,np.linspace(1,nSes_good,nSes_good),np.nanmean(recurr_test,0),SD*np.nanstd(recurr_test,0),col='tab:red',ls='-',label='rnd. data')
        ax.set_ylim([0,1])
        ax.set_xlabel('$\Delta$ sessions')
        ax.set_ylabel('$p(\\%s^+_{s+\Delta s} | \\%s^+_s)$'%(state_label,state_label))#'p(recurr.)')
        ax.set_xlim([0,nSes_good])
        ax.legend(fontsize=10,loc='upper right',bbox_to_anchor=[1.05,1.1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            if mode=='act':
                pl_dat.save_fig('act_dynamics')
            elif mode=='PC':
                pl_dat.save_fig('PC_dynamics')

        steps = min(nSes_good,40)
        dp_pos = np.zeros((steps,2))*np.NaN
        dp_neg = np.zeros((steps,2))*np.NaN
        dp_pos_test = np.zeros((steps,2))*np.NaN
        dp_neg_test = np.zeros((steps,2))*np.NaN
        # print(get_dp(status,status_act,status_dep=status_dep,ds=ds,mode=mode))
        for ds in range(1,steps):
            status_dep = None if mode=='act' else status_act_test

            dp,_ = get_dp(status,status_act,status_dep=status_dep,ds=ds,mode=mode)
            dp_pos[ds,:] = [np.nanmean(dp),np.nanstd(dp)]
            dp_test,_ = get_dp(status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)
            dp_pos_test[ds,:] = [np.nanmean(dp_test),np.nanstd(dp_test)]
            # res = sstats.ttest_ind_from_stats(dp_pos[ds,0],dp_pos[ds,1],nC,dp_pos_test[ds,0],dp_pos_test[ds,1],nC,equal_var=True)

            dp = get_dp(~status,status_act,status_dep=status_dep,ds=ds,mode=mode)
            dp_neg[ds,:] = [np.nanmean(dp),np.nanstd(dp)]
            dp_test = get_dp(~status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)
            dp_neg_test[ds,:] = [np.nanmean(dp_test),np.nanstd(dp_test)]


        plt.figure()
        ax = plt.subplot(211)
        pl_dat.plot_with_confidence(ax,range(steps),dp_pos[:,0],dp_pos[:,1],col='k',ls='-')
        pl_dat.plot_with_confidence(ax,range(steps),dp_pos_test[:,0],dp_pos_test[:,1],col='r',ls='-')
        # plt.plot(dp_pos,'k')
        # plt.plot(dp_pos_test,'r')

        ax = plt.subplot(212)
        pl_dat.plot_with_confidence(ax,range(steps),dp_neg[:,0],dp_neg[:,1],col='k',ls='--')
        pl_dat.plot_with_confidence(ax,range(steps),dp_neg_test[:,0],dp_neg_test[:,1],col='r',ls='--')
        # plt.plot(dp_neg,'k--')
        # plt.plot(dp_neg_test,'r--')
        plt.show(block=False)


        # plt.figure()
        # plt.subplot(121)
        # plt.plot(dp_pos,dp,'k.')
        # plt.subplot(122)
        # plt.plot(dp_pos_test,dp_test,'r.')
        # plt.show(block=False)
        #plt.figure()
        #plt.scatter(numbers[:,0]+0.5*np.random.rand(nC),numbers[:,1]+0.5*np.random.rand(nC),s=10,marker='.')
        #plt.show(block=False)

        # plt.figure()
        # plt.hist(dp_pos,np.linspace(-1,1,201),color='k',histtype='step',cumulative=True,density=True)
        # plt.hist(dp_pos_test,np.linspace(-1,1,201),color='r',histtype='step',cumulative=True,density=True)
        # plt.show(block=False)
        # plt.figure()
        # plt.hist(dp_neg,np.linspace(-1,1,201),color='k',histtype='step',cumulative=True,density=True)
        # plt.hist(dp_neg_test,np.linspace(-1,1,201),color='r',histtype='step',cumulative=True,density=True)
        # plt.show(block=False)



#### ---------------------------- plot matching results --------------------------- ###

    if plot_fig[1]:

        plt.figure(figsize=(4,2))
        ax1 = plt.subplot(111)
        #plt.figure(figsize=(4,3))
        #ax1 = plt.axes([0.15, 0.5, 0.8, 0.45])
        #ax2 = plt.axes([0.15, 0.2, 0.8, 0.25])


        #active_time = np.zeros(nSes)
        #for s in range(nSes):
          #if cluster.sessions['bool'][s]:
            #pathSession = pathcat([cluster.meta['pathMouse'],'Session%02d'%(s+1)]);

            #for file in os.listdir(pathSession):
              #if file.endswith("aligned.mat"):
                #pathBH = os.path.join(pathSession, file)

            #f = h5py.File(pathBH,'r')
            #key_array = ['longrunperiod']

            #dataBH = {}
            #for key in key_array:
              #dataBH[key] = np.squeeze(f.get('alignedData/resampled/%s'%key).value)
            #f.close()

            #active_time[s] = dataBH['longrunperiod'].sum()/len(dataBH['longrunperiod']);

        #ax2.plot(t_ses[cluster.sessions['bool']],active_time[cluster.sessions['bool']],color='k')
        ##ax2.plot(t_measures(1:s_end),active_time,'k')
        #ax2.set_xlim([0,t_ses[-1]])
        #ax2.set_ylim([0,1])
        #ax2.set_xlabel('t [h]',fontsize=14)
        #ax2.set_ylabel('active time',fontsize=14)

        #ax1.plot(t_ses[cluster.sessions['bool']],np.ones(cluster.sessions['bool'].sum())*nC,color='k',linestyle=':',label='# neurons')
        ax1.scatter(t_ses[cluster.sessions['bool']],cluster.status[:,cluster.sessions['bool'],1].sum(0), s=20,color='k',marker='o',facecolor='none',label='# active neurons')
        ax1.set_ylim([0,3000])#nC*1.2])
        ax1.set_xlim([0,t_ses[-1]])
        ax1.legend(loc='upper right')

        ax1.scatter(t_ses[cluster.sessions['bool']],cluster.status[:,cluster.sessions['bool'],2].sum(0),s=20,color='k',marker='o',facecolors='k',label='# place cells')

        ax2 = ax1.twinx()
        ax2.plot(t_ses[cluster.sessions['bool']],cluster.status[:,cluster.sessions['bool'],2].sum(0)/cluster.status[:,cluster.sessions['bool'],1].sum(0),'r')
        ax2.set_ylim([0,0.7])
        ax2.yaxis.label.set_color('red')
        ax2.tick_params(axis='y',colors='red')
        ax2.set_ylabel('fraction PCs')

        ax1.set_xlim([0,t_ses[-1]])
        ax1.set_xlabel('session s',fontsize=14)
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.show(block=False)

        #print(cluster.status[:,cluster.sessions['bool'],2].sum(0)/cluster.status[:,cluster.sessions['bool'],1].sum(0))
        if sv:
            pl_dat.save_fig('neuron_numbers')


    if plot_fig[2]:

        #plt.figure()
        #ax1 = plt.axes([0.2,0.3,0.75,0.65])
        #ax1.plot([0,nSes],[0.75,0.75],color='k',linestyle=':')

        recurrence = {'active': {'all':               np.zeros((nSes,nSes))*np.NaN,
                                 'continuous':        np.zeros((nSes,nSes))*np.NaN,
                                 'overrepresentation':np.zeros((nSes,nSes))*np.NaN},
                      'coding': {'all':               np.zeros((nSes,nSes))*np.NaN,
                                 'ofactive':          np.zeros((nSes,nSes))*np.NaN,
                                 'continuous':        np.zeros((nSes,nSes))*np.NaN,
                                 'overrepresentation':np.zeros((nSes,nSes))*np.NaN}}

        N = {'active': cluster.status[:,:,1].sum(0),
             'coding': cluster.status[:,:,2].sum(0)}
        L=10#00

        #for s in tqdm(range(nSes)):#min(30,nSes)):
        if n_processes>1:
            pool = get_context("spawn").Pool(n_processes)
            res = pool.starmap(get_overlap,zip(range(nSes),itertools.repeat((cluster.status,N,L))))

        for (s,r) in enumerate(res):
            for pop in r.keys():
                for key in r[pop].keys():
                    recurrence[pop][key][s,:] = r[pop][key]

        #for pop in recurrence.keys():
          #for key in recurrence[pop].keys():
            #recurrence[pop][key][:,~cluster.sessions['bool']] = np.NaN
            #recurrence[pop][key][~cluster.sessions['bool'],:] = np.NaN

        #print(recurrence['active']['all'])
        #for s in tqdm(range(nSes)):#min(30,nSes)):

        #recurrence['active']['all'][s,np.where(~cluster.sessions['bool'][s:])[0]] = np.NaN
        #start_recurr = np.zeros(nSes)*np.NaN
        #for s in range(nSes-1):
          #if cluster.sessions['bool'][s] and cluster.sessions['bool'][s+1]:
            #start_recurr[s] = cluster.status[cluster.status[:,s,2],s+1,2].sum()/cluster.status[:,s,2].sum()

        #plt.figure()
        #plt.plot(pl_dat.n_edges,start_recurr)#recurrence['active']['all'][:,1])
        #plt.show(block=False)

        f,axs = plt.subplots(2,2,figsize=(10,4))

        axs[1][0].plot([0,nSes],[0,0],color=[0.8,0.8,0.8],linestyle='--')
        axs[1][1].plot([0,nSes],[0,0],color=[0.8,0.8,0.8],linestyle='--')

        axs[0][0].scatter(pl_dat.n_edges,recurrence['active']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        axs[0][0].scatter(pl_dat.n_edges,recurrence['active']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')

        axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')

        for s in range(nSes):
            axs[0][0].scatter(pl_dat.n_edges,recurrence['active']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
            axs[0][0].scatter(pl_dat.n_edges,recurrence['active']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

            axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
            axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

            axs[1][0].scatter(pl_dat.n_edges,recurrence['active']['overrepresentation'][s,:],5,color=[0.8,0.8,0.8],marker='o')
            axs[1][1].scatter(pl_dat.n_edges,recurrence['coding']['overrepresentation'][s,:],5,color=[0.8,0.8,0.8],marker='o')

        axs[0][0].plot(pl_dat.n_edges,np.nanmean(recurrence['active']['all'],0),color='k')

        axs[0][0].legend(loc='lower right',fontsize=12)

        axs[1][0].plot(pl_dat.n_edges,np.nanmean(recurrence['active']['overrepresentation'],0),color='k')
        axs[0][1].plot(pl_dat.n_edges,np.nanmean(recurrence['coding']['all'],0),color='k')
        axs[1][1].plot(pl_dat.n_edges,np.nanmean(recurrence['coding']['overrepresentation'],0),color='k')

        axs[0][0].set_xticks([])
        axs[0][0].set_title('active cells')
        axs[0][1].set_xticks([])
        axs[0][1].set_title('place cells')
        axs[0][0].set_yticks(np.linspace(0,1,3))
        axs[0][1].set_yticks(np.linspace(0,1,3))

        axs[0][0].set_xlim([0,t_ses[-1]])
        axs[0][1].set_xlim([0,t_ses[-1]])
        axs[1][0].set_xlim([0,t_ses[-1]])
        axs[1][1].set_xlim([0,t_ses[-1]])

        axs[0][0].set_ylim([0,1])
        axs[0][1].set_ylim([0,1])

        axs[1][0].set_ylim([-10,30])
        axs[1][1].set_ylim([-10,30])

        axs[1][0].set_xlabel('session diff. $\Delta$ s',fontsize=14)
        axs[1][1].set_xlabel('session diff. $\Delta$ s',fontsize=14)

        axs[0][0].set_ylabel('fraction',fontsize=14)
        axs[1][0].set_ylabel('overrepr.',fontsize=14)

        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('ROI_stability')

        plt.figure(figsize=(5,2.5))
        ax = plt.subplot(111)
        #ax.scatter(pl_dat.n_edges,recurrence['active']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        #ax.scatter(pl_dat.n_edges,recurrence['active']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
        pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,np.nanmean(recurrence['active']['all'],0),1.96*np.nanstd(recurrence['active']['all'],0),col='k',ls='-',label='recurrence of active cells')
        #for s in range(nSes):
          #ax.scatter(pl_dat.n_edges,recurrence['active']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #ax.scatter(pl_dat.n_edges,recurrence['active']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')
        #ax.plot(pl_dat.n_edges-1,np.nanmean(recurrence['active']['all'],0),color='k')
        ax.legend(loc='top right',fontsize=10)
        ax.set_xlim([0,t_ses[-1]])
        ax.set_ylim([0,1])
        ax.set_ylabel('fraction',fontsize=14)
        ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('ROI_stability_activity')

        plt.figure(figsize=(5,2.5))
        ax = plt.subplot(111)
        #ax.scatter(pl_dat.n_edges,recurrence['coding']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        #ax.scatter(pl_dat.n_edges,recurrence['coding']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
        pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,np.nanmean(recurrence['coding']['ofactive'],0),1.0*np.nanstd(recurrence['coding']['ofactive'],0),col='k',ls='-',label='recurrence of place cells (of active)')
        ax.plot(pl_dat.n_edges-1,np.nanmean(recurrence['coding']['all'],0),'k--',label='recurrence of place cells')
        #for s in range(nSes):
          #ax.scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #ax.scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')
        #ax.plot(pl_dat.n_edges,np.nanmean(recurrence['coding']['all'],0),color='k')
        ax.legend(loc='top right',fontsize=10)
        ax.set_xlim([0,t_ses[-1]])
        ax.set_ylim([0,1])
        ax.set_ylabel('fraction',fontsize=14)
        ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('ROI_stability_PC')



    if plot_fig[3]:
        print('plotting general statistics of PC and nPCs')

        mask_PC = (~cluster.status[...,2])
        mask_active = ~(cluster.status[...,1]&(~cluster.status[...,2]))

        fr_key = 'firingrate'#'firingrate_adapt'
        ### stats of all (PC & nPC) cells
        plt.figure(figsize=(4,3),dpi=300)

        key_arr = ['SNR',fr_key,'Isec_value','MI_value']

        for i,key in enumerate(key_arr):
            ## firingrate
            dat_nPC = np.ma.array(cluster.stats[key], mask=mask_active, fill_value=np.NaN)
            dat_PC = np.ma.array(cluster.stats[key], mask=mask_PC, fill_value=np.NaN)

            dat_PC_mean = np.zeros(nSes)*np.NaN
            dat_PC_CI = np.zeros((2,nSes))*np.NaN
            dat_nPC_mean = np.zeros(nSes)*np.NaN
            dat_nPC_CI = np.zeros((2,nSes))*np.NaN
            for s in np.where(cluster.sessions['bool'])[0]:
                dat_PC_s = dat_PC[:,s].compressed()
                dat_PC_mean[s] = np.mean(dat_PC_s)
                dat_PC_CI[:,s] = np.percentile(dat_PC_s,q=[32.5,67.5])
                dat_nPC_s = dat_nPC[:,s].compressed()
                dat_nPC_mean[s] = np.mean(dat_nPC_s)
                dat_nPC_CI[:,s] = np.percentile(dat_nPC_s,q=[32.5,67.5])

            ax = plt.subplot(2,2,i+1)
            pl_dat.plot_with_confidence(ax,range(nSes),dat_nPC_mean,dat_nPC_CI,col='k',ls='-',label=None)
            pl_dat.plot_with_confidence(ax,range(nSes),dat_PC_mean,dat_PC_CI,col='tab:blue',ls='-',label=None)


            # dat_bs_nPC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_nPC,N_bs)
            # dat_bs_PC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_PC,N_bs)
            # dat_bs_nPC[0][~cluster.sessions['bool']] = np.NaN
            # dat_bs_PC[0][~cluster.sessions['bool']] = np.NaN
            #
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_nPC[0],dat_bs_nPC[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label=None)
            ax.set_ylabel(key)
        ax = plt.subplot(221)
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xticklabels([])

        ax = plt.subplot(222)
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xticklabels([])
        ax.set_ylabel('$\\bar{\\nu}$')

        if nSes > 20:
            s = 20
        else:
            s = 10
        ax = plt.axes([0.8,0.65,0.15,0.075])
        dat_nPC = np.ma.array(cluster.stats[fr_key], mask=mask_active, fill_value=np.NaN)
        dat_PC = np.ma.array(cluster.stats[fr_key], mask=mask_PC, fill_value=np.NaN)
        # ax.hist(dat_nPC[:,s][~mask_active[:,s]],np.linspace(0,0.3,21),density=True,facecolor='k',alpha=0.5)
        # ax.hist(dat_PC[:,s][~mask_PC[:,s]],np.linspace(0,0.3,21),density=True,facecolor='tab:blue',alpha=0.5)
        ax.hist(dat_nPC[:,s][~mask_active[:,s]],np.logspace(-2.5,0,21),density=True,facecolor='k',alpha=0.5)
        ax.hist(dat_PC[:,s][~mask_PC[:,s]],np.logspace(-2.5,0,21),density=True,facecolor='tab:blue',alpha=0.5)
        # ax.set_ylim([0,200])
        # ax.set_xticks()
        ax.set_xscale('log')
        ax.set_xlabel('$\\nu$')
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax = plt.subplot(223)
        # ax.set_ylabel('$r_{value}$')
        ax.set_ylabel('$I/sec$')
        ax.set_xlabel('session')

        ax = plt.subplot(224)
        ax.set_ylabel('MI')
        ax.set_xlabel('session')
        ax.set_ylim([0,ax.get_ylim()[1]])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('neuronStats_nPCvPC')

        # return
        ### stats of PCs
        plt.figure(figsize=(4,3),dpi=300)

        mask_fields = cluster.fields['status']<3

        ax = plt.subplot(2,2,1)
        nPC = cluster.status[...,2].sum(0).astype('float')
        nPC[~cluster.sessions['bool']] = np.NaN
        ax.plot(nPC,'tab:blue')
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_ylabel('# PC')

        ax2 = ax.twinx()
        dat = np.ma.array(cluster.fields['nModes'], mask=mask_PC)
        ax2.plot(dat.mean(0),'k-')
        ax2.set_ylim([1,1.3])

        key_arr = ['width','amplitude','reliability']

        ## field width
        for i,key in enumerate(key_arr):
            print(key)
            if len(cluster.fields[key].shape)==4:
                dat = np.ma.array(cluster.fields[key][...,0], mask=mask_fields, fill_value=np.NaN)
            else:
                dat = np.ma.array(cluster.fields[key], mask=mask_fields, fill_value=np.NaN)

            ax = plt.subplot(2,2,i+2)#axes([0.1,0.6,0.35,0.35])
            dat_mean = np.zeros(nSes)*np.NaN
            dat_CI = np.zeros((4,nSes))*np.NaN
            for s in np.where(cluster.sessions['bool'])[0]:
                dat_s = dat[:,s,:].compressed()
                dat_mean[s] = np.mean(dat_s)
                dat_CI[:,s] = np.percentile(dat_s,q=[2.5,32.5,67.5,97.5])
                # ax.boxplot(dat_s,positions=[s],widths=0.4,whis=[5,95],notch=True,bootstrap=100,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

            # dat_bs = bootstrap_data(lambda x : (np.mean(x,(0,2)),0),dat,N_bs)
            # dat_bs[0][~cluster.sessions['bool']] = np.NaN
            # dat = dat[mask_fields]#[dat.mask] = np.NaN
            # dat[mask_fields] = np.NaN

            # ax.plot(width.mean((0,2)),'k')
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs[0],dat_bs[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat.mean((0,2)),np.percentile(dat.std((0,2))),col='k',ls='-',label=None)

            pl_dat.plot_with_confidence(ax,range(nSes),dat_mean,dat_CI[[0,3],:],col='k',ls='-',label=None)
            pl_dat.plot_with_confidence(ax,range(nSes),dat_mean,dat_CI[[1,2],:],col='k',ls='-',label=None)
            ax.set_ylim([0,ax.get_ylim()[1]])
            ax.set_ylabel(key)

        ax = plt.subplot(222)
        ax.set_ylabel('$\sigma$')
        ax.set_xticklabels([])

        ax = plt.subplot(223)
        ax.set_ylabel('$A$')
        ax.set_xlabel('session')

        ax = plt.subplot(224)
        ax.set_ylabel('reliability')
        ax.set_xlabel('session')

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('neuronStats_PC')


    if plot_fig[4]:
        print('### plot place cell statistics ###')

        fig = plt.figure(figsize=(7,4),dpi=300)

        if nSes > 70:
            s = 70
        else:
            s = 10
        pathLoad = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
        ld = loadmat(pathLoad)
        A = ld['A']#.toarray().reshape(cluster.meta['dims'][0],cluster.meta['dims'][1],-1)
        Cn = A.sum(1).reshape(cluster.meta['dims'])
        Cn -= Cn.min()
        Cn /= Cn.max()

        # adjust to same reference frame
        # x_grid, y_grid = np.meshgrid(np.arange(0., cluster.meta['dims'][0]).astype(np.float32), np.arange(0., cluster.meta['dims'][1]).astype(np.float32))
        # x_remap = (x_grid - \
        #             cluster.sessions['shift'][s+1,0] + cluster.sessions['shift'][s,0] + \
        #             cluster.sessions['flow_field'][s+1,:,:,0] - cluster.sessions['flow_field'][s,:,:,0]).astype('float32')
        # y_remap = (y_grid - \
        #             cluster.sessions['shift'][s+1,1] + cluster.sessions['shift'][s,1] + \
        #             cluster.sessions['flow_field'][s+1,:,:,1] - cluster.sessions['flow_field'][s,:,:,1]).astype('float32')

        ax_ROI = plt.axes([0.01,0.425,0.325,0.55])
        # plot background, based on first sessions
        ax_ROI.imshow(Cn,origin='lower',clim=[0,1],cmap='viridis')

        # plot contours occuring in first and in second session, only, and...
        # plot contours occuring in both sessions (taken from first session)
        idx_act = cluster.status[:,s,1] & (~cluster.status[:,s,2])
        idx_PC = cluster.status[:,s,2]
        c_arr_PC = np.where(idx_PC)[0]

        n_act = cluster.IDs['neuronID'][idx_act,s,1].astype('int')
        n_PC = cluster.IDs['neuronID'][idx_PC,s,1].astype('int')

        twilight = plt.get_cmap('hsv')
        cNorm = colors.Normalize(vmin=0,vmax=100)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm,cmap=twilight)


        if True:   ## enable, when saving
            [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['dotted']) for a in A[:,n_act].T]

            for c,n in zip(np.where(idx_PC)[0],n_PC):
                a = A[:,n]
                f = np.where(cluster.fields['status'][c,s,:]>2)[0][0]
                colVal = scalarMap.to_rgba(cluster.fields['location'][c,s,f,0])
                ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['solid'])

        cbaxes = plt.axes([0.345,0.75,0.01,0.225])
        cb = fig.colorbar(scalarMap,cax = cbaxes,orientation='vertical')

        ax_ROI.plot(np.NaN,np.NaN,'k-',label='PC')
        # ax_ROI.plot(np.NaN,np.NaN,'k--',label='$\\alpha_{s_1}$')
        ax_ROI.plot(np.NaN,np.NaN,'k:',label='nPC')
        # ax_ROI.legend(fontsize=10,bbox_to_anchor=[1.2,1.1],loc='upper right',handlelength=1)

        sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        ax_ROI.add_artist(sbar)
        ax_ROI.set_xticks([])
        ax_ROI.set_yticks([])

        ax = plt.axes([0.525,0.1,0.375,0.275])
        fields = np.zeros((nbin,nSes))
        for i,s in enumerate(np.where(cluster.sessions['bool'])[0]):
            idx_PC = np.where(cluster.fields['status'][:,s,:]>=3)
            # fields[s,:] = np.nansum(cluster.fields['p_x'][:,s,:,:],1).sum(0)
            fields[:,s] = np.nansum(cluster.fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
        fields /= fields.sum(0)
        fields = gauss_smooth(fields,(2,0))

        im = ax.imshow(fields,origin='lower',aspect='auto',cmap='hot')#,clim=[0,1])
        ax.set_xlim([-0.5,nSes-0.5])

        cbaxes = plt.axes([0.92,0.2,0.01,0.175])
        h_cb = plt.colorbar(im,cax=cbaxes)
        h_cb.set_label('place field \ndensity',fontsize=8)
        h_cb.set_ticks([])


        ax.set_xlabel('session')
        ax.set_ylabel('position [bins]')

        # idxes = [range(0,15),range(15,40),range(40,87)]
        # # idxes = [range(0,5),range(5,10),range(10,15)]
        # for (i,idx) in enumerate(idxes):
        #     # print(idx)
        #     ax = plt.axes([0.5,0.475-0.175*i,0.475,0.15])
        #     # ax = plt.subplot(len(idxes),1,i+1)
        #     fields = np.nansum(cluster.fields['p_x'][:,idx,:,:],2).sum(1).sum(0)
        #     fields /= fields.sum()
        #
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['GT'],width=1,facecolor=[0.8,1,0.8],edgecolor='none')
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['RW'],width=1,facecolor=[1,0.8,0.8],edgecolor='none')
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['PC'],width=1,facecolor=[0.7,0.7,1],edgecolor='none')
        #     #ax.bar(pl_dat.bin_edges,fields)
        #     # ax.hist(cluster.fields['location'][:,idx,0,0].flatten(),pl_dat.bin_edges-0.5,facecolor='k',width=0.8,density=True,label='Session %d-%d'%(idx[0]+1,idx[-1]+1))
        #
        #     idx_PC = np.where(cluster.fields['status']>=3)
        #     idx_fields = np.where((idx_PC[1] >= idx[0]) & (idx_PC[1] <=idx[-1]))[0]
        #     cov = cluster.fields['p_x'][idx_PC[0][idx_fields],idx_PC[1][idx_fields],idx_PC[2][idx_fields],:].sum(0)
        #     ax.bar(pl_dat.bin_edges,cov/cov.sum(),facecolor='k',width=0.9,label='Session %d-%d'%(idx[0]+1,idx[-1]+1))
        #     ax.set_xlim([0,L_track])
        #     ax.set_ylim([0,0.04])#np.nanmax(fields)*1.2])
        #
        #     if i==1:
        #         ax.set_ylabel('% of PC')
        #     else:
        #         ax.set_yticks([])
        #     if not (i==2):
        #         ax.set_xticks([])
        #     ax.legend(fontsize=10,loc='upper right')
        # ax.set_xlabel('position [bins]')


        # print('plot fmap corr vs distance for 1. all PC, 2. all active')
        # ax = plt.axes([0.1,0.1,0.25,0.25])
        # D_ROIs_PC = sp.spatial.distance.squareform(sp.spatial.distance.pdist(cluster.stats['com'][c_arr_PC,s,:]))
        # ax.hist(D_ROIs[mat_PC].flat,np.linspace(0,700,201))

        nsteps = 51
        d_arr = np.linspace(0,50,nsteps)
        mean_corr = np.zeros((nsteps,nSes,2))*np.NaN

        for s in tqdm(np.where(cluster.sessions['bool'])[0]):#range(10,15)):
            D_ROIs = sp.spatial.distance.squareform(sp.spatial.distance.pdist(cluster.stats['com'][:,s,:]))
            np.fill_diagonal(D_ROIs,np.NaN)

            idx_PC = cluster.status[:,s,2]
            if idx_PC.sum()>0:
                mat_PC = idx_PC[:,np.newaxis] & idx_PC[:,np.newaxis].T
                D_PCs = D_ROIs[idx_PC,:]
                D_PCs = D_PCs[:,idx_PC]
                NN = np.nanargmin(D_PCs,1)

            C = np.corrcoef(cluster.stats['firingmap'][:,s,:])
            np.fill_diagonal(C,np.NaN)

            for i in range(nsteps-1):
                idx = (D_ROIs>d_arr[i]) & (D_ROIs<=d_arr[i+1])
                if idx_PC.sum()>0:
                    mean_corr[i,s,0] = np.mean(C[idx & mat_PC])
                mean_corr[i,s,1] = np.mean(C[idx])

        dat_bs_PC = bootstrap_data(lambda x : (np.nanmean(x,0),0),mean_corr[...,0].T,N_bs)
        dat_bs = bootstrap_data(lambda x : (np.nanmean(x,0),0),mean_corr[...,1].T,N_bs)

        ax = plt.axes([0.1,0.125,0.25,0.225])
        ax.plot(D_ROIs,C,'.',markerfacecolor=[0.6,0.6,0.6],markersize=0.5,markeredgewidth=0)
        # ax.plot(D_PCs[range(n_PC),NN],C[range(n_PC),NN],'g.',markersize=1,markeredgewidth=0)
        ax.plot([0,50],[0,0],'r:',linewidth=0.75)
        # ax.plot(d_arr,np.nanmean(mean_corr[...,0],1),'r-',linewidth=1)
        pl_dat.plot_with_confidence(ax,d_arr,dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label='place cells')
        pl_dat.plot_with_confidence(ax,d_arr,dat_bs[0],dat_bs[1],col='k',ls='--',label='others')

        # ax.plot(d_arr,np.nanmean(mean_corr[...,1],1),'r--',linewidth=1)
        ax.set_xlim([0,50])
        ax.set_ylim([-0.25,1])
        ax.set_xlabel('d [$\mu$m]')
        ax.set_ylabel('$c_{map(\\nu)}$')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=10,loc='upper right',bbox_to_anchor=[1.2,1.05],handlelength=1)

        mask_PC = (~cluster.status[...,2])
        mask_active = ~(cluster.status[...,1]&(~cluster.status[...,2]))


        fr_key = 'firingrate_adapt'#'firingrate_adapt'
        if not (fr_key in cluster.stats.keys()):
            fr_key = 'firingrate'
        ### stats of all (PC & nPC) cells

        key_arr = [fr_key,'SNR']#,'r_values','MI_value']

        for i,key in enumerate(key_arr):
            ## firingrate
            dat_nPC = np.ma.array(cluster.stats[key], mask=mask_active, fill_value=np.NaN)
            dat_PC = np.ma.array(cluster.stats[key], mask=mask_PC, fill_value=np.NaN)

            dat_PC_mean = np.zeros(nSes)*np.NaN
            dat_PC_CI = np.zeros((2,nSes))*np.NaN
            dat_nPC_mean = np.zeros(nSes)*np.NaN
            dat_nPC_CI = np.zeros((2,nSes))*np.NaN
            for s in np.where(cluster.sessions['bool'])[0]:
                dat_PC_s = dat_PC[:,s].compressed()
                dat_PC_mean[s] = np.mean(dat_PC_s)
                dat_PC_CI[:,s] = np.percentile(dat_PC_s,q=[32.5,67.5])#,q=[2.5,97.5])#
                dat_nPC_s = dat_nPC[:,s].compressed()
                dat_nPC_mean[s] = np.mean(dat_nPC_s)
                dat_nPC_CI[:,s] = np.percentile(dat_nPC_s,q=[32.5,67.5])#,q=[2.5,97.5])#

            # ax = plt.axes([0.525+0.2*i,0.775,0.175,0.2])#subplot(2,2,i+1)
            ax = plt.axes([0.525,0.4+0.2*i,0.375,0.15])#subplot(2,2,i+1)
            pl_dat.plot_with_confidence(ax,range(nSes),dat_PC_mean,dat_PC_CI,col='tab:blue',ls='-',label='place cells')
            pl_dat.plot_with_confidence(ax,range(nSes),dat_nPC_mean,dat_nPC_CI,col='k',ls='--',label='others')
            ax.set_xlim([-0.5,nSes-0.5])
            pl_dat.remove_frame(ax,['top','right'])
            # dat_bs_nPC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_nPC,N_bs)
            # dat_bs_PC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_PC,N_bs)
            # dat_bs_nPC[0][~cluster.sessions['bool']] = np.NaN
            # dat_bs_PC[0][~cluster.sessions['bool']] = np.NaN
            #
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_nPC[0],dat_bs_nPC[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label=None)
            # ax.set_ylabel(key)
            if i==1:
                ax.set_ylabel('SNR')
            else:
                ax.set_ylabel('$\\bar{\\nu}$')
            ax.set_ylim([0,ax.get_ylim()[1]])
            ax.set_xticklabels([])

        ax = plt.axes([0.525,0.8,0.375,0.175])
        ax.plot(np.where(cluster.sessions['bool'])[0],cluster.status[:,cluster.sessions['bool'],1].sum(0),'o',color='k',markersize=2,label='# active neurons')
        ax.plot(np.where(cluster.sessions['bool'])[0],cluster.status[:,cluster.sessions['bool'],2].sum(0),'o',color='tab:blue',markersize=2,label='# place cells')
        ax.set_ylim([0,2000])#nC*1.2])
        ax.set_xlim([-0.5,nSes-0.5])
        ax.set_xticklabels([])
        ax.set_ylabel('# neurons')
        pl_dat.remove_frame(ax,['top','right'])
        # ax.legend(loc='upper right')
        # ax.set_xlabel('session s',fontsize=14)


        ax2 = ax.twinx()
        ax2.plot(np.where(cluster.sessions['bool'])[0],cluster.status[:,cluster.sessions['bool'],2].sum(0)/cluster.status[:,cluster.sessions['bool'],1].sum(0),'--',color='tab:blue',linewidth=0.5)
        ax2.set_ylim([0,0.5])
        ax2.yaxis.label.set_color('tab:blue')
        ax2.tick_params(axis='y',colors='tab:blue')
        ax2.set_ylabel('PC fraction')
        pl_dat.remove_frame(ax2,['top','right'])

        plt.tight_layout()
        plt.show(block=False)


        # ax = plt.subplot(221)
        # ax.set_xticklabels([])

        # ax = plt.subplot(222)
        # ax.set_ylim([0,ax.get_ylim()[1]])
        # ax.set_xticklabels([])
        # ax.set_ylabel('$\\bar{\\nu}$')


        plt.tight_layout()

        plt.show(block=False)
        if sv:
            pl_dat.save_fig('PC_statistics')

#      overrepr = occupancy(:,1:para.nbin)./(sum(nROI(:,3:5),2)/para.nbin);

    if plot_fig[5]:

        print('### plot firingmap over sessions and over time ###')
        if nSes>65:
            s_ref = 61
        else:
            s_ref = 10
        n_plots = 5;
        n_plots_half = (n_plots-1)/2
        # ordered = False

        # if ordered:
            # print('aligned order')
        idxes_tmp = np.where(cluster.status_fields[:,s_ref,:])
        idxes = idxes_tmp[0]
        sort_idx = np.argsort(cluster.fields['location'][idxes_tmp[0],s_ref,idxes_tmp[1],0])

        # idxes = np.where(cluster.status[:,s_ref,2])[0]
        # sort_idx = np.argsort(np.nanmin(cluster.fields['location'][cluster.status[:,s_ref,2],s_ref,:,0],-1))
        sort_idx_ref = idxes[sort_idx]
        nID_ref = len(sort_idx_ref)
        # else:
            # print('non-aligned order')

        width=0.125
        fig = plt.figure(figsize=(7,5))
        for (i,s) in enumerate(range(int(s_ref-n_plots_half),int(s_ref+n_plots_half)+1)):
            ax = plt.axes([0.1+i*width,0.525,width,0.425])
            # ax = plt.subplot(2,n_plots+1,i+1)
            idxes_tmp = np.where(cluster.status_fields[:,s,:])
            idxes = idxes_tmp[0]
            sort_idx = np.argsort(cluster.fields['location'][idxes_tmp[0],s,idxes_tmp[1],0])
            # idxes = np.where(cluster.status[:,s,2])[0]
            # sort_idx = np.argsort(np.nanmin(cluster.fields['location'][cluster.status[:,s,2],s,:,0],-1))
            sort_idx = idxes[sort_idx]
            nID = len(sort_idx)

            firingmap = cluster.stats['firingmap'][sort_idx,s,:]
            firingmap = gauss_smooth(firingmap,[0,2])
            firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,3])


            title_str = "s"
            ds = s-s_ref
            if ds<0:
                title_str += "%d"%ds
            elif ds>0:
                title_str += "+%d"%ds

            ax.set_title(title_str)

            if i == 0:
                #ax.plot([cluster.para['zone_idx']['gate'][0],cluster.para['zone_idx']['gate'][0]],[1,nID],color='r',linestyle=':',linewidth=3)
                #ax.plot([cluster.para['zone_idx']['gate'][1],cluster.para['zone_idx']['gate'][1]],[1,nID],color='r',linestyle=':',linewidth=3)

                #ax.plot([cluster.para['zone_idx']['reward'][0],cluster.para['zone_idx']['reward'][0]],[1,nID],color='g',linestyle=':',linewidth=3)
                #ax.plot([cluster.para['zone_idx']['reward'][1],cluster.para['zone_idx']['reward'][1]],[1,nID],color='g',linestyle=':',linewidth=3)

                #ax.set_xticks(np.linspace(0,nbin,3))
                #ax.set_xticklabels(np.linspace(0,nbin,3))

                ax.set_ylabel('Neuron ID')
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xlim([0,nbin])
            ax.set_ylim([nID,0])

            ax = plt.axes([0.1+i*width,0.08,width,0.425])
            # ax = plt.subplot(2,n_plots+1,i+2+n_plots)
            # if not ordered:

            firingmap = cluster.stats['firingmap'][sort_idx_ref,s,:]
            firingmap = gauss_smooth(firingmap,[0,2])
            firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            im = ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,3])

            ax.set_xlim([0,nbin])
            if i == 0:
                ax.set_ylabel('Neuron ID')
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            ax.set_ylim([nID_ref,0])

            if i==n_plots_half:
                ax.set_xlabel('Location [bin]')

        cbaxes = plt.axes([0.74,0.75,0.01,0.2])
        cb = fig.colorbar(im,cax = cbaxes,orientation='vertical')
        # cb.set_ticks([0,1])
        # cb.set_ticklabels(['low','high'])
        cb.set_label('$\\nu^*$',fontsize=10)

        ax = plt.axes([0.85,0.08,0.125,0.87])
        idx_strong_PC = np.where(cluster.status[...,2].sum(1)>20)[0]
        idx_PC = 28#np.random.choice(idx_strong_PC)    ## 28,1081
        print(idx_PC)
        firingmap = cluster.stats['firingmap'][idx_PC,...]
        firingmap = gauss_smooth(firingmap,[0,4])
        firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
        # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
        ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,3])
        ax.barh(range(nSes),-(cluster.status[idx_PC,:,2]*10.),left=-5,facecolor='r')
        # idx_coding = np.where(cluster.status[idx_PC,:,2])[0]
        # ax.plot(-np.ones_like(idx_coding)*10,idx_coding,'ro')
        ax.set_xlim([-10,nbin])
        ax.set_ylim([nSes,0])
        ax.set_ylabel('Session')
        ax.set_xlabel('Location [bins]')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #plt.set_cmap('jet')
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('PC_mapDynamics')


    if plot_fig[6]:


        #PC_match_certainty = np.zeros((nC,nSes,nSes))
        #t_start = time.time()
        #for s1 in range(nSes):
          #for s2 in range(s1,nSes):
            #PC_match_certainty[:,s1,s2] = np.power(np.nanprod(cluster.sessions['match_score'][:,s1+1:s2+1],1),1/(np.sum(~np.isnan(cluster.sessions['match_score'][:,s1+1:s2+1]),1)+1))*cluster.sessions['match_score'][:,s2]
            ##print('ds : %d'%(s2-s1))
            ##print(PC_match_certainty[:,s1,s2])


        recurrence = {'active': {'all':               np.zeros((nSes,nSes))*np.NaN,
                                 'continuous':        np.zeros((nSes,nSes))*np.NaN,
                                 'overrepresentation':np.zeros((nSes,nSes))*np.NaN},
                      'coding': {'all':               np.zeros((nSes,nSes))*np.NaN,
                                 'ofactive':          np.zeros((nSes,nSes))*np.NaN,
                                 'continuous':        np.zeros((nSes,nSes))*np.NaN,
                                 'overrepresentation':np.zeros((nSes,nSes))*np.NaN}}

        N = {'active': cluster.status[:,:,1].sum(0),
             'coding': cluster.status[:,:,2].sum(0)}
        L=1#00

        #for s in tqdm(range(nSes)):#min(30,nSes)):
        if n_processes>1:
            pool = get_context("spawn").Pool(n_processes)
            res = pool.starmap(get_overlap,zip(range(nSes),itertools.repeat((cluster.status,N,L))))

        for (s,r) in enumerate(res):
            for pop in r.keys():
                for key in r[pop].keys():
                    recurrence[pop][key][s,:] = r[pop][key]

        ### ds = 0
        plt0 = True
        if plt0:
            print(nbin)
            p_shift = np.zeros(nbin)
            for s in range(nSes):
                idx_field = np.where(cluster.status_fields[:,s,:])
                for c,f in zip(idx_field[0],idx_field[1]):
                    roll = round((-cluster.fields['location'][c,s,f,0]+nbin/2)/L_track*nbin).astype('int')
                    p_shift += np.roll(cluster.fields['p_x'][c,s,f,:],roll)
            p_shift /= p_shift.sum()

            PC_idx = np.where(cluster.status[...,2])
            N_data = len(PC_idx[0])
            print('N data: %d'%N_data)

            p_ds0,p_cov = fit_shift_model(p_shift)
            print(p_ds0)

        ### ds > 0
        p = {'all':     {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'cont':    {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'mix':     {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'discont': {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'silent':  {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN}}

        t_start = time.time()
        s1_shifts,s2_shifts,f1,f2 = np.unravel_index(cluster.compare['pointer'].col,(nSes,nSes,cluster.meta['field_count_max'],cluster.meta['field_count_max']))
        c_shifts = cluster.compare['pointer'].row

        celltype = 'all'
        if celltype == 'all':
            idx_celltype = cluster.status[c_shifts,s1_shifts,2]
        if celltype == 'gate':
            idx_celltype = cluster.status[c_shifts,s1_shifts,3]
        if celltype == 'reward':
            idx_celltype = cluster.status[c_shifts,s1_shifts,4]

        if (not('stability' in vars(cluster).keys())) | reprocess:

            if n_processes>1:
                pool = get_context("spawn").Pool(n_processes)
                # pool = mp.Pool(n_processes)
                res = pool.starmap(get_shift_distr,zip(range(1,nSes),itertools.repeat(cluster.compare),itertools.repeat((nSes,nbin,N_bs,idx_celltype))))
                pool.close()
            else:
                res = []
                for ds in range(1,nSes):
                    res.append(get_shift_distr(ds,cluster.compare,(nSes,nbin,N_bs,idx_celltype)))

            for (ds,r) in enumerate(res):
                for pop in r.keys():
                    for key in r[pop].keys():
                        p[pop][key][ds,...] = r[pop][key]
            cluster.stability = p
        else:
            p = cluster.stability
        t_end = time.time()
        print('done - time: %5.3g'%(t_end-t_start))

        # testing = True
        # if testing:
        fig = plt.figure(figsize=(7,4),dpi=300)
        # frac_stable = np.zeros((nSes,3))*np.NaN
        #
        # ### get fraction of cells/fields which actually remains stable (<1.96 \sigma)
        # cbool = cluster.sessions['bool'].copy()
        # #print(cbool)
        # for ds in range(1,nSes):
        #
        #     Ds = s2_shifts-s1_shifts
        #     idx_ds = np.where((Ds==ds) & idx_celltype)[0]
        #     N_data = len(idx_ds)
        #     cbool[-ds:] = False
        #     #print(cbool)
        #     N_ofactive = cluster.status[:,cbool,1].sum()
        #     N_ofcoding = cluster.status[:,cbool,2].sum()
        #
        #     idx_shifts = cluster.compare['pointer'].data[idx_ds].astype('int')-1
        #     shifts = cluster.compare['shifts'][idx_shifts]
        #     N_stable = (np.abs(shifts)<(1.96*cluster.stability['all']['mean'][ds,2])).sum()
        #
        #     frac_stable[ds,0] = N_stable/N_data
        #     frac_stable[ds,1] = N_stable/N_ofcoding
        #     #frac_stable[ds,2] = N_stable/N_ofactive


        # plt.figure()
        # plt.plot(pl_dat.n_edges,frac_stable[:,0],'k',label='of recurring')
        # plt.plot(pl_dat.n_edges,frac_stable[:,1],'k--',label='of coding')
        # plt.plot(pl_dat.n_edges,frac_stable[:,2],'k:',label='of active')
        # plt.show(block=False)

        # f_stable_pos = plt.figure(figsize=(5,2))
        # f_shift_distr = plt.figure(figsize=(5,2))
        ax_distr = plt.axes([0.075,0.1,0.35,0.325])
        pl_dat.add_number(fig,ax_distr,order=2,offset=[-100,50])

        for j,ds in tqdm(enumerate([1,5,10,20,40])):#min(nSes,30)):

            Ds = s2_shifts-s1_shifts
            idx_ds = np.where((Ds==ds) & idx_celltype)[0]
            N_data = len(idx_ds)
            cdf_shifts_ds = np.zeros((N_data,nbin))

            idx_shifts = cluster.compare['pointer'].data[idx_ds].astype('int')-1
            shifts_distr = cluster.compare['shifts_distr'][idx_shifts,:].toarray()
            # for i,_ in enumerate(idx_ds):
            #     roll = round(-shifts[i]+L_track/2).astype('int')
            #     cdf_shifts_ds[i,:] = np.cumsum(np.roll(shifts_distr[i,:],roll))
            #     cdf_shifts_ds[i,:] = np.roll(cdf_shifts_ds[i,:],-roll)

            _, _, _, shift_distr = bootstrap_shifts(fit_shift_model,shifts_distr,N_bs,nbin)

            s1_ds = s1_shifts[idx_ds]
            s2_ds = s2_shifts[idx_ds]
            c_ds = cluster.compare['pointer'].row[idx_ds]

            idxes = cluster.compare['inter_coding'][idx_ds,1]==1

            CI = np.percentile(shift_distr,[5,95],0)
            ax_distr.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),shift_distr.mean(0),color=[0.2*j,0.2*j,1],linewidth=0.5,label='$\Delta$ s = %d'%ds)
            # ax_distr.errorbar(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),shift_distr.mean(0),shift_distr.mean(0)-CI[0,:],CI[1,:]-shift_distr.mean(0),fmt='none',ecolor=[1,0.,0.],elinewidth=0.5)

            # ax_distr.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),F_shifts(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),cluster.stability['all']['mean'][ds,0],cluster.stability['all']['mean'][ds,1],cluster.stability['all']['mean'][ds,2],cluster.stability['all']['mean'][ds,3]),'g',linewidth=2)
        pl_dat.remove_frame(ax_distr,['top','right'])

        ax_distr.plot(np.linspace(-49.5,49.5,100),p_shift,'k--',linewidth=0.5)
        ax_distr.set_xlim([-L_track/2,L_track/2])
        ax_distr.set_ylim([0,0.065])
        ax_distr.set_xlabel('field shift $\Delta \\theta$ [bin]')
        ax_distr.set_ylabel('$p(\Delta \\theta)$')
        ax_distr.set_yticks([])
        ax_distr.legend(loc='upper left',fontsize=8, handlelength=1,bbox_to_anchor=[0.05,1.1])

        dx_arr = np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin)
        N_data = np.zeros(nSes)*np.NaN

        D_KS = np.zeros(nSes)*np.NaN
        N_stable = np.zeros(nSes)*np.NaN
        N_total = np.zeros(nSes)*np.NaN     ### number of PCs which could be stable
        # fig = plt.figure()
        for ds in range(1,nSes):#min(nSes,30)):
            Ds = s2_shifts-s1_shifts
            idx_ds = np.where(Ds==ds)[0]
            N_data[ds] = len(idx_ds)

            idx_shifts = cluster.compare['pointer'].data[idx_ds].astype('int')-1
            shifts = cluster.compare['shifts'][idx_shifts]
            N_stable[ds] = (np.abs(shifts)<(1.96*cluster.stability['all']['mean'][ds,2])).sum()
            shifts_distr = cluster.compare['shifts_distr'][idx_shifts,:].toarray().sum(0)
            shifts_distr /= shifts_distr.sum()
            fun_distr = F_shifts(dx_arr,cluster.stability['all']['mean'][ds,0],cluster.stability['all']['mean'][ds,1],cluster.stability['all']['mean'][ds,2],cluster.stability['all']['mean'][ds,3])

            session_bool = np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False)
            N_total[ds] = cluster.status_fields[:,session_bool,:].sum()
            # if ds < 20:
            #     plt.subplot(5,4,ds)
            #     plt.plot(dx_arr,np.cumsum(shifts_distr),'k')
            #     plt.plot(dx_arr,np.cumsum(fun_distr),'r')
            #     plt.title('$\Delta s=%d$'%ds)
            D_KS[ds] = np.abs(np.cumsum(shifts_distr)-np.cumsum(fun_distr)).max()
        # plt.show(block=False)


        # plt.figure(fig_test.number)
        ax_p1 = plt.axes([0.05,0.825,0.175,0.1])
        ax_p2 = plt.axes([0.05,0.675,0.175,0.1])
        ax_shift1 = plt.axes([0.275,0.825,0.175,0.1])
        ax_shift2 = plt.axes([0.275,0.675,0.175,0.1])
        pl_dat.add_number(fig,ax_p1,order=1,offset=[-50,25])
        # try:
        c = 5
        p1 = cluster.fields['p_x'][c,10,0,:]
        p2 = cluster.fields['p_x'][c,11,0,:]
        ax_p1.plot(p1,color='tab:orange',label='$p(\\theta_s$)')
        ax_p1.plot(p2,color='tab:blue',label='$p(\\theta_{s+\Delta s})$')
        ax_p1.set_xticklabels([])
        ax_p1.legend(fontsize=8,handlelength=1,loc='upper right',bbox_to_anchor=[1.2,1.6])
        pl_dat.remove_frame(ax_p1,['top','left','right'])
        ax_p1.set_yticks([])

        _,dp = periodic_distr_distance(p1,p2,100,100,N_bs=10000,mode='bootstrap')
        ax_shift1.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),dp,'k',label='$p(\Delta \\theta)$')
        ax_shift1.set_xticklabels([])
        ax_shift1.legend(fontsize=8,handlelength=1,loc='upper right',bbox_to_anchor=[1.2,1.6])
        pl_dat.remove_frame(ax_shift1,['top','left','right'])
        ax_shift1.set_yticks([])

        p1 = cluster.fields['p_x'][5,34,0,:]
        p2 = cluster.fields['p_x'][15,63,1,:]
        ax_p2.plot(p1,color='tab:orange')
        ax_p2.plot(p2,color='tab:blue')
        ax_p2.set_yticks([])
        _,dp = periodic_distr_distance(p1,p2,100,100,N_bs=10000,mode='bootstrap')
        ax_shift2.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),dp,'k')
        ax_shift2.set_xlabel('field shift $\Delta \\theta$')
        pl_dat.remove_frame(ax_shift2,['top','left','right'])
        ax_shift2.set_yticks([])

        ax_p2.set_xlabel('position')
        pl_dat.remove_frame(ax_p2,['top','left','right'])
        # except:
            # pass


        ax_img = plt.axes([0.3,0.25,0.175,0.25])
        img = mpimg.imread('/home/wollex/Data/Science/PhD/Thesis/pics/others/shifthist_theory_0.3.png')
        ax_img.imshow(img)
        pl_dat.remove_frame(ax_img,['top','right','bottom','left'])
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        x_lim = np.where(cluster.sessions['bool'])[0][-1] - np.where(cluster.sessions['bool'])[0][0] + 1
        ax_D = plt.axes([0.6,0.8,0.375,0.13])
        ax_D.plot(range(1,nSes+1),D_KS,'k')
        ax_D.set_xlim([0,x_lim])
        ax_D.set_ylabel('$D_{KS}$')
        ax_D.set_xticklabels([])
        ax_D.set_ylim([0,0.2])

        pl_dat.add_number(fig,ax_D,order=3)

        ax_mu = plt.axes([0.6,0.625,0.375,0.13])
        ax_sigma = plt.axes([0.6,0.45,0.375,0.13])
        ax_r = plt.axes([0.6,0.275,0.375,0.13])

        ax_sigma.plot([0,nSes],[p_ds0[2],p_ds0[2]],linestyle='--',color=[0.6,0.6,0.6])
        ax_sigma.text(10,p_ds0[2]+1,'$\sigma_0$',fontsize=8)
        ax_mu.plot([0,nSes],[0,0],linestyle=':',color=[0.6,0.6,0.6])
        ax_r.plot([0,nSes],[0.5,0.5],linestyle=':',color=[0.6,0.6,0.6])

        # pl_dat.plot_with_confidence(ax_mu,range(1,nSes+1),p['all']['mean'][:,3],p['all']['mean'][:,3]+np.array([[-1],[1]])*p['all']['std'][:,3]*1.96,'k','-')
        # pl_dat.plot_with_confidence(ax_sigma,range(1,nSes+1),p['all']['mean'][:,2],p['all']['mean'][:,2]+np.array([[-1],[1]])*p['all']['std'][:,2]*1.96,'k','-')
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p['all']['mean'][:,1],p['all']['mean'][:,1]+np.array([[-1],[1]])*p['all']['std'][:,1]*1.96,'k','-')
        print(p['all']['CI'].shape)
        pl_dat.plot_with_confidence(ax_mu,range(1,nSes+1),p['all']['mean'][:,3],p['all']['CI'][...,3].T,'k','-')
        pl_dat.plot_with_confidence(ax_sigma,range(1,nSes+1),p['all']['mean'][:,2],p['all']['CI'][...,2].T,'k','-')
        pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p['all']['mean'][:,1],p['all']['CI'][...,1].T,'k','-')

        p_corr = np.minimum(1,p['all']['mean'][:,1]+(1-p['all']['mean'][:,1])*(2*1.96*p['all']['mean'][:,2]/nbin))
        p_SD = np.sqrt((1-2*1.96*p['all']['mean'][:,2]/nbin)**2*p['all']['std'][:,1]**2 + ((1-p['all']['mean'][:,1])*2*1.96/nbin)**2 * p['all']['std'][:,2]**2)
        pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p_corr,p_SD,'tab:blue','--')

        # ax_r.plot(range(1,nSes+1),p_corr,'k--')
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p_corr,p_corr+np.array([[-1],[1]])*p['all']['std'][:,1]*1.96,'k','-',label='stable place fields (of rec. place cell)')


        ax_mu.set_xlim([0,x_lim])
        ax_mu.set_ylim([-10,10])
        ax_mu.set_xticklabels([])
        ax_mu.set_ylabel('$\mu_{\Delta \\theta}$')
        ax_sigma.set_xlim([0,x_lim])
        ax_sigma.set_ylim([0,10])
        ax_sigma.set_xticklabels([])
        ax_sigma.set_ylabel('$\sigma_{\Delta \\theta}$')
        ax_r.set_xlim([0,x_lim])
        ax_r.set_ylim([0.0,1])
        ax_r.set_yticks(np.linspace(0,1,3))
        ax_r.set_yticklabels(np.linspace(0,1,3))
        ax_r.set_xticklabels([])
        ax_r.set_ylabel('$r_{stable}$')
        pl_dat.remove_frame(ax_D,['top','right'])
        pl_dat.remove_frame(ax_mu,['top','right'])
        pl_dat.remove_frame(ax_sigma,['top','right'])
        pl_dat.remove_frame(ax_r,['top','right'])
        # axs[0][1].set_ylim([0,1])

        ax_N = plt.axes([0.6,0.1,0.375,0.13])
        ax_N.plot(N_data,'k')
        ax_N.plot(N_stable,'tab:blue')
        ax_N.set_xlabel('session difference $\Delta s$')
        ax_N.set_xlim([0,x_lim])
        ax_N.set_ylabel('$N_{shifts}$')
        pl_dat.remove_frame(ax_N,['top','right'])

        # print(N_stable/N_total)
        plt.tight_layout()
        plt.show(block=False)

        plt.figure()
        plt.plot(range(1,nSes+1),N_stable/N_total,'k--',linewidth=0.5)
        plt.yscale('log')
        plt.show(block=False)


        def plot_shift_distr(p,p_std,p_ds0):
            nSes = p.shape[0]
            f,axs = plt.subplots(2,2,figsize=(6,4),sharex=True)
            axs[1][0].plot([0,nSes],[p_ds0[2],p_ds0[2]],linestyle='--',color=[0.6,0.6,0.6])
            axs[1][1].plot([0,nSes],[0,0],linestyle='--',color=[0.6,0.6,0.6])
            for i in range(4):
                pl_dat.plot_with_confidence(axs[int(np.floor(i/2))][i%2],range(nSes),p[:,i],p[:,i]+np.array([[-1],[1]])*p_std[:,i]*1.96,'k','--')
            #axs[0][1].set_yscale('log')

            axs[0][1].yaxis.set_label_position("right")
            axs[0][1].yaxis.tick_right()

            axs[1][1].yaxis.set_label_position("right")
            axs[1][1].yaxis.tick_right()

            axs[0][0].set_xlim([0,max(20,nSes/2)])
            axs[0][0].set_ylim([0,1])
            axs[0][1].set_ylim([0,1])
            axs[1][0].set_ylim([0,10])
            axs[1][1].set_ylim([-10,10])

            axs[1][0].set_xlabel('$\Delta$ s',fontsize=14)
            axs[1][1].set_xlabel('$\Delta$ s',fontsize=14)
            axs[0][0].set_ylabel('1-$r_{stable}$',fontsize=14)
            axs[0][1].set_ylabel('$r_{stable}$',fontsize=14)
            axs[1][0].set_ylabel('$\sigma$',fontsize=14)
            axs[1][1].set_ylabel('$\mu$',fontsize=14)
            plt.tight_layout()
            #plt.title('cont')
            plt.show(block=False)


        # plot_shift_distr(p['all']['mean'],p['all']['std'],p_ds0)
        #for key in p.keys():
          #plot_shift_distr(p[key]['mean'],p[key]['std'],p_ds0)
        if sv:
            pl_dat.save_fig('stability_dynamics')



        # plt.figure(figsize=(4,4))
        # ax = plt.axes([0.15,0.725,0.8,0.225])
        # pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,np.nanmean(recurrence['active']['all'],0),1.96*np.nanstd(recurrence['active']['all'],0),col='k',ls='-',label='recurrence of active cells')
        # ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])
        # ax.set_xlim([0,t_ses[-1]])
        # ax.set_ylim([0,1.1])
        # ax.set_xticklabels([])
        # ax.set_ylabel('fraction',fontsize=14)
        # #ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # ax = plt.axes([0.15,0.425,0.8,0.225])
        # pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,np.nanmean(recurrence['coding']['ofactive'],0),1.0*np.nanstd(recurrence['coding']['ofactive'],0),col='k',ls='-',label='place cell recurrence (of rec. active)')
        # #ax.plot(pl_dat.n_edges-1,np.nanmean(recurrence['coding']['all'],0),'b--',label='recurrence of place cells')
        # ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])
        # ax.set_xlim([0,t_ses[-1]])
        # ax.set_ylim([0,1.1])
        # ax.set_xticklabels([])
        # ax.set_ylabel('fraction',fontsize=14)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # ax = plt.axes([0.15,0.125,0.8,0.225])
        # p_corr = np.minimum(1,p['all']['mean'][:,1]+(1-p['all']['mean'][:,1])*(2*1.96*p['all']['mean'][:,2]/nbin))
        # pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,p_corr,p_corr+np.array([[-1],[1]])*p['all']['std'][:,1]*1.96,'k','-',label='stable place fields (of rec. place cell)')
        # ax.set_xlim([0,t_ses[-1]])
        # ax.set_ylim([0,1.1])
        # ax.set_ylabel('fraction',fontsize=14)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        # ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])
        #
        # plt.tight_layout()
        # plt.show(block=False)

        # if sv:
        #     pl_dat.save_fig('stability_dynamics_hierarchy')

        plt.figure(figsize=(4,2))
        ax = plt.subplot(111)
        pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,p['all']['mean'][:,2],p['all']['mean'][:,2]+np.array([[-1],[1]])*p['all']['std'][:,2]*1.96,'k','--')
        ax.set_xlim([0,40])
        ax.set_ylim([0,12])
        ax.set_xlabel('session diff. $\Delta$ s')
        ax.set_ylabel('$\sigma$ [bins]',fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        #ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])

        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('stability_dynamics_width')

        #plot_shift_distr(p['cont']['mean'],p['cont']['std'],p_ds0)
        #plot_shift_distr(p['silent']['mean'],p['silent']['std'],p_ds0)

        #if sv:
          #pl_dat.save_fig('stability_dynamics_cont')
        ##plot_shift_distr(p['mix']['mean'],p['mix']['std'],p_ds0)
        #plot_shift_distr(p['discont']['mean'],p['discont']['std'],p_ds0)
        #if sv:
          #pl_dat.save_fig('stability_dynamics_disc')

        #f,axs = plt.subplots(2,2,figsize=(6,4),sharex=True)
        plt.figure(figsize=(4,2.5))
        ax = plt.subplot(111)
        pl_dat.plot_with_confidence(ax,range(nSes),cluster.stability['cont']['mean'][:,1],cluster.stability['cont']['mean'][:,1]+np.array([[-1],[1]])*cluster.stability['cont']['std'][:,1],'b','--',label='coding')
        pl_dat.plot_with_confidence(ax,range(nSes),cluster.stability['discont']['mean'][:,1],cluster.stability['discont']['mean'][:,1]+np.array([[-1],[1]])*cluster.stability['discont']['std'][:,1],'r','--',label='no coding')
        #pl_dat.plot_with_confidence(ax,range(nSes),cluster.stability['silent']['mean'][:,1],cluster.stability['silent']['mean'][:,1]+np.array([[-1],[1]])*cluster.stability['silent']['std'][:,1]*1.96,'g','--',label='silent')
        #ax.set_yscale('log')
        ax.set_ylim([0,1.1])
        ax.set_xlim([0,20])
        ax.set_xlabel('$\Delta$ s',fontsize=14)
        ax.set_ylabel('$r_{stable}$',fontsize=14)
        plt.tight_layout()
        plt.legend(loc='lower right',fontsize=10)
        #plt.title('cont')
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('stability_dynamics_cont_vs_disc')



        maxSes = 6
        print('what are those stable cells coding for?')
        plt.figure(figsize=(5,2.5))

        col_arr = [[0.5,0.5,1],[0.5,0.5,0.5],[1,0.5,0.5],[0.5,1,0.5]]
        label_arr = ['continuous','mixed','non-coding','silent']
        key_arr = ['cont','mix','discont','silent']

        w_bar = 0.2
        nKey = len(key_arr)
        offset_bar = ((nKey+1)%2)*w_bar/2 + (nKey//2 - 1)*w_bar

        for i,key in enumerate(key_arr):

            plt.bar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
            plt.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],cluster.stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')

        plt.xlabel('session difference $\Delta s$',fontsize=14)
        plt.ylabel('$\%$ stable fields',fontsize=14)
        plt.ylim([0,1.1])
        plt.legend(loc='upper right',ncol=2)
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('intercoding_state')


    if plot_fig[7]:

        sesMax = 6

        nSteps = 10
        #val_arr = np.linspace(1,10,nSteps)
        #val_arr = np.linspace(-1,3,nSteps)
        #val_arr = np.append(np.linspace(0,0.2,nSteps-1),1)
        val_arr = [0.99,0.95,0.9,0.8,0.7,0.5,0.3,0.2,0.1,0.05]#[0.05,0.2,0.5,0.9,95]
        print(val_arr)
        SNR_thr = 2
        rval_thr = 0
        pm_thr = 0.05
        alpha = 1
        CI_thr = 10
        Bayes_thr = 3

        if (not('stability' in vars(cluster).keys())) | reprocess:
            p = {'all':     {'mean': np.zeros((nSteps,sesMax,4))*np.NaN,
                             'std':  np.zeros((nSteps,sesMax,4))*np.NaN},
                 'cont':    {'mean': np.zeros((nSteps,sesMax,4))*np.NaN,
                             'std':  np.zeros((nSteps,sesMax,4))*np.NaN},
                 'mix':     {'mean': np.zeros((nSteps,sesMax,4))*np.NaN,
                             'std':  np.zeros((nSteps,sesMax,4))*np.NaN},
                 'discont': {'mean': np.zeros((nSteps,sesMax,4))*np.NaN,
                             'std':  np.zeros((nSteps,sesMax,4))*np.NaN},
                 'silent':  {'mean': np.zeros((nSteps,sesMax,4))*np.NaN,
                             'std':  np.zeros((nSteps,sesMax,4))*np.NaN}}
            t_start = time.time()
            #for (i,pm_thr) in enumerate(val_arr):
            #for (i,alpha) in enumerate(val_arr):
            #for (i,Bayes_thr) in enumerate(val_arr):
            for (i,pm_thr) in enumerate(val_arr):
                #print(pm_thr)
                cluster.update_status(SNR_thr=SNR_thr,rval_thr=rval_thr,pm_thr=pm_thr,alpha=alpha,CI_thr=CI_thr,Bayes_thr=Bayes_thr)
                cluster.compareSessions(n_processes=n_processes,reprocess=True)

                s1_shifts,s2_shifts,f1,f2 = np.unravel_index(cluster.compare['pointer'].col,(nSes,nSes,cluster.meta['field_count_max'],cluster.meta['field_count_max']))
                c_shifts = cluster.compare['pointer'].row

                celltype = 'all'
                if celltype == 'all':
                    idx_celltype = cluster.status[c_shifts,s1_shifts,2]
                if celltype == 'gate':
                    idx_celltype = cluster.status[c_shifts,s1_shifts,3]
                if celltype == 'reward':
                    idx_celltype = cluster.status[c_shifts,s1_shifts,4]

                if n_processes>1:
                    pool = get_context("spawn").Pool(n_processes)
                    res = pool.starmap(get_shift_distr,zip(range(1,sesMax),itertools.repeat(cluster.compare),itertools.repeat((nSes,nbin,N_bs,idx_celltype))))
                    pool.close()
                else:
                    res = []
                for ds in range(1,sesMax):
                    res.append(get_shift_distr(ds,cluster.compare,(nSes,nbin,N_bs,idx_celltype)))

                for (ds,r) in enumerate(res):
                    for pop in r.keys():
                        for key in r[pop].keys():
                            p[pop][key][i,ds,:] = r[pop][key]
            cluster.stability = p
            t_end = time.time()
            print('done - time: %5.3g'%(t_end-t_start))

            plt.figure(figsize=(4,3))
            pop_key = 'all'
            for (i,ds) in enumerate([2]):
                for pop_key in ['cont','discont','silent']:
                    #s1_shifts,s2_shifts = np.unravel_index(cluster.compare['pointer'].col,(nSes,nSes))
                    #idx_ds = np.where(s2_shifts-s1_shifts==ds)[0]
                    if pop_key=='cont':
                        col = [0.2+0.3*i,0.2+0.3*i,1]
                        #idxes = compare['inter_coding'][idx_ds,1]==1
                    if pop_key=='discont':
                        col = [1,0.2+0.3*i,0.2+0.3*i]
                        #idxes = (compare['inter_coding'][idx_ds,1]==0) & (compare['inter_active'][idx_ds,1]==1)
                    if pop_key=='silent':
                        col = [0.2+0.3*i,1,0.2+0.3*i]
                        #idxes = compare['inter_active'][idx_ds,1]==0

                plt.plot(val_arr[:-1],cluster.stability[pop_key]['mean'][:,ds][:-1],color=col,label=pop_key)
                plt.errorbar(val_arr[:-1],cluster.stability[pop_key]['mean'][:,ds][:-1],cluster.stability[pop_key]['std'][:,ds,1][:-1],fmt='none',ecolor='r')

            plt.xlabel('$p_m$',fontsize=14)
            plt.legend(fontsize=10)
            plt.ylim([0,0.6])
            plt.ylabel('$r_{stable}$',fontsize=14)
            plt.tight_layout()
            plt.show(block=False)

            if sv:
                pl_dat.save_fig('stability_impact_pm')
            ##for i in range(nSteps):
              ##col = np.ones(3)*0.2*i
              ##plt.bar(np.arange(1,sesMax+1)-0.1*nSteps+0.2*i,cluster.stability['cont']['mean'][i,:sesMax,1],width=0.2,facecolor=col,label='continuous')
              ##plt.errorbar(np.arange(1,sesMax+1)-0.1*nSteps+0.2*i,cluster.stability['cont']['mean'][i,:sesMax,1],cluster.stability['cont']['std'][i,:sesMax,1],fmt='none',ecolor='r')
            ##plt.show(block=False)
            #return cluster



    if plot_fig[8]:

        s = 10
        idx_PCs = cluster.status[:,:,2]
        plt.figure(figsize=(4,2.5))
        plt.scatter(cluster.stats['MI_p_value'][idx_PCs],cluster.stats['Bayes_factor'][idx_PCs],color='r',s=5)
        plt.scatter(cluster.stats['MI_p_value'][~idx_PCs],cluster.stats['Bayes_factor'][~idx_PCs],color=[0.6,0.6,0.6],s=3)
        plt.xlabel('p-value (mutual information)',fontsize=14)
        plt.ylabel('log($Z_{PC}$) - log($Z_{nPC}$)',fontsize=14)
        plt.ylim([-10,50])
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('PC_choice_s=%d'%s)
        #return arrays,occupancy,ROI_recurr,N_pairs,N_norm,pop_overlap




    if plot_fig[9]:

        print('### Plotting matching score statistics ###')

        print('now add example how to calculate footprint correlation(?), sketch how to fill cost-matrix')

        s = 12
        margins = 18

        D_ROIs = sp.spatial.distance.squareform(sp.spatial.distance.pdist(cluster.stats['com'][:,s,:]))
        np.fill_diagonal(D_ROIs,np.NaN)

        idx_dense = np.where((np.sum(D_ROIs<margins,1)<=8) & np.isfinite(cluster.IDs['neuronID'][:,s+1,1]))[0]
        c = np.random.choice(idx_dense)
        # c = idx_dense[0]
        c = 932
        n = int(cluster.IDs['neuronID'][c,s,1])
        #n = 328
        print(c,n)
        plt.figure(figsize=(7,4),dpi=300)
        props = dict(boxstyle='round', facecolor='w', alpha=0.8)

        ## plot ROIs from a single session

        # c = np.where(cluster.IDs['neuronID'][:,s,1] == n)[0][0]
        idx_close = np.where(D_ROIs[c,:]<margins*2)[0]

        n_close = cluster.IDs['neuronID'][D_ROIs[c,:]<margins*1.5,s,1].astype('int')
        # return
        pathSession = pathcat([cluster.meta['pathMouse'],'Session%02d'%(s+1)])
        pathLoad = pathcat([pathSession,'results_redetect.mat'])
        print('load from %s'%pathLoad)
        ld = loadmat(pathLoad,variable_names=['A'])
        #Cn = ld['Cn']
        A = ld['A']#.toarray().reshape(cluster.meta['dims'][0],cluster.meta['dims'][1],-1)

        cm = com(A,512,512)

        # print(cluster.sessions['shift'][s,:])
        # print(cluster.stats['com'][c,s,:])
        # print(cm[n,:])
        Cn = A.sum(1).reshape(cluster.meta['dims'])
        # x = int(cluster.stats['com'][c,s,0])#+cluster.sessions['shift'][s,0])
        # y = int(cluster.stats['com'][c,s,1])#+cluster.sessions['shift'][s,1])
        x = int(cm[n,0])#-cluster.sessions['shift'][s,0])
        y = int(cm[n,1])#-cluster.sessions['shift'][s,1])

        ax_ROIs1 = plt.axes([0.05,0.55,0.25,0.4])

        #margins = 10
        Cn_tmp = Cn[y-margins:y+margins,x-margins:x+margins]
        Cn -= Cn_tmp.min()
        Cn_tmp -= Cn_tmp.min()
        Cn /= Cn_tmp.max()

        ax_ROIs1.imshow(Cn,origin='lower',clim=[0,1])
        An = A[...,n].reshape(cluster.meta['dims']).toarray()
        for nn in n_close:
            cc = np.where(cluster.IDs['neuronID'][:,s,1]==nn)
            # print('SNR: %.2g'%cluster.stats['SNR'][cc,s])
            ax_ROIs1.contour(A[...,nn].reshape(cluster.meta['dims']).toarray(),[0.2*A[...,nn].max()],colors='w',linestyles='--',linewidths=1)
        ax_ROIs1.contour(An,[0.2*An.max()],colors='w',linewidths=3)
        # ax_ROIs1.plot(cluster.sessions['com'][c,s,0],cluster.sessions['com'][c,s,1],'kx')

        sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        ax_ROIs1.add_artist(sbar)
        ax_ROIs1.set_xlim([x-margins,x+margins])
        ax_ROIs1.set_ylim([y-margins,y+margins])
        ax_ROIs1.text(x-margins+3,y+margins-5,'Session s',bbox=props,fontsize=10)
        ax_ROIs1.set_xticklabels([])
        ax_ROIs1.set_yticklabels([])

        D_ROIs_cross = sp.spatial.distance.cdist(cluster.stats['com'][:,s,:],cluster.stats['com'][:,s+1,:])
        n_close = cluster.IDs['neuronID'][D_ROIs_cross[c,:]<margins*2,s+1,1].astype('int')

        pathSession = pathcat([cluster.meta['pathMouse'],'Session%02d'%(s+2)])
        pathLoad = pathcat([pathSession,'results_redetect.mat'])
        ld = loadmat(pathLoad)
        A = ld['A']#.toarray().reshape(cluster.meta['dims'][0],cluster.meta['dims'][1],-1)
        ## plot ROIs of session 2 compared to one of session 1

        #Cn = cv2.remap(Cn,x_remap,y_remap, interpolation=cv2.INTER_CUBIC)

        x_grid, y_grid = np.meshgrid(np.arange(0., cluster.meta['dims'][0]).astype(np.float32), np.arange(0., cluster.meta['dims'][1]).astype(np.float32))
        x_remap = (x_grid - \
                      cluster.sessions['shift'][s+1,0] + cluster.sessions['shift'][s,0] + \
                      cluster.sessions['flow_field'][s+1,:,:,0] - cluster.sessions['flow_field'][s,:,:,0]).astype('float32')
        y_remap = (y_grid - \
                      cluster.sessions['shift'][s+1,1] + cluster.sessions['shift'][s,1] + \
                      cluster.sessions['flow_field'][s+1,:,:,1] - cluster.sessions['flow_field'][s,:,:,1]).astype('float32')
        # Cn = cv2.remap(Cn,x_remap,y_remap, interpolation=cv2.INTER_CUBIC)

        ax_ROIs2 = plt.axes([0.35,0.55,0.25,0.4])
        ax_ROIs2.imshow(Cn,origin='lower',clim=[0,1])
        n_match = int(cluster.IDs['neuronID'][c,s+1,1])
        for nn in n_close:
            cc = np.where(cluster.IDs['neuronID'][:,s+1,1]==nn)
            # print('SNR: %.2g'%cluster.stats['SNR'][cc,s+1])
            if (not (nn==n_match)) & (cluster.stats['SNR'][cc,s+1]>3):
                A_tmp = cv2.remap(A[...,nn].reshape(cluster.meta['dims']).toarray(),x_remap,y_remap, interpolation=cv2.INTER_CUBIC)
                ax_ROIs2.contour(A_tmp,[0.2*A_tmp.max()],colors='r',linestyles='--',linewidths=1)
        ax_ROIs2.contour(An,[0.2*An.max()],colors='w',linewidths=3)
        A_tmp = cv2.remap(A[...,n_match].reshape(cluster.meta['dims']).toarray(),x_remap,y_remap, interpolation=cv2.INTER_CUBIC)
        ax_ROIs2.contour(A_tmp,[0.2*A_tmp.max()],colors='g',linewidths=3)

        ax_ROIs2.set_xlim([x-margins,x+margins])
        ax_ROIs2.set_ylim([y-margins,y+margins])
        ax_ROIs2.text(x-margins+3,y+margins-5,'Session s+1',bbox=props,fontsize=10)
        ax_ROIs2.set_xticklabels([])
        ax_ROIs2.set_yticklabels([])

        ax_zoom1 = plt.axes([0.075,0.125,0.25,0.275])
        ax_zoom1.hist(D_ROIs.flatten(),np.linspace(0,15,31),facecolor='k',density=True)
        ax_zoom1.set_xlabel('distance [$\mu$m]')
        pl_dat.remove_frame(ax_zoom1,['top','left','right'])
        ax_zoom1.set_ylabel('density')

        ax = plt.axes([0.1,0.345,0.075,0.125])
        plt.hist(D_ROIs.flatten(),np.linspace(0,np.sqrt(2*512**2),101),facecolor='k',density=True)
        ax.set_xlabel('d [$\mu$m]',fontsize=10)
        pl_dat.remove_frame(ax,['top','left','right'])

        D_matches = np.copy(D_ROIs_cross.diagonal())
        np.fill_diagonal(D_ROIs_cross,np.NaN)

        ax_zoom2 = plt.axes([0.35,0.125,0.25,0.275])
        ax_zoom2.hist(D_ROIs_cross.flatten(),np.linspace(0,15,31),facecolor='tab:red',alpha=0.5)
        ax_zoom2.hist(D_ROIs.flatten(),np.linspace(0,15,31),facecolor='k',edgecolor='k',histtype='step')
        ax_zoom2.hist(D_matches,np.linspace(0,15,31),facecolor='tab:green',alpha=0.5)
        ax_zoom2.set_xlabel('distance [$\mu$m]')
        pl_dat.remove_frame(ax_zoom2,['top','left','right'])

        ax = plt.axes([0.38,0.345,0.075,0.125])
        ax.hist(D_ROIs_cross.flatten(),np.linspace(0,np.sqrt(2*512**2),101),facecolor='tab:red',alpha=0.5)
        ax.hist(D_matches,np.linspace(0,np.sqrt(2*512**2),101),facecolor='tab:green',alpha=0.5)
        ax.set_xlabel('d [$\mu$m]',fontsize=10)
        pl_dat.remove_frame(ax,['top','left','right'])


        ax = plt.axes([0.7,0.825,0.25,0.15])#ax_sc1.twinx()
        ax.hist(cluster.stats['match_score'][:,:,0].flat,np.linspace(0,1,51),facecolor='tab:blue',alpha=1,label='$p^*$')
        ax.hist(cluster.stats['match_score'][:,:,1].flat,np.linspace(0,1,51),facecolor='tab:orange',alpha=1,label='max($p\\backslash p^*$)')
        #ax.invert_yaxis()
        ax.set_xlim([0,1])
        ax.set_yticks([])
        ax.set_xlabel('p')
        ax.legend(fontsize=10,bbox_to_anchor=[0.125,1.2],loc='upper left',handlelength=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)


        ax_sc1 = plt.axes([0.7,0.475,0.25,0.21])
        # ax = plt.axes([0.925,0.85,0.225,0.05])#ax_sc1.twiny()
        # ax.set_xticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        ax_sc1.plot(cluster.stats['match_score'][:,:,1].flat,cluster.stats['match_score'][:,:,0].flat,'.',markeredgewidth=0,color='k',markersize=1)
        ax_sc1.plot([0,1],[0,1],'--',color='tab:red',lw=1)
        ax_sc1.plot([0,0.45],[0.5,0.95],'--',color='tab:blue',lw=2)
        ax_sc1.plot([0.45,1],[0.95,0.95],'--',color='tab:blue',lw=2)
        ax_sc1.set_ylabel('$p^{\\asterisk}$')
        ax_sc1.set_xlabel('max($p\\backslash p^*$)')
        ax_sc1.set_xlim([0,1])
        ax_sc1.set_ylim([0,1])
        ax_sc1.spines['top'].set_visible(False)
        ax_sc1.spines['right'].set_visible(False)


        ax_sc2 = plt.axes([0.7,0.125,0.25,0.21])
        #plt.hist(np.nanmean(self.results['p_matched'],1),np.linspace(0,1,51))
        ax = ax_sc2.twinx()
        ax.hist(np.nanmin(cluster.stats['match_score'][:,:,0],1),np.linspace(0,1,51),facecolor='tab:red',alpha=0.3)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = ax_sc2.twiny()
        ax.hist(np.nanmean(cluster.stats['match_score'][:,:,0],axis=1),np.linspace(0,1,51),facecolor='tab:blue',orientation='horizontal',alpha=0.3)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax_sc2.plot(np.nanmin(cluster.stats['match_score'][:,:,0],1),np.nanmean(cluster.stats['match_score'][:,:,0],axis=1),'.',markeredgewidth=0,color='k',markersize=1)
        ax_sc2.set_xlabel('min($p^{\\asterisk}$)')
        ax_sc2.set_ylabel('$\left\langle p^{\\asterisk} \\right\\rangle$')
        ax_sc2.set_xlim([0,1])
        ax_sc2.set_ylim([0,1])
        ax_sc2.spines['top'].set_visible(False)
        ax_sc2.spines['right'].set_visible(False)


        # ax = plt.subplot(248)
        # ax.plot([0,1],[0,1],'--',color='r')
        # ax.scatter(cluster.stats['match_score'][:,:,0],cluster.stats['match_score'][:,:,1],s=1,color='k')
        # ax.set_xlim([0.3,1])
        # ax.set_ylim([-0.05,1])
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        # ax.set_xlabel('matched score',fontsize=14)
        # ax.set_ylabel('2nd best score',fontsize=14)
        # pl_dat.remove_frame(ax,['top'])
        #
        # ax = plt.subplot(244)
        # #ax.hist(cluster.sessions['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='r',alpha=0.5)
        # ax.hist(cluster.stats['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,label='match score')
        # pl_dat.remove_frame(ax,['left','right','top'])
        # ax.yaxis.set_label_position("right")
        # #ax.yaxis.tick_right()
        # ax.set_xlim([0.3,1])
        # ax.set_xticks([])
        # ax.set_ylabel('density',fontsize=14)
        # ax.legend(loc='upper left',fontsize=10)
        #
        # ax = plt.subplot(247)
        # ax.hist(cluster.stats['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,orientation='horizontal',label='2nd best score')
        # #ax.hist(cluster.sessions['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5)
        # pl_dat.remove_frame(ax,['left','bottom','top'])
        # ax.set_ylim([-0.05,1])
        # ax.set_xlim([1.2,0])
        # ax.set_yticks([])
        # ax.legend(loc='upper right',fontsize=10)
        # ax.set_xlabel('density',fontsize=14)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('match_stats')

    if plot_fig[10]:

        print('### Plotting session alignment procedure and statistics ###')

        # s = 10
        # s = s-1

        dims = (512,512)
        com_mean = np.nanmean(cluster.stats['com'][cluster.stats['cluster_bool'],:,:],1)

        W = sstats.norm.pdf(range(dims[0]),dims[0]/2,dims[0]/(0.5*1.96))
        W /= W.sum()
        W = np.sqrt(np.diag(W))
        # x_w = np.dot(W,x)

        y = np.hstack([np.ones((512,1)),np.arange(512).reshape(512,1)])
        y_w = np.dot(W,y)
        x = np.hstack([np.ones((512,1)),np.arange(512).reshape(512,1)])
        x_w = np.dot(W,x)

        pathSession1 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%1])
        ROIs1_ld = loadmat(pathSession1)
        Cn = np.array(ROIs1_ld['A'].sum(1).reshape(dims))
        Cn -= Cn.min()
        Cn /= Cn.max()
        # dims = Cn.shape

        # p_vals = np.zeros((cluster.meta['nSes'],4))*np.NaN
        p_vals = np.zeros((cluster.meta['nSes'],2))*np.NaN
        # fig1 = plt.figure(figsize=(7,5),dpi=300)
        for s in tqdm(np.where(cluster.sessions['bool'])[0][1:]):#cluster.meta['nSes'])):

            # try:
                pathSession2 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
                ROIs2_ld = loadmat(pathSession2,variable_names=['A'])

                Cn2 = np.array(ROIs2_ld['A'].sum(1).reshape(dims))
                Cn2 -= Cn2.min()
                Cn2 /= Cn2.max()
                if cluster.sessions['transpose'][s]:
                    Cn2 = Cn2.T
                # print('adjust session position')

                # t_start = time.time()
                (x_shift,y_shift), flow, (x_grid,y_grid), corr = get_shift_and_flow(Cn,Cn2,dims,projection=None,plot_bool=False)
                (x_shift,y_shift) = cluster.sessions['shift'][s,:]
                flow = cluster.sessions['flow_field'][s,...]

                x_remap = (x_grid - x_shift + flow[...,0])
                y_remap = (y_grid - y_shift + flow[...,1])

                flow_w_y = np.dot(flow[:,:,0],W)
                y0,res,rank,tmp = np.linalg.lstsq(y_w,flow_w_y)
                dy = -y0[0,:]/y0[1,:]
                idx_out = (dy>512) | (dy<0)
                r_y = sstats.linregress(np.where(~idx_out),dy[~idx_out])
                tilt_ax_y = r_y.intercept+r_y.slope*range(512)

                # print((res**2).sum())
                res_y = np.sqrt(((tilt_ax_y-dy)**2).sum())/dims[0]
                # print('y: %.3f'%(np.sqrt(((tilt_ax_y-dy)**2).sum())/dims[0]))

                flow_w_x = np.dot(flow[:,:,1],W)
                x0,res,rank,tmp = np.linalg.lstsq(x_w,flow_w_x)
                dx = -x0[0,:]/x0[1,:]
                idx_out = (dx>512) | (dx<0)
                r_x = sstats.linregress(np.where(~idx_out),dx[~idx_out])
                tilt_ax_x = r_x.intercept+r_x.slope*range(512)
                # print(r_x)
                # print('x:')
                # print((res**2).sum())
                # print('x: %.3f'%(np.sqrt(((tilt_ax_x-dx)**2).sum())/dims[0]))
                res_x = np.sqrt(((tilt_ax_x-dx)**2).sum())/dims[0]
                r = r_y if (res_y < res_x) else r_x
                d = dy if (res_y < res_x) else dx
                tilt_ax = r.intercept+r.slope*range(512)

                com_silent = com_mean[~cluster.status[cluster.stats['cluster_bool'],s,0],:]
                com_active = com_mean[cluster.status[cluster.stats['cluster_bool'],s,1],:]
                com_PCs = com_mean[cluster.status[cluster.stats['cluster_bool'],s,2],:]

                dist_mean = np.abs((r.slope*com_mean[:,0]-com_mean[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))
                dist_silent = np.abs((r.slope*com_silent[:,0]-com_silent[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))
                dist_active = np.abs((r.slope*com_active[:,0]-com_active[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))
                # dist_PCs = np.abs((r.slope*com_PCs[:,0]-com_PCs[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))

                dist = np.abs((r.slope*x_grid-y_grid+r.intercept)/np.sqrt(r.slope**2+1**2))

                # plt.figure()
                # ax_dist = plt.subplot(111)
                # im = ax_dist.imshow(dist,cmap='jet',origin='lower')
                # cb = plt.colorbar(im)
                # cb.set_label('distance [$\mu$m]',fontsize=10)
                # ax_dist.set_xlim([0,dims[0]])
                # ax_dist.set_ylim([0,dims[0]])
                # ax_dist.set_xlabel('x [$\mu$m]')
                # ax_dist.yaxis.tick_right()
                # ax_dist.yaxis.set_label_position("right")
                # ax_dist.set_ylabel('y [$\mu$m]')
                # plt.show(block=False)

                # plt.figure(fig1.number)
                # t_start = time.time()
                r_silent = sstats.ks_2samp(dist_silent,dist_mean)
                r_active = sstats.ks_2samp(dist_active,dist_mean)
                # r_cross = sstats.ks_2samp(dist_active,dist_silent)
                # r_PCs = sstats.ks_2samp(dist_PCs,dist_mean)
                # p_vals[s,:] = [r_silent.pvalue,r_active.pvalue,r_cross.pvalue,r_PCs.pvalue]
                # p_vals[s] = r_cross.pvalue
                p_vals[s,:] = [r_silent.statistic,r_active.statistic]
                # print('time (KS): %.3f'%(time.time()-t_start))
                if s == 9:
                    fig = plt.figure(figsize=(7,5),dpi=300)
                    # plt.figure()
                    C = signal.convolve(Cn-Cn.mean(),Cn2[::-1,::-1]-Cn2.mean(),mode='same')/(np.prod(dims)*Cn.std()*Cn2.std())

                    (x_shift,y_shift), flow, (x_grid,y_grid), corr = get_shift_and_flow(Cn,Cn2,dims,projection=None,plot_bool=False)
                    x_remap = (x_grid - x_shift + flow[...,0])
                    y_remap = (y_grid - y_shift + flow[...,1])

                    Cn2_corr = cv2.remap(Cn2.astype(np.float32), x_remap, y_remap, cv2.INTER_CUBIC)
                    Cn2_corr -= Cn2_corr.min()
                    Cn2_corr /= Cn2_corr.max()
                    props = dict(boxstyle='round', facecolor='w', alpha=0.8)

                    # fig = plt.figure(figsize=(10,5))
                    ax_im1 = plt.axes([0.1,0.625,0.175,0.35])
                    im_col = np.zeros((512,512,3))
                    im_col[:,:,0] = Cn2
                    ax_im1.imshow(im_col,origin='lower')
                    ax_im1.text(50,430,'Session %d'%(s+1),bbox=props,fontsize=8)
                    ax_im1.set_xticks([])
                    ax_im1.set_yticks([])

                    im_col = np.zeros((512,512,3))
                    im_col[:,:,1] = Cn

                    ax_im2 = plt.axes([0.05,0.575,0.175,0.35])
                    ax_im2.imshow(im_col,origin='lower')
                    ax_im2.text(50,430,'Session %d'%1,bbox=props,fontsize=8)
                    ax_im2.set_xticks([])
                    ax_im2.set_yticks([])
                    # ax_im2.set_xlabel('x [px]',fontsize=14)
                    # ax_im2.set_ylabel('y [px]',fontsize=14)
                    sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
                    ax_im2.add_artist(sbar)

                    ax_sShift = plt.axes([0.4,0.575,0.175,0.35])
                    cbaxes = plt.axes([0.4, 0.88, 0.05, 0.02])
                    C -= np.percentile(C,95)
                    C /= C.max()
                    im = ax_sShift.imshow(C,origin='lower',extent=[-dims[0]/2,dims[0]/2,-dims[1]/2,dims[1]/2],cmap='jet',clim=[0,1])
                    cb = fig.colorbar(im,cax = cbaxes,orientation='horizontal')
                    cbaxes.xaxis.set_label_position('top')
                    cbaxes.xaxis.tick_top()
                    cb.set_ticks([0,1])
                    cb.set_ticklabels(['low','high'])
                    cb.set_label('corr.',fontsize=10)
                    ax_sShift.arrow(0,0,float(cluster.sessions['shift'][s,0]),float(cluster.sessions['shift'][s,1]),head_width=1.5,head_length=2,color='k',width=0.1,length_includes_head=True)
                    ax_sShift.text(-13, -13, 'shift: (%d,%d)'%(cluster.sessions['shift'][s,0],cluster.sessions['shift'][s,1]), size=10, ha='left', va='bottom',color='k',bbox=props)

                    #ax_sShift.colorbar()
                    ax_sShift.set_xlim([-15,15])
                    ax_sShift.set_ylim([-15,15])
                    ax_sShift.set_xlabel('$\Delta x$ [px]')
                    ax_sShift.set_ylabel('$\Delta y$ [px]')

                    ax_sShift_all = plt.axes([0.54,0.79,0.1,0.15])
                    for ss in range(nSes):
                        if cluster.sessions['bool'][ss]:
                            col = [0.6,0.6,0.6]
                        else:
                            col = 'tab:red'
                        ax_sShift_all.arrow(0,0,cluster.sessions['shift'][ss,0],cluster.sessions['shift'][ss,1],color=col,linewidth=0.5)
                    ax_sShift_all.arrow(0,0,cluster.sessions['shift'][s,0],cluster.sessions['shift'][s,1],color='k',linewidth=0.5)
                    ax_sShift_all.yaxis.set_label_position("right")
                    ax_sShift_all.yaxis.tick_right()
                    ax_sShift_all.xaxis.set_label_position("top")
                    ax_sShift_all.xaxis.tick_top()
                    ax_sShift_all.set_xlim([-25,50])
                    ax_sShift_all.set_ylim([-25,50])
                    # ax_sShift_all.set_xlabel('x [px]',fontsize=10)
                    # ax_sShift_all.set_ylabel('y [px]',fontsize=10)

                    idxes = 50
                    # tx = dims[0]/2 - 1
                    # ty = tilt_ax_y[int(tx)]
                    ax_OptFlow = plt.axes([0.8,0.625,0.175,0.25])
                    ax_OptFlow.quiver(x_grid[::idxes,::idxes], y_grid[::idxes,::idxes], flow[::idxes,::idxes,0], flow[::idxes,::idxes,1], angles='xy', scale_units='xy', scale=0.1, headwidth=4,headlength=4, width=0.002, units='width')#,label='x-y-shifts')
                    ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),d,':',color='tab:green')
                    # ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),dx,'g:')
                    # ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'g-')
                    ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'-',color='tab:green')

                    ax_OptFlow.set_xlim([0,dims[0]])
                    ax_OptFlow.set_ylim([0,dims[1]])
                    ax_OptFlow.set_xlabel('x [px]')
                    ax_OptFlow.set_ylabel('y [px]')

                    # ax_OptFlow_stats = plt.axes([0.65,0.6,0.075,0.125])
                    # ax_OptFlow_stats.scatter(flow[:,:,0].reshape(-1,1),flow[:,:,1].reshape(-1,1),s=0.2,marker='.',color='k')#,label='xy-shifts')
                    # ax_OptFlow_stats.plot(np.mean(flow[:,:,0]),np.mean(flow[:,:,1]),marker='.',color='r')
                    # ax_OptFlow_stats.set_xlim(-10,10)
                    # ax_OptFlow_stats.set_ylim(-10,10)
                    # ax_OptFlow_stats.set_xlabel('$\Delta$x [px]',fontsize=10)
                    # ax_OptFlow_stats.set_ylabel('$\Delta$y [px]',fontsize=10)
                    # # ax_OptFlow_stats.yaxis.set_label_position("right")
                    # # ax_OptFlow_stats.yaxis.tick_right()
                    # #ax_OptFlow_stats.legend()


                    # dist_mat = np.abs((r.slope*x_grid-y_grid+r.intercept)/np.sqrt(r.slope**2+1**2))
                    # slope_normal = np.array([-r.slope,1])
                    # slope_normal /= np.linalg.norm(slope_normal)
                    # f_perp = np.dot(flow[:,:,:2],slope_normal)
                    # # print(f_perp)
                    # # print(flow[:,:,0]*slope_normal[0] + flow[:,:,1]*slope_normal[1])
                    # h_dat = np.sign(f_perp)*np.sin(np.arccos((dist_mat - np.abs(f_perp))/dist_mat))*dist_mat

                    # ax = plt.axes([0.575,0.125,0.175,0.35])
                    # ax.yaxis.set_label_position("right")
                    # ax.yaxis.tick_right()
                    # im = ax.imshow(h_dat,origin='lower',cmap='jet',clim=[-30,30])
                    # im = ax.imshow(f_perp,origin='lower',cmap='jet',clim=[-3,3])

                    # cbaxes = plt.axes([0.548, 0.3, 0.01, 0.175])
                    # cb = plt.colorbar(im,cax = cbaxes)
                    # cbaxes.yaxis.set_label_position('left')
                    # cbaxes.yaxis.set_ticks_position('left')
                    # cb.set_label('z [$\mu$m]',fontsize=10)

                    ax_sShifted = plt.axes([0.75,0.11,0.2,0.325])
                    im_col = np.zeros((512,512,3))
                    im_col[:,:,0] = Cn
                    im_col[:,:,1] = Cn2_corr
                    ax_sShifted.imshow(im_col,origin='lower')
                    ax_sShifted.text(125,510,'aligned sessions',bbox=props,fontsize=10)
                    ax_sShifted.set_xticks([])
                    ax_sShifted.set_yticks([])

                    ax_scatter = plt.axes([0.1,0.125,0.2,0.3])
                    ax_scatter.scatter(com_silent[:,0],com_silent[:,1],s=0.7,c='k')
                    ax_scatter.scatter(com_active[:,0],com_active[:,1],s=0.7,c='tab:orange')
                    # x_ax = np.linspace(0,dims[0]-1,dims[0])
                    # y_ax = n[0]/n[1]*(p[0]-x_ax) + p[1] + n[2]/n[1]*p[2]
                    ax_scatter.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'-',color='tab:green')
                    # ax_scatter.plot(x_ax,y_ax,'k-')
                    ax_scatter.set_xlim([0,dims[0]])
                    ax_scatter.set_ylim([0,dims[0]])
                    ax_scatter.set_xlabel('x [$\mu$m]')
                    ax_scatter.set_ylabel('y [$\mu$m]')

                    # x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32),
                                                   # np.arange(0., dims[1]).astype(np.float32))

                    ax_hist = plt.axes([0.4,0.125,0.3,0.3])
                    # ax_hist.hist(dist_mean,np.linspace(0,400,21),facecolor='k',alpha=0.5,density=True,label='all neurons')
                    ax_hist.hist(dist_silent,np.linspace(0,400,51),facecolor='k',alpha=0.5,density=True,label='silent')
                    ax_hist.hist(dist_active,np.linspace(0,400,51),facecolor='tab:orange',alpha=0.5,density=True,label='active')
                    ax_hist.legend(loc='lower left',fontsize=8)
                    ax_hist.set_ylabel('density')
                    ax_hist.set_yticks([])
                    ax_hist.set_xlabel('distance from axis [$\mu$m]')
                    ax_hist.set_xlim([0,400])
                    pl_dat.remove_frame(ax_hist,['top','right'])
            # except:
                # pass

        ax_p = plt.axes([0.525,0.325,0.15,0.125])
        ax_p.plot([0,cluster.meta['nSes']],[0.01,0.01],'k--')
        ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],0],'k',linewidth=0.5)
        ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],1],'tab:orange',linewidth=0.5)
        # ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool']],'b')
        #ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],2],'--',color=[0.6,0.6,0.6])
        #ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],3],'g--')
        ax_p.set_yscale('log')
        ax_p.xaxis.set_label_position("top")
        ax_p.yaxis.set_label_position("right")
        ax_p.tick_params(axis='y',which='both',left=False,right=True,labelright=True,labelleft=False)
        ax_p.tick_params(axis='x',which='both',top=True,bottom=False,labeltop=True,labelbottom=False)
        # ax_p.xaxis.tick_top()
        # ax_p.yaxis.tick_right()
        ax_p.set_xlabel('session')
        ax_p.set_ylim([10**(-4),1])
        # ax_p.set_ylim([1,0])
        ax_p.set_ylabel('p-value',fontsize=8,rotation='horizontal',labelpad=-5,y=-0.1)
        pl_dat.remove_frame(ax_p,['bottom','left'])
        # ax_p.tick_params(axis='x',which='both',top=True,bottom=False,labeltop=True,labelbottom=False)

        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('session_align')

    if plot_fig[11]:
        print('### Plotting ROI and cluster statistics of matching ###')

        idx_unsure = cluster.stats['match_score'][...,0]<(cluster.stats['match_score'][...,1]+0.5)

        fig = plt.figure(figsize=(7,4))

        nDisp = 20
        ax_3D = plt.subplot(221,projection='3d')
        ##fig.gca(projection='3d')
        #a = np.arange(30)
        #for c in range(30):
        n_arr = np.random.randint(0,cluster.meta['nC'],nDisp)
        cmap = get_cmap('tab20')
        ax_3D.set_prop_cycle(color=cmap.colors)
        ax_3D.plot(cluster.stats['com'][n_arr,:,0].T,cluster.stats['com'][n_arr,:,1].T,np.arange(nSes),linewidth=3)
        ax_3D.set_xlim([0,512*cluster.para['pxtomu']])
        ax_3D.set_ylim([0,512*cluster.para['pxtomu']])

        ax_3D.set_xlabel('x [$\mu$m]')
        ax_3D.set_ylabel('y [$\mu$m]')
        ax_3D.invert_zaxis()
        # ax_3D.zaxis._axinfo['label']['space_factor'] = 2.8
        ax_3D.set_zlabel('session')

        #ax = plt.subplot(243)
        ax = plt.axes([0.6,0.65,0.175,0.325])
        dx = np.diff(cluster.stats['com'][...,0],axis=1)*cluster.para['pxtomu']
        ax.hist(dx.flatten(),np.linspace(-10,10,101),facecolor='tab:blue',alpha=0.5)
        ax.hist(dx[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='tab:red',alpha=0.5)
        ax.set_xlabel('$\Delta$x [$\mu$m]')
        ax.set_ylabel('density')
        pl_dat.remove_frame(ax,['top','left','right'])
        ax.set_ylim([0,10000])

        #ax = plt.subplot(244)
        ax = plt.axes([0.8,0.65,0.175,0.325])
        dy = np.diff(cluster.stats['com'][...,1],axis=1)*cluster.para['pxtomu']
        ax.hist(dy.flatten(),np.linspace(-10,10,101),facecolor='tab:blue',alpha=0.5)
        ax.hist(dy[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='tab:red',alpha=0.5)
        ax.set_xlabel('$\Delta$y [$\mu$m]')
        pl_dat.remove_frame(ax,['top','left','right'])
        ax.set_ylim([0,10000])

        #ax = plt.axes([0.7,0.875,0.075,0.1])
        #ax.hist(dx.flatten(),np.linspace(-10,10,101),facecolor='k',alpha=0.5)
        #ax.hist(dx[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='r',alpha=0.5)
        #ax.set_xlabel('$\Delta$x [$\mu$m]',fontsize=10)
        #pl_dat.remove_frame(ax,['top','left','right'])
        #ax.set_ylim([0,200])

        #ax = plt.axes([0.9,0.875,0.075,0.1])
        #ax.hist(dy.flatten(),np.linspace(-10,10,101),facecolor='k',alpha=0.5)
        #ax.hist(dy[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='r',alpha=0.5)
        #ax.set_xlabel('$\Delta$y [$\mu$m]',fontsize=10)
        #pl_dat.remove_frame(ax,['top','left','right'])
        #ax.set_ylim([0,200])

        com_ref = np.zeros((nC,2))*np.NaN
        for n in range(nC):
            s_ref = np.where(cluster.status[n,:,1])[0]
            if len(s_ref)>0:
                com_ref[n,:] = cluster.stats['com'][n,s_ref[0],:]
                # print('neuron %d, first session: %d, \tposition: (%.2f,%.2f)'%(n,s_ref[0],com_ref[n,0],com_ref[n,1]))

        ax_mv = plt.axes([0.1,0.11,0.35,0.3])
        # ROI_diff = (cluster.stats['com'].transpose(1,0,2)-cluster.stats['com'][:,0,:]).transpose(1,0,2)#*cluster.para['pxtomu']
        ROI_diff = (cluster.stats['com'].transpose(1,0,2)-com_ref).transpose(1,0,2)#*cluster.para['pxtomu']
        ROI_diff_abs = np.array([np.sqrt(x[:,0]**2+x[:,1]**2) for x in ROI_diff])
        # ROI_diff_abs[~cluster.status[...,1]] = np.NaN

        for n in n_arr:
            ax_mv.plot(range(nSes),ROI_diff_abs[n,:],linewidth=0.5,color=[0.6,0.6,0.6])
        ax_mv.plot(range(nSes),ROI_diff_abs[n,:]*np.NaN,linewidth=0.5,color=[0.6,0.6,0.6],label='displacement')

        pl_dat.plot_with_confidence(ax_mv,range(nSes),np.nanmean(ROI_diff_abs,0),np.nanstd(ROI_diff_abs,0),col='tab:red',ls='-',label='average')
        ax_mv.set_xlabel('session')
        ax_mv.set_ylabel('$\Delta$d [$\mu$m]')
        ax_mv.set_ylim([0,11])
        ax_mv.legend(fontsize=10)
        pl_dat.remove_frame(ax_mv,['top','right'])

        idx_c_unsure = idx_unsure.any(1)

        ax_mv_max = plt.axes([0.6,0.11,0.35,0.35])
        ROI_max_mv = np.nanmax(ROI_diff_abs,1)
        ax_mv_max.hist(ROI_max_mv,np.linspace(0,20,41),facecolor='tab:blue',alpha=0.5,label='certain')
        ax_mv_max.hist(ROI_max_mv[idx_c_unsure],np.linspace(0,20,41),facecolor='tab:red',alpha=0.5,label='uncertain')
        ax_mv_max.set_xlabel('max($\Delta$d) [$\mu$m]')
        ax_mv_max.set_ylabel('# cluster')
        ax_mv_max.legend(fontsize=10)

        pl_dat.remove_frame(ax_mv_max,['top','right'])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('ROI_positions')

    if plot_fig[12]:
        print('### Plotting positions of silent neurons ### ')

        ## get average position of neuron
        com_mean = np.nanmean(cluster.stats['com'][cluster.stats['cluster_bool'],:,:],1)
        #idx_keep = np.all((com_mean[:,:2] < 512) & (com_mean[:,:2]>0),1)
        #print(idx_keep.shape)
        #print(com_mean[~idx_keep,:])



        #nA = 10 ## number zones per row
        #N = com_mean.shape[0]
        #NA_exp = N/nA**2
        #A = np.histogram2d(com_mean[cluster.stats['cluster_bool'],0],com_mean[cluster.stats['cluster_bool'],1],np.linspace(0,512,nA+1))[0]
        #rA = A/NA_exp

        #D = 1-np.sqrt(np.sum((rA-1)**2))/nA**2

        #KS_dist = np.zeros(cluster.meta['nSes'])

        dims = cluster.meta['dims']
        #A_length = [cluster.meta['dims'][0]/nA,cluster.meta['dims'][1]/nA]
        p_vals = np.zeros((cluster.meta['nSes'],4))

        plt.figure()
        for s0 in tqdm(range(cluster.meta['nSes'])):
            #print(s)
            s = s0
            com_silent = com_mean[cluster.status[cluster.stats['cluster_bool'],s,0],:]
            com_active = com_mean[cluster.status[cluster.stats['cluster_bool'],s,1],:]
            com_PCs = com_mean[np.any(cluster.status[cluster.stats['cluster_bool'],s,2:],-1),:]

            n = cluster.sessions['rotation_normal'][s,:]
            p = cluster.sessions['rotation_anchor'][s,:]

            intercept = n.dot(p)/n[1]
            slope = -n[0]/n[1]

            dist_mean = np.abs((slope*com_mean[:,0]-com_mean[:,1]+intercept)/np.sqrt(slope**2+1**2))
            dist_silent = np.abs((slope*com_silent[:,0]-com_silent[:,1]+intercept)/np.sqrt(slope**2+1**2))
            dist_active = np.abs((slope*com_active[:,0]-com_active[:,1]+intercept)/np.sqrt(slope**2+1**2))
            dist_PCs = np.abs((slope*com_PCs[:,0]-com_PCs[:,1]+intercept)/np.sqrt(slope**2+1**2))

            r_silent = sstats.ks_2samp(dist_silent,dist_mean)
            r_active = sstats.ks_2samp(dist_active,dist_mean)
            r_cross = sstats.ks_2samp(dist_active,dist_silent)
            r_PCs = sstats.ks_2samp(dist_PCs,dist_mean)
            p_vals[s,:] = [r_silent.pvalue,r_active.pvalue,r_cross.pvalue,r_PCs.pvalue]
            #dist = KS_test(dist_mean,dist_silent)

            #np.com_mean
            #E_stat_test(com_mean[:,:2],com_mean[:,:2])
            #print('KS_test dist: %5.3g'%dist)
            #E_stat_test(com_mean[:,:2],com_silent[:,:2])
            #dist = sstats.energy_distance(com_mean[:,:2],com_silent[:,:2])
            #print('from sstats: %5.3g'%dist)
            #E_stat_test(com_mean[:,:2],com_active[:,:2])
            #E_stat_test(com_active[:,:2],com_silent[:,:2])
            #zone = com_mean / A_length

            #NA = cluster.status[

            #A = np.histogram2d(com_mean[:,0],com_mean[:,1],np.linspace(0,512,nA+1))


            #print(s)
            #print(s//3+1)
            if s == 10:
                ax_scatter = plt.subplot(221)
                ax_scatter.scatter(com_silent[:,0],com_silent[:,1],s=1,c='r')
                ax_scatter.scatter(com_active[:,0],com_active[:,1],s=1,c='b',cmap='jet')
                x_ax = np.linspace(0,dims[0]-1,dims[0])
                y_ax = n[0]/n[1]*(p[0]-x_ax) + p[1] + n[2]/n[1]*p[2]
                ax_scatter.plot(x_ax,y_ax,'k-')
                ax_scatter.set_xlim([0,dims[0]])
                ax_scatter.set_ylim([0,dims[0]])
                ax_scatter.set_xlabel('x [$\mu$m]',fontsize=14)
                ax_scatter.set_ylabel('y [$\mu$m]',fontsize=14)

                x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32),
                                               np.arange(0., dims[1]).astype(np.float32))
                dist = np.abs((slope*x_grid-y_grid+intercept)/np.sqrt(slope**2+1**2))
                ax_dist = plt.subplot(222)
                im = ax_dist.imshow(dist,cmap='jet',origin='lower')
                cb = plt.colorbar(im)
                cb.set_label('distance [$\mu$m]',fontsize=10)
                ax_dist.set_xlim([0,dims[0]])
                ax_dist.set_ylim([0,dims[0]])
                ax_dist.set_xlabel('x [$\mu$m]',fontsize=14)
                #ax_dist.yaxis.tick_right()
                #ax_dist.yaxis.set_label_position("right")
                ax_dist.set_ylabel('y [$\mu$m]',fontsize=14)

                ax_hist = plt.subplot(223)
                ax_hist.hist(dist_mean,np.linspace(0,400,21),facecolor='k',alpha=0.5,density=True,label='all neurons')
                ax_hist.hist(dist_silent,np.linspace(0,400,21),facecolor='r',alpha=0.5,density=True,label='silent')
                ax_hist.hist(dist_active,np.linspace(0,400,21),facecolor='b',alpha=0.5,density=True,label='active')
                ax_hist.legend(loc='upper right',fontsize=10)
                ax_hist.set_ylabel('density',fontsize=14)
                ax_hist.set_yticks([])
                ax_hist.set_xlabel('distance from axis [$\mu$m]',fontsize=14)
            #plt.subplot(2,6,s0+1+6)

            #
            #plt.colorbar()
            #
            #plt.xlim([0,512])
            #plt.ylim([0,512])

            #height_silent = z_from_point_normal_plane(com_silent[:,0],com_silent[:,1],cluster.sessions['rotation_anchor'][s,:],cluster.sessions['rotation_normal'][s,:])
            #height_active = z_from_point_normal_plane(com_active[:,0],com_active[:,1],cluster.sessions['rotation_anchor'][s,:],cluster.sessions['rotation_normal'][s,:])
            #p['silent']['mean'] = np.histogram(height_silent,np.linspace(0,80,101),density=True)[0]
            #p_active = np.histogram(cluster.sessions['com'][cluster.status[:,s,1],s,2],np.linspace(0,80,101),density=True)[0]
            #KS_dist[s] = KS_test(p['silent']['mean'],p_active)
            #plt.title('Session %d'%(s+1))
        #plt.plot(KS_dist)

        ax_p = plt.subplot(224)
        ax_p.plot([0,cluster.meta['nSes']],[0.1,0.1],'k--')
        ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],0],'r')
        ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],1],'b')
        #ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],2],'--',color=[0.6,0.6,0.6])
        #ax_p.plot(np.where(cluster.sessions['bool'])[0],p_vals[cluster.sessions['bool'],3],'g--')
        ax_p.yaxis.tick_right()
        ax_p.yaxis.set_label_position("right")
        ax_p.set_xlabel('session',fontsize=14)
        ax_p.set_yscale('log')
        ax_p.set_ylim([10**(-2),1])
        ax_p.set_ylabel('p-value (KS-test)',fontsize=14)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('session_align_neuroncheck')


    if plot_fig[13]:

        nDisp = 500
        plt.figure()
        ax = plt.subplot(111)#,projection='3d')

        for n in range(nDisp):
            sc = ax.plot(cluster.stats['com'][n,::5,0],cluster.stats['com'][n,::5,1],'k-')#,c=range(0,nSes,5))#,cmap='jet')#,cluster.stats['com'][n,::5,2]
        #plt.colorbar(sc)

        plt.show(block=False)


    if plot_fig[14]:
        ### plot various light statistics
        for i in range(11,21):
            PC_para = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_para.mat'%i,squeeze_me=True)
            PC_status = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_status.mat'%i,squeeze_me=True)
            PC_firing = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_firingstats.mat'%i,squeeze_me=True)
            idx = (PC_status['Bayes_factor'][:,0,0]>(1/2)) & (PC_para['parameter'][:,0,2,0]>0) & ((PC_para['parameter'][:,0,1,0]/PC_para['parameter'][:,0,0,0])>0.5) & (PC_status['MI_p_value']<1)
            plt.figure(); plt.subplot(221);
            plt.hist(PC_para['parameter'][idx,:,3,0].flat,np.linspace(0,100,101));plt.ylim([0,20]);plt.title('session %02d, nPC=%d'%(i,idx.sum()));
            plt.subplot(222);
            plt.plot(np.ones(2)*np.nanmean(PC_para['parameter'][idx,:,2,0]),[0,1],'r')
            plt.hist(PC_para['parameter'][idx,:,2,0].flat,np.linspace(0,30,51),cumulative=True,density=True,histtype='step');
            plt.subplot(223);
            plt.plot(PC_para['parameter'][idx,:,3,0].flat,PC_para['parameter'][idx,:,2,0].flat,'ko')
            plt.subplot(224);
            idx_sort = np.argsort(PC_para['parameter'][idx,0,3,0])
            fmap = PC_firing['map'][idx,:][idx_sort,:]
            fmap = sp.ndimage.gaussian_filter(fmap,1)
            fmap-=np.nanmin(fmap,1)[:,np.newaxis]
            fmap/=np.nanmax(fmap,1)[:,np.newaxis]
            plt.imshow(fmap,aspect='auto',cmap='jet')
            plt.show(block=False)



    if plot_fig[15]:
        plt.figure()
        s=20
        cfact=5
        idxes = 100
        for ds in range(1,10):
            PC_para = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_para.mat'%s,squeeze_me=True)
            PC_status = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_status.mat'%s,squeeze_me=True)
            PC_firing = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_firingstats.mat'%s,squeeze_me=True)

            PC_para2 = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_para.mat'%(s+ds),squeeze_me=True)
            PC_status2 = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_status.mat'%(s+ds),squeeze_me=True)
            PC_firing2 = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_firingstats.mat'%(s+ds),squeeze_me=True)

            #idx = (PC_status['Bayes_factor'][:,0,0]>(1/2)) & (PC_para['parameter'][:,0,2,0]>0) & ((PC_para['parameter'][:,0,1,0]/PC_para['parameter'][:,0,0,0])>0.5) & (PC_status['MI_p_value']<1)
            #plt.figure(); plt.subplot(221);

            #idx_sort = np.argsort(PC_para['parameter'][idx,0,3,0])
            fmap = PC_firing['map'][:idxes,:]
            fmap = sp.ndimage.gaussian_filter(fmap,1)
            fmap2 = PC_firing2['map'][:idxes,:]
            fmap2 = sp.ndimage.gaussian_filter(fmap2,1)
            fmap-=np.nanmin(fmap,1)[:,np.newaxis]
            fmap/=np.nanmax(fmap,1)[:,np.newaxis]
            fmap2-=np.nanmin(fmap2,1)[:,np.newaxis]
            fmap2/=np.nanmax(fmap2,1)[:,np.newaxis]

            print(fmap2.shape)
            nbin = fmap.shape[1]
            pcorr = np.zeros(nbin)

            for i in range(nbin):
                fmap_c = np.nansum(fmap[max(0,i-cfact):min(i+cfact,nbin),:],0)
                fmap2_c = np.nansum(fmap2[max(0,i-cfact):min(i+cfact,nbin),:],0)
                mask = (~np.isnan(fmap_c)) & (~np.isnan(fmap2_c))
                pcorr[i] = np.corrcoef(fmap_c[mask],fmap2_c[mask])[1,0]

            plt.subplot(3,3,ds)
            plt.hist(pcorr,np.linspace(-0.5,1,21))
            plt.plot([np.mean(pcorr),np.mean(pcorr)],[0,40],'r')
        plt.show(block=False)



    if plot_fig[16]:

        print('### plot time dependence of dynamics ###')

        ### ds > 0
        p = {}

        t_start = time.time()
        s1_shifts,s2_shifts,f1,f2 = np.unravel_index(cluster.compare['pointer'].col,(nSes,nSes,cluster.meta['field_count_max'],cluster.meta['field_count_max']))

        maxSes = 10
        dT_shifts = cluster.para['t_measures'][s2_shifts]-cluster.para['t_measures'][s1_shifts]

        print(cluster.para['t_measures'][s2_shifts])
        #return
        c_shifts = cluster.compare['pointer'].row
        print(c_shifts.shape)
        print(dT_shifts.shape)
        dT_arr = [4,20,24,28,44,48,52,64,68,72,84,88,92]

        N_stable = np.zeros(nSes)*np.NaN
        N_total = np.zeros(nSes)*np.NaN     ### number of PCs which could be stable
        # fig = plt.figure()

        p_rec = {'act':     np.zeros((nSes,nSes))*np.NaN,
                 'PC':      np.zeros((nSes,nSes))*np.NaN,
                 'PF':      np.zeros((nSes,nSes))*np.NaN}

        for ds in range(1,nSes):#min(nSes,30)):
            session_bool = np.where(np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False))[0]

            for s1 in session_bool:
                overlap = cluster.status[cluster.status[:,s1,1],s1+ds,1].sum(0).astype('float')
                N_ref = cluster.status[:,s1,1].sum(0)
                p_rec['act'][ds,s1] = (overlap/N_ref)

                overlap = cluster.status[cluster.status[:,s1,2],s1+ds,2].sum(0).astype('float')
                N_ref = cluster.status[cluster.status[:,s1,2],s1+ds,1].sum(0)
                p_rec['PC'][ds,s1] = (overlap/N_ref)

                Ds = s2_shifts-s1_shifts
                idx = np.where((s1_shifts==s1) & (Ds==ds))[0]

                N_data = len(idx)

                idx_shifts = cluster.compare['pointer'].data[idx].astype('int')-1
                shifts = cluster.compare['shifts'][idx_shifts]
                N_stable = (np.abs(shifts)<(1.96*cluster.stability['all']['mean'][ds,2])).sum()

                p_rec['PF'][ds,s1] = N_stable/N_data
                # N_total[ds] = cluster.status_fields[:,session_bool,:].sum()

        # print(recurr)
        # return recurr
        trials = np.cumsum(cluster.sessions['trial_ct'])

        diff = {'t':        cluster.para['t_measures'][np.newaxis,cluster.sessions['bool']]-cluster.para['t_measures'][cluster.sessions['bool'],np.newaxis],
                's':        np.where(cluster.sessions['bool'])[0][np.newaxis,:] - np.where(cluster.sessions['bool'])[0][:,np.newaxis],
                'trial':    (trials[np.newaxis,cluster.sessions['bool']] - trials[cluster.sessions['bool'],np.newaxis]) // 10}

        s_good = np.where(cluster.sessions['bool'])[0]

        ### test same ds, different dt

        ds_arr = np.unique(np.triu(diff['s']))#[1:]
        dt_arr = np.unique(np.triu(diff['t']))#[1:]

        def calc_pval(diff,p_rec,key1,key1_arr,key2):

            s_good = np.where(cluster.sessions['bool'])[0]
            nSteps = len(key1_arr)
            pval = {'act':  np.zeros(nSteps)*np.NaN,
                    'PC':   np.zeros(nSteps)*np.NaN,
                    'PF':   np.zeros(nSteps)*np.NaN}

            nt = len(dt_arr)
            p = {'act':  np.zeros((nSteps,nt))*np.NaN,
                 'PC':   np.zeros((nSteps,nt))*np.NaN,
                 'PF':   np.zeros((nSteps,nt))*np.NaN}


            for i,dx in enumerate(key1_arr):
                # print('ds: %d'%ds)
                dy_tmp = np.unique(diff[key2][diff[key1]==dx])

                for key in ['act','PC','PF']:
                    tmp = []
                    for dy in dy_tmp:
                        s1,s2 = np.where((diff[key1]==dx) & (diff[key2]==dy))
                        s1 = s_good[s1]
                        s2 = s_good[s2]
                        ds = s2[0]-s1[0]

                        if len(idx)>1:
                            tmp.append(p_rec[key][ds,s1])

                        dt = np.where(dt_arr == (cluster.para['t_measures'][s2[0]] - cluster.para['t_measures'][s1[0]]))[0][0]

                        p[key][ds,dt] = p_rec[key][ds,s1].mean()

                    res = sstats.f_oneway(*tmp)
                    pval[key][i] = res.pvalue

            return pval,p

        pval_s, p_s = calc_pval(diff,p_rec,'s',ds_arr,'t')
        pval_t, p_t = calc_pval(diff,p_rec,'t',dt_arr,'s')


        fig = plt.figure(figsize=(7,4),dpi=300)
        ax1 = plt.axes([0.12,0.11,0.35,0.24])
        ax1.plot([0,ds_arr[-1]],[0.01,0.01],'k--')
        ax1.plot(ds_arr,pval_s['act'],'ko',markersize=2,label='activation')
        ax1.plot(ds_arr,pval_s['PC'],'bo',markersize=2,label='coding')
        ax1.plot(ds_arr,pval_s['PF'],'ro',markersize=2,label='field stability')
        ax1.set_yscale('log')
        ax1.set_ylim([0.1*10**(-5),1])
        ax1.set_xlim([0,10.5])
        ax1.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=True)
        ax1.set_xlabel('session difference $\Delta s$')
        ax1.set_ylabel('p-value')
        ax1.legend(loc='lower right',fontsize=8,bbox_to_anchor=[0.9,0])

        ax2 = plt.axes([0.525,0.11,0.35,0.24])
        ax2.plot([0,dt_arr[-1]],[0.01,0.01],'k--')
        ax2.plot(dt_arr,pval_t['act'],'ko',markersize=2)
        ax2.plot(dt_arr,pval_t['PC'],'bo',markersize=2)
        ax2.plot(dt_arr,pval_t['PF'],'ro',markersize=2)
        ax2.set_yscale('log')
        ax2.set_ylim([0.1*10**(-5),1])
        ax2.set_xlim([0,160])
        ax2.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=False)
        ax2.set_xlabel('time difference $\Delta t$ [h]')

        maxSes = 21
        w_bar = 0.05
        offset_bar = ((maxSes+1)%2)*w_bar/2 + (maxSes//2 - 1)*w_bar

        ax_act = plt.axes([0.12,0.7,0.35,0.25],sharex=ax1)
        ax_PF = plt.axes([0.12,0.4,0.35,0.25],sharex=ax1)
        color_t = iter(plt.cm.rainbow(np.linspace(0,1,maxSes)))
        for i in range(1,maxSes):
            col = next(color_t)
            ax_act.bar(ds_arr-offset_bar+i*w_bar,p_s['act'][:,i],width=w_bar,facecolor=col)
            ax_PF.bar(ds_arr-offset_bar+i*w_bar,p_s['PF'][:,i],width=w_bar,facecolor=col)
            # plt.errorbar(ds_arr-offset_bar+i*w_bar,cluster.stability_dT[dT]['mean'][:maxSes,1],cluster.stability_dT[dT]['std'][:maxSes,1],fmt='none',ecolor='r')
        # ax.set_xlim([0,15])
        plt.setp(ax_act.get_xticklabels(), visible=False)
        plt.setp(ax_PF.get_xticklabels(), visible=False)
        ax_act.set_yticks(np.linspace(0,1,3))
        ax_PF.set_yticks(np.linspace(0,1,3))
        ax_act.tick_params(axis='y',which='both',left=True,right=True,labelright=True,labelleft=False)
        ax_PF.tick_params(axis='y',which='both',left=True,right=True,labelright=True,labelleft=False)
        ax_act.set_ylim([0,1])
        ax_PF.set_ylim([0,1])
        pl_dat.remove_frame(ax_act,['top'])
        pl_dat.remove_frame(ax_PF,['top'])
        ax_act.plot(0,np.NaN,label='activation recurrence $p_{\\alpha}$')
        ax_PF.plot(0,np.NaN,label='field stability $r_{stable}^*$')
        ax_act.legend(loc='upper right',handlelength=0,fontsize=10,bbox_to_anchor=[1,1.1])
        ax_PF.legend(loc='upper right',handlelength=0,fontsize=10,bbox_to_anchor=[1,1.1])

        rainbow = plt.get_cmap('rainbow')
        cNorm = colors.Normalize(vmin=dt_arr[1],vmax=dt_arr[maxSes])
        scalarMap = plt.cm.ScalarMappable(norm=cNorm,cmap=rainbow)
        cbaxes = plt.axes([0.09,0.4,0.01,0.55])
        cb = fig.colorbar(scalarMap,cax=cbaxes,orientation='vertical')
        cbaxes.yaxis.tick_left()
        cbaxes.yaxis.set_label_position('left')
        cbaxes.set_ylabel('$\Delta t$')
        # plt.legend(ncol=3)

        ax_act = plt.axes([0.525,0.7,0.35,0.25],sharex=ax2)
        ax_PF = plt.axes([0.525,0.4,0.35,0.25],sharex=ax2)
        maxSes = 11
        w_bar = 0.4
        offset_bar = ((maxSes+1)%2)*w_bar/2 + (maxSes//2 - 1)*w_bar
        color_s = iter(plt.cm.rainbow(np.linspace(0,1,maxSes)))
        for i in range(1,maxSes):
            col = next(color_s)
            ax_act.bar(dt_arr-offset_bar+i*w_bar,p_s['act'][i,:],width=w_bar,facecolor=col)
            ax_PF.bar(dt_arr-offset_bar+i*w_bar,p_s['PF'][i,:],width=w_bar,facecolor=col)
            # plt.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability_dT[dT]['mean'][:maxSes,1],cluster.stability_dT[dT]['std'][:maxSes,1],fmt='none',ecolor='r')
        plt.setp(ax_act.get_xticklabels(), visible=False)
        plt.setp(ax_PF.get_xticklabels(), visible=False)
        # ax.set_xlim([0,200])
        ax_act.set_ylim([0,1])
        ax_PF.set_ylim([0,1])
        ax_act.set_yticks(np.linspace(0,1,3))
        ax_PF.set_yticks(np.linspace(0,1,3))
        ax_act.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=False)
        ax_PF.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=False)
        pl_dat.remove_frame(ax_act,['top'])
        pl_dat.remove_frame(ax_PF,['top'])

        cNorm = colors.Normalize(vmin=ds_arr[1],vmax=ds_arr[maxSes])
        scalarMap = plt.cm.ScalarMappable(norm=cNorm,cmap=rainbow)
        cbaxes = plt.axes([0.9,0.4,0.01,0.55])
        cb = fig.colorbar(scalarMap,cax=cbaxes,orientation='vertical')
        cbaxes.set_ylabel('$\Delta s$')
        # plt.legend(ncol=3)


        # plt.subplot(313)
        # plt.plot(dtrial_arr,pval_dtr,'ro')
        # plt.plot(dtrial_arr,pval_rec_dtr,'ko')
        # plt.plot(dtrial_arr,pval_recPC_dtr,'bo')
        # plt.ylim([0,1])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('time_dependence')
        # for

        # s_good[np.where((t_diff==4)]
        # return p_stable, t_diff, s_diff



    if plot_fig[17]:


        ### get place field max firing rate
        #for c in range(cluster.meta['nC']):
          #for s in range(cluster.meta['nSes']):


        print('test field width as well')
        print('test peak firing rate as well')

        s1,s2,f1,f2 = np.unravel_index(cluster.compare['pointer'].col,(cluster.meta['nSes'],cluster.meta['nSes'],cluster.meta['field_count_max'],cluster.meta['field_count_max']))
        idx_ds1 = np.where(s2-s1 == 1)[0]

        c_ds1 = cluster.compare['pointer'].row[idx_ds1]
        s1_ds1 = s1[idx_ds1]
        f1_ds1 = f1[idx_ds1]
        idx_shifts_ds1 = cluster.compare['pointer'].data[idx_ds1].astype('int')-1
        shifts_ds1 = cluster.compare['shifts'][idx_shifts_ds1]

        idx_stable_ds1 = np.where(np.abs(shifts_ds1) < 6)[0]
        idx_relocate_ds1 = np.where(np.abs(shifts_ds1) > 12)[0]

        c_stable = c_ds1[idx_stable_ds1]
        s1_stable = s1_ds1[idx_stable_ds1]
        f_stable = f1_ds1[idx_stable_ds1]
        rel_stable = cluster.fields['reliability'][c_stable,s1_stable,f_stable]
        MI_stable = cluster.stats['MI_value'][c_stable,s1_stable]
        Isec_stable = cluster.stats['Isec_value'][c_stable,s1_stable]
        fr_stable = cluster.stats['firingrate'][c_stable,s1_stable]

        c_relocate = c_ds1[idx_relocate_ds1]
        s1_relocate = s1_ds1[idx_relocate_ds1]
        f_relocate = f1_ds1[idx_relocate_ds1]
        rel_relocate = cluster.fields['reliability'][c_relocate,s1_relocate,f_relocate]
        MI_relocate = cluster.stats['MI_value'][c_relocate,s1_relocate]
        Isec_relocate = cluster.stats['Isec_value'][c_relocate,s1_relocate]
        fr_relocate = cluster.stats['firingrate'][c_relocate,s1_relocate]

        idx_loosePC = np.where(np.diff(cluster.status[...,2].astype('int'),1)==-1)
        rel_instable = np.nanmax(cluster.fields['reliability'][idx_loosePC[0],idx_loosePC[1],:],-1)
        MI_instable = cluster.stats['MI_value'][idx_loosePC]
        Isec_instable = cluster.stats['Isec_value'][idx_loosePC]
        fr_instable = cluster.stats['firingrate'][idx_loosePC]

        idx_nPC = np.where(cluster.status[...,1] & ~cluster.status[...,2])
        #rel_instable = np.nanmax(cluster.fields['reliability'][idx_loosePC[0],idx_loosePC[1],:],-1)
        MI_nPC = cluster.stats['MI_value'][idx_nPC]
        Isec_nPC = cluster.stats['Isec_value'][idx_nPC]
        fr_nPC = cluster.stats['firingrate'][idx_nPC]


        col_stable = [0,0.5,0]
        plt.figure(figsize=(6,2.5))
        ax = plt.subplot(132)
        ax.hist(rel_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable)
        #ax.hist(rel_relocate,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='b')
        ax.hist(rel_instable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r')
        #plt.hist(rel_nPC,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':')
        #rel_all = cluster.fields['reliability']
        #rel_all[~cluster.status_fields] = np.NaN
        #rel_all = rel_all[cluster.status[...,2],...]
        #ax.hist(rel_all.flat,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        ax.set_xlabel('reliability [%]',fontsize=14)
        ax.set_xlim([0,1])
        ax.set_yticks([])

        ax = plt.subplot(133)
        plt.hist(MI_nPC,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':',label='nPC')
        plt.hist(MI_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        plt.hist(MI_instable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        ax.set_xlabel('MI [bits]',fontsize=14)
        ax.set_xlim([0,1])
        ax.set_yticks([])
        ax.legend(fontsize=10,loc='lower right')


        #ax = plt.subplot(133)
        #maxrate_stable = cluster.fields['max_rate'][c_stable,s1_stable,f_stable]
        ##idx_loosePC = np.where(np.diff(cluster.status[...,2].astype('int'),1)==-1)
        #maxrate_instable = np.nanmax(cluster.fields['max_rate'][idx_loosePC[0],idx_loosePC[1],:],-1)
        #plt.hist(maxrate_stable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        #plt.hist(maxrate_instable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        #ax.set_xlabel('$\\nu_{max}$',fontsize=14)
        #ax.set_xlim([0,20])

        ax = plt.subplot(131)
        width_stable = cluster.fields['width'][c_stable,s1_stable,f_stable,0]
        #idx_loosePC = np.where(np.diff(cluster.status[...,2].astype('int'),1)==-1)
        width_instable = np.nanmax(cluster.fields['width'][idx_loosePC[0],idx_loosePC[1],:,0],-1)
        plt.hist(width_stable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        plt.hist(width_instable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        ax.set_xlabel('$\sigma$ [bins]',fontsize=14)
        ax.set_xlim([0,20])
        ax.set_yticks(np.linspace(0,1,3))
        ax.set_ylabel('cdf',fontsize=14)

        #plt.subplot(224)
        #plt.hist(Isec_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable)
        #plt.hist(Isec_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r')
        #MI_all = cluster.stats['Isec_value']
        #MI_all = MI_all[cluster.status[...,2]]
        #plt.hist(MI_all.flat,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        #plt.xlabel('I/sec')

        #ax = plt.subplot(131)
        ##a,b = bootstrap_data(lambda x : (np.cumsum(np.histogram(x,np.linspace(0,3,51))[0])/len(x),np.NaN),fr_stable,1000)
        ##pl_dat.plot_with_confidence(ax,np.linspace(0,3,51)[:-1],a,b,col='k',ls='-')
        #plt.hist(fr_nPC,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':',label='nPC')
        #plt.hist(fr_stable,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        ##plt.hist(fr_relocate,np.linspace(0,3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='b')
        #plt.hist(fr_instable,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')

        #fr_all = cluster.stats['firingrate']
        #fr_all = fr_all[cluster.status[...,2]]
        ##a,b = bootstrap_data(lambda x : (np.cumsum(np.histogram(x,np.linspace(0,3,51))[0])/len(x),np.NaN),fr_all,1000)
        ##pl_dat.plot_with_confidence(ax,np.linspace(0,3,51)[:-1],a,b,col='r',ls='-')
        ##ax.hist(fr_all.flat,np.linspace(0,3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        #ax.set_xlabel('activity [Hz]',fontsize=14)
        #ax.set_ylabel('cdf',fontsize=14)
        #ax.set_xlim([0,5])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('codingChange_stats')

    if plot_fig[18]:

        fig = plt.figure()
        ax = plt.subplot(111)

        fields = np.zeros((nbin,nSes))
        for i,s in enumerate(np.where(cluster.sessions['bool'])[0]):
            idx_PC = np.where(cluster.fields['status'][:,s,:]>=3)
            # fields[s,:] = np.nansum(cluster.fields['p_x'][:,s,:,:],1).sum(0)
            fields[:,s] = np.nansum(cluster.fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
        fields /= fields.max(0)
            # fields[:,s] /= fields[:,s].sum()

        fields = gauss_smooth(fields,(2,2))
        im = ax.imshow(fields,origin='lower',cmap='jet')#,clim=[0,1])
        plt.colorbar(im)
        ax.set_xlabel('session')
        ax.set_ylabel('position [bins]')

        plt.show(block=False)


        # s = 10
        ds = 1
        block_size = 10
        plt.figure(figsize=(7,5),dpi=300)
        for s in np.where(cluster.sessions['bool'])[0][:-ds]:#range(5,15):
            if (s%block_size)==0:
                if (s//block_size)>0:
                    ax = plt.subplot(3,4,s//block_size)
                    remapping /= remapping.max()/2
                    ax.imshow(remapping,origin='lower',clim=[0,1],cmap='hot')
                    ax.text(5,90,'Sessions %d-%d'%(s-block_size,s),color='w',fontsize=8)
                    # plt.colorbar()
                remapping = np.zeros((nbin,nbin))

            for c in np.where(cluster.stats['cluster_bool'])[0]:
                if cluster.status[c,s,2] & cluster.status[c,s+ds,2]:
                    for f in np.where(cluster.fields['status'][c,s,:])[0]:
                        for ff in np.where(cluster.fields['status'][c,s+ds,:])[0]:
                            remapping[int(cluster.fields['location'][c,s,f,0]),:] += cluster.fields['p_x'][c,s+ds,ff,:]


        plt.show(block=False)
        # print(remapping.sum(1))
    #print(np.where(cluster.compare['inter_coding'][:,1]==0)[0])
    #print('search for cases, where the neuron loses its  coding ability -> lower MI / lower fr / ...?')

    if plot_fig[19]:
        print('### plot interaction of different dynamical hierarchies ###')

        fig = plt.figure(figsize=(7,5),dpi=300)

        plt.figtext(0.15,0.8,'activation',fontsize=14)


        steps = 4
        p_pre = np.zeros((nSes,2,2,steps+1))*np.NaN
        p_post = np.zeros((nSes,2,2,steps+1))*np.NaN
        for ds in range(1,steps+1):
            # ds = 2  ## directly before sessions
            session_bool = np.pad(cluster.sessions['bool'][:-ds],(ds,0),constant_values=False) & np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False)

            ### activity -> coding
            ## what's the general state before obtaining a place field? (active / silent?; above chance level?

            for s in np.where(session_bool)[0]:
                ## get session-wise prob of neuron being active before coding / non-coding (for statistics)
                p_pre[s,0,0,ds] = cluster.status[cluster.status[:,s,1],s-ds,1].mean()
                p_pre[s,0,1,ds] = cluster.status[cluster.status[:,s,2],s-ds,1].mean()

                p_pre[s,1,0,ds] = cluster.status[~cluster.status[:,s,1],s-ds,1].mean()
                p_pre[s,1,1,ds] = cluster.status[(~cluster.status[:,s,2])&cluster.status[:,s,1],s-ds,1].mean()

                p_post[s,0,0,ds] = cluster.status[cluster.status[:,s,1],s+ds,1].mean()
                p_post[s,0,1,ds] = cluster.status[cluster.status[:,s,2],s+ds,1].mean()

                p_post[s,1,0,ds] = cluster.status[~cluster.status[:,s,1],s+ds,1].mean()
                p_post[s,1,1,ds] = cluster.status[(~cluster.status[:,s,2])&cluster.status[:,s,1],s+ds,1].mean()

        # print(np.nanmean(p_pre[...,1],0))
        # print(np.nanmean(p_post[...,1],0))
        # plt.figure()
        #
        # plt.plot(range(nSes),p_pre[:,0],'k--')
        # plt.plot(range(nSes),p_pre[:,1],'k-')
        # plt.plot(range(nSes),p_pre_not[:,0],'b--')
        # plt.plot(range(nSes),p_pre_not[:,1],'b-')
        #
        # plt.show(block=False)

        # res = sstats.ttest_ind_from_stats(np.nanmean(p_pre[:,0,0]),np.nanstd(p_pre[:,0,0]),(~np.isnan(p_pre[:,0,0])).sum(),np.nanmean(p_pre[:,1,0]),np.nanstd(p_pre[:,1,0]),(~np.isnan(p_pre[:,1,0])).sum(),equal_var=True)
        # print(res)
        # res = sstats.ttest_ind_from_stats(np.nanmean(p_pre[:,0,1]),np.nanstd(p_pre[:,0,1]),(~np.isnan(p_pre[:,0,1])).sum(),np.nanmean(p_pre[:,1,1]),np.nanstd(p_pre[:,1,1]),(~np.isnan(p_pre[:,1,1])).sum(),equal_var=True)
        # print(res)

        # print('sessions before and after coding are more probable to be active')

        # res = sstats.ttest_ind_from_stats(np.nanmean(p_pre[:,0,0]),np.nanstd(p_pre[:,0,0]),(~np.isnan(p_pre[:,1,1])).sum(),np.nanmean(p_pre[:,1,1]),np.nanstd(p_pre[:,1,1]),(~np.isnan(p_pre[:,1,1])).sum(),equal_var=True)
        # print(res)

        ax = plt.axes([0.4,0.7,0.125,0.25])
        # print(p_pre)
        # ax = plt.axes([0.1,0.1,0.2,0.85])
        # ax.errorbar([0.9,1.9],np.nanmean(p_pre[...,0],0),np.nanstd(p_pre[...,0],0),fmt='k-',linewidth=0.5)
        ax.plot([0.5,2.5],[np.nanmean(p_pre[:,0,0,1],0),np.nanmean(p_pre[:,0,0,1],0)],'k--')
        for ds in range(1,5):
            col = [0.2*ds,0.2*ds,1]
            ax.errorbar([1.1,2.1],np.nanmean(p_pre[...,1,ds],0),np.nanstd(p_pre[...,1,ds],0),fmt='-',color=col,linewidth=0.5)
            col = [1,0.2*ds,0.2*ds]
            ax.errorbar([1.1,2.1],np.nanmean(p_post[...,1,ds],0),np.nanstd(p_post[...,1,ds],0),fmt='-',color=col,linewidth=0.5)
        ax.set_xticks([1,2])
        ax.set_ylim([0.5,1])
        ax.set_xticklabels(['$\\alpha^+ / \\beta^+$','$\\alpha^- / \\beta^-$'])
        ax.set_ylabel('$p(\\alpha_{s+\Delta s}^+|\\beta_s^{\pm})$')


        # ax = plt.axes([0.525,0.7,0.125,0.25])
        # ax = plt.axes([0.35,0.1,0.2,0.85])
        # ax.errorbar([0.9,1.9],np.nanmean(p_post[...,0],0),np.nanstd(p_post[...,0],0),fmt='k-',linewidth=0.5)
        ax.set_xticks([1,2])
        ax.set_ylim([0.5,1])
        pl_dat.remove_frame(ax,['top','right'])
        # ax.set_xticklabels(['$\\alpha^+ / \\beta^+$','$\\alpha^- / \\beta^-$'])
        # ax.set_ylabel('$p(\\alpha^+|...)$')
        # plt.show(block=False)


        ### find, if coding sessions usually belong to a period of longer activity
        status_alt = np.zeros_like(cluster.status[...,1],'int')
        for c in range(nC):
            s0 = 0
            inAct = False
            for s in range(nSes):

                if inAct:
                    if ~cluster.status[c,s,1]:
                        status_alt[c,s0:s] = s-s0
                        inAct=False
                else:
                    if cluster.status[c,s,1]:
                        s0 = s
                        inAct = True

        L_code = status_alt[cluster.status[...,2]]

        IPI = get_ICPI(cluster.status[...,2],mode='IPI')

        ax = plt.axes([0.1,0.4,0.25,0.25])
        # ax.hist(L_code,np.linspace(0,50,51))
        # ax.plot(range(nSes),IPI,'k')

        nAct = cluster.status[...,1].sum(1)
        nPC = cluster.status[...,2].sum(1)
        rate = nPC/nAct
        mean_r = np.zeros((nSes,2))*np.NaN
        tmp = []
        for i in range(1,nSes):
            if np.any(nAct==i):
                if i == 0:
                    print(i)
                    print(rate[nAct==1])
                mean_r[i,0] = rate[nAct==i].mean()
                mean_r[i,1] = rate[nAct==i].std()

                tmp.append(rate[nAct==i])

        # print(tmp)
        res = sstats.f_oneway(*tmp)
        print(res)
        # pval[key][i] = res.pvalue

        ax.plot(nAct,nPC/nAct,'k.',markersize=.2)
        # ax.plot(range(nSes),mean_r,'r-')
        pl_dat.plot_with_confidence(ax,range(nSes),mean_r[:,0],mean_r[:,1],col='r')
        # ax = plt.axes([0.7,0.7,0.25,0.25])
        #
        #
        plt.figtext(0.45,0.5,'coding',fontsize=14)
        plt.figtext(0.75,0.15,'  field \nstability',fontsize=14)
        #
        # plt.show(block=False)

        ### coding -> activity




        ### coding -> field stability
        ax = plt.axes([0.7,0.4,0.25,0.2])

        maxSes = 6
        col_arr = [[0.5,0.5,1],[0.5,0.5,0.5],[1,0.5,0.5],[0.5,1,0.5]]
        label_arr = ['continuous','mixed','non-coding','silent']
        # key_arr = ['cont','mix','discont','silent']
        key_arr = ['cont','mix','discont']

        w_bar = 0.2
        nKey = len(key_arr)
        offset_bar = ((nKey+1)%2)*w_bar/2 + ((nKey-1)//2)*w_bar

        for i,key in enumerate(key_arr):
            ax.bar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
            ax.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],cluster.stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')

        ax.set_xlabel('session difference $\Delta s$')
        ax.set_ylabel('$r_{stable}$')
        ax.set_xlim([0.5,maxSes+0.5])
        ax.set_ylim([0,1.1])
        ax.legend(loc='upper right',bbox_to_anchor=[1.1,1.05],fontsize=8)
        pl_dat.remove_frame(ax,['top','right'])



        ### field stability -> coding

        print('find neurons, which are stable and get statistics of reactivation, recoding, enhanced prob of coding, ...')



        ### activity -> field stability
        ax = plt.axes([0.7,0.7,0.25,0.2])
        maxSes = 6
        col_arr = [[1,0.5,0.5],[0.5,1,0.5]]
        label_arr = ['non-coding','silent']
        # key_arr = ['cont','mix','discont','silent']
        key_arr = ['discont','silent']

        w_bar = 0.2
        nKey = len(key_arr)
        offset_bar = ((nKey+1)%2)*w_bar/2 + ((nKey-1)//2)*w_bar
        print(offset_bar)
        offset_bar = 0.1
        for i,key in enumerate(key_arr):
            ax.bar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
            ax.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],cluster.stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')

        ax.set_xlabel('session difference $\Delta s$')
        ax.set_ylabel('$r_{stable}$')
        ax.set_ylim([0,1.1])
        ax.set_xlim([0.5,maxSes+0.5])
        ax.legend(loc='upper right',bbox_to_anchor=[1.1,1.05],fontsize=8)
        pl_dat.remove_frame(ax,['top','right'])



        ### field stability -> activity
        plt.tight_layout
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('dynamics_interaction')


    if plot_fig[20]:

        # need:
        # neurons detected:   SNR, rval, CNN, (p_m) <- stats
        # place fields:       Bayes_factor, reliability, A0, A, pmass <- fields

        f_max = 5
        # SNR_thr = 1
        rval_thr = 0
        CNN_thr = 0.6

        A0_thr = 1
        A_thr = 3
        pmass_thr = 0.5
        Bayes_thr = 10
        rel_thr = 0.1

        if PC is None:

            active = {'SNR':        {},
                'r_values':     {},
                'CNN':          {}}

            PC = {'Bayes_factor':   {},
                'reliability':  {},
                'A_0':          {},
                'A':            {},
                'p_mass':       {}}

            for s in tqdm(range(nSes)):
                PC_para = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_para.mat'%(s+1),variable_names=['Bayes_factor','posterior_mass','parameter'],squeeze_me=True)
                firingstats = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_firingstats.mat'%(s+1),variable_names=['trial_map','map'],squeeze_me=True)
                CNMF_results = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/results_redetect.mat'%(s+1),variable_names=['SNR','r_values','CNN'],squeeze_me=True)

                active['SNR'][s] = CNMF_results['SNR']
                active['r_values'][s] = CNMF_results['r_values']
                active['CNN'][s] = CNMF_results['CNN']
                N = len(active['SNR'][s])

                PC['Bayes_factor'][s] = PC_para['Bayes_factor'][...,0]
                PC['A_0'][s] = PC_para['parameter'][:,:,0,0]
                PC['A'][s] = PC_para['parameter'][:,:,1,0]
                PC['p_mass'][s] = PC_para['posterior_mass']
                PC['reliability'][s] = np.zeros((N,f_max))
                for n in range(N):
                    for f in np.where(~np.isnan(PC['A'][s][n,:]))[0]:
                        PC['reliability'][s][n,f],_,_ = get_reliability(firingstats['trial_map'][n,...],firingstats['map'][n,...],PC_para['parameter'][n,...],f)


        # return active, PC
        ### get SNR dependence of detected neurons
        # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.05) | (cluster.stats['match_score'][...,0]>0.95)
        # idx_other = (cluster.stats['r_values'] > 0) & idx_pm
        # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.5) | (cluster.stats['match_score'][...,0]>0.95)
        # idx_other_certain = (cluster.stats['r_values'] > 0) & idx_pm
        r_thr = 0
        SNR_arr = np.linspace(1,10,10)
        nROI = np.zeros(SNR_arr.shape + (nSes,2))
        nPC = np.zeros(SNR_arr.shape + (nSes,2))
        plt.figure(figsize=(7,2.5))
        width = 0.4
        ax = plt.axes([0.1,0.2,0.25,0.65])
        for i,SNR_thr in enumerate(SNR_arr):

            for s in range(nSes):
                idx_active = (active['SNR'][s] > SNR_thr) & (active['r_values'][s] > r_thr) & (active['CNN'][s] > CNN_thr)
                nROI[i,s,0] = (idx_active).sum()

                morphed_A0_thr = A0_thr-PC['reliability'][s]/2
                idx_fields = (PC['A_0'][s]>morphed_A0_thr) & \
                            (PC['A'][s]>A_thr) & \
                            (PC['p_mass'][s]>pmass_thr) & \
                            (PC['Bayes_factor'][s]>Bayes_thr) & \
                            (PC['reliability'][s]>rel_thr)

                nPC[i,s,0] = (idx_active & np.any(idx_fields,1)).sum()
                # print('active neurons / place cells in session %d: %d / %d'%(s+1,nAct,nPC))


            # nROI[i,:,0] = ((cluster.stats['SNR']>SNR_thr) & idx_other).sum(0)
            # nROI[i,:,1] = ((cluster.stats['SNR']>SNR_thr) & idx_other_certain).sum(0)

            # nPC[i,:,0] = ((cluster.stats['SNR']>SNR_thr) & idx_other & cluster.status[...,2]).sum(0)
            # nPC[i,:,1] = ((cluster.stats['SNR']>SNR_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
            ax.scatter(SNR_thr-width/2 + width*np.random.rand(nSes)[cluster.sessions['bool']],nROI[i,cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
            ax.scatter(SNR_thr-width/2 + width*np.random.rand(nSes)[cluster.sessions['bool']],nPC[i,cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

        ax.plot(SNR_arr,nROI[:,cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')#,label='neurons ($p_m\geq0.05$)')
        # ax.plot(SNR_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')# ($p_m\geq0.95$)')
        ax.plot(SNR_arr,nPC[:,cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
        # ax.plot(SNR_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')

        ax.set_ylim([0,2700])
        ax.set_xlabel('$\Theta_{SNR}$')
        ax.set_ylabel('# neurons')
        ax.spines['top'].set_visible(False)

        ax2 = ax.twinx()
        # ax2.plot(SNR_arr,nPC[...,0].mean(1)/nROI[...,0].mean(1),'k:')
        ax2.plot(SNR_arr,(nPC[:,cluster.sessions['bool'],0]/nROI[:,cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
        ax2.plot([3,3],[0,1],'k--')
        ax2.set_ylim([0,0.54])
        # ax2.set_ylabel('PC fraction')
        ax2.set_yticklabels([])
        ax2.spines['top'].set_visible(False)
        ax.legend(fontsize=10,bbox_to_anchor=[0.3,1.15],loc='upper left')
        # plt.tight_layout()
        # plt.show(block=False)

        # if sv:
            # pl_dat.save_fig('neuronNumbers_SNR')


        ### get r_val dependence of detected neurons
        # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.05) | (cluster.stats['match_score'][...,0]>0.95)
        # idx_other = (cluster.stats['SNR'] > 2) & idx_pm
        # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.5) | (cluster.stats['match_score'][...,0]>0.95)
        # idx_other_certain = (cluster.stats['SNR'] > 2) & idx_pm

        # idx_other = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.5)
        # idx_other_certain = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.95)
        SNR_thr = 3
        r_arr = np.linspace(-0.4,1,8)
        nROI = np.zeros(r_arr.shape + (nSes,2))
        nPC = np.zeros(r_arr.shape + (nSes,2))
        # plt.figure(figsize=(4,2.5))
        width = 0.08
        ax = plt.axes([0.375,0.2,0.25,0.65])
        for i,r_thr in enumerate(r_arr):
            for s in range(nSes):
                idx_active = (active['SNR'][s] > SNR_thr) & (active['r_values'][s] > r_thr) & (active['CNN'][s] > CNN_thr)
                nROI[i,s,0] = (idx_active).sum()

                morphed_A0_thr = A0_thr-PC['reliability'][s]/2
                idx_fields = (PC['A_0'][s]>morphed_A0_thr) & \
                            (PC['A'][s]>A_thr) & \
                            (PC['p_mass'][s]>pmass_thr) & \
                            (PC['Bayes_factor'][s]>Bayes_thr) & \
                            (PC['reliability'][s]>rel_thr)

                nPC[i,s,0] = (idx_active & np.any(idx_fields,1)).sum()

            # nROI[i,:,0] = ((cluster.stats['r_values']>r_thr) & idx_other).sum(0)
            # nROI[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain).sum(0)

            # nPC[i,:,0] = ((cluster.stats['r_values']>r_thr) & idx_other & cluster.status[...,2]).sum(0)
            # nPC[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
            ax.scatter(r_thr-width/2 + width*np.random.rand(nSes)[cluster.sessions['bool']],nROI[i,cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
            ax.scatter(r_thr-width/2 + width*np.random.rand(nSes)[cluster.sessions['bool']],nPC[i,cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

        ax.plot(r_arr,nROI[:,cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')
        # ax.plot(r_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5)
        ax.plot(r_arr,nPC[:,cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
        # ax.plot(r_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5)

        ax.set_ylim([0,2700])
        ax.set_xlabel('$\Theta_{r}$')
        # ax.set_ylabel('# neurons')
        ax.set_yticklabels([])
        # ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)

        ax2 = ax.twinx()
        # ax2.plot(r_arr,nPC[...,0].mean(1)/nROI[...,0].mean(1),'k:')
        ax2.plot(r_arr,(nPC[:,cluster.sessions['bool'],0]/nROI[:,cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
        ax2.plot([0,0],[0,1],'k--')
        ax2.set_ylim([0,0.54])
        ax2.set_yticklabels([])
        # ax2.set_ylabel('PC fraction')
        ax2.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)
        # return
        ### get pm dependence of detected neurons
        idx_other = (cluster.stats['SNR'] > 3) & (cluster.stats['r_values']>0)
        # idx_other_certain = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.95)
        pm_arr = np.linspace(0,1,11)
        nROI = np.zeros(pm_arr.shape + (nSes,2))
        nPC = np.zeros(pm_arr.shape + (nSes,2))
        # plt.figure(figsize=(4,2.5))
        width = 0.04
        ax = plt.axes([0.65,0.2,0.25,0.65])
        for i,pm_thr in enumerate(pm_arr):
            # idx_pm = cluster.stats['match_score'][...,0]>pm_thr
            idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>pm_thr) | (cluster.stats['match_score'][...,0]>0.95)
            # idx_pm = (cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>pm_thr
            nROI[i,:,0] = (idx_pm & idx_other).sum(0)
            # nROI[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain).sum(0)

            nPC[i,:,0] = (idx_pm & idx_other & cluster.status[...,2]).sum(0)
            # nPC[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
            ax.scatter(pm_thr-width/2 + width*np.random.rand(nSes)[cluster.sessions['bool']],nROI[i,cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
            ax.scatter(pm_thr-width/2 + width*np.random.rand(nSes)[cluster.sessions['bool']],nPC[i,cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

        ax.plot(pm_arr,nROI[:,cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')
        # ax.plot(pm_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5)
        ax.plot(pm_arr,nPC[:,cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
        # ax.plot(pm_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'r^',markersize=4,markeredgewidth=0.5)

        ax.set_ylim([0,2700])
        ax.set_xlabel('$\Theta_{p_m}$')
        # ax.set_ylabel('# neurons')
        ax.set_yticklabels([])
        # ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)

        ax2 = ax.twinx()
        ax2.plot(pm_arr,(nPC[:,cluster.sessions['bool'],0]/nROI[:,cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
        # ax2.plot(pm_arr,nPC[...,1].mean(1)/nROI[...,1].mean(1),'k-')
        ax2.plot([0.5,0.5],[0,1],'k--')
        ax2.set_ylim([0,0.54])
        ax2.set_ylabel('PC fraction')
        ax2.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('neuronNumbers_test')


        print('whats with MI of place cells, only?')
        ### get SNR dependence of detected neurons
        idx_other = (cluster.stats['r_values'] > 0) & (cluster.stats['match_score'][...,0]>0.05)
        idx_other_certain = (cluster.stats['r_values'] > 0) & (cluster.stats['match_score'][...,0]>0.95)
        SNR_arr = np.linspace(2,10,5)
        MI = np.zeros(SNR_arr.shape + (2,))
        # nPC = np.zeros(SNR_arr.shape + (nSes,2))
        plt.figure(figsize=(4,2.5))
        width = 0.6
        ax = plt.axes([0.2,0.2,0.65,0.65])
        for i,SNR_thr in enumerate(SNR_arr):
            idx = (cluster.stats['SNR'] >= (SNR_thr-0.5)) & (cluster.stats['SNR'] < (SNR_thr+0.5)) & idx_other
            MI[i,0] = cluster.stats['MI_value'][idx].mean(0)
            idx = (cluster.stats['SNR'] >= (SNR_thr-0.5)) & (cluster.stats['SNR'] < (SNR_thr+0.5)) & idx_other_certain
            MI[i,1] = cluster.stats['MI_value'][idx].mean(0)

            ax.boxplot(cluster.stats['MI_value'][idx],positions=[SNR_thr],widths=width,whis=[5,95],notch=True,bootstrap=100,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

        ax.plot(SNR_arr,MI[...,1],'k^',markersize=5,label='neurons')

        ax.set_ylim([0,0.6])
        ax.set_xlabel('$\Theta_{SNR}$')
        ax.set_ylabel('MI')
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('MI_SNR')



    if plot_fig[21]:

        if True:
            plt.figure(figsize=(7,5),dpi=300)

            nSteps = 11
            SNR_arr = np.linspace(1,11,nSteps)

            margin = 18

            ax = plt.axes([0.1,0.1,0.45,0.85])
            t_arr = np.linspace(0,8989/15,8989)

            s = 1
            pathLoad = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
            ld = loadmat(pathLoad,variable_names=['C','A','SNR','CNN'],squeeze_me=True)
            offset = 0
            for i in tqdm(range(nSteps-1)):
                # idx_SNR = np.where((cluster.stats['SNR'][:,s] >= SNR_arr[i]) & (cluster.stats['SNR'][:,s] < SNR_arr[i+1]))
                idx_SNR = np.where((ld['SNR'] >= SNR_arr[i]) & (ld['SNR'] < SNR_arr[i+1]))
                n_idx = len(idx_SNR[0])
                if n_idx > 0:
                    for j in np.random.choice(n_idx,min(n_idx,3),replace=False):
                        # c = idx_SNR[0][j]
                        # n = int(cluster.IDs['neuronID'][c,s,1])
                        n = idx_SNR[0][j]
                        C = ld['C'][n,:]/ld['C'][n,:].max()
                        ax.plot(t_arr,-C+offset,linewidth=0.5)
                        # ax.text(600,offset,'%.2f'%cluster.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                        offset += 1#= (nSteps-i)

                offset += 1
            ax.set_yticks(np.linspace(1,offset-3,nSteps-1))
            ax.set_yticklabels(['$\\approx %d$'%i for i in SNR_arr])
            ax.set_ylabel('SNR',rotation='horizontal',labelpad=-20,y=1.)
            ax.set_xlabel('time [s]')
            ax.set_ylim([offset-1,-1])
            ax.set_xlim([0,600])
            pl_dat.remove_frame(ax,['top','right'])

            nSteps = 9
            CNN_arr = np.linspace(0.,1.,nSteps)
            acom = com(ld['A'],512,512)
            for i in tqdm(range(nSteps-1)):
                # idx_CNN = np.where((cluster.stats['CNN'][:,s] >= CNN_arr[i]) & (cluster.stats['CNN'][:,s] < CNN_arr[i+1]))
                idx_CNN = np.where((ld['CNN'] >= CNN_arr[i]) & (ld['CNN'] < CNN_arr[i+1]) & ((ld['A']>0).sum(0)>50) & np.all(acom>10,1) & np.all(acom<500,1))
                n_idx = len(idx_CNN[0])
                # print(idx_CNN)
                if n_idx > 0:
                    for j in np.random.choice(n_idx,min(n_idx,1),replace=False):
                        # c = idx_CNN[0][j]
                        # n = int(cluster.IDs['neuronID'][c,s,1])
                        n = idx_CNN[1][j]
                        A = ld['A'][:,n].reshape(512,512).toarray()
                        a_com = com(A.reshape(-1,1),512,512)
                        ax = plt.axes([0.6+(i//(nSteps//2))*0.175,0.75-(i%(nSteps//2))*0.23,0.15,0.21])
                        if i==(nSteps-2):
                            sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
                            ax.add_artist(sbar)
                        A /= A.max()
                        A[A<0.001] = np.NaN
                        ax.imshow(A,cmap='viridis',origin='lower')
                        ax.contour(A, levels=[0.3,0.6,0.9], colors='w', linewidths=[0.5], linestyles=['dotted','dashed','solid'])

                        (x_ref,y_ref) = a_com[0]# print(x_ref,y_ref)
                        x_lims = [x_ref-margin,x_ref+margin]
                        y_lims = [y_ref-margin,y_ref+margin]
                        # ax.plot(t_arr,C+nSteps-offset)
                        # ax.text(600,nSteps-offset,'%.2f'%cluster.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                        ax.set_xlim(x_lims)
                        ax.set_ylim(y_lims)
                        # ax.text(x_ref,y_ref+5,'$CNN = %.3f$'%cluster.stats['CNN'][c,s],fontsize=8)
                        ax.text(x_ref+2,y_ref+12,'$%.3f$'%ld['CNN'][n],fontsize=8)
                        pl_dat.remove_frame(ax)
                        ax.set_xticks([])
                        ax.set_yticks([])
            plt.tight_layout()
            plt.show(block=False)

            if sv:
                pl_dat.save_fig('neuron_stat_examples')

        if False:
            s = 1
            margin = 20
            nSteps = 9
            pathLoad = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s)])
            ld1 = loadmat(pathLoad,variable_names=['A'])
            pathLoad = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
            ld2 = loadmat(pathLoad,variable_names=['A'])

            x_grid, y_grid = np.meshgrid(np.arange(0., cluster.meta['dims'][0]).astype(np.float32), np.arange(0., cluster.meta['dims'][1]).astype(np.float32))
            x_remap = (x_grid - \
                        cluster.sessions['shift'][s-1,0] + cluster.sessions['shift'][s,0] + \
                        cluster.sessions['flow_field'][s-1,:,:,0] - cluster.sessions['flow_field'][s,:,:,0]).astype('float32')
            y_remap = (y_grid - \
                        cluster.sessions['shift'][s-1,1] + cluster.sessions['shift'][s,1] + \
                        cluster.sessions['flow_field'][s-1,:,:,1] - cluster.sessions['flow_field'][s,:,:,1]).astype('float32')

            plt.figure(figsize=(2,4),dpi=300)
            p_arr = np.linspace(0,1,nSteps)
            for i in tqdm(range(nSteps-1)):
                idx_p = np.where((cluster.stats['match_score'][:,s,0] >= p_arr[i]) & (cluster.stats['match_score'][:,s,0] < p_arr[i+1]) & (cluster.status[:,s-1,1]))
                n_idx = len(idx_p[0])
                if n_idx > 0:
                    c = np.random.choice(idx_p[0])
                    # s = idx_SNR[1][j]
                    n1 = int(cluster.IDs['neuronID'][c,s-1,1])
                    n2 = int(cluster.IDs['neuronID'][c,s,1])


                    ax = plt.axes([0.05+(i//(nSteps//2))*0.45,0.75-(i%(nSteps//2))*0.23,0.4,0.2])
                    # ax = plt.axes([0.7,0.8-0.2*]])
                    # for j in np.random.choice(n_idx,min(n_idx,3),replace=False):
                    # offset += 1#= (nSteps-i)
                    A1 = ld1['A'][:,n1].reshape(512,512).toarray()
                    A1 = cv2.remap(A1, x_remap,y_remap, cv2.INTER_CUBIC)
                    A2 = ld2['A'][:,n2].reshape(512,512).toarray()

                    a_com = com(A2.reshape(-1,1),512,512)

                    ax.contour(A1/A1.max(), levels=[0.3,0.6,0.9], colors='k', linewidths=[0.5], linestyles=['dotted','dashed','solid'])
                    ax.contour(A2/A2.max(), levels=[0.3,0.6,0.9], colors='r', linewidths=[0.5], linestyles=['dotted','dashed','solid'])
                    if i==(nSteps-2):
                        sbar = ScaleBar(530.68/512 *10**(-6),location='lower right',box_alpha=0)
                        ax.add_artist(sbar)

                    (x_ref,y_ref) = a_com[0]# print(x_ref,y_ref)
                    x_lims = [x_ref-margin,x_ref+margin]
                    y_lims = [y_ref-margin,y_ref+margin]
                    # ax.plot(t_arr,C+nSteps-offset)
                    # ax.text(600,nSteps-offset,'%.2f'%cluster.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                    ax.set_xlim(x_lims)
                    ax.set_ylim(y_lims)
                    ax.text(x_ref+2,y_ref+8,'$%.2f$'%cluster.stats['match_score'][c,s,0],fontsize=8)
                    pl_dat.remove_frame(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # offset += 1
            plt.tight_layout()
            plt.show(block=False)

            if sv:
                pl_dat.save_fig('neuron_matches_examples')


    if plot_fig[22]:

        print('### plot location-specific stability ###')

        # p_rec = {'all':     np.zeros(nSes)*np.NaN,
        #          'gate':    np.zeros(nSes)*np.NaN,
        #          'reward':  np.zeros(nSes)*np.NaN,
        #          'others':  np.zeros(nSes)*np.NaN}

        s_arr = np.array([0,5,20,30,87])
        # s_arr = np.array([0,10,21])
        n_int = len(s_arr)-1

        ds = 1
        session_bool = np.where(np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False))[0]
        print(session_bool)
        loc_stab = np.zeros((nSes,nbin+2,nbin+2))
        loc_stab_p = np.zeros((nSes,nbin+2,nbin+2))
        for s in session_bool:#range(nSes):#np.where(cluster.sessions['bool'])[0]:
            ### assign bin-specific remapping to rows, active nPC (nbins+1) & silent (nbins+2)
            for c in np.where(cluster.status[:,s,2])[0]:
                ## find belonging fields
                if cluster.status[c,s+ds,2]:
                    d = np.abs(np.mod(cluster.fields['location'][c,s,:,0][:,np.newaxis] - cluster.fields['location'][c,s+ds,:,0]+nbin/2,nbin)-nbin/2)
                    d[np.isnan(d)] = nbin
                    f1,f2 = sp.optimize.linear_sum_assignment(d)
                    for f in zip(f1,f2):
                        if d[f] < nbin:
                            loc_stab[s,int(round(cluster.fields['location'][c,s,f[0],0])),int(round(cluster.fields['location'][c,s+ds,f[1],0]))] += 1
                            loc_stab_p[s,int(round(cluster.fields['location'][c,s,f[0],0])),:nbin] += cluster.fields['p_x'][c,s+ds,f[1],:]

        loc_stab = loc_stab[:,:nbin,:nbin]
        loc_stab_p = loc_stab_p[:,:nbin,:nbin]

        p_rec_loc = np.zeros((n_int,nbin,nSes))*np.NaN

        s1_shifts,s2_shifts,f1,f2 = np.unravel_index(cluster.compare['pointer'].col,(nSes,nSes,cluster.meta['field_count_max'],cluster.meta['field_count_max']))
        c_shifts = cluster.compare['pointer'].row
        sig = 5
        di = 3

        for ds in range(1,min(nSes,41)):
            session_bool = np.where(np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False))[0]

            ### somehow condition this on the location
            # for s1 in session_bool:
            #     overlap = cluster.status[cluster.status[:,s1,1],s1+ds,1].sum(0).astype('float')
            #     N_ref = cluster.status[:,s1,1].sum(0)
            #     p_rec['act'][ds,s1] = (overlap/N_ref)
            #
            #     overlap = cluster.status[cluster.status[:,s1,2],s1+ds,2].sum(0).astype('float')
            #     N_ref = cluster.status[cluster.status[:,s1,2],s1+ds,1].sum(0)
            #     p_rec['PC'][ds,s1] = (overlap/N_ref)

            Ds = s2_shifts-s1_shifts
            idx = np.where(Ds==ds)[0]
            idx_shifts = cluster.compare['pointer'].data[idx].astype('int')-1
            shifts = cluster.compare['shifts'][idx_shifts]

            s = s1_shifts[idx]
            f = f1[idx]
            c = c_shifts[idx]
            loc_shifts = np.round(cluster.fields['location'][c,s,f,0]).astype('int')

            for j in range(len(s_arr)-1):
                for i in range(nbin):
                    i_min = max(0,i-di)
                    i_max = min(nbin,i+di)
                    idx_loc = (loc_shifts>=i_min) & (loc_shifts<i_max) & ((s>=s_arr[j]) & (s<s_arr[j+1]))

                    shifts_loc = shifts[idx_loc]
                    N_data = len(shifts_loc)
                    N_stable = (np.abs(shifts_loc)<(1.96*sig)).sum()

                    p_rec_loc[j,i,ds] = N_stable/N_data

        plt.figure(figsize=(7,5),dpi=300)


        ax_RW = plt.axes([0.1,0.75,0.25,0.225])
        ax_GT = plt.axes([0.4,0.75,0.25,0.225])
        ax_nRnG = plt.axes([0.7,0.75,0.25,0.225])
        ax_RW.bar(range(nbin),1000.*cluster.para['zone_mask']['reward'],width=1,facecolor='tab:red',alpha=0.3)
        ax_RW.set_ylim([0,0.1])
        pl_dat.remove_frame(ax_RW,['top','right'])
        ax_GT.bar(range(nbin),1000.*cluster.para['zone_mask']['gate'],width=1,facecolor='tab:green',alpha=0.3)
        ax_GT.set_ylim([0,0.1])
        pl_dat.remove_frame(ax_GT,['top','right'])
        ax_nRnG.bar(range(nbin),1000.*cluster.para['zone_mask']['others'],width=1,facecolor='tab:blue',alpha=0.3)
        ax_nRnG.set_ylim([0,0.1])
        pl_dat.remove_frame(ax_nRnG,['top','right'])
        for j in range(n_int):

            col = [1,0.3*j,0.3*j]
            occ = loc_stab_p[s_arr[j]:s_arr[j+1],cluster.para['zone_mask']['reward'],:].sum(0).sum(0)
            occ /= occ.sum()
            ax_RW.plot(range(nbin),occ,'-',color=col)
            # ax.bar(range(nbin),loc_stab[:20,cluster.para['zone_mask']['reward'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)

            col = [0.3*j,0.3*j,1]
            occ = loc_stab_p[s_arr[j]:s_arr[j+1],cluster.para['zone_mask']['others'],:].sum(0).sum(0)
            occ /= occ.sum()
            ax_nRnG.plot(range(nbin),occ,'-',color=col)
            # ax.bar(range(nbin),loc_stab[:20,cluster.para['zone_mask']['others'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)

            col = [0.3*j,0.8,0.3*j]
            occ = loc_stab_p[s_arr[j]:s_arr[j+1],cluster.para['zone_mask']['gate'],:].sum(0).sum(0)
            occ /= occ.sum()
            ax_GT.plot(range(nbin),occ,'-',color=col)
            # ax.bar(range(nbin),loc_stab[:20,cluster.para['zone_mask']['others'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)

        for j in range(n_int):
            ax_im = plt.axes([0.1,0.525-j*0.16,0.35,0.125])
            im = ax_im.imshow(gauss_smooth(p_rec_loc[j,...],(2,0)),clim=[0.2,0.7],interpolation='None',origin='lower',aspect='auto')
            plt.colorbar(im)
            ax_im.set_xlim([0.5,20.5])
            if j < (n_int-1):
                ax_im.set_xticklabels([])

            ax = plt.axes([0.6,0.525-j*0.16,0.35,0.125])
            for i,ds in enumerate([1,3,10]):
                col = [0.35*i,0.35*i,0.35*i]
                if j==0:
                    ax_im.annotate(s='',xy=(ds,100),xytext=(ds,115),fontsize=6,annotation_clip=False,arrowprops=dict(arrowstyle='->',color=col))
                ax.plot(p_rec_loc[j,:,ds],color=col)
            ax.set_ylim([0,1])
            pl_dat.remove_frame(ax,['top','right'])
        plt.show(block=False)

        # maxSes = 20
        # print('what are those stable cells coding for?')
        # plt.figure(figsize=(5,2.5))
        #
        # col_arr = ['k',[0.5,1,0.5],[1,0.5,0.5],[0.5,0.5,1]]
        # label_arr = ['all','GT','RW','nRG']
        # key_arr = ['all','gate','reward','others']
        #
        # w_bar = 0.2
        # nKey = len(key_arr)
        # offset_bar = ((nKey+1)%2)*w_bar/2 + (nKey//2 - 1)*w_bar
        #
        # arr = np.arange(1,min(40,nSes),2)
        # for i,key in enumerate(key_arr):

            # plt.bar(arr-offset_bar+i*w_bar,p_rec[key][arr],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
            # plt.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.loc_stability[key]['mean'][:maxSes,1],cluster.loc_stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')

        # plt.xlabel('session difference $\Delta s$')
        # plt.ylabel('$\%$ stable fields')
        # plt.ylim([0,1.1])
        # plt.legend(loc='upper right',ncol=2)
        # plt.tight_layout()
        # plt.show(block=False)


    if plot_fig[23]:

        print('### plot location specific stability at ds = 1 ###')

        ds = 1
        session_bool = np.where(np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False))[0]
        print(session_bool)
        loc_stab = np.zeros((nbin+2,nbin+2))
        loc_stab_p = np.zeros((nbin+2,nbin+2))
        for s in session_bool:#range(nSes):#np.where(cluster.sessions['bool'])[0]:
            ### assign bin-specific remapping to rows, active nPC (nbins+1) & silent (nbins+2)
            for c in np.where(cluster.status[:,s,2])[0]:
                ## find belonging fields
                if cluster.status[c,s+ds,2]:
                    d = np.abs(np.mod(cluster.fields['location'][c,s,:,0][:,np.newaxis] - cluster.fields['location'][c,s+ds,:,0]+nbin/2,nbin)-nbin/2)
                    d[np.isnan(d)] = nbin
                    f1,f2 = sp.optimize.linear_sum_assignment(d)
                    for f in zip(f1,f2):
                        if d[f] < nbin:
                            loc_stab[int(round(cluster.fields['location'][c,s,f[0],0])),int(round(cluster.fields['location'][c,s+ds,f[1],0]))] += 1
                            loc_stab_p[int(round(cluster.fields['location'][c,s,f[0],0])),:nbin] += cluster.fields['p_x'][c,s+ds,f[1],:]

        loc_stab = loc_stab[:nbin,:nbin]
        loc_stab_p = loc_stab_p[:nbin,:nbin]

        ### proper assignment of place fields?
        plt.figure()
        plt.subplot(221)
        plt.imshow(loc_stab,origin='lower')
        plt.subplot(222)

        occ = loc_stab_p[cluster.para['zone_mask']['others'],:].sum(0)
        plt.bar(range(nbin),1000.*cluster.para['zone_mask']['others'],width=1,facecolor='tab:blue',alpha=0.3)
        plt.bar(range(nbin),occ,width=1,facecolor='tab:blue')
        plt.bar(range(nbin),loc_stab[cluster.para['zone_mask']['others'],:].sum(0),width=1,facecolor='k',alpha=0.5)
        plt.ylim([0,occ.max()*1.1])
        # plt.bar(range(nbin),loc_stab[5:15,:].sum(0),width=1,facecolor='k',alpha=0.5)
        # plt.bar(range(nbin),loc_stab[60:70,:].sum(0),width=1,facecolor='k',alpha=0.5)
        # plt.bar(range(nbin),loc_stab[60:70,:].sum(0),width=1,facecolor='k',alpha=0.5)
        plt.subplot(223)
        occ = loc_stab_p[cluster.para['zone_mask']['gate'],:].sum(0)
        plt.bar(range(nbin),1000.*cluster.para['zone_mask']['gate'],width=1,facecolor='g',alpha=0.3)
        plt.bar(range(nbin),occ,width=1,facecolor='tab:green')
        plt.bar(range(nbin),loc_stab[cluster.para['zone_mask']['gate'],:].sum(0),width=1,facecolor='k',alpha=0.5)
        plt.ylim([0,occ.max()*1.1])

        plt.subplot(224)
        occ = loc_stab_p[cluster.para['zone_mask']['reward'],:].sum(0)
        plt.bar(range(nbin),1000.*cluster.para['zone_mask']['reward'],width=1,facecolor='r',alpha=0.3)
        plt.bar(range(nbin),occ,width=1,facecolor='tab:red')
        plt.bar(range(nbin),loc_stab[cluster.para['zone_mask']['reward'],:].sum(0),width=1,facecolor='k',alpha=0.5)
        plt.ylim([0,occ.max()*1.1])
        # plt.bar(range(nbin),loc_stab[55:66,:].sum(0),width=1,facecolor='k',alpha=0.5)
        # plt.ylim()
        plt.tight_layout()
        plt.show(block=False)


    if plot_fig[24]:
        print('## plot population vector correlations etc')

        fmap = np.ma.masked_invalid(cluster.stats['firingmap'][cluster.stats['cluster_bool'],:,:])
        print(fmap.shape)

        if False:
            di = 3

            for ds in [1,2,3,5,10,20]:
                session_bool = np.where(np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False))[0]
                s_corr = np.zeros(nSes)*np.NaN
                plt.figure(figsize=(3,5))
                ax = plt.axes([0.1,0.6,0.85,0.35])
                for s in tqdm(np.where(session_bool)[0]):
                    corr = np.zeros(nbin)
                    for i in range(nbin):

                        idx = np.zeros(nbin,'bool')
                        idx[max(0,i-di):min(nbin+1,i+di)] = True

                        idx_cells = cluster.status[cluster.stats['cluster_bool'],s,2]
                        corr[i] = np.ma.corrcoef(fmap[idx_cells,s,:][:,idx].mean(-1),fmap[idx_cells,s+ds,:][:,idx].mean(-1))[0,1]


                    if s in [10,20,40,60]:
                        ax.plot(corr)

                    s_corr[s] = corr.mean()

                ax.set_ylim([-0.25,0.75])
                ax = plt.axes([0.1,0.15,0.85,0.35])
                ax.plot(gauss_smooth(s_corr,1,mode='constant'))
                ax.set_ylim([-0.25,0.75])
                plt.title('ds=%d'%ds)
                plt.show(block=False)

        if True:

            fmap = gauss_smooth(cluster.stats['firingmap'],(0,0,2))
            corr = np.zeros((nC,nSes,nSes))*np.NaN
            for ds in tqdm(range(1,80)):
                session_bool = np.where(np.pad(cluster.sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(cluster.sessions['bool'][:],(0,0),constant_values=False))[0]
                for s in np.where(session_bool)[0]:
                    for n in np.where(cluster.status[:,s,2] & cluster.status[:,s+ds,2])[0]:
                        corr[n,s,ds] = np.corrcoef(fmap[n,s,:],fmap[n,s+ds,:])[0,1]


            # print(corr)
            plt.figure(figsize=(7,5),dpi=300)
            plt.subplot(121)
            im = plt.imshow(np.nanmean(corr,0),clim=[0,0.5])
            plt.colorbar(im)

            plt.subplot(122)
            plt.plot(np.nanmean(np.nanmean(corr,0),0))
            plt.ylim([0,1])
            plt.show(block=False)


    if plot_fig[25]:

        print('### plot location-specific, static statistics ###')

        ## initialize some arrays
        loc = np.round(cluster.fields['location'][...,0]).astype('int')

        par_keys = ['width','MI_value','max_rate','reliability']
        par_labels = ['$\sigma$','MI','$\\nu^*$','a']
        ranges = np.array([[0,20],[0,1.2],[0,50],[0,1]])

        distr = {}
        for key in par_keys:
            distr[key] = np.zeros((nbin,2))*np.NaN


        fig = plt.figure(figsize=(7,5),dpi=300)

        ### place field density
        ax = plt.axes([0.1,0.45,0.325,0.15])
        pl_dat.add_number(fig,ax,order=2)
        s_range = 20


        ax_sig = plt.axes([0.1,0.1,0.175,0.16])
        pl_dat.add_number(fig,ax_sig,order=4)
        ax_MI = plt.axes([0.325,0.1,0.175,0.16])
        ax_rate = plt.axes([0.55,0.1,0.175,0.16])
        ax_rel = plt.axes([0.775,0.1,0.175,0.16])
        # s_arr = np.arange(0,nSes+s_range,s_range)
        s_arr = np.array([0,10,20,50,87])
        for j in range(len(s_arr)-1):
            idx = (cluster.status_fields & ((np.arange(nSes)>=s_arr[j]) & (np.arange(nSes)<s_arr[j+1]))[np.newaxis,:,np.newaxis])
            density = np.histogram(cluster.fields['location'][idx,0],np.linspace(0,nbin,nbin+1),density=True)
            # print(density)
            col = [0.1+0.225*j,0.1+0.225*j,1]
            ax.plot(np.linspace(0,nbin-1,nbin),density[0],color=col,label='s %d-%d'%(s_arr[j]+1,s_arr[j+1]))

            _,_,patches = ax_sig.hist(cluster.fields['width'][idx,0],np.linspace(0,20,51),color=col,cumulative=True,density=True,histtype='step')
            patches[0].set_xy(patches[0].get_xy()[:-1])
            _,_,patches = ax_MI.hist(cluster.stats['MI_value'][np.any(idx,-1)],np.linspace(0,1,51),color=col,cumulative=True,density=True,histtype='step')
            patches[0].set_xy(patches[0].get_xy()[:-1])
            _,_,patches =ax_rate.hist(cluster.fields['max_rate'][idx],np.linspace(0,50,51),color=col,cumulative=True,density=True,histtype='step')
            patches[0].set_xy(patches[0].get_xy()[:-1])
            _,_,patches =ax_rel.hist(cluster.fields['reliability'][idx],np.linspace(0,1,51),color=col,cumulative=True,density=True,histtype='step')
            patches[0].set_xy(patches[0].get_xy()[:-1])
            # ax.hist(cluster.fields['location'][idx,0],np.linspace(0,nbin-1,nbin),density=True,histtype='step')
        pl_dat.remove_frame(ax_sig,['top','right'])
        pl_dat.remove_frame(ax_MI,['top','right'])
        pl_dat.remove_frame(ax_rate,['top','right'])
        pl_dat.remove_frame(ax_rel,['top','right'])
        ax_sig.set_ylabel('fraction')
        ax_sig.set_xlabel('$\sigma$ [bins]')
        ax_MI.set_yticklabels([])
        ax_MI.set_xlabel('MI [bit]')
        ax_rate.set_yticklabels([])
        ax_rate.set_xlabel('$\\nu^*$ [Hz]')
        ax_rel.set_yticklabels([])
        ax_rel.set_xlabel('a')
        ax.set_xlabel('position [bins]')
        ax.set_ylabel('density')
        ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.25,1.2],handlelength=1)
        pl_dat.remove_frame(ax,['top','right'])

        props = dict(boxstyle='round', facecolor='w', alpha=0.8)
        for j,s in enumerate([2,14,34]):
            ax = plt.axes([0.1+0.125*j,0.675,0.11,0.25])
            if j == 0:
                pl_dat.add_number(fig,ax,order=1,offset=[-100,50])
            idxes_tmp = np.where(cluster.status_fields[:,s,:])
            idxes = idxes_tmp[0]
            sort_idx = np.argsort(cluster.fields['location'][idxes_tmp[0],s,idxes_tmp[1],0])
            sort_idx = idxes[sort_idx]
            nID = len(sort_idx)

            firingmap = cluster.stats['firingmap'][sort_idx,s,:]
            firingmap = gauss_smooth(firingmap,[0,4])
            firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            ax.imshow(firingmap,aspect='auto',origin='upper',cmap='hot',clim=[0,3])
            ax.text(5,nID*0.95,'n = %d'%nID,bbox=props,color='k',fontsize=6)
            ax.text(40,nID/10,'Session %d'%(s+1),bbox=props,color='k',fontsize=6)
            pl_dat.remove_frame(ax)
            ax.set_xticks([])
            ax.set_yticks([])

        ### location-specific parameters
        ## width, rel, MI, max_rate
        for j,key in enumerate(par_keys):
            ax = plt.axes([0.6,0.82-j*0.125,0.35,0.105])
            if j==0:
                pl_dat.add_number(fig,ax,order=3)
            if key == 'MI_value':
                dat = cluster.stats[key]
            elif key == 'width':
                dat = cluster.fields[key][...,0]
            else:
                dat = cluster.fields[key]

            for i in range(nbin):
                idx = ((loc == i) & cluster.status_fields & (np.arange(nSes)<200)[np.newaxis,:,np.newaxis])
                if key == 'MI_value':
                    idx = np.any(idx,-1)
                # print(idx.shape)
                # print(dat.shape)
                distr[key][i,0] = dat[idx].mean()
                distr[key][i,1] = dat[idx].std()

            idx = np.where(cluster.status_fields)
            if key == 'MI_value':
                ax.plot(cluster.fields['location'][idx[0],idx[1],idx[2],0],dat[idx[0],idx[1]],'.',color=[0.6,0.6,0.6],markersize=1,markeredgewidth=0,zorder=0)
            else:
                ax.plot(cluster.fields['location'][idx[0],idx[1],idx[2],0],dat[idx[0],idx[1],idx[2]],'.',color=[0.6,0.6,0.6],markersize=1,markeredgewidth=0,zorder=0)
            pl_dat.plot_with_confidence(ax,np.linspace(0,nbin-1,nbin),distr[key][:,0],distr[key][:,1],col='tab:red')
            ax.set_ylabel(par_labels[j],rotation='vertical',ha='left',va='center')
            ax.yaxis.set_label_coords(-0.175,0.5)
            pl_dat.remove_frame(ax,['top','right'])
            if j < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('position [bins]')
            ax.set_ylim(ranges[j,:])
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('PC_locationStats')


    if plot_fig[26]:

        print('### whats up with multiple peaks? ###')

        nFields = np.ma.masked_array(cluster.status_fields.sum(-1),mask=~cluster.status[...,2])
        idx = nFields>1
        nMultiMode = nFields.mean(0)

        dLoc = np.zeros((nC,nSes))*np.NaN
        corr = np.zeros((nC,nSes))*np.NaN
        overlap = np.zeros((nC,nSes))*np.NaN
        for (c,s) in zip(np.where(idx)[0],np.where(idx)[1]):
            # pass
            loc = cluster.fields['location'][c,s,cluster.status_fields[c,s,:],0]
            dLoc[c,s] = np.abs(np.mod(loc[1] - loc[0]+nbin/2,nbin)-nbin/2)#loc[1]-loc[0]

            idx_loc = np.where(cluster.status_fields[c,s,:])[0]

            corr[c,s] = np.corrcoef(cluster.fields['trial_act'][c,s,idx_loc[0],:cluster.sessions['trial_ct'][s]],cluster.fields['trial_act'][c,s,idx_loc[1],:cluster.sessions['trial_ct'][s]])[0,1]

            overlap[c,s] = (cluster.fields['trial_act'][c,s,idx_loc[0],:cluster.sessions['trial_ct'][s]] & cluster.fields['trial_act'][c,s,idx_loc[1],:cluster.sessions['trial_ct'][s]]).sum()

        fig = plt.figure(figsize=(7,5),dpi=300)
        ax = plt.axes([0.1,0.75,0.35,0.175])
        pl_dat.add_number(fig,ax,order=1)
        ax.plot(nMultiMode,'k')
        ax.set_ylim([0.98,1.2])
        pl_dat.remove_frame(ax,['top','right'])
        ax.set_xlabel('session')
        ax.set_ylabel('$\left \langle \# fields \\right \\rangle$')

        ax = plt.axes([0.55,0.75,0.35,0.175])
        pl_dat.add_number(fig,ax,order=2,offset=[-50,50])

        ax.hist(cluster.fields['location'][nFields==1,:,0].flat,np.linspace(0,100,101),facecolor='k',density=True,label='1 field')
        ax.hist(cluster.fields['location'][idx,:,0].flat,np.linspace(0,100,101),facecolor='tab:orange',alpha=0.5,density=True,label='2 fields')
        pl_dat.remove_frame(ax,['top','right','left'])
        ax.set_yticks([])
        ax.set_xlabel('position [bins]')
        ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[0.85,1.2],handlelength=1)

        ax = plt.axes([0.65,0.1,0.3,0.42])
        pl_dat.add_number(fig,ax,order=5)
        ax.plot(dLoc[overlap==0],corr[overlap==0],'k.',markersize=1,zorder=10)
        ax.plot(dLoc[overlap>0],corr[overlap>0],'.',color='tab:red',markersize=1,zorder=12)
        ax.set_xlim([0,50])
        ax.set_ylim([-1,1])
        ax.set_yticks(np.linspace(-1,1,5))
        ax.set_xlabel('$\Delta \\theta [bins]$')
        ax.set_ylabel('$c_a$')

        ax2 = ax.twiny()
        ax2.hist(corr.flat,np.linspace(-1,1,51),orientation='horizontal',facecolor='tab:orange',alpha=0.5,zorder=0)
        ax2.set_xlim([0,ax2.get_xlim()[1]*4])
        ax2.set_xticks([])

        ax3 = ax.twinx()
        ax3.hist(dLoc.flat,np.linspace(0,50,51),orientation='vertical',facecolor='tab:orange',alpha=0.5,zorder=0)
        ax3.set_ylim([ax3.get_ylim()[1]*4,0])
        ax3.set_yticks([])

        ### plot "proper" 2-field
        idx = np.where(dLoc>30)
        i = np.random.randint(len(idx[0]))
        c = idx[0][i]
        s = idx[1][i]
        c,s = [612,84]
        print(c,s)

        ax_fmap = plt.axes([0.1,0.4,0.35,0.15])
        pl_dat.add_number(fig,ax_fmap,order=3)
        ax_fmap.bar(np.linspace(1,100,100),gauss_smooth(cluster.stats['firingmap'][c,s,:],1),width=1,facecolor='k')
        ax_fmap.set_ylabel('$\\bar{\\nu}$')

        loc = cluster.fields['location'][c,s,cluster.status_fields[c,s,:],0]
        ax_trial = plt.axes([0.375,0.525,0.125,0.1])
        idx_loc = np.where(cluster.status_fields[c,s,:])[0]
        pl_dat.remove_frame(ax_fmap,['top','right'])

        col_arr = ['tab:green','tab:blue']
        for i,f in enumerate(idx_loc):
            ax_fmap.plot(loc[i],1,'v',color=col_arr[i],markersize=5)
            ax_trial.bar(range(cluster.sessions['trial_ct'][s]),cluster.fields['trial_act'][c,s,f,:cluster.sessions['trial_ct'][s]],bottom=i,color=col_arr[i],alpha=0.5)

        ax_fmap.arrow(x=loc.min(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.max()-loc.min(),dy=0,shape='full',color='tab:orange',width=0.02,head_width=0.4,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.arrow(x=loc.max(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.min()-loc.max(),dy=0,shape='full',color='tab:orange',width=0.02,head_width=0.4,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.text(loc.min()/2+loc.max()/2,ax_fmap.get_ylim()[1],'$\Delta \\theta$',color='tab:orange',fontsize=10,ha='center')

        pl_dat.remove_frame(ax_trial,['top','right','left'])
        ax_trial.set_yticks([])
        ax_trial.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_trial.set_xlabel('trial',fontsize=10,labelpad=-5,x=0.4)
        ax_trial.text(-10,1.2,s='$c_a=%.2f$'%corr[c,s],fontsize=6)

        ### plot "improper" 2-field
        idx = np.where(dLoc<20)
        i = np.random.randint(len(idx[0]))
        c = idx[0][i]
        s = idx[1][i]
        c,s = [858,105]
        print(c,s)

        ax_fmap = plt.axes([0.1,0.1,0.35,0.15])
        pl_dat.add_number(fig,ax_fmap,order=4)
        ax_fmap.bar(np.linspace(1,100,100),gauss_smooth(cluster.stats['firingmap'][c,s,:],1),width=1,facecolor='k')
        pl_dat.remove_frame(ax_fmap,['top','right'])
        ax_fmap.set_ylabel('$\\bar{\\nu}$')
        ax_fmap.set_xlabel('position [bins]')

        loc = cluster.fields['location'][c,s,cluster.status_fields[c,s,:],0]
        ax_trial = plt.axes([0.375,0.225,0.125,0.1])
        idx_loc = np.where(cluster.status_fields[c,s,:])[0]
        for i,f in enumerate(idx_loc):

            ax_fmap.plot(loc[i],1,'v',color=col_arr[i],markersize=5)
            ax_trial.bar(range(cluster.sessions['trial_ct'][s]),cluster.fields['trial_act'][c,s,f,:cluster.sessions['trial_ct'][s]],bottom=i,color=col_arr[i],alpha=0.5)

        ax_fmap.arrow(x=loc.min(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.max()-loc.min(),dy=0,shape='full',color='tab:orange',width=0.015,head_width=0.2,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.arrow(x=loc.max(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.min()-loc.max(),dy=0,shape='full',color='tab:orange',width=0.015,head_width=0.2,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.text(loc.min()/2+loc.max()/2,ax_fmap.get_ylim()[1],'$\Delta \\theta$',color='tab:orange',fontsize=10,ha='center')
        pl_dat.remove_frame(ax_trial,['top','right','left'])
        ax_trial.set_yticks([])
        ax_trial.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_trial.set_xlabel('trial',fontsize=10,labelpad=-5,x=0.4)
        ax_trial.text(-10,1.2,s='$c_a=%.2f$'%corr[c,s],fontsize=6)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('multi_modes')

    if plot_fig[27]:

        print('### plot time-specific, static statistics ###')



    if plot_fig[30]:

        plt.figure(figsize=(3,2))

        nC1 = 3000
        L = 1000
        K_act1 = 1200
        K_act2 = 1100
        rand_pull_act = (np.random.choice(nC1,(L,K_act1))<K_act2).sum(1)
        plt.hist(rand_pull_act,np.linspace(0,800,101),facecolor='k',density=True,label='random draws')
        plt.plot([700,700],[0,0.2],'r',label='actual value')
        plt.ylim([0,0.05])
        plt.xlabel('# same activated neurons',fontsize=14)
        plt.yticks([])
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('example_draw')


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


class plot_dat:

  def __init__(self,mouse,pathFigures,nSes,para,sv_suffix='',sv_ext='png'):
    self.pathFigures = pathFigures
    self.mouse = mouse

    L_track = 100
    nbin = para['nbin']

    self.sv_opt = {'suffix':sv_suffix,
                   'ext':sv_ext,
                   'dpi':300}

    self.plt_presi = True;
    self.plot_pop = False;

    self.plot_arr = ['NRNG','GT','RW']
    self.col = ['b','g','r']
    self.col_fill = [[0.5,0.5,1],[0.5,1,0.5],[1,0.5,0.5]]

    self.h_edges = np.linspace(-0.5,nSes+0.5,nSes+2)
    self.n_edges = np.linspace(1,nSes,nSes)
    self.bin_edges = np.linspace(1,L_track,nbin)

    self.bars = {}
    self.bars['PC'] = np.zeros(nbin)
    self.bars['PC'][para['zone_mask']['others']] = 1

    self.bars['GT'] = np.zeros(nbin);

    if np.count_nonzero(para['zone_mask']['gate'])>1:
      self.bars['GT'][para['zone_mask']['gate']] = 1

    self.bars['RW'] = np.zeros(nbin);
    self.bars['RW'][para['zone_mask']['reward']] = 1


    ### build blue-red colormap
    #n = 51;   ## must be an even number
    #cm = ones(3,n);
    #cm(1,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## red
    #cm(2,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## green
    #cm(2,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## green
    #cm(3,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## blue

  def add_number(self,fig,ax,order=1,offset=None):

      offset = [-175,50] if offset is None else offset
      pos = fig.transFigure.transform(plt.get(ax,'position'))
      x = pos[0,0]+offset[0]
      y = pos[1,1]+offset[1]
      ax.text(x=x,y=y,s='%s)'%chr(96+order),ha='center',va='center',transform=None,weight='bold',fontsize=14)


  def remove_frame(self,ax,positions=None):

    if positions is None:
      positions = ['left','right','top','bottom']

    for p in positions:
      ax.spines[p].set_visible(False)

    # if 'left' in positions:
      # ax.set_yticks([])

    # if 'bottom' in positions:
      # ax.set_xticks([])


  def plot_with_confidence(self,ax,x_data,y_data,CI,col='k',ls='-',lw=1,label=None):

    col_fill = np.minimum(np.array(colors.to_rgb(col))+np.ones(3)*0.3,1)
    if len(CI.shape) > 1:
      ax.fill_between(x_data,CI[0,:],CI[1,:],color=col_fill,alpha=0.2)
    else:
      ax.fill_between(x_data,y_data-CI,y_data+CI,color=col_fill,alpha=0.2)
    ax.plot(x_data,y_data,color=col,linestyle=ls,linewidth=lw,label=label)



  def save_fig(self,fig_name,fig_pos=None):
    path = pathcat([self.pathFigures,'m%s_%s%s.%s'%(self.mouse,fig_name,self.sv_opt['suffix'],self.sv_opt['ext'])]);
    plt.savefig(path,format=self.sv_opt['ext'],dpi=self.sv_opt['dpi'])
    print('Figure saved as %s'%path)



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


def get_overlap(s,inVars):

  status,N,L = inVars
  nC,nSes = status.shape[:2]

  recurr = {'active': {'all':                np.zeros(nSes)*np.NaN,
                       'continuous':         np.zeros(nSes)*np.NaN,
                       'overrepresentation': np.zeros(nSes)*np.NaN},
            'coding': {'all':                np.zeros(nSes)*np.NaN,
                       'ofactive':           np.zeros(nSes)*np.NaN,
                       'continuous':         np.zeros(nSes)*np.NaN,
                       'overrepresentation': np.zeros(nSes)*np.NaN}}
  if N['active'][s] == 0:
    return recurr
  overlap_act = status[status[:,s,1],:,1].sum(0)
  overlap_PC = status[status[:,s,2],:,2].sum(0)

  recurr['active']['all'][1:(nSes-s)] = overlap_act[s+1:]/N['active'][s+1:]

  recurr['coding']['all'][1:(nSes-s)] = overlap_PC[s+1:]/N['coding'][s+1:]
  for (i,s1) in enumerate(range(s+1,nSes)):
    recurr['coding']['ofactive'][i+1] = overlap_PC[s1]/status[status[:,s,2],s1,1].sum()

  #print(recurr['active']['all'])
  rand_pull_act = np.zeros((nSes-s,L))*np.NaN
  rand_pull_PC = np.zeros((nSes-s,L))*np.NaN

  for s2 in range(s+1,nSes):
    if (N['active'][s]==0) or (N['active'][s2]==0):
      continue
    rand_pull_act[s2-s,:] = (np.random.choice(nC,(L,N['active'][s]))<N['active'][s2]).sum(1)

    offset = N['active'][s] - overlap_act[s2]
    randchoice_1 = np.random.choice(N['active'][s],(L,N['coding'][s]))
    randchoice_2 = np.random.choice(np.arange(offset,offset+N['active'][s2]),(L,N['coding'][s2]))
    for l in range(L):
      rand_pull_PC[s2-s,l] = np.isin(randchoice_1[l,:],randchoice_2[l,:]).sum()


    ### find continuously coding neurons
    recurr['active']['continuous'][s2-s] = (status[:,s:s2+1,1].sum(1)==(s2-s+1)).sum()/N['active'][s2]#(ic_vals[idx_ds,2] == 1).sum()/N['active'][s2]
    recurr['coding']['continuous'][s2-s] = (status[:,s:s2+1,2].sum(1)==(s2-s+1)).sum()/N['coding'][s2]#(ic_vals[idx_ds,2] == 1).sum()/N['active'][s2]

  recurr['active']['overrepresentation'][:nSes-s] = (overlap_act[s:]-np.nanmean(rand_pull_act,1))/np.nanstd(rand_pull_act,1)
  recurr['coding']['overrepresentation'][:nSes-s] = (overlap_PC[s:]-np.nanmean(rand_pull_PC,1))/np.nanstd(rand_pull_PC,1)
  return recurr


def get_shift_distr(ds,compare,para):

  nSes,nbin,N_bs,idx_celltype = para
  L_track=100
  p = {'all':{},
        'cont':{},
        'mix':{},
        'discont':{},
        'silent':{}}

  s1_shifts,s2_shifts,f1,f2 = np.unravel_index(compare['pointer'].col,(nSes,nSes,5,5))
  #print(idx_celltype)
  Ds = s2_shifts-s1_shifts
  idx_ds = np.where((Ds==ds) & idx_celltype)[0]
  N_data = len(idx_ds)
  # print(N_data)
  # cdf_shifts_ds = np.zeros((N_data,nbin))

  # s1_ds = s1_shifts[idx_ds]
  # c_ds = compare['pointer'].row[idx_ds]

  #len(Counter(frozenset(x) for x in [c_ds,s1_ds].T))

  idx_shifts = compare['pointer'].data[idx_ds].astype('int')-1
  shifts = compare['shifts'][idx_shifts]
  shifts_distr = compare['shifts_distr'][idx_shifts,:].toarray()
  # print(shifts_distr.shape)
  # plt.figure()
  # plt.bar(np.linspace(-50,50,100),shifts_distr.sum(0))
  # plt.show(block=False)
  # for i,_ in enumerate(idx_ds):
  #   roll = round(-shifts[i]+L_track/2).astype('int')
  #   cdf_shifts_ds[i,:] = np.cumsum(np.roll(shifts_distr[i,:],roll))
  #   cdf_shifts_ds[i,:] = np.roll(cdf_shifts_ds[i,:],-roll)

  for pop in p.keys():
    if pop == 'all':
      idxes = np.ones(N_data,'bool')
    elif pop=='cont':
      idxes = compare['inter_coding'][idx_ds,1]==1
    elif pop=='mix':
      idxes = ((compare['inter_coding'][idx_ds,1]>0) & (compare['inter_coding'][idx_ds,1]<1)) & (compare['inter_active'][idx_ds,1]==1)
    elif pop=='discont':
      idxes = (compare['inter_coding'][idx_ds,1]==0) & (compare['inter_active'][idx_ds,1]==1)
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


def get_ICPI(status,mode='ICI'):
    pad = 1
    ISI = np.zeros(status.shape[1])
    for stat in status:
        stat = np.pad(stat,pad_width=(pad,pad),constant_values=True)
        if np.any(stat):
            dCoding = np.diff(np.where(stat)[0])
            dCoding_cleaned = dCoding[dCoding>1]
            if len(dCoding_cleaned)>0:
                for key,val in Counter(dCoding_cleaned).items():
                    if key<status.shape[1]:
                        ISI[key-1] += val
    return ISI

def get_dp(status,status_act,status_dep=None,status_session=None,ds=1,mode='act'):
    if status_dep is None:
        status_dep = np.ones_like(status,'bool')
    if status_session is None:
        status_session = np.ones(status.shape[1],'bool')

    before = np.pad((status[:,:-ds]&status_dep[:,:-ds]),pad_width=((0,0),(ds,0)),constant_values=False)
    if mode=='act':
        cont_score = (before&status)[:,status_session].sum(1)/before[:,status_session].sum(1)
        p_tmp = status[:,status_session].sum(1)/status_dep[:,status_session].sum(1)
        dp = cont_score-p_tmp
    elif mode=='PC':
        cont_score = (before&status&status_dep)[:,status_session].sum(1)/(before&status_dep)[:,status_session].sum(1)
        p_tmp = ((before&status_dep)[:,status_session].sum(1)-1)/(status_act[:,ds:]&status_act[:,:-ds]).sum(1)
        dp = cont_score-p_tmp
    return dp,cont_score
