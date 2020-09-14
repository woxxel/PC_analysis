from multiprocessing import get_context

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, rc
from matplotlib.cm import get_cmap
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Arc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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

from utils import get_nPaths, pathcat, periodic_distr_distance, bootstrap_data, get_average, pickleData, z_from_point_normal_plane, KS_test, E_stat_test, gauss_smooth, calculate_img_correlation, get_shift_and_flow, get_reliability, com, get_status_arr, get_recurr
from get_t_measures import *
from utils_data import set_para
from get_session_specifics import get_session_specifics

warnings.filterwarnings("ignore")

def plot_multiPC_analysis(D,plot_arr=[0,1],mice=None,mouse_ex='762',N_bs=10,n_processes=0,reprocess=False,sv=False,sv_ext='png',PC=None,active=None,steady=False):#,N_bs,s_offset,sv,sv_suffix,sv_ext,arrays,occupancy,ROI_recurr,N_pairs,N_norm,pop_overlap)#,ROI_rec2,ROI_tot2)#pathBase,mouse)

    nbin = 100
    print('a')
    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    if mice is None:
        mice = D.cMice.keys()
    nMice = len(mice)

    plot_fig = np.zeros(100).astype('bool')
    for p in plot_arr:
        plot_fig[p] = True

    if plot_fig[0]:

        pie = True
        pie_labels = ['nPCs','GT','RW','nRG']
        pie_col = [[0.4,0.4,0.4],'w','tab:green','tab:red','tab:blue']
        pie_explode = [0.,0.,200,200,200]
        width=0.5
        # gate_mice = ["34","35","65","66","72","839","840","841","842","879","882","884","886","549","551","756","757","758","918shKO","931wt","943shKO"]
        # nogate_mice = ["231","232","236","762","243","245","246"]#"231",,"243","245","246"
        # i = -1


        recurr = {'act':    {'data':    np.zeros((nMice,200,2))*np.NaN,
                             'test':    np.zeros((nMice,200,2))*np.NaN},
                  'code':   {'data':    np.zeros((nMice,200,2))*np.NaN,
                             'test':    np.zeros((nMice,200,2))*np.NaN}}

        pl_dat = plot_dat(folder='popDyn',sv_ext=sv_ext)

        SD = 1#1.96
        fig = plt.figure(figsize=(7,4),dpi=300)

        ax_recurr_act = plt.axes([0.6,0.8,0.2,0.11])
        ax_recurr_act_para = plt.axes([0.9,0.8,0.075,0.11])
        pl_dat.add_number(fig,ax_recurr_act,order=3,offset=[-200,50])
        ax_recurr_PF = plt.axes([0.6,0.575,0.2,0.11])
        pl_dat.add_number(fig,ax_recurr_PF,order=4,offset=[-200,50])
        ax_recurr_overall = plt.axes([0.6,0.35,0.2,0.11])
        pl_dat.add_number(fig,ax_recurr_overall,order=5,offset=[-200,50])
        ax_recurr_popvect = plt.axes([0.85,0.35,0.125,0.11])

        ax_recurr_SD = plt.axes([0.85,0.575,0.125,0.11])

        ax_corr_p = plt.axes([0.6,0.125,0.125,0.075])
        pl_dat.add_number(fig,ax_corr_p,order=6,offset=[-200,50])
        ax_corr_p2 = plt.axes([0.825,0.125,0.125,0.075])

        nsteps = 9
        SD_arr = np.linspace(0,4,nsteps)
        p_rec_SD = np.zeros((nMice,nsteps))*np.NaN

        act_paras = np.zeros((nMice,4,2))*np.NaN
        overall_paras = np.zeros((nMice,4,2))*np.NaN

        para_act = np.zeros((len(mice),2))*np.NaN
        para_code = np.zeros((len(mice),2))*np.NaN

        fit_fun = lambda x,a,tau,b,m : a*np.exp(-x/tau)+m*x+b
        p_bounds = ([0,0,0,-1],[1,10,1,0])
        x_arr = np.arange(recurr['act']['data'].shape[1])

        color_t_raw = plt.cm.get_cmap('Dark2')(np.linspace(0,1,nMice))
        color_t = iter(plt.cm.get_cmap('Dark2')(np.linspace(0,1,nMice)))

        # for i,mouse in enumerate(D.cMice.keys()):#D.cMice.keys()
        for m,mouse in enumerate(mice):
            print(mouse)
            col_m = next(color_t)

            nSes = D.cMice[mouse].meta['nSes']
            nC = D.cMice[mouse].meta['nC']

            t_measures = get_t_measures(mouse)[:nSes]

            s_bool = np.zeros(nSes,'bool')
            s_bool[D.sessions[mouse]['steady'][0]:D.sessions[mouse]['steady'][1]] = True
            s_bool[~D.cMice[mouse].sessions['bool']] = False
            print(np.where(s_bool)[0])

            ax = plt.axes([0.075-0.065*(m%2),0.775-0.14*m,0.15,0.175])
            if m==0:
                pl_dat.add_number(fig,ax,order=1,offset=[-50,0])
            nROI = D.cMice[mouse].status[:,s_bool,:].sum(0)

            if np.any(D.cMice[mouse].para['zone_mask']['gate']):
                idx = [1,3,4,5]
            else:
                idx = [1,4,5]
                col = [pie_col[i-1] for i in idx]
                col[0] = col_m
                explode = [pie_explode[i-1] for i in idx]
                nROI_norm = nROI[...,idx].mean(0)

                nROI_norm = nROI_norm/nROI_norm.sum()*100
                nTotal = [nROI[...,1].mean(),nROI[...,1].std()]


                # D.cMice[mouse].session_classification(sessions=D.sessions[mouse]['analyze'])
                # D.cMice[mouse].update_status()

                # rad = nROI[...,1].mean()
                prct_labels = ['%.1f%%'%(100*val/nROI_norm.sum()) for val in nROI_norm]
                prct_labels[0] = ''
                rad = 500+nTotal[0]/2
                ax.pie(nROI_norm,explode=explode,startangle=170,radius=rad,colors=col,pctdistance=1.5,textprops={'fontsize':6},labels=prct_labels)#,labels=pie_labels,autopct='%1.1f%%',
                ax.text(0,-rad-500,'n=$%d\pm %d$'%(nTotal[0],nTotal[1]),fontsize=6,color='k',ha='center')
                ax.text(0,-rad+250,'%s'%mouse,fontsize=6,color='w',ha='center')
                ax.set_xlim([-2000,2000])
                ax.set_ylim([-2000,2000])


            status_act = D.cMice[mouse].status[D.cMice[mouse].stats['cluster_bool'],:,1]
            status_act = status_act[:,s_bool]
            status_PC = D.cMice[mouse].status[D.cMice[mouse].stats['cluster_bool'],:,2]
            status_PC = status_PC[:,s_bool]

            nC_good, nSes_good = status_act.shape

            p_act = np.count_nonzero(status_act)/(nC_good*nSes_good)
            p_PC = np.count_nonzero(status_PC)/np.count_nonzero(status_act)

            rnd_var_act = np.random.random(status_act.shape)
            rnd_var_PC = np.random.random(status_PC.shape)
            status_act_test = np.zeros(status_act.shape,'bool')
            status_act_test_rnd = np.zeros(status_act.shape,'bool')
            status_PC_test = np.zeros(status_PC.shape,'bool')
            status_PC_test_rnd = np.zeros(status_PC.shape,'bool')

            for c in range(nC_good):
                nC_act = status_act[c,:].sum()
                status_act_test[c,np.random.choice(nSes_good,nC_act,replace=False)] = True
                status_act_test_rnd[c,:] = rnd_var_act[c,:] < p_act
                # status_PC_test[c,status_act_test[c,:]] = rnd_var_PC[c,status_act_test[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
                # status_PC_test[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
                status_PC_test[c,np.where(status_act[c,:])[0][np.random.choice(nC_act,status_PC[c,:].sum(),replace=False)]] = True
                status_PC_test_rnd[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < p_PC

            for mode in ['act','code']:
                status = status_act if mode=='act' else status_PC
                status_test = status_act_test if mode=='act' else status_PC_test

                status = status_act if mode=='act' else status_PC
                status_dep = np.ones_like(status_act) if mode=='act' else status_act
                recurr_tmp = get_recurr(status,status_dep)
                recurr[mode]['data'][m,:nSes_good,0] = np.nanmean(recurr_tmp,0)
                recurr[mode]['data'][m,:nSes_good,1] = np.nanstd(recurr_tmp,0)

                status_test = status_act_test if mode=='act' else status_PC_test
                status_dep_test = np.ones_like(status_act_test) if mode=='act' else status_act_test
                recurr_tmp = get_recurr(status_test,status_dep_test)
                recurr[mode]['test'][m,:nSes_good,0] = np.nanmean(recurr_tmp,0)
                recurr[mode]['test'][m,:nSes_good,1] = np.nanstd(recurr_tmp,0)

                if mode=='act':
                    pl_dat.plot_with_confidence(ax_recurr_act,np.linspace(1,nSes_good,nSes_good),recurr[mode]['data'][m,:nSes_good,0],SD*recurr[mode]['data'][m,:nSes_good,1],col=col_m,ls='-',label='emp. data')

                    mask = np.isfinite(recurr['act']['data'][m,:,0])
                    res = curve_fit(fit_fun,x_arr[mask],recurr['act']['data'][m,mask,0],bounds=p_bounds)
                    act_paras[m,:,0] = res[0]
                    act_paras[m,:,1] = np.sqrt(np.diag(res[1]))
                    # print('mouse %s'%mouse)
                    # print(recurr['act']['data'][m,mask,0])
                    # print(res)
                    if mouse=='762':
                        ax_recurr_act.plot(x_arr,fit_fun(x_arr,*res[0]),'k--',linewidth=0.5)

                    ax_recurr_act_para.errorbar(act_paras[m,1,0],act_paras[m,3,0],xerr=act_paras[m,1,1],yerr=act_paras[m,3,1],fmt='.',color=col_m,markersize=2,linewidth=0.5)
                    # pl_dat.plot_with_confidence(ax_recurr_act,np.linspace(1,nSes_good,nSes_good),recurr[mode]['test'][m,:nSes_good,0],SD*recurr[mode]['test'][m,:nSes_good,1],col='tab:red',ls='-',label='rnd. data')
                if mode=='code':
                    ax_recurr_PC = plt.axes([0.3,0.82-0.14*m,0.15,0.08])
                    if m==0:
                        pl_dat.add_number(fig,ax_recurr_PC,order=2)
                    pl_dat.plot_with_confidence(ax_recurr_PC,np.linspace(1,nSes_good,nSes_good),recurr[mode]['data'][m,:nSes_good,0],SD*recurr[mode]['data'][m,:nSes_good,1],col='k',ls='-',label='emp. data')
                    pl_dat.plot_with_confidence(ax_recurr_PC,np.linspace(1,nSes_good,nSes_good),recurr[mode]['test'][m,:nSes_good,0],SD*recurr[mode]['test'][m,:nSes_good,1],col='tab:red',ls='--',label='rnd. data')

                    ax_recurr_PC.set_ylim([0,0.75])
                    if m==2:
                        ax_recurr_PC.set_ylabel('$p(\\beta_{s+\Delta s}^+|\\beta_s^+)$',fontsize=8)
                    pl_dat.remove_frame(ax_recurr_PC,['top','right'])
                    if m==nMice-1:
                        ax_recurr_PC.set_xlabel('$\Delta s$')

            s1_shifts,s2_shifts,f1,f2 = np.unravel_index(D.cMice[mouse].compare['pointer'].col,(nSes,nSes,D.cMice[mouse].meta['field_count_max'],D.cMice[mouse].meta['field_count_max']))
            c_shifts = D.cMice[mouse].compare['pointer'].row
            Ds = s2_shifts-s1_shifts

            sig_theta = D.cMice[mouse].stability['all']['mean'][0,2]
            p_rec = np.zeros(nSes)*np.NaN
            p_rec_overall = np.zeros(nSes)*np.NaN

            for ds in range(nSes):
                idx_ds = np.where((Ds==ds) & s_bool[s1_shifts] & s_bool[s2_shifts])[0]
                N_data = len(idx_ds)

                idx_shifts = D.cMice[mouse].compare['pointer'].data[idx_ds].astype('int')-1
                shifts = D.cMice[mouse].compare['shifts'][idx_shifts]

                N_stable = (np.abs(shifts)<(SD*sig_theta)).sum()
                p_rec[ds] = N_stable/N_data

                p_rec_overall[ds] = N_stable/D.cMice[mouse].status[:,np.where(s_bool)[0][:-ds],2].sum()

            r_random = 2*SD*D.cMice[mouse].stability['all']['mean'][0,2]/100
            if m==0:
                ax_recurr_PF.plot([1,nSes],[r_random,r_random],'--',color=[0.6,0.6,0.6],linewidth=0.5)

            ax_recurr_PF.plot(p_rec,'-',color=col_m,linewidth=0.3)

            ax_recurr_overall.plot(p_rec_overall,'-',color=col_m,linewidth=0.3)

            x_arr_overall = np.arange(len(p_rec_overall))
            mask = np.isfinite(p_rec_overall)
            res = curve_fit(fit_fun,x_arr_overall[mask],p_rec_overall[mask],bounds=p_bounds)
            overall_paras[m,:,0] = res[0]
            overall_paras[m,:,1] = np.sqrt(np.diag(res[1]))
            # print('mouse %s'%mouse)
            # print(recurr['act']['data'][m,mask,0])
            # print(res)
            if mouse=='762':
                ax_recurr_overall.plot(x_arr_overall,fit_fun(x_arr_overall,*res[0]),'k--',linewidth=0.5)




            idx_ds = np.where((Ds==1) & s_bool[s1_shifts] & s_bool[s2_shifts])[0]
            N_data = len(idx_ds)
            idx_shifts = D.cMice[mouse].compare['pointer'].data[idx_ds].astype('int')-1
            shifts = D.cMice[mouse].compare['shifts'][idx_shifts]
            for i,SD_i in enumerate(SD_arr):
                N_stable = (np.abs(shifts)<(SD_i*sig_theta)).sum()
                p_rec_SD[m,i] = N_stable/N_data

            ax_recurr_SD.plot(SD_arr,p_rec_SD[m,:],'-',color=col_m,linewidth=0.3)


            smooth_fact = 4
            corr_popvect = np.zeros((nSes,nSes,nbin))*np.NaN
            for s in np.where(s_bool)[0]:

                idx_PC = D.cMice[mouse].status[:,s,2]
                fmap_ref = gauss_smooth(D.cMice[mouse].stats['firingmap'][idx_PC,s,:],(0,smooth_fact))
                fmap_ref[np.isnan(fmap_ref)] = 0

                for ds in range(1,nSes-s):
                    if s_bool[s+ds]:
                        fmap_comp = gauss_smooth(D.cMice[mouse].stats['firingmap'][idx_PC,s+ds,:],(0,smooth_fact))
                        fmap_comp[np.isnan(fmap_comp)] = 0

                        corr_popvect[s,ds,:] = np.diag(np.corrcoef(fmap_ref,fmap_comp,rowvar=False)[nbin:,:nbin])

            ax_recurr_popvect.plot(np.nanmean(corr_popvect,axis=(0,2)),p_rec_overall,'.',color=col_m,markersize=1)



            para_act[m,0] = np.nanmean(D.cMice[mouse].stats['p_post_s']['act']['act'][s_bool,1,0],0)
            para_act[m,1] = np.nanstd(D.cMice[mouse].stats['p_post_s']['act']['act'][s_bool,1,0],0)

            para_code[m,0] = np.nanmean(D.cMice[mouse].stats['p_post_s']['code']['stable'][s_bool,1,0],0)
            para_code[m,1] = np.nanstd(D.cMice[mouse].stats['p_post_s']['code']['stable'][s_bool,1,0],0)

            j = 1
            ax_corr_p.errorbar(para_act[m,0],overall_paras[m,j,0],yerr=overall_paras[m,j,1],xerr=para_act[m,1],fmt='.',color=col_m,markersize=2,linewidth=0.5)
            ax_corr_p2.errorbar(para_code[m,0],overall_paras[m,j,0],yerr=overall_paras[m,j,1],xerr=para_code[m,1],fmt='.',color=col_m,markersize=2,linewidth=0.5)



            # for j,key in enumerate(['act','code','stable'],1):
            #     p_s = np.nanmean(D.cMice[mouse].stats['p_post_s'][key]['act'][s_bool,1,0],0)
            #     # ax_corr_p.plot(p_s,recurr['act']['data'][m,1,0],'.',markersize=j,color=col_m)
            #     ax_corr_p.plot(p_s,act_paras[m,1,0],'.',markersize=j,color=col_m)
            #
            #     p_s = np.nanmean(D.cMice[mouse].stats['p_post_s'][key]['code'][s_bool,1,0],0)
            #     # ax_corr_p2.plot(p_s,recurr['code']['data'][m,1,0],'.',markersize=j,color=col_m)
            #     ax_corr_p2.plot(p_s,act_paras[m,3,0],'.',markersize=j,color=col_m)



            # ax_corr_p.plot(p_s2,recurr['act']['data'][m,1,0],'k.',markersize=2,color=col_m)

            # p = D.cMice[mouse].stability
            # p_corr = np.minimum(1,p['all']['mean'][:,1]+(1-p['all']['mean'][:,1])*(2*SD*p['all']['mean'][1,2]/nbin))

            # p_SD = np.sqrt((1-2*SD*p['all']['mean'][1,2]/nbin)**2*p['all']['std'][:,1]**2 + ((1-p['all']['mean'][:,1])*2*SD/nbin)**2 * p['all']['std'][1,2]**2)
            # pl_dat.plot_with_confidence(ax_recurr_PF,range(1,nSes+1),p_corr,p_SD,'tab:blue','-')

            # pl_dat.plot_with_confidence(ax_recurr_PF,range(1,nSes+1),D.cMice[mouse].stability['all']['mean'][:,1],D.cMice[mouse].stability['all']['CI'][...,1].T,'k','-')

            # mask_PC = (~D.cMice[mouse].status[...,2])
            # mask_fields = D.cMice[mouse].fields['status']<3
            # dat = {}
            # dat['reliability'] = np.ma.array(D.cMice[mouse].fields['reliability'], mask=mask_fields, fill_value=np.NaN)
            # dat['width'] = np.ma.array(D.cMice[mouse].fields['width'][...,0], mask=mask_fields, fill_value=np.NaN)
            # dat['MI_value'] = np.ma.array(D.cMice[mouse].stats['MI_value'], mask=mask_PC, fill_value=np.NaN)
            # dat['max_rate'] = np.ma.array(D.cMice[mouse].fields['max_rate'], mask=mask_fields, fill_value=np.NaN)
            #
            # dat_mean = {'reliability':  np.zeros(nSes)*np.NaN,
            #             'width':        np.zeros(nSes)*np.NaN,
            #             'MI_value':     np.zeros(nSes)*np.NaN,
            #             'max_rate':     np.zeros(nSes)*np.NaN}
            #
            # for key in ['reliability','width','MI_value','max_rate']:
            #
            #     for s in np.where(D.cMice[mouse].sessions['bool'])[0]:
            #         dat_s = dat[key][:,s,...].compressed()
            #         dat_mean[key][s] = np.mean(dat_s)

            # ax = plt.subplot(4,5,i+1)

            # t_half = t_measures.max()/2
            # ax = plt.axes([0.25+0.5*(i//3),0.925-0.35*(i%3),0.175,0.03])
            # ax.boxplot(D.cMice[mouse].fields['reliability'][D.cMice[mouse].fields['status']>=3],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            # ax.set_ylim([0,0.5])
            # ax.set_xticks([])
            # # ax.set_xticks(np.linspace(0,3000,4))
            # # ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,7)])
            # ax.set_xlim([-10,t_measures.max()+10])
            # ax.plot(t_measures,dat_mean['reliability'],'.',markersize=1.5,color='k')
            # ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['reliability'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            # pl_dat.remove_frame(ax,['top'])
            # ax.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=True)
            # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            # ax.set_ylabel('a',fontsize=8)#,rotation='horizontal',labelpad=-20,y=0.9,ha='left')
            #
            # ax = plt.axes([0.25+0.5*(i//3),0.875-0.35*(i%3),0.175,0.03])
            # ax.boxplot(D.cMice[mouse].stats['MI_value'][D.cMice[mouse].status[...,2]],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            # ax.set_ylim([0,0.5])
            # ax.set_xticks([])
            # # ax.set_xticks(np.linspace(0,3000,4))
            # # ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,7)])
            # ax.set_xlim([-10,t_measures.max()+10])
            # ax.plot(t_measures,dat_mean['MI_value'],'.',markersize=1.5,color='k')
            # ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['MI_value'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            # pl_dat.remove_frame(ax,['top'])
            # ax.tick_params(axis='y',which='both',left=True,right=True,labelright=True,labelleft=False)
            # ax.yaxis.set_label_position('right')
            # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            # ax.set_ylabel('MI',fontsize=8)#,rotation='horizontal',labelpad=-50,y=1.0,ha='left')
            #
            #
            # ax = plt.axes([0.25+0.5*(i//3),0.825-0.35*(i%3),0.175,0.03])
            # ax.boxplot(D.cMice[mouse].fields['width'][D.cMice[mouse].fields['status']>=3,0],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            # ax.set_ylim([0,10])
            # # ax.set_xticks(np.linspace(0,3000,4))
            # # ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,4)])
            #
            # # ax.tick_params(which='minor',length=2)
            # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            # ax.set_xlim([-10,t_measures.max()+10])
            # ax.plot(t_measures,dat_mean['width'],'.',markersize=1.5,color='k')
            # ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['width'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            # pl_dat.remove_frame(ax,['top'])
            # ax.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=True)
            # ax.set_ylabel('$\sigma$',fontsize=8)#,rotation='horizontal',labelpad=0,y=1.0,ha='left')
            # # ax.yaxis.tick_right()
            #
            #
            # ax = plt.axes([0.25+0.5*(i//3),0.775-0.35*(i%3),0.175,0.03])
            # ax.boxplot(D.cMice[mouse].fields['max_rate'][D.cMice[mouse].fields['status']>=3],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            # ax.set_ylim([0,30])
            # ax.set_xticks(np.linspace(0,3000,4))
            # ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,4)])
            #
            # # ax.tick_params(which='minor',length=2)
            # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            # ax.set_xlim([-10,t_measures.max()+10])
            # ax.plot(t_measures,dat_mean['max_rate'],'.',markersize=1.5,color='k')
            # ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['max_rate'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            # pl_dat.remove_frame(ax,['top'])
            # ax.tick_params(axis='y',which='both',left=True,right=True,labelright=True,labelleft=False)
            # ax.yaxis.set_label_position('right')
            # ax.set_ylabel('$max(\\nu)$',fontsize=8)#,rotation='horizontal',labelpad=0,y=1.0,ha='left')
            # ax.yaxis.tick_right()

        # ax.set_xticks(range(nMice))
        # ax.set_xticklabels(D.cMice.keys())



        # ax_corr_p.errorbar(-0.1+0.2*np.random.rand(nMice),overall_paras[:,1,0],overall_paras[:,1,1],fmt='.',color='k',markersize=2,linewidth=0.5)
        # ax_corr_p2.errorbar(-0.1+0.2*np.random.rand(nMice),overall_paras[:,3,0],overall_paras[:,3,1],fmt='.',color='k',markersize=2,linewidth=0.5)

        pl_dat.remove_frame(ax_recurr_act_para,['top','right'])
        ax_recurr_act_para.set_xlabel('$\\tau_{p_{\\alpha}}$')
        ax_recurr_act_para.set_ylabel('$m_{p_{\\alpha}}$')



        for axx in [ax_recurr_act,ax_recurr_PF]:
            axx.set_ylim([0,1])
            axx.set_xlim([0,60])
            # axx.set_yscale('log')
            pl_dat.remove_frame(axx,['top','right'])
        ax_recurr_act.set_ylabel('$p(\\alpha_{s+\Delta s}^+|\\alpha_s^+)$',fontsize=8)
        ax_recurr_PF.set_ylabel('$p(\\gamma_{\Delta s}^+|\\beta^+)$',fontsize=8)
        ax_recurr_overall.set_xlim([0,60])
        ax_recurr_overall.set_ylabel('$p^*(\\gamma_{\Delta s}^+|\\beta^+)$',fontsize=8)
        ax_recurr_overall.set_xlabel('$\Delta s$')

        ax_recurr_SD.plot([1,1],[0,1],'--',color=[0.6,0.6,0.6],linewidth=0.5,zorder=0)
        ax_recurr_SD.errorbar(SD_arr,np.nanmean(p_rec_SD,0),np.nanstd(p_rec_SD,0),fmt='k-',linewidth=1.5)
        ax_recurr_SD.set_ylim([0,1])
        ax_recurr_SD.set_xlabel('SD')
        # ax_recurr_SD.set_yticklabels([])
        pl_dat.remove_frame(ax_recurr_SD,['top','right'])

        ax_recurr_overall.set_ylim([0.,0.3])
        pl_dat.remove_frame(ax_recurr_overall,['top','right'])

        ax_recurr_popvect.set_xlim([-0.2,0.5])
        ax_recurr_popvect.set_ylim([0,0.3])
        # ax_recurr_popvect.set_yticklabels([])
        pl_dat.remove_frame(ax_recurr_popvect,['top','right'])
        ax_recurr_popvect.set_xlabel('$c_{pop.vector}$')

        for axx in [ax_corr_p,ax_corr_p2]:
            axx.set_xlim([0,1])
            # axx.set_ylim([0,1])
            pl_dat.remove_frame(axx,['top','right'])
        ax_corr_p.set_xlabel('$p(\\alpha_{s+1}^+|\\alpha_{s}^+)$')
        ax_corr_p2.set_xlabel('$p(\\gamma_{1}^+|\\beta^+)$')
        ax_corr_p.set_ylabel('$\\tau_{p_{\\gamma}}$')
        # ax_corr_p2.set_yticklabels([])

        plt.show(block=False)

        if sv:
            pl_dat.save_fig('placeField_recurrence')

    if plot_fig[1]:

        print('time dependent dynamics in the hippocampus')
        print('GT-mice during learning')

        nbin=100
        if mice is None:
            mice = D.cMice.keys()

        nMice = len(mice)

        col = ['tab:red','tab:orange','tab:brown']

        fig,ax_arr = plt.subplots(3,6,figsize=(7,4),dpi=300)

        for j,axes in enumerate(ax_arr):
            for i,ax in enumerate(axes):
                ax_arr[j][i].set_position([0.1+i*0.125+(i//2)*0.075,0.725-j*0.3,0.075,0.2])

        region_arr = ['GT','RW','nRnG']

        para_arr = ['sig','fr','rel','MI']
        ax = {}
        ax['start'] = ax_arr[0][[0,2,4]]
        ax['density'] = ax_arr[0][[1,3,5]]
        ax['sig'] = ax_arr[1][[0,2,4]]
        ax['fr'] = ax_arr[1][[1,3,5]]
        ax['rel'] = ax_arr[2][[0,2,4]]
        ax['MI'] = ax_arr[2][[1,3,5]]

        s_arr = [0,5,10,15,20]
        nS = len(s_arr)-1
        cmap = get_cmap('tab10')
        cmap_cycle = iter(cmap.colors)

        data = {}
        data['density'] = {}
        for p in para_arr:
            data[p] = {}
        for key in region_arr:
            for p in para_arr:
                data['density'][key] = np.zeros((nMice,nS))
                data[p][key] = np.zeros((nMice,nS,2))

        pl_dat = plot_dat(folder='mapChanges',sv_ext=sv_ext)
        for m,mouse in enumerate(mice):

            nSes = D.cMice[mouse].meta['nSes']
            nC = D.cMice[mouse].meta['nC']
            D.cMice[mouse].session_data = get_session_specifics(mouse,nSes)

            col=next(cmap_cycle)

            fields = np.zeros((nbin,s_arr[-1]-s_arr[0]))
            for i,s in enumerate(range(s_arr[0],s_arr[-1])):
                idx_PC = np.where(D.cMice[mouse].status_fields[:,s,:])
                fields[:,s] = np.nansum(D.cMice[mouse].fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
            fields /= fields.sum(0)
            fields = gauss_smooth(fields,(2,0))

            ## define session ranges (0-5,5-10,10-15)
            loc = D.cMice[mouse].fields['location'][...,0]

            para = {}
            para['sig'] = D.cMice[mouse].fields['width'][...,0]
            para['fr'] = D.cMice[mouse].stats['if_firingrate_adapt']
            para['rel'] = D.cMice[mouse].fields['reliability']
            para['MI'] = np.pad(D.cMice[mouse].stats['MI_value'][...,np.newaxis],((0,0),(0,0),(0,4)),mode='edge')

            for j in range(len(s_arr)-1):

                ## get GT & RW dependent statistics (density, stability, turnover/recruitment, )
                zone_mask = {}
                zone_mask['RW'] = np.zeros(nbin).astype('bool')#range(zone_idx['reward'][0],zone_idx['reward'][-1])
                zone_mask['GT'] = np.zeros(nbin).astype('bool')
                zone_mask['nRnG'] = np.ones(nbin).astype('bool')

                RW_pos = D.cMice[mouse].session_data['RW_pos'][s_arr[j],:].astype('int')
                GT_pos = D.cMice[mouse].session_data['GT_pos'][s_arr[j],:].astype('int')
                zone_mask['RW'][RW_pos[0]:RW_pos[1]] = True
                zone_mask['nRnG'][zone_mask['RW']] = False
                if ~np.isnan(D.cMice[mouse].session_data['GT_pos'][s_arr[0],0]):
                    zone_mask['GT'][GT_pos[0]:GT_pos[1]] = True
                    zone_mask['nRnG'][zone_mask['GT']] = False
                cum_field = np.nanmean(fields[:,s_arr[j]:s_arr[j+1]],1)

                for key in region_arr:
                    data['density'][key][m,j] = cum_field[zone_mask[key]].mean()

                mask = ((np.arange(nSes)>=s_arr[j]) & (np.arange(nSes)<s_arr[j+1]))[np.newaxis,:,np.newaxis] & D.cMice[mouse].status_fields
                idx = {}
                idx['RW'] = (loc>=RW_pos[0]) & (loc<=RW_pos[1])
                idx['GT'] = (loc>=GT_pos[0]) & (loc<=GT_pos[1])
                idx['nRnG'] = ((loc<GT_pos[0]) & ((loc<RW_pos[0]) | (loc>RW_pos[1]))) | ((loc>GT_pos[1]) & ((loc<RW_pos[0]) | (loc>RW_pos[1])))

                for key in region_arr:
                    for p in para_arr:
                        data[p][key][m,j,0] = np.nanmean(para[p][idx[key]&mask])
                        data[p][key][m,j,1] = np.nanstd(para[p][idx[key]&mask])

            for j,key in enumerate(region_arr):
                ax['density'][j].plot(range(nS),data['density'][key][m,:],'-',color=col,linewidth=0.3)
                for p in para_arr:
                    ax[p][j].errorbar(range(nS),data[p][key][m,:,0],data[p][key][m,:,1],fmt='-',color=col,linewidth=0.3)


        for j,key in enumerate(region_arr):
            ax['density'][j].errorbar(range(nS),data['density'][key].mean(0),data['density'][key].std(0),fmt='-',color='k',linewidth=1)
            for p in para_arr:
                ax[p][j].errorbar(range(nS),data[p][key][...,0].mean(0),data[p][key][...,1].std(0),fmt='-',color='k',linewidth=1)


        for axx in ax['density']:
            axx.plot([-2,nS],[1/nbin,1/nbin],'k--',zorder=0,linewidth=0.3)
            axx.set_ylim([0,0.02])
            axx.yaxis.set_label_position('right')
            axx.set_ylabel('density')
            # ax.set_xticklabels(['s%d-%d'%(s_arr[j],s_arr[j+1]) for j in range(nS)],rotation=60)

        for axx in ax['sig']:
            axx.set_ylim([0,10])
            axx.set_ylabel('$\sigma$',fontsize=8,rotation='horizontal')
            axx.yaxis.set_label_coords(0.2,0.8)
            # ax.set_xticklabels(['s%d-%d'%(s_arr[j],s_arr[j+1]) for j in range(nS)],rotation=60)
        for axx in ax['fr']:
            axx.set_ylim([0,2])
            axx.set_ylabel('$\\nu^*$',fontsize=8,rotation='horizontal')
            axx.yaxis.set_label_coords(0.2,0.8)

        for axx in ax['rel']:
            axx.set_ylim([0,1])
            axx.set_ylabel('$a$',fontsize=8,rotation='horizontal')
            axx.yaxis.set_label_coords(0.2,0.8)

        for axx in ax['MI']:
            axx.set_ylim([0,1])
            axx.set_ylabel('$MI$',fontsize=8,rotation='horizontal')
            axx.yaxis.set_label_coords(0.2,0.8)

        for j,axes in enumerate(ax_arr):
            for i,axx in enumerate(axes):
                pl_dat.remove_frame(axx,['top','right'])
                axx.set_xticks(range(nS))
                axx.set_xlim([-1,nS-0.8])
                if j < 2:
                    axx.set_xticklabels([])
                else:
                    axx.set_xticklabels(['s%d-%d'%(s_arr[k],s_arr[k+1]) for k in range(nS)],rotation=60,fontsize=6)

        pic_path = ['/home/wollex/Data/Science/PhD/Thesis/pics/others/gate.jpg','/home/wollex/Data/Science/PhD/Thesis/pics/others/drop.png','']

        for j,axx in enumerate(ax['start']):
            axx.axis('off')
            if pic_path[j] == '':
                continue
            im = mpimg.imread(pic_path[j])
            axx.imshow(im)
            axx.set_xlim([0,im.shape[1]])

        plt.show(block=False)

        if sv:
            pl_dat.save_fig('learning_stats')


    if plot_fig[2]:

        print('### plot probabilities of interaction, etc ###')

        if mice is None:
            mice = D.cMice.keys()

        fig = plt.figure(figsize=(7,5),dpi=300)

        # col = ['tab:red','tab:orange','tab:brown']

        status_arr = ['act','code','stable']
        label_arr = ['alpha','beta','gamma']
        ax = {}
        ax_range = {}
        ax_abs = {}
        ds_max = 20
        p_post_bs = {}
        p_ref = {}
        for j,key in enumerate(status_arr):
            p_post_bs[key] = {}
            p_ref[key] = {}
            ax[key] = {}
            ax_range[key] = {}
            ax_abs[key] = {}
            for i,key2 in enumerate(status_arr):
                p_post_bs[key][key2] = np.zeros((len(mice),ds_max,2,2))*np.NaN
                p_ref[key][key2] = np.zeros(len(mice))*np.NaN

                ID = i+3*j
                ax[key][key2] = plt.subplot(5,7,ID+1)
                ax_range[key][key2] = plt.subplot(5,7,ID+10)
                ax_abs[key][key2] = plt.subplot(5,7,ID+20)


        width1 = 0.075
        height1 = 0.08
        ax['act']['act'].set_position([0.1,0.75,width1,height1])
        ax['act']['code'].set_position([0.1,0.425,width1,height1])
        ax['act']['stable'].set_position([0.1,0.1,width1,height1])
        ax['code']['act'].set_position([0.425,0.75,width1,height1])
        ax['code']['code'].set_position([0.425,0.425,width1,height1])
        ax['code']['stable'].set_position([0.425,0.1,width1,height1])
        ax['stable']['act'].set_position([0.75,0.75,width1,height1])
        ax['stable']['code'].set_position([0.75,0.425,width1,height1])
        ax['stable']['stable'].set_position([0.75,0.1,width1,height1])

        height3 = 0.05
        ax_abs['act']['act'].set_position([0.1,0.875,width1,height3])
        ax_abs['act']['code'].set_position([0.1,0.55,width1,height3])
        ax_abs['act']['stable'].set_position([0.1,0.225,width1,height3])
        ax_abs['code']['act'].set_position([0.425,0.875,width1,height3])
        ax_abs['code']['code'].set_position([0.425,0.55,width1,height3])
        ax_abs['code']['stable'].set_position([0.425,0.225,width1,height3])
        ax_abs['stable']['act'].set_position([0.75,0.875,width1,height3])
        ax_abs['stable']['code'].set_position([0.75,0.55,width1,height3])
        ax_abs['stable']['stable'].set_position([0.75,0.225,width1,height3])

        width2 = 0.1
        height2 = 0.15
        ax_range['act']['act'].set_position([0.22,0.75,width2,height2])
        ax_range['act']['code'].set_position([0.22,0.425,width2,height2])
        ax_range['act']['stable'].set_position([0.22,0.1,width2,height2])
        ax_range['code']['act'].set_position([0.545,0.75,width2,height2])
        ax_range['code']['code'].set_position([0.545,0.425,width2,height2])
        ax_range['code']['stable'].set_position([0.545,0.1,width2,height2])
        ax_range['stable']['act'].set_position([0.87,0.75,width2,height2])
        ax_range['stable']['code'].set_position([0.87,0.425,width2,height2])
        ax_range['stable']['stable'].set_position([0.87,0.1,width2,height2])


        cmap = get_cmap('tab10')
        cmap_cycle = iter(cmap.colors)
        pl_dat = plot_dat(folder='popDyn',sv_ext='png')


        for m,mouse in enumerate(mice):
            print(mouse)
            col = next(cmap_cycle)
            # ds_max = D.cMice[mouse].stats['p_pre_s']['act']['act'].shape[1]
            nSes = D.cMice[mouse].meta['nSes']
            nC = D.cMice[mouse].meta['nC']


            status, status_dep = get_status_arr(D.cMice[mouse])
            # print(D.sessions[mouse]['steady'])
            s_bool = (np.arange(nSes) >= (D.sessions[mouse]['steady'][0]-1)) & (np.arange(nSes)<D.sessions[mouse]['steady'][1])
            for key in status_arr:

                status[key] = status[key][D.cMice[mouse].stats['cluster_bool'],:]
                status[key] = status[key][:,D.cMice[mouse].sessions['bool']&s_bool]
                status_dep[key] = status_dep[key][D.cMice[mouse].stats['cluster_bool'],:]
                status_dep[key] = status_dep[key][:,D.cMice[mouse].sessions['bool']&s_bool]

            for j,key in enumerate(status_arr):
                for i,key2 in enumerate(status_arr):

                    if key=='act':
                        p_ref[key][key2][m] = (status[key2][status_dep[key2]==1]==1).mean()
                        key_ref = ''
                    else:
                        key_ref = status_arr[j-1]
                        # p_ref[key][key2][m] = np.nanmean(D.cMice[mouse].stats['p_post_s'][key_ref][key2][s_bool,1,0],0)
                        p_ref[key][key2][m] = np.nanmean(D.cMice[mouse].stats['p_post_c'][key_ref][key2][D.cMice[mouse].stats['cluster_bool'],1,0],0)
                        # else:
                            # p_ref = np.nanmean(D.cMice[mouse].stats['p1_post_s'][key][key_ref][:,1,0],0)

                    # print('p(%s,%s), ref (key=%s): %.2f'%(key2,key,key_ref,p_ref[key][key2][m]))
                    ax[key][key2].plot([0.5,2.5],[0,0],'k--',linewidth=0.5)
                    ax_range[key][key2].plot([0.5,ds_max+0.5],[0,0],'k--',linewidth=0.5)

                    for ds in range(1,ds_max):
                        # p_post_bs[key][key2][m,ds,0,:] = bootstrap_data(np.nanmean,D.cMice[mouse].stats['p_post_s'][key][key2][s_bool,ds,0],N_bs)
                        # p_post_bs[key][key2][m,ds,1,:] = bootstrap_data(np.nanmean,D.cMice[mouse].stats['p_post_s'][key][key2][s_bool,ds,1],N_bs)

                        p_post_bs[key][key2][m,ds,0,:] = bootstrap_data(np.nanmean,D.cMice[mouse].stats['p_post_c'][key][key2][D.cMice[mouse].stats['cluster_bool'],ds,0],N_bs)
                        p_post_bs[key][key2][m,ds,1,:] = bootstrap_data(np.nanmean,D.cMice[mouse].stats['p_post_c'][key][key2][D.cMice[mouse].stats['cluster_bool'],ds,1],N_bs)


                    ax_abs[key][key2].errorbar([1,2],p_post_bs[key][key2][m,1,:,0],p_post_bs[key][key2][m,1,:,1],fmt='-',color=col,linewidth=0.2,label='$\Delta p(\\%s_{s+\Delta s}^+|\\%s_s^{\pm})$'%(label_arr[i],label_arr[j]))

                    ax[key][key2].errorbar([1,2],p_post_bs[key][key2][m,1,:,0]-p_ref[key][key2][m],p_post_bs[key][key2][m,1,:,1],fmt='-',color=col,linewidth=0.2,label='$\Delta p(\\%s_{s+\Delta s}^+|\\%s_s^{\pm})$'%(label_arr[i],label_arr[j]))

                    ### plot long-range interaction
                    ax_range[key][key2].errorbar(range(ds_max),p_post_bs[key][key2][m,:,0,0]-p_ref[key][key2][m],p_post_bs[key][key2][m,:,0,1],fmt='-',color=col,linewidth=0.2,label='$\Delta p(\\%s_{s+1}^+|\\%s_{s}^{+})$'%(label_arr[i],label_arr[j]) if (j==0) else None)


                    # ax.set_xlabel('$\Delta s$')
                    # ax_range[key][key2].legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.4,1.1])

        for j,key in enumerate(status_arr):
            for i,key2 in enumerate(status_arr):

                ax_abs[key][key2].errorbar([1,2],p_post_bs[key][key2][:,1,:,0].mean(0),p_post_bs[key][key2][:,1,:,1].std(0),fmt='-',color='k',linewidth=1,label='$\Delta p(\\%s_{s+\Delta s}^+|\\%s_s^{\pm})$'%(label_arr[i],label_arr[j]))

                ax[key][key2].errorbar([1,2],(p_post_bs[key][key2][:,1,:,0]-p_ref[key][key2][:,np.newaxis]).mean(0),p_post_bs[key][key2][:,1,:,1].std(0),fmt='-',color='k',linewidth=1,label='$\Delta p(\\%s_{s+\Delta s}^+|\\%s_s^{\pm})$'%(label_arr[i],label_arr[j]))

                ### plot long-range interaction
                ax_range[key][key2].errorbar(range(ds_max),(p_post_bs[key][key2][:,:,0,0]-p_ref[key][key2][:,np.newaxis]).mean(0),p_post_bs[key][key2][:,:,0,1].std(0),fmt='-',color='k',linewidth=1,label='$\Delta p(\\%s_{s+1}^+|\\%s_{s}^{+})$'%(label_arr[i],label_arr[j]) if (j==0) else None)


                ax[key][key2].set_xticks([1,2])
                ax[key][key2].set_xlim([0.75,2.25])
                ax[key][key2].set_ylim([-0.4,0.4])
                ax[key][key2].set_yticks(np.linspace(-0.2,0.2,3))
                ax[key][key2].set_xticklabels(['$\\%s_1^+$'%label_arr[j],'$\\%s_1^-$'%label_arr[j]])
                if j==0:
                    ax_abs[key][key2].set_ylabel('$p(\\%s_{s+\Delta s}|\cdot)$'%label_arr[i])#'(\\alpha_{s+\Delta s}^+|\\beta_s^{\pm})$')
                    ax_abs[key][key2].yaxis.set_label_coords(-0.8,-0.25)
                    # ax[key][key2].set_ylabel('$\Delta p$')#'(\\alpha_{s+\Delta s}^+|\\beta_s^{\pm})$')
                if i==0:
                    ax_abs[key][key2].set_title('$p(\cdot|\\%s_{s})$'%label_arr[j],x=1.3,y=1.2)
                pl_dat.remove_frame(ax[key][key2],['top','right'])

                ax_abs[key][key2].set_xticks([1,2])
                ax_abs[key][key2].set_xticklabels([])
                ax_abs[key][key2].set_xlim([0.75,2.25])
                ax_abs[key][key2].set_ylim([0,1])
                ax_abs[key][key2].set_yticks(np.linspace(0,1,3))
                pl_dat.remove_frame(ax_abs[key][key2],['top','right'])

                pl_dat.remove_frame(ax_range[key][key2],['top','left','right'])
                ax_range[key][key2].set_yticks([])
                ax_range[key][key2].set_ylim([-0.4,0.4])
                ax_range[key][key2].set_yticks(np.linspace(-0.2,0.2,3))
                ax_range[key][key2].set_xlabel('$\Delta s$')

        plt.show(block=False)

        if sv:
            pl_dat.save_fig('interaction_multiple')


    if plot_fig[3]:

        print('### plot stuff ###')
        props = dict(boxstyle='round', facecolor='w', alpha=0.8)

        if mice is None:
            mice = D.cMice.keys()

        plt.figure(figsize=(7,5),dpi=300)

        # col = ['tab:red','tab:orange','tab:brown']

        # ax = plt.axes([0.25,0.9,0.0,0.0])
        # for j,mouse in enumerate(mice):
            # ax.plot(1,np.NaN,'o',color=col[j],label='m%s'%mouse)
        # ax.legend()
        nbin = 100
        for m,mouse in enumerate(mice):

            nSes = D.cMice[mouse].meta['nSes']
            nC = D.cMice[mouse].meta['nC']

            pl_dat = plot_dat(D.cMice[mouse].meta['mouse'],pathcat([D.cMice[mouse].meta['pathMouse'],'Figures']),nSes,D.cMice[mouse].para,sv_ext='png')

            fields = np.zeros((nbin,nSes))
            for i,s in enumerate(np.where(D.cMice[mouse].sessions['bool'])[0]):
                # idx_PC = np.where(D.cMice[mouse].fields['status'][:,s,:]>=3)
                idx_PC = np.where(D.cMice[mouse].status_fields[:,s,:])
                # idx_PC = np.where(~np.isnan(D.cMice[mouse].fields['location'][:,s,:]))
                # fields[s,:] = np.nansum(D.cMice[mouse].fields['p_x'][:,s,:,:],1).sum(0)
                fields[:,s] = np.nansum(D.cMice[mouse].fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
            fields /= fields.sum(0)
            fields = gauss_smooth(fields,(2,0))

            ax = plt.axes([0.1+0.3*m,0.8,0.15,0.15])
            ax.plot(gauss_smooth(np.nanmean(fields[D.cMice[mouse].para['zone_mask']['reward'],:],0),1),color='tab:red',label='reward')
            ax.plot(gauss_smooth(np.nanmean(fields[D.cMice[mouse].para['zone_mask']['gate'],:],0),1),color='tab:green',label='gate')
            ax.plot(gauss_smooth(np.nanmean(fields[D.cMice[mouse].para['zone_mask']['others'],:],0),1),color='tab:blue',label='other')
            ax.set_ylim([0,0.02])
            ax.set_xlim([0,np.where(D.cMice[mouse].sessions['bool'])[0][-1]])
            pl_dat.remove_frame(ax,['top','right'])
            ax.set_title('mouse %s'%mouse)
            ax.set_xlabel('session')
            if m == 0:
                ax.set_ylabel('PC density')
                ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=[0.8,0.3])


            if nSes>50:
                s_arr = np.array([0,5,17,30,87])
            else:
                s_arr = np.array([0,5,15,20,28])
            s_arr += np.where(D.cMice[mouse].sessions['bool'])[0][0]
            print(s_arr)
            n_int = len(s_arr)-1

            p_rec_loc = np.zeros((n_int,nbin,nSes))*np.NaN

            s1_shifts,s2_shifts,f1,f2 = np.unravel_index(D.cMice[mouse].compare['pointer'].col,(nSes,nSes,D.cMice[mouse].meta['field_count_max'],D.cMice[mouse].meta['field_count_max']))
            Ds = s2_shifts-s1_shifts

            c_shifts = D.cMice[mouse].compare['pointer'].row
            sig_theta = D.cMice[mouse].stability['all']['mean'][:10,2].mean()
            di = 3

            for ds in range(1,min(nSes,21)):
                session_bool = np.where(np.pad(D.cMice[mouse].sessions['bool'][ds:],(0,ds),constant_values=False) & np.pad(D.cMice[mouse].sessions['bool'][:],(0,0),constant_values=False))[0]

                idx = np.where(Ds==ds)[0]
                idx_shifts = D.cMice[mouse].compare['pointer'].data[idx].astype('int')-1
                shifts = D.cMice[mouse].compare['shifts'][idx_shifts]

                s = s1_shifts[idx]
                f = f1[idx]
                c = c_shifts[idx]
                loc_shifts = np.round(D.cMice[mouse].fields['location'][c,s,f,0]).astype('int')

                for j in range(len(s_arr)-1):
                    for i in range(nbin):
                        i_min = max(0,i-di)
                        i_max = min(nbin,i+di)
                        idx_loc = (loc_shifts>=i_min) & (loc_shifts<i_max) & ((s>=s_arr[j]) & (s<s_arr[j+1]))

                        shifts_loc = shifts[idx_loc]
                        N_data = len(shifts_loc)
                        N_stable = (np.abs(shifts_loc)<(SD*sig_theta)).sum()

                        p_rec_loc[j,i,ds] = N_stable/N_data

            for j in range(n_int):
                ax_im = plt.axes([0.1+m*0.3,0.55-j*0.15,0.15,0.1])
                im = ax_im.imshow(gauss_smooth(p_rec_loc[j,...],(1,0)),clim=[0.4,0.8],interpolation='None',origin='lower',aspect='auto')
                cb = plt.colorbar(im)
                cb.set_label('stable fraction',fontsize=6)
                ax_im.set_xlim([0.5,10.5])
                ax_im.set_ylim([0,100])
                if m==0:
                    ax_im.text(x=2,y=107,s='Sessions %d-%d'%(s_arr[j]+1,s_arr[j+1]),ha='right',va='bottom',bbox=props,fontsize=8)
                    ax_im.set_ylabel('pos.')
                # if j==0:
                    # ax_im.set_title('stable fraction',fontsize=8)

                if j == (n_int-1):
                    ax_im.set_xlabel('$\Delta s$')


        plt.show(block=False)

        if sv:
            pl_dat.save_fig('timestab_multi')


    if plot_fig[4]:

        nbin=100

        status_arr = ['act','code','stable']

        fig = plt.figure(figsize=(7,4),dpi=300)
        pl_dat = plot_dat(folder='mapChanges',sv_ext='png')
        nSes = D.cMice[mouse_ex].meta['nSes']
        nC = D.cMice[mouse_ex].meta['nC']

        s_arr = [24,44,74]

        ax = plt.axes([0.1,0.825,0.35,0.15])
        pl_dat.add_number(fig,ax,order=1,offset=[-150,0])
        pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/Setup/VR_sketch_noGT.png'

        ax.axis('off')
        im = mpimg.imread(pic_path)
        ax.imshow(im)
        ax.set_xlim([0,im.shape[1]])

        props = dict(boxstyle='round', facecolor='w', alpha=0.8)
        for j,s in enumerate(s_arr):
            if s < nSes:
                ax = plt.axes([0.075+0.12*j,0.425,0.1,0.3])
                if j == 0:
                    pl_dat.add_number(fig,ax,order=2,offset=[-100,50])
                idxes_tmp = np.where(D.cMice[mouse_ex].status_fields[:,s,:] & (D.cMice[mouse_ex].stats['SNR'][:,s]>2)[...,np.newaxis] & (D.cMice[mouse_ex].stats['r_values'][:,s]>0)[...,np.newaxis] & (D.cMice[mouse_ex].stats['match_score'][:,s,0]>0.5)[...,np.newaxis])
                idxes = idxes_tmp[0]
                sort_idx = np.argsort(D.cMice[mouse_ex].fields['location'][idxes_tmp[0],s,idxes_tmp[1],0])
                sort_idx = idxes[sort_idx]
                nID = len(sort_idx)

                firingmap = D.cMice[mouse_ex].stats['firingmap'][sort_idx,s,:]
                firingmap = gauss_smooth(firingmap,[0,2])
                firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
                # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
                im = ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,5])
                ax.text(5,nID*0.95,'n = %d'%nID,bbox=props,color='k',fontsize=6)
                ax.text(95,nID/10,'Session %d'%(s+1),bbox=props,color='k',fontsize=6,ha='right')
                pl_dat.remove_frame(ax)
                ax.set_xticks([])
                ax.set_yticks([])

        cbaxes = plt.axes([0.425,0.65,0.01,0.075])
        h_cb = plt.colorbar(im,cax=cbaxes)
        h_cb.set_label('$Ca^{2+}$',fontsize=8)
        h_cb.set_ticks([0,5])
        h_cb.set_ticklabels(['low','high'])


        par_keys = ['reliability','if_firingrate_adapt','MI_value','width']#'oof_firingrate_adapt'
        par_labels = ['a','$\\nu^*$','MI','$\sigma$']
        ranges = np.array([[0,1],[0,2.5],[0,0.5],[0,20]])

        dense_mean = np.zeros((nMice,2,2))*np.NaN
        distr = {}
        distr_mean = {}
        for key in par_keys:
            distr[key] = np.zeros((nMice,nbin,2))*np.NaN
            distr_mean[key] = np.zeros((nMice,2,2))*np.NaN

        ax_dense = plt.axes([0.9,0.825,0.075,0.1])
        ax_para1 = plt.axes([0.9,0.625,0.075,0.1])
        ax_para2 = plt.axes([0.9,0.455,0.075,0.1])
        ax_para3 = plt.axes([0.9,0.285,0.075,0.1])
        ax_para4 = plt.axes([0.9,0.115,0.075,0.1])

        ax_para = [ax_para1,ax_para2,ax_para3,ax_para4]

        for m,mouse in enumerate(mice):
            nSes = D.cMice[mouse].meta['nSes']
            nC = D.cMice[mouse].meta['nC']
            s_bool = np.zeros(nSes,'bool')
            s_bool[D.sessions[mouse]['steady'][0]:D.sessions[mouse]['steady'][1]] = True
            s_bool[~D.cMice[mouse].sessions['bool']] = False

            RW_pos = D.cMice[mouse].session_data['RW_pos'][D.sessions[mouse]['steady'][0],:].astype('int')
            GT_pos = D.cMice[mouse].session_data['GT_pos'][D.sessions[mouse]['steady'][0],:].astype('int')
            zone_nRnG = np.ones(nbin,'bool')
            zone_nRnG[RW_pos[0]:RW_pos[1]] = False
            zone_nRnG[GT_pos[0]:GT_pos[1]] = False
            zone_nRnG[:10] = False
            zone_nRnG[-10:] = False
            zone_RW = np.zeros(nbin)
            zone_RW[RW_pos[0]:RW_pos[1]] = 100


            fields = np.zeros((nbin,nSes))
            for i,s in enumerate(np.where(D.cMice[mouse].sessions['bool'])[0]):
                idx_PC = np.where(D.cMice[mouse].status_fields[:,s,:])
                fields[:,s] = np.nansum(D.cMice[mouse].fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
            fields = gauss_smooth(fields,(2,0))
            fields /= fields.sum(0)[np.newaxis,:]

            if mouse == mouse_ex:
                ax_im = plt.axes([0.1,0.115,0.325,0.2])
                pl_dat.add_number(fig,ax_im,order=3)
                im = ax_im.imshow(fields,clim=[0.005,0.02],origin='lower',aspect='auto',cmap='hot')#,clim=[0,1])
                ax_im.set_xlim([-0.5,nSes-0.5])
                ax_im.set_xlim([-0.5+np.where(D.cMice[mouse_ex].sessions['bool'])[0][0],np.where(D.cMice[mouse_ex].sessions['bool'])[0][-1]-0.5])
                ax_im.set_ylim([0,100])
                ax_im.set_xlabel('session')
                ax_im.set_ylabel('position')

                for i in range(len(s_arr)):
                    col = [0.7-0.35*i,0.7-0.35*i,0.7-0.35*i]
                    ax_im.annotate(s='',xy=(s_arr[i],100),xytext=(s_arr[i],110),fontsize=6,annotation_clip=False,arrowprops=dict(arrowstyle='->',color=col))

                cbaxes = plt.axes([0.44,0.115,0.01,0.2])
                h_cb = plt.colorbar(im,cax=cbaxes)
                h_cb.set_label('place field \ndensity',fontsize=8)
                h_cb.set_ticks([])

                ax = plt.axes([0.65,0.825,0.2,0.1])
                pl_dat.add_number(fig,ax,order=4,offset=[-250,50])
                ax.plot([10,10],[0,100],'--',color='k',linewidth=0.5,zorder=10)
                ax.plot([90,90],[0,100],'--',color='k',linewidth=0.5,zorder=10)
                ax.bar(np.arange(nbin),zone_RW,width=1,facecolor='tab:red',alpha=0.3,zorder=0)
                ax_dense.plot([0,nbin],[0.01,0.01],'--',color=[0.6,0.6,0.6],linewidth=0.5,zorder=0)
                ax.plot(fields[:,s_bool],'-',color=[0.6,0.6,0.6],linewidth=0.3,alpha=0.5)
                ax.plot(np.nanmean(fields[:,s_bool],1),color='k')
                ax.set_ylim([0,0.03])
                ax.set_xlim([0,100])
                ax.set_xticklabels([])
                # ax.set_yticks([])
                # ax.set_xlabel('position')
                ax.set_ylabel('density')
                ax.yaxis.set_label_coords(-0.35,0.5)
                pl_dat.remove_frame(ax,['top','right'])

            dense_mean[m,0,0] = np.nanmean(fields[zone_nRnG,:][:,s_bool])
            dense_mean[m,0,1] = np.nanstd(fields[zone_nRnG,:][:,s_bool])
            dense_mean[m,1,0] = np.nanmean(fields[RW_pos[0]:RW_pos[1],s_bool])
            dense_mean[m,1,1] = np.nanstd(fields[RW_pos[0]:RW_pos[1],s_bool])
            ax_dense.errorbar([0,1],dense_mean[m,:,0]/0.01,dense_mean[m,:,1]/0.01,linestyle='-',marker='.',color=[0.6,0.6,0.6],linewidth=0.5,markersize=2)

            loc = np.round(D.cMice[mouse].fields['location'][...,0]).astype('int')
            ### location-specific parameters
            ## width, rel, MI, max_rate
            for j,key in enumerate(par_keys):

                if key in ['rate','oof_firingrate_adapt','if_firingrate_adapt','MI_value']:
                    if key in ['rate']:
                        dat = D.cMice[mouse].stats['if_firingrate_adapt']/D.cMice[mouse].stats['oof_firingrate_adapt'][...,np.newaxis]
                    else:
                        dat = D.cMice[mouse].stats[key]
                elif key == 'width':
                    dat = D.cMice[mouse].fields[key][...,0]
                else:
                    dat = D.cMice[mouse].fields[key]

                for i in range(nbin):
                    idx = ((loc == i) & D.cMice[mouse].status_fields & s_bool[np.newaxis,:,np.newaxis])
                    if key in ['oof_firingrate_adapt','if_firingrate_adapt','MI_value','rate']:
                        idx = np.any(idx,-1)
                    distr[key][m,i,0] = np.nanmean(dat[idx])
                    distr[key][m,i,1] = np.nanstd(dat[idx])
                idx = np.where(D.cMice[mouse].status_fields)

                if mouse==mouse_ex:

                    ax = plt.axes([0.65,0.625-j*0.17,0.2,0.1])
                    ax.plot([10,10],[0,100],'--',color='k',linewidth=0.5,zorder=10)
                    ax.plot([90,90],[0,100],'--',color='k',linewidth=0.5,zorder=10)
                    if j==0:
                        pl_dat.add_number(fig,ax,order=5,offset=[-250,50])
                    ax.bar(np.arange(nbin),zone_RW,width=1,facecolor='tab:red',alpha=0.3,zorder=0)
                    if key in ['oof_firingrate_adapt','if_firingrate_adapt','MI_value','rate']:
                        ax.plot(D.cMice[mouse].fields['location'][idx[0],idx[1],idx[2],0],dat[idx[0],idx[1]],'.',color=[0.6,0.6,0.6],markersize=1,markeredgewidth=0,zorder=0)
                    else:
                        ax.plot(D.cMice[mouse].fields['location'][idx[0],idx[1],idx[2],0],dat[idx[0],idx[1],idx[2]],'.',color=[0.6,0.6,0.6],markersize=1,markeredgewidth=0,zorder=0)
                    pl_dat.plot_with_confidence(ax,np.linspace(0,nbin-1,nbin),distr[key][m,:,0],distr[key][m,:,1],col='k')
                    ax.set_ylabel(par_labels[j],rotation='vertical',ha='left',va='center')
                    ax.yaxis.set_label_coords(-0.35,0.5)
                    pl_dat.remove_frame(ax,['top','right'])
                    if j < 3:
                        ax.set_xticklabels([])
                    else:
                        ax.set_xlabel('position [bins]')
                    ax.set_ylim(ranges[j,:])
                    ax.set_xlim([0,100])
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

                distr_mean[key][m,0,0] = np.nanmean(distr[key][m,zone_nRnG,0])
                distr_mean[key][m,0,1] = np.nanstd(distr[key][m,zone_nRnG,0])
                distr_mean[key][m,1,0] = np.nanmean(distr[key][m,RW_pos[0]:RW_pos[1],0])
                distr_mean[key][m,1,1] = np.nanstd(distr[key][m,RW_pos[0]:RW_pos[1],0])

                ax_para[j].errorbar([0,1],distr_mean[key][m,:,0],distr_mean[key][m,:,1],linestyle='-',marker='.',color=[0.6,0.6,0.6],linewidth=0.5,markersize=2)
                # ax_para[j].errorbar(0+np.random.rand()*0.2,np.nanmean(distr[key][m,zone_nRnG,0]),np.nanstd(distr[key][m,zone_nRnG,0]),fmt='.',color=[0.6,0.6,0.6],linewidth=0.5,markersize=2)
                # ax_para[j].errorbar(1+np.random.rand()*0.2,np.nanmean(distr[key][m,RW_pos[0]:RW_pos[1],0]),np.nanstd(distr[key][m,RW_pos[0]:RW_pos[1],0]),fmt='.',color=[0.6,0.6,0.6],linewidth=0.5,markersize=2)


        for j in range(len(ax_para)):
            ax_para[j].errorbar([0,1],np.nanmean(distr_mean[par_keys[j]][...,0],0),np.nanstd(distr_mean[par_keys[j]][...,1],0),fmt='-',color='k',linewidth=1,markersize=2)

            ax_para[j].set_ylim(ranges[j,:])
            ax_para[j].set_xlim([-0.5,1.5])
            ax_para[j].set_xticks([0,1])
            ax_para[j].set_yticklabels([])
            ax_para[j].yaxis.set_minor_locator(AutoMinorLocator(2))
            pl_dat.remove_frame(ax_para[j],['top','right'])
            if j==3:
                ax_para[j].set_xticklabels(['PC','RW'],rotation=60)
            else:
                ax_para[j].set_xticklabels([])

        ax_dense.errorbar([0,1],np.nanmean(dense_mean[...,0],0)/0.01,np.nanstd(dense_mean[...,1],0)/0.01,fmt='-',color='k',linewidth=1,markersize=2)
        ax_dense.plot([-1,2],[1,1],'--',color=[0.6,0.6,0.6],linewidth=0.5,zorder=0)
        ax_dense.set_ylim([0.5,1.5])
        ax_dense.set_xlim([-0.5,1.5])
        ax_dense.set_xticks([0,1])
        ax_dense.set_xticklabels([])
        # ax_dense.text(x=0.75,y=1.25,s='*',fontsize=6)
        # ax_dense.set_yticks([])
        pl_dat.remove_frame(ax_dense,['top','right'])

        res = sstats.kruskal(dense_mean[:,0,0],dense_mean[:,1,0])
        print(res)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('steadystate_paras')

        return

        p_mice = np.zeros((len(mice),120,3))*np.NaN
        p_mice_region = np.zeros((len(mice),3,2,3))*np.NaN
        ax_p1 = plt.axes([0.1,0.3,0.2,0.075])
        pl_dat.add_number(fig,ax_p1,order=5)
        ax_p2 = plt.axes([0.1,0.2,0.2,0.075])
        ax_p3 = plt.axes([0.1,0.1,0.2,0.075])
        ax_p = [ax_p1,ax_p2,ax_p3]

        # ax_p1_region = plt.axes([0.5,0.3,0.1,0.075])
        ax_p12_region = plt.axes([0.5,0.2,0.075,0.075])
        pl_dat.add_number(fig,ax_p12_region,order=6)
        ax_p13_region = plt.axes([0.5,0.1,0.075,0.075])
        ax_p22_region = plt.axes([0.65,0.2,0.075,0.075])
        ax_p23_region = plt.axes([0.65,0.1,0.075,0.075])
        ax_p32_region = plt.axes([0.8,0.2,0.075,0.075])
        ax_p33_region = plt.axes([0.8,0.1,0.075,0.075])

        ax1_region = [None,ax_p12_region,ax_p13_region]
        ax2_region = [None,ax_p22_region,ax_p23_region]
        ax3_region = [None,ax_p32_region,ax_p33_region]

        key2 = 'stable'
        status_label = ['alpha','beta','gamma']
        for m,mouse in enumerate(mice):
            if steady:
                s_range = D.sessions[mouse]['steady']
            else:
                s_range = [np.where(D.cMice[mouse].sessions['bool'])[0][0],np.where(D.cMice[mouse].sessions['bool'])[0][-1]]

            for j,key in enumerate(status_arr):

                ax_p[j].plot(gauss_smooth(D.cMice[mouse].stats['p_post_s'][key][key2][s_range[0]:s_range[1],1,0],1),'k',linestyle=':',linewidth=0.3,label='$p(\\gamma_{s+1}^+|\\%s^+)$'%status_label[j] if m==0 else None)
                p_mice[m,:s_range[1]-s_range[0],j] = D.cMice[mouse].stats['p_post_s'][key][key2][s_range[0]:s_range[1],1,0]

                if j>0:
                    p_mice_region[m,0,0,j] = np.nanmean(D.cMice[mouse].stats['p_post_RW_s'][key]['act'][s_range[0]:s_range[1],1,0])
                    p_mice_region[m,0,1,j] = np.nanmean(D.cMice[mouse].stats['p_post_nRnG_s'][key]['act'][s_range[0]:s_range[1],1,0])
                    ax1_region[j].plot([0,1],p_mice_region[m,0,:,j],'k-',linewidth=0.3)
                    # ax1_region[j].plot(1,p_mice_region[m,1,j],'k.',markersize=2)

                    p_mice_region[m,1,0,j] = np.nanmean(D.cMice[mouse].stats['p_post_RW_s'][key]['code'][s_range[0]:s_range[1],1,0])
                    p_mice_region[m,1,1,j] = np.nanmean(D.cMice[mouse].stats['p_post_nRnG_s'][key]['code'][s_range[0]:s_range[1],1,0])
                    ax2_region[j].plot([0,1],p_mice_region[m,1,:,j],'k-',linewidth=0.3)
                    # ax2_region[j].plot(1,p_mice_region[m,1,j],'k.',markersize=2)

                    p_mice_region[m,2,0,j] = np.nanmean(D.cMice[mouse].stats['p_post_RW_s'][key]['stable'][s_range[0]:s_range[1],1,0])
                    p_mice_region[m,2,1,j] = np.nanmean(D.cMice[mouse].stats['p_post_nRnG_s'][key]['stable'][s_range[0]:s_range[1],1,0])
                    ax3_region[j].plot([0,1],p_mice_region[m,2,:,j],'k-',linewidth=0.3)
                    # ax3_region[j].plot(1,p_mice_region[m,1,j],'k.',markersize=2)

        # ax_p.set_xlim([0,np.where(D.cMice[mouse].sessions['bool'])[0][-1]])
        for j in range(3):
            ax_p[j].set_ylim([0.,1])
            ax_p[j].plot(np.nanmean(p_mice[...,j],0),color='k')
            ax_p[j].set_xlabel('session')
            # ax_p[j].set_ylabel('$p(\\gamma_{s+1}^+|\cdot^+)$',fontsize=8)
            if j<2:
                ax_p[j].set_xticklabels([])
            else:
                ax_p[j].set_xlabel('Session')
            # ax_p[j].legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.3,0.25])
            ax_p[j].set_ylabel('$p(\\gamma_{1}^+|\\%s^+)$'%status_label[j],fontsize=8)
            pl_dat.remove_frame(ax_p[j],['top','right'])
            if j>0:
                ax1_region[j].errorbar([0,1],np.nanmean(p_mice_region[:,0,:,j],0),np.nanstd(p_mice_region[:,0,:,j],0),fmt='k-')
                ax2_region[j].errorbar([0,1],np.nanmean(p_mice_region[:,1,:,j],0),np.nanstd(p_mice_region[:,1,:,j],0),fmt='k-')
                ax3_region[j].errorbar([0,1],np.nanmean(p_mice_region[:,2,:,j],0),np.nanstd(p_mice_region[:,2,:,j],0),fmt='k-')
                # ax1_region[j].plot([0,1],p_mice_region[m,:,j],'k-',linewidth=0.3)

                ax1_region[j].set_xlim([-0.5,1.5])
                ax1_region[j].set_ylim([0.3,1])
                pl_dat.remove_frame(ax1_region[j],['top','right'])
                ax2_region[j].set_xlim([-0.5,1.5])
                ax2_region[j].set_ylim([0.3,1])
                pl_dat.remove_frame(ax2_region[j],['top','right'])
                ax3_region[j].set_xlim([-0.5,1.5])
                ax3_region[j].set_ylim([0.3,1])
                pl_dat.remove_frame(ax3_region[j],['top','right'])
                ax2_region[j].set_yticklabels([])
                ax3_region[j].set_yticklabels([])
            if j==1:
                ax1_region[j].set_xticklabels([])
                ax2_region[j].set_xticklabels([])
                ax3_region[j].set_xticklabels([])

                ax1_region[j].set_title('$p(\\alpha^+|\cdot)$',fontsize=8)
                ax2_region[j].set_title('$p(\\beta^+|\cdot)$',fontsize=8)
                ax3_region[j].set_title('$p(\\gamma^+|\cdot)$',fontsize=8)
            elif j==2:
                ax1_region[j].set_xticks([0,1])
                ax1_region[j].set_xticklabels(['RW','PC'],rotation=60,fontsize=8)
                ax2_region[j].set_xticks([0,1])
                ax2_region[j].set_xticklabels(['RW','PC'],rotation=60,fontsize=8)
                ax3_region[j].set_xticks([0,1])
                ax3_region[j].set_xticklabels(['RW','PC'],rotation=60,fontsize=8)
        ax_p12_region.set_ylabel('$p(\cdot|\\beta^+)$',fontsize=8)
        ax_p13_region.set_ylabel('$p(\cdot|\\gamma^+)$',fontsize=8)
        plt.show(block=False)

        for j in range(3):
            for i in range(3):
                print(p_mice_region[:,i,0,j])
                res = sstats.kruskal(p_mice_region[:,i,0,j],p_mice_region[:,i,1,j])
                print(res)
        # res = sstats.kruskal(p_mice_region[:,1,0,1],p_mice_region[:,1,0,1])
        # print(res)
        # res = sstats.kruskal(p_mice_region[:,2,0,1],p_mice_region[:,2,0,1])
        # print(res)

    if plot_fig[5]:

        SD = 1
        # s_arr += np.where(cluster.sessions['bool'])[0][0]

        status_arr = ['act','code','stable']

        nSes_max = 20
        s_arr = np.array([0,5,10,15,20])
        same_loc = [15,15,15,20]
        n_int = len(s_arr)-1

        ds_max = 11

        di = 3


        p_rec_loc = np.zeros((len(mice),n_int,nbin,ds_max))*np.NaN
        N_rec_loc = np.zeros((len(mice),n_int,nbin,ds_max))

        p_act_loc = np.zeros((len(mice),nSes_max,nbin,ds_max))*np.NaN
        N_act_loc = np.zeros((len(mice),nSes_max,nbin,ds_max))

        p_post = np.zeros((len(mice),4,n_int,3,3,2))

        RW_dense = np.zeros((len(mice),n_int))
        GT_dense = np.zeros((len(mice),n_int))
        nRnG_dense = np.zeros((len(mice),n_int))

        N_rec_all = np.zeros((len(mice),n_int))
        N_rec_RW = np.zeros((len(mice),n_int))
        N_rec_GT = np.zeros((len(mice),n_int))
        N_act_all = np.zeros((len(mice),n_int))
        N_act_RW = np.zeros((len(mice),n_int))
        N_act_GT = np.zeros((len(mice),n_int))

        fig = plt.figure(figsize=(7,7),dpi=300)

        ax = plt.axes([0.1,0.9,0.35,0.075])
        pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/Setup/VR_sketch.png'
        ax.axis('off')
        im = mpimg.imread(pic_path)
        ax.imshow(im)
        ax.set_xlim([0,im.shape[1]])

        ax = plt.axes([0.6,0.9,0.35,0.075])
        # pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/Setup/VR_sketch.png'
        ax.axis('off')
        ax.set_title('sketch of experiment',fontsize=8)
        # im = mpimg.imread(pic_path)
        # ax.imshow(im)
        # ax.set_xlim([0,im.shape[1]])

        ax_rec = plt.axes([0.9,0.65,0.075,0.075])
        ax_rec.plot([0,n_int],[0.2,0.2],color=[0.5,0.5,0.5],linestyle='--',linewidth=0.5,zorder=0)
        ax_rec.set_ylim([0,1])
        ax_rec.set_ylabel('$\\frac{\# stable region}{\# stable all}$',fontsize=8)

        ax_act = plt.axes([0.4,0.65,0.075,0.075])
        ax_act.plot([0,n_int],[0.2,0.2],color=[0.5,0.5,0.5],linestyle='--',linewidth=0.5,zorder=0)
        ax_act.set_ylim([0,1])
        ax_act.set_ylabel('$\\frac{\# react. region}{\# react. all}$',fontsize=8)

        pl_dat = plot_dat(folder='mapChanges',sv_ext='png')
        for m,mouse in enumerate(mice):

            sig_theta = D.cMice[mouse_ex].stability['all']['mean'][0,2]
            nSes = D.cMice[mouse].meta['nSes']
            nC = D.cMice[mouse].meta['nC']

            s1_shifts,s2_shifts,f1,f2 = np.unravel_index(D.cMice[mouse].compare['pointer'].col,(nSes,nSes,D.cMice[mouse].meta['field_count_max'],D.cMice[mouse].meta['field_count_max']))
            c_shifts = D.cMice[mouse].compare['pointer'].row
            for ds in range(ds_max):
                Ds = s2_shifts-s1_shifts
                idx = np.where(Ds==ds)[0]
                idx_shifts = D.cMice[mouse].compare['pointer'].data[idx].astype('int')-1
                shifts = D.cMice[mouse].compare['shifts'][idx_shifts]

                s = s1_shifts[idx]
                f = f1[idx]
                c = c_shifts[idx]
                loc_shifts = np.round(D.cMice[mouse].fields['location'][c,s,f,0]).astype('int')

                for j in range(len(s_arr)-1):
                    for i in range(nbin):
                        i_min = max(0,i-di)
                        i_max = min(nbin,i+di)
                        idx_loc = (loc_shifts>=i_min) & (loc_shifts<i_max) & ((s>=s_arr[j]) & (s<s_arr[j+1])) & (s+ds<same_loc[j])

                        shifts_loc = shifts[idx_loc]
                        N_data = len(shifts_loc)
                        N_stable = (np.abs(shifts_loc)<(SD*sig_theta)).sum()

                        p_rec_loc[m,j,i,ds] = N_stable/N_data
                        N_rec_loc[m,j,i,ds] = N_stable

            for s in np.where(D.cMice[mouse].sessions['bool'])[0][:20]:
                # s_thr = 15 if s<15 else nSes
                if s >= nSes_max:
                    continue
                s_thr = nSes_max
                for ds in range(min(min(nSes,s_thr)-s,ds_max)):
                    if D.cMice[mouse].sessions['bool'][s+ds]:
                        loc = D.cMice[mouse].fields['location'][:,s,:]

                        for i in range(nbin):
                            i_min = max(0,i-di)
                            i_max = min(nbin,i+di)
                            idx_loc = np.where((i_min<=loc) & (loc<i_max))
                            p_act_loc[m,s,i,ds] = D.cMice[mouse].status[idx_loc[0],s+ds,1].mean()
                            N_act_loc[m,s,i,ds] = D.cMice[mouse].status[idx_loc[0],s+ds,1].sum()

            suffix_arr = ['','_nRnG','_RW','_GT']
            for k in range(n_int):

                for j,key in enumerate(status_arr):
                    for i,key2 in enumerate(status_arr):
                        for l,sf in enumerate(suffix_arr):
                            p_post[m,l,k,j,i,0] = np.nanmean(D.cMice[mouse].stats['p_post%s_s'%sf][key][key2][s_arr[k]:s_arr[k+1],1,0])
                            p_post[m,l,k,j,i,1] = np.nanstd(D.cMice[mouse].stats['p_post%s_s'%sf][key][key2][s_arr[k]:s_arr[k+1],1,0])

            fields = np.zeros((nSes,nbin))
            for i,s in enumerate(np.where(D.cMice[mouse].sessions['bool'])[0]):
                idx_PC = np.where(D.cMice[mouse].status_fields[:,s,:])
                fields[s,:] = np.nansum(D.cMice[mouse].fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)

            ax = plt.axes([0.6,0.775,0.2,0.075])
            ax_dense = plt.axes([0.9,0.775,0.075,0.075])
            for j in range(n_int):

                col = [0.25*j,0.25*j,0.25*j]
                RW_pos = D.cMice[mouse].session_data['RW_pos'][s,:].astype('int')
                GT_pos = D.cMice[mouse].session_data['GT_pos'][s,:].astype('int')
                zone_nRnG = np.ones(nbin,'bool')
                zone_nRnG[RW_pos[0]:RW_pos[1]] = False
                zone_nRnG[GT_pos[0]:GT_pos[1]] = False

                fields_s = np.nansum(fields[s_arr[j]:s_arr[j+1],:],0)
                fields_s /= fields_s.sum()
                # print(fields_s)
                RW_dense[m,j] = np.nanmean(fields_s[RW_pos[0]:RW_pos[1]])
                GT_dense[m,j] = np.nanmean(fields_s[GT_pos[0]:GT_pos[1]])
                nRnG_dense[m,j] = np.nanmean(fields_s[zone_nRnG])

                ax_dense.plot(j,RW_dense[m,j],'.',color='tab:red',markersize=1)
                ax_dense.plot(j,GT_dense[m,j],'.',color='tab:green',markersize=1)
                ax_dense.plot(j,nRnG_dense[m,j],'.',color='tab:blue',markersize=1)

            fields = np.nansum(fields[:15,:],0)
            ax_dense.set_ylim([0.,0.03])
            ax.plot(fields/fields.sum(),'-',color='k',linewidth=1)
            ax.set_ylim([0.,0.03])
            ax.set_ylabel('PC density',fontsize=8)

            pl_dat.remove_frame(ax,['top','right'])

            ax = plt.axes([0.1,0.775,0.2,0.075])
            ax.plot(D.cMice[mouse].status[...,2].sum(0)/D.cMice[mouse].status[...,1].sum(0),'k.',markersize=1)
            # ax.plot(D.cMice[mouse].status[...,2].sum(0)/D.cMice[mouse].sessions['time_active'],'k.',markersize=1)
            ax.set_xlim([0,np.where(D.cMice[mouse].sessions['bool'])[0][-1]])
            ax.set_xticklabels([])
            ax.set_ylabel('% PC')
            pl_dat.remove_frame(ax,['top','right'])

            label_arr = ['nRnG','RW','GT']

            # for j,key in enumerate(status_arr):
            key = 'code'
            ax = plt.axes([0.6,0.65,0.2,0.075])
            for j in range(len(s_arr)-1):
                ax.plot([s_arr[j],s_arr[j]],[0,2],'--',color=[0.5,0.5,0.5],linewidth=0.5,zorder=0)
            ax.plot(gauss_smooth(D.cMice[mouse].stats['p_post_s'][key]['stable'][:,1,0],1),'k',linewidth=0.5)
            ax.set_ylim([0,1])
            ax.set_xlim([0,np.where(D.cMice[mouse].sessions['bool'])[0][-1]])
            # ax.set_xticklabels([])
            ax.set_ylabel('$p(\\gamma_{s+1}^+|\cdot^+)$',fontsize=8)
            pl_dat.remove_frame(ax,['top','right'])
            # ax.plot(gauss_smooth(D.cMice[mouse].stats['p_post_s']['code']['stable'][:,1,0],1),'r',linewidth=0.5)
            # ax.plot(gauss_smooth(D.cMice[mouse].stats['p_post_s']['stable']['stable'][:,1,0],1),'b',linewidth=0.5)
            ax.set_xlabel('session')

            # for j,key in enumerate(status_arr):
            ax = plt.axes([0.1,0.65,0.2,0.075])
            for j in range(n_int):
                ax.plot([s_arr[j],s_arr[j]],[0,2],'--',color=[0.5,0.5,0.5],linewidth=0.5,zorder=0)
            ax.plot(gauss_smooth(D.cMice[mouse].stats['p_post_s'][key]['act'][:,1,0],1),'k',linewidth=0.5)
            ax.set_ylim([0.5,1])
            ax.set_xlim([0,np.where(D.cMice[mouse].sessions['bool'])[0][-1]])
            # ax.set_xticklabels([])
            ax.set_ylabel('$p(\\alpha_{s+1}^+|\cdot^+)$',fontsize=8)
            pl_dat.remove_frame(ax,['top','right'])
            # ax.plot(gauss_smooth(D.cMice[mouse].stats['p_post_s']['code']['act'][:,1,0],1),'r',linewidth=0.5)
            # ax.plot(gauss_smooth(D.cMice[mouse].stats['p_post_s']['stable']['act'][:,1,0],1),'b',linewidth=0.5)
            ax.set_xlabel('session')

            status_key = ['alpha','beta','gamma']
            # col_arr = []
            idx = 1
            ax11_region = plt.axes([0.6,0.425,0.075,0.075])
            ax12_region = plt.axes([0.75,0.425,0.075,0.075])
            ax13_region = plt.axes([0.85,0.425,0.075,0.075])

            ax11_region.errorbar(np.arange(n_int),p_post[m,1,:,idx,2,0],p_post[m,1,:,idx,2,1],fmt='bo',linewidth=0.5,markersize=1)
            ax12_region.errorbar(np.arange(n_int),p_post[m,2,:,idx,2,0]-p_post[m,1,:,idx,2,0],p_post[m,1,:,idx,2,1],fmt='ro',linewidth=0.5,markersize=1)
            ax13_region.errorbar(np.arange(n_int),p_post[m,3,:,idx,2,0]-p_post[m,1,:,idx,2,0],p_post[m,1,:,idx,2,1],fmt='go',linewidth=0.5,markersize=1)

            ax21_region = plt.axes([0.1,0.425,0.075,0.075])
            ax22_region = plt.axes([0.25,0.425,0.075,0.075])
            ax23_region = plt.axes([0.35,0.425,0.075,0.075])

            ax21_region.errorbar(np.arange(n_int),p_post[m,1,:,idx,0,0],p_post[m,1,:,idx,2,1],fmt='bo',linewidth=0.5,markersize=1)
            ax22_region.errorbar(np.arange(n_int),p_post[m,2,:,idx,0,0]-p_post[m,1,:,idx,0,0],p_post[m,1,:,idx,0,1],fmt='ro',linewidth=0.5,markersize=1)
            ax23_region.errorbar(np.arange(n_int),p_post[m,3,:,idx,0,0]-p_post[m,1,:,idx,0,0],p_post[m,1,:,idx,0,1],fmt='go',linewidth=0.5,markersize=1)


            for j in range(n_int):
                if s_arr[j] >= nSes:
                    continue
                ax = plt.axes([0.1+j*0.2,0.075,0.15,0.1])
                zone_mask = {}
                zone_mask['RW'] = np.zeros(nbin).astype('bool')#range(zone_idx['reward'][0],zone_idx['reward'][-1])
                zone_mask['GT'] = np.zeros(nbin).astype('bool')
                zone_mask['nRnG'] = np.ones(nbin).astype('bool')

                RW_pos = D.cMice[mouse].session_data['RW_pos'][s_arr[j],:].astype('int')
                GT_pos = D.cMice[mouse].session_data['GT_pos'][s_arr[j],:].astype('int')
                zone_mask['RW'][RW_pos[0]:RW_pos[1]] = True
                zone_mask['nRnG'][zone_mask['RW']] = False
                if ~np.isnan(D.cMice[mouse].session_data['GT_pos'][s_arr[0],0]):
                    zone_mask['GT'][GT_pos[0]:GT_pos[1]] = True
                    zone_mask['nRnG'][zone_mask['GT']] = False

                ax.plot(gauss_smooth(np.nanmean(p_rec_loc[m,j,zone_mask['RW'],:],0),0,mode='constant'),'tab:red',linewidth=0.5)
                ax.plot(gauss_smooth(np.nanmean(p_rec_loc[m,j,zone_mask['GT'],:],0),0,mode='constant'),'tab:green',linewidth=0.5)
                ax.plot(gauss_smooth(np.nanmean(p_rec_loc[m,j,zone_mask['nRnG'],:],0),0,mode='constant'),'tab:blue',linewidth=0.5)

                ax.set_ylim([0,1])

                N_rec_all[m,j] = N_rec_loc[m,j,:,1].sum()
                N_rec_RW[m,j] = N_rec_loc[m,j,RW_pos[0]:RW_pos[1],1].sum()
                N_rec_GT[m,j] = N_rec_loc[m,j,GT_pos[0]:GT_pos[1],1].sum()

                ax_rec.plot(j,N_rec_RW[m,j]/N_rec_all[m,j],'o',color='tab:red',markersize=1)
                ax_rec.plot(j,N_rec_GT[m,j]/N_rec_all[m,j],'o',color='tab:green',markersize=1)
                ax_rec.plot(j,(N_rec_all[m,j]-N_rec_RW[m,j]-N_rec_GT[m,j])/N_rec_all[m,j],'o',color='tab:blue',markersize=1)

                N_act_all[m,j] = N_act_loc[m,s_arr[j]:s_arr[j+1],:,1].sum(axis=(0,1))
                N_act_RW[m,j] = N_act_loc[m,s_arr[j]:s_arr[j+1],RW_pos[0]:RW_pos[1],1].sum(axis=(0,1))
                N_act_GT[m,j] = N_act_loc[m,s_arr[j]:s_arr[j+1],GT_pos[0]:GT_pos[1],1].sum(axis=(0,1))

                ax_act.plot(j,N_act_RW[m,j]/N_act_all[m,j],'o',color='tab:red',markersize=1)
                ax_act.plot(j,N_act_GT[m,j]/N_act_all[m,j],'o',color='tab:green',markersize=1)
                ax_act.plot(j,(N_act_all[m,j]-N_act_RW[m,j]-N_act_GT[m,j])/N_act_all[m,j],'o',color='tab:blue',markersize=1)


        for k,axx in enumerate([[ax11_region,ax12_region,ax13_region],[ax21_region,ax22_region,ax23_region]]):
            for l,axxx in enumerate(axx):
                axxx.set_xticks(np.arange(n_int))
                axxx.set_xticklabels(['s%d-%d'%(s_arr[j]+1,s_arr[j+1]) for j in range(n_int)],rotation=60,fontsize=8)
                axxx.set_title(label_arr[l])
                axxx.set_yticklabels([])
                pl_dat.remove_frame(axxx,['top','right'])

        for axx in [ax12_region,ax13_region,ax22_region,ax23_region]:
            axx.plot([-0.5,n_int-0.5],[0,0],'--',color=[0.5,0.5,0.5],linewidth=0.5,zorder=0)
            axx.set_ylim([-0.5,0.5])

        ax11_region.set_ylim([0,1])
        ax11_region.set_ylabel('$p(\\gamma_{s+1}^+|\\beta_s^+)$')
        ax12_region.set_ylabel('$\Delta p$')
        ax21_region.set_ylim([0,1])
        ax21_region.set_ylabel('$p(\\alpha_{s+1}^+|\\beta_s^+)$')
        ax22_region.set_ylabel('$\Delta p$')

        idx = 1
        ax11_region.errorbar(np.arange(n_int),np.nanmean(p_post[:,1,:,idx,2,0],0),np.nanstd(p_post[:,1,:,idx,2,0],0),fmt='k-',linewidth=1,markersize=1)
        ax12_region.errorbar(np.arange(n_int),np.nanmean(p_post[:,2,:,idx,2,0]-p_post[:,1,:,idx,2,0],0),np.nanstd(p_post[:,2,:,idx,2,0]-p_post[:,1,:,idx,2,0],0),fmt='k-',linewidth=1,markersize=1)
        ax13_region.errorbar(np.arange(n_int),np.nanmean(p_post[:,3,:,idx,2,0]-p_post[:,1,:,idx,2,0],0),np.nanstd(p_post[:,3,:,idx,2,0]-p_post[:,1,:,idx,2,0],0),fmt='k-',linewidth=1,markersize=1)

        ax21_region.errorbar(np.arange(n_int),np.nanmean(p_post[:,1,:,idx,0,0],0),np.nanstd(p_post[:,1,:,idx,0,0],0),fmt='k-',linewidth=1,markersize=1)
        ax22_region.errorbar(np.arange(n_int),np.nanmean(p_post[:,2,:,idx,0,0]-p_post[:,1,:,idx,0,0],0),np.nanstd(p_post[:,2,:,idx,2,0]-p_post[:,1,:,idx,2,0],0),fmt='k-',linewidth=1,markersize=1)
        ax23_region.errorbar(np.arange(n_int),np.nanmean(p_post[:,3,:,idx,0,0]-p_post[:,1,:,idx,0,0],0),np.nanstd(p_post[:,3,:,idx,2,0]-p_post[:,1,:,idx,2,0],0),fmt='k-',linewidth=1,markersize=1)

        props = dict(boxstyle='round', facecolor='w', alpha=0.8)

        m = 1
        for j in range(n_int):
            ax_im = plt.axes([0.1+0.2*j,0.2,0.15,0.075])
            im = ax_im.imshow(gauss_smooth(p_rec_loc[m,j,...],(1,0)),clim=[0.25,0.75],interpolation='None',origin='lower',aspect='auto')
            ax_im.set_xlim([0.5,10.5])
            ax_im.set_xticklabels([1,5,10])
            ax_im.text(x=0.5,y=110,s='Sessions %d-%d'%(s_arr[j]+1,s_arr[j+1]),ha='left',va='bottom',bbox=props,fontsize=8)
            ax_im.set_ylim([0,100])
            ax_im.set_xticks([1,5,10])
            ax_im.set_xticklabels([])
            if j == 0:
                ax_im.set_ylabel('pos.')
            else:
                ax_im.set_yticklabels([])
            if j == (n_int-1):
                cb = plt.colorbar(im)
                cb.set_label('$p(\\gamma_{\Delta s}^+|\\beta^+)$',fontsize=8)

            ax_dense.plot(np.arange(n_int),np.nanmean(RW_dense,0),'-',color='tab:red')
            ax_dense.plot(np.arange(n_int),np.nanmean(GT_dense,0),'-',color='tab:green')
            ax_dense.plot(np.arange(n_int),np.nanmean(nRnG_dense,0),'-',color='tab:blue')
            pl_dat.remove_frame(ax_dense,['top','right'])
            # p_rec_loc[:,j,zone_mask['RW'],:].shape
            ax = plt.axes([0.1+j*0.2,0.075,0.15,0.1])
            ax.plot(gauss_smooth(np.nanmean(p_rec_loc[:,j,zone_mask['RW'],:],axis=(0,1)),0,mode='constant'),'r',linewidth=1.5)
            ax.plot(gauss_smooth(np.nanmean(p_rec_loc[:,j,zone_mask['GT'],:],axis=(0,1)),0,mode='constant'),'g',linewidth=1.5)
            ax.plot(gauss_smooth(np.nanmean(p_rec_loc[:,j,zone_mask['nRnG'],:],axis=(0,1)),0,mode='constant'),'b',linewidth=1.5)
            pl_dat.remove_frame(ax,['top','right'])
            ax.set_xticks([1,5,10])
            ax.set_xlabel('$\Delta s$')
            if j == 0:
                ax.set_ylabel('$p(\\gamma_{\Delta s}^+|\\beta^+)$')

        ax_dense.set_xticks(np.arange(n_int))
        ax_dense.set_xticklabels([])
        ax_dense.set_ylabel('$\\left \\langle density \\right \\rangle$',fontsize=8)


        pl_dat.remove_frame(ax_rec,['top','right'])
        pl_dat.remove_frame(ax_act,['top','right'])

        ax_act.plot(np.arange(n_int),np.nanmean(N_act_RW/N_act_all,0),'r-',linewidth=1)
        ax_act.plot(np.arange(n_int),np.nanmean(N_act_GT/N_act_all,0),'g-',linewidth=1)
        ax_act.plot(np.arange(n_int),np.nanmean((N_act_all-N_act_RW-N_act_GT)/N_act_all,0),'b-',linewidth=1)

        ax_rec.plot(np.arange(n_int),np.nanmean(N_rec_RW/N_rec_all,0),'r-',linewidth=1)
        ax_rec.plot(np.arange(n_int),np.nanmean(N_rec_GT/N_rec_all,0),'g-',linewidth=1)
        ax_rec.plot(np.arange(n_int),np.nanmean((N_rec_all-N_rec_RW-N_rec_GT)/N_rec_all,0),'b-',linewidth=1)

        ax_rec.set_xticks(np.arange(n_int))
        ax_rec.set_xticklabels(['s%d-%d'%(s_arr[j]+1,s_arr[j+1]) for j in range(n_int)],rotation=60,fontsize=8)
        ax_act.set_xticks(np.arange(n_int))
        ax_act.set_xticklabels(['s%d-%d'%(s_arr[j]+1,s_arr[j+1]) for j in range(n_int)],rotation=60,fontsize=8)

        # ax.set_xlabel('session')
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('timedep_dynamics')

        plt.figure()
        # plt.subplot(221)



            # plt.plot(np.nanmean(D.cMice[mouse_ex].stats['p_post_RW_s']['code']['stable'][s_arr[j]:s_arr[j+1],:,0],0),'r')
            # plt.plot(np.nanmean(D.cMice[mouse_ex].stats['p_post_GT_s']['code']['stable'][s_arr[j]:s_arr[j+1],:,0],0),'g')
            # plt.plot(np.nanmean(D.cMice[mouse_ex].stats['p_post_nRnG_s']['code']['stable'][s_arr[j]:s_arr[j+1],:,0],0),'b')
        plt.show(block=False)


    if plot_fig[6]:

        print('changes between coding state')
        SD = 1

        rel = np.zeros((len(mice),3,2))
        fr = np.zeros((len(mice),3,2))
        fr1 = np.zeros((len(mice),3,2))
        MI = np.zeros((len(mice),3,2))
        sig = np.zeros((len(mice),3,2))
        pl_dat = plot_dat(folder='popDyn',sv_ext='png')
        ds_max = 6
        p_rec = {'all':       np.zeros((len(mice),ds_max,2))*np.NaN,
                  'cont':       np.zeros((len(mice),ds_max,2))*np.NaN,
                  'mix':        np.zeros((len(mice),ds_max,2))*np.NaN,
                  'discont':    np.zeros((len(mice),ds_max,2))*np.NaN,
                  'silent_mix': np.zeros((len(mice),ds_max,2))*np.NaN,
                  'silent':     np.zeros((len(mice),ds_max,2))*np.NaN}
        nsteps=51
        stab_loc = np.zeros((len(mice),nsteps,2))
        act_loc = np.zeros((len(mice),nsteps,3))

        fig = plt.figure(figsize=(7,3.5),dpi=300)
        ax_rel = plt.axes([0.1,0.15,0.075,0.2])
        ax_fr = plt.axes([0.275,0.15,0.075,0.2])
        ax_fr1 = plt.axes([0.45,0.15,0.075,0.2])
        ax_MI = plt.axes([0.625,0.15,0.075,0.2])
        ax_sig = plt.axes([0.8,0.15,0.075,0.2])

        rel_tmp = {'stable':    [],
                    'relocate':    [],
                    'instable': []}
        fr_tmp = {'stable':    [],
                    'relocate':    [],
                    'instable': []}
        fr1_tmp = {'stable':    [],
                    'relocate':    [],
                    'instable': []}
        MI_tmp = {'stable':    [],
                    'relocate':    [],
                    'instable': []}
        sig_tmp = {'stable':    [],
                    'relocate':    [],
                    'instable': []}

        ax_stable = plt.axes([0.5,0.625,0.325,0.17])
        ax_dstable = plt.axes([0.5,0.8,0.325,0.125])
        ax_recode = plt.axes([0.1,0.625,0.25,0.27])

        pl_dat.add_number(fig,ax_rel,order=3)
        pl_dat.add_number(fig,ax_recode,order=1)
        pl_dat.add_number(fig,ax_dstable,order=2,offset=[-150,25])

        for m,mouse in enumerate(mice):
            print(mouse)
            nSes = D.cMice[mouse].meta['nSes']
            s_bool = np.zeros(nSes,'bool')
            # s_bool[17:87] = True
            s_bool[5:] = True

            sig_theta = D.cMice[mouse].stability['all']['mean'][0,2]

            s1,s2,f1,f2 = np.unravel_index(D.cMice[mouse].compare['pointer'].col,(D.cMice[mouse].meta['nSes'],D.cMice[mouse].meta['nSes'],D.cMice[mouse].meta['field_count_max'],D.cMice[mouse].meta['field_count_max']))
            idx_ds1 = np.where((s2-s1 == 1) & s_bool[s1] & s_bool[s2])[0]

            c_ds1 = D.cMice[mouse].compare['pointer'].row[idx_ds1]
            s1_ds1 = s1[idx_ds1]
            f1_ds1 = f1[idx_ds1]
            idx_shifts_ds1 = D.cMice[mouse].compare['pointer'].data[idx_ds1].astype('int')-1
            shifts_ds1 = D.cMice[mouse].compare['shifts'][idx_shifts_ds1]

            idx_stable_ds1 = np.where(np.abs(shifts_ds1) < sig_theta)[0]
            idx_relocate_ds1 = np.where(np.abs(shifts_ds1) > 2*sig_theta)[0]
            idx_loosePC = np.where(np.diff(D.cMice[mouse].status[:,s_bool,2].astype('int'),1)==-1)

            c_stable = c_ds1[idx_stable_ds1]
            s1_stable = s1_ds1[idx_stable_ds1]
            f_stable = f1_ds1[idx_stable_ds1]

            c_relocate = c_ds1[idx_relocate_ds1]
            s1_relocate = s1_ds1[idx_relocate_ds1]
            f_relocate = f1_ds1[idx_relocate_ds1]

            idx_nPC = np.where(D.cMice[mouse].status[:,s_bool,1] & ~D.cMice[mouse].status[:,s_bool,2])

            rel_dat = {}
            rel_dat['stable'] = D.cMice[mouse].fields['reliability'][c_stable,s1_stable,f_stable]
            rel_dat['relocate'] = D.cMice[mouse].fields['reliability'][c_relocate,s1_relocate,f_relocate]
            rel_dat['instable'] = np.nanmax(D.cMice[mouse].fields['reliability'][idx_loosePC[0],idx_loosePC[1],:],-1)

            sig_dat = {}
            sig_dat['stable'] = D.cMice[mouse].fields['width'][c_stable,s1_stable,f_stable]
            sig_dat['relocate'] = D.cMice[mouse].fields['width'][c_relocate,s1_relocate,f_relocate]
            sig_dat['instable'] = np.nanmax(D.cMice[mouse].fields['width'][idx_loosePC[0],idx_loosePC[1],:],-1)

            fr_dat = {}
            fr_dat['stable'] = D.cMice[mouse].stats['oof_firingrate_adapt'][c_stable,s1_stable]
            fr_dat['relocate'] = D.cMice[mouse].stats['oof_firingrate_adapt'][c_relocate,s1_relocate]
            fr_dat['instable'] = D.cMice[mouse].stats['oof_firingrate_adapt'][idx_loosePC[0],idx_loosePC[1]]

            fr1_dat = {}
            fr1_dat['stable'] = D.cMice[mouse].stats['if_firingrate_adapt'][c_stable,s1_stable,f_stable]
            fr1_dat['relocate'] = D.cMice[mouse].stats['if_firingrate_adapt'][c_relocate,s1_relocate,f_relocate]
            fr1_dat['instable'] = np.nanmax(D.cMice[mouse].stats['if_firingrate_adapt'][idx_loosePC[0],idx_loosePC[1],:],-1)

            MI_dat = {}
            MI_dat['stable'] = D.cMice[mouse].stats['MI_value'][c_stable,s1_stable]
            MI_dat['relocate'] = D.cMice[mouse].stats['MI_value'][c_relocate,s1_relocate]
            MI_dat['instable'] = D.cMice[mouse].stats['MI_value'][idx_loosePC[0],idx_loosePC[1]]

            for j,key in enumerate(['stable','relocate','instable']):
                rel[m,j,0] = np.nanmean(rel_dat[key])
                rel[m,j,1] = np.nanstd(rel_dat[key])

                sig[m,j,0] = np.nanmean(sig_dat[key])
                sig[m,j,1] = np.nanstd(sig_dat[key])

                fr[m,j,0] = np.nanmean(fr_dat[key])
                fr[m,j,1] = np.nanstd(fr_dat[key])

                fr1[m,j,0] = np.nanmean(fr1_dat[key])
                fr1[m,j,1] = np.nanstd(fr1_dat[key])

                MI[m,j,0] = np.nanmean(MI_dat[key])
                MI[m,j,1] = np.nanstd(MI_dat[key])

                rel_tmp[key].extend(rel_dat[key])
                fr_tmp[key].extend(fr_dat[key])
                fr1_tmp[key].extend(fr1_dat[key])
                MI_tmp[key].extend(MI_dat[key])
                sig_tmp[key].extend(sig_dat[key])

            ax_rel.errorbar(np.arange(3)-0.05+np.random.rand()*0.1,rel[m,:,0],rel[m,:,1],fmt='-',color=[0.6,0.6,0.6],mec=[0.6,0.6,0.6],ecolor=[0.6,0.6,0.6],markersize=2,linewidth=0.3)
            ax_fr.errorbar(np.arange(3)-0.05+np.random.rand()*0.1,fr[m,:,0],fr[m,:,1],fmt='-',color=[0.6,0.6,0.6],mec=[0.6,0.6,0.6],ecolor=[0.6,0.6,0.6],markersize=2,linewidth=0.3)
            ax_fr1.errorbar(np.arange(3)-0.05+np.random.rand()*0.1,fr1[m,:,0],fr1[m,:,1],fmt='-',color=[0.6,0.6,0.6],mec=[0.6,0.6,0.6],ecolor=[0.6,0.6,0.6],markersize=2,linewidth=0.3)
            ax_MI.errorbar(np.arange(3)-0.05+np.random.rand()*0.1,MI[m,:,0],MI[m,:,1],fmt='-',color=[0.6,0.6,0.6],mec=[0.6,0.6,0.6],ecolor=[0.6,0.6,0.6],markersize=2,linewidth=0.3)
            ax_sig.errorbar(np.arange(3)-0.05+np.random.rand()*0.1,sig[m,:,0],sig[m,:,1],fmt='-',color=[0.6,0.6,0.6],mec=[0.6,0.6,0.6],ecolor=[0.6,0.6,0.6],markersize=2,linewidth=0.3)

            # status,status_dep = get_status_arr(D.cMice[mouse])

            key_arr = ['cont','discont','silent']#'mix','silent_mix','all',
            s1_shifts,s2_shifts,f1,f2 = np.unravel_index(D.cMice[mouse].compare['pointer'].col,(nSes,nSes,5,5))
            Ds = s2_shifts-s1_shifts

            idx_ds = np.where(s_bool[s1_shifts] & s_bool[s2_shifts])[0]
            idx_shifts = D.cMice[mouse].compare['pointer'].data[idx_ds].astype('int')-1
            shifts = D.cMice[mouse].compare['shifts'][idx_shifts]

            s1_shifts = s1_shifts[idx_ds]
            s2_shifts = s2_shifts[idx_ds]

            ### coding -> field stability+
            p_rec_bs = {'cont':       np.zeros((N_bs,ds_max))*np.NaN,
                     'discont':    np.zeros((N_bs,ds_max))*np.NaN,
                     'silent':     np.zeros((N_bs,ds_max))*np.NaN}

            N_data = len(s1_shifts)
            samples = np.random.randint(0,N_data,(N_bs,N_data))
            # print(samples)
            for L in range(N_bs):
                s1_bs = s1_shifts[samples[L,:]]
                s2_bs = s2_shifts[samples[L,:]]
                shifts_bs = shifts[samples[L,:]]

                Ds_bs = s2_bs-s1_bs
                # print(Ds_bs)
                for ds in range(1,ds_max):

                    idx_shifts = np.where(Ds_bs==ds)[0]
                    shifts_bs_ds = shifts_bs[idx_shifts]
                    if len(shifts_bs_ds)>10:

                        for pop in key_arr:

                            # if pop=='all':
                                # idxes = np.ones(len(idx_ds),'bool')
                            if pop=='cont':
                                # idxes = D.cMice[mouse].compare['inter_coding'][idx_ds,1]==1
                                idxes = D.cMice[mouse].compare['inter_coding'][samples[L,idx_shifts],1]==1
                            # elif pop=='mix':
                                # idxes = ((D.cMice[mouse].compare['inter_coding'][idx_ds,1]>0) & (D.cMice[mouse].compare['inter_coding'][idx_ds,1]<1)) & (D.cMice[mouse].compare['inter_active'][idx_ds,1]==1)
                            elif pop=='discont':
                                # idxes = (D.cMice[mouse].compare['inter_coding'][idx_ds,1]==0) & (D.cMice[mouse].compare['inter_active'][idx_ds,1]==1)
                                idxes = (D.cMice[mouse].compare['inter_coding'][samples[L,idx_shifts],1]==0) & (D.cMice[mouse].compare['inter_active'][samples[L,idx_shifts],1]==1)
                            # elif pop=='silent_mix':
                                # idxes =(cluster.compare['inter_active'][idx_ds,1]>0) & (cluster.compare['inter_active'][idx_ds,1]<1)
                            elif pop=='silent':
                                # idxes = D.cMice[mouse].compare['inter_active'][idx_ds,1]==0
                                idxes = D.cMice[mouse].compare['inter_active'][samples[L,idx_shifts],1]==0

                            N_data = idxes.sum()
                            N_stable = (np.abs(shifts_bs_ds[idxes])<(sig_theta)).sum()

                            p_rec_bs[pop][L,ds] = N_stable/N_data

            for pop in key_arr:
                for ds in range(1,ds_max):
                    p_rec[pop][m,ds,0] = np.nanmean(p_rec_bs[pop][:,ds])
                    p_rec[pop][m,ds,1] = np.nanstd(p_rec_bs[pop][:,ds])



            N_data = len(idx_ds)
            s_range = 2+10

            s1_shifts,s2_shifts,f1,f2 = np.unravel_index(D.cMice[mouse].compare['pointer'].col,(nSes,nSes,5,5))
            c_shifts = D.cMice[mouse].compare['pointer'].row
            Ds = s2_shifts-s1_shifts

            idx_ds = np.where(s_bool[s1_shifts] & s_bool[s2_shifts])[0]

            s1_shifts_ds = s1_shifts[idx_ds]
            s2_shifts_ds = s2_shifts[idx_ds]
            c_shifts_ds = c_shifts[idx_ds]
            f1_ds = f1[idx_ds]
            idx_shifts = D.cMice[mouse].compare['pointer'].data[idx_ds].astype('int')-1
            shifts_ds = D.cMice[mouse].compare['shifts'][idx_shifts]
            dist = np.linspace(0,4*sig_theta,nsteps)
            for i in tqdm(range(nsteps-1)):
                ## idxes of ds=1 shifts with certain distance
                idx = ((np.abs(shifts_ds)>=dist[i]) & (np.abs(shifts_ds) < dist[i+1]))
                for j in np.where(idx)[0]:
                    s = s1_shifts_ds[j]
                    c = c_shifts_ds[j]
                    idx_c_shifts = (c_shifts==c) & (s1_shifts==s) & ((s+1)<s2_shifts) & (s2_shifts<(s+s_range))

                    idx_shifts = D.cMice[mouse].compare['pointer'].data[idx_c_shifts].astype('int')-1
                    c_shifts_now = D.cMice[mouse].compare['shifts'][idx_shifts]

                    stab_loc[m,i,0] += (np.abs(c_shifts_now) < sig_theta).sum()
                    stab_loc[m,i,1] += len(c_shifts_now)

                    # act_loc[m,i,0] += D.cMice[mouse].status[c,s+2:s+s_range,1].sum()
                    # act_loc[m,i,1] += D.cMice[mouse].status[c,s+2:s+s_range,2].sum()
                    # act_loc[m,i,2] += s_bool[s+2:s+s_range].sum()


            # ax_recode.plot(np.linspace(0,4,nsteps),gauss_smooth(stab_loc[m,:,0]/stab_loc[m,:,1],0,mode='constant'),linewidth=0.3,color=[0.6,0.6,0.6])
            ax_recode.plot(np.linspace(0,4,nsteps),stab_loc[m,:,0]/stab_loc[m,:,1],linewidth=0.3,color=[0.6,0.6,0.6])

            # ax_recode.plot(np.linspace(0,4,nsteps),gauss_smooth(act_loc[m,:,0]/act_loc[m,:,2],0,mode='constant'),linewidth=0.3,color='k')
            # ax_recode.plot(np.linspace(0,4,nsteps),gauss_smooth(act_loc[m,:,1]/act_loc[m,:,2],0,mode='constant'),linewidth=0.3,color='b')

            col_arr = [[0.5,0.5,1],[1,0.5,0.5],[0.5,1,0.5]]#[0.5,0.5,0.5],
            label_arr = ['continuous','non-coding','silent']#,'mixed'
            # key_arr = ['cont','mix','discont','silent']
            key_arr = ['cont','discont','silent']#'mix',

            w_bar = 0.25
            nKey = len(key_arr)
            offset_bar = ((nKey+1)%2)*w_bar/2 + ((nKey-1)//2)*w_bar

            for i,key in enumerate(key_arr):
                ax_stable.errorbar(np.arange(2,ds_max)-offset_bar+i*w_bar-w_bar/4+w_bar/2*np.random.rand(ds_max-2),p_rec[key][m,2:ds_max,0],p_rec[key][m,2:ds_max,1],fmt='.',color='k',markersize=1,linewidth=0.3)#col_arr[i]

            # ax_dstable.plot(np.arange(2,ds_max)-w_bar/4+w_bar/2*np.random.rand(ds_max-2),p_rec['silent'][m,2:ds_max,0]-p_rec['discont'][m,2:ds_max,0],'k.',markersize=1)
            ax_dstable.errorbar(np.arange(2,ds_max)-w_bar/4+w_bar/2*np.random.rand(ds_max-2),p_rec['silent'][m,2:ds_max,0]-p_rec['discont'][m,2:ds_max,0],p_rec['silent'][m,2:ds_max,1]+p_rec['discont'][m,2:ds_max,1],fmt='.',color='k',markersize=1,linewidth=0.3)

        for i,key in enumerate(key_arr):
            ax_stable.bar(np.arange(2,ds_max)-offset_bar+i*w_bar,np.nanmean(p_rec[key][:,2:ds_max,0],0),width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i],zorder=0)
        # print(p_rec)
        ax_dstable.bar(np.arange(2,ds_max),np.nanmean((p_rec['silent']-p_rec['discont'])[:,2:ds_max,0],0),width=w_bar,facecolor=[0.6,0.6,0.6],edgecolor='k',zorder=0,label='silent - non-coding')
        ax_dstable.errorbar(np.arange(2,ds_max),np.nanmean((p_rec['silent']-p_rec['discont'])[:,2:ds_max,0],0),np.nanstd((p_rec['silent']-p_rec['discont'])[:,2:ds_max,0],0),ls='none',ecolor='r',linewidth=1)

        ax_stable.legend(loc='upper left',bbox_to_anchor=[0.95,1.1],fontsize=8)
        ax_dstable.set_ylabel('$\Delta p$')
        for j in range(3):
            ax_rel.errorbar(j,rel[:,j,0].mean(),rel[:,j,0].std(),fmt='k.',markersize=4,linewidth=0.5)
            ax_fr.errorbar(j,fr[:,j,0].mean(),fr[:,j,0].std(),fmt='k.',markersize=4,linewidth=0.5)
            ax_fr1.errorbar(j,fr1[:,j,0].mean(),fr1[:,j,0].std(),fmt='k.',markersize=4,linewidth=0.5)
            ax_MI.errorbar(j,MI[:,j,0].mean(),MI[:,j,0].std(),fmt='k.',markersize=4,linewidth=0.5)
            ax_sig.errorbar(j,sig[:,j,0].mean(),sig[:,j,0].std(),fmt='k.',markersize=4,linewidth=0.5)

        for axx in [ax_rel,ax_fr,ax_fr1,ax_MI,ax_sig]:
            axx.set_xticks(range(3))
            axx.set_xticklabels(['stable','reloc.','vanish'],rotation=60)
            axx.set_xlim([-0.5,2.5])
            pl_dat.remove_frame(axx,['top','right'])

        res = sstats.kruskal(rel[:,0,0],rel[:,1,0])
        ax_rel.text(x=0,y=0.7,s='*'*int(-np.log10(res.pvalue)),fontsize=6,ha='center')
        ax_rel.set_ylabel('a')
        ax_rel.set_ylim([0.,0.8])
        ax_fr.set_ylabel('$\\nu^-$')
        ax_fr.set_ylim([0.,0.3])
        ax_fr1.set_ylabel('$\\nu^*$')
        ax_fr1.set_ylim([0.,2])
        ax_MI.set_ylabel('MI')
        ax_MI.set_ylim([0.,0.4])
        ax_sig.set_ylabel('$\\sigma$')
        ax_sig.set_ylim([0.,12])
        res = sstats.kruskal(sig[:,0,0],sig[:,2,0])
        ax_sig.text(x=2,y=10,s='*'*int(-np.log10(res.pvalue)),fontsize=6,ha='center')

        sig_theta = D.cMice[mice[0]].stability['all']['mean'][0,2]
        p_rec_chance = (2*SD*sig_theta) / 100
        ax_stable.plot([0.5,ds_max+0.5],[p_rec_chance,p_rec_chance],'--',color=[0.5,0.5,0.5],zorder=0)
        ax_stable.set_xlabel('session difference $\Delta s$')
        ax_stable.set_ylabel('$p(\\gamma_{\Delta s}^+)$')
        ax_stable.set_xlim([1.5,ds_max-0.5])
        ax_stable.set_ylim([0,1])
        ax_dstable.plot([0,ds_max],[0,0],linewidth=0.5,color=[0.6,0.6,0.6],linestyle='--')
        ax_dstable.set_ylim([-0.1,0.2])
        ax_dstable.set_xlim([1.5,ds_max-0.5])
        ax_dstable.set_xticklabels([])
        ax_dstable.legend(fontsize=8,loc='lower left',bbox_to_anchor=[0.2,0.9])

        ax_recode.plot([0,4],[p_rec_chance,p_rec_chance],'--',color=[0.5,0.5,0.5],zorder=0)
        stab_overall = stab_loc[:,:,0]/stab_loc[:,:,1]
        pl_dat.plot_with_confidence(ax_recode,np.linspace(0,4,nsteps),np.nanmean(stab_overall,0),np.nanstd(stab_overall,0),col='k')
        ax_recode.set_xlabel('$\Delta \\theta (\Delta s = 1) / \\sigma_{\\theta}$')
        ax_recode.set_ylabel('$p(\\gamma_{\Delta s>1}^+|\\beta^+)$')
        ax_recode.set_ylim([0,1])
        # ax_recode.set_ylabel('$p(\\gamma_{\Delta s>1}^+|\\beta^+)$')
        # ax.legend(loc='upper right',bbox_to_anchor=[0.7,1.25],fontsize=8,ncol=2)
        pl_dat.remove_frame(ax_stable,['top','right'])
        pl_dat.remove_frame(ax_dstable,['top','right'])
        pl_dat.remove_frame(ax_recode,['top','right'])
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('fieldshift_impact')



    if plot_fig[7]:

        pl_dat = plot_dat(folder='mapChanges',sv_ext=sv_ext)

        SD = 1#1.96
        fig = plt.figure(figsize=(7,5),dpi=300)

        ax_sketch = plt.axes([0.075,0.85,0.15,0.1])
        pl_dat.add_number(fig,ax_sketch,order=1,offset=[-50,25])
        ax_sketch.axis('off')
        print('sketch of place fields remapping / recruiting from other state')

        ax_transitions_silent = plt.axes([0.39,0.85,0.225,0.1])
        pl_dat.add_number(fig,ax_transitions_silent,order=2,offset=[-150,25])
        ax_transitions_active = plt.axes([0.39,0.725,0.225,0.1])
        ax_transitions_coding = plt.axes([0.39,0.6,0.225,0.1])
        ax_transitions_stable = plt.axes([0.39,0.475,0.225,0.1])
        ax_transitions = [ax_transitions_silent,ax_transitions_active,ax_transitions_coding]
        for axx in ax_transitions:
            axx.plot([0,nbin],[0,0],'--',color=[0.5,0.5,0.5],linewidth=0.5)

        ax_transitions_silent1 = plt.axes([0.675,0.85,0.065,0.1])
        pl_dat.add_number(fig,ax_transitions_silent1,order=3,offset=[-50,25])
        ax_transitions_silent2 = plt.axes([0.775,0.85,0.065,0.1])
        ax_transitions_silent3 = plt.axes([0.9,0.85,0.065,0.1])
        ax_trans = {}
        ax_trans['silent'] = [ax_transitions_silent1,ax_transitions_silent2,ax_transitions_silent3]

        ax_transitions_active1 = plt.axes([0.675,0.725,0.065,0.1])
        ax_transitions_active2 = plt.axes([0.775,0.725,0.065,0.1])
        ax_transitions_active3 = plt.axes([0.9,0.725,0.065,0.1])
        ax_trans['active'] = [ax_transitions_active1,ax_transitions_active2,ax_transitions_active3]

        ax_transitions_coding1 = plt.axes([0.675,0.6,0.065,0.1])
        ax_transitions_coding2 = plt.axes([0.775,0.6,0.065,0.1])
        ax_transitions_coding3 = plt.axes([0.9,0.6,0.065,0.1])
        ax_trans['coding'] = [ax_transitions_coding1,ax_transitions_coding2,ax_transitions_coding3]

        ax_trans_stable = plt.axes([0.675,0.475,0.05,0.1])


        ax_d_shift = plt.axes([0.35,0.125,0.175,0.1])
        ax_d_shift_SD = plt.axes([0.35,0.25,0.175,0.075])
        pl_dat.add_number(fig,ax_d_shift_SD,order=6)
        d_remap_mean = np.zeros((nMice,nbin,2))*np.NaN
        F_shifts = lambda x,A0,A,sig,theta : A/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-theta)**2/(2*sig**2)) + A0/len(x)     ## gaussian + linear offset


        status_key = ['silent','active','coding']
        trans_mean = {}
        for key in status_key:
            trans_mean[key] = np.zeros((nMice,3,3,2))*np.NaN    ### mouse, transition-type (recruit/dismiss/diff), field-loc, mean/std
        stab_mean = np.zeros((nMice,3,2))*np.NaN
        smooth=1
        ls_arr = ['-','--',':']
        for m,mouse in enumerate(mice):

            nSes = D.cMice[mouse].meta['nSes']
            D.cMice[mouse].session_data = get_session_specifics(mouse,nSes)

            s_bool = np.zeros(nSes,'bool')
            s_bool[D.sessions[mouse]['steady'][0]:D.sessions[mouse]['steady'][1]] = True
            s_bool[~D.cMice[mouse].sessions['bool']] = False

            RW_pos = D.cMice[mouse].session_data['RW_pos'][D.sessions[mouse]['steady'][0],:].astype('int')
            GT_pos = D.cMice[mouse].session_data['GT_pos'][D.sessions[mouse]['steady'][0],:].astype('int')
            zone_nRnG = np.ones(nbin,'bool')
            zone_nRnG[RW_pos[0]:RW_pos[1]] = False
            zone_nRnG[GT_pos[0]:GT_pos[1]] = False
            zone_nRnG[:10] = False
            zone_nRnG[-10:] = False
            zone_RW = np.zeros(nbin,'bool')
            zone_RW[RW_pos[0]:RW_pos[1]] = True
            zone_edge = np.zeros(nbin,'bool')
            zone_edge[:10] = True
            zone_edge[-10:] = True


            D.cMice[mouse].get_locTransition_prob()

            for j in range(3):
                key = status_key[j]
                trans_mean[key][m,0,0,0] = np.nanmean(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j][:,zone_nRnG])
                trans_mean[key][m,0,0,1] = np.nanstd(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j][:,zone_nRnG])
                trans_mean[key][m,0,1,0] = np.nanmean(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j][:,zone_RW])
                trans_mean[key][m,0,1,1] = np.nanstd(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j][:,zone_RW])
                trans_mean[key][m,0,2,0] = np.nanmean(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j][:,zone_edge])
                trans_mean[key][m,0,2,1] = np.nanstd(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j][:,zone_edge])

                trans_mean[key][m,1,0,0] = np.nanmean(D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j][:,zone_nRnG])
                trans_mean[key][m,1,0,1] = np.nanstd(D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j][:,zone_nRnG])
                trans_mean[key][m,1,1,0] = np.nanmean(D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j][:,zone_RW])
                trans_mean[key][m,1,1,1] = np.nanstd(D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j][:,zone_RW])
                trans_mean[key][m,1,2,0] = np.nanmean(D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j][:,zone_edge])
                trans_mean[key][m,1,2,1] = np.nanstd(D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j][:,zone_edge])

                diff = D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j]-D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j]
                trans_mean[key][m,2,0,0] = np.nanmean(diff[:,zone_nRnG])
                trans_mean[key][m,2,0,1] = np.nanstd(diff[:,zone_nRnG])
                trans_mean[key][m,2,1,0] = np.nanmean(diff[:,zone_RW])
                trans_mean[key][m,2,1,1] = np.nanstd(diff[:,zone_RW])
                trans_mean[key][m,2,2,0] = np.nanmean(diff[:,zone_edge])
                trans_mean[key][m,2,2,1] = np.nanstd(diff[:,zone_edge])

                ax_transitions[j].plot(range(nbin),gauss_smooth(np.nanmean(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j],0),smooth),color=[0.6,0.6,0.6],linewidth=0.3,label='recruiting' if j==0 else None)

                ax_transitions[j].plot(range(nbin),-gauss_smooth(np.nanmean(D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j],0),smooth),color=[0.6,0.6,0.6],linewidth=0.3,label='dismissing' if j==0 else None)

                ax_transitions[j].plot(range(nbin),gauss_smooth(np.nanmean(D.cMice[mouse].stats['transition']['recruitment'][s_bool,:,j]-D.cMice[mouse].stats['transition']['dismissal'][s_bool,:,j],0),smooth),color='tab:red',linestyle=ls_arr[j],linewidth=0.3)

            for key in status_key:
                for k in range(3):
                    ax_trans[key][k].errorbar([0,1,2],trans_mean[key][m,k,:,0],trans_mean[key][m,k,:,1],marker='.',markersize=2,color=[0.6,0.6,0.6],linewidth=0.3)
                # ax_trans_silent[k].errorbar([0,1,2],trans_mean['active'][m,k,:,0],trans_mean['silent'][m,k,:,1],linestyle='-',marker='.',markersize=2,color=[0.6,0.6,0.6])
                # ax_trans_silent[k].errorbar([0,1,2],trans_mean['coding'][m,k,:,0],trans_mean['coding'][m,k,:,1],linestyle='-',marker='.',markersize=2,color=[0.6,0.6,0.6])

            stab_tmp = D.cMice[mouse].stats['transition']['stabilization'][s_bool,:]
            stab_mean[m,0,0] = np.nanmean(stab_tmp[:,zone_nRnG])
            stab_mean[m,0,1] = np.nanstd(stab_tmp[:,zone_nRnG])
            stab_mean[m,1,0] = np.nanmean(stab_tmp[:,zone_RW])
            stab_mean[m,1,1] = np.nanstd(stab_tmp[:,zone_RW])
            stab_mean[m,2,0] = np.nanmean(stab_tmp[:,zone_edge])
            stab_mean[m,2,1] = np.nanstd(stab_tmp[:,zone_edge])

            ax_transitions_stable.plot(range(nbin),gauss_smooth(np.nanmean(D.cMice[mouse].stats['transition']['stabilization'][s_bool,:],0),smooth),color='tab:green',linewidth=0.5,label='stabilizing')
            ax_trans_stable.errorbar([0,1,2],stab_mean[m,:,0],stab_mean[m,:,1],linestyle='-',marker='.',markersize=2,color=[0.6,0.6,0.6],linewidth=0.3)



            ### calculate remapp distances
            s1_shifts,s2_shifts,f1,f2 = np.unravel_index(D.cMice[mouse].compare['pointer'].col,(nSes,nSes,D.cMice[mouse].meta['field_count_max'],D.cMice[mouse].meta['field_count_max']))
            c_shifts = D.cMice[mouse].compare['pointer'].row

            Ds = s2_shifts-s1_shifts
            idx = np.where(Ds==1)[0]
            idx_shifts = D.cMice[mouse].compare['pointer'].data[idx].astype('int')-1
            shifts = D.cMice[mouse].compare['shifts'][idx_shifts]
            shifts_distr = D.cMice[mouse].compare['shifts_distr'][idx_shifts,:].toarray()

            s = s1_shifts[idx]
            f = f1[idx]
            c = c_shifts[idx]
            loc_shifts = np.round(D.cMice[mouse].fields['location'][c,s,f,0]).astype('int')
            loc_to_shifts = D.cMice[mouse].fields['p_x'][c,s2_shifts[idx],f2[idx],:]

            remap = np.zeros((nbin,nbin))
            d_remap = np.zeros((nbin,nbin))

            loc_ct = np.zeros(nbin)
            for i in range(nbin):
                idx_i = loc_shifts==i

                loc_ct[i] = idx_i.sum()

                d_remap[:,i] = shifts_distr[idx_i,:].mean(0)
                remap[:,i] = loc_to_shifts[idx_i,:].mean(0)

                d_remap_mean[m,i,0] = np.abs(shifts[idx_i]).mean()
                d_remap_mean[m,i,1] = np.abs(shifts[idx_i]).std()

            if mouse == mouse_ex:
                ax = plt.axes([0.075,0.45,0.175,0.225])
                pl_dat.add_number(fig,ax,order=4,offset=[-50,50])
                ax.imshow(remap,origin='lower',clim=[0,0.03])
                ax.set_ylabel('$\\theta_{s+1}$')
                ax.set_yticks(np.linspace(0,100,3))
                ax.yaxis.set_label_coords(-0.3,0.5)

                ax = plt.axes([0.075,0.125,0.175,0.225])
                pl_dat.add_number(fig,ax,order=5,offset=[-50,50])
                ax.imshow(d_remap,origin='lower',clim=[0,0.03])
                ax.set_yticks(np.linspace(0,100,3))
                ax.set_yticklabels(np.linspace(-50,50,3))
                ax.set_xlim([0,nbin])
                ax.set_ylim([0,nbin])
                ax.set_xlabel('PF location $\\theta_s$')
                ax.set_ylabel('$\\Delta\\theta$')
                ax.yaxis.set_label_coords(-0.3,0.5)

                ax = plt.axes([0.625,0.125,0.1,0.175])
                pl_dat.add_number(fig,ax,order=7,offset=[-75,50])
                F_field_shift = F_shifts(np.linspace(-50,50,100),*D.cMice[mouse].stability['all']['mean'][0,:])
                F_total_RW = np.zeros(nbin)
                for i in np.where(zone_RW)[0]:
                    F_total_RW += np.roll(F_field_shift*loc_ct[i],int(i+nbin/2))
                F_total_RW /= F_total_RW.sum()
                ax.bar(np.arange(nbin),zone_RW*2,width=1,facecolor='tab:red',alpha=0.2)
                remap_RW = remap[:,zone_RW].sum(1)
                ax.plot(np.arange(nbin),remap_RW/remap_RW.sum(),'tab:red')
                ax.plot(np.arange(nbin),F_total_RW,'k',linestyle='--',linewidth=0.75)
                ax.set_ylim([0,0.08])
                ax.set_xlabel('$\\theta_s$')
                pl_dat.remove_frame(ax,['top','right'])

                ax = plt.axes([0.75,0.125,0.1,0.175])
                F_total_edge = np.zeros(nbin)
                for i in np.where(zone_edge)[0]:
                    F_total_edge += np.roll(F_field_shift*loc_ct[i],int(i+nbin/2))
                F_total_edge /= F_total_edge.sum()
                ax.bar(np.arange(nbin),zone_edge*2,width=1,facecolor='tab:green',alpha=0.2)
                remap_edge = remap[:,zone_edge].sum(1)
                ax.plot(np.arange(nbin),remap_edge/remap_edge.sum(),'tab:green')
                ax.plot(np.arange(nbin),F_total_edge,'k',linestyle='--',linewidth=0.75)
                ax.set_ylim([0,0.08])
                ax.set_yticklabels([])
                ax.set_xlabel('$\\theta_s$')
                pl_dat.remove_frame(ax,['top','right'])

                ax = plt.axes([0.875,0.125,0.1,0.175])
                F_total_nRnG = np.zeros(nbin)
                for i in np.where(zone_nRnG)[0]:
                    F_total_nRnG += np.roll(F_field_shift*loc_ct[i],int(i+nbin/2))
                F_total_nRnG /= F_total_nRnG.sum()
                ax.bar(np.arange(nbin),zone_nRnG*2,width=1,facecolor='tab:blue',alpha=0.2)
                remap_nRnG = remap[:,zone_nRnG].sum(1)
                ax.plot(np.arange(nbin),remap_nRnG/remap_nRnG.sum(),'tab:blue')
                ax.plot(np.arange(nbin),F_total_nRnG,'k',linestyle='--',linewidth=0.75)
                ax.set_ylim([0,0.08])
                ax.set_yticklabels([])
                ax.set_xlabel('$\\theta_s$')
                pl_dat.remove_frame(ax,['top','right'])

            # d_remap * np.linspace(-50,50,100)[:,np.newaxis]
            ax_d_shift.plot(np.arange(nbin),d_remap_mean[m,:,0],color=[0.6,0.6,0.6],linewidth=0.3)
            ax_d_shift_SD.plot(np.arange(nbin),d_remap_mean[m,:,1],color=[0.6,0.6,0.6],linewidth=0.3)
            # pl_dat.plot_with_confidence(ax_d_shift,np.arange(nbin),d_remap_mean[:,0],d_remap_mean[:,1],col='k')

            ### get matrix of remapping locations (from-to)

            ### get average remapping distances

            ### get rw specific remapping, compared with "theory"



        for axx in ax_transitions:
            axx.set_ylim([-3,3])
            pl_dat.remove_frame(axx,['top','right'])
            axx.set_xticklabels([])

        ax_transitions_silent.set_ylabel('$\\frac{transition}{bin}$ \n silent',fontsize=8)
        ax_transitions_silent.text(x=40,y=2,s='from/to silent',fontsize=6)
        ax_transitions_active.set_ylabel('$\\frac{transition}{bin}$ \n active',fontsize=8)
        ax_transitions_active.text(x=40,y=2,s='from/to active',fontsize=6)
        ax_transitions_coding.set_ylabel('$\\frac{transition}{bin}$ \n coding',fontsize=8)
        ax_transitions_coding.text(x=40,y=2,s='from/to coding',fontsize=6)

        for key in status_key:
            for k in range(3):
                ax_trans[key][k].errorbar([0,1,2],np.nanmean(trans_mean[key][:,k,:,0],0),np.nanstd(trans_mean[key][:,k,:,0],0),linestyle='-',color='k',linewidth=1)

                ax_trans[key][k].set_xlim([-0.5,2.5])
                ax_trans[key][k].set_xticks([0,1,2])
                pl_dat.remove_frame(ax_trans[key][k],['top','right'])
                if (key=='coding') & (k>0):
                    ax_trans[key][k].set_xticklabels(['PC','RW','edge'],rotation=60)
                else:
                    ax_trans[key][k].set_xticklabels([])
            ax_trans[key][0].set_ylim([0,3])
            ax_trans[key][1].set_ylim([0,3])
            ax_trans[key][2].set_ylim([-1.5,1.5])
            ax_trans[key][1].set_yticklabels([])

        ax_trans['silent'][0].set_title('recruit',fontsize=8)
        ax_trans['silent'][1].set_title('dismiss',fontsize=8)
        ax_trans['silent'][2].set_title('difference',fontsize=8)

        ax_transitions_stable.set_ylim([0,3])
        ax_transitions_stable.set_xlabel('position $\\theta_s$')
        ax_transitions_stable.set_ylabel('$\\frac{stable PF}{bin}$',fontsize=8)
        pl_dat.remove_frame(ax_transitions_stable,['top','right'])

        ax_trans_stable.errorbar([0,1,2],np.nanmean(stab_mean[:,:,0],0),np.nanstd(stab_mean[:,:,0],0),linestyle='-',color='k',linewidth=1)
        ax_trans_stable.set_xticks([0,1,2])
        ax_trans_stable.set_xticklabels(['PC','RW','edge'],rotation=60)
        ax_trans_stable.set_xlim([-0.5,2.5])
        ax_trans_stable.set_ylim([0,3])
        pl_dat.remove_frame(ax_trans_stable,['top','right'])

        ax_d_shift.plot(np.arange(nbin),d_remap_mean[...,0].mean(0),'k')
        ax_d_shift_SD.plot(np.arange(nbin),d_remap_mean[...,1].mean(0),'k')
        ax_d_shift.set_ylim([0,30])
        ax_d_shift_SD.set_ylim([0,20])
        ax_d_shift_SD.set_xticklabels([])
        ax_d_shift.set_xlabel('$\\theta_s$')
        ax_d_shift.set_ylabel('$\\Delta \\theta$')
        ax_d_shift_SD.set_ylabel('$\\sigma_{\\Delta \\theta}$')
        pl_dat.remove_frame(ax_d_shift,['top','right'])
        pl_dat.remove_frame(ax_d_shift_SD,['top','right'])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('steadystate_dynamics')



    if plot_fig[8]:

        pl_dat = plot_dat(folder='mapChanges',sv_ext=sv_ext)

        SD = 1#1.96
        ds = 5


        # ax_sketch = plt.axes([0.075,0.85,0.15,0.1])
        print('sketch showing, how remapping is happening at some session (reward zone shifted by arrow?)')

        for m,mouse in enumerate(mice):
            fig = plt.figure(figsize=(7,5),dpi=300)
            nSes = D.cMice[mouse].meta['nSes']
            D.cMice[mouse].session_data = get_session_specifics(mouse,nSes)

            RW_changes = np.where(np.diff(D.cMice[mouse].session_data['RW_pos'][:,0]))[0] + 1

            print(mouse)
            print(RW_changes)

            ax_RW_density = plt.axes([0.1,0.1,0.35,0.35])
            ax_RW_density1 = plt.axes([0.6,0.1,0.125,0.35])

            for j,change in enumerate(RW_changes):
                RW_pos_pre = D.cMice[mouse].session_data['RW_pos'][change-1,:].astype('int') + 5
                RW_pos_post = D.cMice[mouse].session_data['RW_pos'][change,:].astype('int') + 5

                print(RW_pos_pre)
                print(RW_pos_post)

                zone_RW_pre = np.zeros(nbin,'bool')
                zone_RW_post = np.zeros(nbin,'bool')
                zone_RW_pre[RW_pos_pre[0]:RW_pos_pre[1]] = True
                zone_RW_post[RW_pos_post[0]:RW_pos_post[1]] = True

                # zone_nRnG = np.ones(nbin,'bool')
                # zone_nRnG[RW_pos[0]:RW_pos[1]] = False
                # zone_nRnG[GT_pos[0]:GT_pos[1]] = False
                # zone_nRnG[:10] = False
                # zone_nRnG[-10:] = False
                # zone_edge = np.zeros(nbin,'bool')
                # zone_edge[:10] = True
                # zone_edge[-10:] = True

                s_bool = np.zeros(nSes,'bool')
                s_bool[change-ds:change+ds] = True

                fields = np.zeros((2*ds,nbin))
                for i,s in enumerate(np.where(s_bool)[0]):
                    idx_PC = np.where(D.cMice[mouse].status_fields[:,s,:])
                    fields[i,:] = np.nansum(D.cMice[mouse].fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
                fields_pre = fields[:ds,:].mean(0)
                fields_post = fields[ds:,:].mean(0)
                fields_pre /= fields_pre.sum()
                fields_post /= fields_post.sum()

                fields /= fields.sum(1)[:,np.newaxis]
                fields = gauss_smooth(fields,(0,2))

                ax_fields = plt.axes([0.1+0.3*j,0.6,0.2,0.3])
                ax_fields.imshow(fields,origin='lower',aspect='auto')

                # print(zone_RW_pre)
                # print(zone_RW_post)
                ax_RW_density.plot(np.arange(-5,5),fields[:,zone_RW_pre].mean(1),'k-',linewidth=0.5)
                ax_RW_density.plot(np.arange(-5,5),fields[:,zone_RW_post].mean(1),'r-',linewidth=0.5)

                ax_RW_density1.plot(0,fields_pre[zone_RW_pre].mean(),'k.',markersize=2)
                ax_RW_density1.plot(1,fields_post[zone_RW_pre].mean(),'k.',markersize=2)

                ax_RW_density1.plot(0,fields_pre[zone_RW_post].mean(),'r.',markersize=2)
                ax_RW_density1.plot(1,fields_post[zone_RW_post].mean(),'r.',markersize=2)



            ax_RW_density.set_ylim([0,0.02])
            ax_RW_density1.set_ylim([0,0.02])
            plt.show(block=False)

class plot_dat:

    def __init__(self,folder='others',sv_suffix='',sv_ext='png'):
        # self.pathFigures = pathFigures
        # self.mouse = mouse

        self.pathFigures = '/home/wollex/Data/Science/PhD/Thesis/pics/%s'%folder

        L_track = 100
        nbin=100
        # nbin = para['nbin']

        self.sv_opt = {'suffix':sv_suffix,
                       'ext':sv_ext,
                       'dpi':300}

        self.plt_presi = True;
        self.plot_pop = False;

        # self.plot_arr = ['NRNG','GT','RW']
        # self.col = ['b','g','r']
        # self.col_fill = [[0.5,0.5,1],[0.5,1,0.5],[1,0.5,0.5]]

        # self.h_edges = np.linspace(-0.5,nSes+0.5,nSes+2)
        # self.n_edges = np.linspace(1,nSes,nSes)
        # self.bin_edges = np.linspace(1,L_track,nbin)

        # self.bars = {}
        # self.bars['PC'] = np.zeros(nbin)
        # self.bars['PC'][para['zone_mask']['others']] = 1
        #
        # self.bars['GT'] = np.zeros(nbin);
        #
        # if np.count_nonzero(para['zone_mask']['gate'])>1:
        #   self.bars['GT'][para['zone_mask']['gate']] = 1
        #
        # self.bars['RW'] = np.zeros(nbin);
        # self.bars['RW'][para['zone_mask']['reward']] = 1


        ### build blue-red colormap
        #n = 51;   ## must be an even number
        #cm = ones(3,n);
        #cm(1,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## red
        #cm(2,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## green
        #cm(2,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## green
        #cm(3,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## blue

    def remove_frame(self,ax,positions=None,ticks=False):

        if positions is None:
            positions = ['left','right','top','bottom']

        for p in positions:
            ax.spines[p].set_visible(False)

        if ticks:
            ax.set_yticks([])
            ax.set_xticks([])

    def plot_with_confidence(self,ax,x_data,y_data,CI,col='k',ls='-',lw=1,label=None):

        col_fill = np.minimum(np.array(colors.to_rgb(col))+np.ones(3)*0.3,1)
        if len(CI.shape) > 1:
          ax.fill_between(x_data,CI[0,:],CI[1,:],color=col_fill,alpha=0.2)
        else:
          ax.fill_between(x_data,y_data-CI,y_data+CI,color=col_fill,alpha=0.2)
        ax.plot(x_data,y_data,color=col,linestyle=ls,linewidth=lw,label=label)

    def add_number(self,fig,ax,order=1,offset=None):

        # offset = [-175,50] if offset is None else offset
        offset = [-150,50] if offset is None else offset
        offset = np.multiply(offset,self.sv_opt['dpi']/300)
        pos = fig.transFigure.transform(plt.get(ax,'position'))
        x = pos[0,0]+offset[0]
        y = pos[1,1]+offset[1]
        ax.text(x=x,y=y,s='%s)'%chr(96+order),ha='center',va='center',transform=None,weight='bold',fontsize=14)


    def save_fig(self,fig_name,fig_pos=None):
        path = pathcat([self.pathFigures,'mice_%s%s.%s'%(fig_name,self.sv_opt['suffix'],self.sv_opt['ext'])]);
        plt.savefig(path,format=self.sv_opt['ext'],dpi=self.sv_opt['dpi'])
        print('Figure saved as %s'%path)
