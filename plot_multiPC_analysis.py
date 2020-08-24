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

from utils import get_nPaths, pathcat, periodic_distr_distance, bootstrap_data, get_average, pickleData, z_from_point_normal_plane, KS_test, E_stat_test, gauss_smooth, calculate_img_correlation, get_shift_and_flow, get_reliability, com
from get_t_measures import *
from utils_data import set_para

warnings.filterwarnings("ignore")

def plot_multiPC_analysis(D,plot_arr=[0,1],N_bs=10,n_processes=0,reprocess=False,sv=False,sv_ext='png',PC=None,active=None):#,N_bs,s_offset,sv,sv_suffix,sv_ext,arrays,occupancy,ROI_recurr,N_pairs,N_norm,pop_overlap)#,ROI_rec2,ROI_tot2)#pathBase,mouse)

    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    nMice = len(D.cMice.keys())
    plot_fig = np.zeros(100).astype('bool')
    for p in plot_arr:
        plot_fig[p] = True

    if plot_fig[0]:

        pie = True
        pie_labels = ['nPCs','GT','RW','nRG']
        pie_col = [[0.4,0.4,0.4],'w','tab:green','tab:red','tab:blue']
        pie_explode = [0.,0.,200,200,200]
        plt.figure(figsize=(7,5),dpi=300)
        width=0.5
        gate_mice = ["34","35","65","66","72","839","840","841","842","879","882","884","886","549","551","756","757","758","918shKO","931wt","943shKO"]
        nogate_mice = ["232","236","243","245","246","762"]#"231",
        i = -1
        # for i,mouse in enumerate(D.cMice.keys()):#D.cMice.keys()
        for mouse in nogate_mice:
            print(mouse)
            if not (mouse in D.cMice.keys()):
                continue
            i+=1

            if i//6 > 0:
                i = 0
                plt.show(block=False)
                plt.figure(figsize=(7,5),dpi=300)

            nSes = D.cMice[mouse].meta['nSes']
            nC = D.cMice[mouse].meta['nC']
            pl_dat = plot_dat(D.cMice[mouse].meta['mouse'],pathcat([D.cMice[mouse].meta['pathMouse'],'Figures']),nSes,D.cMice[mouse].para,sv_ext='png')

            t_measures = get_t_measures(mouse)[:nSes]
            print(t_measures)

            ### get stats of all sessions
            D.cMice[mouse].session_classification()
            D.cMice[mouse].update_status()

            mask_PC = (~D.cMice[mouse].status[...,2])
            mask_fields = D.cMice[mouse].fields['status']<3
            dat = {}
            dat['reliability'] = np.ma.array(D.cMice[mouse].fields['reliability'], mask=mask_fields, fill_value=np.NaN)
            dat['width'] = np.ma.array(D.cMice[mouse].fields['width'][...,0], mask=mask_fields, fill_value=np.NaN)
            dat['MI_value'] = np.ma.array(D.cMice[mouse].stats['MI_value'], mask=mask_PC, fill_value=np.NaN)
            dat['max_rate'] = np.ma.array(D.cMice[mouse].fields['max_rate'], mask=mask_fields, fill_value=np.NaN)

            dat_mean = {'reliability':  np.zeros(nSes)*np.NaN,
                        'width':        np.zeros(nSes)*np.NaN,
                        'MI_value':     np.zeros(nSes)*np.NaN,
                        'max_rate':     np.zeros(nSes)*np.NaN}

            for key in ['reliability','width','MI_value','max_rate']:

                for s in np.where(D.cMice[mouse].sessions['bool'])[0]:
                    dat_s = dat[key][:,s,...].compressed()
                    dat_mean[key][s] = np.mean(dat_s)

            # ax = plt.subplot(4,5,i+1)
            ax = plt.axes([0.05+0.5*(i//3),0.75-0.35*(i%3),0.15,0.21])
            nROI = D.cMice[mouse].status[:,D.cMice[mouse].sessions['bool'],:].sum(0)

            if np.any(D.cMice[mouse].para['zone_mask']['gate']):
                idx = [1,3,4,5]
            else:
                idx = [1,4,5]
            col = [pie_col[i-1] for i in idx]
            explode = [pie_explode[i-1] for i in idx]
            nROI_norm = nROI[...,idx].mean(0)

            nROI_norm = nROI_norm/nROI_norm.sum()*100
            nTotal = [nROI[...,1].mean(),nROI[...,1].std()]


            D.cMice[mouse].session_classification(sessions=D.sessions[mouse]['analyze'])
            D.cMice[mouse].update_status()

            # rad = nROI[...,1].mean()
            rad = 500+nTotal[0]/2
            ax.pie(nROI_norm,explode=explode,startangle=170,autopct='%1.1f%%',radius=rad,colors=col,pctdistance=1.4,textprops={'fontsize':6})#,labels=pie_labels
            ax.text(0,-rad-500,'n=$%d\pm %d$'%(nTotal[0],nTotal[1]),fontsize=6,color='k',ha='center')
            ax.text(0,-rad+250,'%s'%mouse,fontsize=6,color='w',ha='center')
            ax.set_xlim([-2000,2000])
            ax.set_ylim([-2000,2000])

            t_half = t_measures.max()/2
            ax = plt.axes([0.25+0.5*(i//3),0.925-0.35*(i%3),0.175,0.03])
            ax.boxplot(D.cMice[mouse].fields['reliability'][D.cMice[mouse].fields['status']>=3],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            ax.set_ylim([0,0.5])
            ax.set_xticks([])
            # ax.set_xticks(np.linspace(0,3000,4))
            # ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,7)])
            ax.set_xlim([-10,t_measures.max()+10])
            ax.plot(t_measures,dat_mean['reliability'],'.',markersize=1.5,color='k')
            ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['reliability'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            pl_dat.remove_frame(ax,['top'],ticks=False)
            ax.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=True)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_ylabel('a',fontsize=8)#,rotation='horizontal',labelpad=-20,y=0.9,ha='left')

            ax = plt.axes([0.25+0.5*(i//3),0.875-0.35*(i%3),0.175,0.03])
            ax.boxplot(D.cMice[mouse].stats['MI_value'][D.cMice[mouse].status[...,2]],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            ax.set_ylim([0,0.5])
            ax.set_xticks([])
            # ax.set_xticks(np.linspace(0,3000,4))
            # ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,7)])
            ax.set_xlim([-10,t_measures.max()+10])
            ax.plot(t_measures,dat_mean['MI_value'],'.',markersize=1.5,color='k')
            ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['MI_value'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            pl_dat.remove_frame(ax,['top'],ticks=False)
            ax.tick_params(axis='y',which='both',left=True,right=True,labelright=True,labelleft=False)
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_ylabel('MI',fontsize=8)#,rotation='horizontal',labelpad=-50,y=1.0,ha='left')


            ax = plt.axes([0.25+0.5*(i//3),0.825-0.35*(i%3),0.175,0.03])
            ax.boxplot(D.cMice[mouse].fields['width'][D.cMice[mouse].fields['status']>=3,0],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            ax.set_ylim([0,10])
            # ax.set_xticks(np.linspace(0,3000,4))
            # ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,4)])

            # ax.tick_params(which='minor',length=2)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_xlim([-10,t_measures.max()+10])
            ax.plot(t_measures,dat_mean['width'],'.',markersize=1.5,color='k')
            ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['width'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            pl_dat.remove_frame(ax,['top'],ticks=False)
            ax.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=True)
            ax.set_ylabel('$\sigma$',fontsize=8)#,rotation='horizontal',labelpad=0,y=1.0,ha='left')
            # ax.yaxis.tick_right()


            ax = plt.axes([0.25+0.5*(i//3),0.775-0.35*(i%3),0.175,0.03])
            ax.boxplot(D.cMice[mouse].fields['max_rate'][D.cMice[mouse].fields['status']>=3],positions=[t_half],widths=t_half/2,whis=[5,95],showfliers=False)
            ax.set_ylim([0,30])
            ax.set_xticks(np.linspace(0,3000,4))
            ax.set_xticklabels(['%d'%i for i in np.linspace(0,3000,4)])

            # ax.tick_params(which='minor',length=2)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_xlim([-10,t_measures.max()+10])
            ax.plot(t_measures,dat_mean['max_rate'],'.',markersize=1.5,color='k')
            ax.plot(t_measures[D.cMice[mouse].sessions['bool']],dat_mean['max_rate'][D.cMice[mouse].sessions['bool']],'.',markersize=2,color='tab:green')
            pl_dat.remove_frame(ax,['top'],ticks=False)
            ax.tick_params(axis='y',which='both',left=True,right=True,labelright=True,labelleft=False)
            ax.yaxis.set_label_position('right')
            ax.set_ylabel('$max(\\nu)$',fontsize=8)#,rotation='horizontal',labelpad=0,y=1.0,ha='left')
            # ax.yaxis.tick_right()

        # ax.set_xticks(range(nMice))
        # ax.set_xticklabels(D.cMice.keys())
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('session_align')


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

  def remove_frame(self,ax,positions=None,ticks=True):

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



  def save_fig(self,fig_name,fig_pos=None):
    path = pathcat([self.pathFigures,'m%s_%s%s.%s'%(self.mouse,fig_name,self.sv_opt['suffix'],self.sv_opt['ext'])]);
    plt.savefig(path,format=self.sv_opt['ext'],dpi=self.sv_opt['dpi'])
    print('Figure saved as %s'%path)
