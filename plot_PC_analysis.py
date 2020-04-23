import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import scipy as sp
import scipy.ndimage as spim
import scipy.stats as sstats
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from tqdm import *
import os, time, math, h5py, pickle, multiprocessing, random, cv2

from utils import get_nPaths, pathcat, periodic_distr_distance, bootstrap_data, get_average, pickleData, z_from_point_normal_plane, KS_test, E_stat_test
from utils_data import set_para

def plot_PC_analysis(cluster,plot_arr=[0,1],sv=False,sv_ext='png'):#,N_bs,s_offset,sv,sv_suffix,sv_ext,arrays,occupancy,ROI_recurr,N_pairs,N_norm,pop_overlap)#,ROI_rec2,ROI_tot2)#pathBase,mouse)
  
  nSes = cluster.meta['nSes']
  nC = cluster.meta['nC']
  nbin = cluster.para['nbin']
  t_ses = np.linspace(1,nSes,nSes)
  
  ### think about: doing all the plots in an UI, able to change parameters such as gate & reward location, sessions considered, etc
  ### -> which plots are needed, which ones are not that important?
  
  ## fitting functions and options
  F_shifts = lambda x,A0,A,sig,theta : A/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-theta)**2/(2*sig**2)) + A0/len(x)     ## gaussian + linear offset
  def fit_shift_model(data):
    shift_hist = np.histogram(data,np.linspace(-50,50,101),density=True)[0]
    try:
      return curve_fit(F_shifts,np.linspace(-49.5,49.5,100),shift_hist)
    except:
      return np.zeros(4)*np.NaN, np.NaN
  
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
  
  ## 13:     
  ## 14:     
  ## 15:     
  ## 16:     
  ## 17:     
  ## 18:     
  ## 19:     
  ## 20:     
  
  
  #plot_fig[0] = True;        ## 
  #plot_fig[1] = True;        ## 
  #plot_fig[2] = True;        ## 
  #plot_fig[3] = True;        ## 
  #plot_fig[4] = True;        ## 
  #plot_fig[5] = True;        ## 
  #plot_fig[6] = True;        ## 
  
  #plot_fig[7] = True;        ## 
  #plot_fig[8] = True;        ## 
  
  #plot_fig[9] = True;        ## 
  
  
  
  
  #plot_fig(6) = true;         ## remapping
  #plot_fig(7) = true;         ## trial threshold (not needed!!)
  #plot_fig(8) = true;         ## MI per session
  #plot_fig(10) = true;        ## PC remapping (presi)
  #plot_fig(11) = true;        ## overrepresentation //correlation of activity / PCs
  #plot_fig(12) = true;        ## plot pie charts of neuron numbers / session
  #plot_fig(13) = true;        ## firing rates
  #plot_fig(14) = true;        ## (no plot) pie charts: silent in- and out- transitions
  #plot_fig(15) = true;        ## (old) shift-distributions
  #plot_fig(16) = false;       ## (check) traces of PCs
  #plot_fig(17) = true;        ## ROI recurrence

#    plot_fig(18) = true;      ## population test
#    plot_fig(19) = true;      ## (shift to 10) PC/active numbers histogram
#    plot_fig(20) = true;      ## (not yet done) correlated population changes
#    plot_fig(21) = true;      ## firing rate / population
#    plot_fig(22) = true;      ## coding / session
#    plot_fig(23) = true;      ## firing rate correlation
#    plot_fig(24) = true;      ## population fraction reliability
#    plot_fig(25) = true;      ## number of fields
#    plot_fig(26) = true;      ## stability in populations (hard threshold)
#    plot_fig(27) = true;      ## population sizes
#    plot_fig(28) = true;      ## formation / deformation ratio
#    plot_fig(29) = true;      ## intercoding state
#    plot_fig(30) = true;      ## stability in populations (fit to distribution)
#    plot_fig(31) = true;      ## waiting time to next coding
#    plot_fig(32) = true;      ## shift distributions dependence on time passed
#    plot_fig(33) = true;
#    plot_fig(34) = true;
#    plot_fig(:) = true;
  
#    if nargin < 8
    ## get neurons & PCs per session
    
    
#      PC_recurr = zeros(nSes,t_ses(end),3);  ## 1 = NRNG, 2 = GT, 3 = RW
#      PC_stable = zeros(nSes,t_ses(end),3);  ## 1 = NRNG, 2 = GT, 3 = RW
#      stable_pos = zeros(nSes,t_ses(end),nbin);

#      remap_pos = zeros(nSes,nSes-1,nbin+2,nbin+2);
#      remap_df = struct('hist',zeros(nSes-1,nbin,3),'ds',struct('rw',cell(nSes-1,1),'gt',cell(nSes-1,1),'nrng',cell(nSes-1,1)));
#      remap_df = zeros(nC,nSes-1,t_ses(end))*NaN;    ## session, delta session
    
  
  nSes_real = cluster.boolSessions.sum()
  print(np.where(cluster.boolSessions)[0])
  
  t_start = time.time()
    
  #nROI = np.zeros((nSes,5)); ## 1 = silent, 2 = active (nPC), 3 = NRNG, 4 = GT, 5 = RW
  #nROI(:,1) = cluster.activity['status'][:,:,0].sum(0)
  #nROI(:,2) = sum(cluster.activity['status'][:,:,1]>0 & ~np.any(cluster.activity['status'](:,:,3:5),3),1);
  
  
#### ------------ get information about populations ------------- ###
  
  nOcc = cluster.activity['status'][:,cluster.boolSessions,:].sum(1);
  nOcc[:,1] = nOcc[:,1] - np.any(cluster.activity['status'][:,cluster.boolSessions,2:],2).sum(1);
  
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
    
    numbers = np.zeros((nC,2))
    numbers[:,0] = cluster.activity['status'][:,cluster.boolSessions,1].sum(1)        ## number of neurons
    numbers[:,1] = nSes_real - nOcc[:,:2].sum(1)      ## number of place cells
    
    #numbers = cluster.activity['status'][:,cluster.boolSessions,1].sum(1)        ## number of neurons
    
    plt.figure(figsize=(4,2.5))
    plt.axes([0.2,0.2,0.75,0.75])
    
    #plt.hist(numbers,pl_dat.h_edges,color=[0.6,0.6,0.6],width=0.4,label='# sessions active');
    plt.hist(numbers,pl_dat.h_edges,color=[[0.6,0.6,0.6],'k'],width=0.4,label=['# sessions active','# sessions coding']);
    plt.plot([nSes_real,nSes_real],[0,500],'r--')
    plt.xlabel('# sessions',fontsize=14)
    plt.ylabel('# neurons',fontsize=14)
    plt.legend(loc='upper right')
    plt.xlim([-0.5,nSes+0.5])
    plt.ylim([0,300])
    plt.tight_layout()
    plt.show(block=False)
    
    #plt.figure()
    #plt.scatter(numbers[:,0]+0.5*np.random.rand(nC),numbers[:,1]+0.5*np.random.rand(nC),s=10,marker='.')
    #plt.show(block=False)
    
    if sv:
      pl_dat.save_fig('Nactive_nPC')
  
#### ---------------------------- plot matching results --------------------------- ###
  
  if plot_fig[1]:
    
    #fig_pos = [200 200 400 200];
    plt.figure(figsize=(4,3))
    
    ax1 = plt.axes([0.15, 0.5, 0.8, 0.45])
    ax2 = plt.axes([0.15, 0.2, 0.8, 0.25])
    
    
    active_time = np.zeros(nSes)
    for s in range(nSes):
      if cluster.boolSessions[s]:
        pathSession = pathcat([cluster.meta['pathMouse'],'Session%02d'%(s+1)]);
        
        for file in os.listdir(pathSession):
          if file.endswith("aligned.mat"):
            pathBH = os.path.join(pathSession, file)
        
        f = h5py.File(pathBH,'r')
        key_array = ['longrunperiod']
        
        dataBH = {}
        for key in key_array:
          dataBH[key] = np.squeeze(f.get('alignedData/resampled/%s'%key).value)
        f.close()
        
        active_time[s] = dataBH['longrunperiod'].sum()/len(dataBH['longrunperiod']);
    
    ax2.plot(t_ses[cluster.boolSessions],active_time[cluster.boolSessions],color='k')
    #ax2.plot(t_measures(1:s_end),active_time,'k')
    ax2.set_xlim([0,t_ses[-1]])
    ax2.set_ylim([0,1])
    ax2.set_xlabel('t [h]',fontsize=14)
    ax2.set_ylabel('active time',fontsize=14)
    
    
    ax1.plot(t_ses[cluster.boolSessions],np.ones(cluster.boolSessions.sum())*nC,color='k',linestyle=':',label='# neurons')
    ax1.scatter(t_ses[cluster.boolSessions],cluster.activity['status'][:,cluster.boolSessions,1].sum(0),s=20,color='k',marker='o',facecolor='none',label='# active neurons')
    ax1.set_ylim([0,nC*1.2])
    ax1.set_xlim([0,t_ses[-1]])
    ax1.legend(loc='upper right')
    ax1.set_xticks([])
    if pl_dat.plt_presi:
      plt.tight_layout()
      pl_dat.save_fig('neuron_numbers_act')
      #path = pathcat(pathFigures,sprintf('ROInum#s_2.png',sv_suffix));
      #print(path,'-dpng','-r300')
    
    ax1.scatter(t_ses[cluster.boolSessions],np.any(cluster.activity['status'][:,cluster.boolSessions,2:],2).sum(0),s=20,color='k',marker='o',facecolors='k',label='# place cells')
    #legend('Location',[0.4 0.79 0.3 0.15])
    #if plt_presi
      #path = pathcat(pathFigures,sprintf('ROInum%s_3.png',sv_suffix));
      #print(path,'-dpng','-r300')
    
##      plot(t_ses,nROI(:,3),'b-','DisplayName','# NRNG')
##      plot(t_ses,nROI(:,4),'r-','DisplayName','# GT')
##      plot(t_ses,nROI(:,5),'g-','DisplayName','# RW')
    plt.tight_layout()
    plt.show(block=False)
    
    if sv:
      pl_dat.save_fig('neuron_numbers')
    
    
  if plot_fig[2]:
    
    #plt.figure()
    f,axs = plt.subplots(2,2,figsize=(10,4))
    
    axs[1][0].plot([0,nSes],[0,0],color=[0.8,0.8,0.8],linestyle='--')
    axs[1][1].plot([0,nSes],[0,0],color=[0.8,0.8,0.8],linestyle='--')
    #ax1 = plt.axes([0.2,0.3,0.75,0.65])
    #ax1.plot([0,nSes],[0.75,0.75],color='k',linestyle=':')
    
    
    K_act = cluster.activity['status'][:,:,1].sum(0)
    K_PC = np.any(cluster.activity['status'][:,:,2:],-1).sum(0)
    
    overrepr_act = np.zeros((nSes,nSes))*np.NaN
    overrepr_PC = np.zeros((nSes,nSes))*np.NaN
    
    ic_keys = np.array(list(cluster.compare['inter_coding'].keys()))
    ic_vals = np.array(list(cluster.compare['inter_coding'].values()))
    
    L=50
    r_recurr = np.zeros((nSes,nSes))*np.NaN
    r_recurr_cont = np.zeros((nSes,nSes))*np.NaN
    PC_recurr = np.zeros((nSes,nSes))*np.NaN
    PC_recurr_cont = np.zeros((nSes,nSes))*np.NaN
    for s in tqdm(range(nSes)):#min(30,nSes)):
      
      if cluster.boolSessions[s]:
        overlap_act = cluster.activity['status'][cluster.activity['status'][:,s,1],:,1].sum(0)
        overlap_PC = np.any(cluster.activity['status'][np.any(cluster.activity['status'][:,s,2:],-1),:,2:],-1).sum(0)
        
        r_recurr[s,1:(nSes-s)] = overlap_act[s+1:]/K_act[s+1:]
        r_recurr[s,np.where(~cluster.boolSessions[s:])[0]] = np.NaN
        
        PC_recurr[s,1:(nSes-s)] = overlap_PC[s+1:]/K_PC[s+1:]
        PC_recurr[s,np.where(~cluster.boolSessions[s:])[0]] = np.NaN
        
        
        rand_pull_act = np.zeros((nSes-s,L))*np.NaN
        rand_pull_PC = np.zeros((nSes-s,L))*np.NaN
        
        for s2 in range(s+1,nSes):
          rand_pull_act[s2-s,:] = (np.random.choice(nC,(L,K_act[s]))<K_act[s2]).sum(1)
          
          offset = K_act[s] - overlap_act[s2]
          randchoice_1 = np.random.choice(K_act[s],(L,K_PC[s]))
          randchoice_2 = np.random.choice(np.arange(offset,offset+K_act[s2]),(L,K_PC[s2]))
          for l in range(L):
            rand_pull_PC[s2-s,l] = np.isin(randchoice_1[l,:],randchoice_2[l,:]).sum()
          
          
          ### find continuously coding neurons
          r_recurr_cont[s,s2-s] = (cluster.activity['status'][:,s:s2+1,1].sum(1)==(s2-s+1)).sum()/K_act[s2]#(ic_vals[idx_ds,2] == 1).sum()/K_act[s2]
          PC_recurr_cont[s,s2-s] = (np.any(cluster.activity['status'][:,s:s2+1,2:],2).sum(1)==(s2-s+1)).sum()/K_PC[s2]#(ic_vals[idx_ds,2] == 1).sum()/K_act[s2]
        r_recurr_cont[s,np.where(~cluster.boolSessions[s:])[0]] = np.NaN
        PC_recurr_cont[s,np.where(~cluster.boolSessions[s:])[0]] = np.NaN
        
        
        overrepr_act[s,:nSes-s] = (overlap_act[s:]-rand_pull_act.mean(1))/rand_pull_act.std(1) 
        overrepr_act[s,np.where(~cluster.boolSessions[s:])[0]] = np.NaN
        
        overrepr_PC[s,:nSes-s] = (overlap_PC[s:]-rand_pull_PC.mean(1))/rand_pull_PC.std(1) 
        overrepr_PC[s,np.where(~cluster.boolSessions[s:])[0]] = np.NaN
        
        axs[1][0].scatter(pl_dat.n_edges,overrepr_act[s,:],5,color=[0.8,0.8,0.8],marker='o')
        axs[1][1].scatter(pl_dat.n_edges,overrepr_PC[s,:],5,color=[0.8,0.8,0.8],marker='o')
        
        if s==0:
          axs[0][0].scatter(pl_dat.n_edges,r_recurr[s,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
          axs[0][0].scatter(pl_dat.n_edges,r_recurr_cont[s,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
          
          axs[0][1].scatter(pl_dat.n_edges,PC_recurr[s,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
          axs[0][1].scatter(pl_dat.n_edges,PC_recurr_cont[s,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
          
        else:
          axs[0][0].scatter(pl_dat.n_edges,r_recurr[s,:],5,color=[0.8,0.8,0.8],marker='o')
          axs[0][0].scatter(pl_dat.n_edges,r_recurr_cont[s,:],5,color=[0.6,1,0.6],marker='o')
          
          axs[0][1].scatter(pl_dat.n_edges,PC_recurr[s,:],5,color=[0.8,0.8,0.8],marker='o')
          axs[0][1].scatter(pl_dat.n_edges,PC_recurr_cont[s,:],5,color=[0.6,1,0.6],marker='o')
    
    axs[0][0].plot(pl_dat.n_edges,np.nanmean(r_recurr,0),color='k')
    
    axs[0][0].legend(loc='lower right',fontsize=12)
    
    axs[1][0].plot(pl_dat.n_edges,np.nanmean(overrepr_act,0),color='k')
    axs[0][1].plot(pl_dat.n_edges,np.nanmean(PC_recurr,0),color='k')
    axs[1][1].plot(pl_dat.n_edges,np.nanmean(overrepr_PC,0),color='k')
    
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
    
    
  
  if plot_fig[3]:
    
    ### # of fields
    mean_fc = np.zeros(nSes)
    for s in range(nSes):
      if cluster.boolSessions[s]:
        #print(s)
        #print(cluster.activity['status'][:,s,1].sum())
        #print(cluster.PCs['fields']['nModes'][:,s])
        #print(cluster.PCs['fields']['nModes'][np.any(cluster.activity['status'][:,s,2:],1),s])
        mean_fc[s] = cluster.PCs['fields']['nModes'][np.any(cluster.activity['status'][:,s,2:],1),s].mean(0)
    
    plt.figure(figsize=(3,2))
    plt.plot(mean_fc,color='k',marker='o')
    plt.ylim([0.9,1.1])
    plt.xlabel('Session')
    plt.ylabel('avg. #fields')
    plt.show(block=False)
    if sv:
      pl_dat.save_fig('field_num')
  
  
  if plot_fig[4]:
    
    #idxes = [range(0,10),range(10,30),range(30,nSes)]
    idxes = [range(0,5),range(5,10),range(10,15)]
    
    plt.figure(figsize=(6,4))
    for (idx,i) in zip(idxes,range(len(idxes))):
      ax = plt.subplot(len(idxes),1,i+1)
      fields = np.nansum(cluster.PCs['fields']['p_x'][:,idx,:,1,:],2).sum(1).sum(0)
      fields /= fields.sum()
      
      ax.bar(pl_dat.bin_edges,pl_dat.bars['GT'],width=1,facecolor=[0.8,1,0.8],edgecolor='none')
      ax.bar(pl_dat.bin_edges,pl_dat.bars['RW'],width=1,facecolor=[1,0.8,0.8],edgecolor='none')
      ax.bar(pl_dat.bin_edges,pl_dat.bars['PC'],width=1,facecolor=[0.7,0.7,1],edgecolor='none')
      #ax.bar(pl_dat.bin_edges,fields)
      ax.hist(cluster.PCs['fields']['parameter'][:,idx,0,3,0].flatten(),pl_dat.bin_edges-0.5,facecolor='k',width=0.8,density=True)
      ax.set_xlim([0,nbin])
      ax.set_ylim([0,0.025])#np.nanmax(fields)*1.2])
      ax.set_ylabel('% of PC',fontsize=14)
      
    ax.set_xlabel('position x',fontsize=14)
    plt.tight_layout()
    #axes(ax1)
    #yyaxis right
    
    #plot(imgaussfilt(sum(occupancy(4:s_end,1:para.nbin))/(sum(sum(nROI(4:s_end,3:5)))/para.nbin),2,'Padding','circular'),'r','LineWidth',2)
    #ylim([0,4])
    #ylabel('ratio','FontSize',14)
    #set(gca,'XTick',linspace(1,para.nbin,5),'XTickLabel',linspace(0,100,5))
    
    #yyaxis left
#        text(19,0.055,'GT','FontSize',14)
#        text(67,0.055,'RW','FontSize',14)
    plt.show(block=False)
    if sv:
      pl_dat.save_fig('PC_coverage_single')

#      overrepr = occupancy(:,1:para.nbin)./(sum(nROI(:,3:5),2)/para.nbin);
    
  if plot_fig[5]:
    
    ### get centers of all coding cells in session s
    s_ref = 10;
    n_plots = 7;
    n_plots_half = (n_plots-1)/2
    ordered = True
    
    if ordered:
      print('ordered')
      idxes = np.where(np.any(cluster.activity['status'][:,s_ref,2:],1))[0]
      sort_idx = np.argsort(cluster.PCs['fields']['parameter'][np.any(cluster.activity['status'][:,s_ref,2:],1),s_ref,0,3,0])
      sort_idx = idxes[sort_idx]
      nID = len(sort_idx)
    
    plt.figure(figsize=(10,3))
    for (s,i) in zip(range(int(s_ref-n_plots_half),int(s_ref+n_plots_half)+1),range(n_plots)):
      ax = plt.subplot(1,n_plots,i+1)
      if not ordered:
        print('not ordered')
        idxes = np.where(np.any(cluster.activity['status'][:,s,2:],1))[0]
        sort_idx = np.argsort(cluster.PCs['fields']['parameter'][np.any(cluster.activity['status'][:,s,2:],1),s,0,3,0])
        sort_idx = idxes[sort_idx]
        nID = len(sort_idx)
      
      firingmap = cluster.activity['firingmap'][sort_idx,s,0,:]
      firingmap = firingmap - firingmap.min(1)[:,np.newaxis]
      firingmap = firingmap / firingmap.sum(1)[:,np.newaxis]
      
      firingmap = spim.gaussian_filter(firingmap,[0,3])
      
      ax.imshow(firingmap)
      
      if i == 0:
        ax.plot([cluster.para['zone_idx']['gate'][0],cluster.para['zone_idx']['gate'][0]],[1,nID],color='r',linestyle=':',linewidth=3)
        ax.plot([cluster.para['zone_idx']['gate'][1],cluster.para['zone_idx']['gate'][1]],[1,nID],color='r',linestyle=':',linewidth=3)
        
        ax.plot([cluster.para['zone_idx']['reward'][0],cluster.para['zone_idx']['reward'][0]],[1,nID],color='g',linestyle=':',linewidth=3)
        ax.plot([cluster.para['zone_idx']['reward'][1],cluster.para['zone_idx']['reward'][1]],[1,nID],color='g',linestyle=':',linewidth=3)
        
        ax.set_xticks(np.linspace(1,nbin,5))
        ax.set_xlim([1,nbin])
        ax.set_ylim([1,nID])
        ax.set_xlabel('Location [cm]')
        ax.set_ylabel('Neuron ID')
      
    plt.set_cmap('jet')
    plt.show(block=False)
        
        
        #ax_hist = axes('Position',[0.25,0.8,0.65,0.15]);
        #bar(ax_hist,linspace(1,para.nbin,para.nbin),occupancy(s,1:end-2),'FaceColor','k')
        #set(ax_hist,'Xtick',[],'YTick',[],'visible','off')
        #suptitle(sprintf('s=%d',s))
##          set()
        
##          histcounts(centers,linspace(1,para.nbin,para.nbin));
##        end
        
##        ax = subplot(1,n_plots+1,n_plots+1);
##        colormap('jet')
##        cb = colorbar();#'Location','West');
##        ylabel(cb,'normalized Ca^{2+} activity above baseline')
##        set(ax,'visible','off')
        
        
        #if sv
          #if ordered
            #path = pathcat(pathFigures,sprintf('PC_coverage_session_%d_aligned_%d%s.%s',s,s_ref,sv_suffix,sv_ext));
          #else
            #path = pathcat(pathFigures,sprintf('PC_coverage_session_%d_nonaligned_%d%s.%s',s,s_ref,sv_suffix,sv_ext));
          #end
          #export_fig(path,sprintf('-%s',sv_ext),'-transparent','-r300')
##            print(path,sprintf('-d%s',sv_ext),'-r300')
          #disp(sprintf('Figure saved as %s',path))
        #end
      #end

#      if sv
#        path = pathcat(pathFigures,sprintf('PC_coverage_multiple%s.png',sv_suffix));
#        print(path,sprintf('-d%s',sv_ext),'-r300')
#        disp(sprintf('Figure saved as %s',path))
#      end
    
  
  if plot_fig[6]:
    
    
    #PC_match_certainty = np.zeros((nC,nSes,nSes))
    #t_start = time.time()
    #for s1 in range(nSes):
      #for s2 in range(s1,nSes):
        #PC_match_certainty[:,s1,s2] = np.power(np.nanprod(cluster.sessions['match_score'][:,s1+1:s2+1],1),1/(np.sum(~np.isnan(cluster.sessions['match_score'][:,s1+1:s2+1]),1)+1))*cluster.sessions['match_score'][:,s2]
        ##print('ds : %d'%(s2-s1))
        ##print(PC_match_certainty[:,s1,s2])
    
    ic_keys = np.array(list(cluster.compare['inter_coding'].keys()))
    idx_good = np.zeros(len(ic_keys)).astype('bool')
    idx_good[np.isin(ic_keys[:,0],np.where(cluster.clusters_bool)[0])] = True
    ic_keys = ic_keys[idx_good,:]
    ic_vals = np.array(list(cluster.compare['inter_coding'].values()))
    ic_vals = ic_vals[idx_good,:]
    
    #ic_vals = ic_vals[ic_vals[:,4]==1,:]
    
    distr_keys=np.array(list(cluster.compare['shifts_distr'].keys()))
    distr_keys = distr_keys[idx_good,:]
    distr_shifts=np.array(list(cluster.compare['shifts_distr'].values()))
    distr_shifts = distr_shifts[idx_good,:]
    
    fpos_keys = np.array(list(cluster.compare['ref_pos'].keys()))
    fpos_keys = fpos_keys[idx_good,:]
    fpos_vals = np.array(list(cluster.compare['ref_pos'].values()))
    print(fpos_vals.shape)
    fpos_vals = fpos_vals[idx_good]
    
    #print(distr_keys)
    #print(fpos_keys)
    #print(ic_keys)
    
    cdf_shifts = np.cumsum(distr_shifts,1)
    
    ### ds = 0
    PC_idx = np.any(cluster.activity['status'][:,:,2:],2)
    p_shift = np.zeros(nbin)
    for s in range(nSes):
      for c in np.where(PC_idx[:,s])[0]:
        roll = round(-cluster.PCs['fields']['parameter'][c,s,0,3,0]+nbin/2).astype('int')
        p_shift += np.roll(cluster.PCs['fields']['p_x'][c,s,0,1,:],roll)
    #print('time took: %5.3f'%(time.time()-t_start))
    p_shift /= p_shift.sum()
    
    PC_idx = np.where(PC_idx)
    N_data = len(PC_idx[0])
    print('N data: %d'%N_data)
    
    cdf_shifts_ds0 = np.cumsum(cluster.PCs['fields']['p_x'][PC_idx[0],PC_idx[1],0,1,:],-1)
    print(cdf_shifts_ds0.shape)
    N_bs = 200
    sample_randval = np.random.rand(N_bs,N_data,2)
    
    ### for ds == 0 (pretty much same as summing up)
    for i in range(N_bs):
      x1 = np.argmin(abs(cdf_shifts_ds0-sample_randval[i,:,0,np.newaxis]),1)
      x2 = np.argmin(abs(cdf_shifts_ds0-sample_randval[i,:,1,np.newaxis]),1)
      shift_distr_ds0 = (x2-x1 + nbin/2)%nbin -nbin/2
    #plt.figure()
    #plt.hist(shift_distr_ds0,np.linspace(-50,50,101),density=True)
    #plt.plot(np.linspace(-49.5,49.5,100),p_shift,'r')
    #plt.show(block=False)
      #data_bs = data[samples[i,:],...]    ### get bootstrap sample
    
    
    
    ### ds > 0
    #PC_shifts = np.reshape(np.array(PC_shifts.todense()),(nC,nSes,nSes))
    
    
    p_ds0,p_cov = fit_shift_model(shift_distr_ds0)
    
    p = np.zeros((nSes,4))*np.NaN
    p_std = np.zeros((nSes,4))*np.NaN
    
    p_cont = np.zeros((nSes,4))*np.NaN
    p_cont_std = np.zeros((nSes,4))*np.NaN
    
    p_mix = np.zeros((nSes,4))*np.NaN
    p_mix_std = np.zeros((nSes,4))*np.NaN
    
    p_disc = np.zeros((nSes,4))*np.NaN
    p_disc_std = np.zeros((nSes,4))*np.NaN
    
    p_silent = np.zeros((nSes,4))*np.NaN
    p_silent_std = np.zeros((nSes,4))*np.NaN
    
    t_start = time.time()
    f_stable_pos = plt.figure(figsize=(7,2.5))
    
    f_shift_distr = plt.figure(figsize=(7,2.5))
    for ds in tqdm(range(1,nSes)):#min(nSes,30)):
      
      #PC_idx_uncertain = np.diagonal(PC_match_certainty,ds,axis1=1,axis2=2) < 0.9
      
      #shifts = np.diagonal(PC_shifts,ds,axis1=1,axis2=2)
      #shifts_hist = np.histogram(shifts[shifts!=0],np.linspace(-50,50,101),density=True)
      ##shifts_hist_certain = np.histogram(shifts[(shifts!=0) & ~PC_idx_uncertain],np.linspace(-50,50,101),density=False)
      ##shifts_hist_uncertain = np.histogram(shifts[(shifts!=0) & PC_idx_uncertain],np.linspace(-50,50,101),density=False)
      
      #shifts_all = shifts[shifts!=0]
      #p[ds,:],p_std[ds,:] = bootstrap_data(fit_shift_model,shifts[shifts!=0],100)
      
      idx_ds = np.where((distr_keys[:,2]-distr_keys[:,1])==ds)[0]
      N_data = len(idx_ds)
      cdf_shifts_ds = distr_shifts[idx_ds,:]
      
      
      ic = ic_vals[idx_ds,:]
      ## all
      p[ds,:], p_std[ds,:], shift_distr = bootstrap_shifts(fit_shift_model,cdf_shifts_ds,N_bs,nbin)
      
      ## cont
      p_cont[ds,:], p_cont_std[ds,:], tmp = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[np.where(ic[:,3]==1)[0]],N_bs,nbin)
      
      ## mixed
      p_mix[ds,:], p_mix_std[ds,:], tmp = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[np.where((ic[:,3]>0) & (ic[:,3]<1))[0]],N_bs,nbin)
      
      ## discont
      p_disc[ds,:], p_disc_std[ds,:], tmp = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[np.where(ic[:,3]==0)[0]],N_bs,nbin)
      
      p_silent[ds,:], p_silent_std[ds,:], tmp = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[np.where(ic[:,2]==0)[0]],N_bs,nbin)
      
      
      if ((ds-1)%3==0) & (ds < 9):
        plt.figure(f_stable_pos.number)
        plt.subplot(1,3,int((ds-1)/3+1))
        fpos_ds = fpos_vals[idx_ds]
        #plt.bar(range(100),fpos_ds[np.where(ic[:,3]==1)])
        plt.hist(fpos_ds[np.where(ic[:,3]==1)],np.linspace(0,100,101),density=True,facecolor=[0.6,0.6,0.6])
        plt.xlabel('field pos. x',fontsize=14)
        if ds == 1:
          plt.ylabel('density',fontsize=14)
        else:
          plt.yticks([])
        plt.tight_layout()
        
      #print(p[ds,:])
      
      if ((ds-1)%3==0) & (ds < 9):
        
        plt.figure(f_shift_distr.number)
        plt.subplot(1,3,int((ds-1)/3+1))
        CI = np.percentile(shift_distr,[5,95],0)
        
        plt.bar(np.linspace(-49.5,49.5,100),shift_distr.mean(0),facecolor=[0.6,0.6,0.6])
        plt.errorbar(np.linspace(-49.5,49.5,100),shift_distr.mean(0),shift_distr.mean(0)-CI[0,:],CI[1,:]-shift_distr.mean(0),fmt='none',ecolor=[1,0.8,0.8],elinewidht=0.3)
        
        plt.plot(np.linspace(-50,50,101),F_shifts(np.linspace(-50,50,101),p[ds,0],p[ds,1],p[ds,2],p[ds,3]),'g',linewidth=2)
        
        #plt.bar(shifts_hist[1][:-1]+0.5,shifts_hist[0])
        #plt.plot(shifts_hist[1][:-1]+0.5,F_shifts(shifts_hist[1][:-1]+0.5,p[ds,0],p[ds,1],p[ds,2],p[ds,3]),'r--')
        #plt.bar(shifts_hist_certain[1][:-1]+0.5,shifts_hist_certain[0],facecolor='g',alpha=0.5)
        #plt.bar(shifts_hist_uncertain[1][:-1]+0.5,shifts_hist_uncertain[0],facecolor='r',alpha=0.5)
        
        #plt.plot(shifts_hist[1][:-1]+0.5,p_shift,'r')
        plt.xlim([-nbin/2,nbin/2])
        plt.ylim([0,0.08])
        plt.xlabel('field shift $\Delta$ x',fontsize=14)
        if ds==1:
          plt.ylabel('density',fontsize=14)
        else:
          plt.yticks([])
        plt.title('$\Delta$ s = %d'%ds)
        plt.tight_layout()
        #plt.show(block=False)
        #plt.ylim([0,200])#shifts_hist[0].max()*1.5])
    plt.show(block=False)
    if sv:
      plt.figure(f_shift_distr.number)
      pl_dat.save_fig('shift_distr')
      plt.figure(f_stable_pos.number)
      pl_dat.save_fig('stable_pos')
    
    def plot_shift_distr(p,p_std,p_ds0):
      nSes = p.shape[0]
      f,axs = plt.subplots(2,2,figsize=(6,4),sharex=True)
      axs[1][0].plot([0,nSes],[p_ds0[2],p_ds0[2]],linestyle='--',color=[0.6,0.6,0.6])
      axs[1][1].plot([0,nSes],[0,0],linestyle='--',color=[0.6,0.6,0.6])
      for i in range(4):
        pl_dat.plot_with_confidence(axs[int(np.floor(i/2))][i%2],range(nSes),p[:,i],p[:,i]+np.array([[-1],[1]])*p_std[:,i]*1.96,'k','--')
      axs[0][1].set_yscale('log')
      
      axs[0][1].yaxis.set_label_position("right")
      axs[0][1].yaxis.tick_right()

      axs[1][1].yaxis.set_label_position("right")
      axs[1][1].yaxis.tick_right()
      
      axs[0][0].set_xlim([0,max(20,nSes/2)])
      axs[0][0].set_ylim([0,1])
      axs[0][1].set_ylim([10**(-2),1])
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
    
    
    plot_shift_distr(p,p_std,p_ds0)
    if sv:
      pl_dat.save_fig('stability_dynamics')
    #plot_shift_distr(p_cont,p_cont_std,p_ds0)
    #if sv:
      #pl_dat.save_fig('stability_dynamics_cont')
    ##plot_shift_distr(p_mix,p_mix_std,p_ds0)
    #plot_shift_distr(p_disc,p_disc_std,p_ds0)
    #if sv:
      #pl_dat.save_fig('stability_dynamics_disc')
    
    #f,axs = plt.subplots(2,2,figsize=(6,4),sharex=True)
    plt.figure(figsize=(4,2))
    ax = plt.subplot(111)
    pl_dat.plot_with_confidence(ax,range(nSes),p_cont[:,1],p_cont[:,1]+np.array([[-1],[1]])*p_cont_std[:,1]*1.96,'b','--')
    pl_dat.plot_with_confidence(ax,range(nSes),p_disc[:,1],p_disc[:,1]+np.array([[-1],[1]])*p_cont_std[:,1]*1.96,'r','--')
    ax.set_yscale('log')
    ax.set_ylim([10**(-2),1.1])
    ax.set_xlim([0,20])
    ax.set_xlabel('$\Delta$ s',fontsize=14)
    ax.set_ylabel('$r_{stable}$',fontsize=14)
    plt.tight_layout()
    #plt.title('cont')
    plt.show(block=False)
    if sv:
      pl_dat.save_fig('stability_dynamics_cont_vs_disc')
    
    
    
    maxSes = 8
    print('what are those stable cells coding for?')
    plt.figure(figsize=(4,3))
    plt.bar(np.arange(maxSes)-0.3,p_cont[:maxSes,1],width=0.2,facecolor=[0.8,0.8,1],label='continuous')
    plt.errorbar(np.arange(maxSes)-0.3,p_cont[:maxSes,1],p_cont_std[:maxSes,1],fmt='none',ecolor='r')
    
    plt.bar(np.arange(maxSes)-0.1,p_mix[:maxSes,1],width=0.2,facecolor=[0.8,0.8,0.8],label='mixed')
    plt.errorbar(np.arange(maxSes)-0.1,p_mix[:maxSes,1],p_mix_std[:maxSes,1],fmt='none',ecolor='r')
    
    plt.bar(np.arange(maxSes)+0.1,p_disc[:maxSes,1],width=0.2,facecolor=[1,0.8,0.8],label='non-coding')
    plt.errorbar(np.arange(maxSes)+0.1,p_disc[:maxSes,1],p_disc_std[:maxSes,1],fmt='none',ecolor='r')
    
    plt.bar(np.arange(maxSes)+0.3,p_silent[:maxSes,1],width=0.2,facecolor=[0.8,1,0.8],label='silent')
    plt.errorbar(np.arange(maxSes)+0.3,p_silent[:maxSes,1],p_silent_std[:maxSes,1],fmt='none',ecolor='r')
    
    plt.xlabel('session diff. $\Delta s$',fontsize=14)
    plt.ylabel('$r_{stable}$',fontsize=14)
    plt.ylim([0,1])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=False)
    
    if sv:
      pl_dat.save_fig('intercoding_state')
    
    #return PC_match_certainty
    
    #idx_stable = np.where(~np.isnan(shifts))
    #plt.figure()
    #plt.hist(shifts[idx_stable],pl_dat.bin_edges,facecolor='r')
    #plt.show(block=True)
    #for s in range(10):
      #print('Session %d'%s)
      
      #idx_stable = np.where(~np.isnan(cluster.PCs['fields']['shift'][:,s,s+1]))[0]
      
      #plt.figure()
      ##plt.subplot(212)
      #plt.hist(cluster.PCs['fields']['parameter'][idx_stable,s,0,3,0],pl_dat.bin_edges)
      #plt.subplot(211)
      
      #idx_stable2 = np.where(shifts[:,s]<10)
      #plt.hist(cluster.PCs['fields']['parameter'][idx_stable2,s,0,3,0],pl_dat.bin_edges,facecolor='r')
      ##plt.hist(shifts[:,s],pl_dat.bin_edges,facecolor='r')
      #plt.show(block=True)
    ##print(idx_stable)
  
  if plot_fig[7]:
    
    
    s = 10
    idx_PCs = np.any(cluster.activity['status'][:,:,2:],-1)
    plt.figure(figsize=(4,2.5))
    plt.scatter(cluster.PCs['MI']['p_value'][idx_PCs[:,s],s],cluster.PCs['Bayes']['factor'][idx_PCs[:,s],s,0],color='r',s=5)
    plt.scatter(cluster.PCs['MI']['p_value'][~idx_PCs[:,s],s],cluster.PCs['Bayes']['factor'][~idx_PCs[:,s],s,0],color=[0.6,0.6,0.6],s=3)
    plt.xlabel('p-value (mutual information)',fontsize=14)
    plt.ylabel('log($Z_{PC}$) - log($Z_{nPC}$)',fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    if sv:
      pl_dat.save_fig('PC_choice_s=%d'%s)
  #return arrays,occupancy,ROI_recurr,N_pairs,N_norm,pop_overlap
  
  
  if plot_fig[8]:
    
    
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

  
  if plot_fig[9]:
    
    print('### Plotting matching score statistics ###')
    
    print('now add example how to calculate footprint correlation(?), sketch how to fill cost-matrix')
    
    s = 12
    
    D_ROIs = sp.spatial.distance.squareform(sp.spatial.distance.pdist(cluster.sessions['com'][:,s,:2]))
    np.fill_diagonal(D_ROIs,np.NaN)
    
    plt.figure(figsize=(12,6))
    props = dict(boxstyle='round', facecolor='w', alpha=0.8)
    
    ## plot ROIs from a single session
    idx_dense = np.where(np.sum(D_ROIs<30,1)==8)[0]
    #c = idx_dense[np.random.randint(len(idx_dense))]
    #n = int(cluster.IDs['neuronID'][c,s,1])
    n = 328
    c = np.where(cluster.IDs['neuronID'][:,s,1] == n)[0][0]
    n_close = cluster.IDs['neuronID'][D_ROIs[c,:]<50,s,1].astype('int')
    
    pathSession = pathcat([cluster.meta['pathMouse'],'Session%02d'%(s+1)])
    pathLoad = pathcat([pathSession,'results_OnACID.mat'])
    ld = loadmat(pathLoad)
    Cn = ld['Cn']
    A = ld['A'].toarray().reshape(cluster.meta['dims'][0],cluster.meta['dims'][1],-1)
    
    x = int(cluster.sessions['com'][c,s,0]-cluster.sessions['shift'][s,0])
    y = int(cluster.sessions['com'][c,s,1]-cluster.sessions['shift'][s,1])
    
    ax_ROIs1 = plt.subplot(241)
    margins = 30
    #margins = 10
    Cn_tmp = Cn[x-margins:x+margins,y-margins:y+margins]
    Cn -= Cn_tmp.min()
    Cn_tmp -= Cn_tmp.min()
    Cn /= Cn_tmp.max()
    
    ax_ROIs1.imshow(Cn,origin='lower',clim=[0,1])
    An = A[...,n]
    for nn in n_close:
      ax_ROIs1.contour(A[...,nn].T,[0.01*A[...,nn].max()],colors='w',linestyles='--',linewidths=1)
    ax_ROIs1.contour(An.T,[0.01*An.max()],colors='w',linewidths=3)
    #ax_ROIs1.plot(cluster.sessions['com'][c,s,0],cluster.sessions['com'][c,s,1],'kx')
    
    ax_ROIs1.set_xlim([x-margins,x+margins])
    ax_ROIs1.set_ylim([y-margins,y+margins])
    ax_ROIs1.text(x-margins+5,y+margins-10,'Session s',bbox=props,fontsize=10)
    
    D_ROIs_cross = sp.spatial.distance.cdist(cluster.sessions['com'][:,s,:2],cluster.sessions['com'][:,s+1,:2])
    n_close = cluster.IDs['neuronID'][D_ROIs_cross[c,:]<50,s+1,1].astype('int')
    
    pathSession = pathcat([cluster.meta['pathMouse'],'Session%02d'%(s+2)])
    pathLoad = pathcat([pathSession,'results_OnACID.mat'])
    ld = loadmat(pathLoad)
    A = ld['A'].toarray().reshape(cluster.meta['dims'][0],cluster.meta['dims'][1],-1)
    ## plot ROIs of session 2 compared to one of session 1
    ax_ROIs2 = plt.subplot(242)
    ax_ROIs2.imshow(Cn,origin='lower',clim=[0,1])
    n_match = int(cluster.IDs['neuronID'][c,s+1,1])
    for nn in n_close:
      if not (nn==n_match):
        ax_ROIs2.contour(A[...,nn].T,[0.01*A[...,nn].max()],colors='r',linestyles='--',linewidths=1)
    ax_ROIs2.contour(A[...,n_match].T,[0.01*A[...,n_match].max()],colors='g',linewidths=3)
    ax_ROIs2.contour(An.T,[0.01*An.max()],colors='w',linewidths=3)
    
    ax_ROIs2.set_xlim([x-margins,x+margins])
    ax_ROIs2.set_ylim([y-margins,y+margins])
    ax_ROIs2.text(x-margins+5,y+margins-10,'Session s+1',bbox=props,fontsize=10)

    
    
    ax_zoom1 = plt.subplot(245)
    ax_zoom1.hist(D_ROIs.flatten(),np.linspace(0,15,31),facecolor='k',density=True)
    ax_zoom1.set_xlabel('distance [$\mu$m]',fontsize=14)
    pl_dat.remove_frame(ax_zoom1,['top','left','right'])
    ax_zoom1.set_ylabel('density',fontsize=14)
    
    ax = plt.axes([0.05,0.35,0.075,0.125])
    plt.hist(D_ROIs.flatten(),np.linspace(0,np.sqrt(2*512**2),101),facecolor='k',density=True)
    ax.set_xlabel('d [$\mu$m]',fontsize=10)
    pl_dat.remove_frame(ax,['top','left','right'])
    
    D_matches = np.copy(D_ROIs_cross.diagonal())
    np.fill_diagonal(D_ROIs_cross,np.NaN)
    
    ax_zoom2 = plt.subplot(246)
    ax_zoom2.hist(D_ROIs_cross.flatten(),np.linspace(0,15,31),facecolor='r',alpha=0.5)
    ax_zoom2.hist(D_matches,np.linspace(0,15,31),facecolor='g',alpha=0.5)
    ax_zoom2.set_xlabel('distance [$\mu$m]',fontsize=14)
    pl_dat.remove_frame(ax_zoom2,['top','left','right'])
    
    ax = plt.axes([0.3,0.35,0.075,0.125])
    ax.hist(D_ROIs_cross.flatten(),np.linspace(0,np.sqrt(2*512**2),101),facecolor='r',alpha=0.5)
    ax.hist(D_matches,np.linspace(0,np.sqrt(2*512**2),101),facecolor='g',alpha=0.5)
    ax.set_xlabel('d [$\mu$m]',fontsize=10)
    pl_dat.remove_frame(ax,['top','left','right'])
    
    ax = plt.subplot(248)
    ax.plot([0,1],[0,1],'--',color='r')
    ax.scatter(cluster.sessions['match_score'][:,:,0],cluster.sessions['match_score'][:,:,1],s=1,color='k')
    ax.set_xlim([0.3,1])
    ax.set_ylim([-0.05,1])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('matched score',fontsize=14)
    ax.set_ylabel('2nd best score',fontsize=14)
    pl_dat.remove_frame(ax,['top'])
    
    ax = plt.subplot(244)
    #ax.hist(cluster.sessions['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='r',alpha=0.5)
    ax.hist(cluster.sessions['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,label='match score')
    pl_dat.remove_frame(ax,['left','right','top'])
    ax.yaxis.set_label_position("right")
    #ax.yaxis.tick_right()
    ax.set_xlim([0.3,1])
    ax.set_xticks([])
    ax.set_ylabel('density',fontsize=14)
    ax.legend(loc='upper left',fontsize=10)
    
    ax = plt.subplot(247)
    ax.hist(cluster.sessions['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,orientation='horizontal',label='2nd best score')
    #ax.hist(cluster.sessions['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5)
    pl_dat.remove_frame(ax,['left','bottom','top'])
    ax.set_ylim([-0.05,1])
    ax.set_xlim([1.2,0])
    ax.set_yticks([])
    ax.legend(loc='upper right',fontsize=10)
    ax.set_xlabel('density',fontsize=14)
    
    plt.tight_layout()
    plt.show(block=False)
    
    if sv:
      pl_dat.save_fig('match_stats')
  
  if plot_fig[10]:
    
    print('### Plotting session alignment procedure and statistics ###')
    
    s = 10
    s = s-1
    
    pathSession1 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_OnACID.mat'%1])
    ROIs1_ld = loadmat(pathSession1)
    
    pathSession2 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_OnACID.mat'%(s+1)])
    ROIs2_ld = loadmat(pathSession2)
    
    Cn = ROIs1_ld['Cn']
    Cn2 = ROIs2_ld['Cn']
    Cn -= Cn.min()
    Cn /= Cn.max()
    Cn2 -= Cn2.min()
    Cn2 /= Cn2.max()
    dims = Cn.shape
    
    x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32))
    
    print('adjust session position')
    C = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(Cn) * np.fft.fft2(np.rot90(Cn2,2)))))
    
    #/np.abs(np.fft.fft2(Cn) * np.fft.fft2(np.rot90(Cn2,2)))
    
    max_pos = np.where(C==np.max(C))
    x_shift = (max_pos[1] - (dims[1]/2-1))#.astype(int)
    y_shift = (max_pos[0] - (dims[0]/2-1))#.astype(int)
    print([x_shift,y_shift])
    print('shift by x,y: %5.3f,%5.3f'%(cluster.sessions['shift'][s,0],cluster.sessions['shift'][s,1]))
    x_remap = (x_grid - cluster.sessions['shift'][s,0]).astype(np.float32)
    y_remap = (y_grid - cluster.sessions['shift'][s,1]).astype(np.float32)
    
    
    Cn2_shift = cv2.remap(Cn2, x_remap, y_remap, cv2.INTER_NEAREST)
    
    Cn_norm = np.uint8(Cn*(Cn > 0)*255)
    Cn2_norm = np.uint8(Cn2_shift*(Cn2_shift > 0)*255)
    flow = cv2.calcOpticalFlowFarneback(np.uint8(Cn_norm*255),
                                        np.uint8(Cn2_norm*255),
                                        None,0.5,3,128,3,7,1.5,0)
    x_remap = (x_grid + flow[:,:,0]).astype(np.float32) 
    y_remap = (y_grid + flow[:,:,1]).astype(np.float32)
    
    
    x = np.hstack([np.ones((512,1)),np.arange(512).reshape(512,1)]) 
  
    W = sstats.norm.pdf(range(dims[0]),dims[0]/2,dims[0]/(1*1.96))
    #W = np.ones(512)
    W /= W.sum()
    W = np.sqrt(np.diag(W))
    x_w = np.dot(W,x)
    flow_w = np.dot(flow[:,:,1],W)
    x0,res,rank,tmp = np.linalg.lstsq(x_w,flow_w)
    
    d = -x0[0,:]/x0[1,:]
    W = np.sqrt(np.diag(1/res))
    r = sstats.linregress(range(dims[0]),d)
    
    tilt_ax = r.intercept+r.slope*range(512)
    
    #U = x_remap - x_grid
    #V = y_remap - y_grid
    
    Cn2_corr = cv2.remap(Cn2_shift.astype(np.float32), x_remap, y_remap, cv2.INTER_NEAREST)
    
    
    props = dict(boxstyle='round', facecolor='w', alpha=0.8)
    
    fig = plt.figure(figsize=(10,5))
    ax_im1 = plt.axes([0.15,0.625,0.175,0.35])
    im_col = np.zeros((512,512,3))
    im_col[:,:,0] = Cn2
    ax_im1.imshow(im_col,origin='lower')
    ax_im1.text(50,75,'Session %d'%s,bbox=props,fontsize=10)
    ax_im1.set_xticks([])
    ax_im1.set_yticks([])
    
    im_col = np.zeros((512,512,3))
    im_col[:,:,1] = Cn
    
    ax_im2 = plt.axes([0.1,0.575,0.175,0.35])
    ax_im2.imshow(im_col,origin='lower')
    ax_im2.text(50,75,'Session %d'%1,bbox=props,fontsize=10)
    #ax_im2.set_xticks([])
    #ax_im2.set_yticks([])
    ax_im2.set_xlabel('x [px]',fontsize=14)
    ax_im2.set_ylabel('y [px]',fontsize=14)
    
    ax_sShift = plt.axes([0.5,0.625,0.175,0.35])
    cbaxes = plt.axes([0.7, 0.625, 0.05, 0.02])
    C -= np.percentile(C,95)
    C /= C.max()
    im = ax_sShift.imshow(C,origin='lower',extent=[-dims[0]/2,dims[0]/2,-dims[1]/2,dims[1]/2],cmap='jet',clim=[0,1])
    cb = fig.colorbar(im,cax = cbaxes,orientation='horizontal')
    cb.set_ticks([0,1])
    cb.set_ticklabels(['low','high'])
    cb.set_label('corr.',fontsize=10)
    ax_sShift.arrow(0,0,float(cluster.sessions['shift'][s,0]),float(cluster.sessions['shift'][s,1]),head_width=1.5,head_length=2,color='k',width=0.1,length_includes_head=True)
    ax_sShift.text(-13, -13, 'shift: (%d,%d)'%(cluster.sessions['shift'][s,0],cluster.sessions['shift'][s,1]), size=10, ha='left', va='bottom',color='k',bbox=props)
    
    #ax_sShift.colorbar()
    ax_sShift.set_xlim([-15,15])
    ax_sShift.set_ylim([-15,15])
    ax_sShift.set_xlabel('x [px]',fontsize=14)
    ax_sShift.set_ylabel('y [px]',fontsize=14)
    
    
    ax_sShift_all = plt.axes([0.685,0.825,0.075,0.15])
    for ss in range(nSes):
      if cluster.boolSessions[ss]:
        ax_sShift_all.arrow(0,0,cluster.sessions['shift'][ss,0],cluster.sessions['shift'][ss,1],color=[0.6,0.6,0.6])
    ax_sShift_all.arrow(0,0,cluster.sessions['shift'][s,0],cluster.sessions['shift'][s,1],color='r')
    ax_sShift_all.yaxis.set_label_position("right")
    ax_sShift_all.yaxis.tick_right()
    ax_sShift_all.set_xlim([-20,20])
    ax_sShift_all.set_ylim([-20,20])
    ax_sShift_all.set_xlabel('x [px]',fontsize=10)
    ax_sShift_all.set_ylabel('y [px]',fontsize=10)
    
    
    idxes = 50
    tx = dims[0]/2 - 1
    ty = tilt_ax[int(tx)]
    ax_OptFlow = plt.axes([0.1,0.125,0.175,0.325])
    ax_OptFlow.quiver(x_grid[::idxes,::idxes], y_grid[::idxes,::idxes], flow[::idxes,::idxes,0], flow[::idxes,::idxes,1], angles='xy', scale_units='xy', scale=0.25, headwidth=4,headlength=4, width=0.002, units='width')#,label='x-y-shifts')
    ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),d,'r:')
    ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'r-')
    
    ### display angle on plot
    #idx_intersect = 400
    #slope_perp = -1/slope
    #slope_perp_theta = -(slope_perp+20)
    #intercept_perp = idx_intersect*slope_perp - tilt_ax[idx_intersect]
    #intercept_perp_theta = idx_intersect*slope_perp_theta - tilt_ax[idx_intersect]
    #tilt_ax_perp = slope_perp*range(512) - intercept_perp
    #tilt_ax_perp_theta = slope_perp_theta*range(512) - intercept_perp_theta
    #angle_perp = abs(math.degrees(math.atan(slope_perp)))
    #angle_perp_theta = abs(math.degrees(math.atan(slope_perp_theta)))
    #theta = angle_perp_theta - angle_perp
    #ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax_perp,'b-')
    #ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax_perp_theta,'b:')
    #ax_OptFlow.add_patch(Arc([idx_intersect,tilt_ax[idx_intersect]], 450, 450, 0, angle_perp, angle_perp_theta, color='b'))
    #ax_OptFlow.text(idx_intersect,tilt_ax[idx_intersect]+150,'$\\theta$',fontsize=10,color='b')
    
    ax_OptFlow.set_xlim([0,dims[0]])
    ax_OptFlow.set_ylim([0,dims[1]])
    ax_OptFlow.set_xlabel('x [px]',fontsize=14)
    ax_OptFlow.set_ylabel('y [px]',fontsize=14)
    #ax_OptFlow.legend(loc='lower left')
    #ax_OptFlow.set_title('optical flow')
    
    ax_OptFlow_stats = plt.axes([0.285,0.3,0.075,0.15])
    ax_OptFlow_stats.scatter(flow[:,:,0].reshape(-1,1),flow[:,:,1].reshape(-1,1),s=0.2,marker='.',color='k')#,label='xy-shifts')
    ax_OptFlow_stats.plot(np.mean(flow[:,:,0]),np.mean(flow[:,:,1]),marker='.',color='r')
    ax_OptFlow_stats.set_xlim(-10,10)
    ax_OptFlow_stats.set_ylim(-10,10)
    ax_OptFlow_stats.set_xlabel('$\Delta$x [px]',fontsize=10)
    ax_OptFlow_stats.set_ylabel('$\Delta$y [px]',fontsize=10)
    ax_OptFlow_stats.yaxis.set_label_position("right")
    ax_OptFlow_stats.yaxis.tick_right()
    #ax_OptFlow_stats.legend()
    
    
    dist_mat = np.abs((r.slope*x_grid-y_grid+r.intercept)/np.sqrt(r.slope**2+1**2))
    slope_normal = np.array([-r.slope,1])
    slope_normal /= np.linalg.norm(slope_normal)
    f_perp = np.dot(flow[:,:,:2],slope_normal)
    h_dat = np.sign(f_perp)*np.sin(np.arccos((dist_mat - np.abs(f_perp))/dist_mat))*dist_mat
    
    ax = plt.axes([0.575,0.125,0.175,0.35])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    im = ax.imshow(h_dat,origin='lower',cmap='jet',clim=[-60,60])
    
    
    cbaxes = plt.axes([0.548, 0.3, 0.01, 0.175])
    cb = plt.colorbar(im,cax = cbaxes)
    cbaxes.yaxis.set_label_position('left')
    cbaxes.yaxis.set_ticks_position('left')
    cb.set_label('z [$\mu$m]',fontsize=10)
    
    
    angles = np.arccos(cluster.sessions['rotation_normal'])/(2*np.pi)*360
    ax_tilt = plt.axes([0.495,0.125,0.075,0.15])
    #ax_tilt.errorbar(range(nSes),cluster.PCs['mic_axis'][:,0],cluster.PCs['mic_axis'][:,1],fmt='.',color='b',markersize=1)
    #ax_tilt.errorbar(s,cluster.PCs['mic_axis'][s,0],cluster.PCs['mic_axis'][s,1],fmt='.',color='r',markersize=5)
    ax_tilt.plot([0,cluster.nSes],[0,0],color=[0.6,0.6,0.6],linestyle='--')
    ax_tilt.plot(np.where(cluster.boolSessions)[0],90-angles[cluster.boolSessions,0],'k')
    ax_tilt.plot(np.where(cluster.boolSessions)[0],90-angles[cluster.boolSessions,1],'b')
    ax_tilt.plot(np.where(cluster.boolSessions)[0],angles[cluster.boolSessions,2],'r')
    #ax_tilt.yaxis.set_label_position("right")
    #ax_tilt.yaxis.tick_right()
    ax_tilt.set_ylim([-5,20])
    ax_tilt.set_xlabel('session',fontsize=10)
    ax_tilt.set_ylabel('$\phi$',fontsize=10)
    
    #ax_tilt2 = plt.axes([0.6,0.3,0.1,0.15])
    #ax_tilt2.errorbar(range(nSes),cluster.PCs['mic_theta'][:,0],cluster.PCs['mic_theta'][:,1],fmt='.',color='b',markersize=1)
    #ax_tilt2.errorbar(s,cluster.PCs['mic_theta'][s,0],cluster.PCs['mic_theta'][s,1],fmt='.',color='r',markersize=5)
    #ax_tilt2.yaxis.set_label_position("right")
    #ax_tilt2.yaxis.tick_right()
    #ax_tilt2.set_xticks([])
    ##ax_tilt2.set_xlabel('session',fontsize=10)
    #ax_tilt2.set_ylabel('$\\theta$',fontsize=10)
    
    
    ax_sShifted = plt.axes([0.8,0.125,0.175,0.35])
    #ax_sShifted = plt.subplot(111)
    im_col = np.zeros((512,512,3))
    im_col[:,:,0] = Cn
    im_col[:,:,1] = Cn2_corr
    ax_sShifted.imshow(im_col,origin='lower')
    ax_sShifted.text(50,75,'aligned sessions',bbox=props,fontsize=10)
    ax_sShifted.set_xticks([])
    ax_sShifted.set_yticks([])
    
    plt.tight_layout()
    plt.show(block=False)
    if sv:
      pl_dat.save_fig('session_align')
    
  if plot_fig[11]:
    print('### Plotting ROI and cluster statistics of matching ###')
    
    print('calculate ROI position from corrected imaging plane')
    
    ax = plt.subplot(243)
    ax.hist(np.nanmin(cluster.sessions['match_score'][...,0],1),np.linspace(0,1,101),facecolor='r',alpha=0.5,density=True,label='minimal score')
    ax.hist(np.nanmean(cluster.sessions['match_score'][...,0],1),np.linspace(0,1,101),facecolor='b',alpha=0.5,density=True,label='average score')
    ax.set_xlim([0.3,1])
    ax.set_xlabel('matching score',fontsize=14)
    ax.set_ylabel('density',fontsize=14)
    ax.legend(loc='upper left',fontsize=10)
    
    
    idx_unsure = cluster.sessions['match_score'][...,0]<(cluster.sessions['match_score'][...,1]+0.0)
    
    fig = plt.figure(figsize=(10,6))
    
    nDisp = 10
    ax_3D = plt.subplot(221,projection='3d')
    ##fig.gca(projection='3d')
    #a = np.arange(30)
    #for c in range(30):
    ax_3D.plot(cluster.sessions['com'][:nDisp,:,0].T*cluster.para['pxtomu'],cluster.sessions['com'][:nDisp,:,1].T,np.arange(nSes)*cluster.para['pxtomu'],'k')
    ax_3D.set_xlim([0,512*cluster.para['pxtomu']])
    ax_3D.set_ylim([0,512*cluster.para['pxtomu']])
    
    ax_3D.set_xlabel('x [$\mu$m]',fontsize=14)
    ax_3D.set_ylabel('y [$\mu$m]',fontsize=14)
    ax_3D.invert_zaxis()
    ax_3D.set_zlabel('session',fontsize=14)
    
    #ax = plt.subplot(243)
    ax = plt.axes([0.6,0.65,0.175,0.325])
    dx = np.diff(cluster.sessions['com'][...,0],axis=1)*cluster.para['pxtomu']
    ax.hist(dx.flatten(),np.linspace(-10,10,101),facecolor='k',alpha=0.5)
    ax.hist(dx[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='r',alpha=0.5)
    ax.set_xlabel('$\Delta$x [$\mu$m]',fontsize=14)
    ax.set_ylabel('density',fontsize=14)
    pl_dat.remove_frame(ax,['top','left','right'])
    ax.set_ylim([0,6000])
    
    #ax = plt.subplot(244)
    ax = plt.axes([0.8,0.65,0.175,0.325])
    dy = np.diff(cluster.sessions['com'][...,1],axis=1)*cluster.para['pxtomu']
    ax.hist(dy.flatten(),np.linspace(-10,10,101),facecolor='k',alpha=0.5)
    ax.hist(dy[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='r',alpha=0.5)
    ax.set_xlabel('$\Delta$y [$\mu$m]',fontsize=14)
    pl_dat.remove_frame(ax,['top','left','right'])
    ax.set_ylim([0,6000])
    
    ax = plt.axes([0.7,0.875,0.075,0.1])
    ax.hist(dx.flatten(),np.linspace(-10,10,101),facecolor='k',alpha=0.5)
    ax.hist(dx[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='r',alpha=0.5)
    ax.set_xlabel('$\Delta$x [$\mu$m]',fontsize=10)
    pl_dat.remove_frame(ax,['top','left','right'])
    ax.set_ylim([0,50])
    
    ax = plt.axes([0.9,0.875,0.075,0.1])
    ax.hist(dy.flatten(),np.linspace(-10,10,101),facecolor='k',alpha=0.5)
    ax.hist(dy[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='r',alpha=0.5)
    ax.set_xlabel('$\Delta$y [$\mu$m]',fontsize=10)
    pl_dat.remove_frame(ax,['top','left','right'])
    ax.set_ylim([0,50])
    
    
    ax_mv = plt.subplot(223)
    ROI_diff = (cluster.sessions['com'].transpose(1,0,2)-cluster.sessions['com'][:,0,:]).transpose(1,0,2)*cluster.para['pxtomu']
    ROI_diff_abs = np.array([np.sqrt(x[:,0]**2+x[:,1]**2) for x in ROI_diff])
    for n in range(nDisp):
      ax_mv.plot(range(nSes),ROI_diff_abs[n,:],linewidth=0.5,color=[0.6,0.6,0.6])
    
    pl_dat.plot_with_confidence(ax_mv,range(nSes),np.nanmean(ROI_diff_abs,0),np.nanstd(ROI_diff_abs,0),col='r',ls='-')
    ax_mv.set_xlabel('session',fontsize=14)
    ax_mv.set_ylabel('$\Delta$d [$\mu$m]',fontsize=14)
    pl_dat.remove_frame(ax_mv,['top','right'])
    
    idx_c_unsure = idx_unsure.any(1)
    
    ax_mv_max = plt.subplot(224)
    ROI_max_mv = np.nanmax(ROI_diff_abs,1)
    ax_mv_max.hist(ROI_max_mv,np.linspace(0,40,41),facecolor='k')
    ax_mv_max.hist(ROI_max_mv[idx_c_unsure],np.linspace(0,40,41),facecolor='r')
    ax_mv_max.set_xlabel('max($\Delta$d) [$\mu$m]',fontsize=14)
    ax_mv_max.set_ylabel('# cluster',fontsize=14)
    pl_dat.remove_frame(ax_mv_max,['top','right'])
    
    plt.tight_layout()
    plt.show(block=False)
    
  
  if plot_fig[12]:
    print('### Plotting positions of silent neurons ### ')
    
    ## get average position of neuron
    com_mean = np.nanmean(cluster.sessions['com'][cluster.clusters_bool,:,:],1)
    #idx_keep = np.all((com_mean[:,:2] < 512) & (com_mean[:,:2]>0),1)
    #print(idx_keep.shape)
    #print(com_mean[~idx_keep,:])
    
    
    
    #nA = 10 ## number zones per row
    #N = com_mean.shape[0]
    #NA_exp = N/nA**2
    #A = np.histogram2d(com_mean[cluster.clusters_bool,0],com_mean[cluster.clusters_bool,1],np.linspace(0,512,nA+1))[0]
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
      com_silent = com_mean[cluster.activity['status'][cluster.clusters_bool,s,0],:]
      com_active = com_mean[cluster.activity['status'][cluster.clusters_bool,s,1],:]
      com_PCs = com_mean[np.any(cluster.activity['status'][cluster.clusters_bool,s,2:],-1),:]
      
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
      
      #NA = cluster.activity['status'][
      
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
      #p_silent = np.histogram(height_silent,np.linspace(0,80,101),density=True)[0]
      #p_active = np.histogram(cluster.sessions['com'][cluster.activity['status'][:,s,1],s,2],np.linspace(0,80,101),density=True)[0]
      #KS_dist[s] = KS_test(p_silent,p_active)
      #plt.title('Session %d'%(s+1))
    #plt.plot(KS_dist)
    
    ax_p = plt.subplot(224)
    ax_p.plot([0,cluster.meta['nSes']],[0.1,0.1],'k--')
    ax_p.plot(np.where(cluster.boolSessions)[0],p_vals[cluster.boolSessions,0],'r')
    ax_p.plot(np.where(cluster.boolSessions)[0],p_vals[cluster.boolSessions,1],'b')
    #ax_p.plot(np.where(cluster.boolSessions)[0],p_vals[cluster.boolSessions,2],'--',color=[0.6,0.6,0.6])
    #ax_p.plot(np.where(cluster.boolSessions)[0],p_vals[cluster.boolSessions,3],'g--')
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
    
    nDisp = 100
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    
    for n in range(nDisp):
      sc = ax.scatter(cluster.sessions['com'][n,::5,0],cluster.sessions['com'][n,::5,1],cluster.sessions['com'][n,::5,2],c=range(0,nSes,5),cmap='jet')
    plt.colorbar(sc)
    
    plt.show(block=False)
  
  #if plot_fig[14]:
    
    ## load results from manual matching
    

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
    self.bin_edges = np.linspace(1,para['nbin'],para['nbin'])
    
    self.bars = {}
    self.bars['PC'] = np.zeros(para['nbin'])
    self.bars['PC'][para['zone_mask']['others']] = 1
    
    self.bars['GT'] = np.zeros(para['nbin']);
    
    if np.count_nonzero(para['zone_mask']['gate'])>1:
      self.bars['GT'][para['zone_mask']['gate']] = 1
    
    self.bars['RW'] = np.zeros(para['nbin']);
    self.bars['RW'][para['zone_mask']['reward']] = 1
    
    
    ### build blue-red colormap
    #n = 51;   ## must be an even number
    #cm = ones(3,n);
    #cm(1,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## red
    #cm(2,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## green
    #cm(2,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## green
    #cm(3,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## blue
  
  def remove_frame(self,ax,positions=None):
    
    if positions is None:
      positions = ['left','right','top','bottom']
    
    for p in positions:
      ax.spines[p].set_visible(False)
    
    if 'left' in positions:
      ax.set_yticks([])
    
    if 'bottom' in positions:
      ax.set_xticks([])
    
  
  def plot_with_confidence(self,ax,x_data,y_data,CI,col='k',ls='-'):
    
    col_fill = np.minimum(np.array(colors.to_rgb(col))+np.ones(3)*0.3,1)
    if len(CI.shape) > 1:
      ax.fill_between(x_data,CI[0,:],CI[1,:],color=col_fill,alpha=0.2)
    else:
      ax.fill_between(x_data,y_data-CI,y_data+CI,color=col_fill,alpha=0.2)
    ax.plot(x_data,y_data,color=col,linestyle=ls)
    
    
    
  def save_fig(self,fig_name,fig_pos=None):
    path = pathcat([self.pathFigures,'m%s_%s%s.%s'%(self.mouse,fig_name,self.sv_opt['suffix'],self.sv_opt['ext'])]);
    plt.savefig(path,format=self.sv_opt['ext'],dpi=self.sv_opt['dpi'])
    print('Figure saved as %s'%path)
    
    

def bootstrap_shifts(fun,shifts,N_bs,nbin):
  
  N_data = len(shifts)
  if N_data == 0:
    return np.zeros(4)*np.NaN,np.zeros(4)*np.NaN,np.zeros((2,nbin))*np.NaN
  
  samples = np.random.randint(0,N_data,(N_bs,N_data))
  sample_randval = np.random.rand(N_bs,N_data)
  shift_distr_bs = np.zeros((N_bs,nbin))
  par = np.zeros((N_bs,4))*np.NaN
  for i in range(N_bs):
    x1 = np.argmin(abs(shifts[samples[i,:],:]-sample_randval[i,:,np.newaxis]),1)-nbin/2
    
    shift_distr_bs[i,:] = np.histogram(x1,np.linspace(-50,50,101),density=True)[0]
    par[i,:],p_cov = fun(x1)
    
  p = np.nanmean(par,0)
  p_std = np.nanstd(par,0)
  
  return p,p_std,shift_distr_bs
