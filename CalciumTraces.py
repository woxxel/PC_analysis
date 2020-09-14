import sys, scipy, time
import numpy as np
from scipy.io import loadmat
from scipy.stats import norm
from scipy import fftpack
import scipy.sparse
from tqdm import *
import matplotlib.pyplot as plt

#sys.path.append('/home/wollex/Data/Science/PhD/Programs/PC_analysis')
from utils import pathcat, find_modes, normalize_sparse_array, _hsm


class CalciumTraces:

  def __init__(self,basePath=None,mouse=None,s=None):

    self.f = 15  ## recording rate in Hz

    if not (basePath is None):
      pathMouse = pathcat([basePath,mouse])
      self.pathSession = pathcat([pathMouse,'Session%02d'%s])

    self.data = {}

  def plot_from_input(self,C,S,SNR=None,r_values=None,n=None,sig=2):
    self.load_data(C,S,SNR,r_values)
    self.process_data()
    self.plot_trace(n=n,sig=sig)
    self.plot_stats(SNR_thr=4,r_thr=0,sig=sig)

  def load_data(self,C=None,S=None,SNR=None,r_values=None):
    ## do I need noise level for either method? (rerun redetection??)
    if C is None:
      pathResults = pathcat([self.pathSession,'results_redetect.mat'])
      ld = loadmat(pathResults,variable_names=['C','S','A','b','f','SNR','r_values','CNN'],squeeze_me=True)
      for key in ld.keys():
        self.data[key] = ld[key]
      self.data['C'] -= self.data['C'].min(1)[:,np.newaxis]
      self.data['S'][self.data['S']<0] = 0
    else:
      self.data['C'] = C
      self.data['S'] = S
      self.data['C'] -= self.data['C'].min(1)[:,np.newaxis]
      self.data['S'][self.data['S']<0] = 0
      self.data['SNR'] = SNR
      self.data['r_values'] = r_values

    #self.data['A'] = normalize_sparse_array(self.data['A'])
    self.nCells, self.T = self.data['C'].shape
    ## plot results

  def process_data(self):

    self.baseline_C = np.median(self.data['C'],1)
    trace = self.data['C']-self.baseline_C[:,np.newaxis]
    trace = -trace * (trace <= 0)
    N_s = (trace>0).sum(1)
    self.noise_C = np.sqrt((trace**2).sum(1)/(N_s*(1-2/np.pi)))

    self.baseline_S = np.zeros(self.nCells)
    self.noise_S = np.zeros(self.nCells)

    for n in range(self.nCells):
      trace = self.data['S'][n,self.data['S'][n,:]>0]
      self.baseline_S[n] = np.median(trace)
      trace -= self.baseline_S[n]
      trace *= -1*(trace <= 0)
      N_s = (trace>0).sum()
      self.noise_S[n] = np.sqrt((trace**2).sum()/(N_s*(1-2/np.pi)))


  def plot_trace(self,n,sig=2,SNR_thr=2):

    self.baseline_C_med = np.zeros(self.nCells)
    self.baseline_C_hsm = np.zeros(self.nCells)

    self.baseline_S_med = np.zeros(self.nCells)
    self.baseline_S_hsm = np.zeros(self.nCells)

    self.baseline_C_med[n] = np.median(self.data['C'][n,:])
    self.baseline_C_hsm[n] = _hsm(np.sort(self.data['C'][n,:]))

    S = self.data['S'][n,self.data['S'][n,:]>0]
    self.baseline_S_med[n] = np.median(S)
    self.baseline_S_hsm[n] = _hsm(np.sort(S))

    #data_thr_C = self.data_thr_C[:,0] + sig*self.data_thr_C[:,1]

    #self.N_spikes_C = np.floor(self.data['C'] / (self.baseline_C[:,np.newaxis] + sig*self.noise_C[:,np.newaxis])).sum(1)
    #self.N_spikes_S = np.floor(self.data['S'] / (self.baseline_S[:,np.newaxis] + sig*self.noise_S[:,np.newaxis])).sum(1)

    #self.N_spikes_S = np.zeros(self.nCells)*np.NaN
    #data_thr_S = self.data_thr_S[:,0] + sig*self.data_thr_S[:,1]
    #self.N_spikes_S = np.floor(np.maximum(0,self.data['S'] / data_thr_S[:,np.newaxis])).sum(1)
    #return N_spikes_C, N_spikes_S
    nCells,T = self.data['C'].shape
    t_arr = np.linspace(0,T/self.f,T)

    plt.figure()
    ax1 = plt.subplot(221)
    ax1.plot(t_arr,self.data['C'][n,:],'k')
    ax1.plot(t_arr,np.ones(T)*self.baseline_C_med[n],'g--')
    ax1.plot(t_arr,np.ones(T)*self.baseline_C_hsm[n],'r--')
    ax1.plot(t_arr,np.ones(T)*self.baseline_C[n]+sig*self.noise_C[n],'g')

    ax = plt.subplot(222)
    ax.plot(np.sort(self.data['C'][n,:]))
    ax.plot(np.ones(self.T)*self.baseline_C_med[n],'g--')
    ax.plot(np.ones(self.T)*self.baseline_C_hsm[n],'r--')
    #ax.plot(np.ones(self.T)*self.data_thr_C[n,0],'r')

    #print(N_spikes_S.sum())
    ax2 = plt.subplot(223,sharex=ax1)
    ax2.plot(t_arr,self.data['S'][n,:],'r')
    ax2.plot(t_arr,np.ones(T)*self.baseline_S_med[n],'g--')
    ax2.plot(t_arr,np.ones(T)*self.baseline_S_hsm[n],'r--')
    ax2.plot(t_arr,np.ones(T)*self.baseline_S[n]+sig*self.noise_S[n],'g')

    #ax2.plot(t_arr,self.df_f[n,:],'r')
    #ax2.plot(t_arr,np.ones(T)*self.data_thr_C[n,0],'g--')
    #ax2.plot(t_arr,np.ones(T)*data_thr_S[n],'g')

    ax = plt.subplot(224)
    ax.plot(np.sort(self.data['S'][n,:]),'r')
    ax.plot(np.ones(T)*self.baseline_S_med[n],'g--')
    ax.plot(np.ones(T)*self.baseline_S_hsm[n],'r--')
    ax.plot(np.ones(T)*self.baseline_S[n]+sig*self.noise_S[n],'g')

    #ax = plt.subplot(426)
    #ax.plot(np.cumsum(np.diff(np.sort(self.data['C'][n,:])))/np.arange(self.T)[1:])

    #ax.plot(np.sort(self.data['S'][n,:]))

    plt.show(block=True)

  def plot_stats(self,sig=2,SNR_thr=2,r_thr=0):

    self.N_spikes_C = np.floor(self.data['C'] / (self.baseline_C[:,np.newaxis] + sig*self.noise_C[:,np.newaxis])).sum(1)

    self.N_spikes_S = np.floor(self.data['S'] / (self.baseline_S[:,np.newaxis] + sig*self.noise_S[:,np.newaxis])).sum(1)

    idxes = (self.data['SNR']>SNR_thr) & (self.data['r_values']>r_thr)

    T = self.T
    plt.figure()
    plt.subplot(221)
    plt.hist(self.N_spikes_C[idxes]/(T/self.f),np.linspace(0,2,101),facecolor='g',alpha=0.5)
    plt.hist(self.N_spikes_C[~idxes]/(T/self.f),np.linspace(0,2,101),facecolor='r',alpha=0.5)
    plt.xlabel('firing rate (C)')
    #plt.subplot(223)
    #plt.hist(self.N_spikes_S/(T/self.f),np.linspace(0,10,101),facecolor='r',alpha=0.5)
    #plt.xlabel('firing rate (S)')
    plt.subplot(222)
    plt.scatter(self.data['SNR'][idxes],self.N_spikes_S[idxes]/(T/self.f),c='g')
    plt.scatter(self.data['SNR'][~idxes],self.N_spikes_S[~idxes]/(T/self.f),c='r')
    #plt.ylim([0,10])

    plt.subplot(223)
    plt.scatter(self.data['r_values'][idxes],self.N_spikes_S[idxes]/(T/self.f),c='g')
    plt.scatter(self.data['r_values'][~idxes],self.N_spikes_S[~idxes]/(T/self.f),c='r')
    #plt.ylim([0,10])

    plt.subplot(224)
    plt.hist(self.N_spikes_S[idxes]/(T/self.f),np.linspace(0,2,101),facecolor='g',alpha=0.5)
    plt.hist(self.N_spikes_S[~idxes]/(T/self.f),np.linspace(0,2,101),facecolor='r',alpha=0.5)
    plt.xlabel('firing rate (S)')

    plt.show(block=False)

    #plt.figure()
    #plt.plot([0,1000],[0,1000],'k--')
    #plt.scatter(self.data['SNR'],self.SNR_C)
    #plt.xlim([0,40])
    #plt.ylim([0,40])
    #plt.show(block=False)

  def plot_thesis(self,n,sig=[2,3,4],SNR_thr=2,r_thr=0):

    print('plotting for neurons %d and %d'%(n[0],n[1]))

    # n=[0,2452] ## in session 10 of mouse 762

    self.nCells, self.T = self.data['C'].shape
    T=self.T
    t_arr = np.linspace(0,self.T/self.f,self.T)

    plt.rcParams['font.size'] = 12
    idxes = (self.data['SNR']>SNR_thr) & (self.data['r_values']>r_thr)

    zlen=500
    zstart=1500
    zrange=range(zstart,(zstart+zlen))
    fig = plt.figure(figsize=(7,5),dpi=150)
    col = np.array([1,0,0])
    ax_C_zoom = plt.axes([0.075,0.7,0.2,0.25])
    add_number(fig,ax_C_zoom,order=1,offset=[-50,0])

    ax_C_zoom = plt.axes([0.075+0.29,0.7,0.2,0.25])
    add_number(fig,ax_C_zoom,order=2,offset=[-50,0])
    for i in range(2):
      ax_C_zoom = plt.axes([0.075+(0.29*i),0.7,0.2,0.25])
      ax_C_zoom.plot(t_arr[zrange],self.data['C'][n[i],zrange],'k',linewidth=0.5)
      ax_C_zoom.set_ylim([0,10*self.baseline_C[n[i]]])
      ax_C_zoom.set_xlim([t_arr[zstart],t_arr[zstart+zlen]])
      ax_C_zoom.set_xticks([])
      ax_C_zoom.set_yticks([])
      ax_C_zoom.spines['top'].set_visible(False)
      ax_C_zoom.spines['right'].set_visible(False)
      if i==0:
        ax_C_zoom.set_ylabel('C (a.u.)')

      ax_C = plt.axes([0.1+(0.29*i),0.85,0.125,0.1])
      ax_C.plot(t_arr,self.data['C'][n[i],:],'k',linewidth=0.5)
      ax_C.set_yticks([])
      ax_C.set_xlim([0,self.T/self.f])
      ax_C.spines['top'].set_visible(False)
      ax_C.spines['right'].set_visible(False)

        #print(self.T)

      ax_S_zoom = plt.axes([0.075+(0.29*i),0.425,0.2,0.275])
      y_arr = np.linspace(0,self.baseline_S[n[i]]+4*self.noise_S[n[i]],100)
      x1 = norm.pdf(y_arr,loc=self.baseline_S[n[i]],scale=self.noise_S[n[i]])
      x1 = x1/x1.max()*5+t_arr[zrange[-1]]-5
      x2 = t_arr[zrange[-1]]*np.ones(100)-5

      #plt.plot(x_offset,A_0,'ko')
      ax_S_zoom.fill_betweenx(y_arr,x1,x2,facecolor='r',alpha=1,edgecolor=None)
      ax_S_zoom.vlines(t_arr[zrange],np.zeros(zlen),self.data['S'][n[i],zrange],colors='k',linewidth=0.5)
      #ax_S_zoom.bar(t_arr,self.data['S'][n[i],:],width=1/self.f,facecolor='r')
      ax_S_zoom.plot(t_arr[zrange],np.ones(zlen)*self.baseline_S[n[i]],'r-',linewidth=0.8)
      for j,s in enumerate(np.flipud(sig)):
        ax_S_zoom.plot(t_arr[zrange],np.ones(zlen)*self.baseline_S[n[i]]+s*self.noise_S[n[i]],color=np.array([1,0.3*(2-j),0.3*(2-j)]),linestyle='--',label='$\\theta_S = %d$'%s,linewidth=0.8)

      ax_S_zoom.set_ylim([0,10*self.baseline_S[n[i]]])
      ax_S_zoom.set_xlabel('time in s')
      ax_S_zoom.set_yticks([])
      ax_S_zoom.set_xticks(np.linspace(0,600,61))
      ax_S_zoom.set_xlim([t_arr[zstart],t_arr[zstart+zlen]])
      ax_S_zoom.spines['top'].set_visible(False)
      ax_S_zoom.spines['right'].set_visible(False)

      if i==0:
        ax_S_zoom.set_ylabel('S (a.u.)')

        ax_S = plt.axes([0.1+(0.29*i),0.575,0.125,0.1])
        ax_S.vlines(t_arr,np.zeros(T),self.data['S'][n[i],:],colors='k',linewidth=0.5)
        ax_S.set_yticks([])
        ax_S.set_xlim([0,self.T/self.f])
        ax_S.spines['top'].set_visible(False)
        ax_S.spines['right'].set_visible(False)
      else:
        ax_S_zoom.legend(loc='upper right',bbox_to_anchor=[1,1],fontsize=8)


    xlim_arr = [2,1,0.5]
    ax_fr = plt.axes([0.075,0.125,0.175,0.15])
    add_number(fig,ax_fr,order=3,offset=[-50,10])
    for j,s in enumerate(sig):
      self.N_spikes_S = np.floor(self.data['S'] / (self.baseline_S[:,np.newaxis] + s*self.noise_S[:,np.newaxis])).sum(1)

      ax_fr = plt.axes([0.075+0.22*j,0.125,0.175,0.15])
      ax_fr.hist(self.N_spikes_S[idxes]/(T/self.f),np.linspace(0,xlim_arr[j],51),facecolor='tab:blue',alpha=0.5,label='SNR$\geq$%d'%SNR_thr)
      ax_fr.hist(self.N_spikes_S[~idxes]/(T/self.f),np.linspace(0,xlim_arr[j],51),facecolor='tab:red',alpha=0.5,label='SNR<%d'%SNR_thr)
      #ax_fr.bar(0,0,color=np.array([1,0.3*j,0.3*j]),label='$\\theta_S = %d$'%s)
      ax_fr.set_xlim([0,xlim_arr[j]])
      ax_fr.set_yticks([])
      _,ymax = ax_fr.get_ylim()
      if j>0:
          ax_fr.text(xlim_arr[j]*0.5,ymax*0.8,'$\\theta_S = %d$'%s,fontsize=10)

      if j==0:
          ax_fr.text(0.2,ymax*0.8,'$\\theta_S = %d$'%s,fontsize=10)
          ax_fr.set_ylabel('count')
      elif j==1:
          ax_fr.set_xlabel('$\\bar{\\nu}$ [Hz]')

      ax_fr.spines['top'].set_visible(False)
      ax_fr.spines['right'].set_visible(False)

    Ns = (self.data['S']>0).sum(1)

    s_adapt = norm.ppf((1-0.01)**(1/Ns))
    self.N_spikes_S = np.floor(self.data['S'] / (self.baseline_S[:,np.newaxis] + s_adapt[:,np.newaxis]*self.noise_S[:,np.newaxis])).sum(1)

    ax_fr = plt.axes([0.75,0.125,0.175,0.15])
    ax_fr.hist(self.N_spikes_S[idxes]/(T/self.f),np.linspace(0,(0.5)**j,51),facecolor='tab:blue',alpha=0.5,label='SNR$\geq$%d'%SNR_thr)
    ax_fr.hist(self.N_spikes_S[~idxes]/(T/self.f),np.linspace(0,(0.5)**j,51),facecolor='tab:red',alpha=0.5,label='SNR<%d'%SNR_thr)
    #ax_fr.bar(0,0,color=np.array([1,0.3*j,0.3*j]),label='$\\theta_S = %d$'%s)
    ax_fr.set_xlim([0,0.5**j])
    ax_fr.set_xlabel('$\\bar{\\nu}$ [Hz]')
    ax_fr.set_yticks([])
    if j==0:
      ax_fr.set_ylabel('count')
    ax_fr.legend(loc='lower right',bbox_to_anchor=[1.2,0.4],fontsize=10)
    ax_fr.spines['top'].set_visible(False)
    ax_fr.spines['right'].set_visible(False)
    _,ymax = ax_fr.get_ylim()
    #ax_fr.text(0.5**j*0.5,ymax*0.8,'$\\theta_S$ adapti'%s,fontsize=10)
    #ax_fr.text(0.2**j,ymax*0.9,'$\\theta_S = %d$'%s,fontsize=10)

    ax_theory = plt.axes([0.725,0.75,0.25,0.2])
    add_number(fig,ax_theory,order=4,offset=[-50,10])
    x_arr = np.linspace(1,10,101)
    for j,t in enumerate([500,2000,10000]):
      ax_theory.plot(x_arr,1-np.exp(t*np.log(norm.cdf(x_arr))),color=np.array([0.3*j,0.3*j,0.3*j]),label='T=%d'%t)
    for j,s in enumerate(sig):
      ax_theory.plot(s,1.1,'v',color=np.array([1,0.3*j,0.3*j]))

    ax_theory.legend(fontsize=10,loc='upper right',bbox_to_anchor=[1.1,1.1])
    ax_theory.set_xlabel('x')
    ax_theory.set_ylabel('$p(M_T > x$)')
    ax_theory.spines['top'].set_visible(False)
    ax_theory.spines['right'].set_visible(False)

    ax_theory2 = plt.axes([0.725,0.425,0.25,0.2])
    add_number(fig,ax_theory2,order=5,offset=[-50,10])
    T_arr = np.linspace(0,10000,1000)
    plt.plot(T_arr,norm.ppf((1-0.1)**(1/T_arr)),'k')
    plt.xlabel('$T$')
    plt.ylabel('$\\theta_S$')

    plt.tight_layout()
    plt.show(block=False)

    svPath = pathcat([self.pathSession,'get_firingrates.png'])
    plt.savefig(svPath,format='png',dpi=150)

def add_number(fig,ax,order=1,offset=None):

    # offset = [-175,50] if offset is None else offset
    offset = [-75,25] if offset is None else offset
    pos = fig.transFigure.transform(plt.get(ax,'position'))
    x = pos[0,0]+offset[0]
    y = pos[1,1]+offset[1]
    ax.text(x=x,y=y,s='%s)'%chr(96+order),ha='center',va='center',transform=None,weight='bold',fontsize=14)
