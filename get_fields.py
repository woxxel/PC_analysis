import os, random, h5py, time, math, cmath, copy, importlib, warnings, pickle
import multiprocessing as mp
from multiprocessing import get_context

from skimage import measure
from collections import Counter
import scipy as sp
import scipy.stats as sstats
import scipy.io as sio
from scipy.io import savemat, loadmat

import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#from scipy.signal import savgol_filter

### UltraNest toolbox ###
### from https://github.com/JohannesBuchner/UltraNest
### Documentation on https://johannesbuchner.github.io/UltraNest/performance.html
import ultranest
from ultranest.plot import cornerplot
import ultranest.stepsampler

from spike_shuffling import shuffling

from utils import pathcat, _hsm, get_nPaths, extend_dict
from utils_data import set_para

warnings.filterwarnings("ignore")



class detect_PC:
  
  def __init__(self,basePath,mouse,s,nP,plt_bool=False,sv_bool=False):
    
    print('----------- mouse %s --- session %d -------------'%(mouse,s))
    
    ### set global parameters and data for all processes to access
    self.para = set_para(basePath,mouse,s,nP,plt_bool,sv_bool)
    self.get_behavior()       ## load and process behavior
    
    
  def run_detection(self,S=None,rerun=False,f_max=2,return_results=False,specific_n=None,artificial=False,dataSet='redetect',mode_info='MI'):
    
    global t_start
    t_start = time.time()
    self.dataSet = dataSet
    self.f_max = f_max
    self.para['modes']['info'] = mode_info
    if S is None:
      S, other = load_activity(self.para['pathSession'],dataSet=dataSet)
      if dataSet == 'redetect':
        idx_evaluate = other[0]
        idx_previous = other[1]
        SNR = other[2]
        r_values = other[3]
    else:
      nCells = S.shape[0]
      SNR = np.zeros(nCells)*np.NaN
      r_values = np.zeros(nCells)*np.NaN
      
    nCells = S.shape[0]
    
    if not (specific_n is None):
      self.para['n'] = specific_n
      result = self.PC_detect(S[specific_n,:])
      return result
    if rerun:
      if artificial:
        #nDat,pathData = get_nPaths(self.para['pathSession'],'artificialData_analyzed_n')
        #for i in range(nDat):
        print(self.para['svname_art'])
        f = open(self.para['svname_art'],'rb')
        PC_processed = pickle.load(f)
        f.close()
        print(PC_processed.keys())
        #PC_processed = extend_dict(PC_processed,ld_tmp['fields']['parameter'].shape[0],ld_tmp)
      else:
        PC_processed = {}
        PC_processed['status'] = loadmat(self.para['svname_status'],squeeze_me=True)
        PC_processed['fields'] = loadmat(self.para['svname_fields'],squeeze_me=True)
        PC_processed['firingstats'] = loadmat(self.para['svname_firingstats'],squeeze_me=True)
      
      #self.para['modes']['info'] = False
      
      idx_process = np.where(np.isnan(PC_processed['status']['Bayes_factor'][:,0]))[0]
    else:
      idx_process = np.arange(nCells)
    nCells_process = len(idx_process)
    if nCells_process:
      #print(idx_process)
      print('run detection on %d neurons'%nCells_process)
      
      ### run multiprocessing from here, obtain list of result-dictionaries
      result_tmp = []
      if self.para['nP'] > 0:
        
        pool = get_context("spawn").Pool(self.para['nP'])
        batchSz = 50
        nBatch = nCells_process//batchSz
        
        for i in range(nBatch+1):
          idx_batch = idx_process[i*batchSz:min(nCells_process,(i+1)*batchSz)]
          #result_tmp.extend(pool.map(self.PC_detect,S[idx_batch,:]))
          #res = pool.starmap(self.PC_detect,zip(S[idx_batch,:],SNR[idx_batch],r_values[idx_batch]))
          #result_tmp.extend(res)
          result_tmp.extend(pool.starmap(self.PC_detect,zip(S[idx_batch,:],SNR[idx_batch],r_values[idx_batch])))
          print('\t\t\t ------ mouse %s --- session %d ------ %d / %d neurons processed\t ------ \t time passed: %7.2fs'%(self.para['mouse'],self.para['session'],min(nCells_process,(i+1)*batchSz),nCells_process,time.time()-t_start))
      else:
        for n0 in range(nCells_process):
          n = idx_process[n0]
          result_tmp.append(self.PC_detect(S[n,:],SNR[n],r_values[n]))
          print('\t\t\t ------ mouse %s --- session %d ------ %d / %d neurons processed\t -----\t time passed: %7.2fs'%(self.para['mouse'],self.para['session'],n0+1,nCells_process,time.time()-t_start))
      
      results = self.build_PC_results(nCells)   ## pre-allocate array
      
      for n in range(nCells):
        for key_type in result_tmp[0].keys():
          for key in result_tmp[0][key_type].keys():
            if key[0] == '_':
              continue
            if rerun:
              if ((~np.isnan(PC_processed['status']['Bayes_factor'][n,0])) | (key in ['MI_value','MI_p_value','MI_z_score'])):# | (n>=idx_process[10])):
                results[key_type][key][n,...] = PC_processed[key_type][key][n,...]
              else:
                n0 = np.where(idx_process==n)[0][0]
                results[key_type][key][n,...] = result_tmp[n0][key_type][key]
            else:
              n0 = np.where(idx_process==n)[0][0]
              results[key_type][key][n,...] = result_tmp[n0][key_type][key]
      
      #for (r,n) in zip(result_tmp,range(nCells)):
        #for key_type in r.keys():
          #for key in r[key_type].keys():
            #results[key_type][key][n,...] = r[key_type][key]
        
      print('time passed (overall): %7.2f'%(time.time()-t_start))
      
      if return_results:
        return results
      else:
        print('saving results...')
        savemat(self.para['svname_status'],results['status'])
        savemat(self.para['svname_fields'],results['fields'])
        savemat(self.para['svname_firingstats'],results['firingstats'])
        return  
    else:
      print('nothing here to process')
  
  
  def get_behavior(self,T=None):
    
    for file in os.listdir(self.para['pathSession']):
      if file.endswith("aligned.mat"):
          pathBH = os.path.join(self.para['pathSession'], file)
    
    f = h5py.File(pathBH,'r')
    key_array = ['position','time']
    
    load_behavior = {}
    for key in key_array:
      load_behavior[key] = np.squeeze(f.get('alignedData/resampled/%s'%key).value)
    f.close()
    if T is None:
      T = load_behavior['time'].shape[0]
    self.dataBH = {}
    position = load_behavior['position'][:T].astype('int')
    position -= position.min()
    self.dataBH['binpos'] = (position*(self.para['nbin']-1)/position.max()).astype('int')
    
    velocity = np.diff(np.append(position[0],position))*self.para['f']*120/position.max()
    velocity[velocity<0] = 0
    velocity = sp.ndimage.gaussian_filter(velocity,7)
    self.dataBH['active'] = (velocity) > 5
    
    #self.dataBH['binpos'] = load_behavior['binpos'][:T].astype('int') - 1 ## correct for different indexing
    
    self.dataBH['binpos_active'] = self.dataBH['binpos'][self.dataBH['active']]
    
    nbin_coarse = (self.para['nbin']/self.para['coarse_factor']-1)
    self.dataBH['binpos_coarse'] = (position*nbin_coarse/position.max()).astype('int')
    self.dataBH['binpos_coarse_active'] = self.dataBH['binpos_coarse'][self.dataBH['active']]
    
    self.dataBH['time'] = load_behavior['time'][:T]
    self.dataBH['time_active'] = self.dataBH['time'][self.dataBH['active']]
    self.dataBH['T'] = np.count_nonzero(self.dataBH['active'])
    #plt.figure()
    #plt.plot(self.dataBH['time_active'],self.dataBH['binpos_coarse'],'r')
    #plt.plot(self.dataBH['time_active'],self.dataBH['binpos_active'],'b')
    #plt.show()
    
    ###### define trials
    self.dataBH['trials'] = {}
    self.dataBH['trials']['frame'] = np.hstack([0, np.where(np.diff(self.dataBH['binpos_active'])<-10)[0]+1,len(self.dataBH['time_active'])])
    self.dataBH['trials']['t'] = np.hstack([self.dataBH['time_active'][self.dataBH['trials']['frame'][:-1]],self.dataBH['time_active'][-1]])
    dt = np.diff(self.dataBH['trials']['t'])
    ct = len(self.dataBH['trials']['frame'])-1
    #if dt[0] < 2:
      #self.dataBH['trials']['t'] = np.delete(self.dataBH['trials']['t'],0)
      #self.dataBH['trials']['frame'] = np.delete(self.dataBH['trials']['frame'],0)
      #ct -=1
      #print('cut')
    #if dt[-1] < 2:
      #self.dataBH['trials']['t'] = np.delete(self.dataBH['trials']['t'],ct)
      #self.dataBH['trials']['frame'] = np.delete(self.dataBH['trials']['frame'],ct)
      #ct -=1
      #print('cut')
    self.dataBH['trials']['ct'] = ct
    self.dataBH['trials']['dwelltime'] = np.zeros((self.dataBH['trials']['ct'],self.para['nbin']))
    self.dataBH['trials']['T'] = np.zeros(self.dataBH['trials']['ct']).astype('int')
    
    t_offset = 0
    self.dataBH['trials']['trial'] = {}
    for t in range(self.dataBH['trials']['ct']):
      self.dataBH['trials']['trial'][t] = {}
      self.dataBH['trials']['trial'][t]['binpos_active'] = self.dataBH['binpos_active'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]]
      self.dataBH['trials']['dwelltime'][t,:] = np.histogram(self.dataBH['trials']['trial'][t]['binpos_active'],self.para['bin_array_centers'])[0]/self.para['f']
      self.dataBH['trials']['T'][t] = len(self.dataBH['trials']['trial'][t]['binpos_active'])
    return self.dataBH


  def PC_detect(self,S,SNR=None,r_value=None):
  #def PC_detect(varin):
    t_start = time.time()
    result = self.build_PC_fields()
    S[S<0] = 0
    if not (SNR is None):
      result['status']['SNR'] = SNR
      result['status']['r_value'] = r_value
    T = S.shape[0]
    #try:
    active = {}
    active['S'] = S[self.dataBH['active']]    ### only consider activity during continuous runs
    if np.count_nonzero(active['S'])==0:
      print('no activity for this neuron')
      result['firingstats']['rate'] = 0
      return result
    
    
    ### calculate firing rate
    #baseline = _hsm(active['S'][active['S']>0])
    #active['S'] /= baseline
    #[spikeNr,md,sd_r] = get_spikeNr(active['S'][active['S']>0])
    
    #t_start = time.time()
    result['firingstats']['rate'],result['firingstats']['rate_thr'] = self.get_firingrate(active['S'])
    #t_end = time.time()
    #print('get spikeNr - time taken: %5.3g'%(t_end-t_start))
    
    #print("spike nr: %d"%spikeNr)
    #rate_tmp = spikeNr / (self.dataBH['T']/self.para['f'])
    if self.para['modes']['info']:
      if self.para['modes']['info'] == 'MI':
        ## obtain quantized firing rate for MI calculation
        active['qtl'] = sp.ndimage.gaussian_filter(S,self.para['sigma'])
        active['qtl'] = active['qtl'][self.dataBH['active']]
        qtls = np.quantile(active['qtl'][active['qtl']>0],np.linspace(0,1,self.para['qtl_steps']+1))
        active['qtl'] = np.count_nonzero(active['qtl'][:,np.newaxis]>=qtls[np.newaxis,1:-1],1)
    
    ### get trial-specific activity
    #t_start = time.time()
    trials_S, trials_firingmap = self.get_trials_activity(active)
    
    ## obtain firingmap
    firingstats_tmp = self.get_firingstats_from_trials(trials_firingmap)
    for key in firingstats_tmp.keys():
      result['firingstats'][key] = firingstats_tmp[key]
    
    #t_end = time.time()
    #print('get firingstats: %5.3g'%(t_end-t_start))
    
    #print(result['firingstats']['parNoise'])
    #return
    #t_start = time.time()
    #return self.calc_Icorr(active['S'],trials_S)
    #return
    if self.para['modes']['info']:
      ## obtain mutual information first - check if (computational cost of) finding fields is worth it at all
      MI_tmp = self.test_MI(active,trials_S)
      for key in MI_tmp.keys():
        result['status'][key] = MI_tmp[key]
    else:
      result['status']['MI_value'] = np.nan
      result['status']['MI_p_value'] = np.nan
      result['status']['MI_z_score'] = np.nan
    #t_end = time.time()
    #print('calc MI - time taken: %5.3g'%(t_end-t_start))
    
    #return MI_tmp
      
    ### do further tests only if there is "significant" mutual information
    #if (not self.para['modes']['info']) | (result['status']['MI_p_value']<0.2):
      
    #### -------------------------- tuning curve model -------------------------------------
    #PC_fields['firingmap'] = savgol_filter(PC_fields['firingmap'], 5, 3)
    #self.para['bin_array'] = np.linspace(2*math.pi/self.para['nbin'],2*math.pi,self.para['nbin']);
    
    hbm = HierarchicalBayesModel(result['firingstats']['map'],self.para['bin_array'],result['firingstats']['parNoise'],0)
    
    ### test models with 0 vs 1 fields
    pnames = [self.para['names'][0]]
    break_it = False
    for f in range(self.f_max+1):
      
      if break_it:
        continue
      hbm.change_model(f)
      #print('model: %d'%f)
      #t_start = time.time()
      paramnames = pnames.copy()
      paramnames.extend(self.para['names'][1:]*f)
      
      ## hand over functions for sampler
      my_prior_transform = hbm.transform_p
      my_likelihood = hbm.set_logl_func()
      
      sampler = ultranest.ReactiveNestedSampler(paramnames, my_likelihood, my_prior_transform,wrapped_params=hbm.pTC['wrap'],vectorized=True,num_bootstraps=20)#,log_dir='/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Programs/PC_analysis/test_ultra')   ## set up sampler...
      num_samples = 200
      if f>1:
        sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=3)#, adaptive_nsteps='move-distance')
        num_samples = 200
      
      sampling_result = sampler.run(min_num_live_points=num_samples,max_iters=10000,cluster_num_live_points=40,max_num_improvement_loops=5,show_status=False,viz_callback=False)  ## ... and run it #max_ncalls=500000,(f+1)*100,
      #t_end = time.time()
      #print('nested sampler done, time: %5.3g'%(t_end-t_start))
      
      A_0 = sampling_result['posterior']['mean'][0]
      
      result['status']['Z'][f,:] = [sampling_result['logz'],sampling_result['logzerr']]    ## store evidences
      
      if f > 0:
        
        fields_tmp = self.detect_modes_from_posterior(sampler)
        
        result['status']['Bayes_factor'][f-1,0] = result['status']['Z'][f,0]-result['status']['Z'][f-1,0]
        result['status']['Bayes_factor'][f-1,1] = np.sqrt(result['status']['Z'][f-1,1]**2 + result['status']['Z'][f,1]**2)
        
        if f==2:
          #try:
          print('peaks to compare:')
          print(result['fields']['parameter'][:,3,0])
          f_major = np.nanargmax(result['fields']['posterior_mass'])
          theta_major = result['fields']['parameter'][f_major,3,0]
          print(theta_major)
          print(fields_tmp['parameter'][:,3,0])
          #print(fields_tmp)
          dTheta = np.abs(np.mod(theta_major-fields_tmp['parameter'][:,3,0]+self.para['nbin']/2,self.para['nbin'])-self.para['nbin']/2)
          print(dTheta)
          if result['status']['Bayes_factor'][f-1,0] > 0:
            
            if np.nanmin(dTheta) < 5:
              
              for key in fields_tmp.keys():
                result['fields'][key] = fields_tmp[key]
              result['fields']['major'] = np.nanargmin(dTheta)
            
            #result['status']['Bayes_factor'][-1,:] = np.NaN
            #return result,sampler
          #except:
            #pass
          return result, sampler
        else:
          for key in fields_tmp.keys():
            result['fields'][key] = fields_tmp[key]
        
        if result['status']['Bayes_factor'][f-1,0]<=0:
          break_it = True
          
        
      
    if self.para['plt_bool']:
      print('for display: draw tuning curves from posterior distribution and evaluate TC-value for each bin. then, each bin has distribution of values and can be plotted! =)')
      style_arr = ['--','-']
      #col_arr = []
      #fig,ax = plt.subplots(figsize=(5,3),dpi=150)
      ax = plt.axes([0.6,0.625,0.35,0.25])
      ax.bar(self.para['bin_array'],result['firingstats']['map'],facecolor='b',width=100/self.para['nbin'],alpha=0.2)
      ax.errorbar(self.para['bin_array'],result['firingstats']['map'],result['firingstats']['CI'],ecolor='r',linestyle='',fmt='',elinewidth=0.3)#,label='$95\\%$ confidence')
      
      ax.plot(self.para['bin_array'],hbm.TC(np.array([A_0])),'k',linestyle='--',linewidth=1,label='$log(Z)=%4.1f\\pm%4.1f$ (non-coding)'%(result['status']['Z'][0,0],result['status']['Z'][0,1]))
      
      #try:
      #print(result['fields']['nModes'])
      for c in range(min(2,result['fields']['nModes'])):
        if result['fields']['nModes']>1:
          if c==0:
            label_str = '(mode #%d)\t$log(Z)=%4.1f\\pm%4.1f$'%(c+1,result['status']['Z'][1,0],result['status']['Z'][1,1])
          else:
            label_str = '(mode #%d)'%(c+1)
        else:
          label_str = '$log(Z)=%4.1f\\pm%4.1f$ (coding)'%(result['status']['Z'][1,0],result['status']['Z'][1,1])
        
        ax.plot(self.para['bin_array'],hbm.TC(result['fields']['parameter'][c,:,0]),'r',linestyle='-',linewidth=0.5+result['fields']['posterior_mass'][c]*2,label=label_str)
        #except:
          #1
      #ax.plot(self.para['bin_array'],hbm.TC(par_results[1]['mean']),'r',label='$log(Z)=%5.3g\\pm%5.3g$'%(par_results[1]['Z'][0],par_results[1]['Z'][1]))
      ax.legend(title='evidence',fontsize=8,loc='upper left',bbox_to_anchor=[0.05,1.4])
      ax.set_xlabel('Location [bin]')
      ax.set_ylabel('$\\bar{\\nu}$')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      plt.tight_layout()
      if self.para['plt_sv']:
        pathSv = pathcat([self.para['pathFigs'],'PC_analysis_fit_results.png'])
        plt.savefig(pathSv)
        print('Figure saved @ %s'%pathSv)
      plt.show(block=False)
      
    t_process = time.time()-t_start
    
    #print('get spikeNr - time taken: %5.3g'%(t_end-t_start))
    if SNR is None:
      print('p-value (MI): %5.2g, z-score (MI): %5.2g, \t bayes factor (1): %5.2g+/-%5.2g, \t (2): %5.2g+/-%5.2g \t time passed: %5.2gs'%(result['status']['MI_p_value'],result['status']['MI_z_score'],result['status']['Bayes_factor'][0,0],result['status']['Bayes_factor'][0,1],result['status']['Bayes_factor'][1,0],result['status']['Bayes_factor'][1,1],t_process))
    else:
      print('p-value (MI): %5.2g, z-score (MI): %5.2g, \t bayes factor (1): %5.2g+/-%5.2g, \t (2): %5.2g+/-%5.2g \t SNR: %5.2g,\t r_value: %5.2g \t time passed: %5.2gs'%(result['status']['MI_p_value'],result['status']['MI_z_score'],result['status']['Bayes_factor'][0,0],result['status']['Bayes_factor'][0,1],result['status']['Bayes_factor'][1,0],result['status']['Bayes_factor'][1,1],SNR,r_value,t_process))
    
    #print('p-value (MI): %5.3g, z-score (MI): %5.3g, \t bayes factor: %7.5g+/-%7.5g \t SNR: %5.3g,\t r_value: %5.3g'%(result['status']['MI_p_value'],result['status']['MI_z_score'],result['status']['Bayes_factor'][0],result['status']['Bayes_factor'][1],SNR,r_value))# \t time passed: %7.2fs,time.time()-t_start))
    #global n_ct
    #print('tried')
    #n_ct.increment()
    #print('%d neurons processed, \t %7.5gs passed'%(n_ct.value,time.time()-t_start))
    #except (KeyboardInterrupt, SystemExit):
      #raise
    #except:
      #print('analysis failed: (-) p-value (MI): %5.3g, \t bayes factor: %7.5g+/-%7.5g'%(result['status']['MI_p_value'],result['status']['Bayes_factor'][0],result['status']['Bayes_factor'][1]))
      ##result['fields']['nModes'] = -1
      
    return result


  def detect_modes_from_posterior(self,sampler,plt_bool=False):
    ### handover of sampled points
    data_tmp = ultranest.netiter.logz_sequence(sampler.root,sampler.pointpile)[0]
    logp_prior = np.log(-0.5*(np.diff(np.exp(data_tmp['logvol'][1:]))+np.diff(np.exp(data_tmp['logvol'][:-1])))) ## calculate prior probabilities (phasespace-slice volume from change in prior-volume (trapezoidal form)
    
    data = {}
    data['logX'] = np.array(data_tmp['logvol'][1:-1])
    data['logl'] = np.array(data_tmp['logl'][1:-1])
    data['logz'] = np.array(data_tmp['logz'][1:-1])
    data['logp_posterior'] = logp_prior + data['logl'] - data['logz'][-1]   ## normalized posterior weight
    data['samples'] = data_tmp['samples'][1:-1,:]
    
    if False:#self.para['plt_bool']:
      plt.figure(figsize=(2.5,1.5),dpi=300)
      ## plot weight
      ax1 = plt.subplot(111)
      dZ = np.diff(np.exp(data['logz']))
      ax1.fill_between(data['logX'][1:],dZ/dZ.max(),color=[0.5,0.5,0.5],zorder=0,label='$\Delta Z$')
      
      w = np.exp(logp_prior)
      ax1.plot(data['logX'],w/w.max(),'r',zorder=5,label='$w$')
      
      L = np.exp(data['logl'])
      ax1.plot(data['logX'],L/L.max(),'k',zorder=10,label='$\mathcal{L}$')
      
      ax1.set_yticks([])
      ax1.set_xlabel('ln X')
      ax1.legend(fontsize=8,loc='lower left')
      plt.tight_layout()
      plt.show(block=False)
      
      if self.para['plt_sv']:
        pathSv = pathcat([self.para['pathFigs'],'PC_analysis_NS_contributions.png'])
        plt.savefig(pathSv)
        print('Figure saved @ %s'%pathSv)
    
      print('add colorbar to other plot')
    
    nPars = data_tmp['samples'].shape[-1]
    nf = int((nPars - 1)/3)
    
    testing = True
    bins = 2*self.para['nbin']
    offset = 50
    
    fields = {}
    for f in range(nf):
      
      fields[f] = {}
      fields[f]['nModes'] = 0
      fields[f]['posterior_mass'] = np.zeros(3)*np.NaN
      fields[f]['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
      fields[f]['p_x'] = np.zeros((3,self.para['nbin']))*np.NaN
      
      data['pos_samples'] = np.array(data['samples'][:,3+3*f])
      logp = np.exp(data['logp_posterior'])   ## even though its not logp, but p!!
      
      
      
      ### search for baseline (where whole prior space is sampled)
      x_space = np.linspace(0,100,11)
      logX_top = -(data['logX'].min())
      logX_bottom = -(data['logX'].max())
      for i in range(10):
        logX_base = (logX_top + logX_bottom)/2
        mask_logX = -data['logX']>logX_base
        cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
        if np.mean(cluster_hist) > 0.9:
          logX_bottom = logX_base
        else:
          logX_top = logX_base
        i+=1
      
      mask = -data['logX'] > logX_base
      
      post,post_bin = np.histogram(data['pos_samples'],bins=np.linspace(0,self.para['nbin'],bins+1),weights=logp*(np.random.rand(len(logp))<(logp/logp.max())))
      post /= post.sum()
      
      # construct wrapped and smoothed histogram
      post_cat = np.concatenate([post[-offset:],post,post[:offset]])
      post_smooth = sp.ndimage.gaussian_filter(post,10,mode='wrap')
      post_smooth = np.concatenate([post_smooth[-offset:],post_smooth,post_smooth[:offset]])
      
      ## find peaks and troughs
      mode_pos, prop = sp.signal.find_peaks(post_smooth,distance=20)
      mode_pos = mode_pos[(mode_pos>offset) & (mode_pos<(bins+offset))]
      trough_pos, prop = sp.signal.find_peaks(-post_smooth,distance=20)
      
      modes = {}
      c_ct = 0
      for (i,p) in enumerate(mode_pos):
        
        ## find neighbouring troughs
        dp = trough_pos-p
        t_right = dp[dp>0].min()
        t_left = dp[dp<0].max()
        
        
        p_mass = post_cat[p+t_left:p+t_right].sum()    # obtain probability mass between peaks
        
        if p_mass > 0.2:
          modes[c_ct] = {}
          modes[c_ct]['p_mass'] = p_mass
          modes[c_ct]['left'] = post_bin[np.mod(p+t_left-offset,bins)]
          modes[c_ct]['right'] = post_bin[np.mod(p+t_right-offset,bins)]
          c_ct += 1
        
        if testing:
          print('peak @x=%.1f'%post_bin[p-offset])
          print('\ttroughs: [%.1f, %.1f]'%(post_bin[np.mod(p+t_left-offset,bins)],post_bin[np.mod(p+t_right-offset,bins)]))
          print('\tposterior mass: %5.3g'%p_mass)
        
      nsamples = len(logp)
      print(modes)
      
      plt.figure()
      plt.subplot(311)
      plt.scatter(data['pos_samples'],-data['logX'],c=np.exp(data['logp_posterior']),marker='.',label='samples')
      plt.plot([0,100],[logX_base,logX_base],'k--')
      plt.xlabel('field position $\\theta$')
      plt.ylabel('-ln(X)')
      plt.legend(loc='lower right')
      #plt.show(block=False)
      
      for (p,m) in enumerate(modes.values()):
        
        print(m)
        if m['p_mass'] > 0.1:
          
          mask_mode = np.ones(nsamples,'bool')
          
          ## calculate further statistics
          for (p2,m2) in enumerate(modes.values()):
            if not (p==p2):
              print(p,p2)
              print(m2)
              # obtain samples first 
              if m2['left']<m2['right']:
                mask_mode[(data['pos_samples']>m2['left']) & (data['pos_samples']<m2['right']) & (-data['logX']>logX_base)] = False
              else:
                mask_mode[(data['pos_samples']>m2['left']) | (data['pos_samples']<m2['right']) & (-data['logX']>logX_base)] = False
          
          mode_logp = logp[mask_mode]#/posterior_mass
          mode_logp /= mode_logp.sum()
          
          plt.subplot(312)
          plt.scatter(data['pos_samples'][mask_mode],-data['logX'][mask_mode],c=np.exp(data['logp_posterior'][mask_mode]),marker='.',label='samples')
          plt.plot([0,100],[logX_base,logX_base],'k--')
          plt.xlabel('field position $\\theta$')
          plt.ylabel('-ln(X)')
          plt.legend(loc='lower right')
          #plt.show(block=False)
          
          
          ## obtain parameters
          if m['left']<m['right']:
            fields[f]['posterior_mass'][p] = mode_logp[(data['pos_samples'][mask_mode]>m['left']) & (data['pos_samples'][mask_mode]<m['right'])].sum()
          else:
            fields[f]['posterior_mass'][p] = mode_logp[(data['pos_samples'][mask_mode]>m['left']) | (data['pos_samples'][mask_mode]<m['right'])].sum()
          
          fields[f]['parameter'][p,0,0] = get_average(data['samples'][mask_mode,0],mode_logp)
          fields[f]['parameter'][p,1,0] = get_average(data['samples'][mask_mode,1+3*f],mode_logp)
          fields[f]['parameter'][p,2,0] = get_average(data['samples'][mask_mode,2+3*f],mode_logp)
          fields[f]['parameter'][p,3,0] = get_average(data['samples'][mask_mode,3+3*f],mode_logp,True,[0,self.para['nbin']])
          
          print(fields[f]['parameter'][p,3,0])
          for i in range(4):
            ### get confidence intervals from cdf
            if i==0:
              samples = data['samples'][mask_mode,0]
            elif i==3:
              samples = (data['samples'][mask_mode,3+3*f]+self.para['nbin']/2-fields[f]['parameter'][p,3,0])%self.para['nbin']-self.para['nbin']/2        ## shift whole axis such, that peak is in the center, to get proper errorbars
            else:
              samples = data['samples'][mask_mode,i+3*f]
            
            x_cdf_posterior, y_cdf_posterior = ecdf(samples,mode_logp)
            for j in range(len(self.para['CI_arr'])):
              fields[f]['parameter'][p,i,1+j] = x_cdf_posterior[np.where(y_cdf_posterior>=self.para['CI_arr'][j])[0][0]]
          
          fields[f]['p_x'][p,:],_ = np.histogram(data['pos_samples'][mask_mode],bins=np.linspace(0,self.para['nbin'],self.para['nbin']+1),weights=mode_logp*(np.random.rand(len(mode_logp))<(mode_logp/mode_logp.max())))
          fields[f]['p_x'][p,fields[f]['p_x'][p,:]<(0.001*fields[f]['p_x'][p,:].max())] = 0
          
          fields[f]['parameter'][p,3,0] = fields[f]['parameter'][p,3,0] % 100
          fields[f]['parameter'][p,3,1:] = (fields[f]['parameter'][p,3,0] + fields[f]['parameter'][p,3,1:]) % 100
          
          fields[f]['nModes'] += 1
          plt.subplot(313)
          plt.plot(fields[f]['p_x'][p,:])
          plt.show(block=False)
          
      
      if testing:
        plt.figure()
        plt.subplot(211)
        #bin_arr = np.linspace(-25,125,bins+2*offset)
        bin_arr = np.linspace(0,bins+2*offset,bins+2*offset)
        plt.bar(bin_arr,post_smooth)
        plt.plot(bin_arr[mode_pos],post_smooth[mode_pos],'ro')
        plt.subplot(212)
        plt.bar(post_bin[:-1],post,width=0.5,facecolor='b',alpha=0.5)
        plt.plot(post_bin[np.mod(mode_pos-offset,bins)],post[np.mod(mode_pos-offset,bins)],'ro')
        plt.plot(post_bin[np.mod(trough_pos-offset,bins)],post[np.mod(trough_pos-offset,bins)],'bo')
        plt.show(block=False)
      
          ### calculate other parameters
          #for i in range(4):
            #if i==0:
              #samples = data['samples'][mask,i]
            #else:
              #samples = data['samples'][mask,i+3*f]
            ##samples = data['samples'][clusters[c]['mask'],i+3*f]
              
            #if i==2:
              #for j in range(len(sig_space)-1):
                #thr_low = sig_space[j]
                #thr_up = sig_space[j+1]
                #mask_px = np.ones(samples.shape).astype('bool')
                #mask_px[samples<thr_low] = False
                #mask_px[samples>=thr_up] = False
                #fields[f]['p_x'][c_ct,0,j] = post_mode[mask_px].sum()
            #if i==3:
              #for j in range(len(x_space)-1):
                #thr_low = x_space[j]
                #thr_up = x_space[j+1]
                #mask_px = np.ones(samples.shape).astype('bool')
                #mask_px[samples<thr_low] = False
                #mask_px[samples>=thr_up] = False
                #fields[f]['p_x'][c_ct,1,j] = post_mode[mask_px].sum()
              ##fields[f]['parameter'][c_ct,i,0] = get_average(data['samples'][clusters[c]['mask'],i+3*f],post_mode,True,[0,100])
              #fields[f]['parameter'][c_ct,i,0] = get_average(samples,post_mode,True,[0,100])
              #samples = (data['samples'][mask,i+3*f]+100/2-fields[f]['parameter'][c_ct,i,0])%100-100/2        ## shift whole axis such, that peak is in the center, to get proper errorbars
            #else:
              ##fields[f]['parameter'][c_ct,i,0] = get_average(data['samples'][clusters[c]['mask'],i+3*f],post_mode)
              #fields[f]['parameter'][c_ct,i,0] = get_average(samples,post_mode)
            
            #### get confidence intervals from cdf
            #x_cdf_posterior, y_cdf_posterior = ecdf(samples,post_mode)
            #for j in range(len(self.para['CI_arr'])):
              #fields[f]['parameter'][c_ct,i,1+j] = x_cdf_posterior[np.where(y_cdf_posterior>=self.para['CI_arr'][j])[0][0]]
            ##fields[f]['parameter'][c_ct,i,2] = x_cdf_posterior[np.where(y_cdf_posterior>0.975)[0][0]]
            
            #if i==3:
              #fields[f]['parameter'][c_ct,i,0] = fields[f]['parameter'][c_ct,i,0] % 100
              #fields[f]['parameter'][c_ct,i,1:] = (fields[f]['parameter'][c_ct,i,0] + fields[f]['parameter'][c_ct,i,1:]) % 100
              
            #fields[f]['posterior_mass'][c_ct] = clusters[c]['posterior_mass']
          
          #if c_ct == 2:
            #break   ## maximum 2 clusters detected. should be kinda sorted, since they are detected starting from smallest lnX
          #c_ct += 1
          
      
      print(fields[f])
      
      
      
      
      
      
      
      #### check from baseline up
      #### gather occupied phase space & occupied probability mass
      #x_space = np.linspace(0,100,101)
      
      #nsteps = 31
      #clusters = {}
      #blob_center = np.zeros((nsteps,1))*np.NaN
      #blob_phase_space = np.zeros((nsteps,1))*np.NaN
      #blob_probability_mass = np.zeros((nsteps,1))*np.NaN
      #blob_center_CI = np.zeros((nsteps,2,1))*np.NaN
      
      #periodic = False
      #nClusters = 0
      #logX_arr = np.linspace(logX_top,logX_base,nsteps)
      #for (logX,i) in zip(logX_arr,range(nsteps)):
        #mask_logX = -data['logX']>logX
        
        #cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
        
        ### remove "noise" clusters
        #cluster_hist = sp.ndimage.morphology.binary_opening(sp.ndimage.morphology.binary_closing(np.concatenate((cluster_hist[-10:],cluster_hist,cluster_hist[:10]))))[10:-10]
        #if not any(cluster_hist):
          #continue
        
        #blobs = measure.label(cluster_hist)
        #nblobs = blobs.max()
        
        #if (blobs[0]>0) & (blobs[-1]>0):
          #periodic = True
          #nblobs-=1
        
        #for c in range(1,nblobs+1):
          
          #if len(blob_phase_space) < nblobs:
            #blob_center.append([])
            #blob_phase_space.append([])
            #blob_probability_mass.append([])
            #blob_center_CI.append([])
            
          #if c == 1 & periodic:
            #val_last = x_space[np.where(blobs==c)[0][-1]]
            #val_first = x_space[np.where(blobs==blobs[-1])[0][0]]
            #mask_cluster = np.logical_or((val_first <= data['pos_samples']),(data['pos_samples'] < val_last))
            #blobs[blobs==blobs[-1]] = blobs[0]      ## assign components to each other, wrapped by periodicity of phase space
            
          #else:
            #val_first = x_space[np.where(blobs==c)[0][0]]
            #val_last = x_space[np.where(blobs==c)[0][-1]]
            #mask_cluster = (val_first <= data['pos_samples']) & (data['pos_samples'] < val_last)
          
          #mask = (mask_cluster & mask_logX)
          
          #if np.count_nonzero(mask)>50:
            #p_posterior_cluster = np.exp(data['logp_posterior'][mask])
            #posterior_mass = (p_posterior_cluster).sum()
            #p_posterior_cluster /= posterior_mass
            #masked_samples = data['pos_samples'][mask]
            
            #center_tmp = get_average(masked_samples,p_posterior_cluster,True,[0,100])  ## project to positive axis again
            
            #### test for overlap with other clusters
            #assigned = overlap = False
            #for cc in clusters.keys():
              
              #mask_joint = mask & clusters[cc]['mask']
                
              #if any(mask_joint):     ### does the cluster have an overlap with another one?
                
                #if clusters[cc]['active']:    ### clusters can only be assigned to active clusters
                
                  #if assigned:
                    
                    #if (clusters[c_id]['posterior_mass'] > clusters[cc]['posterior_mass']):
                      #clusters[cc]['active'] = False
                      #if clusters[cc]['appeared'] > 3:
                        #clusters[c_id]['active'] = False
                        #overlap = True
                    #else:
                      #clusters[c_id]['active'] = False
                      #if clusters[c_id]['appeared'] > 3:
                        #clusters[cc]['active'] = False
                        #overlap = True
                    
                  #elif ((np.exp(data['logp_posterior'][mask_joint]).sum() / clusters[cc]['posterior_mass']) > 0):# & abs((center_tmp -:
                    #### if it's a significant overlap, not changing the clusterstats significantly, assign to existing one
                    #c_id = cc
                    #appeared = clusters[c_id]['appeared']+1
                    #assigned = True
                  #else:
                    #### if overlap is non-significant, disable "small" cluster
                    #clusters[cc]['active'] = False
                #else:
                  #overlap = True
                
            #if overlap:
              #continue
            #elif not assigned:
              #### if no overlap was found, start new cluster
              #c_id = nClusters
              #nClusters+=1
              #appeared = 1
            
            #n_samples = len(p_posterior_cluster)
            
            #clusters[c_id] = {'periodic':periodic & (c==1),'mask':mask,'center':center_tmp,'phase_space':np.mean(blobs==c),'posterior_mass':posterior_mass,'baseline':logX,'n_samples':n_samples,'appeared':appeared,'active':True}
            
            #if c_id >= blob_center.shape[1]:
              #cat_cluster = np.zeros((nsteps,1))*np.nan
              #blob_center = np.concatenate((blob_center,cat_cluster),1)
              #blob_phase_space = np.concatenate((blob_phase_space,cat_cluster),1)
              #blob_probability_mass = np.concatenate((blob_probability_mass,cat_cluster),1)
              
              #cat_cluster = np.zeros((nsteps,2,1))*np.nan
              #blob_center_CI = np.concatenate((blob_center_CI,cat_cluster),2)
            
            #blob_center[i,c_id] = center_tmp
            #blob_phase_space[i,c_id] = np.mean(blobs==c)
            #blob_probability_mass[i,c_id] = posterior_mass
            
            #### get confidence intervals from cdf
            #x_cdf_posterior, y_cdf_posterior = ecdf(masked_samples,p_posterior_cluster)
            #blob_center_CI[i,0,c_id] = x_cdf_posterior[np.where(y_cdf_posterior>=0.025)[0][0]]
            #blob_center_CI[i,1,c_id] = x_cdf_posterior[np.where(y_cdf_posterior>0.975)[0][0]]
          
      ##print(clusters)
      ##print(blob_center)
      ##print(blob_center_CI)
      ##print(blob_phase_space)
      ##print(blob_probability_mass)
      
      #### remove overlapping and non-significant clusters
      #logX_baseline = np.inf
      #c_ct=0
      #c_arr = []
      #for c in range(nClusters):
        #clusters[c]['active'] = True
        #if c in clusters:
          #if (clusters[c]['posterior_mass'] < 0.05):
            #clusters[c]['active'] = False
            #continue
          #if (clusters[c]['appeared'] <= 3):
            #clusters[c]['active'] = False
            #continue
          #c_arr.append(c)
        #logX_baseline = min(clusters[c]['baseline'],logX_baseline)
        #c_ct+=1
      
      #### for each cluster, obtain self.para values and confidence intervals
      #fields[f] = {}
      #fields[f]['nModes'] = min(3,c_ct)
      #fields[f]['posterior_mass'] = np.zeros(3)*np.NaN
      #fields[f]['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
      #fields[f]['p_x'] = np.zeros((3,2,self.para['nbin']))*np.NaN
      
      #mask_logX = -data['logX']>logX_baseline
      #cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
      ### remove "noise" clusters
      #cluster_hist = sp.ndimage.morphology.binary_opening(sp.ndimage.morphology.binary_closing(np.concatenate((cluster_hist[-10:],cluster_hist,cluster_hist[:10]))))[10:-10]
      
      #blobs = measure.label(cluster_hist)
      #if (blobs[0]>0) & (blobs[-1]>0):
        #periodic = True
      #masks = {}
      #for i in range(1,blobs.max()):
        #masks[i] = {}
        #if i == 1 & periodic:
          #val_last = x_space[np.where(blobs==i)[0][-1]]
          #val_first = x_space[np.where(blobs==blobs[-1])[0][0]]
          #mask_cluster = np.logical_or((val_first <= data['pos_samples']),(data['pos_samples'] < val_last))
          ##blobs[blobs==blobs[-1]] = blobs[0]      ## assign components to each other, wrapped by periodicity of phase space
        #else:
          #val_first = x_space[np.where(blobs==i)[0][0]]
          #val_last = x_space[np.where(blobs==i)[0][-1]]
          #mask_cluster = (val_first <= data['pos_samples']) & (data['pos_samples'] < val_last)
        #masks[i]['baseline'] = (mask_cluster & mask_logX)
      
      #c_ct = 0
      #sig_space = np.linspace(0,10,101)
      #for c in clusters.keys():
        #if clusters[c]['active']:
          #mask = np.ones(data['samples'].shape[0]).astype('bool')
          
          #for i in range(1,len(masks)):
            #if np.count_nonzero(masks[i]['baseline'] & clusters[c]['mask'])==0:
              #mask[masks[i]['baseline']] = False
          
          ### find, which cluster mask belongs to
          #for cc in clusters.keys():
            #if not cc==c:
              #mask[clusters[cc]['mask']] = False
              ##posterior_mass -= clusters[cc]['posterior_mass']
          
          #p_posterior_cluster = np.exp(data['logp_posterior'][mask])#/posterior_mass
          #p_posterior_cluster /= p_posterior_cluster.sum()
          
          ### calculate other parameters
          #for i in range(4):
            #if i==0:
              #samples = data['samples'][mask,i]
            #else:
              #samples = data['samples'][mask,i+3*f]
            ##samples = data['samples'][clusters[c]['mask'],i+3*f]
              
            #if i==2:
              #for j in range(len(sig_space)-1):
                #thr_low = sig_space[j]
                #thr_up = sig_space[j+1]
                #mask_px = np.ones(samples.shape).astype('bool')
                #mask_px[samples<thr_low] = False
                #mask_px[samples>=thr_up] = False
                #fields[f]['p_x'][c_ct,0,j] = p_posterior_cluster[mask_px].sum()
            #if i==3:
              #for j in range(len(x_space)-1):
                #thr_low = x_space[j]
                #thr_up = x_space[j+1]
                #mask_px = np.ones(samples.shape).astype('bool')
                #mask_px[samples<thr_low] = False
                #mask_px[samples>=thr_up] = False
                #fields[f]['p_x'][c_ct,1,j] = p_posterior_cluster[mask_px].sum()
              ##fields[f]['parameter'][c_ct,i,0] = get_average(data['samples'][clusters[c]['mask'],i+3*f],p_posterior_cluster,True,[0,100])
              #fields[f]['parameter'][c_ct,i,0] = get_average(samples,p_posterior_cluster,True,[0,100])
              #samples = (data['samples'][mask,i+3*f]+100/2-fields[f]['parameter'][c_ct,i,0])%100-100/2        ## shift whole axis such, that peak is in the center, to get proper errorbars
            #else:
              ##fields[f]['parameter'][c_ct,i,0] = get_average(data['samples'][clusters[c]['mask'],i+3*f],p_posterior_cluster)
              #fields[f]['parameter'][c_ct,i,0] = get_average(samples,p_posterior_cluster)
            
            #### get confidence intervals from cdf
            #x_cdf_posterior, y_cdf_posterior = ecdf(samples,p_posterior_cluster)
            #for j in range(len(self.para['CI_arr'])):
              #fields[f]['parameter'][c_ct,i,1+j] = x_cdf_posterior[np.where(y_cdf_posterior>=self.para['CI_arr'][j])[0][0]]
            ##fields[f]['parameter'][c_ct,i,2] = x_cdf_posterior[np.where(y_cdf_posterior>0.975)[0][0]]
            
            #if i==3:
              #fields[f]['parameter'][c_ct,i,0] = fields[f]['parameter'][c_ct,i,0] % 100
              #fields[f]['parameter'][c_ct,i,1:] = (fields[f]['parameter'][c_ct,i,0] + fields[f]['parameter'][c_ct,i,1:]) % 100
              
            #fields[f]['posterior_mass'][c_ct] = clusters[c]['posterior_mass']
          
          #if c_ct == 2:
            #break   ## maximum 2 clusters detected. should be kinda sorted, since they are detected starting from smallest lnX
          #c_ct += 1
        
        
        
        
        
        
        #cc+=1
          #print('val: %5.3g, \t (%5.3g,%5.3g)'%(val[c,i],CI[c,i,0],CI[c,i,1]))
      #print('time took (post-process posterior): %5.3g'%(time.time()-t_start))
      #print(fields[f]['parameter'])
      if False:#self.para['plt_bool'] or plt_bool:
        #plt.figure()
        #### plot nsamples
        #### plot likelihood
        #plt.subplot(313)
        #plt.plot(-data['logX'],np.exp(data['logl']))
        #plt.ylabel('likelihood')
        #### plot importance weight
        #plt.subplot(312)
        #plt.plot(-data['logX'],np.exp(data['logp_posterior']))
        #plt.ylabel('posterior weight')
        #### plot evidence
        #plt.subplot(311)
        #plt.plot(-data['logX'],np.exp(data['logz']))
        #plt.ylabel('evidence')
        #plt.show(block=False)
        
        col_arr = ['tab:blue','tab:orange','tab:green']
        
        fig = plt.figure(figsize=(7,4),dpi=300)
        ax_NS = plt.axes([0.1,0.11,0.2,0.85])
        #ax_prob = plt.subplot(position=[0.6,0.675,0.35,0.275])
        #ax_center = plt.subplot(position=[0.6,0.375,0.35,0.275])
        ax_phase_1 = plt.axes([0.4,0.11,0.125,0.2])
        ax_phase_2 = plt.axes([0.55,0.11,0.125,0.2])
        ax_phase_3 = plt.axes([0.4,0.335,0.125,0.2])
        ax_hist_1 = plt.axes([0.7,0.11,0.1,0.2])
        ax_hist_2 = plt.axes([0.55,0.335,0.125,0.15])
        ax_hist_3 = plt.axes([0.4,0.56,0.125,0.15])
        
        
        ax_NS.scatter(data['pos_samples'],-data['logX'],c=np.exp(data['logp_posterior']),marker='.',label='samples')
        ax_NS.plot([0,100],[logX_base,logX_base],'k--')
        ax_NS.set_xlabel('field position $\\theta$')
        ax_NS.set_ylabel('-ln(X)')
        ax_NS.legend(loc='lower right')
        for c in range(fields[f]['nModes']):
          #if fields[f]['posterior_mass'][c] > 0.05:
          #ax_center.plot(logX_arr,blob_center[:,c],color=col_arr[c])
          #ax_center.fill_between(logX_arr,blob_center_CI[:,0,c],blob_center_CI[:,1,c],facecolor=col_arr[c],alpha=0.5)
          
          ax_phase_1.plot(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],data['samples'][clusters[c_arr[c]]['mask'],3+3*f],'k.',markeredgewidth=0,markersize=1)
          ax_phase_2.plot(data['samples'][clusters[c_arr[c]]['mask'],1+3*f],data['samples'][clusters[c_arr[c]]['mask'],3+3*f],'k.',markeredgewidth=0,markersize=1)
          ax_phase_3.plot(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],data['samples'][clusters[c_arr[c]]['mask'],1+3*f],'k.',markeredgewidth=0,markersize=1)
          
          ax_hist_1.hist(data['samples'][clusters[c_arr[c]]['mask'],3+3*f],np.linspace(0,self.para['nbin'],50),facecolor='k',orientation='horizontal')
          ax_hist_2.hist(data['samples'][clusters[c_arr[c]]['mask'],1+3*f],np.linspace(0,10,20),facecolor='k')
          ax_hist_3.hist(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],np.linspace(0,5,20),facecolor='k')
          
          #ax_phase.plot(logX_arr,blob_phase_space[:,c],color=col_arr[c],label='mode #%d'%(c+1))
          #ax_prob.plot(logX_arr,blob_probability_mass[:,c],color=col_arr[c])
          
          #if c < 3:
            #ax_NS.annotate('',(fields[f]['parameter'][c,3,0],logX_top),xycoords='data',xytext=(fields[f]['parameter'][c,3,0]+5,logX_top+2),arrowprops=dict(facecolor=ax_center.lines[-1].get_color(),shrink=0.05))
        
        nsteps = 5
        logX_arr = np.linspace(logX_top,logX_base,nsteps)
        for (logX,i) in zip(logX_arr,range(nsteps)):
          ax_NS.plot([0,100],[logX,logX],'--',color=[1,i/(2*nsteps),i/(2*nsteps)],linewidth=0.5)
        
        #ax_center.set_xticks([])
        #ax_center.set_xlim([logX_base,logX_top])
        #ax_prob.set_xlim([logX_base,logX_top])
        #ax_center.set_ylim([0,100])
        #ax_center.set_ylabel('$\\theta$')
        #ax_prob.set_ylim([0,1])
        #ax_prob.set_xlabel('-ln(X)')
        #ax_prob.set_ylabel('posterior')
        
        ax_phase_1.set_xlabel('$\\sigma$')
        ax_phase_1.set_ylabel('$\\theta$')
        ax_phase_2.set_xlabel('$A$')
        ax_phase_3.set_ylabel('$A$')
        ax_phase_2.set_yticks([])
        ax_phase_3.set_xticks([])
        
        ax_hist_1.set_xticks([])
        ax_hist_2.set_xticks([])
        ax_hist_3.set_xticks([])
        ax_hist_1.set_yticks([])
        ax_hist_2.set_yticks([])
        ax_hist_3.set_yticks([])
        
        ax_hist_1.spines['top'].set_visible(False)
        ax_hist_1.spines['right'].set_visible(False)
        ax_hist_1.spines['bottom'].set_visible(False)
        
        ax_hist_2.spines['top'].set_visible(False)
        ax_hist_2.spines['right'].set_visible(False)
        ax_hist_2.spines['left'].set_visible(False)
        
        ax_hist_3.spines['top'].set_visible(False)
        ax_hist_3.spines['right'].set_visible(False)
        ax_hist_3.spines['left'].set_visible(False)
        
        #ax_phase_1.set_xticks([])
        #ax_phase.set_xlim([logX_base,logX_top])
        #ax_phase.set_ylim([0,1])
        #ax_phase.set_ylabel('% phase space')
        #ax_phase.legend(loc='upper right')
        
        if self.para['plt_sv']:
            pathSv = pathcat([self.para['pathFigs'],'PC_analysis_NS_results.png'])
            plt.savefig(pathSv)
            print('Figure saved @ %s'%pathSv)
        plt.show(block=False)
    
    
    if nf > 1:
      
      print('detected from nested sampling:')
      print(fields[0]['parameter'][:,3,0])
      print(fields[0]['posterior_mass'])
      print(fields[1]['parameter'][:,3,0])
      print(fields[1]['posterior_mass'])
      
      fields_return = {}
      fields_return['nModes'] = 0
      fields_return['posterior_mass'] = np.zeros(3)*np.NaN
      fields_return['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
      fields_return['p_x'] = np.zeros((3,self.para['nbin']))*np.NaN
      
      for f in range(fields[0]['nModes']):
        p_cluster = fields[0]['posterior_mass'][f]
        dTheta = np.abs(np.mod(fields[0]['parameter'][f,3,0]-fields[1]['parameter'][:,3,0]+self.para['nbin']/2,self.para['nbin'])-self.para['nbin']/2)
        if np.any(dTheta<5):  ## take field with larger probability mass to have better sampling
          f2 = np.where(dTheta<5)[0][0]
          if fields[0]['posterior_mass'][f] > fields[1]['posterior_mass'][f2]:
            handover_f = 0
            f2 = f
          else:
            handover_f = 1
            p_cluster = fields[1]['posterior_mass'][f2]
        else:
          handover_f = 0
          f2 = f
          
        if p_cluster>0.5:
          fields_return['parameter'][fields_return['nModes'],...] = fields[handover_f]['parameter'][f2,...]
          fields_return['p_x'][fields_return['nModes'],...] = fields[handover_f]['p_x'][f2,...]
          fields_return['posterior_mass'][fields_return['nModes']] = fields[handover_f]['posterior_mass'][f2]
          fields_return['nModes'] += 1
      
      for f in range(fields[1]['nModes']):
        
        dTheta = np.abs(np.mod(fields[1]['parameter'][f,3,0]-fields[0]['parameter'][:,3,0]+self.para['nbin']/2,self.para['nbin'])-self.para['nbin']/2)
        
        dTheta2 = np.abs(np.mod(fields[1]['parameter'][f,3,0]-fields_return['parameter'][:,3,0]+self.para['nbin']/2,self.para['nbin'])-self.para['nbin']/2)
        
        
        if (not np.any(dTheta<5)) and (not np.any(dTheta2<5)) and (fields[1]['posterior_mass'][f]>0.5):  ## take field with larger probability mass to have better sampling
          fields_return['parameter'][fields_return['nModes'],...] = fields[1]['parameter'][f,...]
          fields_return['p_x'][fields_return['nModes'],...] = fields[1]['p_x'][f,...]
          fields_return['posterior_mass'][fields_return['nModes']] = fields[1]['posterior_mass'][f]
          fields_return['nModes'] += 1
          
      
    else:
      fields_return = fields[0]
      
      
      
      
    #print(fields_return)
    return fields_return


  def get_trials_activity(self,active):
    
    ## preallocate
    trials_map = np.zeros((self.dataBH['trials']['ct'],self.para['nbin']))
    
    trials_S = {}
    for t in range(self.dataBH['trials']['ct']):
      trials_S[t] = {}
      trials_S[t]['S'] = active['S'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]];    ## should be quartiles?!
      if self.para['modes']['info'] == 'MI':
        trials_S[t]['qtl'] = active['qtl'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]];    ## should be quartiles?!
      
      if self.para['modes']['activity'] == 'spikes':
        trials_S[t]['spike_times'] = np.where(trials_S[t]['S']);
        trials_S[t]['spikes'] = trials_S[t]['S'][trials_S[t]['spike_times']];
        trials_S[t]['ISI'] = np.diff(trials_S[t]['spike_times']);
      
      trials_S[t]['rate'] = trials_S[t]['S'].sum()/(self.dataBH['trials']['T'][t]/self.para['f']);
      
      if trials_S[t]['rate'] > 0:
        trials_map[t,:] = self.get_firingmap(trials_S[t]['S'],self.dataBH['trials']['trial'][t]['binpos_active'],self.dataBH['trials']['dwelltime'][t,:])#/trials_S[t]['rate']
      
      #[spikeNr,md,sd_r] = get_spikeNr(trials_S[t]['S'][trials_S[t]['S']>0]);
      #trials_rate[t] = spikeNr/(self.dataBH['trials']['T'][t]/self.para['f']);
    trials_map /= np.nansum(trials_map,1)[:,np.newaxis]
    
    return trials_S, trials_map#, trials_rate


  def get_firingstats_from_trials(self,trials_firingmap):
    
    ### construct firing rate map from bootstrapping over (normalized) trial firing maps
    
    firingstats = {}
    firingmap_bs = np.zeros((self.para['nbin'],self.para['N_bs']))
    
    base_sample = np.random.randint(0,self.dataBH['trials']['ct'],(self.para['N_bs'],self.dataBH['trials']['ct']))
    
    for L in range(self.para['N_bs']):
      #dwelltime = self.dataBH['trials']['dwelltime'][base_sample[L,:],:].sum(0)
      #firingmap_bs[:,L] = trials_firingmap[base_sample[L,:],:].sum(0)/dwelltime
      #mask = (dwelltime==0)
      #firingmap_bs[mask,L] = 0
      
      firingmap_bs[:,L] = np.nanmean(trials_firingmap[base_sample[L,:],:],0)# / self.dataBH['trials']['dwelltime'][base_sample[L,:],:],0)
    firingstats['map'] = np.nanmean(firingmap_bs,1)
    firingstats['map'][~(firingstats['map']>0)] = 1/(self.para['nbin']*self.dataBH['T'])     ## set 0 firing rates to lowest possible (0 leads to problems in model, as 0 noise, thus likelihood = 0)
    
    ### estimate noise of model
    ## parameters of gamma distribution can be directly inferred from mean and std
    firingstats['std'] = np.nanstd(firingmap_bs,1)
    prc = [2.5,97.5]
    firingstats['CI'] = np.nanpercentile(firingmap_bs,prc,1);   ## width of gaussian - from 1-SD confidence interval
    ### fit linear dependence of noise on amplitude (with 0 noise at fr=0)
    firingstats['parNoise'] = jackknife(firingstats['map'],firingstats['std'])
    
    if self.para['plt_theory_bool']:
      self.plt_model_selection(firingmap_bs,firingstats['parNoise'],trials_firingmap)
      #return
    return firingstats


  def get_info_value(self,activity,dwelltime):
    
    if self.para['modes']['info'] == 'MI':
      p_joint = self.get_p_joint(activity)   ## need activity trace
      return get_MI(p_joint,dwelltime,self.para['qtl_weight'])
      
    elif self.para['modes']['info'] == 'I_sec':
      fmap = self.get_firingmap(activity,self.dataBH['binpos_coarse_active'],dwelltime,coarse=True)
      fmap /= fmap.sum()
      return np.nansum(dwelltime*fmap*np.log2(fmap/fmap.mean()))
      
      #firingmap = imgaussfilt(get_firingmap(activity,binpos,dwelltime,self.para,True),self.para['sigma'],
                              #'Padding','circular','FilterDomain','spatial')
      #return get_I_sec(firingmap,dwelltime,self.para)
    

  def get_p_joint(self,activity):
    
    ### need as input:
    ### - activity (quantiled or something)
    ### - behavior trace
    p_joint = np.zeros((self.para['nbin_coarse'],self.para['qtl_steps']))
    
    for q in range(self.para['qtl_steps']):
      for (x,ct) in Counter(self.dataBH['binpos_coarse_active'][activity==q]).items():
        p_joint[x,q] = ct;
    p_joint = p_joint/p_joint.sum();    ## normalize
    return p_joint

  
  def calc_Icorr(self,S,trials_S):
    
    S /= S[S>0].mean()
    lag = [0,self.para['f']*2]
    nlag = lag[1]-lag[0]
    T = S.shape[0]
    
    print('check if range is properly covered in C_cross (and PSTH)')
    print('speed, speed, speed!! - what can be vectorized? how to get rid of loops?')
    print('spike train generation: generate a single, long one at once (or smth)')
    print('how to properly generate surrogate data? single trials? conjoint trials? if i put in a rate only, sums will even out to homogenous process for N large')
    
    PSTH = np.zeros((self.para['nbin'],nlag))
    C_cross = np.zeros((self.para['nbin'],nlag))
    for x in range(self.para['nbin']):
      for t in range(self.dataBH['trials']['ct']):
        idx_x = np.where(self.dataBH['trials']['trial'][t]['binpos_active']==x)[0]
        #print(idx_x)
        if len(idx_x):
          i = self.dataBH['trials']['frame'][t] + idx_x[0]    ## find entry to position x in trial t
          #print('first occurence of x=%d in trial %d (start: %d) at frame %d'%(x,t,self.dataBH['trials']['frame'][t],i))
          PSTH[x,:min(nlag,T-(i+lag[0]))] += S[i+lag[0]:min(T,i+lag[1])]
      for i in range(1,nlag):
        C_cross[x,i] = np.corrcoef(PSTH[x,:-i],PSTH[x,i:])[0,1]
      C_cross[np.isnan(C_cross)] = 0
      #C_cross[x,:] = np.fft.fft(C_cross[x,:])
    #PSTH /= nlag/self.para['f']*self.dataBH['trials']['ct']
    fC_cross = np.fft.fft(C_cross)
    
    rate = PSTH.sum(1)/(nlag*self.dataBH['trials']['ct'])
    #print(rate)
    
    Icorr = np.zeros(self.para['nbin'])
    Icorr_art = np.zeros(self.para['nbin'])
    Icorr_art_std = np.zeros(self.para['nbin'])
    
    for x in range(self.para['nbin']):
      print(x)
      Icorr[x] = -1/2*rate[x] * np.log2(1 - fC_cross[x,:]/(rate[x]+fC_cross[x,:])).sum()
      Icorr_art[x], Icorr_art_std[x] = self.calc_Icorr_data(rate[x],nlag)
    
    plt.figure()
    plt.plot(Icorr)
    plt.errorbar(range(100),Icorr_art,yerr=Icorr_art_std)
    #plt.plot(Icorr_art,'r')
    plt.show(block=False)
    
    return PSTH, C_cross, Icorr
      #self.dataBH['trials']['frame'][t]
  
  def calc_Icorr_data(self,rate,T,N_bs=10):
    
    t = np.linspace(0,T-1,T)
    Icorr = np.zeros(N_bs)
    
    nGen = int(math.ceil(1.1*T*rate)) 
    u = np.random.rand(N_bs,nGen)   ## generate random variables to cover the whole time
    t_AP = np.cumsum(-(1/rate)*np.log(u),1) ## generate points of homogeneous pp
    #print(t_AP)
    for L in range(N_bs):
      t_AP_now = t_AP[L,t_AP[L,:]<T];
      idx_AP = np.argmin(np.abs(t_AP_now[:,np.newaxis]-t[np.newaxis,:]),1)
      
      PSTH = np.zeros(T)
      for AP in idx_AP:
        PSTH[AP] += 1
      
      #C_cross = np.correlate(PSTH,PSTH)
      #print(C_cross)
      C_cross = np.zeros(T)
      for i in range(1,T):
        C_cross[i] = np.corrcoef(PSTH[:-i],PSTH[i:])[0,1]
      C_cross[np.isnan(C_cross)] = 0
      fC_cross = np.fft.fft(C_cross)
      Icorr[L] = -1/2*rate * np.log2(1 - fC_cross/(rate+fC_cross)).sum()
      
    return Icorr.mean(), Icorr.std()
  
  
  def test_MI(self,active,trials_S):
    
    shuffle_peaks = False;
    if self.para['modes']['info'] == 'MI':
      S_key = 'qtl'
    else:
      S_key = 'S'
    
    MI = {'MI_p_value':np.NaN,'MI_value':np.NaN,'MI_z_score':np.NaN}
    rand_distr = np.zeros(self.para['repnum'])*np.NaN
    
    ### first, get actual MI value
    norm_dwelltime = self.dataBH['trials']['dwelltime'].sum(0)*self.para['f']/self.dataBH['T']
    norm_dwelltime_coarse = np.zeros(self.para['nbin_coarse'])
    for i in range(self.para['nbin_coarse']):
      norm_dwelltime_coarse[i] = norm_dwelltime[i*self.para['coarse_factor']:(i+1)*self.para['coarse_factor']].sum()
    
    MI['MI_value'] = self.get_info_value(active[S_key],norm_dwelltime_coarse)
    #return MI
    ### shuffle according to specified mode
    #t_start_shuffle = time.time()
    
    trial_ct = self.dataBH['trials']['ct']
    for L in range(self.para['repnum']):
      
      ## shift single trials to destroy characteristic timescale
      if self.para['modes']['shuffle'] == 'shuffle_trials':
        
        ## trial shuffling
        trials = np.random.permutation(trial_ct)
        
        shuffled_activity = np.roll(np.hstack([np.roll(trials_S[t][S_key],int(random.random()*self.dataBH['trials']['T'][t])) for t in trials]),int(random.random()*self.dataBH['T']))
        
        ### subsequent global shuffling
        #if self.para['modes']['activity'] == 'spikes':
          #spike_times_L = np.where(shuffled_activity)[0]
          #spikes_L = shuffled_activity[spike_times_L]
          #ISI_L = np.diff(spike_times_L)
          #shuffled_activity = shuffling('dithershift',shuffle_peaks,spike_times=spike_times_L,spikes=spikes_L,T=self.dataBH['T'],ISI=ISI_L,w=2*self.para['f'])
        #else:
          #shuffled_activity = shuffling('shift',shuffle_peaks,spike_train=shuffled_activity)
      
      
      elif self.para['modes']['shuffle'] == 'shuffle_global':
        if self.para['modes']['activity'] == 'spikes':
          shuffled_activity = shuffling('dithershift',shuffle_peaks,spike_times=spike_times,spikes=spikes,T=self.dataBH['T'],ISI=ISI,w=2*self.para['f'])
        else:
          shuffled_activity = shuffling('shift',shuffle_peaks,spike_train=active[S_key])
        
      elif self.para['modes']['shuffle'] == 'randomize':
        shuffled_activity = active[S_key][np.random.permutation(len(active[S_key]))];
      #
      #t_start_info = time.time()
      #print(shuffled_activity)
      rand_distr[L] = self.get_info_value(shuffled_activity,norm_dwelltime_coarse)
      #print('info calc: time taken: %5.3g'%(time.time()-t_start_info))
    #print('shuffle: time taken: %5.3g'%(time.time()-t_start_shuffle))
    
    MI_mean = np.nanmean(rand_distr)
    MI_std = np.nanstd(rand_distr)
    MI['MI_z_score'] = (MI['MI_value'] - MI_mean)/MI_std
    if MI['MI_value'] > rand_distr.max():
      MI['MI_p_value'] = 1e-10#1/self.para['repnum']
    else:
      x,y = ecdf(rand_distr)
      min_idx = np.argmin(abs(x-MI['MI_value']));
      MI['MI_p_value'] = 1 - y[min_idx];
    #print('p_value: %7.5g'%MI['MI_p_value'])
    
    #if pl['bool']:
    #plt.figure()
    #plt.hist(rand_distr)
    #plt.plot(MI['MI_value'],0,'kx')
    #plt.show(block=True)
    
    return MI



  def build_PC_fields(self):
    
    result = {}
    result['status'] = {'MI_value':np.NaN,
                        'MI_p_value':np.NaN,
                        'MI_z_score':np.NaN,
                        'Z':np.zeros((self.f_max+1,2))*np.NaN,
                        'Bayes_factor':np.zeros((self.f_max,2))*np.NaN,
                        'SNR':np.NaN,
                        'r_value':np.NaN}
    
    result['fields'] = {'parameter':np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN,          ### (field1,field2) x (A0,A,std,theta) x (mean,CI_low,CI_top)
                        'posterior_mass':np.zeros(3)*np.NaN,
                        'p_x':np.zeros((3,2,self.para['nbin']))*np.NaN,
                        'nModes':0,
                        'major':np.NaN}
    
    result['firingstats'] = {'rate':np.NaN,
                            'rate_thr':np.NaN,
                            'map':np.zeros(self.para['nbin'])*np.NaN,
                            'std':np.zeros(self.para['nbin'])*np.NaN,
                            'CI':np.zeros((2,self.para['nbin']))*np.NaN,
                            'parNoise':np.NaN}
    return result
    
    
  def build_PC_results(self,nCells):
    results = {}
    results['status'] = {'MI_value':np.zeros(nCells)*np.NaN,
                        'MI_p_value':np.zeros(nCells)*np.NaN,
                        'MI_z_score':np.zeros(nCells)*np.NaN,
                        'Z':np.zeros((nCells,self.f_max+1,2))*np.NaN,
                        'Bayes_factor':np.zeros((nCells,self.f_max,2))*np.NaN,
                        'SNR':np.zeros(nCells)*np.NaN,
                        'r_value':np.zeros(nCells)*np.NaN}
                  
    results['fields'] = {'parameter':np.zeros((nCells,3,4,1+len(self.para['CI_arr'])))*np.NaN,          ### (mean,std,CI_low,CI_top)
                        'p_x':sp.sparse.COO((nCells,3,self.para['nbin'])),#np.zeros((nCells,3,2,self.para['nbin'])),##
                        'posterior_mass':np.zeros((nCells,3))*np.NaN,
                        'nModes':np.zeros(nCells).astype('int'),
                        'major':np.zeros(nCells)*np.NaN}
    
    results['firingstats'] = {'rate':np.zeros(nCells)*np.NaN,
                              'rate_thr':np.zeros(nCells)*np.NaN,
                              'map':np.zeros((nCells,self.para['nbin']))*np.NaN,
                              'std':np.zeros((nCells,self.para['nbin']))*np.NaN,
                              'CI':np.zeros((nCells,2,self.para['nbin']))*np.NaN,
                              'parNoise':np.zeros(nCells)*np.NaN}
    return results


  #def set_para(self,basePath,mouse,s,nP,plt_bool,sv_bool):
    
    ### set paths:
    #pathMouse = pathcat([basePath,mouse])
    #pathSession = pathcat([pathMouse,'Session%02d'%s])
    
    #nbin = 100
    #qtl_steps = 5
    #coarse_factor = 4
    #self.para = {'nbin':nbin,'f':15,
                #'bin_array':np.linspace(0,nbin-1,nbin),
                #'bin_array_centers':np.linspace(0,nbin,nbin+1)-0.5,
                #'coarse_factor':coarse_factor,
                #'nbin_coarse':int(nbin/coarse_factor),
                
                #'nP':nP,
                #'N_bs':10000,'repnum':1000,
                #'qtl_steps':qtl_steps,'sigma':1,
                #'qtl_weight':np.ones(qtl_steps)/qtl_steps,
                #'names':['A_0','A','SD','theta'],
                #'CI_arr':[0.001,0.025,0.05,0.159,0.5,0.841,0.95,0.975,0.999],
                
                #'plt_bool':plt_bool&(nP==0),
                #'plt_theory_bool':False&(nP==0),
                #'plt_sv':sv_bool&(nP==0),
                
                #'mouse':mouse,
                #'session':s,
                #'pathSession':pathSession,
                #'pathFigs':'/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Results/pics/Methods',
                
                #### provide names for figures
                #'svname_status':pathcat([pathSession,'PC_fields_status.mat']),
                #'svname_fields':pathcat([pathSession,'PC_fields_para.mat']),
                #'svname_firingstats':pathcat([pathSession,'PC_fields_firingstats.mat']),
                
                #### modes, how to perform PC detection
                #'modes':{'activity':'calcium',          ## data provided: 'calcium' or 'spikes'
                          #'info':'MI',                   ## information calculated: 'MI', 'Isec' (/second), 'Ispike' (/spike)
                          #'shuffle':'shuffle_trials'     ## how to shuffle: 'shuffle_trials', 'shuffle_global', 'randomize'
                        #}
                #}
    
  def get_firingmap(self,S,binpos,dwelltime,coarse=False):
    
    ### calculates the firing map
    spike_times = np.where(S)
    spikes = S[spike_times]
    binpos = binpos[spike_times]#.astype('int')
    
    firingmap = np.zeros(int(self.para['nbin']/self.para['coarse_factor'])) if coarse else np.zeros(self.para['nbin'])
    for (p,s) in zip(binpos,spikes):#range(len(binpos)):
      firingmap[p] = firingmap[p]+s
    
    if not (dwelltime is None):
      firingmap = firingmap/dwelltime
      firingmap[dwelltime==0] = np.NaN
    
    return firingmap
  
  
  def get_firingrate(self,S):
    
    S[S<0.0001*S.max()]=0
    Ns = (S>0).sum()
    if Ns==0:
      return 0
    else:
      trace = S[S>0]
      baseline = np.median(trace)
      trace -= baseline
      trace *= -1*(trace <= 0)
      
      Ns_baseline = (trace>0).sum()
      noise = np.sqrt((trace**2).sum()/(Ns_baseline*(1-2/np.pi)))
      
      firing_threshold_adapt = sstats.norm.ppf((1-0.01)**(1/Ns))
      N_spikes = np.floor(S / (baseline + firing_threshold_adapt*noise)).sum()
      return N_spikes/(S.shape[0]/self.para['f']),firing_threshold_adapt
  
  
  def plt_model_selection(self,fmap_bs,parsNoise,trials_fmap):
    
    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)
    sig2 = False   ## estimate variance from 2-sigma or from 1-sigma interval (more accurate from 1-sigma interval, as it becomes less gaussian, the more data it encompasses
    if sig2: 
      prc = [2.5,97.5]
      sig_fact = 1.96
    else:
      prc = [15.8,84.2]
      sig_fact = 1
    fr_mu = np.nanmean(fmap_bs,1)                 ## average of firing rate (centre of gaussian)
    fr_CI = np.nanpercentile(fmap_bs,prc,1);   ## width of gaussian - from 1-SD confidence interval
    fr_std = np.nanstd(fmap_bs,1)
    
    #fig = plt.figure(figsize=(4,3),dpi=150)
    #ax = plt.axes([0.15,0.2,0.8,0.75])
    
    #ax.bar(self.para['bin_array'],fr_mu,color='b',alpha=0.2,width=1,label='$\\bar{\\nu}$')
    #ax.errorbar(self.para['bin_array'],fr_mu,fr_CI,ecolor='r',elinewidth=0.5,linestyle='',fmt='',label='$95\\%$ confidence')
    
    #ax.set_xticks(np.linspace(0,100,6))
    #ax.set_xticklabels(np.linspace(0,100,6).astype('int'))
    #ax.set_ylim([-0.1,np.nanmax(fr_mu)*2])
    #ax.set_ylabel('Ca$^{2+}$-event rate $\\nu$',fontsize=12)
    #ax.set_xlabel('Position on track',fontsize=12)
    #ax.legend(loc='upper left')
    #plt.show(block=False)
    
    fig = plt.figure(figsize=(7,5),dpi=300)
    
    ## get data
    pathDat = os.path.join(self.para['pathSession'],'results_redetect.mat')
    ld = loadmat(pathDat,variable_names=['S','C'])
    
    C = ld['C'][self.para['n'],:]
    S = ld['S'][self.para['n'],:]
    
    t_start = 0#200
    t_end = 600#470
    n_trial = 12
    
    ax_Ca = plt.axes([0.1,0.8,0.5,0.175])
    idx_longrun = self.dataBH['active']
    t_longrun = self.dataBH['time'][idx_longrun]
    t_stop = self.dataBH['time'][~idx_longrun]
    ax_Ca.bar(t_stop,np.ones(len(t_stop))*1.2*S.max(),color=[0.9,0.9,0.9],zorder=0)
    
    ax_Ca.fill_between([self.dataBH['trials']['t'][n_trial],self.dataBH['trials']['t'][n_trial+1]],[0,0],[1.2*S.max(),1.2*S.max()],color=[0,0,1,0.2],zorder=1)
    
    ax_Ca.plot(self.dataBH['time'],C,'k',linewidth=0.2)
    ax_Ca.plot(self.dataBH['time'],S,'r',linewidth=1)
    ax_Ca.set_ylim([0,1.2*S.max()])
    ax_Ca.set_xlim([t_start,t_end])
    ax_Ca.set_xticks([])
    ax_Ca.set_ylabel('Ca$^{2+}$')
    ax_Ca.set_yticks([])
    
    ax_loc = plt.axes([0.1,0.5,0.5,0.3])
    
    ax_loc.plot(self.dataBH['time'],self.dataBH['binpos'],'.',color=[0.6,0.6,0.6],zorder=5,markeredgewidth=0,markersize=1)
    idx_active = (S>0) & self.dataBH['active']
    idx_inactive = (S>0) & ~self.dataBH['active']
    
    t_active = self.dataBH['time'][idx_active]
    pos_active = self.dataBH['binpos'][idx_active]
    S_active = S[idx_active]
    
    t_inactive = self.dataBH['time'][idx_inactive]
    pos_inactive = self.dataBH['binpos'][idx_inactive]
    S_inactive = S[idx_inactive]
    ax_loc.scatter(t_active,pos_active,s=S_active/S.max()*10+0.5,c='r',zorder=10)
    ax_loc.scatter(t_inactive,pos_inactive,s=S_inactive/S.max()*10+0.5,c='k',zorder=10)
    ax_loc.bar(t_stop,np.ones(len(t_stop))*100,color=[0.9,0.9,0.9],zorder=0)
    ax_loc.fill_between([self.dataBH['trials']['t'][n_trial],self.dataBH['trials']['t'][n_trial+1]],[0,0],[100,100],color=[0,0,1,0.2],zorder=1)
    
    ax_loc.set_ylim([0,100])
    ax_loc.set_xlim([t_start,t_end])
    ax_loc.set_xlabel('t [s]')
    ax_loc.set_ylabel('Location [bin]')
    
    nC,T = ld['C'].shape
    ax_acorr = plt.axes([0.7,0.9,0.2,0.075])
    n_arr = np.random.randint(0,nC,20)
    lags = 300
    t=np.linspace(0,lags/15,lags+1)
    for n in n_arr:
      acorr = np.zeros(lags+1)
      acorr[0] = 1
      for i in range(1,lags+1):
        acorr[i] = np.corrcoef(ld['C'][n,:-i],ld['C'][n,i:])[0,1]
      #acorr = np.correlate(ld['S'][n,:],ld['S'][n,:],mode='full')[T-1:T+lags]
      #ax_acorr.plot(t,acorr/acorr[0])
      ax_acorr.plot(t,acorr,linewidth=0.5)
    
    ax_acorr.set_xlabel('$\Delta t$ [s]')
    ax_acorr.set_ylabel('corr.')
    
    ax1 = plt.axes([0.6,0.5,0.35,0.3])
    i = random.randint(0,self.para['nbin']-1)
    #ax1 = plt.subplot(211)
    ax1.barh(self.para['bin_array'],fr_mu,facecolor='b',alpha=0.2,height=1,label='$\\bar{\\nu}$')
    ax1.barh(self.para['bin_array'][i],fr_mu[i],facecolor='b',height=1)
    
    ax1.errorbar(fr_mu,self.para['bin_array'],xerr=fr_CI,ecolor='r',linewidth=0.2,linestyle='',fmt='',label='1 SD confidence')
    Y = trials_fmap/self.dataBH['trials']['dwelltime']
    mask = ~np.isnan(Y)
    Y = [y[m] for y, m in zip(Y.T, mask.T)]
    
    #flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5)
    #h_bp = ax1.boxplot(Y,flierprops=flierprops)#,positions=self.para['bin_array'])
    ax1.set_yticks([])#np.linspace(0,100,6))
    #ax1.set_yticklabels(np.linspace(0,100,6).astype('int'))
    ax1.set_ylim([0,100])
    ax1.set_xlim([0,np.nanmax(fr_mu)*1.2])
    ax1.set_xticks([])
    #ax1.set_xlabel('Ca$^{2+}$-event rate $\\nu$')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    #ax1.set_ylabel('Position on track')
    ax1.legend(title='# trials = %d'%self.dataBH['trials']['ct'],loc='lower left',bbox_to_anchor=[0.55,0.025],fontsize=8)#[h_bp['boxes'][0]],['trial data'],
    
    ax2 = plt.axes([0.6,0.275,0.35,0.175])
    #ax2 = plt.subplot(426)
    ax2.plot(np.linspace(0,5,101),parsNoise*np.linspace(0,5,101),'--',color=[0.5,0.5,0.5],label='lq-fit')
    ax2.plot(fr_mu,fr_std,'r.',markersize=1)#,label='$\\sigma(\\nu)$')
    ax2.set_xlim([0,np.nanmax(fr_mu)*1.2])
    #ax2.set_xlabel('firing rate $\\nu$')
    ax2.set_xticks([])
    ax2.set_ylim([0,np.nanmax(fr_std)*1.2])
    ax2.set_ylabel('$\\sigma_{\\nu}$')
    
    ax3 = plt.axes([0.1,0.08,0.4,0.25])
    #ax3 = plt.subplot(223)
    
    
    ax1.bar(self.para['bin_array'][i],fr_mu[i],color='b')
    
    x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,self.para['nbin']+1)
    offset = (x_arr[1]-x_arr[0])/2
    act_hist = np.histogram(fmap_bs[i,:],x_arr,normed=True)
    ax3.bar(act_hist[1][:-1],act_hist[0],width=x_arr[1]-x_arr[0],color='b',alpha=0.2,label='data (bin %d)'%i)
    
    alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
    mu, shape = lognorm_paras(fr_mu[i],fr_std[i])
    
    def gamma_fun(x,alpha,beta):
      return beta**alpha * x**(alpha-1) * np.exp(-beta*x) / sp.special.gamma(alpha)
    
    #fm = trials_fmap/self.dataBH['trials']['dwelltime']
    #ax3.plot(fm[:,i],np.zeros(fm.shape[0]),'kx')
    
    ax3.plot(x_arr,sstats.gamma.pdf(x_arr,alpha,0,1/beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')
    #ax3.plot(x_arr,gamma_fun(x_arr,alpha,beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')
    
    #D,p = sstats.kstest(fmap_bs[i,:1000],'gamma',args=(alpha,0,1/beta))
    
    #sstats.gamma.rvs()
    #ax3.plot(x_arr,sstats.lognorm.pdf(x_arr,s=shape,loc=0,scale=np.exp(mu)),'b',label='fit: $lognorm(\\alpha,\\beta)$')
    #ax3.plot(x_arr,sstats.truncnorm.pdf(x_arr,(0-fr_mu[i])/fr_std[i],np.inf,loc=fr_mu[i],scale=fr_std[i]),'g',label='fit: $gauss(\\mu,\\sigma)$')
    
    ax3.set_xlabel('$\\nu$')
    ax3.set_ylabel('$p_{bs}(\\nu)$')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    
    ax2.legend(fontsize=8)
    #ax2.set_title("estimating noise")
    
    ax3.legend(fontsize=8)
    #ax3.set_title('bootstrapped data at bin %d'%i)
    
    ax4 = plt.axes([0.6,0.08,0.35,0.175])
    #ax4 = plt.subplot(428)
    
    D_KL_gamma = np.zeros(self.para['nbin'])
    D_KL_gauss = np.zeros(self.para['nbin'])
    D_KL_lognorm = np.zeros(self.para['nbin'])
    
    D_KS_stats = np.zeros(self.para['nbin'])
    p_KS_stats = np.zeros(self.para['nbin'])
    
    
    
    
    #print('this is ordered by bins, not by firing rate?!')
    for i in range(self.para['nbin']):
      x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,self.para['nbin']+1)
      offset = (x_arr[1]-x_arr[0])/2
      act_hist = np.histogram(fmap_bs[i,:],x_arr,normed=True)
      alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
      mu, shape = lognorm_paras(fr_mu[i],fr_std[i])
      
      #print(fmap_bs[i,:])
      
      D_KS_stats[i],p_KS_stats[i] = sstats.kstest(fmap_bs[i,:],'gamma',args=(alpha,offset,1/beta))
      #print('bin #%d'%i)
      #print(fm[:,i])
      #if np.any(np.isfinite(fm[:,i])):
        #D_KS_stats[i],p_KS_stats[i] = sstats.kstest(fm[~np.isnan(fm[:,i]),i],'gamma',args=(alpha,offset,1/beta))
        #print(D_KS_stats[i],p_KS_stats[i])
        
      #hist_cdf = np.cumsum(act_hist[0])
      #hist_cdf /= hist_cdf[-1]
      #offset = (x_arr[1]-x_arr[0])/2
      #D_KL_gamma[i] = abs(sstats.gamma.cdf(act_hist[1][:-1]+offset,alpha,0,1/beta) - hist_cdf).max()
      #D_KL_gauss[i] = abs(sstats.truncnorm.cdf(act_hist[1][:-1]+offset,(0-fr_mu[i])/fr_std[i],np.inf,loc=fr_mu[i],scale=fr_std[i]) - hist_cdf).max()
      #D_KL_lognorm[i] = abs(sstats.lognorm.cdf(act_hist[1][:-1]+offset,s=shape,loc=0,scale=np.exp(mu)) - hist_cdf).max()
      
      #if D_KL_gamma[i] > 0.1:
        
        #print('D_KL (Gamma): ',D_KL_gamma[i])
        #plt.figure()
        #plt.subplot(121)
        #plt.bar(act_hist[1][:-1],act_hist[0],width=x_arr[1]-x_arr[0])
        
        #plt.plot(x_arr,sstats.gamma.pdf(act_hist[1],alpha,0,1/beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')
        ##plt.plot(x_arr,sstats.lognorm.pdf(x_arr,s=shape,loc=0,scale=np.exp(mu)),'b',label='fit: $lognorm(\\alpha,\\beta)$')
        #plt.plot(x_arr,sstats.norm.pdf(x_arr,loc=fr_mu[i],scale=fr_std[i]),'g',label='fit: $gauss(\\mu,\\sigma)$')
        
        #plt.legend()
        
        #plt.subplot(122)
        #plt.bar(act_hist[1][:-1],hist_cdf,width=x_arr[1]-x_arr[0])
        #plt.plot(x_arr,sstats.gamma.cdf(act_hist[1],alpha,0,1/beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')
        ##plt.plot(x_arr,sstats.lognorm.cdf(x_arr,s=shape,loc=0,scale=np.exp(mu)),'b',label='fit: $lognorm(\\alpha,\\beta)$')
        #plt.plot(x_arr,sstats.norm.cdf(x_arr,loc=fr_mu[i],scale=fr_std[i]),'g',label='fit: $gauss(\\mu,\\sigma)$')
        
        #plt.legend()
        
        #plt.show(block=False)
    
    #ax4.plot(fr_mu,p_KS_stats,'k.')
    #ax4.set_yscale('log')
    #ax4.set_ylim([10**(-30),1])
    #ax4.plot(fr_mu,D_KL_gamma,'rx')
    ax4.plot(fr_mu,D_KS_stats,'k.',markersize=1)
    #ax4.plot(fr_mu,D_KL_gauss,'gx')
    #ax4.plot(fr_mu,D_KL_lognorm,'bx')
    ax4.set_xlim([0,np.nanmax(fr_mu)*1.2])
    ax4.set_xlabel('$\\bar{\\nu}$')
    ax4.set_ylabel('$D_{KS}$')
    #ax4.set_title('comparing data vs model')
    
    plt.tight_layout()
    if self.para['plt_sv']:
      pathSv = pathcat([self.para['pathFigs'],'PC_analysis_get_noise.png'])
      plt.savefig(pathSv)
      print('Figure saved @ %s'%pathSv)
    plt.show(block=False)
    
    
    x_arr = np.linspace(0,100,1001)
    hbm = HierarchicalBayesModel(np.random.rand(1001),x_arr,parsNoise,1)
    A_0 = 0.6
    A = 1
    std = 5
    theta = 63
    TC = hbm.TC(np.array([A_0,A,std,theta]))
    
    plt.figure(figsize=(6,4),dpi=150)
    plt.bar(np.linspace(0,100,100),fr_mu,color='b',alpha=0.2,width=1,label='$\\bar{\\nu}$')
    plt.plot(x_arr,TC,'k',label='tuning curve model TC($x$;$A_0$,A,$\\sigma$,$\\theta$)')
    
    x_offset = 10
    y_arr = np.linspace(0,2.5,1001)
    alpha, beta = gamma_paras(A_0,A_0/2)
    x1 = sstats.gamma.pdf(y_arr,alpha,0,1/beta)
    x1 = -10*x1/x1.max()+x_offset
    x2 = x_offset*np.ones(1001)
    plt.plot(x_offset,A_0,'ko')
    plt.fill_betweenx(y_arr,x1,x2,facecolor='b',alpha=0.2,edgecolor=None)
    
    #idx = 550
    #x_offset = x_arr[idx]
    #plt.plot(x_offset,TC[idx],'ko')
    
    #alpha, beta = gamma_paras(TC[idx],TC[idx]/2)
    #x1 = sstats.gamma.pdf(y_arr,alpha,0,1/beta)
    #x1 = -10*x1/x1.max()+x_offset
    #x2 = x_offset*np.ones(1001)
    #plt.fill_betweenx(y_arr,x1,x2,facecolor='b',alpha=0.2,edgecolor=None,label='assumed noise')
    
    ### add text to show parameters
    plt_text = False
    if plt_text:
      plt.annotate("", xy=(theta+3*std, A_0), xytext=(theta+3*std, A_0+A),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
      plt.text(theta+3*std+2,A_0+A/2,'A')
      
      plt.annotate("", xy=(theta, A_0*0.9), xytext=(theta+2*std, A_0*0.9),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
      plt.text(theta+2,A_0*0.6,'$2\\sigma$')
      
      plt.annotate("", xy=(90, 0), xytext=(90, A_0),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
      plt.text(92,A_0/2,'$A_0$')
      
      plt.annotate("", xy=(theta, 0), xytext=(theta,A_0+A),
            arrowprops=dict(arrowstyle="-"))
      plt.text(theta,A_0+A*1.1,'$\\theta$')
    
    plt.xlabel('Position on track $x$')
    plt.ylabel('Ca$^{2+}$ event rate (model)')
    plt.yticks([])
    #plt.legend(loc='upper left')
    #plt.ylim([0,2.5])
    plt.tight_layout()
    plt.show(block=False)

#### ---------------- end of class definition -----------------


class HierarchicalBayesModel:
  
  ### possible speedup through... 
  ###   parallelization of code
  
  def __init__(self, data, x_arr, parsNoise, f):
    
    self.data = data
    self.parsNoise = parsNoise
    self.x_arr = x_arr
    self.x_max = x_arr.max()
    self.Nx = len(self.x_arr)
    self.change_model(f)
    
    self.nbin = 100
    
    ### steps for lookup-table (if used)
    #self.lookup_steps = 100000
    #self.set_beta_prior(5,4)
  
  def set_logl_func(self):
    def get_logl(p):
      if len(p.shape)==1:
        p = p[np.newaxis,:]
      p = p[...,np.newaxis]
      
      mean_model = np.ones((p.shape[0],self.Nx))*p[:,0,:]
      if p.shape[1] > 1:
        for j in [-1,0,1]:   ## loop, to have periodic boundary conditions
          
          #mean_model += p[:,1]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,3]+self.x_max*j)**2/(2*p[:,2]**2))
          mean_model += (p[:,slice(1,None,3),:]*np.exp(-(self.x_arr[np.newaxis,np.newaxis,:]-p[:,slice(3,None,3),:]+self.x_max*j)**2/(2*p[:,slice(2,None,3),:]**2))).sum(1)
      
      #plt.figure()
      #for i in range(p.shape[0]):
        #plt.subplot(p.shape[0],1,i+1)
        #plt.plot(self.x_arr,np.squeeze(mean_model[i,:]))
      #plt.show(block=False)
      
      SD_model = self.parsNoise*mean_model
      
      alpha = (mean_model/SD_model)**2
      beta = mean_model/SD_model**2
      
      logl = (alpha*np.log(beta) - np.log(sp.special.gamma(alpha)) + (alpha-1)*np.log(self.data) - beta*self.data ).sum(1)
      if self.f>1:
        p_theta = p[:,slice(3,None,3)]
        dTheta = np.squeeze(np.abs(np.mod(p_theta[:,1]-p_theta[:,0]+self.nbin/2,self.nbin)-self.nbin/2))
        logl[dTheta<10] = -1e300
      
      return logl
      
    return get_logl
  
  ### want beta-prior for std - costly, though, so define lookup-table
  def set_beta_prior(self,a,b):
    self.lookup_beta = sp.stats.beta.ppf(np.linspace(0,1,self.lookup_steps),a,b)
    #return sp.special.gamma(a+b)/(sp.special.gamma(a)*sp.special.gamma(b))*x**(a-1)*(1-x)**(b-1)
  
  def transform_p(self,p):
    
    
    if p.shape[-1]>1:
      p_out = p*self.prior_stretch + self.prior_offset
      #print(p_out[:,slice(3,None,3)])
      #if self.f > 1:
        #p_theta = p_out[:,slice(3,None,3)]
        #for f in range(1,self.f):
          #dTheta = np.diag(np.abs(np.mod(p_theta[:,f]-p_theta[:,:f]+nbin/2,nbin)-nbin/2))
          #idx_update = dTheta<10
          #while np.any(idx_update):
            #p_theta[idx_update,f] = np.random.uniform(self.prior_offset[3*f],self.prior_offset[3*f]+self.prior_stretch[3*f],idx_update.sum())
            
            #dTheta = np.diag(np.abs(np.mod(p_theta[:,f]-p_theta[:,:f]+nbin/2,nbin)-nbin/2))
            #idx_update = dTheta<10
        
        #p_out[:,slice(3,None,3)] = p_theta
        ####for i in range(p.shape[0]):
          ####if p_theta[i,1] < p_theta[i,0]:
            ####p1 = p_out[i,1:4]
            ####p2 = p_out[i,4:7]
            ####p_out[i,1:4] = p2
            ####p_out[i,4:7] = p1
          
          
      #p_out[...,2] = self.prior_stretch[2]*self.lookup_beta[(p[...,2]*self.lookup_steps).astype('int')]
    else:
      p_out = p*self.prior_stretch[0] + self.prior_offset[0]
    #print(p_out[:,slice(3,None,3)])
    return p_out
  
  def set_priors(self):
    self.prior_stretch = np.array(np.append(1,[1,9.5,100]*self.f))
    self.prior_offset = np.array(np.append(0,[0,1,0]*self.f))
  
  def change_model(self,f):
    self.f = f
    self.nPars = 1+3*f
    self.TC = self.build_TC_func()
    self.pTC = {}
    self.set_priors()
    self.pTC['wrap'] = np.zeros(self.nPars).astype('bool')
    self.pTC['wrap'][slice(3,None,3)] = True
  
    self.transform_ct = 0
  
  def build_TC_func(self):        ## general function to build tuning curve model
    def TC_func(p):
      if len(p.shape)==1:
        p = p[np.newaxis,:]
      p = p[...,np.newaxis]
      TC = np.ones((p.shape[0],self.Nx))*p[:,0]
      if p.shape[1] > 1:
        for j in [-1,0,1]:   ## loop, to have periodic boundary conditions
          #TC += p[:,1]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,3]+self.x_max*j)**2/(2*p[:,2]**2))
          TC += (p[:,slice(1,None,3)]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,slice(3,None,3)]+self.x_max*j)**2/(2*p[:,slice(2,None,3)]**2))).sum(-1)
      
      return np.squeeze(TC)
    
    return TC_func
 
####------------------------ end of HBM definition ----------------------------- ####






def load_activity(pathSession,dataSet='redetect'):
  ## load activity data from CaImAn results
  
  pathAct = pathcat([pathSession,'results_%s.mat'%dataSet])
  
  ld = sio.loadmat(pathAct,squeeze_me=True)
  S = ld['S']
  if S.shape[0] > 8000:
    S = S.transpose()
  
  if dataSet=='redetect':
    idx_evaluate = ld['idx_evaluate'].astype('bool')
    idx_previous = ld['idx_previous'].astype('bool')
    SNR = ld['SNR']
    r_values = ld['r_values']
    other = [idx_evaluate,idx_previous,SNR,r_values]
  else:
    other = None
  
  return S, other

def get_MI(p_joint,p_x,p_f):
  
  ### - joint distribution
  ### - behavior distribution
  ### - firing rate distribution
  ### - all normalized, such that sum(p) = 1
  
  p_tot = p_joint * np.log2(p_joint/(p_x[:,np.newaxis]*p_f[np.newaxis,:]))
  #return p_tot[p_joint>0].sum()
  #plt.figure()
  #plt.plot(p_tot.sum(1))
  #plt.show(block=False)
  return np.nansum(p_tot)

def _hsm(data,sort_it=True):
  ### adapted from caiman
  ### Robust estimator of the mode of a data set using the half-sample mode.
  ### versionadded: 1.0.3
    
  ### Create the function that we can use for the half-sample mode
  ### sorting done as first step, if not specified else
  
  data = data[np.isfinite(data)]
  if sort_it:
    data = np.sort(data)
  
  if data.size == 1:
      return data[0]
  elif data.size == 2:
      return data.mean()
  elif data.size == 3:
      i1 = data[1] - data[0]
      i2 = data[2] - data[1]
      if i1 < i2:
          return data[:2].mean()
      elif i2 > i1:
          return data[1:].mean()
      else:
          return data[1]
  else:

      wMin = np.inf
      N = data.size//2 + data.size % 2
      for i in range(N):
          w = data[i + N - 1] - data[i]
          if w < wMin:
              wMin = w
              j = i

      return _hsm(data[j:j + N])


def get_spikeNr(data):
  
  if np.count_nonzero(data)==0:
    return 0,np.NaN,np.NaN
  else:
    md = _hsm(data,True);       #  Find the mode
    
    # only consider values under the mode to determine the noise standard deviation
    ff1 = data - md;
    ff1 = -ff1 * (ff1 < 0);
    
    # compute 25 percentile
    ff1.sort()
    ff1[ff1==0] = np.NaN
    Ns = round((ff1>0).sum() * .5).astype('int')
    
    # approximate standard deviation as iqr/1.349
    iqr_h = ff1[-Ns];
    sd_r = 2 * iqr_h / 1.349;
    data_thr = md+2*sd_r;
    spikeNr = np.floor(data/data_thr).sum();
    return spikeNr,md,sd_r



def ecdf(x,p=None):
  
  if type(p)==np.ndarray:
    #assert abs(1-p.sum()) < 10**(-2), 'probability is not normalized, sum(p) = %5.3g'%p.sum()
    if abs(1-p.sum()) < 10**(-2):
      p /= p.sum()
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = np.cumsum(p[sort_idx])
  else:
    x = np.sort(x)
    y = np.cumsum(np.ones(x.shape)/x.shape)
  
  return x,y


def get_average(x,p,periodic=False,bounds=None):
  
  #assert abs(1-p.sum()) < 10**(-2), 'probability not normalized, sum(p) = %5.3g'%p.sum()
  if abs(1-p.sum()) < 10**(-2):
      p /= p.sum()
  if periodic:
    assert bounds, 'bounds not specified'
    L = bounds[1]-bounds[0]
    scale = L/(2*np.pi)
    avg = (cmath.phase((p*np.exp(+complex(0,1)*(x-bounds[0])/scale)).sum())*scale + bounds[0]) % L
  else:
    avg = (x*p).sum()
  return avg


#def get_average(x,p,periodic=False,bounds=None):
  
  ##assert abs(1-p.sum()) < 10**(-2), 'probability not normalized, sum(p) = %5.3g'%p.sum()
  #if abs(1-p.sum()) < 10**(-2):
      #p /= p.sum()
  #if periodic:
    #assert bounds, 'bounds not specified'
    #L = bounds[1]-bounds[0]
    #scale = L/(2*np.pi)
    #avg = (cmath.phase((p*periodic_to_complex(x,bounds)).sum())*scale) % L + bounds[0]
  #else:
    #avg = (x*p).sum()
  #return avg


def periodic_difference(x,y,bounds):
  scale = (bounds[1]-bounds[0])/(2*np.pi)
  print(scale)
  print(periodic_to_complex(x,bounds))
  print(periodic_to_complex(y,bounds))
  print(cmath.phase(periodic_to_complex(y,bounds) - periodic_to_complex(x,bounds)))
  diff = cmath.phase(periodic_to_complex(y,bounds) - periodic_to_complex(x,bounds))*scale + bounds[0]
  return diff
  

def periodic_to_complex(x,bounds):
  scale = (bounds[1]-bounds[0])/(2*np.pi)
  return np.exp(complex(0,1)*(x-bounds[0])/scale)

def complex_to_periodic(phi,bounds):
  L = bounds[1]-bounds[0]
  scale = L/(2*np.pi)
  
  return (cmath.phase(phi)*scale) % L + bounds[0] 
  


def jackknife(X,Y,W=None):
  
  ## jackknifing a linear fit (with possible weights)
  ## W_i = weights of value-tuples (X_i,Y_i)
  
  if len(X.shape) < 2:
    X = X[:,np.newaxis]
  
  if type(W) == np.ndarray:
    print('weights given (not working)')
    W = np.ones(Y.shape)
    Xw = X * np.sqrt(W[:,np.newaxis])
    Yw = Y * np.sqrt(W)
  else:
    Xw = X
    Yw = Y
  
  N_data = len(Y);
  
  fit_jk = np.zeros(N_data);
  mask_all = np.ones(N_data).astype('bool')
  for i in range(N_data):
    mask = np.copy(mask_all)
    mask[i] = False;
    try:
      fit_jk[i] = np.linalg.lstsq(Xw[mask,:],Yw[mask])[0]
    except:
      fit_jk[i] = np.NaN
  
  return np.nanmean(fit_jk)

### -------------- lognorm distribution ---------------------
def lognorm_paras(mean,sd):
  shape = np.sqrt(np.log(sd/mean**2+1))
  mu = np.log(mean/np.sqrt(sd/mean**2 + 1))
  return mu, shape

### -------------- Gamma distribution -----------------------
def gamma_paras(mean,SD):
  alpha = (mean/SD)**2
  beta = mean/SD**2
  return alpha, beta


def pathcat(strings):
  return '/'.join(strings)
