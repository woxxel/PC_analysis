import os, random, h5py, time, math, cmath, copy, importlib, warnings, pickle
import multiprocessing as mp

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
    
    
  def run_detection(self,S=None,rerun=False,return_results=False,specific_n=False,artificial=False,redetect=True):
    
    global t_start
    t_start = time.time()
    
    self.para['modes']['info'] = 'MI'
    if S is None:
      S, other = load_activity(self.para['pathSession'],redetect=True)
      if redetect:
        idx_evaluate = other[0]
        idx_previous = other[1]
        SNR = other[2]
        r_values = other[3]
    
    nCells = S.shape[0]
    
    if specific_n:
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
      
      self.para['modes']['info'] = False
      
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
        
        pool = mp.Pool(self.para['nP'])
        batchSz = 500
        nBatch = nCells_process//batchSz
        
        for i in range(nBatch+1):
          idx_batch = idx_process[i*batchSz:min(nCells_process,(i+1)*batchSz)]
          #result_tmp.extend(pool.map(self.PC_detect,S[idx_batch,:]))
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
  
  
  def get_behavior(self):
    
    for file in os.listdir(self.para['pathSession']):
      if file.endswith("aligned.mat"):
          pathBH = os.path.join(self.para['pathSession'], file)
    
    f = h5py.File(pathBH,'r')
    key_array = ['binpos','time','longrunperiod']
    
    load_behavior = {}
    for key in key_array:
      load_behavior[key] = np.squeeze(f.get('alignedData/resampled/%s'%key).value)
    f.close()
    
    self.dataBH = {}
    self.dataBH['longrunperiod'] = load_behavior['longrunperiod'].astype(bool)
    self.dataBH['binpos'] = load_behavior['binpos'][self.dataBH['longrunperiod']].astype('int') - 1 ## correct for different indexing
    self.dataBH['binpos_raw'] = load_behavior['binpos'].astype('int') - 1 ## correct for different indexing
    self.dataBH['binpos_coarse'] = (self.dataBH['binpos']/self.para['coarse_factor']).astype('int')
    self.dataBH['time'] = load_behavior['time'][self.dataBH['longrunperiod']]
    self.dataBH['time_raw'] = load_behavior['time']
    self.dataBH['T'] = np.count_nonzero(self.dataBH['longrunperiod'])
    
    #plt.figure()
    #plt.plot(self.dataBH['time'],self.dataBH['binpos_coarse'],'r')
    #plt.plot(self.dataBH['time'],self.dataBH['binpos'],'b')
    #plt.show()
    
    ###### define trials
    self.dataBH['trials'] = {}
    self.dataBH['trials']['frame'] = np.hstack([0, np.where(np.diff(self.dataBH['binpos'])<-10)[0]+1,len(self.dataBH['time'])-1])
    self.dataBH['trials']['t'] = np.hstack([self.dataBH['time'][self.dataBH['trials']['frame'][:-1]],self.dataBH['time'][-1]])
    dt = np.diff(self.dataBH['trials']['t'])
    ct = len(self.dataBH['trials']['frame'])-1
    if dt[0] < 2:
      self.dataBH['trials']['t'] = np.delete(self.dataBH['trials']['t'],0)
      self.dataBH['trials']['frame'] = np.delete(self.dataBH['trials']['frame'],0)
      ct -=1
    if dt[-1] < 2:
      self.dataBH['trials']['t'] = np.delete(self.dataBH['trials']['t'],ct)
      self.dataBH['trials']['frame'] = np.delete(self.dataBH['trials']['frame'],ct)
      ct -=1
    
    self.dataBH['trials']['ct'] = ct
    self.dataBH['trials']['dwelltime'] = np.zeros((self.dataBH['trials']['ct'],self.para['nbin']))
    self.dataBH['trials']['T'] = np.zeros(self.dataBH['trials']['ct']).astype('int')
    
    t_offset = 0
    self.dataBH['trials']['trial'] = {}
    for t in range(self.dataBH['trials']['ct']):
      self.dataBH['trials']['trial'][t] = {}
      self.dataBH['trials']['trial'][t]['binpos'] = self.dataBH['binpos'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]]
      self.dataBH['trials']['dwelltime'][t,:] = np.histogram(self.dataBH['trials']['trial'][t]['binpos'],self.para['bin_array_centers'])[0]/self.para['f']
      self.dataBH['trials']['T'][t] = len(self.dataBH['trials']['trial'][t]['binpos'])
    return self.dataBH


  def PC_detect(self,S,SNR=None,r_value=None):
  #def PC_detect(varin):
    result = self.build_PC_fields()
    S[S<0] = 0
    if not (SNR is None):
      result['status']['SNR'] = SNR
      result['status']['r_value'] = r_value
      
    #try:
    active = {}
    active['S'] = S[self.dataBH['longrunperiod']]    ### only consider activity during continuous runs
    if np.count_nonzero(active['S'])==0:
      print('no activity for this neuron')
      result['firingstats']['rate'] = 0
      return result
    ## normalize activity by baseline (obtained via half-sampling mode)
    baseline = _hsm(active['S'][active['S']>0])
    active['S'] /= baseline
    
    ### calculate firing rate
    [spikeNr,md,sd_r] = get_spikeNr(active['S'][active['S']>0])
    
    #plt.figure()
    #plt.subplot(212)
    
    #plt.scatter(self.dataBH['time'],self.dataBH['binpos'],marker='.',s=3)
    #S_plot = S/(md+2*sd_r);
    #plt.scatter(self.dataBH['time'][active['S']>1],self.dataBH['binpos'][active['S']>1],color='r',s=10+S_plot[self.dataBH['longrunperiod']&(S>(md+2*sd_r))]*2)
    #plt.subplot(211)
    
    #plt.plot(S)
    #plt.show(block=False)
    
    #print("spike nr: %d"%spikeNr)
    result['firingstats']['rate'] = spikeNr / (self.dataBH['T']/self.para['f'])
    
    if self.para['modes']['info']:
      if self.para['modes']['info'] == 'MI':
        ## obtain quantized firing rate for MI calculation
        active['qtl'] = sp.ndimage.gaussian_filter(S,self.para['sigma'])
        active['qtl'] = active['qtl'][self.dataBH['longrunperiod']]
        qtls = np.quantile(active['qtl'][active['qtl']>0],np.linspace(0,1,self.para['qtl_steps']+1))
        active['qtl'] = np.count_nonzero(active['qtl'][:,np.newaxis]>=qtls[np.newaxis,1:-1],1)
    
    ### get trial-specific activity
    trials_S, trials_firingmap = self.get_trials_activity(active)
    
    ## obtain firingmap
    firingstats_tmp = self.get_firingstats_from_trials(trials_firingmap)
    for key in firingstats_tmp.keys():
      result['firingstats'][key] = firingstats_tmp[key]
    #print(result['firingstats']['parNoise'])
    
    if self.para['modes']['info']:
      ## obtain mutual information first - check if (computational cost of) finding fields is worth it at all
      MI_tmp = self.test_MI(active,trials_S)
      for key in MI_tmp.keys():
        result['status'][key] = MI_tmp[key]
    else:
      result['status']['MI_value'] = np.nan
      result['status']['MI_p_value'] = np.nan
      result['status']['MI_z_score'] = np.nan
      
      
    ### do further tests only if there is "significant" mutual information
    #if (not self.para['modes']['info']) | (result['status']['MI_p_value']<0.2):
      
    #### -------------------------- tuning curve model -------------------------------------
    #PC_fields['firingmap'] = savgol_filter(PC_fields['firingmap'], 5, 3)
    #self.para['bin_array'] = np.linspace(2*math.pi/self.para['nbin'],2*math.pi,self.para['nbin']);
    
    hbm = HierarchicalBayesModel(result['firingstats']['map'],self.para['bin_array'],result['firingstats']['parNoise'],0)
    
    ### test models with 0 vs 1 fields
    for f in range(2):
      
      hbm.change_model(f)
      
      #tic = time.time()
      
      ## hand over functions for sampler
      my_prior_transform = hbm.transform_p
      my_likelihood = hbm.set_logl_func()
      
      sampler = ultranest.ReactiveNestedSampler(self.para['names'][:hbm.nPars], my_likelihood, my_prior_transform,wrapped_params=hbm.pTC['wrap'],vectorized=True,num_bootstraps=30)#,log_dir='/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Programs/PC_analysis/test_ultra')   ## set up sampler...
      sampling_result = sampler.run(frac_remain=0.5,min_num_live_points=(f+1)*200,show_status=False,viz_callback=False)  ## ... and run it
      
      A_0 = sampling_result['posterior']['mean'][0]
      
      result['status']['Z'][f,:] = [sampling_result['logz'],sampling_result['logzerr']]    ## store evidences
      
      if f > 0:
        #toc = time.time()
        #print('time taken: %5.3gs'%(toc-tic))
        
        result['status']['Bayes_factor'][0] = result['status']['Z'][1,0]-result['status']['Z'][0,0]
        result['status']['Bayes_factor'][1] = np.sqrt(result['status']['Z'][0,1]**2 + result['status']['Z'][1,1]**2)
        bayes_thr = result['status']['Bayes_factor'][0] - 1.96*result['status']['Bayes_factor'][1]
        
        #if (result['status']['Z'][1,0]-result['status']['Z'][1,1]) > (result['status']['Z'][0,0]+result['status']['Z'][0,1]):
        #if bayes_thr > 1:
        fields_tmp = self.detect_modes_from_posterior(sampler)
        for key in fields_tmp.keys():
          result['fields'][key] = fields_tmp[key]
          
        #if self.para['plt_bool']:
          #cornerplot(sampling_result)
      
    if self.para['plt_bool']:
      print('for display: draw tuning curves from posterior distribution and evaluate TC-value for each bin. then, each bin has distribution of values and can be plotted! =)')
      style_arr = ['--','-']
      #col_arr = []
      fig,ax = plt.subplots(figsize=(5,3),dpi=150)
      ax.bar(self.para['bin_array'],result['firingstats']['map'],facecolor='b',width=100/self.para['nbin'],alpha=0.2)
      ax.errorbar(self.para['bin_array'],result['firingstats']['map'],result['firingstats']['CI'],ecolor='r',linestyle='',fmt='',elinewidth=0.3)#,label='$95\\%$ confidence')
      
      ax.plot(self.para['bin_array'],hbm.TC(np.array([A_0])),'k',linestyle='--',linewidth=1,label='(constant)\t $log(Z)=%5.2g\\pm%5.2g$'%(result['status']['Z'][0,0],result['status']['Z'][0,1]))
      
      #try:
      #print(result['fields']['nModes'])
      for c in range(min(2,result['fields']['nModes'])):
        if result['fields']['nModes']>1:
          if c==0:
            label_str = '(mode #%d)\t$log(Z)=%5.2g\\pm%5.2g$'%(c+1,result['status']['Z'][1,0],result['status']['Z'][1,1])
          else:
            label_str = '(mode #%d)'%(c+1)
        else:
          label_str = '(field)\t $log(Z)=%5.2g\\pm%5.2g$'%(result['status']['Z'][1,0],result['status']['Z'][1,1])
        
        ax.plot(self.para['bin_array'],hbm.TC(result['fields']['parameter'][c,:,0]),'r',linestyle='-',linewidth=0.5+result['fields']['posterior_mass'][c]*2,label=label_str)
        #except:
          #1
      #ax.plot(self.para['bin_array'],hbm.TC(par_results[1]['mean']),'r',label='$log(Z)=%5.3g\\pm%5.3g$'%(par_results[1]['Z'][0],par_results[1]['Z'][1]))
      ax.legend(title='evidence from nested sampling')
      ax.set_xlabel('Position on track')
      ax.set_ylabel('Ca$^{2+}$ event rate')
      plt.tight_layout()
      if self.para['plt_sv']:
        pathSv = pathcat([self.para['pathFigs'],'PC_analysis_fit_results.png'])
        plt.savefig(pathSv)
        print('Figure saved @ %s'%pathSv)
      plt.show(block=True)
    
    print('p-value (MI): %5.3g, z-score (MI): %5.3g, \t bayes factor: %7.5g+/-%7.5g \t SNR: %5.3g,\t r_value: %5.3g \t time passed: %7.2fs'%(result['status']['MI_p_value'],result['status']['MI_z_score'],result['status']['Bayes_factor'][0],result['status']['Bayes_factor'][1],SNR,r_value,time.time()-t_start))
    #global n_ct
    #print('tried')
    #n_ct.increment()
    #print('%d neurons processed, \t %7.5gs passed'%(n_ct.value,time.time()-t_start))
    #except (KeyboardInterrupt, SystemExit):
      #raise
    #except:
      #print('analysis failed: (-) p-value (MI): %5.3g, \t bayes factor: %7.5g+/-%7.5g \t SNR: %5.3g,\t r_value: %5.3g \t time passed: %7.2fs'%(result['status']['MI_p_value'],result['status']['Bayes_factor'][0],result['status']['Bayes_factor'][1],SNR,r_value,time.time()-t_start))
      #result['fields']['nModes'] = -1
      
    return result


  def detect_modes_from_posterior(self,sampler):
    ### handover of sampled points
    data_tmp = ultranest.netiter.logz_sequence(sampler.root,sampler.pointpile)[0]
    logp_prior = np.log(-0.5*(np.diff(np.exp(data_tmp['logvol'][1:]))+np.diff(np.exp(data_tmp['logvol'][:-1])))) ## calculate prior probabilities (phasespace-slice volume from change in prior-volume (trapezoidal form)
    
    data = {}
    data['logX'] = data_tmp['logvol'][1:-1]
    data['logl'] = data_tmp['logl'][1:-1]
    data['logz'] = data_tmp['logz'][1:-1]
    data['logp_posterior'] = logp_prior + data['logl'] - data['logz'][-1]   ## normalized posterior weight
    data['samples'] = data_tmp['samples'][1:-1,:]
    data['pos_samples'] = data['samples'][:,3]
    
    ### get number of clusters:
    ### ...slice at different levels 
    
    logX_top = logX_top_tmp = -(data['logX'].min())
    logX_bottom = logX_bottom_tmp = -(data['logX'].max())
    
    ### search for baseline (where whole prior space is sampled)
    x_space = np.linspace(0,100,11)
    i=0
    while i < 10:
      logX_base = (logX_top_tmp + logX_bottom_tmp)/2
      mask_logX = -data['logX']>logX_base
      cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
      if np.mean(cluster_hist) > 0.9:
        logX_bottom_tmp = logX_base
      else:
        logX_top_tmp = logX_base
      i+=1
    
    ### check from baseline up
    ### gather occupied phase space & occupied probability mass
    x_space = np.linspace(0,100,101)
    
    nsteps = 31
    clusters = {}
    blob_center = np.zeros((nsteps,1))*np.NaN
    blob_phase_space = np.zeros((nsteps,1))*np.NaN
    blob_probability_mass = np.zeros((nsteps,1))*np.NaN
    blob_center_CI = np.zeros((nsteps,2,1))*np.NaN
    
    periodic = False
    nClusters = 0
    logX_arr = np.linspace(logX_top,logX_base,nsteps)
    for (logX,i) in zip(logX_arr,range(nsteps)):
      mask_logX = -data['logX']>logX
      
      cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
      ## remove "noise" clusters
      cluster_hist = sp.ndimage.morphology.binary_opening(sp.ndimage.morphology.binary_closing(np.concatenate((cluster_hist[-10:],cluster_hist,cluster_hist[:10]))))[10:-10]
      if not any(cluster_hist):
        continue
      
      blobs = measure.label(cluster_hist)
      nblobs = blobs.max()
      
      if (blobs[0]>0) & (blobs[-1]>0):
        periodic = True
        nblobs-=1
      
      for c in range(1,nblobs+1):
        
        if len(blob_phase_space) < nblobs:
          blob_center.append([])
          blob_phase_space.append([])
          blob_probability_mass.append([])
          blob_center_CI.append([])
          
        if c == 1 & periodic:
          val_last = x_space[np.where(blobs==c)[0][-1]]
          val_first = x_space[np.where(blobs==blobs[-1])[0][0]]
          mask_cluster = np.logical_or((val_first <= data['pos_samples']),(data['pos_samples'] < val_last))
          blobs[blobs==blobs[-1]] = blobs[0]      ## assign components to each other, wrapped by periodicity of phase space
          
        else:
          val_first = x_space[np.where(blobs==c)[0][0]]
          val_last = x_space[np.where(blobs==c)[0][-1]]
          mask_cluster = (val_first <= data['pos_samples']) & (data['pos_samples'] < val_last)
        
        mask = (mask_cluster & mask_logX)
        
        if np.count_nonzero(mask)>50:
          p_posterior_cluster = np.exp(data['logp_posterior'][mask])
          posterior_mass = (p_posterior_cluster).sum()
          p_posterior_cluster /= posterior_mass
          masked_samples = data['pos_samples'][mask]
          
          center_tmp = get_average(masked_samples,p_posterior_cluster,True,[0,100])  ## project to positive axis again
          
          ### test for overlap with other clusters
          assigned = overlap = False
          for cc in clusters.keys():
            
            mask_joint = mask & clusters[cc]['mask']
              
            if any(mask_joint):     ### does the cluster have an overlap with another one?
              
              if clusters[cc]['active']:    ### clusters can only be assigned to active clusters
              
                if assigned:
                  
                  if (clusters[c_id]['posterior_mass'] > clusters[cc]['posterior_mass']):
                    clusters[cc]['active'] = False
                    if clusters[cc]['appeared'] > 3:
                      clusters[c_id]['active'] = False
                      overlap = True
                  else:
                    clusters[c_id]['active'] = False
                    if clusters[c_id]['appeared'] > 3:
                      clusters[cc]['active'] = False
                      overlap = True
                  
                elif ((np.exp(data['logp_posterior'][mask_joint]).sum() / clusters[cc]['posterior_mass']) > 0):# & abs((center_tmp -:
                  ### if it's a significant overlap, not changing the clusterstats significantly, assign to existing one
                  c_id = cc
                  appeared = clusters[c_id]['appeared']+1
                  assigned = True
                else:
                  ### if overlap is non-significant, disable "small" cluster
                  clusters[cc]['active'] = False
              else:
                overlap = True
              
          if overlap:
            continue
          elif not assigned:
            ### if no overlap was found, start new cluster
            c_id = nClusters
            nClusters+=1
            appeared = 1
          
          n_samples = len(p_posterior_cluster)
          
          clusters[c_id] = {'periodic':periodic & (c==1),'mask':mask,'center':center_tmp,'phase_space':np.mean(blobs==c),'posterior_mass':posterior_mass,'baseline':logX,'n_samples':n_samples,'appeared':appeared,'active':True}
          
          if c_id >= blob_center.shape[1]:
            cat_cluster = np.zeros((nsteps,1))*np.nan
            blob_center = np.concatenate((blob_center,cat_cluster),1)
            blob_phase_space = np.concatenate((blob_phase_space,cat_cluster),1)
            blob_probability_mass = np.concatenate((blob_probability_mass,cat_cluster),1)
            
            cat_cluster = np.zeros((nsteps,2,1))*np.nan
            blob_center_CI = np.concatenate((blob_center_CI,cat_cluster),2)
          
          blob_center[i,c_id] = center_tmp
          blob_phase_space[i,c_id] = np.mean(blobs==c)
          blob_probability_mass[i,c_id] = posterior_mass
          
          ### get confidence intervals from cdf
          x_cdf_posterior, y_cdf_posterior = ecdf(masked_samples,p_posterior_cluster)
          blob_center_CI[i,0,c_id] = x_cdf_posterior[np.where(y_cdf_posterior>=0.025)[0][0]]
          blob_center_CI[i,1,c_id] = x_cdf_posterior[np.where(y_cdf_posterior>0.975)[0][0]]
        
    #print(clusters)
    #print(blob_center)
    #print(blob_center_CI)
    #print(blob_phase_space)
    #print(blob_probability_mass)
    
    ### remove overlapping and non-significant clusters
    logX_baseline = np.inf
    c_ct=0
    c_arr = []
    for c in range(nClusters):
      clusters[c]['active'] = True
      if c in clusters:
        if (clusters[c]['posterior_mass'] < 0.05):
          clusters[c]['active'] = False
          continue
        if (clusters[c]['appeared'] <= 3):
          clusters[c]['active'] = False
          continue
        c_arr.append(c)
      logX_baseline = min(clusters[c]['baseline'],logX_baseline)
      c_ct+=1
    
    ### for each cluster, obtain self.para values and confidence intervals
    fields = {}
    fields['nModes'] = min(3,c_ct)
    fields['posterior_mass'] = np.zeros(3)*np.NaN
    fields['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
    fields['p_x'] = np.zeros((3,2,self.para['nbin']))*np.NaN
    
    #print('masses:')
    #print(clusters)
    
    mask_logX = -data['logX']>logX_baseline
    cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
    ## remove "noise" clusters
    cluster_hist = sp.ndimage.morphology.binary_opening(sp.ndimage.morphology.binary_closing(np.concatenate((cluster_hist[-10:],cluster_hist,cluster_hist[:10]))))[10:-10]
    
    blobs = measure.label(cluster_hist)
    if (blobs[0]>0) & (blobs[-1]>0):
      periodic = True
    masks = {}
    for i in range(1,blobs.max()):
      masks[i] = {}
      if i == 1 & periodic:
        val_last = x_space[np.where(blobs==i)[0][-1]]
        val_first = x_space[np.where(blobs==blobs[-1])[0][0]]
        mask_cluster = np.logical_or((val_first <= data['pos_samples']),(data['pos_samples'] < val_last))
        #blobs[blobs==blobs[-1]] = blobs[0]      ## assign components to each other, wrapped by periodicity of phase space
      else:
        val_first = x_space[np.where(blobs==i)[0][0]]
        val_last = x_space[np.where(blobs==i)[0][-1]]
        mask_cluster = (val_first <= data['pos_samples']) & (data['pos_samples'] < val_last)
      masks[i]['baseline'] = (mask_cluster & mask_logX)
    
    c_ct = 0
    sig_space = np.linspace(0,10,101)
    for c in clusters.keys():
      if clusters[c]['active']:
        mask = np.ones(data['samples'].shape[0]).astype('bool')
        #posterior_mass = 1
        #print(clusters[c])
        #plt.figure()
        #plt.subplot(131)
        #plt.scatter(data['pos_samples'][mask],-data['logX'][mask],c=np.exp(data['logp_posterior'][mask]),marker='.',label='samples')
        for i in range(1,len(masks)):
          if np.count_nonzero(masks[i]['baseline'] & clusters[c]['mask'])==0:
            mask[masks[i]['baseline']] = False
        #plt.subplot(132)
        #plt.scatter(data['pos_samples'][mask],-data['logX'][mask],c=np.exp(data['logp_posterior'][mask]),marker='.',label='samples')
        
        ## find, which cluster mask belongs to
        for cc in clusters.keys():
          if not cc==c:
            mask[clusters[cc]['mask']] = False
            #posterior_mass -= clusters[cc]['posterior_mass']
        
        #plt.subplot(133)
        #plt.scatter(data['pos_samples'][mask],-data['logX'][mask],c=np.exp(data['logp_posterior'][mask]),marker='.',label='samples')
        #plt.show(block=False)
        
        p_posterior_cluster = np.exp(data['logp_posterior'][mask])#/posterior_mass
        p_posterior_cluster /= p_posterior_cluster.sum()
        
        ## calculate other parameters
        for i in range(4):
          samples = data['samples'][mask,i]
          #samples = data['samples'][clusters[c]['mask'],i]
            
          if i==2:
            for j in range(len(sig_space)-1):
              thr_low = sig_space[j]
              thr_up = sig_space[j+1]
              mask_px = np.ones(samples.shape).astype('bool')
              mask_px[samples<thr_low] = False
              mask_px[samples>=thr_up] = False
              fields['p_x'][c_ct,0,j] = p_posterior_cluster[mask_px].sum()
          if i==3:
            for j in range(len(x_space)-1):
              thr_low = x_space[j]
              thr_up = x_space[j+1]
              mask_px = np.ones(samples.shape).astype('bool')
              mask_px[samples<thr_low] = False
              mask_px[samples>=thr_up] = False
              fields['p_x'][c_ct,1,j] = p_posterior_cluster[mask_px].sum()
            #fields['parameter'][c_ct,i,0] = get_average(data['samples'][clusters[c]['mask'],i],p_posterior_cluster,True,[0,100])
            fields['parameter'][c_ct,i,0] = get_average(samples,p_posterior_cluster,True,[0,100])
            samples = (data['samples'][mask,i]+100/2-fields['parameter'][c_ct,i,0])%100-100/2        ## shift whole axis such, that peak is in the center, to get proper errorbars
          else:
            #fields['parameter'][c_ct,i,0] = get_average(data['samples'][clusters[c]['mask'],i],p_posterior_cluster)
            fields['parameter'][c_ct,i,0] = get_average(samples,p_posterior_cluster)
          
          ### get confidence intervals from cdf
          x_cdf_posterior, y_cdf_posterior = ecdf(samples,p_posterior_cluster)
          for j in range(len(self.para['CI_arr'])):
            fields['parameter'][c_ct,i,1+j] = x_cdf_posterior[np.where(y_cdf_posterior>=self.para['CI_arr'][j])[0][0]]
          #fields['parameter'][c_ct,i,2] = x_cdf_posterior[np.where(y_cdf_posterior>0.975)[0][0]]
          
          if i==3:
            fields['parameter'][c_ct,i,0] = fields['parameter'][c_ct,i,0] % 100
            fields['parameter'][c_ct,i,1:] = (fields['parameter'][c_ct,i,0] + fields['parameter'][c_ct,i,1:]) % 100
            
            fields['parameter']
          
          fields['posterior_mass'][c_ct] = clusters[c]['posterior_mass']
        
        if c_ct == 2:
          break   ## maximum 2 clusters detected. should be kinda sorted, since they are detected starting from smallest lnX
        c_ct += 1
        #cc+=1
          #print('val: %5.3g, \t (%5.3g,%5.3g)'%(val[c,i],CI[c,i,0],CI[c,i,1]))
    #print('time took (post-process posterior): %5.3g'%(time.time()-t_start))
    #print(fields['parameter'])
    if self.para['plt_bool']:
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
      
      fig = plt.figure(figsize=(5,4),dpi=150)
      ax_NS = plt.subplot(position=[0.13,0.15,0.32,0.75])
      ax_prob = plt.subplot(position=[0.6,0.675,0.35,0.275])
      ax_center = plt.subplot(position=[0.6,0.375,0.35,0.275])
      ax_phase = plt.subplot(position=[0.6,0.15,0.35,0.2])
      
      ax_NS.scatter(data['pos_samples'],-data['logX'],c=np.exp(data['logp_posterior']),marker='.',label='samples')
      ax_NS.plot([0,100],[logX_base,logX_base],'k--')
      ax_NS.set_xlabel('field position $\\theta$')
      ax_NS.set_ylabel('-ln(X)')
      ax_NS.legend(loc='lower right')
      for c in range(fields['nModes']):
        #if fields['posterior_mass'][c] > 0.05:
        ax_center.plot(logX_arr,blob_center[:,c],color=col_arr[c])
        ax_center.fill_between(logX_arr,blob_center_CI[:,0,c],blob_center_CI[:,1,c],facecolor=col_arr[c],alpha=0.5)
        
        ax_phase.scatter(data['samples'][clusters[c_arr[c]]['mask'],3],data['samples'][clusters[c_arr[c]]['mask'],2],marker='.')
        #ax_phase.plot(logX_arr,blob_phase_space[:,c],color=col_arr[c],label='mode #%d'%(c+1))
        ax_prob.plot(logX_arr,blob_probability_mass[:,c],color=col_arr[c])
        
        if c < 3:
          ax_NS.annotate('',(fields['parameter'][c,3,0],logX_top),xycoords='data',xytext=(fields['parameter'][c,3,0]+5,logX_top+2),arrowprops=dict(facecolor=ax_center.lines[-1].get_color(),shrink=0.05))
      
      
      ax_center.set_xticks([])
      ax_center.set_xlim([logX_base,logX_top])
      ax_prob.set_xlim([logX_base,logX_top])
      ax_center.set_ylim([0,100])
      ax_center.set_ylabel('$\\theta$')
      ax_prob.set_ylim([0,1])
      ax_prob.set_xlabel('-ln(X)')
      ax_prob.set_ylabel('posterior')
      
      #ax_phase.set_xticks([])
      #ax_phase.set_xlim([logX_base,logX_top])
      #ax_phase.set_ylim([0,1])
      #ax_phase.set_ylabel('% phase space')
      #ax_phase.legend(loc='upper right')
      
      if self.para['plt_sv']:
          pathSv = pathcat([self.para['pathFigs'],'PC_analysis_NS_results.png'])
          plt.savefig(pathSv)
          print('Figure saved @ %s'%pathSv)
      plt.show(block=False)
      
      
    return fields


  def get_trials_activity(self,active):
    
    ## preallocate
    trials_map = np.zeros((self.dataBH['trials']['ct'],self.para['nbin']))
    #trials_rate = np.zeros(self.dataBH['trials']['ct'])
    
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
        trials_map[t,:] = get_firingmap(trials_S[t]['S'],self.dataBH['trials']['trial'][t]['binpos'],self.dataBH['trials']['dwelltime'][t,:],False)/trials_S[t]['rate']
      
      
      #[spikeNr,md,sd_r] = get_spikeNr(trials_S[t]['S'][trials_S[t]['S']>0]);
      #trials_rate[t] = spikeNr/(self.dataBH['trials']['T'][t]/self.para['f']);
      
    return trials_S, trials_map#, trials_rate


  def get_firingstats_from_trials(self,trials_firingmap):
    
    firingstats = {}
    ### construct firing rate map from bootstrapping over trials (is that valid? what does it imply?)
    firingmap_bs = np.zeros((self.para['nbin'],self.para['N_bs']))
    for L in range(self.para['N_bs']):
      base_sample = np.random.randint(0,self.dataBH['trials']['ct'],(self.dataBH['trials']['ct'],))
      dwelltime = self.dataBH['trials']['dwelltime'][base_sample,:].sum(0)
      firingmap_bs[:,L] = trials_firingmap[base_sample,:].sum(0)/dwelltime
      mask = (dwelltime==0)
      firingmap_bs[mask,L] = 0
    firingstats['map'] = np.nanmean(firingmap_bs,1)
    firingstats['map'][firingstats['map']<=0] = 1/self.dataBH['T']     ## set 0 firing rates to lowest possible (0 leads to problems in model, as 0 noise, thus likelihood = 0)
    
    ### estimate noise of model
    ## parameters of gamma distribution can be directly inferred from mean and std
    firingstats['std'] = np.nanstd(firingmap_bs,1)
    prc = [2.5,97.5]
    firingstats['CI'] = np.nanpercentile(firingmap_bs,prc,1);   ## width of gaussian - from 1-SD confidence interval
    ### fit linear dependence of noise on amplitude (with 0 noise at fr=0)
    firingstats['parNoise'] = jackknife(firingstats['map'],firingstats['std'])
    
    if self.para['plt_theory_bool']:
      self.plt_model_selection(firingmap_bs,firingstats['parNoise'],trials_firingmap)
    
    return firingstats


  def get_info_value(self,activity, dwelltime):
    
    #if self.para['modes']['info'] == 'MI':
      
    t_joint = time.time()
    p_joint = self.get_p_joint(activity)   ## need activity trace
    t_MI = time.time()
    MI = get_MI(p_joint,dwelltime,self.para['qtl_weight'])
    firingmap = []
      
    #elif self.para['modes']['info'] == 'Isec':
      
      #firingmap = imgaussfilt(get_firingmap(activity,binpos,dwelltime,self.para,True),self.para['sigma'],
                              #'Padding','circular','FilterDomain','spatial')
      #MI = get_I_sec(firingmap,dwelltime,self.para)
      
    return MI#, firingmap


  def get_p_joint(self,activity):
    
    ### need as input:
    ### - activity (quantiled or something)
    ### - behavior trace
    p_joint = np.zeros((self.para['nbin_coarse'],self.para['qtl_steps']))
    for f in range(self.para['qtl_steps']):
      for (x,ct) in Counter(self.dataBH['binpos_coarse'][activity==f]).items():
        p_joint[x,f] = ct;
    p_joint = p_joint/p_joint.sum();    ## normalize
    return p_joint


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
    MI['MI_value'] = self.get_info_value(active[S_key],norm_dwelltime_coarse);    
    
    ### shuffle according to specified mode
    #t_start_shuffle = time.time()
    
    for L in range(self.para['repnum']):
      
      ## shift single trials to destroy characteristic timescale
      if self.para['modes']['shuffle'] == 'shuffle_trials':
        
        ## trial shuffling
        shuffled_activity = np.zeros(self.dataBH['T'])
        idx = 0
        for t in np.random.permutation(self.dataBH['trials']['ct']):
          if np.nonzero(trials_S[t][S_key]):
            shuffled_activity[idx:idx+self.dataBH['trials']['T'][t]] = shuffling('shift',shuffle_peaks,spike_train=trials_S[t][S_key])
          else:
            shuffled_activity[idx:idx+self.dataBH['trials']['T'][t]] = trials_S[t][S_key]
          
          idx = idx+self.dataBH['trials']['T'][t]
        
        ## subsequent global shuffling
        if self.para['modes']['activity'] == 'spikes':
          spike_times_L = np.where(shuffled_activity)[0]
          spikes_L = shuffled_activity[spike_times_L]
          ISI_L = np.diff(spike_times_L)
          shuffled_activity = shuffling('dithershift',shuffle_peaks,spike_times=spike_times_L,spikes=spikes_L,T=self.dataBH['T'],ISI=ISI_L,w=2*self.para['f'])
        else:
          shuffled_activity = shuffling('shift',shuffle_peaks,spike_train=shuffled_activity)
        
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
      MI['MI_p_value'] = 1/self.para['repnum']
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
                        'Z':np.zeros((2,2))*np.NaN,
                        'Bayes_factor':np.zeros(2)*np.NaN,
                        'SNR':np.NaN,
                        'r_value':np.NaN}
    
    result['fields'] = {'parameter':np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN,          ### (field1,field2) x (A0,A,std,theta) x (mean,CI_low,CI_top)
                        'posterior_mass':np.zeros(3)*np.NaN,
                        'p_x':np.zeros((3,2,self.para['nbin']))*np.NaN,
                        'nModes':0}
    
    result['firingstats'] = {'rate':np.NaN,
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
                        'Z':np.zeros((nCells,2,2))*np.NaN,
                        'Bayes_factor':np.zeros((nCells,2))*np.NaN,
                        'SNR':np.zeros(nCells)*np.NaN,
                        'r_value':np.zeros(nCells)*np.NaN}
                  
    results['fields'] = {'parameter':np.zeros((nCells,3,4,1+len(self.para['CI_arr'])))*np.NaN,          ### (mean,std,CI_low,CI_top)
                        'p_x':np.zeros((nCells,3,2,self.para['nbin'])),
                        'posterior_mass':np.zeros((nCells,3))*np.NaN,
                        'nModes':np.zeros(nCells).astype('int')}
    
    results['firingstats'] = {'rate':np.zeros(nCells)*np.NaN,
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
    


  def plt_model_selection(self,fmap_bs,parsNoise,trials_fmap):
    
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
    
    fig = plt.figure(figsize=(4,3),dpi=150)
    ax = plt.axes([0.15,0.2,0.8,0.75])
    
    ax.bar(self.para['bin_array'],fr_mu,color='b',alpha=0.2,width=1,label='$\\bar{\\nu}$')
    ax.errorbar(self.para['bin_array'],fr_mu,fr_CI,ecolor='r',elinewidth=0.5,linestyle='',fmt='',label='$95\\%$ confidence')
    
    ax.set_xticks(np.linspace(0,100,6))
    ax.set_xticklabels(np.linspace(0,100,6).astype('int'))
    ax.set_ylim([-0.1,np.nanmax(fr_mu)*2])
    ax.set_ylabel('Ca$^{2+}$-event rate $\\nu$',fontsize=14)
    ax.set_xlabel('Position on track',fontsize=14)
    ax.legend(loc='upper left')
    plt.show(block=False)
    
    fig = plt.figure(figsize=(10,5),dpi=150)
    
    ax1 = plt.axes([0.1,0.55,0.85,0.42])
    #ax1 = plt.subplot(211)
    ax1.bar(self.para['bin_array'],fr_mu,color='b',alpha=0.2,width=1,label='$\\bar{\\nu}$')
    ax1.errorbar(self.para['bin_array'],fr_mu,fr_CI,ecolor='r',linestyle='',fmt='',label='$95\\%$ confidence')
    Y = trials_fmap/self.dataBH['trials']['dwelltime']
    mask = ~np.isnan(Y)
    Y = [y[m] for y, m in zip(Y.T, mask.T)]
    
    #flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5)
    #h_bp = ax1.boxplot(Y,flierprops=flierprops)#,positions=self.para['bin_array'])
    ax1.set_xticks(np.linspace(0,100,6))
    ax1.set_xticklabels(np.linspace(0,100,6).astype('int'))
    ax1.set_ylim([-0.1,np.nanmax(fr_mu)*2])
    ax1.set_ylabel('Ca$^{2+}$-event rate $\\nu$',fontsize=14)
    ax1.set_xlabel('Position on track',fontsize=14)
    ax1.legend(title='# trials = %d'%self.dataBH['trials']['ct'],fontsize=12)#[h_bp['boxes'][0]],['trial data'],
    
    ax2 = plt.axes([0.6,0.27,0.35,0.13])
    #ax2 = plt.subplot(426)
    ax2.plot(np.linspace(0,5,101),parsNoise*np.linspace(0,5,101),'k--',label='lq-fit')
    ax2.plot(fr_mu,fr_std,'kx',label='$\\sigma(\\nu)$')
    ax2.set_xlim([0,np.nanmax(fr_mu)*1.2])
    #ax2.set_xlabel('firing rate $\\nu$')
    ax2.set_xticks([])
    ax2.set_ylim([0,np.nanmax(fr_mu)*1.2])
    ax2.set_ylabel('$\\sigma_{\\nu}$',fontsize=14)
    
    ax3 = plt.axes([0.1,0.1,0.4,0.3])
    #ax3 = plt.subplot(223)
    
    i = random.randint(0,self.para['nbin']-1)
    ax1.bar(self.para['bin_array'][i],fr_mu[i],color='b')
    
    x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,101)
    act_hist = np.histogram(fmap_bs[i,:],x_arr,normed=True)
    ax3.bar(act_hist[1][:-1],act_hist[0],width=x_arr[1]-x_arr[0],color='b',alpha=0.2,label='data spread')
    
    alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
    mu, shape = lognorm_paras(fr_mu[i],fr_std[i])
    
    ax3.plot(x_arr,sstats.gamma.pdf(x_arr,alpha,0,1/beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')
    #ax3.plot(x_arr,sstats.lognorm.pdf(x_arr,s=shape,loc=0,scale=np.exp(mu)),'b',label='fit: $lognorm(\\alpha,\\beta)$')
    #ax3.plot(x_arr,sstats.truncnorm.pdf(x_arr,(0-fr_mu[i])/fr_std[i],np.inf,loc=fr_mu[i],scale=fr_std[i]),'g',label='fit: $gauss(\\mu,\\sigma)$')
    
    ax3.set_xlabel('$\\nu$',fontsize=14)
    ax3.set_ylabel('$p_{bs}(\\nu)$',fontsize=14)
    ax2.legend(fontsize=12)
    ax2.set_title("estimating noise",fontsize=12)
    
    ax3.legend(fontsize=12)
    ax3.set_title('bootstrapped data at bin %d'%i)
    
    ax4 = plt.axes([0.6,0.1,0.35,0.13])
    #ax4 = plt.subplot(428)
    
    D_KL_gamma = np.zeros(self.para['nbin'])
    D_KL_gauss = np.zeros(self.para['nbin'])
    D_KL_lognorm = np.zeros(self.para['nbin'])
    
    for i in range(self.para['nbin']):
      x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,101)
      act_hist = np.histogram(fmap_bs[i,:],x_arr,normed=True)
      alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
      mu, shape = lognorm_paras(fr_mu[i],fr_std[i])
      
      hist_cdf = np.cumsum(act_hist[0])
      hist_cdf /= hist_cdf[-1]
      offset = (x_arr[1]-x_arr[0])/2
      D_KL_gamma[i] = abs(sstats.gamma.cdf(act_hist[1][:-1]+offset,alpha,0,1/beta) - hist_cdf).max()
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
    
    ax4.plot(fr_mu,D_KL_gamma,'rx')
    #ax4.plot(fr_mu,D_KL_gauss,'gx')
    #ax4.plot(fr_mu,D_KL_lognorm,'bx')
    ax4.set_xlim([0,fr_mu.max()*1.2])
    ax4.set_xlabel('$\\nu$',fontsize=14)
    ax4.set_ylabel('$D_{KS}$',fontsize=14)
    #ax4.set_title('comparing data vs model',fontsize=14)
    
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
      plt.text(theta+3*std+2,A_0+A/2,'A',fontsize=14)
      
      plt.annotate("", xy=(theta, A_0*0.9), xytext=(theta+2*std, A_0*0.9),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
      plt.text(theta+2,A_0*0.6,'$2\\sigma$',fontsize=14)
      
      plt.annotate("", xy=(90, 0), xytext=(90, A_0),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
      plt.text(92,A_0/2,'$A_0$',fontsize=14)
      
      plt.annotate("", xy=(theta, 0), xytext=(theta,A_0+A),
            arrowprops=dict(arrowstyle="-"))
      plt.text(theta,A_0+A*1.1,'$\\theta$',fontsize=14)
    
    plt.xlabel('Position on track $x$',fontsize=14)
    plt.ylabel('Ca$^{2+}$ event rate (model)',fontsize=14)
    plt.yticks([])
    #plt.legend(loc='upper left',fontsize=14)
    plt.ylim([0,2.5])
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
    
    ### steps for lookup-table (if used)
    #self.lookup_steps = 100000
    #self.set_beta_prior(5,4)
  
  def set_logl_func(self):
    def get_logl(p):
      if len(p.shape)==1:
        p = p[np.newaxis,:]
      p = p[...,np.newaxis]
      
      mean_model = np.ones((p.shape[0],self.Nx))*p[:,0]
      if p.shape[1] > 1:
        for j in [-1,0,1]:   ## loop, to have periodic boundary conditions
          mean_model += p[:,1]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,3]+self.x_max*j)**2/(2*p[:,2]**2))
      
      SD_model = self.parsNoise*mean_model
      
      alpha = (mean_model/SD_model)**2
      beta = mean_model/SD_model**2
      
      return (alpha*np.log(beta) - np.log(sp.special.gamma(alpha)) + (alpha-1)*np.log(self.data) - beta*self.data ).sum(1)
      
    return get_logl
  
  ### want beta-prior for std - costly, though, so define lookup-table
  def set_beta_prior(self,a,b):
    self.lookup_beta = sp.stats.beta.ppf(np.linspace(0,1,self.lookup_steps),a,b)
    #return sp.special.gamma(a+b)/(sp.special.gamma(a)*sp.special.gamma(b))*x**(a-1)*(1-x)**(b-1)
  
  def transform_p(self,p):
    
    if p.shape[-1]>1:
      p_out = p*self.prior_stretch + self.prior_offset
      #p_out[...,2] = self.prior_stretch[2]*self.lookup_beta[(p[...,2]*self.lookup_steps).astype('int')]
    else:
      p_out = p*self.prior_stretch[0] + self.prior_offset[0]
      
    return p_out
  
  def set_priors(self):
    self.prior_stretch = np.array([5,10,5,100])
    self.prior_offset = np.array([0,0,0.5,0])
  
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
          TC += p[:,1]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,3]+self.x_max*j)**2/(2*p[:,2]**2))
      
      return np.squeeze(TC)
    
    return TC_func
 
####------------------------ end of HBM definition ----------------------------- ####






def load_activity(pathSession,redetect=True):
  ## load activity data from CaImAn results
  
  if redetect:
    pathAct = pathcat([pathSession,'results_postSilent.mat'])
  else:
    pathAct = pathcat([pathSession,'results_OnACID.mat'])
  
  ld = sio.loadmat(pathAct,squeeze_me=True)
  S = ld['S']
  if S.shape[0] > 8000:
    S = S.transpose()
  
  if redetect:
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
  return p_tot[p_joint>0].sum()


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


def get_firingmap(S,binpos,dwelltime,bool_norm):
  
  ### calculates the firing map
  
  #if not bool_norm:
    #dwelltime = np.histogram(binpos,parameter['bin_array_centers'])[0]/self.para['f']
  #else:
    #dwelltime = dwelltime_in
  
  nbin = len(dwelltime);
  
  spike_times = np.where(S)
  spikes = S[spike_times]
  binpos = binpos[spike_times].astype('int')
  
  firingmap = np.zeros(nbin)
  for i in range(len(binpos)):
    firingmap[binpos[i]] = firingmap[binpos[i]]+spikes[i]
  
  if bool_norm:
    firingmap = firingmap/dwelltime
    firingmap[~dwelltime] = np.NaN
  
  return firingmap



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
