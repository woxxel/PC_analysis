from multiprocessing import get_context
#set_start_method("spawn")

import os, time, cv2, warnings, h5py, itertools
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as sstats
from tqdm import *
from time import sleep
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp


from utils import get_nPaths, pathcat, extend_dict, clean_dict, pickleData, fdr_control, periodic_distr_distance, fit_plane, z_from_point_normal_plane, get_shift_and_flow, com, calculate_img_correlation
from utils_data import set_para

warnings.filterwarnings("ignore")


class cluster:
  
  def __init__(self,basePath,mouse,nSes,dataSet='OnACID',session_order=None,s_corr_min=0.3):
    
    t_start = time.time()
    
    self.mouse = mouse
    self.pathMouse = pathcat([basePath,mouse])
    self.pathMatching = pathcat([self.pathMouse,'matching/Sheintuch_registration_results_%s.pkl'%dataSet])
    self.CNMF_name = 'results_%s.mat'%dataSet
    
    if not (nSes is None):
      self.nSes = nSes
    else:
      self.nSes, tmp = get_nPaths(self.pathMouse,'Session')
    
    if session_order is None:
      self.session_order = range(1,self.nSes+1)
    else:
      self.session_order = list(chain.from_iterable(session_order)) if (type(session_order[0]) is range) else list(chain(session_order))
    
    self.para = set_para(basePath,mouse,1)
    
    self.svCluster = pathcat([self.pathMouse,'cluster_%s.pkl'%dataSet])
    
    self.meta = {'mouse':self.mouse,
                 'pathMouse':self.pathMouse,
                 'CNMFname':self.CNMF_name,
                 'nC':np.NaN,
                 'nSes':self.nSes,
                 'f': 15,   ## measurement frequency
                 'dims':(512,512),#np.zeros(2)*np.NaN,
                 'svSessions':pathcat([self.pathMouse,'clusterSessions_%s.pkl'%dataSet]),
                 'svIDs':pathcat([self.pathMouse,'clusterIDs_%s.pkl'%dataSet]),
                 'svStats':pathcat([self.pathMouse,'clusterStats_%s.pkl'%dataSet]),
                 'svPCs':pathcat([self.pathMouse,'clusterPCs_%s.pkl'%dataSet]),
                 'svCompare':pathcat([self.pathMouse,'clusterCompare_%s.pkl'%dataSet]),
                 
                 'field_count_max':            5,
                 
                 'session_min_correlation':    s_corr_min,
                 'session_max_shift':          50,
                 'border_margin':              10,
                 'min_cluster_count':          2,
                 
                 'SNR_thr':                    2,
                 'rval_thr':                   0,
                 'CNN_thr':                    0.3,
                 'pm_thr':                     0.05,
                 
                 'fr_thr':                     0,
                 
                 'MI_alpha':                   1,
                 'MI_thr':                     0.1,
                 
                 'Bayes_thr':                  0,
                 'reliability_thr':            0.1,
                 'ampl_thr':                   0,
                 'Arate_thr':                  0,
                 'pmass_thr':                  0.5,
                 'CI_thr':                     self.para['nbin']
                 }
    
    
  def run_complete(self,sessions=None,n_processes=0,reprocess=False):
        
    if (not os.path.exists(self.svCluster)) | reprocess:
      self.process_sessions(sessions=sessions,n_processes=n_processes,reprocess=reprocess)
      self.get_IDs()
      self.get_stats(n_processes=n_processes)
      
      self.cluster_classification()
      
      self.get_PC_fields()
      
      self.update_status()
      self.compareSessions(n_processes=n_processes,reprocess=reprocess)
      self.save()
    else:
      print('already present')
      #self.load('cluster.pkl')
      #self.meta['nC'] = self.PCs['status'].shape[0]
      #self.session_classification(sessions)
  
  def process_sessions(self,sessions=None,n_processes=0,reprocess=False):
    if reprocess | (not os.path.exists(self.meta['svSessions'])):
      
      self.sessions = {'shift':np.zeros((self.nSes,2)),
                       'borders':np.zeros((2,2))*np.NaN,
                       'corr':np.zeros((self.nSes,2))*np.NaN,
                       'flow_field':np.zeros(((self.nSes,)+self.meta['dims']+(2,))),
                       #'rotation_anchor':np.zeros((self.nSes,3))*np.NaN,     ## point on plane
                       #'rotation_normal':np.zeros((self.nSes,3))*np.NaN,     ## normal describing plane}
                       'bool':np.zeros(self.meta['nSes']).astype('bool')}
      
      self.get_reference_frame()
      self.session_classification(sessions=sessions)
      #self.save([False,True,False,False,False])
    
  def get_reference_frame(self):
    self.progress = tqdm(enumerate(self.session_order),total=self.nSes,leave=False)
    for (s,s0) in self.progress:
      self.progress.set_description('Loading data from Session %d'%s0)
      pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%s0])
      pathLoad = pathcat([pathSession,self.CNMF_name])
      if os.path.exists(pathLoad):
        ld = sio.loadmat(pathLoad,variable_names=['A','Cn'])
        A2 = ld['A']
        try:
          Cn2 = ld['Cn']
        except:
          Cn2 = np.array(A2.sum(1).reshape(self.meta['dims']))
        
        if s>0:
          self.sessions['shift'][s,:], self.sessions['flow_field'][s,...], _, self.sessions['corr'][s,0] = get_shift_and_flow(Aref,A2,self.meta['dims'],projection=1,plot_bool=False)
          
          self.sessions['corr'][s,1],(y_shift,x_shift) = calculate_img_correlation(Cnref,Cn2,plot_bool=False)
          #self.sessions['corr'][s,1],(y_shift,x_shift) = calculate_img_correlation(A2ref.sum(1),A2.sum(1),plot_bool=False)
          if self.sessions['corr'][s,1]>=self.meta['session_min_correlation']:
            A2ref = A2.copy()
            Cnref = Cn2.copy()
        else:
          Aref = A2.copy()    ## keep template of first session for calculating session alignment statistics
          A2ref = A2.copy()
          
          Cnref = Cn2.copy()
  
  def session_classification(self,sessions=None,max_shift=None,min_corr=None):
    
    self.sessions['bool'][:] = False
    if sessions is None:
      self.sStart = 0
      self.sEnd = self.meta['nSes']
    else:
      self.sStart = sessions[0]-1
      self.sEnd = sessions[-1]+1
    
    print(self.sStart)
    print(self.sEnd)
    if not (max_shift is None):
      self.meta['session_max_shift'] = max_shift
    if not (min_corr is None):
      self.meta['session_min_corr'] = min_corr
    
    self.sessions['bool'][self.sStart:self.sEnd] = True
    self.sessions['bool'][np.array([np.sqrt(x**2+y**2) for (x,y) in self.sessions['shift']])>self.meta['session_max_shift']] = False ## huge shift
    self.sessions['bool'][self.sessions['corr'][:,1]<self.meta['session_min_correlation']] = False ## huge shift
    self.sessions['bool'][np.isnan(self.sessions['corr'][:,1])] = False
    if self.sStart == 0:
      self.sessions['bool'][0] = True
    #if self.meta['mouse'] == '762':
      #try:
        #self.sessions['bool'][40] = False  ## very bad imaging quality (check video)
        #self.sessions['bool'][66] = False     ## very bad imaging quality (check video)
      #except:
        #1
  
  def get_IDs(self):
    
    self.IDs = {'clusterID':np.zeros((0,2)).astype('uint16'), 
                'neuronID':np.zeros((0,self.nSes,2))}
    
    ld_dat = pickleData([],self.pathMatching,'load')
    try:
      assignments = ld_dat['assignments']
    except:
      assignments = ld_dat['assignment']
    self.meta['nC'] = nC = assignments.shape[0]
    extend_dict(self.IDs,nC)
    self.IDs['clusterID'][range(nC),:] = np.vstack([np.ones(nC),range(nC)]).T;
    print(assignments.shape)
    print(self.session_order)
    for (s,_) in tqdm(enumerate(self.session_order),total=self.meta['nSes'],leave=False):
      if s >= assignments.shape[1]:
        break
      ### assign neuron IDs
      idx_c = np.where(~np.isnan(assignments[:,s]))[0]
      n_arr = assignments[idx_c,s].astype('int')
      
      self.IDs['neuronID'][idx_c,s,:] = np.vstack([np.ones(len(n_arr)),n_arr]).T
    
    #self.save([True,False,False,False,False])
  
  #def find_rotation_from_flow(self,flow,plot_bool=False):
    #dims = self.meta['dims']
    #x = np.hstack([np.ones((dims[0],1)),np.arange(dims[0]).reshape(dims[0],1)]) 
    #x_grid, y_grid, z_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32),
                                        #np.arange(0., dims[1]).astype(np.float32),
                                        #np.arange(0,1).astype(np.float32))
    
    #W = sstats.norm.pdf(range(dims[0]),dims[0]/2,dims[0]/2)
    #W /= W.sum()
    #W = np.sqrt(np.diag(W))
    #x_w = np.dot(W,x)
    #flow_w = np.dot(flow[:,:,1],W)
    #x0,res,rank,s = np.linalg.lstsq(x_w,flow_w,rcond=None)
    
    #d = -x0[0,:]/x0[1,:]
    #r = sstats.linregress(range(dims[0]),d)
    
    #tilt_ax = r.intercept+r.slope*range(512)
    
    #dist_mat = np.abs((r.slope*x_grid[:,:,0]-y_grid[:,:,0]+r.intercept)/np.sqrt(r.slope**2+1**2))
    #slope_normal = np.array([-r.slope,1])
    #slope_normal /= np.linalg.norm(slope_normal)
    #f_perp = np.dot(flow[:,:,:2],slope_normal)
    #f_perp_vec = f_perp[...,np.newaxis].dot(slope_normal[np.newaxis,:]) #multiplay by normal
    
    ##print(f_perp_vec.shape)
    #idxes = 15
    #plt.figure()
    #plt.subplot(121)
    #plt.quiver(x_grid[::idxes,::idxes], y_grid[::idxes,::idxes], flow[::idxes,::idxes,0], flow[::idxes,::idxes,1], angles='xy', scale_units='xy', scale=0.25, headwidth=4,headlength=4, width=0.002, units='width')
    #plt.plot(tilt_ax,'b-')
    #plt.xlim([0,dims[0]])
    #plt.ylim([0,dims[0]])
    
    #abs_flow = np.linalg.norm(flow,axis=2)
    #margin = 50
    #abs_flow = sp.ndimage.filters.gaussian_filter(abs_flow[margin:-margin,margin:-margin],20)
    
    #print(abs_flow)
    #print(abs_flow.shape)
    #min_pos = np.add(np.unravel_index(np.argmin(abs_flow),abs_flow.shape),margin)
    
    #plt.subplot(122)
    #plt.imshow(abs_flow,extent=[margin,512-margin,margin,512-margin])
    #plt.colorbar()
    #plt.plot(min_pos[1],min_pos[0],'rx')
    #plt.xlim([0,dims[0]])
    #plt.ylim([0,dims[0]])
    #plt.show(block=True)
    
    ### need two cases to capture both, flows away from and towards axis (rotation to a more even vs rotation to a more skewed plane) - not entirely precise, only holds for relatively small angles theta
    #h_dat = np.sign(f_perp)*np.sin(np.arccos((dist_mat - np.abs(f_perp))/dist_mat))*dist_mat
    ##h_dat = np.zeros(x_grid[...,0].shape)
    
    #data = np.stack([x_grid[...,0],y_grid[...,0],h_dat],2)
    #data = data.reshape(dims[0]*dims[1],3)
    #mask = ~np.isnan(data[:,2])
    
    #### fit plane to data using covariance of data points. choose between calculating centroid to rotate around or providing from center of tilt axis
    ##(p,n),tmp = fit_plane(data[mask,:],anchor=[dims[0]/2-1,tilt_ax[int(dims[0]/2-1)],0])
    #(p,n),tmp = fit_plane(data[mask,:])
    #angles = np.arccos(n)/(2*np.pi)*360
    ##print('angles:',angles)
    
    
    #if plot_bool:
      #h_plane = z_from_point_normal_plane(x_grid[...,0],y_grid[...,0],p,n)
      
      #plt.figure(figsize=(12,12))
      #ax = plt.subplot(221)
      #im = ax.imshow(h_dat,origin='lower',cmap='jet',clim=[-30,30])
      #plt.colorbar(im)
      #ax.plot(d,'b:')
      #ax.plot(tilt_ax,'b-')
      #ax.set_xlim([0,dims[0]])
      #ax.set_ylim([0,dims[1]])
      
      #ax2 = plt.subplot(222)
      #im2 = ax2.imshow(h_plane,origin='lower',cmap='jet',clim=[-30,30])
      #ax2.plot(d,'b:')
      #ax2.plot(tilt_ax,'b-')
      #ax2.plot(p[0],p[1],'rx')
      #ax2.set_xlim([0,dims[0]])
      #ax2.set_ylim([0,dims[1]])
      #plt.colorbar(im2)
      
      #plt.subplot(223)
      #plt.imshow(h_dat-h_plane,origin='lower',cmap='jet',clim=[-10,10])
      #plt.colorbar()
      
      #idxes = 1
      #ax = plt.subplot(224,projection='3d')
      #ax.plot_surface(x_grid[::50,::50,0],y_grid[::50,::50,0],h_plane[::50,::50],color='k')
      #ax.plot_surface(x_grid[::idxes,::idxes,0],y_grid[::idxes,::idxes,0],h_dat[::idxes,::idxes],color='r',alpha=0.5)
      #ax.set_xlabel('x')
      #ax.set_ylabel('y')
      #ax.set_zlabel('z')
      
      #plt.show(block=True)
    
    #return (p,n)
  
  
  def get_stats(self,n_processes=0,complete=False):
    
    t_start = time.time()
    
    self.stats = {'com':np.zeros((0,self.nSes,2))*np.NaN,
                  'match_score':np.zeros((0,self.nSes,2)),
                  'cluster_bool':np.zeros(0,'bool'),
                                    
                  'firingrate':np.zeros((0,self.nSes))*np.NaN,
                  'firingmap':np.zeros((0,self.nSes,self.para['nbin'])),
                  
                  'SNR':np.zeros((0,self.nSes)),
                  'r_values':np.zeros((0,self.nSes)),
                  'CNN':np.zeros((0,self.nSes)),
                  
                  'MI_value':np.zeros((0,self.nSes)),
                  'MI_p_value':np.zeros((0,self.nSes)),
                  'MI_z_score':np.zeros((0,self.nSes)),
                  
                  'Isec_value':np.zeros((0,self.nSes)),
                  
                  'Bayes_factor':np.zeros((0,self.nSes))
                  }
    
    extend_dict(self.stats,self.meta['nC'])
    if complete:
      res = []
      if n_processes>1:
        batchSz = n_processes
        nBatch = self.meta['nSes']//batchSz + (np.mod(self.meta['nSes'],batchSz)>0)
        
        for i in tqdm(range(nBatch)):
          S = {}
          for s in range(i*batchSz,min(self.meta['nSes'],(i+1)*batchSz)):
            s0 = self.session_order[s]
            pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%(s0)]);
            pathLoad = pathcat([pathSession,self.meta['CNMFname']])
            for file in os.listdir(pathSession):
              if file.endswith("aligned.mat"):
                pathBH = os.path.join(pathSession, file)
                break
            
            ### obtain firing rate
            f = h5py.File(pathBH,'r')
            longrun = np.squeeze(f.get('alignedData/resampled/%s'%'longrunperiod').value).astype('bool')
            f.close()
            
            results_CNMF = sio.loadmat(pathLoad,variable_names=['S'],squeeze_me=True)
            if results_CNMF['S'].shape[0] > 8000:
              results_CNMF['S'] = results_CNMF['S'].transpose()
            
            S[s0] = results_CNMF['S'][:,longrun]
          
          pool = get_context("spawn").Pool(n_processes)
          #res = [pool.apply(get_firingrate,args=(s0,self.meta)) for (s,s0) in enumerate(self.session_order)]
          res.extend(pool.starmap(get_firingrate,zip(S.items(),itertools.repeat(self.meta))))#self.session_order[i*batchSz:(i+1)*batchSz],
          pool.close()
      else:
        for s in tqdm(range(self.meta['nSes'])):
          res.append(get_firingrate(s0,self.meta))
      firingrate = {}
      for (s,x) in res:#[r.get() for r in res]:
        firingrate[s] = x
    
    ## get matching results
    ld_dat = pickleData([],self.pathMatching,'load')
    p_matched = ld_dat['p_matched']
    p_all = ld_dat['p_same']
    cm = ld_dat['cm']
    
    for (s,s0) in tqdm(enumerate(self.session_order),total=self.meta['nSes'],leave=False):
      
      idx_c = np.where(~np.isnan(self.IDs['neuronID'][:,s,1]))[0]
      n_arr = self.IDs['neuronID'][idx_c,s,1].astype('int')
      nCells = len(n_arr)
      
      ### get neuron centroid positions
      self.stats['com'][idx_c,s,:] = cm[idx_c,s,:]
      
      if complete:
        self.stats['firingrate'][idx_c,s] = firingrate[s0][n_arr]
      else:
        pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%s0]);
        pathFiringstats = pathcat([pathSession,'PC_fields_firingstats.mat']);
        firingstats_tmp = sio.loadmat(pathFiringstats,squeeze_me=True)
        
        self.stats['firingrate'][idx_c,s] = firingstats_tmp['rate'][n_arr]
        self.stats['firingmap'][idx_c,s,:] = firingstats_tmp['map'][n_arr,:]
        #self.stats['firingmap'][idx_c,s,1,:] = firingstats_tmp['std'][n_arr,:]
        #self.stats['firingmap'][idx_c,s,2:,:] = firingstats_tmp['CI'][n_arr,:,:]
        
        pathStatus = pathcat([pathSession,'PC_fields_status.mat']);
        status = sio.loadmat(pathStatus,squeeze_me=True);
        
        self.stats['MI_value'][idx_c,s] = status['MI_value'][n_arr]
        self.stats['MI_p_value'][idx_c,s] = status['MI_p_value'][n_arr]
        self.stats['MI_z_score'][idx_c,s] = status['MI_z_score'][n_arr]
        
        self.stats['Isec_value'][idx_c,s] = status['Isec_value'][n_arr]
        
        #self.stats['Bayes_factor'][idx_c,s] = status['Bayes_factor'][n_arr,0,0]-status['Bayes_factor'][n_arr,0,1]
        #self.stats['Bayes']['Z'][idx_c,s,:,:] = status['Z'][n_arr,:,:]
        
        idx_active = self.stats['firingrate'][:,s]>0
        
        try:
          pathLoad = pathcat([pathSession,self.meta['CNMFname']])
          results_CNMF = sio.loadmat(pathLoad,variable_names=['SNR','r_values','CNN'],squeeze_me=True)
          self.stats['SNR'][idx_c,s] = results_CNMF['SNR'][n_arr]
          self.stats['r_values'][idx_c,s] = results_CNMF['r_values'][n_arr]
          self.stats['CNN'][idx_c,s] = results_CNMF['CNN'][n_arr]
        except:
          1
      
      self.stats['match_score'][idx_c,s,0] = 1
      if (s>0) & self.sessions['bool'][s]:     ## store matched score and best score (with matched removed)
        idx_c = idx_c[idx_c<p_all[s].shape[0]]    # remove entries of first-occurence neurons (no matching possible)
        self.stats['match_score'][idx_c,s,0] = p_matched[idx_c,s]
        scores_now = p_all[s].toarray()
        self.stats['match_score'][idx_c,s,1] = [max(scores_now[c,np.where(scores_now[c,:]!=self.stats['match_score'][c,s,0])[0]]) for c in idx_c]
    self.save([False,False,True,False,False])
    print('stats obtained - time taken: %5.3g'%(time.time()-t_start))
  
  
  def cluster_classification(self,min_cluster_count=None,border_margin=None):
    
    if not (min_cluster_count is None):
      self.meta['min_cluster_count'] = min_cluster_count
    if not (border_margin is None):
      self.meta['border_margin'] = border_margin
    
    self.stats['cluster_bool'] = np.ones(self.meta['nC']).astype('bool')
    self.stats['cluster_bool'][(~np.isnan(self.IDs['neuronID'][...,1])).sum(1)<self.meta['min_cluster_count']] = False
    
    thr_high = self.meta['dims'] + self.sessions['shift'][self.sessions['bool'],:].min(0)
    thr_low = self.sessions['shift'][self.sessions['bool'],:].max(0)
    
    self.sessions['borders'] = np.vstack([thr_low,thr_high])
    
    for i in range(2):
      idx_remove_low = self.stats['com'][:,self.sessions['bool'],i] < (thr_low[i]+self.meta['border_margin'])
      self.stats['cluster_bool'][np.any(idx_remove_low,1)] = False
      
      idx_remove_high = self.stats['com'][:,self.sessions['bool'],i] > (thr_high[i]-self.meta['border_margin'])
      self.stats['cluster_bool'][np.any(idx_remove_high,1)] = False
  
  
  def get_PC_fields(self):
    
    self.fields = {'nModes':np.zeros((0,self.nSes)).astype('uint8'),
                   'status':np.zeros((0,self.nSes,self.meta['field_count_max'])).astype('uint8'),
                   'Bayes_factor':np.zeros((0,self.nSes,self.meta['field_count_max'])),
                   'reliability':np.zeros((0,self.nSes,self.meta['field_count_max'])),
                   'posterior_mass':np.zeros((0,self.nSes,self.meta['field_count_max'])),
                   'baseline':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                   'amplitude':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                   'width':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                   'location':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                   'p_x':np.zeros((0,self.nSes,self.meta['field_count_max'],self.para['nbin']))}
    extend_dict(self.fields,self.meta['nC'])
    
    t_start = time.time()
    for (s,_) in tqdm(enumerate(self.session_order),total=self.meta['nSes'],leave=False):
      
      pathSession = pathcat([self.pathMouse,'Session%02d'%(s+1)]);
      pathFields = pathcat([pathSession,'PC_fields_para.mat']);
      
      if os.path.exists(pathFields):
        
        idx_c = np.where(~np.isnan(self.IDs['neuronID'][:,s,1]))[0]
        n_arr = self.IDs['neuronID'][idx_c,s,1].astype('int')
        nCells = len(n_arr)
        
        ### hand over all other values
        fields = sio.loadmat(pathFields,squeeze_me=True);
        self.fields['nModes'][idx_c,s] = np.minimum(2,(np.isfinite(fields['parameter'][n_arr,:,3,0]).sum(1)).astype('int'))
        
        for (c,n) in zip(idx_c,n_arr):
          
          if self.fields['nModes'][c,s] > 0:   ## cell is PC
            
            ### hand over field parameters
            self.fields['location'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],3,(0,1,4)].transpose(1,0)
            self.fields['width'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],2,(0,1,4)].transpose(1,0)
            self.fields['amplitude'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],1,(0,1,4)].transpose(1,0)
            self.fields['baseline'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],0,(0,1,4)].transpose(1,0)
            
            self.fields['p_x'][c,s,:,:] = fields['p_x'][n,:self.meta['field_count_max'],:]
            self.fields['posterior_mass'][c,s,:] = fields['posterior_mass'][n,:self.meta['field_count_max']]
            
            self.fields['Bayes_factor'][c,s,...] = fields['Bayes_factor'][n,:self.meta['field_count_max'],0]
            self.fields['reliability'][c,s,:] = fields['reliability'][n,:self.meta['field_count_max']]
            
      else:
        print('Data for Session %d does not exist'%s)
    
    self.save([False,False,False,True,False])
    t_end = time.time()
    print('Fields obtained and saved, time spend: %6.4f'%(t_end-t_start))
  
  
  ### calculate shifts within clusters
  def compareSessions(self,reprocess=False,n_processes=0):
    
    print('method chooses first field automatically - though I want most significant (mostly same) - how to?')
    t_start = time.time()
    nSes = self.meta['nSes']
    
    pathComp = pathcat([self.para['pathMouse'],'compareSessions.pkl'])
    if reprocess | (not os.path.exists(pathComp)):
      self.compare = {'pointer':sp.sparse.lil_matrix((self.meta['nC'],nSes**2*self.meta['field_count_max']**2)),
                      'shifts':[],
                      'shifts_distr':[],
                      'inter_active':[],
                      'inter_coding':[]}
      
      t_start=time.time()
      
      if n_processes>1:
        pool = get_context("spawn").Pool(n_processes)
        loc = self.fields['location'][...,0]
        #loc[~self.status_fields] = np.NaN
        res = pool.starmap(get_field_shifts,zip(self.status,self.fields['p_x'],loc))#self.meta['nC']))
        i=0
        for c,r in enumerate(res):
          for key in r['shifts_distr']:
            i += 1
            self.compare['pointer'][c,key] = i
            #print(r['shifts'][key])
            self.compare['shifts'].append(r['shifts'][key])
            self.compare['shifts_distr'].append(r['shifts_distr'][key])
            self.compare['inter_active'].append(r['inter_active'][key])
            self.compare['inter_coding'].append(r['inter_coding'][key])
        
        self.compare['shifts'] = np.array(self.compare['shifts'])
        
        self.compare['inter_active'] = np.array(self.compare['inter_active'])
        self.compare['inter_coding'] = np.array(self.compare['inter_coding'])
        
        self.compare['shifts_distr'] = sp.sparse.csr_matrix(self.compare['shifts_distr'])
        self.compare['pointer'] = self.compare['pointer'].tocoo()
      else:
        print('please use parallel processing for this')
      
    else:
      self.compare = pickleData([],self.meta['svCompare'],'load')
    t_end = time.time()
    print('Place field shifts calculated - time %5.3f'%(t_end-t_start))
    
  
  
  def update_status(self,complete=True,
                    SNR_thr=None,rval_thr=None,CNN_thr=None,pm_thr=None,
                    fr_thr=None,Bayes_thr=None,reliability_thr=None,alpha=None,MI_thr=None,
                    ampl_thr=None,Arate_thr=None,pmass_thr=None,CI_thr=None):
    
    print('further, implement method to calculate inter-coding intervals, etc, after updating statusses')
    
    self.thr = {'SNR':          self.meta['SNR_thr'] if SNR_thr is None else SNR_thr,
                'r_values':     self.meta['rval_thr'] if rval_thr is None else rval_thr,
                'CNN':          self.meta['CNN_thr'] if CNN_thr is None else CNN_thr,
                'p_matched':    self.meta['pm_thr'] if pm_thr is None else pm_thr,
                
                'firingrate':   self.meta['fr_thr'] if fr_thr is None else fr_thr,
                
                'alpha':        self.meta['MI_alpha'] if alpha is None else alpha,
                'MI':           self.meta['MI_thr'] if MI_thr is None else MI_thr,
                'CNN':          self.meta['CNN_thr'] if CNN_thr is None else CNN_thr,
                
                'Bayes':        self.meta['Bayes_thr'] if Bayes_thr is None else Bayes_thr,
                'reliability':  self.meta['reliability_thr'] if reliability_thr is None else reliability_thr,
                
                'ampl':         self.meta['ampl_thr'] if ampl_thr is None else ampl_thr,
                'A_rate':       self.meta['Arate_thr'] if Arate_thr is None else Arate_thr,
                'p_mass':       self.meta['pmass_thr'] if pmass_thr is None else pmass_thr,
                'CI':           self.meta['CI_thr'] if CI_thr is None else CI_thr
                }
    
    t_start = time.time()
    
    ### reset all statuses
    self.status = np.zeros((self.meta['nC'],self.meta['nSes'],6),'bool')
    
    ### appearance in cluster: defined by (SNR,r_values,CNN), p_matched
    if np.any(~np.isnan(self.stats['SNR'])):
      self.status[...,0] = (self.stats['SNR']>self.thr['SNR']) & \
                            (self.stats['r_values']>self.thr['r_values']) & \
                            (self.stats['CNN']>self.thr['CNN']) & \
                            (self.stats['match_score'][...,0]>self.thr['p_matched'])
    else:
      self.status[...,0] = (self.stats['match_score'][...,0]>self.thr['p_matched'])
    
    if complete:
      ### activity: defined by: firingrate
      self.status[...,1] = (self.stats['firingrate']>self.thr['firingrate']) & self.status[...,0]
      
      self.fields['status'] = np.zeros((self.meta['nC'],self.meta['nSes'],self.meta['field_count_max']),'int')
      ### place field: amplitude, A_rate, p_mass, CI-width, width(?), 
      A_rate = self.fields['amplitude'][...,0]/self.fields['baseline'][...,0]
      CI_width = np.mod(self.fields['location'][...,2]-self.fields['location'][...,1],self.para['nbin'])
      
      idx_fields = (self.fields['amplitude'][...,0]>self.thr['ampl']) & \
                  (A_rate>self.thr['A_rate']) & \
                  (self.fields['posterior_mass']>self.thr['p_mass']) & \
                  (CI_width<self.thr['CI']) & \
                  (self.fields['Bayes_factor']>self.thr['Bayes']) & \
                  (self.fields['reliability']>self.thr['reliability'])
      
      self.status_fields = idx_fields
      ### place cell: defined by: Bayes factor, MI(val,p_val,z_score)
      self.stats['MI_p_value'][self.stats['MI_p_value']==0.001] = 10**(-10)    ## need this - can't get any lower than 0.001 with 1000 shuffles...
      idx_PC = np.ones((self.meta['nC'],self.meta['nSes']),'bool')
      for s in np.where(self.sessions['bool'])[0]:
        idx_PC[:,s] = fdr_control(self.stats['MI_p_value'][:,s],self.thr['alpha'])
      idx_PC = idx_PC & np.any(idx_fields,-1) & (self.stats['MI_value']>self.thr['MI'])
      
      self.status[...,2] = idx_PC & self.status[...,1]
      
      self.status[:,~self.sessions['bool'],:] = False
      self.status[~self.stats['cluster_bool'],:,:] = False
      
      #idx_reward = (self.fields['location'][...,0]<=self.para['zone_idx']['reward'][-1]) & \
                  #(self.fields['location'][...,0]>=self.para['zone_idx']['reward'][0])
      #self.fields['status'][idx_reward] = 4
      #self.fields['status'][~self.status[...,2],:] = False
      
      for c in tqdm(range(self.meta['nC'])):
        for s in range(self.meta['nSes']):
          if self.status[c,s,2]:
            
            for f in np.where(idx_fields[c,s,:])[0]:#range(self.fields['nModes'][c,s]):
              if idx_fields[c,s,f]:
                
                if self.para['zone_idx']['reward'][0] <= self.fields['location'][c,s,f,0] <= self.para['zone_idx']['reward'][-1]:
                  self.fields['status'][c,s,f] = 4;
                elif self.para['zone_idx']['gate'][0] <= self.fields['location'][c,s,f,0] <= self.para['zone_idx']['gate'][-1]:
                  self.fields['status'][c,s,f] = 3;
                else:
                  self.fields['status'][c,s,f] = 5;
                self.status[c,s,self.fields['status'][c,s,f]] = True;
            
            self.fields['nModes'][c,s] = np.count_nonzero(self.fields['status'][c,s,:])
        
        
    t_end = time.time()
    print('PC-characterization done. Time taken: %7.5f'%(t_end-t_start))
  
  
  def get_reliability(self):
  
    fmap = gauss_smooth(result['firingstats']['trial_map'],(0,5*self.para['nbin']/self.para['L_track']))
    field_bin = int(result['fields']['parameter'][t,3,0]*(self.para['nbin']/self.para['L_track']))
    print(field_bin)
    print(fmap[:,field_bin])
    print(fmap[:,field_bin]>self.para['rate_thr'])
    print(result['firingstats']['trial_field'][t,:])
    
    
  
  def save(self,svBool=np.ones(5).astype('bool')):
    
    if svBool[0]:
      pickleData(self.IDs,self.meta['svIDs'],'save')
    if svBool[1]:
      pickleData(self.sessions,self.meta['svSessions'],'save')
    if svBool[2]:
      pickleData(self.stats,self.meta['svStats'],'save')
    if svBool[3]:
      pickleData(self.fields,self.meta['svPCs'],'save')
    if svBool[4]:
      pickleData(self.compare,self.meta['svCompare'],'save')
    
  def load(self,ldBool=np.ones(5).astype('bool')):
    #self.allocate_cluster()
    if ldBool[0]:
      self.IDs = pickleData([],self.meta['svIDs'],'load')
      self.meta['nC'] = self.IDs['neuronID'].shape[0]
    if ldBool[1]:
      self.sessions = pickleData([],self.meta['svSessions'],'load')
    if ldBool[2]:
      self.stats = pickleData([],self.meta['svStats'],'load')
    if ldBool[3]:
      self.fields = pickleData([],self.meta['svPCs'],'load')
    if ldBool[4]:
      self.compare = pickleData([],self.meta['svCompare'],'load')


def get_field_shifts(status,p_x,loc):
  nSes = status.shape[0]
  nfields = p_x.shape[1]
  nbin = p_x.shape[-1]
  L_track=100
  out = {'shifts': {},
         'shifts_distr': {},
         'inter_active':{},
         'inter_coding':{}}
  
  for s1 in range(nSes):
    if np.any(status[s1,2:]):
      for s2 in range(s1+1,nSes):
        if np.any(status[s2,2:]):
          
          #f1_arr = np.where(~np.isnan(loc[s1,:]))[0]
          #f2_arr = np.where(~np.isnan(loc[s2,:]))[0]
          
          #f_norm = len(f1_arr)*len(f2_arr)
          #i=0
          #for f1 in f1_arr:
            #for f2 in f2_arr:
              #idx = np.ravel_multi_index((s1,s2,i),(nSes,nSes,nfields**2))
              #shifts, shifts_distr = periodic_distr_distance(p_x[s1,f1,:],p_x[s2,f2,:],nbin,L_track,mode='bootstrap')
              #shifts_distr /= f_norm
          
          d = np.abs(np.mod(loc[s1,:][:,np.newaxis] - loc[s2,:]+nbin/2,nbin)-nbin/2)
          d[np.isnan(d)] = nbin
          f1,f2 = sp.optimize.linear_sum_assignment(d)
          for f in zip(f1,f2):
            
            if d[f] < nbin:
              
              idx = np.ravel_multi_index((s1,s2,f[0]),(nSes,nSes,nfields**2))
              shifts, shifts_distr = periodic_distr_distance(p_x[s1,f[0],:],p_x[s2,f[1],:],nbin,L_track,mode='bootstrap')
              
              out['shifts'][idx] = shifts
              out['shifts_distr'][idx] = list(shifts_distr)
              
              inter_active = status[s1+1:s2,1].sum()
              inter_coding = status[s1+1:s2,2].sum()
              if s2-s1==1:
                out['inter_active'][idx] = [inter_active,1] 
                out['inter_coding'][idx] = [inter_coding,1]
              else:
                out['inter_active'][idx] = [inter_active,inter_active/(s2-s1-1)] 
                out['inter_coding'][idx] = [inter_coding,inter_coding/(s2-s1-1)]
              #i+=1
  
  return out


def get_firingrate(S_in,meta):
  
  s,S = S_in
  S[S<0.0001*S.max()]=0
  nCells = S.shape[0]
  baseline = np.ones(nCells)
  noise = np.zeros(nCells)
  Ns = (S>0).sum(1)
  n_arr = np.where(Ns>0)[0]
  for n in n_arr:
    trace = S[n,S[n,:]>0]
    baseline[n] = np.median(trace)
    trace -= baseline[n]
    trace *= -1*(trace <= 0)
    N_s = (trace>0).sum()
    noise[n] = np.sqrt((trace**2).sum()/(N_s*(1-2/np.pi)))
  
  s_adapt = sstats.norm.ppf((1-0.01)**(1/Ns))
  N_spikes = np.floor(S / (baseline[:,np.newaxis] + s_adapt[:,np.newaxis]*noise[:,np.newaxis])).sum(1)
  return (s,N_spikes/(S.shape[1]/meta['f']))
