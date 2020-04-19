import os, time, cv2, warnings
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as sstats
import progressbar
from time import sleep
import matplotlib.pyplot as plt

from utils import get_nPaths, pathcat, extend_dict, clean_dict, pickleData, fdr_control, periodic_distr_distance, fit_plane, z_from_point_normal_plane
from utils_data import set_para

warnings.filterwarnings("ignore")


class cluster:
  
  def __init__(self,basePath,mouse,nSes):
    
    t_start = time.time()
    
    self.mouse = mouse
    self.pathMouse = pathcat([basePath,mouse])
    
    self.nSes, tmp = get_nPaths(self.pathMouse,'Session')
    if not (nSes is None):
      self.nSes = nSes
    self.para = set_para(basePath,mouse,1)
    
    self.svCluster = pathcat([self.pathMouse,'cluster.pkl'])
    
  
  def allocate_cluster(self):
    
    self.f_max = 2
    
    ### general meta-statistics / data
    self.meta = {'mouse':self.mouse,
                 'pathMouse':self.pathMouse,
                 'nC':np.NaN,
                 'nSes':self.nSes,
                 'dims':(512,512),#np.zeros(2)*np.NaN,
                 'svSessions':pathcat([self.pathMouse,'clusterSessions.pkl']),
                 'svIDs':pathcat([self.pathMouse,'clusterIDs.pkl']),
                 'svActivity':pathcat([self.pathMouse,'clusterActivity.pkl']),
                 'svPCs':pathcat([self.pathMouse,'clusterPCs.pkl']),
                 'svCompare':pathcat([self.pathMouse,'clusterCompare.pkl'])}
    
    ### identity of neurons
    self.IDs = {'clusterID':np.zeros((0,2)).astype('uint16'), 
                'neuronID':np.zeros((0,self.nSes,2))}
    
    self.sessions = {'shift':np.zeros((self.nSes,2))*np.NaN,
                     'rotation_anchor':np.zeros((self.nSes,3))*np.NaN,     ## point on plane
                     'rotation_normal':np.zeros((self.nSes,3))*np.NaN,     ## normal describing plane}
                     'com':np.zeros((0,self.nSes,3))*np.NaN,
                     'match_score':np.zeros((0,self.nSes,2))}
    
    ### activity of neurons
    self.activity = {'status':np.zeros((0,self.nSes,5)).astype('bool'),     ### vals: 0=silent, 1=active, 2=PC(others), 3=PC(gate), 4=PC(reward)
                     'firingrate':np.zeros((0,self.nSes)),
                     'firingmap':np.zeros((0,self.nSes,4,self.para['nbin']))}
    
    ### place fields
    self.PCs = {'fields':{'nModes':np.zeros((0,self.nSes)).astype('uint8'),
                          'status':np.zeros((0,self.nSes,self.f_max)).astype('uint8'),
                          'posterior_mass':np.zeros((0,self.nSes,self.f_max)),
                          'parameter':np.zeros((0,self.nSes,self.f_max,4,3)),
                          'p_x':np.zeros((0,self.nSes,self.f_max,2,self.para['nbin']))},
                'MI':{'value':np.zeros((0,self.nSes)),
                      'p_value':np.zeros((0,self.nSes)),
                      'z_score':np.zeros((0,self.nSes))},
                'Bayes':{'Z':np.zeros((0,self.nSes,2,2)),
                         'factor':np.zeros((0,self.nSes,2))}}
    
  def extend_dicts(self,nC=None):
    if nC is None:
      pathMatching = pathcat([self.pathMouse,'matching/match_results.pkl']);
      ld_dat = pickleData([],pathMatching,'load')
      nC = ld_dat['assignments'].shape[0]
    
    extend_dict(self.IDs,nC)
    extend_dict(self.sessions,nC,exclude=['shift','rotation_anchor','rotation_normal'])
    extend_dict(self.activity,nC)
    extend_dict(self.PCs,nC)
    
    
  def run_complete(self,sessions=None,reprocess=False):
    
    self.allocate_cluster()
    
    if not os.path.exists(self.svCluster) | reprocess:
      self.extend_dicts()
      self.get_matching()
      #self.extend_dicts(self.meta['nC'])
      self.session_classification(sessions)
      self.cluster_classification()
      self.get_PC_fields()
      self.find_PCs()
      self.compareSessions()
      self.save()
    #else:
      #self.load('cluster.pkl')
      #self.meta['nC'] = self.PCs['status'].shape[0]
      #self.session_classification(sessions)
    
  
  def get_matching(self):
    
    #pathMatching = pathcat([self.pathMouse,'matching/results_matching_multi_std=0_thr=70_w=33_OnACID.mat']);
    #ld_dat = sio.loadmat(pathMatching,squeeze_me=True)
    
    ## get template of first session for calculating session alignment statistics
    pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%1])
    pathLoad = pathcat([pathSession,'results_OnACID.mat'])
    ld = sio.loadmat(pathLoad)
    Aref = ld['A']
    #Cn = ld['Cn']
    #Cn -= Cn.min()
    #Cn /= Cn.max()
    #Cn_norm = np.uint8(Cn*(Cn > 0)*255)
    
    self.meta['dims'] = dims = ld['Cn'].shape
    #x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32))
    
    ## get neuron
    pathMatching = pathcat([self.pathMouse,'matching/match_results.pkl']);
    ld_dat = pickleData([],pathMatching,'load')
    assignments = ld_dat['assignments']
    scores = ld_dat['scores']
    self.meta['nC'],nSes = assignments.shape
    
    assert (nSes==self.meta['nSes']), 'Session numbers dont agree - please check %d vs %d'%(nSes,self.meta['nSes'])
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(ld['Cn'],origin='lower')
    plt.subplot(122)
    plt.imshow(Aref.sum(1).reshape(dims),origin='lower')
    plt.title('Session %d'%(s+1))
    plt.show(block=False)
    
    
    #plt.figure()
    #plt.subplot(121)
    #plt.imshow(self.dataIn['A'].sum(1).reshape(dims),origin='lower')
    #plt.subplot(122)
    #plt.imshow(A_tmp.sum(1).reshape(dims),origin='lower')
    #plt.title('Session %d aligned'%(s_ld+1))
    #plt.show(block=False)
    
    #plt.pause(1)
    
    bar = progressbar.ProgressBar(maxval=self.meta['nSes'],widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for s in range(self.meta['nSes']):
      
      ### assign neuron IDs
      idx_c = np.where(~np.isnan(assignments[:,s]))[0]
      n_arr = assignments[idx_c,s].astype('int')
      N = len(n_arr)
      self.IDs['neuronID'][idx_c,s,:] = np.vstack([np.ones(N),n_arr]).T
      
      ### get neuron centroid positions
      pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%(s+1)])
      pathLoad = pathcat([pathSession,'results_OnACID.mat'])
      ld = sio.loadmat(pathLoad)
      A2 = ld['A']
      cm = com(A2,dims[0],dims[1])
      
      if s > 0:
        
        self.sessions['shift'][s,:], flow, (x_grid,y_grid) = get_shift_and_flow(Aref,A2,self.dims,projection=1,plot_bool=True)
        
        x_remap = (x_grid - self.sessions['shift'][s,0] + flow[:,:,0])
        y_remap = (y_grid - self.sessions['shift'][s,1] + flow[:,:,1])
        
        angles = self.find_rotation_from_flow(flow)
        self.sessions['rotation_anchor'][s,:] = angles[0]
        self.sessions['rotation_normal'][s,:] = angles[1]
        
        ## correct for optical flow (rotations, etc)
        self.sessions['com'][idx_c,s,:2] = np.hstack([x_remap[np.round(cm[:,0])],y_remap[np.round(cm[:,1])]]).T
        print('check correction of CoM')
        print(cm)
        print(self.sessions['com'][idx_c,s,:2])
        
        ## get z-position
        self.sessions['com'][idx_c,s,2] = np.squeeze(z_from_point_normal_plane(com_tmp[:,0],com_tmp[:,1],self.sessions['rotation_anchor'][s,:],self.sessions['rotation_normal'][s,:]))
        
        ## store matched score and best score (with matched removed)
        # remove entries of neurons, that were not present before (no matching possible)
        idx_c = idx_c[idx_c<scores[s-1].shape[0]]
        n_arr = assignments[idx_c,s].astype('int')
        
        scores_now = scores[s-1].toarray()
        self.sessions['match_score'][idx_c,s,0] = scores_now[idx_c,n_arr]
        self.sessions['match_score'][idx_c,s,1] = [max(scores_now[c,np.where(scores_now[c,:]!=self.sessions['match_score'][c,s,0])[0]]) for c in idx_c]
        
      else:
        self.sessions['shift'][s,:] = 0
        self.sessions['rotation_anchor'][s,:] = 0
        self.sessions['rotation_normal'][s,:] = 0
        
        self.sessions['com'][idx_c,s,:2] = cm
      
      bar.update(s)
                                                                     
    bar.finish()
    
    pickleData(self.sessions,self.meta['svSessions'],'save')
  
  def find_rotation_from_flow(self,flow,Cn):
    dims = self.meta['dims']
    x = np.hstack([np.ones((dims[0],1)),np.arange(dims[0]).reshape(dims[0],1)]) 
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32),
                                        np.arange(0., dims[1]).astype(np.float32),
                                        np.arange(0,1).astype(np.float32))
    
    W = sstats.norm.pdf(range(dims[0]),dims[0]/2,dims[0]/2)
    W /= W.sum()
    W = np.sqrt(np.diag(W))
    x_w = np.dot(W,x)
    flow_w = np.dot(flow[:,:,1],W)
    x0,res,rank,s = np.linalg.lstsq(x_w,flow_w,rcond=None)
    
    d = -x0[0,:]/x0[1,:]
    r = sstats.linregress(range(dims[0]),d)
    
    tilt_ax = r.intercept+r.slope*range(512)
    
    dist_mat = np.abs((r.slope*x_grid[:,:,0]-y_grid[:,:,0]+r.intercept)/np.sqrt(r.slope**2+1**2))
    slope_normal = np.array([-r.slope,1])
    slope_normal /= np.linalg.norm(slope_normal)
    f_perp = np.dot(flow[:,:,:2],slope_normal)
    
    ## need two cases to capture both, flows away from and towards axis (rotation to a more even vs rotation to a more skewed plane) - not entirely precise, only holds for relatively small angles theta
    h_dat = np.sign(f_perp)*np.sin(np.arccos((dist_mat - np.abs(f_perp))/dist_mat))*dist_mat
    
    data = np.stack([x_grid[...,0],y_grid[...,0],h_dat],2)
    data = data.reshape(dims[0]*dims[1],3)
    mask = ~np.isnan(data[:,2])
    
    ### fit plane to data using covariance of data points. choose between calculating centroid to rotate around or providing from center of tilt axis
    #(p,n),tmp = fit_plane(data[mask,:],anchor=[dims[0]/2-1,tilt_ax[int(dims[0]/2-1)],0])
    (p,n),tmp = fit_plane(data[mask,:])
    angles = np.arccos(n)/(2*np.pi)*360
    #print('angles:',angles)
    
    pl_bool=False
    if pl_bool:
      h_plane = z_from_point_normal_plane(x_grid[...,0],y_grid[...,0],p,n)
      
      pl.figure(figsize=(12,12))
      ax = pl.subplot(221)
      im = ax.imshow(h_dat,origin='lower',cmap='jet',clim=[-60,60])
      pl.colorbar(im)
      ax.plot(d,'b:')
      ax.plot(tilt_ax,'b-')
      ax.set_xlim([0,dims[0]])
      ax.set_ylim([0,dims[1]])
      
      ax2 = pl.subplot(222)
      im2 = ax2.imshow(h_plane,origin='lower',cmap='jet',clim=[-60,60])
      ax2.plot(d,'b:')
      ax2.plot(tilt_ax,'b-')
      ax2.plot(p[0],p[1],'rx')
      ax2.set_xlim([0,dims[0]])
      ax2.set_ylim([0,dims[1]])
      pl.colorbar(im2)
      
      pl.subplot(223)
      pl.imshow(h_dat-h_plane,origin='lower',cmap='jet',clim=[-10,10])
      pl.colorbar()
      
      idxes = 1
      ax = pl.subplot(224,projection='3d')
      ax.plot_surface(x_grid[::50,::50,0],y_grid[::50,::50,0],h_plane[::50,::50],color='k')
      ax.plot_surface(x_grid[::idxes,::idxes,0],y_grid[::idxes,::idxes,0],h_dat[::idxes,::idxes],color='r',alpha=0.5)
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')
      
      pl.show(block=False)
    
    return (p,n)
  
  
  def session_classification(self,sessions=None,max_shift=30):
    
    if sessions is None:
      self.sStart = 0
      self.sEnd = self.meta['nSes']
    else:
      self.sStart = sessions[0]
      self.sEnd = sessions[-1]
    
    self.boolSessions = np.zeros(self.meta['nSes']).astype('bool')
    self.boolSessions[sessions[0]:sessions[-1]] = True
    
    self.boolSessions = np.zeros(self.meta['nSes']).astype('bool')
    self.boolSessions[self.sStart:self.sEnd] = True
    self.boolSessions[np.array([np.sqrt(x**2+y**2) for (x,y) in self.sessions['shift']])>max_shift] = False ## huge shift
    if self.meta['mouse'] == '762':
      try:
        self.boolSessions[40] = False  ## very bad imaging quality (check video)
        self.boolSessions[66] = False     ## very bad imaging quality (check video)
      except:
        1
  
    
  def get_PC_fields(self):
    
    bar = progressbar.ProgressBar(maxval=self.meta['nSes'],widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    t_start = time.time()
    for s in range(self.meta['nSes']):
      bar.update(s)
      pathSession = pathcat([self.pathMouse,'Session%02d'%(s+1)]);
      pathOnACID = pathcat([pathSession,'results_OnACID.mat'])
      
      pathStatus = pathcat([pathSession,'PC_fields_status.mat']);
      pathFields = pathcat([pathSession,'PC_fields_para.mat']);
      pathFiringstats = pathcat([pathSession,'PC_fields_firingstats.mat']);
      
      if os.path.exists(pathFields):
        #print('Now reading %s'%pathSession)
        
        results = {'fields':{},'status':{},'firingstats':{}};
        
        results['fields'] = sio.loadmat(pathFields,squeeze_me=True);
        results['status'] = sio.loadmat(pathStatus,squeeze_me=True);
        
        firingstats_tmp = sio.loadmat(pathFiringstats,squeeze_me=True);
        
        results['firingstats']['rate'] = firingstats_tmp['rate'];
        nCells = len(results['firingstats']['rate'])
        
        results['firingstats']['map'] = np.zeros((nCells,4,self.para['nbin']))
        results['firingstats']['map'][:,0,:] = firingstats_tmp['map'];
        results['firingstats']['map'][:,1,:] = firingstats_tmp['std'];
        results['firingstats']['map'][:,2:,:] = firingstats_tmp['CI'];
        
        idx_c = np.where(~np.isnan(self.IDs['neuronID'][:,s,1]))[0]
        n_arr = self.IDs['neuronID'][idx_c,s,1].astype('int')
        N = len(n_arr)
        
        self.activity['status'][idx_c,s,1] = True
        self.activity['firingrate'][idx_c,s] = results['firingstats']['rate'][n_arr]
        self.activity['firingmap'][idx_c,s,:,:] = results['firingstats']['map'][n_arr,:,:]
        
        self.PCs['MI']['value'][idx_c,s] = results['status']['MI_value'][n_arr]
        self.PCs['MI']['p_value'][idx_c,s] = results['status']['MI_p_value'][n_arr]
        self.PCs['Bayes']['Z'][idx_c,s,:,:] = results['status']['Z'][n_arr,:,:]
        self.PCs['Bayes']['factor'][idx_c,s,:] = results['status']['Bayes_factor'][n_arr,:]
        self.PCs['fields']['nModes'][idx_c,s] = np.minimum(2,(np.isfinite(results['fields']['parameter'][n_arr,:,3,0]).sum(1)).astype('int'))
        for c in range(self.meta['nC']):
          n = self.IDs['neuronID'][c,s,1]
          
          if ~np.isnan(n):
            n = int(n)
            if results['firingstats']['rate'][n] > 0:      ## cell is active
              try:
                self.PCs['MI']['z_score'][c,s] = results['status']['MI_z_score'][n]
              except:
                1
                #print('z score not in here')
              
              if self.PCs['fields']['nModes'][c,s] > 0:   ## cell is PC
                #print(n)
                #print(results['fields']['parameter'][n,:self.f_max,:,(0,3,7)].transpose(1,2,0))
                
                ### hand over field parameters
                self.PCs['fields']['parameter'][c,s,:,:,:] = results['fields']['parameter'][n,:self.f_max,:,(0,3,7)].transpose(1,2,0)
                self.PCs['fields']['p_x'][c,s,:,:,:] = results['fields']['p_x'][n,:self.f_max,:,:]
                self.PCs['fields']['posterior_mass'][c,s,:] = results['fields']['posterior_mass'][n,:self.f_max]
                
            else:
              self.activity['firingrate'][c,s] = 0
              self.activity['status'][c,s,0] = True
          else:
            self.activity['firingrate'][c,s] = 0
            self.activity['status'][c,s,0] = True
          
          
          if s == 0:
            self.IDs['clusterID'][c,:] = [1,c];
      else:
        print('Data for Session %d does not exist'%s)
    t_end = time.time()
    print('Cluster built, time spend: %6.4f'%(t_end-t_start))
    
    ### remove clusters, consisting of single neurons, only
    self.cluster_classification()
    #idx_keep_c = self.PCs['status'][:,:,1].sum(1) >= self.ct_thr
    #clean_dict(self.PCs,idx_keep_c,0)
    
    #idx_keep_s = PCs['status'][:,:,1].sum(0) >= 0
    bar.finish()
  
  
  def cluster_classification(self,ct_thr=2):
    
    margin = 3
    self.clusters_bool = np.ones(self.meta['nC']).astype('bool')
    self.clusters_bool[self.activity['status'][:,:,1].sum(1)<ct_thr] = False
    
    thr_high = self.meta['dims'] + self.sessions['shift'][self.boolSessions,:].min(0)
    thr_low = self.sessions['shift'][self.boolSessions,:].max(0)
    for i in range(2):
      idx_remove_low = self.sessions['com'][:,self.boolSessions,i] < (thr_low[i]+margin)
      self.clusters_bool[np.any(idx_remove_low,1)] = False
      
      idx_remove_high = self.sessions['com'][:,self.boolSessions,i] > (thr_high[i]-margin)
      self.clusters_bool[np.any(idx_remove_high,1)] = False
  
  
  def find_PCs(self,alpha=1,Bayes_thr=0,A_ratio_thr=0,p_mass_thr=0.5):
    
    ### --- post-process: only take into account fields that pass thresholds --- ###
    bar = progressbar.ProgressBar(maxval=self.meta['nC'],widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    t_start = time.time()
    self.activity['status'][:,:,2:] = False     ## reset values
    
    self.PCs['MI']['p_value'][self.PCs['MI']['p_value']==0.001] = 10**(-10)    ## need this - can't get any lower than 0.001 with 1000 shuffles...
    PC_mask = np.zeros((self.meta['nC'],self.meta['nSes'])).astype('bool')
    for s in range(self.meta['nSes']):
      PC_mask[:,s] = fdr_control(self.PCs['MI']['p_value'][:,s],alpha)
    
    A_ratio = self.PCs['fields']['parameter'][:,:,:,1,0]/self.PCs['fields']['parameter'][:,:,:,0,0]
    Bayes = self.PCs['Bayes']['factor'][:,:,0] - self.PCs['Bayes']['factor'][:,:,1]
    for c in range(self.meta['nC']):
      bar.update()
      for s in range(self.meta['nSes']):
        if self.activity['status'][c,s,1]:
          
          if PC_mask[c,s] & (Bayes[c,s] > Bayes_thr):
            for f in range(self.PCs['fields']['nModes'][c,s]):
              if (A_ratio[c,s,f] >= A_ratio_thr) & (self.PCs['fields']['posterior_mass'][c,s,f] >= p_mass_thr):
                if self.para['zone_idx']['reward'][0] <= self.PCs['fields']['parameter'][c,s,f,3,0] <= self.para['zone_idx']['reward'][-1]:
                  self.PCs['fields']['status'][c,s,f] = 4;
                elif self.para['zone_idx']['gate'][0] <= self.PCs['fields']['parameter'][c,s,f,3,0] <= self.para['zone_idx']['gate'][-1]:
                  self.PCs['fields']['status'][c,s,f] = 3;
                else:
                  self.PCs['fields']['status'][c,s,f] = 2;
                self.activity['status'][c,s,self.PCs['fields']['status'][c,s,f]] = True;
            
            self.PCs['fields']['nModes'][c,s] = np.count_nonzero(self.PCs['fields']['status'][c,s,:])
    
    t_end = time.time()
    bar.finish()
    print('PC-characterization done. Time taken: %7.5f'%(t_end-t_start))
  
  
  ### calculate shifts within clusters
  def compareSessions(self,reprocess=False):
    
    t_start = time.time()
    
    pathComp = pathcat([self.para['pathMouse'],'compareSessions.pkl'])
    if reprocess | (not os.path.exists(pathComp)):
      bar = progressbar.ProgressBar(maxval=self.meta['nC'],widgets=[progressbar.Timer(),' / ',progressbar.ETA(),progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
      bar.start()
      self.compare = {'shifts':{},
                      'shifts_distr':{},
                      'inter_coding':{},
                      'ref_pos':{}}
      
      t_start=time.time()
      
      for c in range(self.meta['nC']):
        bar.update(c)
        for s1 in range(self.meta['nSes']):
          if np.any(self.activity['status'][c,s1,2:]):
            for s2 in range(s1+1,self.meta['nSes']):
              if np.any(self.activity['status'][c,s2,2:]):
                
                ic_tmp=np.zeros(5)                ## 0: #active sessions, 1: #coding sessions, 2: fraction active sessions, 3: fraction coding sessions, 4: ?
                ic_tmp[0] = np.sum(self.activity['status'][c,s1+1:s2,1])
                ic_tmp[1] = np.sum(np.any(self.activity['status'][c,s1+1:s2,2:],-1))
                ic_tmp[2] = ic_tmp[0]/(s2-s1-1) if (s2-s1)>1 else 1
                ic_tmp[3] = ic_tmp[1]/(s2-s1-1) if (s2-s1)>1 else 1
                
                #if any(self.PCs['status'][c,s1,2:]) & any(self.PCs['status'][c,s2,2:]):
                self.compare['shifts'][c,s1,s2], self.compare['shifts_distr'][c,s1,s2] = periodic_distr_distance(self.PCs['fields']['p_x'][c,s1,0,1,:],self.PCs['fields']['p_x'][c,s2,0,1,:],self.PCs['fields']['parameter'][c,s1,0,3,0],self.PCs['fields']['parameter'][c,s2,0,3,0],self.para['nbin'],mode='bootstrap')
                
                self.compare['ref_pos'][c,s1,s2] = self.PCs['fields']['parameter'][c,s1,0,3,0]
                self.compare['inter_coding'][c,s1,s2] = ic_tmp
      
      pickleData(self.compare,self.meta['svCompare'],'save')
      bar.finish()
    else:
      self.compare = pickleData([],self.meta['svCompare'],'load')
    t_end = time.time()
    print('Place cells processed - time %5.3f'%(t_end-t_start))
    
  
  def save(self,svBool=np.ones(5).astype('bool')):
    
    if svBool[0]:
      pickleData(self.IDs,self.meta['svIDs'],'save')
    if svBool[1]:
      pickleData(self.sessions,self.meta['svSessions'],'save')
    if svBool[2]:
      pickleData(self.activity,self.meta['svActivity'],'save')
    if svBool[3]:
      pickleData(self.PCs,self.meta['svPCs'],'save')
    if svBool[4]:
      pickleData(self.compare,self.meta['svCompare'],'save')
    
  def load(self,ldBool=np.ones(5).astype('bool')):
    self.allocate_cluster()
    if ldBool[0]:
      self.IDs = pickleData([],self.meta['svIDs'],'load')
      self.meta['nC'] = self.IDs['neuronID'].shape[0]
    if ldBool[1]:
      self.sessions = pickleData([],self.meta['svSessions'],'load')
    if ldBool[2]:
      self.activity = pickleData([],self.meta['svActivity'],'load')
    if ldBool[3]:
      self.PCs = pickleData([],self.meta['svPCs'],'load')
    if ldBool[4]:
      self.compare = pickleData([],self.meta['svCompare'],'load')