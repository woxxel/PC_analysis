''' contains several functions, defining data and parameters used for analysis

  set_para

'''
import os, pickle
import numpy as np

class cluster_parameters:

	# params = {}
	# paths = {}

    def __init__(self,mouse,s_corr_min):
        self.params = {
            
            'f': 15,   ## measurement frequency
            'dims': (512,512),#np.zeros(2)*np.NaN,


            'field_count_max':            5,

            'session_min_correlation':    s_corr_min,
            'session_max_shift':          50,
            'border_margin':              5,
            'min_cluster_count':          2,

            'SNR_thr':                    3,
            'rval_thr':                   0.2,
            'CNN_thr':                    0.6,
            'pm_thr':                     0.5,

            'fr_thr':                     0,

            'MI_alpha':                   1,
            'MI_thr':                     0.0,

            'Bayes_thr':                  10,
            'reliability_thr':            0.1,
            'A0_thr':                     1,
            'A_thr':                      3,
            'Arate_thr':                  0,
            'sigma_thr':                  2,
            'pmass_thr':                  0.5,
            'CI_thr':                     100,### was nbin before - what does it do?

            'nCluster':                   2
            # 'pathMouse':pathMouse,
            # 'zone_idx':zone_idx,
            # 'zone_mask':zone_mask
        }

        self.data = {
            'mouse':mouse,
            'nSes': None,
            'nC': None,
            'nbin': None,
        }
	
    def set_paths(self,pathAssignments,pathResults,suffix=''):
        
        ## load stored filenames in order in which they were matched
        with open(pathAssignments,'rb') as f_open:
            ld = pickle.load(f_open)
        paths = [os.path.split(session['filePath'])[0] for key,session in ld['data'].items() if isinstance(key,int)]
        
        self.data['nC'],self.data['nSes'] = ld['results']['assignments'].shape
        
        self.paths = {
            'sessions':               paths,
            # 'data':                   pathData,
            'assignments':            pathAssignments,
            'figures':                os.path.join(pathResults,'figures'),#'/home/wollex/Data/Science/PhD/Thesis/pics/Methods',
            
            ### provide names for distinct result files (needed?)
            'fileNamePCFields':       'PC_fields%s.pkl'%suffix,
            'fileNameCNMF':           'OnACID_results.hdf5',
            'fileNameBehavior':       'aligned_behavior.pkl',

            'pathResults':            pathResults,
            'svSessions':             os.path.join(pathResults,'clusterSessions_%s.pkl'%suffix),
            'svIDs':                  os.path.join(pathResults,'clusterIDs_%s.pkl'%suffix),
            'svStats':                os.path.join(pathResults,'clusterStats_%s.pkl'%suffix),
            'svPCs':                  os.path.join(pathResults,'clusterPCs_%s.pkl'%suffix),
            'svCompare':              os.path.join(pathResults,'clusterCompare_%s.pkl'%suffix)
        }

        ## get nbins
        pathPCFields = os.path.join(paths[0],self.paths['fileNamePCFields'])
        with open(pathPCFields,'rb') as f_open:
            PCFields = pickle.load(f_open)
        self.data['nbin'] = PCFields['fields']['p_x'].shape[-1]

        
        
        




def extend_dict(D,n,D2=None,dim=0,exclude=[]):

  '''
    Extends all entries of dictionary D along axis dim to contain n entries
    filled with either placeholder values, or values provided by D2. Can exclude
    keys by providing a list to 'exclude'
  '''
  if not bool(D):
    return D2
  for key in D.keys():
    if not (key in exclude):
      if type(D[key]) is dict:
        if not (D2 is None):
          extend_dict(D[key],n,D2[key],dim)
        else:
          extend_dict(D[key],n,None,dim)
      else:
        dims = np.array(D[key].shape[:])
        dims[dim] = n
        if D[key].dtype == 'float':
          D[key] = np.append(D[key],np.full(dims,np.NaN),dim)
        else:
          D[key] = np.append(D[key],np.zeros(dims).astype(D[key].dtype),dim)
        if not (D2 is None):
          D[key][-n:,...] = D2[key]
  return D


# def clean_dict(D,idx,dim=0):
#   assert dim==0, 'Only works for dimension 0 for now'
#   print('cleaning dictionary of %d entries'%np.count_nonzero(~idx))
#   for key in D.keys():
#     if not (key=='session_shift'):
#       if type(D[key]) is dict:
#         clean_dict(D[key],idx,dim)
#       else:
#         D[key] = D[key][idx,...]



# def set_para(basePath,mouse,s,nP=0,nbin=100,plt_bool=False,sv_bool=False,suffix='2'):

#   ## set paths:
#   pathMouse = pathcat([basePath,mouse])
#   pathSession = pathcat([pathMouse,'Session%02d'%s])

#   coarse_factor = int(nbin/20)
#   #nbin = 100
#   #coarse_factor = 5
#   qtl_steps = 4


#   fact = 1 ## factor from path length to bin number


#   gate_mice = ["34","35","65","66","72","839","840","841","842","879","882","884","886","67","68","91","549","551","756","757","758","918shKO","931wt","943shKO"]
#   nogate_mice = ["231","232","236","243","245","246","762","",""]

#   zone_idx = {}
#   if any(mouse==m for m in gate_mice):        ## gate
#     zone_idx['gate'] = [18,33]
#     zone_idx['reward'] = [75,95]
#     have_gt = True
#   elif any(mouse==m for m in nogate_mice):    ## no gate
#     zone_idx['reward'] = [50,70]#[50,66]#
#     zone_idx['gate'] = [np.NaN,np.NaN]
#     have_gt = False

#   zone_mask = {}
#   zone_mask['reward'] = np.zeros(nbin).astype('bool')#range(zone_idx['reward'][0],zone_idx['reward'][-1])
#   zone_mask['gate'] = np.zeros(nbin).astype('bool')
#   zone_mask['others'] = np.ones(nbin).astype('bool')

#   zone_mask['reward'][zone_idx['reward'][0]:zone_idx['reward'][-1]] = True
#   zone_mask['others'][zone_mask['reward']] = False
#   if have_gt:
#     zone_mask['gate'][zone_idx['gate'][0]:zone_idx['gate'][-1]] = True
#     zone_mask['others'][zone_mask['gate']] = False

#   # zone_mask['others'][40:50] = False  ## remove central wall pattern change?!
#   zone_mask['active'] = nbin+1
#   zone_mask['silent'] = nbin+2

#   print('now')

  


## -----------------------------------------------------------------------------------------------------------------------

  #if nargin == 3:

    #para.t_s = get_t_measures(mouse);
    #para.nSes = length(para.t_s);

    #time_real = false;
  #if time_real
    #t_measures = get_t_measures(mouse);
    #t_mask_m = false(1,t_measures(nSes));
    #for s = 1:nSes-1
      #for sm = s+1:nSes
        #dt = t_measures(sm)-t_measures(s);
        #t_mask_m(dt) = true;
      #end
    #end
    #t_data_m = find(t_mask_m);
    #t_ses = t_measures;
    #t_mask = t_mask_m;
    #t_data = t_data_m;
    #nT = length(t_data);
  #else
    #t_measures = get_t_measures(mouse);
    #t_measures = t_measures(s_offset:s_offset+nSes-1);
##      t_measures = 1:nSes;    ## remove!
##      t_measures
    #t_ses = linspace(1,nSes,nSes);
    #t_data = linspace(1,nSes,nSes);
    #t_mask = true(nSes,1);
    #nT = nSes;
  #end

#   return para
