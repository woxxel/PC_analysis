from multiprocessing import get_context

import os, time, warnings, itertools, pickle
import numpy as np
import scipy as sp
from tqdm import *
import itertools

from caiman.utils.utils import load_dict_from_hdf5

from .utils import pickleData, fdr_control, periodic_distr_distance, get_reliability, get_status_arr, get_average
from .utils import cluster_parameters

from matplotlib import pyplot as plt

from .PC_detection.detection.utils import prepare_behavior, get_firingrate
# from plot_PC_analysis import plot_PC_analysis
from .mouse_data_scripts.get_session_specifics import *


warnings.filterwarnings("ignore")

class cluster:
    '''
        class of functions and objects to accumulate previously obtained 
        results processing sessions separately and calculating cross session statistics

        requires
            - processed footprint data 
                'pathAssignments' (single path to matching results file)
                from module "neuron_matching" @ https://github.com/woxxel/neuron_matching.git
            - processed calcium activity
                'paths' (list of processed temporal activity files)
                from module "PC_detection" @ https://github.com/woxxel/PC_detection.git
            - processed behavior files (from... where? extra module?)

    '''

    def __init__(self,
            pathMouse,
            mouse,
            # dataSet='redetect',session_order=None,
            s_corr_min=0.2,suffix=''):

        paramsObj = cluster_parameters(mouse,s_corr_min)
        paramsObj.set_paths(pathMouse,suffix)

        self.params = paramsObj.params
        self.paths = paramsObj.paths
        self.data = paramsObj.data

        self.data['nSes'] = len(self.paths['sessions'])

        
        # if not (nSes is None):
        #     self.data['nSes'] = nSes
        # else:
        #     self.data['nSes'], tmp = get_nPaths(self.pathMouse,'Session')


    def run_complete(self,sessions=None,n_processes=0,reprocess=False):
        
        ### loading data from different states of the analysis process
        # if reprocess: #| (not os.path.exists(self.paths['assignments'])) | 

        self.get_matching()
        # self.classify_sessions(sessions=sessions)

        self.get_behavior()
        

        # self.process_sessions(sessions=sessions,n_processes=n_processes,reprocess=reprocess)

        self.get_stats()

        # self.classify_cluster()

        self.get_PC_fields()

        self.update_status()
        self.compareSessions(n_processes=n_processes)
        # _,_ = self.recalc_firingrate()
        self.save()
        # else:
        #     print('already present')
        #     #self.load('cluster.pkl')
        #     #self.data['nC'] = self.PCs['status'].shape[0]
        #     #self.classify_sessions(sessions)


    def prepare_dicts(self,which=None,overwrite=False):
        
        if not which:
            which = ['status','alignment','matching','stats','compare']
        

        if 'status' in which and (not hasattr(self,'status') or overwrite):
            self.status = {
                'sessions': np.zeros(self.data['nSes'],'bool'),
                'clusters': np.zeros(self.data['nC'],'bool'),
            }

        if 'alignment' in which and (not hasattr(self,'alignment') or overwrite):
            self.alignment = {
                'shift':np.zeros((self.data['nSes'],2)),
                'corr':np.full(self.data['nSes'],np.NaN),
                'borders':np.full((2,2),np.NaN),
                'flow': np.zeros((self.data['nSes'],2)+self.params['dims'])
            }

        if 'matching' in which and (not hasattr(self,'matching') or overwrite):
            self.matching = {
                'IDs': None,
                'com':np.full((self.data['nC'],self.data['nSes'],2),np.NaN),
                'score':np.zeros((self.data['nC'],self.data['nSes'],2)),
            }

        if 'stats' in which and (not hasattr(self,'stats') or overwrite):
            self.stats = {
                'firingrate':   np.full((self.data['nC'],self.data['nSes']),np.NaN),
                'firingmap':    np.zeros((self.data['nC'],self.data['nSes'],self.data['nbin'])),
                # 'trial_map':np.zeros((0,self.data['nSes'],self.params['field_count_max'],self.sessions['trial_ct'].max()),'bool'),

                'SNR':          np.zeros((self.data['nC'],self.data['nSes'])),
                'CNN':          np.zeros((self.data['nC'],self.data['nSes'])),
                'r_values':     np.zeros((self.data['nC'],self.data['nSes'])),

                'MI_value':     np.zeros((self.data['nC'],self.data['nSes'])),
                'MI_p_value':   np.zeros((self.data['nC'],self.data['nSes'])),
                'MI_z_score':   np.zeros((self.data['nC'],self.data['nSes'])),

                'Isec_value':   np.zeros((self.data['nC'],self.data['nSes'])),

                'Bayes_factor': np.zeros((self.data['nC'],self.data['nSes']))
            }

        if 'behavior' in which and (not hasattr(self,'behavior') or overwrite):

            self.behavior = {
                'trial_ct':np.zeros(self.data['nSes'],'int'),
                'trial_frames': {},
                'time_active': np.zeros(self.data['nSes']),
                'speed': np.zeros(self.data['nSes']),
                'performance': {},
            }

        if 'fields' in which and (not hasattr(self,'fields') or overwrite):
            self.fields = {
                'nModes':np.zeros((self.data['nC'],self.data['nSes'])).astype('uint8'),
                'status':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'])).astype('uint8'),
                'Bayes_factor':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'])),
                'reliability':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'])),
                'trial_act':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'],self.behavior['trial_ct'].max()),'bool'),
                'max_rate':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'])),
                'posterior_mass':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'])),
                'baseline':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'],3)),
                'amplitude':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'],3)),
                'width':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'],3)),
                'location':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'],3)),
                'p_x':np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max'],self.data['nbin']))
            }

        if 'compare' in which and (not hasattr(self,'compare') or overwrite):
            self.compare = {

            }


    def process_sessions(self,sessions=None,n_processes=0,reprocess=False):
        
        '''
            obtain basic information from sessions to allow processing and plotting 
            analyses combining results from different steps of the process
        '''

        if reprocess | (not os.path.exists(self.paths['svSessions'])):

            self.get_matching()
            self.get_behavior()
            #self.save([False,True,False,False,False])

#         ## load originally detected data
#         pathLoad = pathcat([path,'results_OnACID.mat'])
        # if os.path.exists(pathLoad):
        #     results_original = sio.whosmat(pathLoad)
        #     if results_original[1][0] == 'C':
        #         self.sessions['N_original'][s] = results_original[1][1][0]
        #     else:
        #         print('nope')

    def get_matching(self):
        '''
            loads information from matching algorithm, such as session-specifics (e.g. shift and session-correlation) and neuron specifics (e.g. center of mass, matching probability, IDs)
        '''

        self.prepare_dicts(which=['alignment','matching','stats'])

        with open(self.paths['assignments'],'rb') as f_open:
            ldData = pickle.load(f_open)
        self.matching['IDs'] = ldData['results']['assignments']

        ## get matching results
        p_matched = ldData['results']['p_matched']
        self.matching['com'] = ldData['results']['cm']

        self.stats['SNR'] = ldData['results']['SNR_comp']
        self.stats['r_values'] = ldData['results']['r_values']
        self.stats['CNN'] = ldData['results']['cnn_preds']
        
        has_reference = False
        for s in range(self.data['nSes']):

            if not os.path.exists(os.path.join(self.paths['sessions'][s],self.paths['fileNamePCFields'])):
                continue

            self.alignment['shift'][s,:] = ldData['data'][s]['remap']['shift'] if has_reference else [0,0]
            self.alignment['corr'][s] = ldData['data'][s]['remap']['c_max'] if has_reference else 1

            self.alignment['flow'][s,...] = ldData['data'][s]['remap']['flow'] if has_reference else np.NaN

            idx_c = np.where(np.isfinite(self.matching['IDs'][:,s]))[0]

            ## match- and best non-match-score should be calculated and stored in matching algorithm
            if not has_reference:
                self.matching['score'][idx_c,s,0] = 1
                self.matching['score'][idx_c,s,1] = np.NaN
            elif s in ldData['data'].keys():

                p_all = ldData['data'][s]['p_same']

                idx_c_first = idx_c[idx_c>=p_all.shape[0]]    # first occurence of a neuron is always certain match!
                self.matching['score'][idx_c_first,s,0] = 1
                self.matching['score'][idx_c_first,s,1] = np.NaN

                idx_c = idx_c[idx_c<p_all.shape[0]]    # remove entries of first-occurence neurons (no matching possible)
                self.matching['score'][idx_c,s,0] = p_matched[idx_c,s]
                scores_now = p_all.toarray()
                self.matching['score'][idx_c,s,1] = [max(scores_now[c,np.where(scores_now[c,:]!=self.matching['score'][c,s,0])[0]]) for c in idx_c]
            has_reference = True
        self.classify_sessions()
        self.classify_cluster()


    def classify_sessions(self,sessions=None):#,max_shift=None,min_corr=None):
        '''
            checks all sessions to pass certain criteria to 
            be included in the further analysis
        '''

        self.prepare_dicts(which=['status'])

        self.status['sessions'][:] = False
        
        ## if 'sessions' is provided (tuple), it specifies range
        ## of sessions to be included
        if sessions is None:
            self.sStart = 0
            self.sEnd = self.data['nSes']
        else:
            self.sStart = max(0,sessions[0]-1)
            self.sEnd = sessions[-1]
        self.status['sessions'][self.sStart:self.sEnd] = True

        ## check for coherence with other sessions (low shift, high correlation)
        abs_shift = np.array([np.sqrt(x**2+y**2) for (x,y) in self.alignment['shift']])
        self.status['sessions'][abs_shift>self.params['session_max_shift']] = False ## huge shift
        
        self.status['sessions'][self.alignment['corr']<self.params['session_min_correlation']] = False ## huge shift
        self.status['sessions'][np.isnan(self.alignment['corr'])] = False
        
        ## reset first session to True if needed (doesnt pass correlation check)
        if self.sStart == 0:
            self.status['sessions'][0] = True
        
        ## finally, check if data can be loaded properly
        for s in np.where(self.status['sessions'])[0]:
            if not os.path.exists(os.path.join(self.paths['sessions'][s],self.paths['fileNameCNMF'])):
                self.status['sessions'][s] = False
        
        thr_high = self.params['dims'] + self.alignment['shift'][self.status['sessions'],:].min(0)
        thr_low = self.alignment['shift'][self.status['sessions'],:].max(0)

        self.alignment['borders'] = np.vstack([thr_low,thr_high])


    def classify_cluster(self,idxes=None,min_cluster_count=None,border_margin=None):

        self.prepare_dicts(which=['status'])

        if not (min_cluster_count is None):
            self.params['min_cluster_count'] = min_cluster_count
        if not (border_margin is None):
            self.params['border_margin'] = border_margin

        self.status['clusters'] = np.ones(self.data['nC']).astype('bool')
        if idxes is None:
            idxes = (self.stats['SNR']>self.params['SNR_thr']) & (self.stats['r_values']>self.params['rval_thr']) & (self.stats['CNN']>self.params['CNN_thr'])
        # self.status['clusters'][(~np.isnan(self.matching['IDs'])).sum(1)<self.params['min_cluster_count']] = False
        self.status['clusters'][idxes[:,self.status['sessions']].sum(1)<self.params['min_cluster_count']] = False

        for i in range(2):
            idx_remove_low = self.matching['com'][:,self.status['sessions'],i] < (self.alignment['borders'][0,i]+self.params['border_margin'])
            self.status['clusters'][np.any(idx_remove_low,1)] = False

            idx_remove_high = self.matching['com'][:,self.status['sessions'],i] > (self.alignment['borders'][1,i]-self.params['border_margin'])
            self.status['clusters'][np.any(idx_remove_high,1)] = False



    def get_behavior(self):
        '''
            accumulates information from all sessions of this mouse
        '''

        self.prepare_dicts(which=['behavior'])

        pathsBehavior = [os.path.join(path,self.paths['fileNameBehavior']) for path in self.paths['sessions']]
        
        for s,path in enumerate(pathsBehavior):
            if not self.status['sessions'][s]: continue
            
            ldData = prepare_behavior(path,nbin_coarse=20,calculate_performance=True)
            self.behavior['trial_ct'][s] = ldData['trials']['ct']
            self.behavior['trial_frames'][s] = ldData['trials']['nFrames']

            self.behavior['speed'][s] = np.nanmean(ldData['velocity'])
            self.behavior['time_active'][s] = ldData['active'].sum()/self.params['f']

            if 'performance' in ldData.keys():
                self.behavior['performance'][s] = ldData['performance']

        self.session_data = get_session_specifics(self.data['mouse'],self.data['nSes'])


    def get_stats(self):

        t_start = time.time()

        self.prepare_dicts(which=['stats'])

        for (s,path) in tqdm(enumerate(self.paths['sessions']),total=self.data['nSes'],leave=False):
            if not self.status['sessions'][s]: continue
            
            idx_c = np.where(np.isfinite(self.matching['IDs'][:,s]))[0]
            n_arr = self.matching['IDs'][idx_c,s].astype('int')
                
            ## load results from the place field detection algorithm
            pathPCFields = os.path.join(path,self.paths['fileNamePCFields'])
            if os.path.exists(pathPCFields):

                with open(pathPCFields,'rb') as f_open:
                    PCFields = pickle.load(f_open)

                self.stats['firingrate'][idx_c,s] = PCFields['firingstats']['rate'][n_arr]
                self.stats['firingmap'][idx_c,s,:] = PCFields['firingstats']['map'][n_arr,:]

                # print('loading MI')
                self.stats['MI_value'][idx_c,s] = PCFields['status']['MI_value'][n_arr]
                self.stats['MI_p_value'][idx_c,s] = PCFields['status']['MI_p_value'][n_arr]
                self.stats['MI_z_score'][idx_c,s] = PCFields['status']['MI_z_score'][n_arr]

                self.stats['Isec_value'][idx_c,s] = PCFields['status']['Isec_value'][n_arr]


        # self.save([False,False,True,False,False])
        # print('stats obtained - time taken: %5.3g'%(time.time()-t_start))


    def get_PC_fields(self):

        t_start = time.time()
        
        self.prepare_dicts(which=['fields'])
        
        for (s,path) in tqdm(enumerate(self.paths['sessions']),total=self.data['nSes'],leave=False):
            if not self.status['sessions'][s]: continue
            # for (s,s0) in enumerate(self.session_order):
            # pathSession = pathcat([self.params['pathMouse'],'Session%02d'%s0])
            pathFields = os.path.join(path,self.paths['fileNamePCFields'])
            if os.path.exists(pathFields):

                idx_c = np.where(~np.isnan(self.matching['IDs'][:,s]))[0]
                n_arr = self.matching['IDs'][idx_c,s].astype('int')
                nCells = len(n_arr)

                with open(pathFields,'rb') as f_open:
                    ld = pickle.load(f_open)
                firingstats_tmp = ld['firingstats']
                fields = ld['fields']

                ### hand over all other values
                self.fields['nModes'][idx_c,s] = np.minimum(2,(np.isfinite(fields['parameter'][n_arr,:,3,0]).sum(1)).astype('int'))

                for (c,n) in zip(idx_c,n_arr):

                    if self.fields['nModes'][c,s] > 0:   ## cell is PC

                        ### hand over field parameters
                        self.fields['location'][c,s,:,:] = fields['parameter'][n,:self.params['field_count_max'],3,(0,1,4)].transpose(1,0)
                        self.fields['width'][c,s,:,:] = fields['parameter'][n,:self.params['field_count_max'],2,(0,1,4)].transpose(1,0)
                        self.fields['amplitude'][c,s,:,:] = fields['parameter'][n,:self.params['field_count_max'],1,(0,1,4)].transpose(1,0)
                        self.fields['baseline'][c,s,:,:] = fields['parameter'][n,:self.params['field_count_max'],0,(0,1,4)].transpose(1,0)

                        self.fields['p_x'][c,s,:,:] = fields['p_x'][n,:self.params['field_count_max'],:]
                        self.fields['posterior_mass'][c,s,:] = fields['posterior_mass'][n,:self.params['field_count_max']]
                        self.fields['Bayes_factor'][c,s,...] = fields['Bayes_factor'][n,:self.params['field_count_max'],0]

                        for f in np.where(~np.isnan(fields['parameter'][n,:,3,0]))[0]:
                            if firingstats_tmp['trial_map'].shape[1]==self.behavior['trial_ct'][s]:
                                self.fields['reliability'][c,s,f], self.fields['max_rate'][c,s,f], self.fields['trial_act'][c,s,f,:self.behavior['trial_ct'][s]] = get_reliability(firingstats_tmp['trial_map'][n,...],firingstats_tmp['map'][n,...],fields['parameter'][n,...],f)
                            else:
                                self.fields['reliability'][c,s,f], self.fields['max_rate'][c,s,f], trial_act = get_reliability(firingstats_tmp['trial_map'][n,1:,...],firingstats_tmp['map'][n,...],fields['parameter'][n,...],f)
                                # self.fields['reliability'][c,s,f] = rel
                                # self.fields['max_rate'][c,s,f] = max_rate
                                print
                                self.fields['trial_act'][c,s,f,:len(trial_act)] = trial_act

                            # self.fields['trial_act'][c,s,f,:self.behavior['trial_ct'][s]] = firingstats_tmp['trial_field'][n,f,:]
                        # print(self.fields['trial_act'][c,s,:,:self.behavior['trial_ct'][s]])
                        # self.fields['reliability'][c,s,:] = fields['reliability'][n,:self.params['field_count_max']]

            else:
                print(f'Data for Session {path} does not exist')

        # self.save([False,False,False,True,False])
        t_end = time.time()
        print('Fields obtained and saved, time spend: %6.4f'%(t_end-t_start))


    ### calculate shifts within clusters
    def compareSessions(self,n_processes=0):
        t_start = time.time()

        self.prepare_dicts(which=['compare'])

        # pathComp = os.path.join(self.paths['pathResults'],'compareSessions.pkl')
        # if reprocess | (not os.path.exists(pathComp)):
        self.compare = {
            'pointer':sp.sparse.lil_matrix((self.data['nC'],self.data['nSes']**2*self.params['field_count_max']**2)),
            'shifts':[],
            'shifts_distr':[],
            'inter_active':[],
            'inter_coding':[]
        }

        t_start=time.time()

        if n_processes>1:
            pool = get_context("spawn").Pool(n_processes)
            loc = np.copy(self.fields['location'][...,0])
            loc[~self.status_fields] = np.NaN
            res = pool.starmap(get_field_shifts,zip(self.status['activity'],self.fields['p_x'],loc))#self.data['nC']))
            
        else:
            loc = np.copy(self.fields['location'][...,0])
            loc[~self.status_fields] = np.NaN
            res = []
            for c in range(self.data['nC']):
                res.append(get_field_shifts(self.status['activity'][c,...],self.fields['p_x'][c,...],loc[c,...]))
            # print('please use parallel processing for this')
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


        # else:
            # self.compare = pickleData([],self.params['svCompare'],'load')
        t_end = time.time()
        # print('Place field shifts calculated - time %5.3f'%(t_end-t_start))



    def update_status(self,complete=True,
            SNR_thr=None,rval_thr=None,CNN_thr=None,pm_thr=None,
            fr_thr=None,Bayes_thr=None,reliability_thr=None,alpha=None,MI_thr=None,
            A0_thr=None,A_thr=None,Arate_thr=None,sigma_thr=None,pmass_thr=None,CI_thr=None,nCluster_thr=None):

        # print('further, implement method to calculate inter-coding intervals, etc, after updating statusses')

        self.thr = {
            'SNR':          self.params['SNR_thr'] if SNR_thr is None else SNR_thr,
            'r_values':     self.params['rval_thr'] if rval_thr is None else rval_thr,
            'CNN':          self.params['CNN_thr'] if CNN_thr is None else CNN_thr,
            'p_matched':    self.params['pm_thr'] if pm_thr is None else pm_thr,

            'firingrate':   self.params['fr_thr'] if fr_thr is None else fr_thr,

            'alpha':        self.params['MI_alpha'] if alpha is None else alpha,
            'MI':           self.params['MI_thr'] if MI_thr is None else MI_thr,
            'CNN':          self.params['CNN_thr'] if CNN_thr is None else CNN_thr,

            'Bayes':        self.params['Bayes_thr'] if Bayes_thr is None else Bayes_thr,
            'reliability':  self.params['reliability_thr'] if reliability_thr is None else reliability_thr,

            'A_0':          self.params['A0_thr'] if A0_thr is None else A0_thr,
            'A':            self.params['A_thr'] if A_thr is None else A_thr,
            'A_rate':       self.params['Arate_thr'] if Arate_thr is None else Arate_thr,
            'sigma':        self.params['sigma_thr'] if sigma_thr is None else sigma_thr,
            'p_mass':       self.params['pmass_thr'] if pmass_thr is None else pmass_thr,
            'CI':           self.params['CI_thr'] if CI_thr is None else CI_thr,

            'nCluster':     self.params['nCluster'] if nCluster_thr is None else nCluster_thr,
        }
        # print(self.thr)

        t_start = time.time()

        ### reset all statuses
        self.status['activity'] = np.zeros((self.data['nC'],self.data['nSes'],6),'bool')

        ### appearance in cluster: defined by (SNR,r_values,CNN), p_matched
        if np.any(~np.isnan(self.stats['SNR'])):
            self.status['activity'][...,0] = (self.stats['SNR']>self.thr['SNR']) & \
                                (self.stats['r_values']>self.thr['r_values']) & \
                                (self.stats['CNN']>self.thr['CNN']) & \
                                (self.matching['score'][...,0]>self.thr['p_matched'])
                                # (((self.matching['score'][...,0]-self.matching['score'][...,1])>self.thr['p_matched']) | (self.matching['score'][...,0]>0.95))
        else:
            self.status['activity'][...,0] = ((self.matching['score'][...,0]-self.matching['score'][...,1])>self.thr['p_matched']) | (self.matching['score'][...,0]>0.95)

        self.status['activity'][...,1] = (self.stats['firingrate']>self.thr['firingrate']) & self.status['activity'][...,0]
        self.classify_cluster(idxes=self.status['activity'][...,1],min_cluster_count=nCluster_thr)

        if complete:
            print('update fields')

            self.fields['status'] = np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max']),'int')
            ### place field: amplitude, A_rate, p_mass, CI-width, width(?),
            
            ## characterize place cells by whether their field passes some thresholds:
            A_rate = self.fields['amplitude'][...,0]/self.fields['baseline'][...,0]
            CI_width = np.mod(self.fields['location'][...,2]-self.fields['location'][...,1],self.data['nbin'])

            morphed_A0_thr = self.thr['A_0']-self.fields['reliability']/2

            idx_fields = (self.fields['baseline'][...,0]>morphed_A0_thr) & \
                        (self.fields['amplitude'][...,0]>self.thr['A']) & \
                        (A_rate>self.thr['A_rate']) & \
                        (self.fields['width'][...,0]>self.thr['sigma']) & \
                        (self.fields['posterior_mass']>self.thr['p_mass']) & \
                        (CI_width<self.thr['CI']) & \
                        (self.fields['Bayes_factor']>self.thr['Bayes']) & \
                        (self.fields['reliability']>self.thr['reliability'])

            self.fix_missed_loc()

            self.status_fields = idx_fields

            
            ##   - Bayes factor, MI(val,p_val,z_score)
            self.stats['MI_p_value'][self.stats['MI_p_value']==0.001] = 10**(-10)    ## need this - can't get any lower than 0.001 with 1000 shuffles...
            idx_PC = np.ones((self.data['nC'],self.data['nSes']),'bool')
            for s in np.where(self.status['sessions'])[0]:
                idx_PC[:,s] = fdr_control(self.stats['MI_p_value'][:,s],self.thr['alpha'])
            idx_PC = idx_PC & np.any(idx_fields,-1) & (self.stats['MI_value']>self.thr['MI'])

            self.status['activity'][...,2] = idx_PC & self.status['activity'][...,1]

            self.status['activity'][:,~self.status['sessions'],:] = False
            self.status['activity'][~self.status['clusters'],:,:] = False

            self.status_fields = self.status_fields & self.status['activity'][...,2][...,np.newaxis]# & (~np.isnan(self.fields['location'][...,0]))
            #idx_reward = (self.fields['location'][...,0]<=self.para['zone_idx']['reward'][-1]) & \
                      #(self.fields['location'][...,0]>=self.para['zone_idx']['reward'][0])
            #self.fields['status'][idx_reward] = 4
            #self.fields['status'][~self.status[...,2],:] = False

            self.session_data = get_session_specifics(self.data['mouse'],self.data['nSes'])

            nbin = self.data['nbin']
            ct_field_remove = 0

            for c in range(self.data['nC']):
                for s in range(self.data['nSes']):
                    # print(self.session_data['RW_pos'])
                    # print(self.session_data['GT_pos'])
                    # rw_pos = self.session_data['RW_pos'][s,:]
                    # gt_pos = self.session_data['GT_pos'][s,:]

                    rw_pos = [50,70]
                    gt_pos = [5,6]

                    if self.status['activity'][c,s,2]:

                        for f in np.where(self.status_fields[c,s,:])[0]:#range(self.fields['nModes'][c,s]):
                            break_it = False
                            # if idx_fields[c,s,f]:
                            field_loc = self.fields['location'][c,s,f,0]
                            field_sig = self.fields['width'][c,s,f,0]
                            field_bin_l = int(field_loc-field_sig) % nbin
                            field_bin_r = int(field_loc+field_sig+1) % nbin

                            field_bool = np.zeros(nbin,'bool')
                            if field_bin_l < field_bin_r:
                                field_bool[field_bin_l:field_bin_r] = True
                            else:
                                field_bool[field_bin_l:] = True
                                field_bool[:field_bin_r] = True

                            ## if cell shows several place fields, check whether they are highly correlated and remove, if so
                            for ff in np.where(self.status_fields[c,s,:])[0]:
                                if f==ff:
                                    continue
                                field2_loc = self.fields['location'][c,s,ff,0]
                                field2_sig = self.fields['width'][c,s,ff,0]
                                field_bin_l = int(field2_loc-field2_sig) % nbin
                                field_bin_r = int(field2_loc+field2_sig+1) % nbin

                                field2_bool = np.zeros(nbin,'bool')
                                if field_bin_l < field_bin_r:
                                    field2_bool[field_bin_l:field_bin_r] = True
                                else:
                                    field2_bool[field_bin_l:] = True
                                    field2_bool[:field_bin_r] = True

                                corr_trial = np.corrcoef(self.fields['trial_act'][c,s,f,:self.behavior['trial_ct'][s]],self.fields['trial_act'][c,s,ff,:self.behavior['trial_ct'][s]])[0,1]

                                if ((field_bool & field2_bool).sum() > 3) & (corr_trial > 0.3):
                                    ct_field_remove += 1
                                    if self.fields['Bayes_factor'][c,s,f] > self.fields['Bayes_factor'][c,s,ff]:
                                        self.status_fields[c,s,ff] = False
                                    else:
                                        self.status_fields[c,s,f] = False
                                        break_it = True
                                        break        ## continue, when this field is bad

                            if break_it:
                                continue
                            
                            ## set status of field according to which region of the VR the field encodes
                            if rw_pos[0] <= self.fields['location'][c,s,f,0] <= rw_pos[1]:
                                self.fields['status'][c,s,f] = 4
                            elif gt_pos[0] <= self.fields['location'][c,s,f,0] <= gt_pos[1]:
                                self.fields['status'][c,s,f] = 3
                            else:
                                self.fields['status'][c,s,f] = 5
                            self.status[c,s,self.fields['status'][c,s,f]] = True

                        self.fields['nModes'][c,s] = np.count_nonzero(self.fields['status'][c,s,:])

            print('fields removed: %d'%ct_field_remove)
        t_end = time.time()
        print('PC-characterization done. Time taken: %7.5f'%(t_end-t_start))


    

    def recalc_firingrate(self,sd_r=-1):

        oof_frate = np.zeros((self.data['nC'],self.data['nSes']))*np.NaN   # out-of-field firingrate
        if_frate = np.zeros((self.data['nC'],self.data['nSes'],self.params['field_count_max']))*np.NaN    # in-field firingrate
        
        for (s,path) in tqdm(enumerate(self.paths['sessions']),total=self.data['nSes'],leave=False):
        
            if not self.status['sessions'][s]: continue

            dataBH = prepare_behavior(os.path.join(path, self.paths['fileNameBehavior']))
            pathLoad = os.path.join(path,self.paths['fileNameCNMF'])
            if os.path.exists(pathLoad):
                ld = load_dict_from_hdf5(pathLoad)
                S = ld['S'][:,dataBH['active']]

                c_arr = np.where(np.isfinite(self.matching['IDs'][:,s]))[0]
                n_arr = self.matching['IDs'][c_arr,s].astype('int')

                for (c,n) in zip(c_arr,n_arr):
                    bool_arr = np.ones(S.shape[1],'bool')
                    if self.status['activity'][c,s,2]:
                        for f in np.where(self.fields['status'][c,s,:])[0]:
                            field_bin = int(self.fields['location'][c,s,f,0])
                            field_bin_l = int(self.fields['location'][c,s,f,0]-self.fields['width'][c,s,f,0]) % self.data['nbin']
                            field_bin_r = int(self.fields['location'][c,s,f,0]+self.fields['width'][c,s,f,0]+1) % self.data['nbin']
                            if field_bin_l < field_bin_r:
                                bool_arr[(dataBH['binpos']>field_bin_l) & (dataBH['binpos']<field_bin_r)] = False
                            else:
                                bool_arr[(dataBH['binpos']>field_bin_l) | (dataBH['binpos']<field_bin_r)] = False
                    oof_frate[c,s],_,_ = get_firingrate(S[n,bool_arr],self.params['f'],sd_r=sd_r)

                    if self.status['activity'][c,s,2]:

                        for f in np.where(self.status_fields[c,s,:])[0]:
                            bool_arr = np.ones(S.shape[1],'bool')
                            field_bin = int(self.fields['location'][c,s,f,0])
                            field_bin_l = int(self.fields['location'][c,s,f,0]-self.fields['width'][c,s,f,0]) % self.data['nbin']
                            field_bin_r = int(self.fields['location'][c,s,f,0]+self.fields['width'][c,s,f,0]+1) % self.data['nbin']
                            # print(field_bin_l,field_bin_r)
                            if field_bin_l < field_bin_r:
                                bool_arr[(dataBH['binpos']<field_bin_l) | (dataBH['binpos']>field_bin_r)] = False
                            else:
                                bool_arr[(dataBH['binpos']<field_bin_l) & (dataBH['binpos']>field_bin_r)] = False

                            for t in range(dataBH['trials']['ct']):
                                if ~self.fields['trial_act'][c,s,f,t]:
                                    bool_arr[dataBH['trials']['start'][t]:dataBH['trials']['start'][t+1]] = False

                            if_frate[c,s,f],_,_ = get_firingrate(S[n,bool_arr],self.params['f'],sd_r=sd_r)
        self.stats['oof_firingrate_adapt'] = oof_frate
        self.stats['if_firingrate_adapt'] = if_frate
        return oof_frate, if_frate



    def get_transition_prob(self,which=['p_post_c','p_post_s']):

        status_arr = ['act','code','stable']


        nSes = self.data['nSes']
        nC = self.data['nC']

        ds_max = 20

        if 'p_post_s' in which:
            self.stats['p_post_s'] = {}
            self.stats['p_post_nodepend_s'] = {}
            self.stats['p_post_RW_s'] = {}
            self.stats['p_post_GT_s'] = {}
            self.stats['p_post_nRnG_s'] = {}
            for status_key in status_arr:
                self.stats['p_post_s'][status_key] = {}
                self.stats['p_post_nodepend_s'][status_key] = {}
                self.stats['p_post_RW_s'][status_key] = {}
                self.stats['p_post_GT_s'][status_key] = {}
                self.stats['p_post_nRnG_s'][status_key] = {}
                # self.stats['p_pre_s'][status_key] = {}
                for status2_key in status_arr:
                    # if status_key=='stable':
                    #     self.stats['p_post_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2,2))*np.NaN
                    #     self.stats['p_post_RW_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2,2))*np.NaN
                    # else:
                    self.stats['p_post_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2))*np.NaN
                    self.stats['p_post_nodepend_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2))*np.NaN
                    self.stats['p_post_RW_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2))*np.NaN
                    self.stats['p_post_GT_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2))*np.NaN
                    self.stats['p_post_nRnG_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2))*np.NaN
                    # self.stats['p_pre_s'][status_key][status2_key] = np.zeros((nSes,ds_max+1,2))*np.NaN

            status, status_dep = get_status_arr(self)
            for key in status_arr:
                status[key] = status[key][self.status['clusters'],...]
                status_dep[key] = status_dep[key][self.status['clusters'],...]
            # status['act'] = status['act'][cluster.stats['cluster_bool'],:]
            # status['act'] = status['act'][cluster.stats['cluster_bool'],:]

            for ds in range(ds_max):

                for s in np.where(self.status['sessions'])[0][:-ds]:
                    if self.status['sessions'][s+ds]:

                        # loc = self.fields['location'][self.status['clusters'],s,:,0]
                        # idx_gt = (loc>GT_pos[0]) & (loc<GT_pos[1])
                        # idx_rw = (loc>RW_pos[0]) & (loc<RW_pos[1])

                        for key in status_arr:
                            for key2 in status_arr:
                                # if key i
                                status_pos = status[key2][status[key][:,s,1] & status_dep[key2][:,s+ds],s+ds,ds]
                                self.stats['p_post_s'][key][key2][s,ds,0] = np.nanmean(status_pos)
                                status_neg = status[key2][(~status[key][:,s,1]) & status_dep[key][:,s] & status_dep[key2][:,s+ds],s+ds,ds]
                                self.stats['p_post_s'][key][key2][s,ds,1] = np.nanmean(status_neg)

                                status_pos = status[key2][status[key][:,s,1],s+ds,ds]
                                self.stats['p_post_nodepend_s'][key][key2][s,ds,0] = np.nanmean(status_pos)
                                status_neg = status[key2][(~status[key][:,s,1]) & status_dep[key][:,s],s+ds,ds]
                                self.stats['p_post_nodepend_s'][key][key2][s,ds,1] = np.nanmean(status_neg)

                                if key in ['code','stable']:
                                    idx_gt = np.any(self.fields['status'][self.status['clusters'],s,:] == 3,1)
                                    idx_rw = np.any(self.fields['status'][self.status['clusters'],s,:] == 4,1)
                                    idx_nRnG = np.any(self.fields['status'][self.status['clusters'],s,:] == 5,1)

                                    status_pos = status[key2][status[key][:,s,1] & status_dep[key2][:,s+ds] & idx_rw,s+ds,ds]
                                    self.stats['p_post_RW_s'][key][key2][s,ds,0] = np.nanmean(status_pos)
                                    status_neg = status[key2][(~status[key][:,s,1]) & status_dep[key][:,s] & status_dep[key2][:,s+ds] & idx_rw,s+ds,ds]
                                    self.stats['p_post_RW_s'][key][key2][s,ds,1] = np.nanmean(status_neg)

                                    status_pos = status[key2][status[key][:,s,1] & status_dep[key2][:,s+ds] & idx_gt,s+ds,ds]
                                    self.stats['p_post_GT_s'][key][key2][s,ds,0] = np.nanmean(status_pos)
                                    status_neg = status[key2][(~status[key][:,s,1]) & status_dep[key][:,s] & status_dep[key2][:,s+ds] & idx_gt,s+ds,ds]
                                    self.stats['p_post_GT_s'][key][key2][s,ds,1] = np.nanmean(status_neg)

                                    status_pos = status[key2][status[key][:,s,1] & status_dep[key2][:,s+ds] & idx_nRnG,s+ds,ds]
                                    self.stats['p_post_nRnG_s'][key][key2][s,ds,0] = np.nanmean(status_pos)
                                    status_neg = status[key2][(~status[key][:,s,1]) & status_dep[key][:,s] & status_dep[key2][:,s+ds] & idx_nRnG,s+ds,ds]
                                    self.stats['p_post_nRnG_s'][key][key2][s,ds,1] = np.nanmean(status_neg)


        if 'p_post_c' in which:
            status, status_dep = get_status_arr(self)
            status_above = {}
            status_above['act'] = ~status['code']
            status_above['code'] = ~status['stable']
            status_above['stable'] = np.ones_like(status['act'],'bool')

            self.stats['p_post_c'] = {}
            for status_key in status_arr:
                self.stats['p_post_c'][status_key] = {}
                for status2_key in status_arr:
                    self.stats['p_post_c'][status_key][status2_key] = np.zeros((nC,ds_max+1,2))*np.NaN

            for ds in tqdm(range(1,ds_max)):
                for c in np.where(self.status['clusters'])[0]:

                    counts = {}
                    for status_key in status_arr:
                        counts[status_key] = {}
                        for status2_key in status_arr:
                            counts[status_key][status2_key] = np.zeros((2,2))

                    for s in np.where(self.status['sessions'])[0][:-ds]:
                        if self.status['sessions'][s+ds]:

                            for status_key in status_arr:
                                for status2_key in status_arr:
                                    if status[status_key][c,s,1]:# & status_above[status_key][c,s,1]:
                                        if status_dep[status2_key][c,s+ds]:
                                            counts[status_key][status2_key][0,0] += 1
                                            if status[status2_key][c,s+ds,ds]:
                                                counts[status_key][status2_key][0,1] += 1

                                    if status_dep[status_key][c,s] & (~status[status_key][c,s,1]):
                                        if status_dep[status2_key][c,s+ds]:
                                            counts[status_key][status2_key][1,0] += 1
                                            if status[status2_key][c,s+ds,ds]:# & status_dep[status2_key][c,s+ds]:
                                                counts[status_key][status2_key][1,1] += 1


                    for status_key in status_arr:
                        for status2_key in status_arr:
                            self.stats['p_post_c'][status_key][status2_key][c,ds,0] = counts[status_key][status2_key][0,1]/counts[status_key][status2_key][0,0] if counts[status_key][status2_key][0,0]>0 else np.NaN
                            self.stats['p_post_c'][status_key][status2_key][c,ds,1] = counts[status_key][status2_key][1,1]/counts[status_key][status2_key][1,0] if counts[status_key][status2_key][1,0]>0 else np.NaN



    def get_locTransition_prob(self,which=['recruit','dismiss','stable']):

        ## for each location, get probability of
        ### 1. recruitment (from silent / non-coding / coding)
        ### 2. stability of place fields
        ### 3. dismissal (towards silent / non-coding / coding)
        nSes = self.data['nSes']
        nbin = self.data['nbin']

        SD = 1.96
        sig_theta = self.stability['all']['mean'][0,2]
        stab_thr = SD*sig_theta

        self.stats['transition'] = {'recruitment':       np.zeros((nSes,nbin,3))*np.NaN,
                                    'stabilization':    np.zeros((nSes,nbin)),
                                    'dismissal':        np.zeros((nSes,nbin,3))*np.NaN}

        for s in np.where(self.status['sessions'])[0]:
            if self.status['sessions'][s-1]:

                ### recruitment
                idx_recruit_silent = (~self.status['activity'][:,s-1,1]) & self.status['activity'][:,s,2]                             # neurons turning from silence to coding
                idx_recruit_active = self.status['activity'][:,s-1,1] & (~self.status['activity'][:,s-1,2]) & self.status['activity'][:,s,2]   # neurons turning from silence to coding
                idx_recruit_coding = self.status['activity'][:,s-1,2] & self.status['activity'][:,s,2]   # neurons turning from silence to coding

                idx_fields = np.where(idx_recruit_silent[:,np.newaxis] & self.status_fields[:,s,:])
                self.stats['transition']['recruitment'][s,:,0] = np.nansum(self.fields['p_x'][idx_fields[0],s,idx_fields[1]],0)

                idx_fields = np.where(idx_recruit_active[:,np.newaxis] & self.status_fields[:,s,:])
                self.stats['transition']['recruitment'][s,:,1] = np.nansum(self.fields['p_x'][idx_fields[0],s,idx_fields[1]],0)

                idx_fields = np.where(idx_recruit_coding[:,np.newaxis] & self.status_fields[:,s,:])
                self.stats['transition']['recruitment'][s,:,2] = np.nansum(self.fields['p_x'][idx_fields[0],s,idx_fields[1]],0)


                ### dismissal
                idx_dismiss_silent = self.status['activity'][:,s-1,2] & (~self.status['activity'][:,s,1])                             # neurons turning from silence to coding
                idx_dismiss_active = self.status['activity'][:,s-1,2] & (~self.status['activity'][:,s,2]) & self.status['activity'][:,s,1]     # neurons turning from silence to coding
                idx_dismiss_coding = self.status['activity'][:,s-1,2] & self.status['activity'][:,s,2]   # neurons turning from silence to coding

                idx_fields = np.where(idx_dismiss_silent[:,np.newaxis] & self.status_fields[:,s-1,:])
                self.stats['transition']['dismissal'][s,:,0] = np.nansum(self.fields['p_x'][idx_fields[0],s-1,idx_fields[1],:],0)

                idx_fields = np.where(idx_dismiss_active[:,np.newaxis] & self.status_fields[:,s-1,:])
                self.stats['transition']['dismissal'][s,:,1] = np.nansum(self.fields['p_x'][idx_fields[0],s-1,idx_fields[1],:],0)

                idx_fields = np.where(idx_dismiss_coding[:,np.newaxis] & self.status_fields[:,s-1,:])
                self.stats['transition']['dismissal'][s,:,2] = np.nansum(self.fields['p_x'][idx_fields[0],s-1,idx_fields[1],:],0)


                ### stabilization
                idx_stabilization = self.status['activity'][:,s-1,2] & self.status['activity'][:,s,2]   # neurons turning from silence to coding
                for c in np.where(idx_stabilization)[0]:
                    field_ref = self.fields['location'][c,s-1,self.status_fields[c,s-1,:],0]
                    field_compare = self.fields['location'][c,s,self.status_fields[c,s,:],0]

                    d = np.abs(np.mod(field_ref[:,np.newaxis]-field_compare[np.newaxis,:]+nbin/2,nbin)-nbin/2)
                    if np.any(d<stab_thr):
                        # print(field_ref)
                        # print(d)
                        f_stable = np.where(d<stab_thr)[0][0]
                        # print(f_stable)
                        # print(np.round(field_ref[f_stable]))
                        self.stats['transition']['stabilization'][s,int(field_ref[f_stable])] += 1


    





    def fix_missed_loc(self):

        idx_c, idx_s, idx_f = np.where(np.isnan(self.fields['location'][...,0]) & (~np.isnan(self.fields['baseline'][...,0])))

        for c,s,f in zip(idx_c,idx_s,idx_f):
            loc = get_average(np.arange(100),self.fields['p_x'][c,s,f,:],periodic=True,bounds=[0,100])
            self.fields['location'][c,s,f,0] = loc
            # print('field at c=%d, s=%d:'%(c,s))
            # print(self.fields['location'][c,s,f,:])
            # print(loc)





    def save(self,svBool=np.ones(5).astype('bool')):
        pass
        # if svBool[0]:
        #     pickleData(self.matching['IDs'],self.paths['svIDs'],'save')
        # if svBool[1]:
        #     pickleData(self.sessions,self.paths['svSessions'],'save')
        # if svBool[2]:
        #     pickleData(self.stats,self.params['svStats'],'save')
        # if svBool[3]:
        #     pickleData(self.fields,self.params['svPCs'],'save')
        # if svBool[4]:
        #     pickleData(self.compare,self.params['svCompare'],'save')

    def load(self,ldBool=np.ones(5).astype('bool')):
        #self.allocate_cluster()
        if ldBool[0]:
            self.matching['IDs'] = pickleData([],self.params['svIDs'],'load')
            self.data['nC'] = self.matching['IDs'].shape[0]
        if ldBool[1]:
            self.sessions = pickleData([],self.params['svSessions'],'load')
        if ldBool[2]:
            self.stats = pickleData([],self.params['svStats'],'load')
        if ldBool[3]:
            self.fields = pickleData([],self.params['svPCs'],'load')
        if ldBool[4]:
            self.compare = pickleData([],self.params['svCompare'],'load')
    



### ------------------------ CURRENTLY UNUSED CODE (START) ----------------------------- ###
            
    def get_trial_fr(self,n_processes):

        self.sessions['trials_fr'] = np.zeros((self.data['nC'],self.data['nSes'],self.sessions['trial_ct'].max()))*np.NaN
        for s,path in tqdm(enumerate(self.paths['sessions'])):

            c_arr = np.where(~np.isnan(self.matching['IDs'][:,s]))[0]
            n_arr = self.matching['IDs'][c_arr,s].astype('int')
            nCells = len(n_arr)

            # pathSession = self.params['paths']['session'][s]
            # pathLoad = os.path.join(path,'results_redetect.mat')
            ld = pickleData(None,self.params['paths']['session'][s],'load')

            pool = get_context("spawn").Pool(n_processes)
            res = pool.starmap(get_firingrate,zip(ld['S'][n_arr,:],itertools.repeat(15),itertools.repeat(2)))

            fr = np.zeros((self.data['nC'],ld['S'].shape[1]))
            for j,r in enumerate(res):
                fr[c_arr[j],:] = r[2]

            for i in range(self.sessions['trial_ct'][s]):
                t_start = self.sessions['trial_frames'][s][i]
                t_end = self.sessions['trial_frames'][s][i+1]
                self.sessions['trials_fr'][:,s,i] = fr[:,t_start:t_end].sum(1)/((t_end-t_start)/self.para['f'])

### ------------------------ CURRENTLY UNUSED CODE (END) ----------------------------- ###



def get_field_shifts(status,p_x,loc):
    nSes = status.shape[0]
    # nSes = 10
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
                    # print(d)
                    d[np.isnan(d)] = nbin
                    f1,f2 = sp.optimize.linear_sum_assignment(d)
                    for f in zip(f1,f2):

                        if d[f] < nbin:
                            idx = np.ravel_multi_index((s1,s2,f[0],f[1]),(nSes,nSes,nfields,nfields))
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