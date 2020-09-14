from multiprocessing import get_context
#set_start_method("spawn")

import os, time, cv2, warnings, h5py, itertools
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as sstats
from tqdm import *
from time import sleep
import itertools
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp

from utils import get_nPaths, pathcat, extend_dict, clean_dict, pickleData, fdr_control, periodic_distr_distance, fit_plane, z_from_point_normal_plane, get_shift_and_flow, com, calculate_img_correlation, gauss_smooth, get_reliability, get_firingrate, get_status_arr, get_average
from utils_data import set_para
from utils_analysis import get_performance, define_active
from plot_PC_analysis import plot_PC_analysis
from get_session_specifics import get_session_specifics


warnings.filterwarnings("ignore")


class cluster:

    def __init__(self,basePath,mouse,nSes,dataSet='redetect',session_order=None,s_corr_min=0.3,suffix='2'):

        t_start = time.time()

        self.mouse = mouse
        self.pathMouse = pathcat([basePath,mouse])
        self.dataSet = dataSet
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

        self.para = set_para(basePath,mouse,1,suffix=suffix)

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
                     'CI_thr':                     self.para['nbin'],

                     'nCluster':                   2
                     }


    def run_complete(self,sessions=None,n_processes=0,reprocess=False):

        if (not os.path.exists(self.svCluster)) | reprocess:
            self.process_sessions(sessions=sessions,n_processes=n_processes,reprocess=reprocess)
            self.get_IDs()
            self.get_stats(n_processes=n_processes)

            # self.cluster_classification()

            self.get_PC_fields()

            self.update_status()
            self.compareSessions(n_processes=n_processes,reprocess=reprocess)
            _,_ = self.recalc_firingrate()
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
                           'transpose':np.zeros(self.nSes,'bool'),
                           'flow_field':np.zeros(((self.nSes,)+self.meta['dims']+(2,))),
                           'N_original':np.zeros(self.nSes)*np.NaN,
                           #'rotation_anchor':np.zeros((self.nSes,3))*np.NaN,     ## point on plane
                           #'rotation_normal':np.zeros((self.nSes,3))*np.NaN,     ## normal describing plane}
                           'bool':np.zeros(self.meta['nSes']).astype('bool'),
                           'trial_ct':np.zeros(self.meta['nSes'],'int'),
                           'time_active':np.zeros(self.meta['nSes']),
                           'speed':np.zeros(self.meta['nSes'])}

            self.get_reference_frame()
            self.get_behavior()
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
                    self.sessions['corr'][s,1],(y_shift,x_shift) = calculate_img_correlation(Cnref,Cn2,plot_bool=False)
                    corr_T,(y_shift_T,x_shift_T) = calculate_img_correlation(Cnref,Cn2.T,plot_bool=False)

                    if (corr_T > self.sessions['corr'][s,1]) & (corr_T > self.meta['session_min_correlation']):
                        # print('corr: %.3f vs %.3f'%(self.sessions['corr'][s,1],corr_T))
                        self.sessions['corr'][s,1] = corr_T
                        self.sessions['transpose'][s] = ~self.sessions['transpose'][s-1]
                        x_shift, y_shift = [x_shift_T,y_shift_T]
                    else:
                        self.sessions['transpose'][s] = self.sessions['transpose'][s-1]

                    self.sessions['shift'][s,:], self.sessions['flow_field'][s,...], _, self.sessions['corr'][s,0] = get_shift_and_flow(Aref,A2,self.meta['dims'],projection=1,transpose_it=self.sessions['transpose'][s],plot_bool=False)
                    # print('Session %d'%s)

                    #self.sessions['corr'][s,1],(y_shift,x_shift) = calculate_img_correlation(A2ref.sum(1),A2.sum(1),plot_bool=False)
                    if self.sessions['corr'][s,1]>=self.meta['session_min_correlation']:
                        A2ref = A2.copy()
                        Cnref = Cn2.copy()
                else:
                    Aref = A2.copy()    ## keep template of first session for calculating session alignment statistics
                    A2ref = A2.copy()

                    Cnref = Cn2.copy()

            ## load originally detected data
            pathLoad = pathcat([pathSession,'results_OnACID.mat'])
            if os.path.exists(pathLoad):
                results_original = sio.whosmat(pathLoad)
                if results_original[1][0] == 'C':
                    self.sessions['N_original'][s] = results_original[1][1][0]
                else:
                    print('nope')


    def get_behavior(self):

        self.sessions['time_active'] = np.zeros(self.meta['nSes'])
        self.sessions['speed'] = np.zeros(self.meta['nSes'])
        self.sessions['trial_frames'] = {}
        for s,s0 in enumerate(self.session_order):
            pathSession = os.path.join(self.meta['pathMouse'],'Session%02d'%s0)

            data = define_active(pathSession)
            if data is None:
                continue

            self.sessions['trial_ct'][s] = data['trials']['ct']
            self.sessions['trial_frames'][s] = data['trials']['frame']

            self.sessions['speed'][s] = data['velocity'][data['active']].mean()
            self.sessions['time_active'][s] = data['active'].sum()/self.para['f']


        self.session_data = get_session_specifics(self.para['mouse'],self.meta['nSes'])
        self.performance = get_performance(self.meta['pathMouse'],self.session_order,rw_pos=self.session_data['RW_pos'],rw_delay=self.session_data['delay'])

    def get_trial_fr(self,n_processes):

        self.sessions['trials_fr'] = np.zeros((self.meta['nC'],self.meta['nSes'],self.sessions['trial_ct'].max()))*np.NaN
        for s,s0 in tqdm(enumerate(self.session_order)):

            c_arr = np.where(~np.isnan(self.IDs['neuronID'][:,s,1]))[0]
            n_arr = self.IDs['neuronID'][c_arr,s,1].astype('int')
            nCells = len(n_arr)

            pathSession = os.path.join(self.meta['pathMouse'],'Session%02d'%s0)
            pathLoad = os.path.join(pathSession,'results_redetect.mat')
            ld = sio.loadmat(pathLoad,variable_names=['S'])

            pool = get_context("spawn").Pool(n_processes)
            res = pool.starmap(get_firingrate,zip(ld['S'][n_arr,:],itertools.repeat(15),itertools.repeat(2)))

            fr = np.zeros((self.meta['nC'],ld['S'].shape[1]))
            for j,r in enumerate(res):
                fr[c_arr[j],:] = r[2]

            for i in range(self.sessions['trial_ct'][s]):
                t_start = self.sessions['trial_frames'][s][i]
                t_end = self.sessions['trial_frames'][s][i+1]
                self.sessions['trials_fr'][:,s,i] = fr[:,t_start:t_end].sum(1)/((t_end-t_start)/self.para['f'])





    def session_classification(self,sessions=None,max_shift=None,min_corr=None):

        self.sessions['bool'][:] = False
        if sessions is None:
            self.sStart = 0
            self.sEnd = self.meta['nSes']
        else:
            self.sStart = max(0,sessions[0]-1)
            self.sEnd = sessions[-1]

        # print([self.sStart,self.sEnd])
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
        for s in np.where(self.sessions['bool'])[0]:
            pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%self.session_order[s]])
            pathLoad = pathcat([pathSession,self.meta['CNMFname']])
            if not os.path.exists(pathLoad):
                self.sessions['bool'][s] = False

        if self.meta['mouse'] == '762':
          #try:
            self.sessions['bool'][39] = False  ## very bad imaging quality (check video)
            # self.sessions['bool'][66] = False     ## very bad imaging quality (check video)
          #except:
            #1

    def get_IDs(self):

        self.IDs = {'clusterID':np.zeros((0,2)).astype('uint16'),
                    'neuronID':np.zeros((0,self.nSes,3))}

        ld_dat = pickleData([],self.pathMatching,'load')
        try:
            assignments = ld_dat['assignments']
        except:
            assignments = ld_dat['assignment']
        self.meta['nC'] = nC = assignments.shape[0]
        extend_dict(self.IDs,nC)
        self.IDs['clusterID'][range(nC),:] = np.vstack([np.ones(nC),range(nC)]).T;

        for (s,s0) in tqdm(enumerate(self.session_order),total=self.meta['nSes'],leave=False):
            if s >= assignments.shape[1]:
                break
            ### assign neuron IDs
            idx_c = np.where(~np.isnan(assignments[:,s]))[0]
            n_arr = assignments[idx_c,s].astype('int')

            self.IDs['neuronID'][idx_c,s,:] = np.vstack([np.ones(len(n_arr)),n_arr,np.ones(len(n_arr))*s0]).T


    def get_stats(self,n_processes=0,complete=False):

        t_start = time.time()

        self.stats = {'com':np.zeros((0,self.nSes,2))*np.NaN,
                      'match_score':np.zeros((0,self.nSes,2)),
                      'cluster_bool':np.zeros(0,'bool'),

                      'firingrate':np.zeros((0,self.nSes))*np.NaN,
                      'firingmap':np.zeros((0,self.nSes,self.para['nbin'])),
                      # 'trial_map':np.zeros((0,self.nSes,self.meta['field_count_max'],self.sessions['trial_ct'].max()),'bool'),

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

        ## get matching results
        ld_dat = pickleData([],self.pathMatching,'load')
        p_matched = ld_dat['p_matched']
        p_all = ld_dat['p_same']
        cm = ld_dat['cm']
        # print(cm.shape)

        for (s,s0) in tqdm(enumerate(self.session_order),total=self.meta['nSes'],leave=False):
            # print('Session: %d'%s0)
            # if not self.sessions['bool'][s]:
                # continue
            # try:
            idx_c = np.where(np.isfinite(self.IDs['neuronID'][:,s,1]))[0]
            n_arr = self.IDs['neuronID'][idx_c,s,1].astype('int')
            nCells = len(n_arr)

            pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%s0])
            pathLoad = pathcat([pathSession,self.meta['CNMFname']])

            if os.path.exists(pathLoad):
                results_CNMF = sio.loadmat(pathLoad,variable_names=['A','SNR','r_values','CNN'],squeeze_me=True)

                self.stats['com'][idx_c,s,:] = cm[idx_c,s,:]
                if self.dataSet == 'redetect':
                    self.stats['SNR'][idx_c,s] = results_CNMF['SNR'][n_arr]
                    self.stats['r_values'][idx_c,s] = results_CNMF['r_values'][n_arr]
                    self.stats['CNN'][idx_c,s] = results_CNMF['CNN'][n_arr]

            #if complete:
            #self.stats['firingrate'][idx_c,s] = firingrate[s0][n_arr]
            #else:

            pathFiring = os.path.join(pathSession,self.para['svname_firingstats'])
            if os.path.exists(pathFiring):
                try:
                    pathFiring = os.path.join(pathSession,os.path.splitext(self.para['svname_firingstats'])[0] + '.pkl')
                    firingstats_tmp = pickleData([],pathFiring,prnt=False)
                except:
                    print('new detection not found')
                    pathFiring = os.path.join(pathSession,self.para['svname_firingstats'])
                    firingstats_tmp = sio.loadmat(pathFiring,squeeze_me=True)
                # print(idx_c.shape,n_arr.shape)
                self.stats['firingrate'][idx_c,s] = firingstats_tmp['rate'][n_arr]
                self.stats['firingmap'][idx_c,s,:] = firingstats_tmp['map'][n_arr,:]

                try:
                    pathStatus = os.path.join(pathSession,os.path.splitext(self.para['svname_status'])[0] + '.pkl')
                    status = pickleData([],pathStatus,prnt=False)
                except:
                    print('new detection not found')
                    pathStatus = os.path.join(pathSession,self.para['svname_status'])
                    status = sio.loadmat(pathStatus,squeeze_me=True)
                try:
                # print('loading MI')
                    self.stats['MI_value'][idx_c,s] = status['MI_value'][n_arr]
                    self.stats['MI_p_value'][idx_c,s] = status['MI_p_value'][n_arr]
                    self.stats['MI_z_score'][idx_c,s] = status['MI_z_score'][n_arr]

                    self.stats['Isec_value'][idx_c,s] = status['Isec_value'][n_arr]
                except:
                    # print('error in loading MI')
                    pass

            if s == 0:
                self.stats['match_score'][idx_c,s,0] = 1
                self.stats['match_score'][idx_c,s,1] = np.NaN
            elif s in p_all.keys():

                idx_c_first = idx_c[idx_c>=p_all[s].shape[0]]    # first entries are always certain!
                self.stats['match_score'][idx_c_first,s,0] = 1
                self.stats['match_score'][idx_c_first,s,1] = np.NaN

                idx_c = idx_c[idx_c<p_all[s].shape[0]]    # remove entries of first-occurence neurons (no matching possible)
                self.stats['match_score'][idx_c,s,0] = p_matched[idx_c,s]
                scores_now = p_all[s].toarray()
                self.stats['match_score'][idx_c,s,1] = [max(scores_now[c,np.where(scores_now[c,:]!=self.stats['match_score'][c,s,0])[0]]) for c in idx_c]
            # except:
                # print('passing session %d'%s)


        # self.save([False,False,True,False,False])
        # print('stats obtained - time taken: %5.3g'%(time.time()-t_start))


    def get_PC_fields(self):

        self.fields = {'nModes':np.zeros((0,self.nSes)).astype('uint8'),
                       'status':np.zeros((0,self.nSes,self.meta['field_count_max'])).astype('uint8'),
                       'Bayes_factor':np.zeros((0,self.nSes,self.meta['field_count_max'])),
                       'reliability':np.zeros((0,self.nSes,self.meta['field_count_max'])),
                       'trial_act':np.zeros((0,self.nSes,self.meta['field_count_max'],self.sessions['trial_ct'].max()),'bool'),
                       'max_rate':np.zeros((0,self.nSes,self.meta['field_count_max'])),
                       'posterior_mass':np.zeros((0,self.nSes,self.meta['field_count_max'])),
                       'baseline':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                       'amplitude':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                       'width':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                       'location':np.zeros((0,self.nSes,self.meta['field_count_max'],3)),
                       'p_x':np.zeros((0,self.nSes,self.meta['field_count_max'],self.para['nbin']))}
        extend_dict(self.fields,self.meta['nC'])

        t_start = time.time()
        for (s,s0) in tqdm(enumerate(self.session_order),total=self.meta['nSes'],leave=False):

            pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%s0])
            pathFields = os.path.join(pathSession,self.para['svname_fields'])
            if os.path.exists(pathFields):

                idx_c = np.where(~np.isnan(self.IDs['neuronID'][:,s,1]))[0]
                n_arr = self.IDs['neuronID'][idx_c,s,1].astype('int')
                nCells = len(n_arr)

                try:
                    pathFiring = os.path.join(pathSession,os.path.splitext(self.para['svname_firingstats'])[0] + '.pkl')
                    firingstats_tmp = pickleData([],pathFiring,prnt=False)
                except:
                    print('new detection not found')
                    pathFiring = os.path.join(pathSession,self.para['svname_firingstats'])
                    firingstats_tmp = sio.loadmat(pathFiring,squeeze_me=True)

                ### hand over all other values
                try:
                    pathFields = os.path.join(pathSession,os.path.splitext(self.para['svname_fields'])[0] + '.pkl')
                    fields = pickleData([],pathFields,prnt=False)
                except:
                    print('new detection not found')
                    pathFields = os.path.join(pathSession,self.para['svname_fields'])
                    fields = sio.loadmat(pathFields,squeeze_me=True);
                self.fields['nModes'][idx_c,s] = np.minimum(2,(np.isfinite(fields['parameter'][n_arr,:,3,0]).sum(1)).astype('int'))
                # print(firingstats_tmp['trial_field'].shape)
                # return
                for (c,n) in zip(idx_c,n_arr):

                    # self.stats['trial_map'][c,s,:] = firingstats_tmp['trial_map'][n,...]

                    if self.fields['nModes'][c,s] > 0:   ## cell is PC

                        ### hand over field parameters
                        self.fields['location'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],3,(0,1,4)].transpose(1,0)
                        self.fields['width'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],2,(0,1,4)].transpose(1,0)
                        self.fields['amplitude'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],1,(0,1,4)].transpose(1,0)
                        self.fields['baseline'][c,s,:,:] = fields['parameter'][n,:self.meta['field_count_max'],0,(0,1,4)].transpose(1,0)

                        self.fields['p_x'][c,s,:,:] = fields['p_x'][n,:self.meta['field_count_max'],:]
                        self.fields['posterior_mass'][c,s,:] = fields['posterior_mass'][n,:self.meta['field_count_max']]
                        self.fields['Bayes_factor'][c,s,...] = fields['Bayes_factor'][n,:self.meta['field_count_max'],0]

                        for f in np.where(~np.isnan(fields['parameter'][n,:,3,0]))[0]:
                            # if firingstats_tmp['trial_map'].shape[1]==self.sessions['trial_ct'][s]:
                            self.fields['reliability'][c,s,f], self.fields['max_rate'][c,s,f], self.fields['trial_act'][c,s,f,:self.sessions['trial_ct'][s]] = get_reliability(firingstats_tmp['trial_map'][n,...],firingstats_tmp['map'][n,...],fields['parameter'][n,...],f)
                            # else:
                                # self.fields['reliability'][c,s,f], self.fields['max_rate'][c,s,f], trial_act = get_reliability(firingstats_tmp['trial_map'][n,1:,...],firingstats_tmp['map'][n,...],fields['parameter'][n,...],f)
                                # self.fields['reliability'][c,s,f] = rel
                                # self.fields['max_rate'][c,s,f] = max_rate
                                # self.fields['trial_act'][c,s,f,:len(trial_act)] = trial_act

                            # self.fields['trial_act'][c,s,f,:self.sessions['trial_ct'][s]] = firingstats_tmp['trial_field'][n,f,:]
                        # print(self.fields['trial_act'][c,s,:,:self.sessions['trial_ct'][s]])
                        # self.fields['reliability'][c,s,:] = fields['reliability'][n,:self.meta['field_count_max']]

            else:
                print('Data for Session %d does not exist'%s0)

        # self.save([False,False,False,True,False])
        t_end = time.time()
        # print('Fields obtained and saved, time spend: %6.4f'%(t_end-t_start))


    ### calculate shifts within clusters
    def compareSessions(self,reprocess=False,n_processes=0):
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
                loc = np.copy(self.fields['location'][...,0])
                loc[~self.status_fields] = np.NaN
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
                pass
                # print('please use parallel processing for this')

        else:
            self.compare = pickleData([],self.meta['svCompare'],'load')
        t_end = time.time()
        # print('Place field shifts calculated - time %5.3f'%(t_end-t_start))



    def update_status(self,complete=True,
                        SNR_thr=None,rval_thr=None,CNN_thr=None,pm_thr=None,
                        fr_thr=None,Bayes_thr=None,reliability_thr=None,alpha=None,MI_thr=None,
                        A0_thr=None,A_thr=None,Arate_thr=None,sigma_thr=None,pmass_thr=None,CI_thr=None,nCluster_thr=None):

        # print('further, implement method to calculate inter-coding intervals, etc, after updating statusses')

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

                    'A_0':          self.meta['A0_thr'] if A0_thr is None else A0_thr,
                    'A':            self.meta['A_thr'] if A_thr is None else A_thr,
                    'A_rate':       self.meta['Arate_thr'] if Arate_thr is None else Arate_thr,
                    'sigma':        self.meta['sigma_thr'] if sigma_thr is None else sigma_thr,
                    'p_mass':       self.meta['pmass_thr'] if pmass_thr is None else pmass_thr,
                    'CI':           self.meta['CI_thr'] if CI_thr is None else CI_thr,

                    'nCluster':     self.meta['nCluster'] if nCluster_thr is None else nCluster_thr,
                    }
        print(self.thr)

        t_start = time.time()

        ### reset all statuses
        self.status = np.zeros((self.meta['nC'],self.meta['nSes'],6),'bool')

        ### appearance in cluster: defined by (SNR,r_values,CNN), p_matched
        if np.any(~np.isnan(self.stats['SNR'])):
            self.status[...,0] = (self.stats['SNR']>self.thr['SNR']) & \
                                (self.stats['r_values']>self.thr['r_values']) & \
                                (self.stats['CNN']>self.thr['CNN']) & \
                                (self.stats['match_score'][...,0]>self.thr['p_matched'])
                                # (((self.stats['match_score'][...,0]-self.stats['match_score'][...,1])>self.thr['p_matched']) | (self.stats['match_score'][...,0]>0.95))
        else:
            self.status[...,0] = ((self.stats['match_score'][...,0]-self.stats['match_score'][...,1])>self.thr['p_matched']) | (self.stats['match_score'][...,0]>0.95)

        self.status[...,1] = (self.stats['firingrate']>self.thr['firingrate']) & self.status[...,0]
        self.cluster_classification(idxes=self.status[...,1],min_cluster_count=nCluster_thr)

        if complete:
            print('update fields')

            self.fields['status'] = np.zeros((self.meta['nC'],self.meta['nSes'],self.meta['field_count_max']),'int')
            ### place field: amplitude, A_rate, p_mass, CI-width, width(?),
            # try:
                # A_rate = self.stats['if_firingrate_adapt']/self.stats['oof_firingrate_adapt'][...,np.newaxis]
            # except:
            A_rate = self.fields['amplitude'][...,0]/self.fields['baseline'][...,0]
            CI_width = np.mod(self.fields['location'][...,2]-self.fields['location'][...,1],self.para['nbin'])

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

            print(self.status_fields.sum(axis=(0,2)))

            ### place cell: defined by: Bayes factor, MI(val,p_val,z_score)
            self.stats['MI_p_value'][self.stats['MI_p_value']==0.001] = 10**(-10)    ## need this - can't get any lower than 0.001 with 1000 shuffles...
            idx_PC = np.ones((self.meta['nC'],self.meta['nSes']),'bool')
            for s in np.where(self.sessions['bool'])[0]:
                idx_PC[:,s] = fdr_control(self.stats['MI_p_value'][:,s],self.thr['alpha'])

            # print(np.any(idx_fields,-1).sum(0))
            idx_PC = idx_PC & np.any(idx_fields,-1) & (self.stats['MI_value']>self.thr['MI'])

            # print(idx_PC.sum(0))
            self.status[...,2] = idx_PC & self.status[...,1]

            self.status[:,~self.sessions['bool'],:] = False
            self.status[~self.stats['cluster_bool'],:,:] = False

            self.status_fields = self.status_fields & self.status[...,2][...,np.newaxis]# & (~np.isnan(self.fields['location'][...,0]))
            #idx_reward = (self.fields['location'][...,0]<=self.para['zone_idx']['reward'][-1]) & \
                      #(self.fields['location'][...,0]>=self.para['zone_idx']['reward'][0])
            #self.fields['status'][idx_reward] = 4
            #self.fields['status'][~self.status[...,2],:] = False

            nbin = self.para['nbin']
            ct_field_remove = 0

            for c in range(self.meta['nC']):
                for s in range(self.meta['nSes']):

                    rw_pos = self.session_data['RW_pos'][s,:]
                    gt_pos = self.session_data['GT_pos'][s,:]

                    if self.status[c,s,2]:

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

                                corr_trial = np.corrcoef(self.fields['trial_act'][c,s,f,:self.sessions['trial_ct'][s]],self.fields['trial_act'][c,s,ff,:self.sessions['trial_ct'][s]])[0,1]

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

                            if rw_pos[0] <= self.fields['location'][c,s,f,0] <= rw_pos[1]:
                                self.fields['status'][c,s,f] = 4;
                            elif gt_pos[0] <= self.fields['location'][c,s,f,0] <= gt_pos[1]:
                                self.fields['status'][c,s,f] = 3;
                            else:
                                self.fields['status'][c,s,f] = 5;
                            self.status[c,s,self.fields['status'][c,s,f]] = True;

                        self.fields['nModes'][c,s] = np.count_nonzero(self.fields['status'][c,s,:])

            print('fields removed: %d'%ct_field_remove)
        t_end = time.time()
        # print('PC-characterization done. Time taken: %7.5f'%(t_end-t_start))


    def cluster_classification(self,idxes=None,min_cluster_count=None,border_margin=None):

        if not (min_cluster_count is None):
            self.meta['min_cluster_count'] = min_cluster_count
        if not (border_margin is None):
            self.meta['border_margin'] = border_margin

        self.stats['cluster_bool'] = np.ones(self.meta['nC']).astype('bool')
        if idxes is None:
            idxes = (self.stats['SNR']>self.meta['SNR_thr']) & (self.stats['r_values']>self.meta['rval_thr']) & (self.stats['CNN']>self.meta['CNN_thr'])
        # self.stats['cluster_bool'][(~np.isnan(self.IDs['neuronID'][...,1])).sum(1)<self.meta['min_cluster_count']] = False
        self.stats['cluster_bool'][idxes[:,self.sessions['bool']].sum(1)<self.meta['min_cluster_count']] = False

        thr_high = self.meta['dims'] + self.sessions['shift'][self.sessions['bool'],:].min(0)
        thr_low = self.sessions['shift'][self.sessions['bool'],:].max(0)

        self.sessions['borders'] = np.vstack([thr_low,thr_high])

        for i in range(2):
            idx_remove_low = self.stats['com'][:,self.sessions['bool'],i] < (thr_low[i]+self.meta['border_margin'])
            self.stats['cluster_bool'][np.any(idx_remove_low,1)] = False

            idx_remove_high = self.stats['com'][:,self.sessions['bool'],i] > (thr_high[i]-self.meta['border_margin'])
            self.stats['cluster_bool'][np.any(idx_remove_high,1)] = False



    def recalc_firingrate(self,sd_r=-1):

        oof_frate = np.zeros((self.meta['nC'],self.meta['nSes']))*np.NaN   # out-of-field firingrate
        if_frate = np.zeros((self.meta['nC'],self.meta['nSes'],self.meta['field_count_max']))*np.NaN    # in-field firingrate
        for (s,s0) in tqdm(enumerate(self.session_order),total=self.meta['nSes'],leave=False):
            pathSession = pathcat([self.meta['pathMouse'],'Session%02d'%s0])
            pathLoad = pathcat([pathSession,self.CNMF_name])

            for file in os.listdir(pathSession):
                if file.endswith("aligned.mat"):
                    pathBH = os.path.join(pathSession, file)
                    f = h5py.File(pathBH,'r')
                    binpos = np.squeeze(f.get('alignedData/resampled/binpos'))
                    f.close()

            if os.path.exists(pathLoad):
                ld = sio.loadmat(pathLoad,variable_names=['S'])
                S = ld['S']

                c_arr = np.where(np.isfinite(self.IDs['neuronID'][:,s,1]))[0]
                n_arr = self.IDs['neuronID'][c_arr,s,1].astype('int')

                for (c,n) in zip(c_arr,n_arr):
                    bool_arr = np.ones(S.shape[1],'bool')
                    if self.status[c,s,2]:
                        for f in np.where(self.fields['status'][c,s,:])[0]:
                            field_bin = int(self.fields['location'][c,s,f,0])
                            field_bin_l = int(self.fields['location'][c,s,f,0]-self.fields['width'][c,s,f,0]) % self.para['nbin']
                            field_bin_r = int(self.fields['location'][c,s,f,0]+self.fields['width'][c,s,f,0]+1) % self.para['nbin']
                            if field_bin_l < field_bin_r:
                                bool_arr[(binpos>field_bin_l) & (binpos<field_bin_r)] = False
                            else:
                                bool_arr[(binpos>field_bin_l) | (binpos<field_bin_r)] = False
                    oof_frate[c,s],_,_ = get_firingrate(S[n,bool_arr],self.meta['f'],sd_r=sd_r)

                    if self.status[c,s,2]:
                        binpos -= binpos.min()
                        binpos *= 100/binpos.max()
                        idx_teleport = np.where(np.diff(binpos)<-10)[0]+1

                        for f in np.where(self.status_fields[c,s,:])[0]:
                            bool_arr = np.ones(S.shape[1],'bool')
                            field_bin = int(self.fields['location'][c,s,f,0])
                            field_bin_l = int(self.fields['location'][c,s,f,0]-self.fields['width'][c,s,f,0]) % self.para['nbin']
                            field_bin_r = int(self.fields['location'][c,s,f,0]+self.fields['width'][c,s,f,0]+1) % self.para['nbin']
                            # print(field_bin_l,field_bin_r)
                            if field_bin_l < field_bin_r:
                                bool_arr[(binpos<field_bin_l) | (binpos>field_bin_r)] = False
                            else:
                                bool_arr[(binpos<field_bin_l) & (binpos>field_bin_r)] = False

                            for t in range(len(idx_teleport)-1):
                                if ~self.fields['trial_act'][c,s,f,t]:
                                    bool_arr[idx_teleport[t]:idx_teleport[t+1]] = False
                            # print(self.fields['location'][c,s,:,0])
                            # print(self.fields['width'][c,s,:,0])
                            # print(bool_arr.sum())
                            if_frate[c,s,f],_,_ = get_firingrate(S[n,bool_arr],self.meta['f'],sd_r=sd_r)
                    #self.stats['firingrate'][c,s]
        self.stats['oof_firingrate_adapt'] = oof_frate
        self.stats['if_firingrate_adapt'] = if_frate
        return oof_frate, if_frate



    def get_transition_prob(self,which=['p_post_c','p_post_s']):

        status_arr = ['act','code','stable']

        nbin = 100


        nSes = self.meta['nSes']
        nC = self.meta['nC']

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
                status[key] = status[key][self.stats['cluster_bool'],...]
                status_dep[key] = status_dep[key][self.stats['cluster_bool'],...]
            # status['act'] = status['act'][cluster.stats['cluster_bool'],:]
            # status['act'] = status['act'][cluster.stats['cluster_bool'],:]

            for ds in range(ds_max):

                for s in np.where(self.sessions['bool'])[0][:-ds]:
                    if self.sessions['bool'][s+ds]:
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
                                    idx_gt = np.any(self.fields['status'][self.stats['cluster_bool'],s,:] == 3,1)
                                    idx_rw = np.any(self.fields['status'][self.stats['cluster_bool'],s,:] == 4,1)
                                    idx_nRnG = np.any(self.fields['status'][self.stats['cluster_bool'],s,:] == 5,1)

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
                for c in np.where(self.stats['cluster_bool'])[0]:

                    counts = {}
                    for status_key in status_arr:
                        counts[status_key] = {}
                        for status2_key in status_arr:
                            counts[status_key][status2_key] = np.zeros((2,2))

                    for s in np.where(self.sessions['bool'])[0][:-ds]:
                        if self.sessions['bool'][s+ds]:

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
        nSes = self.meta['nSes']
        nbin = self.para['nbin']

        SD = 1.96
        sig_theta = self.stability['all']['mean'][0,2]
        stab_thr = SD*sig_theta

        self.stats['transition'] = {'recruitment':       np.zeros((nSes,nbin,3))*np.NaN,
                                    'stabilization':    np.zeros((nSes,nbin)),
                                    'dismissal':        np.zeros((nSes,nbin,3))*np.NaN}

        for s in np.where(self.sessions['bool'])[0]:
            if self.sessions['bool'][s-1]:

                ### recruitment
                idx_recruit_silent = (~self.status[:,s-1,1]) & self.status[:,s,2]                             # neurons turning from silence to coding
                idx_recruit_active = self.status[:,s-1,1] & (~self.status[:,s-1,2]) & self.status[:,s,2]   # neurons turning from silence to coding
                idx_recruit_coding = self.status[:,s-1,2] & self.status[:,s,2]   # neurons turning from silence to coding

                idx_fields = np.where(idx_recruit_silent[:,np.newaxis] & self.status_fields[:,s,:])
                self.stats['transition']['recruitment'][s,:,0] = np.nansum(self.fields['p_x'][idx_fields[0],s,idx_fields[1]],0)

                idx_fields = np.where(idx_recruit_active[:,np.newaxis] & self.status_fields[:,s,:])
                self.stats['transition']['recruitment'][s,:,1] = np.nansum(self.fields['p_x'][idx_fields[0],s,idx_fields[1]],0)

                idx_fields = np.where(idx_recruit_coding[:,np.newaxis] & self.status_fields[:,s,:])
                self.stats['transition']['recruitment'][s,:,2] = np.nansum(self.fields['p_x'][idx_fields[0],s,idx_fields[1]],0)


                ### dismissal
                idx_dismiss_silent = self.status[:,s-1,2] & (~self.status[:,s,1])                             # neurons turning from silence to coding
                idx_dismiss_active = self.status[:,s-1,2] & (~self.status[:,s,2]) & self.status[:,s,1]     # neurons turning from silence to coding
                idx_dismiss_coding = self.status[:,s-1,2] & self.status[:,s,2]   # neurons turning from silence to coding

                idx_fields = np.where(idx_dismiss_silent[:,np.newaxis] & self.status_fields[:,s-1,:])
                self.stats['transition']['dismissal'][s,:,0] = np.nansum(self.fields['p_x'][idx_fields[0],s-1,idx_fields[1],:],0)

                idx_fields = np.where(idx_dismiss_active[:,np.newaxis] & self.status_fields[:,s-1,:])
                self.stats['transition']['dismissal'][s,:,1] = np.nansum(self.fields['p_x'][idx_fields[0],s-1,idx_fields[1],:],0)

                idx_fields = np.where(idx_dismiss_coding[:,np.newaxis] & self.status_fields[:,s-1,:])
                self.stats['transition']['dismissal'][s,:,2] = np.nansum(self.fields['p_x'][idx_fields[0],s-1,idx_fields[1],:],0)


                ### stabilization
                idx_stabilization = self.status[:,s-1,2] & self.status[:,s,2]   # neurons turning from silence to coding
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

class multi_cluster:
    def __init__(self):
        self.cMice = {}
        self.set_sessions()

    def set_sessions(self,start_ses=None):
        self.sessions = {'34':   {'total':22,
                             'order':range(1,23),
                             'analyze':[3,15]},
                    '35':   {'total':22,
                             'order':range(1,23),
                             'analyze':[3,15],
                             'steady':  [3,64]},
                    '65':   {'total':44,
                             'order':range(1,45),
                             'analyze':[1,20],
                             'steady':  [3,20]}, ## maybe until 28
                    '66':   {'total':45,
                             'order':range(1,46),
                             'analyze':[1,21],
                             'steady':  [3,20]}, ## maybe until 29
                    '72':   {'total':44,
                             'order':range(1,45),
                             'analyze':[1,20],
                             'steady':  [3,20]}, ## maybe until 28
                    '243':  {'total':   71,
                             'order':[range(69,72),range(66,69),range(57,60),range(63,66),range(60,63),range(54,57),range(51,54),range(1,51)],
                             'analyze': [1,71],
                             'steady':  [53,71]},
                    '245':  {'total':73,
                             'order':[range(68,74),range(65,68),range(62,65),range(1,53),range(56,62),range(53,56)],
                             'analyze':[1,73],
                             'steady':  [45,64]},
                    '246':  {'total':63,
                             'order':[range(1,32),range(32,43),range(46,49),range(61,64),range(58,61),range(55,58),range(49,55),range(43,46)],       ## achtung!! between 32 & 32 there are 10 measurements missing over 150h
                             'analyze':[1,60],
                             'steady':  [33,60]},
                    '756':  {'total':30,    ## super huge gap between 22 & 23
                             'order':range(1,31),
                             'analyze':[1,20],
                             'steady':  [3,15]},
                    '757':  {'total':28,    ## super huge gap between 20 & 21
                             'order':range(1,29),
                             'analyze':[1,20],
                             'steady':  [3,15]},
                    '758':  {'total':28,    ## super huge gap between 20 & 21
                             'order':range(1,29),
                             'analyze':[1,20],
                             'steady':  [3,20]},
                    '839':  {'total':24,    ## cpp injections starting at 20
                             'order':range(1,25),
                             'analyze':[3,20]},
                    '840':  {'total':25,    ## cpp injections starting at 20
                             'order':range(1,25),
                             'analyze':[1,18],
                             'steady':  [3,18]},
                    '841':  {'total':np.NaN,
                             'analyze':[np.NaN,np.NaN]},
                    '842':  {'total':np.NaN,
                             'analyze':[np.NaN,np.NaN]},
                    '879':  {'total':15,
                             'order':range(1,16),
                             'analyze':[1,15],
                             'steady':  [3,15]},
                    '882':  {'total':np.NaN,
                             'analyze':[np.NaN,np.NaN]},
                    '884':  {'total':24,    ## cpp injections starting at 20
                             'order':range(1,25),
                             'analyze':[3,16]},
                    '886':  {'total':24,    ## saline injections starting at 22
                             'order':range(1,25),
                             'analyze':[3,19]}, ## dont know yet
                    '549':  {'total':29,    ## super huge gap between 20 & 21, kainate acid @ s21
                             'order':range(1,30),
                             'analyze':[3,20]}, #bad matching?!
                    '551':  {'total':28,    ## super huge gap between 20 & 21, kainate acid @ s21
                             'order':range(1,29),
                             'analyze':[3,20]},
                    '918shKO':  {'total':28,    # RW change at s16
                                 'order':range(1,29),
                             'analyze':[3,15]},
                    '931wt':    {'total':28,    # RW change at s16
                                 'order':range(1,29),
                             'analyze':[3,15]},
                    '943shKO':  {'total':28,    # RW change at s16
                                 'order':range(1,29),
                             'analyze':[3,15]},
                    '231':  {'total':   87,
                             'order':   range(1,88),   ## RW change at s11, s21, s31
                             'analyze': [1,87],
                             'steady':  [32,87]},
                    '232':  {'total':   74,
                             'order':   range(1,75),   ## RW change at s73, s83, s93,s94,s95
                             'analyze': [15,72],
                             'steady':  [15,72]},
                    '236':  {'total':28,
                             'order':range(1,29),
                             'analyze':[3,28]},
                    '762':  {'total':   112,
                             'order':   range(1,113),
                             'analyze': [1,112],
                             'steady':  [17,87]},
                    }
        if not (start_ses is None):
            for mouse in self.cMice.keys():
                self.sessions[mouse]['analyze'][0] = start_ses



    def load_mice(self,mice,load=False,reload=False,session_start=None,suffix='2'):

        for mouse in mice:
            if (not (mouse in self.cMice.keys())) | reload:

                if mouse in ['34','35','65','66','72','243','244','756','757','758','839','840','841','842','879','882','884','886']:
                    basePath = '/media/wollex/Analyze_AS3/Data'
                elif mouse in ['549','551']:
                    basePath = '/media/wollex/Analyze_AS1/others'
                elif mouse in ['918shKO','931wt','943shKO']:
                    basePath = '/media/wollex/Analyze_AS1/Shank'
                elif mouse in ['231','232','236','245','246','762']:
                    basePath = '/media/wollex/Analyze_AS1/linstop'
                else:
                    print('Mouse %s does not exist'% mouse)

                self.cMice[mouse] = cluster(basePath,mouse,self.sessions[mouse]['total'],session_order=self.sessions[mouse]['order'],dataSet='redetect',suffix=suffix)
                if not load:
                    self.cMice[mouse].run_complete(sessions=self.sessions[mouse]['analyze'],n_processes=10,reprocess=True)
                else:
                    self.cMice[mouse].load()

        # if load:
            # self.update_all(which=['sessions','status','compare'],session_start=session_start)

    def update_all(self,mice=None,which=None,SNR_thr=2,rval_thr=0,Bayes_thr=10,rel_thr=0.1,A_thr=3,A0_thr=1,Arate_thr=2,pm_thr=0.3,nCluster_thr=2,session_start=None,session_end=None,sd_r=-1,steady=False):

        if mice is None:
            mice = self.cMice.keys()

        progress = tqdm(mice,leave=True)


        for mouse in tqdm(mice):
            progress.set_description('updating mouse %s...'%mouse)
            if 'sessions' in which:
                if steady:
                    ses = self.sessions[mouse]['steady']
                else:
                    ses = self.sessions[mouse]['analyze']

                if not (session_start is None):
                    ses[0] = session_start
                if ((not (session_end is None)) & (not (session_end==-1))):
                    ses[1] = session_end
                elif session_end==-1:
                    ses[1] = self.sessions[mouse]['total']
                print(mouse)
                print(ses)
                self.cMice[mouse].session_classification(ses)
                # self.cMice[mouse].session_classification()
            if 'behavior' in which:
                self.cMice[mouse].get_behavior()
            if 'stats' in which:
                self.cMice[mouse].get_stats(n_processes=4)
            elif 'firingrate' in which:
                _,_ = self.cMice[mouse].recalc_firingrate(sd_r)

            if 'status' in which:
                self.cMice[mouse].update_status(SNR_thr=SNR_thr,rval_thr=rval_thr,Bayes_thr=Bayes_thr,reliability_thr=rel_thr,A_thr=A_thr,A0_thr=A0_thr,Arate_thr=Arate_thr,MI_thr=0,pm_thr=pm_thr,nCluster_thr=nCluster_thr)
            if 'compare' in which:
                self.cMice[mouse].compareSessions(reprocess=True,n_processes=10)
            if 'transition' in which:
                if not ('stability' in self.cMice[mouse].__dict__):
                    plot_PC_analysis(self.cMice[mouse],plot_arr=[6],N_bs=100,n_processes=10,sv=False,reprocess=True)
                self.cMice[mouse].get_transition_prob()
                self.cMice[mouse].get_locTransition_prob()
            if 'save' in which:
                self.cMice[mouse].save()

# def get_firingrate(S,f=15,sd_r=1):
#
#     S[S<0.0001*S.max()]=0
#     nCells = S.shape[0]
#     baseline = np.ones(nCells)
#     noise = np.zeros(nCells)
#     Ns = (S>0).sum(1)
#     n_arr = np.where(Ns>0)[0]
#     for n in n_arr:
#         trace = S[n,S[n,:]>0]
#         baseline[n] = np.median(trace)
#         trace -= baseline[n]
#         trace *= -1*(trace <= 0)
#         N_s = (trace>0).sum()
#         noise[n] = np.sqrt((trace**2).sum()/(N_s*(1-2/np.pi)))
#
#     sd_r = sstats.norm.ppf((1-0.01)**(1/Ns)) if (sd_r==-1) else sd_r
#     firing_threshold_adapt = baseline[:,np.newaxis] + sd_r[:,np.newaxis]*noise[:,np.newaxis]
#
#     N_spikes = np.floor(S / (firing_threshold_adapt)).sum(1)
#     return N_spikes/(S.shape[1]/f),firing_threshold_adapt,S > firing_threshold_adapt
