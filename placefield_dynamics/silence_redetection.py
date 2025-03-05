import cv2, os, time, scipy
import numpy as np
from matplotlib import pyplot as plt, colors as mcolors, rc, patches as mppatches, lines as mplines
from matplotlib_scalebar.scalebar import ScaleBar
from tqdm import *
from scipy.optimize import linear_sum_assignment

import caiman as cm
from caiman.source_extraction import cnmf as cnmf

from .neuron_detection import *
from .neuron_matching import get_shift_and_flow, calculate_statistics, calculate_p, load_data, save_data, set_paths_default
from .cluster_analysis import cluster_analysis


class silence_redetection:

    def __init__(self,
            pathsSession,
            pathsResults,
            pathsImages=None,
            matlab=False,
            params_in={}):
        # filePath_images=None,
        # fileName_suffix_out='_redetected',
        # fileName_results='results_CaImAn*'):
        '''
            find an as-complete-as-possible set of neurons active in this session by using input footprints from
                1. neurons previously detected in this session
                2. neurons detected in other sessions, that could not be matched to any in this
        '''

        self.matlab = matlab

        # mousePath = os.path.join(path_detected,dataset,mouse)

        # self.fileName_results = fileName_in#fileName_results
        print(pathsSession)
        print(pathsResults)

        pathMouse = Path(pathsSession[0]).parent
        print(f"{pathMouse=}")

        self.cluster = cluster_analysis(
            pathMouse,
            paths_neuron_detection=pathsResults,
            matlab=matlab,
            matching_only=True,
        )
        self.cluster.get_matching()

        self.paths_images = pathsImages
        self.params_in = params_in

        self.params = {
            's_corr_min': 0.05,
            'dims': self.cluster.params['dims'],
        }

        self.dims = self.cluster.params['dims']

    def run_all(self,n_processes=8,plt_bool=False):

        '''
            TODO:
                * split into one code for a single session, and one code iterating over it, to allow hpc-queuing
                (what kind of data is required by each instance? what can be removed?)
        '''

        for s,path in enumerate(self.cluster.paths['sessions']):

            ## test some stuff (such as session correlation and shift distance)
            try:
                self.process_session(s,n_processes=n_processes,ssh_alias=None,plt_bool=plt_bool)
            except:
                print('Error in processing session %d'%s)
                continue

            # if s==0:
            #     process = True
            # else:
            #     process = self.cluster['alignment']['corr'][s] > self.params['s_corr_min']
            #     print(path,self.cluster['alignment']['shifts'][s,:])
            #     print(np.sqrt(np.sum([x**2 for x in self.cluster['alignment']['shifts'][s,:]])))
            #     process &= np.sqrt(np.sum([x**2 for x in self.cluster['alignment']['shifts'][s,:]])) < self.params['max_shift']

            # if process:
            #     self.process_session(s)
            #     # print('process this!')
            #     # pass

    def process_session(self,s,n_processes=8,path_tmp='data/tmp',pathImage=None,ssh_alias='hpc-sofja',plt_bool=True):

        print(self.cluster.status['sessions'])
        if not self.cluster.status['sessions'][s]:
            print(f'Session has not been included in matching or is skipped for another reason - skip redetection of "silent" cells on session {s+1}')
            return

        self.currentS = s
        self.currentSession = self.cluster.paths['sessions'][s]
        self.currentSession_source = pathImage if pathImage else self.paths_images[s]
        self.path_tmp = path_tmp#os.path.join(path_tmp,os.path.split(self.currentSession)[-1])

        print('process',self.currentSession)
        print(f'tmp path: {self.path_tmp}')
        t_start = time.time()

        self.obtain_footprints(s)
        print('### --------- footprints constructed - time took: %5.3fs ---------- ###'%(time.time()-t_start))

        self.prepare_CNMF(n_processes=n_processes,ssh_alias=ssh_alias)    ## for some reason, motion correction gets slower with parallel processing
        self.run_detection(s,n_processes,cleanup=True)
        print('### --------- rerun completed - time took: %5.3fs ---------- ###'%(time.time()-t_start))

        self.analyze_traces()
        self.save_results()
        print('### --------- analysis and saving completed - time took: %5.3fs ---------- ###'%(time.time()-t_start))

        if plt_bool:
            self.analyze_pre_plot()
            self.plot_analysis()
            self.plot_traces()
            self.plot_detection()
            plt.close('all')
        print('### --------- plotting completed - time took: %5.3fs ---------- ###'%(time.time()-t_start))

    def obtain_footprints(self,s,max_diff=None,complete_new=False):

        print(f'Run redetection of "silent" cells on session {s+1}')

        ## prepare some dictionaries for storing in- and output data
        if max_diff is None:
            max_diff = self.cluster.data['nSes']

        dims = self.dims
        self.nC = self.cluster.status['clusters'].sum()

        self.idxes = {
            'in': {
                'active': np.zeros(self.nC,'bool'),
                'silent': np.zeros(self.nC,'bool'),
                'match_to_c': None,     # indexing of matching cluster number
                'match_to_n': None,     # indexing to session neuron number
            },
            'out': {
                'active': None, 'silent': None,
            },
        }

        self.dataIn = {}
        self.dataOut = {}

        ld = load_data(self.cluster.paths['CaImAn_results'][s])
        T = ld['C'].shape[1]

        self.dataIn['A'] = scipy.sparse.csc_matrix((np.prod(self.dims),self.nC))
        self.dataIn['C'] = np.random.rand(self.nC,T)

        if complete_new:
            ## if (for whatever reason) you just want to throw in nC random footprints
            self.idxes['in']['silent'] = np.ones(self.nC,'bool')
            self.dataIn['b'] = np.random.rand(np.prod(dims),1)
            self.dataIn['f'] = np.random.rand(1,T)
        else:
            ## find active and silent neurons in session s
            isSilent = self.cluster.status['clusters'] & np.isnan(self.cluster.matching['IDs'][:,s])
            self.idxes['in']['nSilent'] = isSilent.sum()
            isActive = self.cluster.status['clusters'] & np.isfinite(self.cluster.matching['IDs'][:,s])
            self.idxes['in']['nActive'] = isActive.sum()

            self.idxes['in']['active'][:self.idxes['in']['nActive']] = True
            self.idxes['in']['silent'][self.idxes['in']['nActive']:] = True

            print('silent:', self.idxes['in']['nSilent'], 'active:', self.idxes['in']['nActive'])

            ## load footprints of active cells from session s
            ld = load_data(self.cluster.paths['CaImAn_results'][s])
            if 'Cn' in ld.keys():
                Cn_ref = ld['Cn'].T 
            else:
                # self.log.warning('Cn not in result files. constructing own Cn from footprints!')
                Cn_ref = np.array(ld['A'].sum(axis=1).reshape(*self.params['dims']))

            # Cn_ref = ld['Cn'].T

            c_idx = np.concatenate([np.where(isActive)[0],np.where(isSilent)[0]])
            n_idx = self.cluster.matching['IDs'][isActive,s].astype('int')

            self.idxes['in']['match_to_c'] = c_idx
            self.idxes['in']['match_to_n'] = n_idx

            if self.cluster.alignment['transposed'][s]:
                print(f'load transposed from session {s+1}')
                self.dataIn['A'][:,:self.idxes['in']['nActive']] = scipy.sparse.hstack([(a/a.sum()).reshape(self.dims).transpose().reshape(-1,1) for a in ld['A'][:,n_idx].T])
            else:
                self.dataIn['A'][:,:self.idxes['in']['nActive']] = scipy.sparse.vstack([a/a.sum() for a in ld['A'][:,n_idx].T]).T

            # if not (ld['C'].shape[0] == ld['A'].shape[1]):
            # ld['C'] = ld['C'].transpose()
            ## load temporal components of active cells from session s
            T1 = ld['C'].shape[1]   # adjusted for a session, where T != T1
            self.dataIn['C'][:self.idxes['in']['nActive'],:T1] = ld['C'][n_idx,:]

            ## load background components from session s
            if not (ld['b'].shape[0] == ld['A'].shape[0]):
                ld['b'] = ld['b'].transpose()
            self.dataIn['b'] = ld['b']
            if not (ld['f'].shape[1] == self.dataIn['C'].shape[1]):
                ld['f'] = ld['f'].transpose()
            self.dataIn['f'] = ld['f']

        ## for each silent neuron identify footprints as close as possible before & after current session
        s_ref = np.full((self.idxes['in']['nSilent'],2),-1,'int')
        n_ref = np.full((self.idxes['in']['nSilent'],2),-1,'int')

        # '''
        #     INSERT HERE CHECK, SUCH THAT STATUS_CLUSTER AND STATUS_SESSION IS RESPECTED!
        # '''
        for i,c in enumerate(self.idxes['in']['match_to_c'][self.idxes['in']['silent']]):
            s_pre = np.where(np.isfinite(self.cluster.matching['IDs'][c,:s]))[0]
            s_post = s+1+np.where(np.isfinite(self.cluster.matching['IDs'][c,s+1:]))[0]
            # print(c,s_pre,s_post)
            if len(s_pre)>0:
                s_ref[i,0] = s_pre[-1] if (s-s_pre[-1]) <= max_diff else -1
                if s_ref[i,0]>=0:
                    n_ref[i,0] = self.cluster.matching['IDs'][c,int(s_ref[i,0])].astype(int)

            if len(s_post)>0:
                s_ref[i,1] = s_post[0] if (s_post[0] - s) <= max_diff else -1
                if s_ref[i,1]>=0:
                    n_ref[i,1] = self.cluster.matching['IDs'][c,int(s_ref[i,1])].astype(int)

            # if n_ref[i,1] == 510:
            # print('@510:',s_ref[i,1],n_ref[i,1])
        s_load = np.unique(s_ref[s_ref>=0])

        # print('s_ref:',s_ref)
        # print('n:',n_ref)
        # assert False, 'check n_ref!'
        ## construct footprint of silent cells as interpolation between footprints of adjacent sessions and adjust for shift & rotation

        progress = tqdm.tqdm(s_load,total=len(s_load),desc='Loading footprints for processing Session %d...'%(s+1))
        for s_ld in progress:

            ld = load_data(self.cluster.paths['CaImAn_results'][s_ld])
            # print(f"{s_ld=}, path={self.cluster.paths['CaImAn_results'][s_ld]}")
            # print(f'n=,{n_ref[s_ref==s_ld]}')
            A_tmp = ld['A'].tocsc()
            # Cn = ld['Cn'].T
            if 'Cn' in ld.keys():
                Cn = ld['Cn'].T
            else:
                # self.log.warning('Cn not in result files. constructing own Cn from footprints!')
                Cn = np.array(ld['A'].sum(axis=1).reshape(*self.params['dims']))

            if self.cluster.alignment['transposed'][s_ld]:
                # print('load transposed from session %d'%(s_ld+1))
                A_tmp = scipy.sparse.hstack([img.reshape(self.dims).transpose().reshape(-1,1) for img in A_tmp.transpose()]).tocsc()
                Cn = Cn.T

            x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32), np.arange(0., dims[1]).astype(np.float32))
            (x_shift,y_shift),flow,_,_ = get_shift_and_flow(Cn_ref,Cn,dims,projection=None,plot_bool=False)

            x_remap = (x_grid - x_shift + flow[:,:,0])
            y_remap = (y_grid - y_shift + flow[:,:,1])

            A_tmp = scipy.sparse.hstack([
                scipy.sparse.csc_matrix(                    # cast results to sparse type
                    cv2.remap(
                        fp.reshape(dims),   # reshape image to original dimensions
                        x_remap, y_remap,                 # apply reverse identified shift and flow
                        cv2.INTER_CUBIC                 
                    ).reshape(-1,1)                       # reshape back to allow sparse storage
                    ) for fp in A_tmp[:,n_ref[s_ref==s_ld]].toarray().T        # loop through all footprints
                ])

            self.dataIn['A'][:,self.idxes['in']['nActive'] + np.where(s_ref==s_ld)[0]] += 1./abs(s_ld-s) * scipy.sparse.vstack([a/a.sum() for a in A_tmp.T]).T
            # print('%d neuron footprints taken from session %d'%(A_tmp.shape[1],s_ld+1))

        max_thr = 0.001
        self.dataIn['A'] = scipy.sparse.vstack([a.multiply(a>(max_thr*a.max()))/a.sum() for a in self.dataIn['A'].T]).T

    def prepare_CNMF(self,
                n_processes=8,
                ssh_alias='hpc-sofja'):
        '''
            function to set up a CaImAn batch-processing instance
        '''

        isFolder = not len(os.path.splitext(self.currentSession_source)[1])

        if ssh_alias:
            ## if file(s) are on remote, copy over data, first
            # path_tmp_images = os.path.join(self.path_tmp,'images')
            path_tmp_images = self.path_tmp
            get_data_from_server(self.currentSession_source,path_tmp_images,ssh_alias)
        else:
            # path_tmp_images = os.path.join(self.currentSession_source,'images')
            path_tmp_images = self.currentSession_source

        ## if files present in single tifs, only (its the case for my data), run batch-creation
        if isFolder:
            path_to_stack = make_stack_from_single_tifs(path_tmp_images,self.path_tmp,data_type='float16',clean_after_stacking=True)
        else:
            path_to_stack = os.path.join(self.path_tmp,os.path.basename(self.currentSession_source))
            shutil.copy(self.currentSession_source,path_to_stack)

            # path_to_stack = path_tmp_images

        # print(f'{path_to_stack=}')
        # run motion correction separately (to not having to call everything manually...)
        CaImAn_params['fnames'] = None   # reset to remove previously set data
        path_to_motion_correct = motion_correct(path_to_stack,CaImAn_params,n_processes=n_processes)

        # path_to_motion_correct = os.path.join(self.currentSession,'20240607_58_ROI1_00001_els__d1_512_d2_512_d3_1_order_F_frames_13329.mmap')
        CaImAn_params['fnames'] = [path_to_motion_correct]
        for key in self.params_in:
            CaImAn_params[key] = self.params_in[key]

        self.opts = cnmf.params.CNMFParams(params_dict=CaImAn_params)

    def run_detection(self,s,n_processes,as_c=False,cleanup=False):

        '''
            calls cnmf to run redetection with predefined neuron footprints from current, as well as adjacent sessions
                1. temporal trace updates on ROIs and background
                2. spatial update on silent neurons (?) - rather not!
        '''
        # if not self.preprocessed:
        use_parallel = n_processes>1
        if use_parallel:
            c, self.dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)
        else:
            self.dview=None
            n_processes=1

        self.cnm = cnmf.CNMF(n_processes,dview=self.dview)

        Yr, dims, T = cm.load_memmap(CaImAn_params['fnames'][0])
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        Y = np.reshape(Yr.T, [T] + list(dims), order='F')

        self.cnm.estimates.dims = self.cnm.dims = dims
        if self.cluster.alignment['transposed'][s]:
            print('transpose all footprints')
            self.dataIn['A'] = scipy.sparse.hstack([img.reshape(self.dims).transpose().reshape(-1,1) for img in self.dataIn['A'].transpose()]).tocsc()

        self.cnm.estimates.A = self.dataIn['A']

        self.cnm.estimates.C = self.dataIn['C']
        self.cnm.estimates.b = self.dataIn['b']
        self.cnm.estimates.f = self.dataIn['f']

        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))

        t_start = time.time()
        if as_c:
            print('Switch memory order to C-style')
            Yr = np.ascontiguousarray(Yr)
            print(f'done! after {time.time()-t_start}s')
        # if np.isfortran(Yr):
        #     raise Exception('The file is in F order, it should be in C order (see save_memmap function')

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
        except AttributeError:  # if no memmapping cause working with small data
            pass

        if self.cnm.estimates.sn is None:
            print('preprocessing...')
            Yr = self.cnm.preprocess(Yr)
            self.preprocessed = True
            print(f'done! after {time.time()-t_start}s')

        print('temporal and spatial update...')
        self.cnm.update_temporal(Yr)
        self.cnm.update_spatial(Yr)
        self.cnm.update_temporal(Yr)
        print(f'done! after {time.time()-t_start}s')

        print('deconvolving and evaluating...')
        self.cnm.deconvolve()
        self.cnm.estimates.normalize_components()
        self.cnm.estimates.evaluate_components(Y,self.opts)

        self.cnm.estimates.Cn = cm.load(Y.filename, subindices=slice(0,None,10)).local_correlations(swap_dim=False)

        if use_parallel:
            cm.cluster.stop_server(dview=self.dview)      ## restart server to clean up memory

        print(f'done! after {time.time()-t_start}s')
        # if cleanup:
        #     os.remove(CaImAn_params['fnames'][0])
        #     if os.path.isdir(self.path_tmp):
        #         shutil.rmtree(self.path_tmp)

    def analyze_traces(self,redo=False):

        dims = self.dims
        # nC = self.cluster.data['nC']

        # self.cnm.estimates.C -= np.percentile(self.cnm.estimates.C,0.05,axis=1)[:,np.newaxis]

        # self.idxes['in'] = np.array(self.dataIn['A'].sum(0)>0).ravel()
        # idxes_Ain = np.where(self.idxes['in'])[0]

        ### find proper IDs --- some are removed, some are weirdly remapped to completely other places

        self.analysis = {}
        print('Finding proper IDs...')
        # nC_in = self.idxes['in'].sum()
        nC_in = self.dataIn['A'].shape[1]
        nC_out = self.cnm.estimates.A.shape[1]
        if nC_out < nC_in:
            print('\t %d neurons were removed ...'%(nC_in-nC_out))

        Cout = self.cnm.estimates.C - np.percentile(self.cnm.estimates.C,0.05,axis=1)[:,np.newaxis]
        ## find next footprints with low distance

        Cin = self.dataIn['C']

        self.idxes['out']['active'] = np.zeros(nC_out,'bool')
        self.idxes['out']['active'][self.cnm.estimates.idx_components] = True

        ### check whether some neurons have significant overlap and need to be removed
        ## ?? run twice, once to calculate statistics for all, once to remove footprints from the pool of active neurons
        _,_,idx_remove = calculate_statistics(
            self.cnm.estimates.A,
            SNR_comp=self.cnm.estimates.SNR_comp,C=self.cnm.estimates.C,
            # idx_eval=self.idxes['out']['active'],
            binary=False)
        # _,_,idx_remove = calculate_statistics(
        #     self.cnm.estimates.A,
        #     SNR_comp=self.cnm.estimates.SNR_comp,C=self.cnm.estimates.C,
        #     idx_eval=self.idxes['out']['active'],
        #     binary=False)
        # print(idx_remove)
        self.idxes['out']['active'][idx_remove] = False

        self.analysis['D_ROIs'],self.analysis['fp_corr'], _ = calculate_statistics(self.cnm.estimates.A,A_ref=self.dataIn['A'],binary=False,dims=dims)
        d_ROIs, fp_corr, _ = calculate_statistics(self.cnm.estimates.A,A_ref=self.dataIn['A'],idx_eval=self.idxes['out']['active'],binary=False,dims=dims)

        p_same = calculate_p(d_ROIs,fp_corr[1,...],self.cluster.matching['f_same'])
        matches = linear_sum_assignment(1 - p_same.toarray())
        p_matched = p_same.toarray()[matches]

        matches = np.vstack([*matches]).T

        is_match = p_matched > 0.1  ## low match probability is fine here

        ## for previously active, check whether activity is 'significantly' correlated
        for i,(n_in,n_out) in enumerate(matches):
            if is_match[i] & self.idxes['in']['active'][n_in]:
                C_corr = np.corrcoef(Cin[n_in,:],Cout[n_out,:])[0,1]

                is_match[i] &= C_corr > 0.3
                if not is_match[i]:
                    print('match ',n_in,n_out," not accepted")
        matches = matches[is_match,:]
        p_matched = p_matched[is_match]

        idx_matched = np.zeros(nC_out,'bool')
        idx_matched[matches[:,1]] = True

        # cast prev active to new idxes and update arrays accordingly
        self.idxes['in']['match_to_out'] = np.full(nC_in,-1,dtype='int')
        self.idxes['in']['match_to_out'][matches[:,0]] = matches[:,1]

        self.idxes['out']['match_to_in'] = np.full(nC_out,-1,dtype='int')
        self.idxes['out']['match_to_in'][matches[:,1]] = matches[:,0]

        idx_out_active_before = np.zeros(nC_out,'bool')
        idx_out_active_before[self.idxes['in']['match_to_out'][(self.idxes['in']['match_to_out']>=0) & self.idxes['in']['active']].astype('int')] = True

        self.idxes['out']['active_still'] = self.idxes['out']['active'] & idx_out_active_before
        self.idxes['out']['active_new'] = self.idxes['out']['active'] & ~idx_out_active_before & idx_matched
        self.idxes['out']['silent_still'] = ~self.idxes['out']['active'] & ~idx_out_active_before
        self.idxes['out']['silent_new'] = ~self.idxes['out']['active'] & idx_out_active_before

        self.idxes['out']['active_unmatched'] = self.idxes['out']['active'] & ~idx_out_active_before & ~idx_matched

        self.idxes['out']['silent'] = []

        nC_new = self.idxes['out']['active_unmatched'].sum()
        print('%d new neurons detected'%nC_new)

        print(f"active before: {self.idxes['in']['active'].sum()}, \t silent before: {(~self.idxes['in']['active']).sum()}")
        print(f"active still: {self.idxes['out']['active_still'].sum()}, \t active new: {self.idxes['out']['active_new'].sum()}")
        print(f"silent still: {self.idxes['out']['silent_still'].sum()}, \t silent new: {self.idxes['out']['silent_new'].sum()}")

        self.dataOut = {}
        for key in ['A','C','S','SNR_comp','r_values','cnn_preds']:
            self.dataOut[key] = getattr(self.cnm.estimates,key)
        self.dataOut['C'] -= np.percentile(self.dataOut['C'],0.05,axis=1)[:,np.newaxis] # correct for offset

        # ### analyze active neurons: did they maintain same behavior? (correlation > 0.9)
        # #self.Cout[self.idxes['in'],:] = np.vstack([Ca/Ca.max() for Ca in self.cnm.estimates.C])

        # #t_start = time.time()
        # #self.dataOut['fitness'] = np.zeros(nC+nC_new)
        # #self.dataOut['fitness'][self.idxes['in']], erfc, sd_r, md = compute_event_exceptionality(self.dataOut['C'][self.idxes['in'],:])
        # #if nC_new > 0:
        # #self.dataOut['fitness'][nC:], erfc, sd_r, md = compute_event_exceptionality(self.dataOut['C'][nC:,:])
        # #self.fitness[self.fitness<-10**4] = -10**4

        # #t_end = time.time()
        # #print('fitness computed - time took: %5.3g'%(t_end-t_start))

    def others(self):

        print('check this stuff for analysis')

        # def plot_contour(ax,A,col,linewidth=1.):
        #     a = A.reshape(512,512).todense()
        #     a /= a.max()
        #     ax.contour(a,levels=[0.1,0.5],colors=col,linewidths=linewidth,linestyles=['--','-'])

        ### check, why some components are not matched
        # cols = ['r','g','b']
        # for i in np.where(new_comps)[0]:

        #     print('\n neuron ',i)

        #     why_no_match = np.where(p_same[:,i].todense()>0.1)[0]
        #     if len(why_no_match):

        #         print('why no match?',why_no_match)
        #         print(p_same[:,i].data)

        #         why_no_match = why_no_match[0]
        #         idx_confused = np.where(p_same[why_no_match,:].todense())[1]
        #         print(idx_confused)

        #         fig,ax = plt.subplots(1,2);
        #         plot_contour(ax[0],self.dataIn['A'][:,why_no_match],'k')
        #         if self.idxes['in']['active'][why_no_match]:
        #             ax[1].plot(Cin[why_no_match,:],color='k')
        #         for a,j in enumerate(idx_confused):
        #             plot_contour(ax[0],self.cnm.estimates.A[:,j],cols[a],linewidth=p_same[why_no_match,j])

        #             ax[1].plot(Cout[j,:],color=cols[a])

        #         plt.show(block=False)

        ## firingrate distributions
        # plt.figure()
        # plt.hist(ts.analysis['nu'][ts.idxes['out']['active_still']],np.linspace(0,1,51),facecolor='k',alpha=0.7)
        # plt.hist(ts.analysis['nu'][ts.idxes['out']['active_new']],np.linspace(0,1,51),facecolor='g',alpha=0.7)
        # plt.hist(ts.analysis['nu'][ts.idxes['out']['silent_still']],np.linspace(0,1,51),facecolor='r',alpha=0.3)
        # plt.show()

    def analyze_pre_plot(self,redo=False):

        nC = self.idxes['out']['active'].shape[0]
        # dims = self.dims

        ## remove non-zero offset
        # self.dataOut['C'] = self.dataOut['C'] - np.percentile(self.dataOut['C'],0.05,axis=1)[:,np.newaxis]

        # self.analysis = {}
        self.analysis['nu'] = np.full(nC,np.NaN)
        for c in tqdm.tqdm(range(nC),leave=False):
            if (self.dataOut['S'][c,:]>0).sum():
                spike_nr = (self.dataOut['S'][c,:]>0).sum()
                # spike_nr, md, sd_r = get_spikeNr(self.dataOut['S'][c,self.dataOut['S'][c,:]>0])
            else:
                spike_nr = 0
            self.analysis['nu'][c] = spike_nr/(8989/15)

        self.analysis['C_corr'] = np.zeros((self.dataOut['A'].shape[1],self.dataIn['A'].shape[1]))
        self.dataIn['C_std'] = self.dataIn['C'].std(1)
        self.dataOut['C_std'] = self.dataOut['C'].std(1)
        for c in tqdm.tqdm(np.where(self.idxes['out']['active_still'])[0],leave=False):

            # if self.idxes['out']['active_still'][c]:
            c_in = int(self.idxes['out']['match_to_in'][c])
            # print(c_in)
            # self.analysis['fp_corr'][c,c] = self.dataOut['A'][:,c].multiply(self.dataIn['A'][:,c]).sum()/(self.dataOut['A_norm'][c]*self.dataIn['A_norm'][c])
            self.analysis['C_corr'][c,c] = np.cov(self.dataIn['C'][c_in,:],self.dataOut['C'][c,:])[0,1]/(self.dataIn['C_std'][c_in]*self.dataOut['C_std'][c])

            idx_close = np.where(self.analysis['D_ROIs'][c,:]<12)[0]
            if len(idx_close)>0:
                for cc in idx_close:
                    # self.analysis['fp_corr'][c,cc] = self.dataOut['A'][:,c].multiply(self.dataOut['A'][:,cc]).sum()/(self.dataOut['A_norm'][c]*self.dataOut['A_norm'][cc])

                    if self.idxes['out']['active_still'][cc]:
                        cc_in = int(self.idxes['out']['match_to_in'][cc])
                        # print(cc_in)
                        # corr_tmp = np.cov(self.Cout[c,:],self.Cout[cc,:])[0,1]/(self.dataOut['C_std'][c]*self.dataOut['C_std'][cc])
                        self.analysis['C_corr'][c,cc] = np.cov(self.dataOut['C'][c,:],self.dataIn['C'][cc_in,:])[0,1]/(self.dataOut['C_std'][c]*self.dataIn['C_std'][cc_in])

        # self.analysis['fp_corr'].tocsc()

    def save_results(self):

        ### save everything important from running the detection

        fileParts = os.path.splitext(self.cluster.paths['CaImAn_results'][self.currentS])
        svPath = os.path.join(f'{fileParts[0]}_redetected{fileParts[1]}')

        results = {
            'A':self.dataOut['A'],
            'C':self.dataOut['C'],
            'S':self.dataOut['S'],
            'b':self.cnm.estimates.b,
            'f':self.cnm.estimates.f,
            'Cn':self.cnm.estimates.Cn,
            'dims': self.dims,
            'SNR_comp':self.dataOut['SNR_comp'],
            'r_values':self.dataOut['r_values'],
            'cnn_preds':self.dataOut['cnn_preds']
        }
        save_data(results,svPath)

        compares = {
            'Ain':self.dataIn['A'],
            'Cin':self.dataIn['C'],
            'idxes': self.idxes,
        }
        svPath = os.path.join(f'{fileParts[0]}_redetected_compares{fileParts[1]}')
        save_data(compares,svPath)
        # with open(svPath,'wb') as f_open:
        # pickle.dump(compares,f_open)

        ### analyze silent ones: do they show firing behavior at all? are they correlated with nearby active neurons?

    def load(self,s):

        pathData = os.path.join(self.currentSession,self.fileName_results)
        # pathData = pathcat([pathSession,'results_postSilent.%s'%ext])
        print('loading data from %s'%pathData)

        ld = load_data(pathData)

        self.dataOut = {'A':ld['A'],
                    'C':ld['C'],
                    'S':ld['S'],
                    'SNR':ld['SNR'],
                    'r_values':ld['r_values'],
                    'CNN':ld['CNN']}
        #'fitness':ld['fitness']}

        self.dataIn = {'A':ld['Ain'],
                   #'Cn':ld['Cn'],
                   'C':ld['Cin']}

        self.idxes = {'previous':ld['idx_previous'].astype('bool'),
                  'in':ld['idx_Ain'].astype('bool'),
                  'evaluate':ld['idx_evaluate'].astype('bool')}

        # self.data = {}

    def plot_analysis(self,SNR_thr=2,r_val_thr=0,CNN_thr=0.6):

        ext = 'png'

        idxs = [self.idxes['out']['active_still'],self.idxes['out']['active_new'],self.idxes['out']['silent_still'],self.idxes['out']['silent_new']]
        # col_arr = ['k','b','r','g']
        col_arr = ['k','g','r','b']

        ### plot changes from input to output
        acorr = np.diag(self.analysis['C_corr'])
        fp_acorr = np.diag(self.analysis['fp_corr'][1,...])

        al=0.5

        plt.figure(figsize=(4,3))
        ax = plt.axes([0.1,0.2,0.25,0.4])
        for i in range(2):
            ax.hist(acorr[idxs[i]],np.linspace(0.5,1,21),facecolor=col_arr[i],alpha=al,orientation='horizontal')
        # ax.xlabel('$C_{Ca^{2+}}^{in-out}$',fontsize=14)
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top','left']].set_visible(False)

        ax = plt.axes([0.4,0.65,0.35,0.3])
        for i in range(2):
            # if i==1:
            # continue
            plt.hist(fp_acorr[idxs[i]],np.linspace(0.5,1,21),facecolor=col_arr[i],alpha=al)
        # plt.xlabel('$C_{fp}^{in-out}$',fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top','left']].set_visible(False)
        ax.legend(handles=[
            mppatches.Patch(color='k',label=f'active (still)'),
            mppatches.Patch(color='g',label=f'active (new)'),
        ],loc='upper left',bbox_to_anchor=[-0.3,1.1])

        ax = plt.axes([0.4,0.2,0.35,0.4])
        for i in range(2):
            ax.scatter(fp_acorr[idxs[i]],acorr[idxs[i]],s=2,c=col_arr[i])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlabel('$C_{fp}$',fontsize=14)
        ax.set_ylabel('$C_{Ca^{2+}}$',fontsize=14)
        ax.set_yticks(np.linspace(0.5,1,3))
        ax.set_xlim([0.5,1])
        ax.set_ylim([0.5,1])

        plt.tight_layout()
        plt.show(block=False)
        pathFigure = os.path.join(self.currentSession,f'find_silent_similarity.{ext}')
        plt.savefig(pathFigure,format=ext,dpi=300)
        print(f'Figure saved as {pathFigure}')

        ### plot goodness of detected ROIs
        plt.figure(figsize=(4,3))

        ax = plt.axes([0.15,0.65,0.4,0.25])
        for i in range(4):
            ax.hist(self.dataOut['r_values'][idxs[i]],np.linspace(-1,1,21),facecolor=col_arr[i],alpha=al)
        # plt.xscale('log')
        ax.set_xlim([-0.5,1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.set_xlabel('r value',fontsize=14)

        ax = plt.axes([0.65,0.2,0.25,0.4])
        for i in range(4):
            ax.hist(self.dataOut['SNR_comp'][idxs[i]],np.linspace(0,30,21),facecolor=col_arr[i],alpha=al,orientation='horizontal')
        # ax.invert_xaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.yaxis.tick_right()
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.yaxis.set_label_position("right")
        # ax.set_ylabel('SNR',fontsize=14)
        ax.set_ylim([0,20])
        ax.set_xlabel('occurence',fontsize=14)
        ax.legend(handles=[
            mppatches.Patch(color='k',label=f'active (still)'),
            mppatches.Patch(color='g',label=f'active (new)'),
            mppatches.Patch(color='r',label=f'silent (still)'),
            mppatches.Patch(color='b',label=f'silent (new)'),
        ],loc='upper left',bbox_to_anchor=[-0.15,2.])

        ax = plt.axes([0.15,0.2,0.4,0.4])
        for i in range(4):
            ax.plot(self.dataOut['r_values'][idxs[i]],self.dataOut['SNR_comp'][idxs[i]],'o',color=col_arr[i],markersize=1)
        ax.set_xlabel('r value',fontsize=14)
        ax.yaxis.tick_right()
        ax.set_ylabel('SNR',fontsize=14)
        ax.set_xlim([-0.5,1])
        ax.set_ylim([0,20])
        ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # ax.set_yticks([])

        plt.tight_layout()
        plt.show(block=False)

        pathFigure = os.path.join(self.currentSession,f'find_silent_evaluation.{ext}');
        plt.savefig(pathFigure,format=ext,dpi=300)
        print(f'Figure saved as {pathFigure}')

    def plot_traces(self,idxs=None,n=3,ext='png'):

        ## some nice plots:
        ###   1. active vs silent ROI shapes
        ###   2. correlation vs activity

        plt.figure()
        if idxs is None:
            idxs = np.where(self.idxes['out']['active_still'])[0]
        for i in range(n):

            c = idxs[np.random.randint(len(idxs))]

            c_in = int(self.idxes['out']['match_to_in'][c])
            print(c)
            plt.subplot(2,n,i+1)
            plt.imshow(self.dataIn['A'][:,c_in].reshape(512,512).toarray(),origin='lower')
            plt.xlabel('x [mum]')
            plt.ylabel('y [mum]')
            plt.title('Ca-correlation: %5.3f'%self.analysis['C_corr'][c,c])

            plt.subplot(4,n,2*n+i+1)
            plt.plot(self.dataOut['C'][c,:],linewidth=1)
            plt.plot(self.dataOut['S'][c,:],linewidth=0.5)
            if i==0:
                plt.ylabel('$C_{out}$')

            plt.subplot(4,n,3*n+i+1)
            plt.plot(self.dataIn['C'][c_in,:],linewidth=1)
            # plt.plot(self.dataIn['S'][c,:])
            if i==0:
                plt.ylabel('$C_{in}$')

        # plt.subplot(3,2,6)
        # plt.plot(self.cnm.estimates.S[c,:])
        # plt.ylabel('$S_{out}$')

        # plt.subplot(315)
        # plt.plot(Cout[c,:])
        # if not self.idxes['in']['silent'][c]:
        # plt.plot(Cin[c,:])
        plt.tight_layout()
        pathFigure = os.path.join(self.currentSession,f'find_silent_examples.{ext}');
        plt.savefig(pathFigure,format=ext,dpi=300)
        print('Figure saved as %s'%pathFigure)

        plt.show(block=False)

    def plot_detection(self):

        ext = 'png'
        plt.figure(figsize=(10,4))
        ax = plt.subplot(121)
        ax.imshow(self.dataOut['A'].sum(1).reshape(self.dims),origin='lower')
        # ax.imshow(self.cnm.estimates.Cn.T,origin='lower')

        [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=0.5) for a in self.dataIn['A'][:,self.idxes['in']['active']].T]
        [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=0.5) for a in self.dataIn['A'][:,self.idxes['in']['silent']].T]
        # plt.tight_layout()
        # plt.show(block=False)

        # pathFigure = os.path.join(self.currentSession,'find_silent_only_before.%s'%(ext));
        # plt.savefig(pathFigure,format=ext,dpi=300)
        # print('Figure saved as %s'%pathFigure)

        # plt.figure(figsize=(5,4))
        ax = plt.subplot(122)
        ax.imshow(self.dataOut['A'].sum(1).reshape(self.dims),origin='lower')
        # ax.imshow(self.cnm.estimates.Cn.T,origin='lower')

        [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['out']['active_still']].T]
        [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=1.) for a in self.dataOut['A'][:,self.idxes['out']['active_new']].T]
        [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='tab:orange', linewidths=1.) for a in self.dataOut['A'][:,self.idxes['out']['active_unmatched']].T]

        silents = np.copy(self.idxes['in']['silent'])
        silents[self.idxes['out']['match_to_in'][self.idxes['out']['active_still'] | self.idxes['out']['active_new']].astype('int')] = False
        [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=0.5) for a in self.dataIn['A'][:,silents].T]

        # [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['out']['silent_still']].T]
        # [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=1.) for a in self.dataOut['A'][:,self.idxes['out']['silent_new']].T]

        # [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=1) for a in self.dataIn['A'][:,self.idxes['in']['silent']].T]
        ax.legend(handles=[
            mplines.Line2D([0],[0],color='k',linestyle='-',linewidth=0.5,label=f"active (still) ({self.idxes['out']['active_still'].sum()})"),
            mplines.Line2D([0],[0],color='k',linestyle='-',linewidth=1,label=f"active (new) ({self.idxes['out']['active_new'].sum()})"),
            mplines.Line2D([0],[0],color='tab:orange',linestyle='-',linewidth=1,label=f"active (unmatched) ({self.idxes['out']['active_unmatched'].sum()})"),
            mplines.Line2D([0],[0],color='r',linestyle='-',linewidth=0.5,label=f"silent (still) ({self.idxes['out']['silent_still'].sum()})"),
        ],loc='upper left',bbox_to_anchor=[-0.4,0.1],framealpha=0.9)

        plt.tight_layout()
        plt.show(block=False)
        pathFigure = os.path.join(self.currentSession,'find_silent_after.%s'%(ext));
        plt.savefig(pathFigure,format=ext,dpi=300)
        print('Figure saved as %s'%pathFigure)


class plot_test_undetected:

  def __init__(self,basePath,mouse):

    dataSet='redetect'  ## nothing else makes sense

    self.pathMouse = os.path.join(basePath,mouse)
    self.pathMatching = os.path.join(self.pathMouse,'matching/Sheintuch_registration_results_%s.pkl'%dataSet)


  def load_eval(self,ext='mat'):
      ld_dat = pickleData([],self.pathMatching,'load')

      self.eval_data = {}
      self.progress = tqdm(range(self.nSes))
      # self.matches = {}
      for s in self.progress:
          ## load results
          pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
          pathData = os.path.join(pathSession,'results_redetect.%s'%ext)
          self.progress.set_description('Now processing Session %d'%(s+1))

          ld = load_data(pathData)

          self.eval_data[s] = {}#   {'SNR':       np.zeros((self.nC,self.nSes))*np.NaN,
                                #'CNN':       np.zeros((self.nC,self.nSes))*np.NaN,
                                #'r_values':  np.zeros((self.nC,self.nSes))*np.NaN}
          for key in ['SNR','r_values','CNN','idx_previous']:
              self.eval_data[s][key] = ld[key]


  def load(self,ext='mat'):

    ### load matching results
    ld_dat = load_data(self.pathMatching)
    # ld_dat = pickleData([],self.pathMatching,'load')
    try:
      assignments = ld_dat['assignments']
    except:
      assignments = ld_dat['assignment']
    self.nC,self.nSes = assignments.shape

    ### preallocate arrays
    self.data = {'eval':        {'SNR':       np.zeros((self.nC,self.nSes))*np.NaN,
                                 'CNN':       np.zeros((self.nC,self.nSes))*np.NaN,
                                 'r_values':  np.zeros((self.nC,self.nSes))*np.NaN},
                 'nonMatched':{'SNR':           np.zeros((10000,self.nSes))*np.NaN,
                                'CNN':          np.zeros((10000,self.nSes))*np.NaN,
                                'r_values':     np.zeros((10000,self.nSes))*np.NaN},
                 'fp_corr':   np.zeros((self.nC,self.nSes))*np.NaN,
                 'C_corr':    np.zeros((self.nC,self.nSes))*np.NaN,
                 'idxes':       {'previous':  np.zeros((self.nC,self.nSes),'bool'),
                                 'in':        np.zeros((self.nC,self.nSes),'bool'),
                                 'detected':  np.zeros((self.nC,self.nSes),'bool')}
                 }

    ### for all s, store eval and idxes and calculate A-A and C-C correlation
    self.progress = tqdm(range(self.nSes))
    self.matches = {}
    for s in self.progress:

      ## load results
      pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
      pathData = os.path.join(pathSession,'results_redetect.%s'%ext)
      self.progress.set_description('Now processing Session %d'%(s+1))

      ld = load_data(pathData)

      idx_c = np.where(~np.isnan(assignments[:,s]))[0]
      n_arr = assignments[idx_c,s].astype('int')

      ## store eval
      non_n_arr = np.zeros(10000,'bool')
      non_n_arr[:len(ld['SNR'])] = True
      non_n_arr[n_arr] = False
      for key in ['SNR','r_values','CNN']:
        self.analysis['eval'][key][idx_c,s] = ld[key][n_arr]
        self.analysis['nonMatched'][key][non_n_arr,s] = ld[key][np.where(non_n_arr)]

      # print(np.isfinite(self.analysis['nonMatched']['SNR']).sum())
      ## store indices
      self.analysis['idxes']['detected'][idx_c,s] = True
      self.analysis['idxes']['previous'][idx_c,s] = ld['idx_previous'][n_arr]
      self.analysis['idxes']['in'][idx_c,s] = ld['idx_Ain'][n_arr]

      dims = (512,512)
      ## find next footprints with low distance
      cm_in = com(ld['Ain'],dims[0],dims[1])
      cm_out = com(ld['A'],dims[0],dims[1])
      #return ld['A'], ld['Ain']

      nC_in = ld['Ain'].shape[1]
      nC_out = ld['A'].shape[1]

      D_ROIs = scipy.spatial.distance.cdist(cm_in,cm_out)
      D_ROI_mask = np.ma.array(D_ROIs, mask = np.isnan(D_ROIs))

      #idx_out_bool = np.zeros(nC_out,'bool')
      #idx_in_bool = np.zeros(nC_in,'bool')

      self.matches[s] = np.zeros(cm_out.shape[0])*np.NaN
      idx_new = []
      nC_new = 0
      dc = 0              ## difference in index
      for c in np.where((ld['A']>0).sum(0)>50)[1]:
        idx_close = np.where(D_ROI_mask[:,c]<5)[0]
        if len(idx_close):
            d_idx = np.min(np.abs(idx_close-c))
            closeby = idx_close[np.argmin(np.abs(idx_close-c))]
            if np.isscalar(closeby) & (d_idx <= dc):
                self.matches[s][c] = closeby
            elif np.isscalar(closeby) & (d_idx < (dc+10)):
                dc = d_idx
                self.matches[s][c] = closeby

      C_std = ld['C'].std(1)
      Cin_std = ld['Cin'].std(1)

      chunks = 1000
      idx_Ain = np.where((ld['Ain']>0).sum(0)>50)[1]#np.where(self.analysis['idxes']['in'][:,s])[0]
      Ain_norm = np.zeros(ld['Ain'].shape[1])*np.NaN
      nC_in = len(idx_Ain)
      steps = int(nC_in/chunks)
      for i in tqdm(range(steps+(np.mod(nC_in,chunks)>0)),leave=False):
        c_start = i*chunks
        c_end = min(nC_in,(i+1)*chunks)
        Ain_norm[idx_Ain[c_start:c_end]] = np.linalg.norm(ld['Ain'][:,idx_Ain[c_start:c_end]].toarray(),axis=0)

      idx_A = np.where((ld['A']>0).sum(0)>50)[1]#np.where(self.analysis['idxes']['in'][:,s])[0]
      A_norm = np.zeros(ld['A'].shape[1])*np.NaN
      nC_out = len(idx_A)
      steps = int(nC_out/chunks)
      for i in tqdm(range(steps+(np.mod(nC_out,chunks)>0)),leave=False):
        c_start = i*chunks
        c_end = min(nC_out,(i+1)*chunks)
        A_norm[idx_A[c_start:c_end]] = np.linalg.norm(ld['A'][:,idx_A[c_start:c_end]].toarray(),axis=0)

      ## calculate correlation between input and output
      for (n2,n1) in tqdm(enumerate(self.matches[s]),total=len(self.matches[s]),leave=False):
        if ~np.isnan(n1):
            c = np.where(assignments[:,s]==n2)[0]
            n1 = int(n1)
            self.analysis['fp_corr'][c,s] = ld['A'][:,n2].multiply(ld['Ain'][:,n1]).sum()/(A_norm[n2]*Ain_norm[n1])
            self.analysis['C_corr'][c,s] = np.cov(ld['C'][n2,:],ld['Cin'][n1,:])[0,1]/(C_std[n2]*Cin_std[n1])


  def get_cluster(self,basePath,mouse,nSes,dataSet='redetect',n_processes=0,s_corr_min=0.3):

    self.cluster = cluster(basePath,mouse,nSes,dataSet=dataSet,s_corr_min=s_corr_min)
    self.cluster.load()

    # self.cluster.process_sessions(n_processes=n_processes,reprocess=True)
    # self.cluster.get_IDs()
    # self.cluster.get_stats(n_processes=n_processes,complete=False)

  def plot(self,SNR_thr=2,rval_thr=0,CNN_thr=0.6,pm_thr=0.5,sv=False):

    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    dims = (512,512)
    idx = self.analysis['idxes']['previous']

    idx_active = (self.analysis['eval']['SNR']>SNR_thr) & (self.analysis['eval']['r_values']>rval_thr) & (self.analysis['eval']['CNN']>CNN_thr)
    idx_silent = ~idx_active

    idxes = {}
    idxes['active_good'] = idx & idx_active
    idxes['active_bad'] = idx & idx_silent

    idxes['silent_good'] = ~idx & idx_active
    idxes['silent_bad'] = ~idx & idx_silent

    pathMatching = os.path.join(self.pathMouse,'matching/Sheintuch_registration_results_OnACID.pkl')
    dt = pickleData([],pathMatching,'load')
    assignments_OnACID = dt['assignments']

    pathMatching = os.path.join(self.pathMouse,'matching/Sheintuch_registration_results_redetect.pkl')
    ld_dat = pickleData([],pathMatching,'load')
    assignments = ld_dat['assignments']

    self.cluster.meta['SNR_thr'] = SNR_thr
    self.cluster.meta['rval_thr'] = rval_thr
    self.cluster.meta['CNN_thr'] = CNN_thr
    self.cluster.cluster_classification()

    idx_pm = (self.cluster.stats['match_score'][...,0]>0.5)# ((self.cluster.stats['match_score'][...,0]-self.cluster.stats['match_score'][...,1])>pm_thr) | (self.cluster.stats['match_score'][...,0]>0.95)

    if False:
        fig = plt.figure(figsize=(7,4),dpi=300)

        ax_ROIs = plt.axes([0.05,0.625,0.2,0.325])
        add_number(fig,ax_ROIs,order=1,offset=[-25,25])
        ax_ROIs2 = plt.axes([0.3,0.625,0.2,0.325])
        add_number(fig,ax_ROIs2,order=2,offset=[-25,25])

        s = 10

        n_arr = assignments_OnACID[np.where(~np.isnan(assignments_OnACID[:,s-1]))[0],s-1].astype('int')

        pathSession = os.path.join(self.pathMouse,'Session%02d'%(s))
        pathData = os.path.join(pathSession,'results_redetect.mat')
        ld = load_data(pathData)

        cm_in = com(ld['Ain'],dims[0],dims[1])
        cm_out = com(ld['A'],dims[0],dims[1])

        idx_A = ld['idx_previous'][:ld['Ain'].shape[1]].astype('bool')
        #idx_A = np.zeros(ld['Ain'].shape[1],'bool')
        #idx_A[n_arr] = True

        bounds = [[175,225],[175,225]]
        self.plot_detection(ax_ROIs,ld['Ain'][:,idx_A],ld['Ain'][:,~idx_A],bounds=bounds)
        ax_ROIs.set_xlim(bounds[0])
        ax_ROIs.set_ylim(bounds[1])
        ax_ROIs.set_xticks([])
        ax_ROIs.set_yticks([])
        sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        ax_ROIs.add_artist(sbar)

        idx_A = ld['idx_previous'].astype('bool')# & idx_n
        # print(idx_A.shape)
        #idx_A = np.zeros(ld['A'].shape[1],'bool')
        #idx_A[n_arr] = True
        # print(ld['A'].shape)
        nC = self.analysis['eval']['SNR'].shape[0]
        nN = ld['A'].shape[1]
        c_arr = np.where(assignments[:,s-1]>=0)[0]
        n_arr = assignments[c_arr,s-1].astype('int')
        idx_n = np.in1d(range(nN),n_arr)
        idx_c = np.in1d(range(nC),c_arr)
        idx_mapping_cn = -np.ones(nC,'int')
        idx_mapping_cn[c_arr] = n_arr
        idx_mapping_nc = -np.ones(nN,'int')
        idx_mapping_nc[n_arr] = c_arr

        idx_active_n = np.in1d(range(nN),idx_mapping_cn[np.squeeze(idx_active[:,s-1])])

        # CM = com(ld['A'],512,512)
        # idx_bounds = (CM[:,0] > bounds[0][0]) & (CM[:,0] < bounds[0][1]) & (CM[:,1] > bounds[1][0]) & (CM[:,1] < bounds[1][1])

        self.plot_detection(ax_ROIs2,ld['A'][:,idx_A],ld['A'][:,~idx_A],bounds=bounds,lw=0.5,ls='dotted')
        self.plot_detection(ax_ROIs2,ld['A'][:,idx_A & idx_active_n],ld['A'][:,(~idx_A) & idx_active_n],bounds=bounds)

        ax_ROIs2.set_xlim(bounds[0])
        ax_ROIs2.set_ylim(bounds[1])
        ax_ROIs2.set_xticks([])
        ax_ROIs2.set_yticks([])

        # return
        #ax1 = ax.twinx()
        #ax2 = ax.twiny()
        #ax1.hist(self.analysis['fp_corr'][idx],np.linspace(0,1,101),facecolor='k',alpha=0.5)
        #ax1.hist(self.analysis['fp_corr'][~idx],np.linspace(0,1,101),facecolor='r',alpha=0.5)
        #ax1.set_yticks([])
        #ax1.invert_yaxis()

        #ax2.hist(self.analysis['C_corr'][idx],np.linspace(0,1,101),facecolor='k',alpha=0.5,orientation='horizontal')
        #ax2.hist(self.analysis['C_corr'][~idx],np.linspace(0,1,101),facecolor='r',alpha=0.5,orientation='horizontal')
        #ax2.set_xticks([])

        #plt.figure(figsize=(6,2.5))
        ax = plt.axes([0.55,0.65,0.17,0.3])
        add_number(fig,ax,order=5,offset=[-25,25])

        ax.plot(self.analysis['fp_corr'][idx].flat,self.analysis['C_corr'][idx].flat,'ko',markersize=0.5,markeredgewidth=0,alpha=0.5)
        ax.plot(self.analysis['fp_corr'][~idx].flat,self.analysis['C_corr'][~idx].flat,'o',color='tab:orange',markersize=0.5,markeredgewidth=0,alpha=0.5)
        ax.plot(-1,np.NaN,'ko',markersize=5,markeredgewidth=0,alpha=0.5,label='initial')
        ax.plot(-1,np.NaN,'o',color='tab:orange',markersize=5,markeredgewidth=0,alpha=0.5,label='inferred')

        ax.set_xlabel('$c_{fp}$')
        ax.set_ylabel('$c_{Ca}$')
        ax.set_xlim([-0.1,1])
        ax.tick_params(axis='y',which='both',left=False,right=True,labelright=True,labelleft=False)
        ax.yaxis.set_label_position("right")
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([-0.1,1])
        ax.set_yticks(np.linspace(0,1,3))

        ax.legend(loc='upper left',bbox_to_anchor=[-0.1,0.95],fontsize=8)

        SNR = []
        SNR_not = []
        r_val = []
        r_val_not = []
        for s in range(self.nSes):
            SNR.extend(self.eval_data[s]['SNR'][self.eval_data[s]['idx_previous'].astype('bool')])
            SNR_not.extend(self.eval_data[s]['SNR'][~self.eval_data[s]['idx_previous'].astype('bool')])
            r_val.extend(self.eval_data[s]['r_values'][self.eval_data[s]['idx_previous'].astype('bool')])
            r_val_not.extend(self.eval_data[s]['r_values'][~self.eval_data[s]['idx_previous'].astype('bool')])
        # r_val_not.extend(self.analysis['nonMatched']['r_values'].flat)
        # r_val_not.extend(self.analysis['eval']['r_values'][~idx])
        # SNR_not.extend(self.analysis['nonMatched']['SNR'].flat)
        # SNR_not.extend(self.analysis['eval']['SNR'][~idx])
        # print(SNR)
        # print(len(SNR))

        ax = plt.axes([0.55,0.125,0.17,0.3])
        add_number(fig,ax,order=8,offset=[-25,25])
        ax.plot(r_val_not,SNR_not,'o',color='tab:orange',markersize=0.25,markeredgewidth=0,alpha=0.5)
        ax.plot(r_val,SNR,'ko',markersize=0.25,markeredgewidth=0,alpha=0.5)
        # ax.plot(self.analysis['eval']['r_values'][idx],self.analysis['eval']['SNR'][idx],'ko',markersize=0.25,markeredgewidth=0,alpha=0.5)
        # ax.plot(self.analysis['nonMatched']['r_values'],self.analysis['nonMatched']['SNR'],'o',color='tab:orange',markersize=0.25,markeredgewidth=0,alpha=0.5)
        # ax.plot(self.analysis['eval']['r_values'][~idx],self.analysis['eval']['SNR'][~idx],'o',color='tab:orange',markersize=0.25,markeredgewidth=0,alpha=0.5)
        ax.plot([-1,1],[3,3],'k--',linewidth=0.5)
        ax.plot([0.5,0.5],[1,40],'k--',linewidth=0.5)

        ax.set_yscale('log')
        ax.set_ylabel('SNR')
        ax.set_xlim([-0.5,1])
        ax.set_ylim([1,40])
        ax.set_xlabel('r-value')
        ax.tick_params(axis='y',which='both',left=False,right=True,labelright=True,labelleft=False)
        ax.yaxis.set_label_position("right")
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax = plt.axes([0.85,0.85,0.125,0.1])
        add_number(fig,ax,order=6,offset=[-50,25])
        ax.hist(self.analysis['fp_corr'][idx],np.linspace(0.5,1,21),density=True,facecolor='k',alpha=0.8)
        ax.hist(self.analysis['fp_corr'][~idx],np.linspace(0.5,1,21),density=True,facecolor='tab:orange',alpha=0.8)
        ax.set_xlabel('$c_{fp}$')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.85,0.625,0.125,0.1])
        add_number(fig,ax,order=7,offset=[-50,25])
        ax.hist(self.analysis['C_corr'][idx],np.linspace(-0.2,1,25),density=True,facecolor='k',alpha=0.8)
        ax.hist(self.analysis['C_corr'][~idx],np.linspace(-0.2,1,25),density=True,facecolor='tab:orange',alpha=0.8)
        ax.set_xlabel('$c_{Ca}$')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.85,0.35,0.125,0.1])
        add_number(fig,ax,order=9,offset=[-50,25])
        # ax.hist(self.analysis['eval']['r_values'][idx],np.linspace(-0.5,1,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(r_val,np.linspace(-0.5,1,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(r_val_not,np.linspace(-0.5,1,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.analysis['eval']['r_values'][~idx],np.linspace(-0.5,1,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.analysis['nonMatched']['r_values'].flat,np.linspace(-0.5,1,31),density=True,facecolor='tab:orange',alpha=0.8)
        ax.plot([0.5,0.5],[0,ax.get_ylim()[1]],'k--')
        ax.set_xlabel('r-value')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.85,0.125,0.125,0.1])
        add_number(fig,ax,order=10,offset=[-50,25])
        # ax.hist(self.analysis['eval']['SNR'][idx],np.linspace(0,30,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(SNR,np.linspace(0,30,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(SNR_not,np.linspace(0,30,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.analysis['eval']['SNR'][~idx],np.linspace(0,30,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.analysis['nonMatched']['SNR'].flat,np.linspace(0,30,31),density=True,facecolor='tab:orange',alpha=0.8)
        ax.plot([3,3],[0,ax.get_ylim()[1]],'k--')
        ax.set_xlabel('SNR')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ax1.hist(self.analysis['eval']['r_values'][idx],np.linspace(-0.5,1,151),facecolor='k',alpha=0.5)
        # ax1.hist(self.analysis['eval']['r_values'][~idx],np.linspace(-0.5,1,151),facecolor='r',alpha=0.5)
        # ax1.set_yticks([])
        # ax1.invert_yaxis()
        #
        # ax2 = ax.twiny()
        # ax2.hist(self.analysis['eval']['SNR'][idx],np.linspace(0,40,101),facecolor='k',alpha=0.5,orientation='horizontal')
        # ax2.hist(self.analysis['eval']['SNR'][~idx],np.linspace(0,40,101),facecolor='r',alpha=0.5,orientation='horizontal')
        # ax2.set_xticks([])


        # plt.tight_layout()
        # plt.show(block=False)

        #ext = 'png'
        #path = pathcat([self.pathMouse,'complete_set_evaluation.%s'%ext])
        #plt.savefig(path,format=ext,dpi=300)

        ax_A = plt.axes([0.05,0.375,0.125,0.2])
        add_number(fig,ax_A,order=3,offset=[-25,25])
        ax_Ca = plt.axes([0.2,0.375,0.275,0.2])

        idx_active = np.where((self.analysis['C_corr']>0.8) & idx)
        i = np.random.randint(len(idx_silent[0]))
        c = idx_active[0][i]
        s = idx_active[1][i]
        s,c,n = (7,0,0)
        # ld_dat = pickleData([],self.pathMatching,'load')
        # assignments = ld_dat['assignments']
        n = int(assignments[c,s])
        #plt.draw()
        print(s,c,n)    ## (73,1441,2406)

        pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
        pathData = os.path.join(pathSession,'results_redetect.mat')
        ld = load_data(pathData)
        # ld = loadmat(pathData,variable_names=['Cin','C','Ain','A'],squeeze_me=True)

        dims=(512,512)
        ax_A.imshow(ld['A'].sum(1).reshape(dims))

        Cin = ld['Cin']/ld['Cin'].max(1)[:,np.newaxis]
        C = ld['C']/ld['C'].max(1)[:,np.newaxis]

        cm = com(ld['A'],dims[0],dims[1])

        # calculate CoMs and distance
        D_ROIs = scipy.spatial.distance.cdist(cm,cm)
        n_close = np.where(D_ROIs[n,:]<10)[0]
        print(n_close)
        A_norm = np.linalg.norm(ld['A'].toarray(),axis=0)

        # plot all closeby ones, highlight overlapping ones
        t_arr = np.linspace(0,8989/15,8989)
        ax_Ca.plot(t_arr,Cin[n,:],'k',linewidth=0.5)
        ax_Ca.plot(t_arr,C[n,:]+1,'tab:green',linewidth=0.5)
        #ax_Ca.text(6000/15,1.5,'corr: %5.3g, SNR: %5.3g'%(self.analysis['fp_corr'][c,s],self.analysis['eval']['SNR'][c,s]))

        ax_A.contour((ld['Ain'][:,n]/ld['Ain'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='w', linewidths=2)
        ax_A.contour((ld['A'][:,n]/ld['A'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:green', linewidths=2)

        i = 0
        for nn in n_close:
          A_corr = ld['A'][:,n].multiply(ld['A'][:,nn]).sum()/(A_norm[n]*A_norm[nn])
          if (A_corr>0.3) and not (n==nn):
            ax_A.contour((ld['A'][:,nn]/ld['A'][:,nn].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:orange', linewidths=1.5)

            # cc = np.where(assignments[:,s] == nn)[0][0]
            #print(c)
            #print(self.analysis['corr'][cc,s])
            #ax_Ca.text(6000,i+2.5,'corr: %5.3g, SNR: %5.3g'%(A_corr,self.analysis['eval']['SNR'][cc,s]))
            ax_Ca.plot(t_arr,C[nn,:]+i+2,'tab:orange',linewidth=0.5)
            i+=1

        ax_A.set_xlim([cm[n,0]-15,cm[n,0]+15])
        ax_A.set_ylim([cm[n,1]-15,cm[n,1]+15])
        ax_A.set_xticks([])
        ax_A.set_yticks([])
        ax_A.axis('off')
        #ax_A.text(cm[n,0],cm[n,1]-14,'$c_{fp}=%4.2g$\n$c_{Ca}=%4.2g$'%(self.analysis['fp_corr'][c,s],self.analysis['C_corr'][c,s]),color='w',fontsize=12)
        ax_Ca.spines['right'].set_visible(False)
        ax_Ca.spines['top'].set_visible(False)
        ax_Ca.set_xticks([])
        ax_Ca.set_yticks([])
        #ax_Ca.set_xlabel('time [s]')

        #plt.tight_layout()
        #plt.show(block=False)
        #ext = 'png'
        #path = pathcat([self.pathMouse,'complete_set_evaluation_example_good.%s'%ext])
        #plt.savefig(path,format=ext,dpi=300)

        ax_A = plt.axes([0.05,0.125,0.125,0.2])
        add_number(fig,ax_A,order=4,offset=[-25,25])
        ax_Ca = plt.axes([0.2,0.125,0.275,0.2])

        idx_silent = np.where((self.analysis['C_corr']<0.5) & (self.analysis['fp_corr']<0.8) & idx)
        i = np.random.randint(len(idx_silent[0]))
        c = idx_silent[0][i]
        s = idx_silent[1][i]
        s,c,n = (5,2099,1853)
        # ld_dat = pickleData([],self.pathMatching,'load')
        # assignments = ld_dat['assignments']
        n = int(assignments[c,s])
        #plt.draw()
        print(s,c,n)    ## (73,1441,2406)

        pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
        pathData = os.path.join(pathSession,'results_redetect.mat')
        ld = load_data(pathData)
        # ld = loadmat(pathData,variable_names=['Cin','C','Ain','A'],squeeze_me=True)

        #plt.figure(figsize=(6,2.5))
        #dims=(512,512)
        #ax_A = plt.axes([0.1,0.2,0.35,0.75])
        ax_A.imshow(ld['A'].sum(1).reshape(dims))

        Cin = ld['Cin']/ld['Cin'].max(1)[:,np.newaxis]
        C = ld['C']/ld['C'].max(1)[:,np.newaxis]

        cm = com(ld['A'],dims[0],dims[1])

        # calculate CoMs and distance
        D_ROIs = scipy.spatial.distance.cdist(cm,cm)
        n_close = np.where(D_ROIs[n,:]<10)[0]
        A_norm = np.linalg.norm(ld['A'].toarray(),axis=0)

        # plot all closeby ones, highlight overlapping ones
        t_arr = np.linspace(0,8989/15,8989)
        ax_Ca.plot(t_arr,Cin[n,:],'k',linewidth=0.5)
        ax_Ca.plot(t_arr,C[n,:]+1,'tab:green',linewidth=0.5)
        #ax_Ca.text(6000/15,1.5,'corr: %5.3g, SNR: %5.3g'%(self.analysis['fp_corr'][c,s],self.analysis['eval']['SNR'][c,s]))

        ax_A.contour((ld['Ain'][:,n]/ld['Ain'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='w', linewidths=2)
        ax_A.contour((ld['A'][:,n]/ld['A'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:green', linewidths=2)

        i = 0
        for nn in n_close:
          A_corr = ld['A'][:,n].multiply(ld['A'][:,nn]).sum()/(A_norm[n]*A_norm[nn])
          if (A_corr>0.3) and not (n==nn):
            ax_A.contour((ld['A'][:,nn]/ld['A'][:,nn].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:orange', linewidths=1.5)

            # cc = np.where(assignments[:,s] == nn)[0][0]
            #print(c)
            #print(self.analysis['corr'][cc,s])
            #ax_Ca.text(6000,i+2.5,'corr: %5.3g, SNR: %5.3g'%(A_corr,self.analysis['eval']['SNR'][cc,s]))
            ax_Ca.plot(t_arr,C[nn,:]+i+2,'tab:orange',linewidth=0.5)
            i+=1

        ax_A.set_xlim([cm[n,0]-15,cm[n,0]+15])
        ax_A.set_ylim([cm[n,1]-15,cm[n,1]+15])
        ax_A.set_xticks([])
        ax_A.set_yticks([])
        ax_A.axis('off')
        #ax_A.text(cm[n,0],cm[n,1]-14,'$c_{fp}=%4.2g$\n$c_{Ca}=%4.2g$'%(self.analysis['fp_corr'][c,s],self.analysis['C_corr'][c,s]),color='w',fontsize=12)
        ax_Ca.spines['right'].set_visible(False)
        ax_Ca.spines['top'].set_visible(False)
        ax_Ca.set_yticks([])
        ax_Ca.set_xlabel('time [s]')

        if sv:
            ext = 'png'
            path = pathcat([self.pathMouse,'Figures/complete_set_evaluation1.%s'%ext])
            plt.savefig(path,format=ext,dpi=300)

    if True:
        plt.figure(figsize=(3.5,4),dpi=300)
        dataSet = 'redetect'
        pathLoad = pathcat([self.pathMouse,'clusterStats_%s.pkl'%dataSet])
        ld = pickleData([],pathLoad,'load')
        idxes = (ld['SNR']>SNR_thr) & (ld['r_values']>rval_thr) & (ld['CNN']>CNN_thr) & (ld['firingrate']>0)
        idx_prev = self.analysis['idxes']['previous']

        colors = [(1,0,0,0),mcolors.to_rgba('tab:orange')]
        RedAlpha = mcolors.LinearSegmentedColormap.from_list('RedAlpha',colors,N=2)
        colors = [(0,0,0,0),(0,0,0,1)]
        BlackAlpha = mcolors.LinearSegmentedColormap.from_list('BlackAlpha',colors,N=2)
        colors = [(0,0,0,0),mcolors.to_rgba('tab:green')]
        GreenAlpha = mcolors.LinearSegmentedColormap.from_list('GreenAlpha',colors,N=2)

        nC,nS = assignments.shape
        ax_oc = plt.axes([0.2,0.1,0.45,0.55])
        ax_oc2 = ax_oc.twinx()
        # ax_oc.imshow((~np.isnan(assignments))&idxes,cmap=BlackAlpha,aspect='auto')
        # ax_oc2.imshow((~np.isnan(assignments))&(~idxes),cmap=RedAlpha,aspect='auto')
        ax_oc.imshow((idx_pm&idxes)[self.cluster.stats['cluster_bool'],:],cmap=BlackAlpha,aspect='auto')
        ax_oc.imshow((idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:],cmap=GreenAlpha,aspect='auto')
        ax_oc2.imshow((idx_pm&(~idxes))[self.cluster.stats['cluster_bool'],:],cmap=RedAlpha,aspect='auto')
        #ax_oc.imshow(self.results['p_matched'],cmap='binary',aspect='auto')
        ax_oc.set_xlabel('session')
        ax_oc.set_ylabel('neuron ID')


        ax = plt.axes([0.65,0.1,0.175,0.55])
        nC_clean = self.cluster.stats['cluster_bool'].sum()
        # ax.plot(((~np.isnan(assignments)) & idxes).sum(1),np.linspace(0,nC,nC),'ko',markersize=0.5)
        ax.plot((idx_pm & idxes)[self.cluster.stats['cluster_bool'],:].sum(1),np.linspace(1,nC_clean,nC_clean),'ko',markersize=1,mew=0)
        ax.invert_yaxis()
        ax.set_ylim([nC_clean,0])
        ax.set_yticks([])
        ax.set_xlabel('occ.')
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.65,0.65,0.175,0.3])
        # ax.hist((~np.isnan(assignments)).sum(1),np.linspace(0,nS,nS),color='r',cumulative=True,density=True,histtype='step')
        # ax.hist((ld_dat['p_matched']>pm_thr).sum(1),np.linspace(0,nS,nS),color='r',cumulative=True,density=True,histtype='step')
        ax.hist((idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:].sum(1),np.linspace(0,nS,nS),color='tab:green',cumulative=True,density=True,histtype='step')
        # ax.hist(((~np.isnan(assignments)) & idxes).sum(1),np.linspace(0,nS,nS),color='k',alpha=0.5,cumulative=True,density=True,histtype='step')
        ax.hist((idx_pm & idxes)[self.cluster.stats['cluster_bool'],:].sum(1),np.linspace(0,nS,nS),color='k',cumulative=True,density=True,histtype='step')
        ax.set_xticks([])
        #ax.set_yticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylim([0,1])
        ax.set_yticks(np.linspace(0,1,3))
        ax.set_ylabel('cdf')
        ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)

        ax = plt.axes([0.2,0.65,0.45,0.3])
        # ax.plot(np.linspace(0,nS,nS),(~np.isnan(assignments_OnACID)).sum(0),'ro',markersize=1,markerfacecolor='None')
        # ax.plot(np.linspace(0,nS,nS),(~np.isnan(assignments)).sum(0),'ro',markersize=1)
        # ax.plot(np.linspace(0,nS,nS),((~np.isnan(assignments)) & idxes).sum(0),'ko',markersize=1)
        nROI_prev = (idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:].sum(0)
        nROI_total = (idx_pm&idxes)[self.cluster.stats['cluster_bool'],:].sum(0)
        frac_ROI = nROI_total/nROI_prev-1
        print(np.nanmean(frac_ROI[frac_ROI<1.5]))
        print(np.nanstd(frac_ROI[frac_ROI<1.5]))
        # print()

        ax.plot(np.linspace(0,nS,nS),(idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:].sum(0),'^',color='tab:green',markersize=2,label='initial')
        # ax.plot(np.linspace(0,nS,nS),idx_pm[self.cluster.stats['cluster_bool'],:].sum(0),'^',color='tab:orange',markersize=2,label='inferred')
        ax.plot(np.linspace(0,nS,nS),(idx_pm & idxes)[self.cluster.stats['cluster_bool'],:].sum(0),'k^',markersize=2,label='total')
        ax.set_xlim([0,nS])
        ax.set_ylim([0,1300])
        ax.set_xticks([])
        ax.set_ylabel('# neurons')
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=8,bbox_to_anchor=[0.0,0.0],loc='lower left')



        plt.tight_layout()
        plt.show(block=False)
        if sv:
            ext = 'png'
            path = pathcat([self.pathMouse,'Figures/complete_set_evaluation.%s'%ext])
            plt.savefig(path,format=ext,dpi=300)


  def plot_threshold(self):
      print('### plot test of thresholds for neuron number ###')

      nSes = self.cluster.meta['nSes']
      f_max = 5
      # SNR_thr = 1
      rval_thr = 0
      CNN_thr = 0.6

      A0_thr = 1
      A_thr = 3
      pmass_thr = 0.5
      Bayes_thr = 10
      rel_thr = 0.1

      # if PC is None:
      #
      #     active = {'SNR':        {},
      #         'r_values':     {},
      #         'CNN':          {}}
      #
      #     PC = {'Bayes_factor':   {},
      #         'reliability':  {},
      #         'A_0':          {},
      #         'A':            {},
      #         'p_mass':       {}}
      #
      #     for s in tqdm(range(nSes)):
      #         PC_para = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_para.mat'%(s+1),variable_names=['Bayes_factor','posterior_mass','parameter'],squeeze_me=True)
      #         firingstats = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_firingstats.mat'%(s+1),variable_names=['trial_map','map'],squeeze_me=True)
      #         CNMF_results = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/results_redetect.mat'%(s+1),variable_names=['SNR','r_values','CNN'],squeeze_me=True)
      #
      #         active['SNR'][s] = CNMF_results['SNR']
      #         active['r_values'][s] = CNMF_results['r_values']
      #         active['CNN'][s] = CNMF_results['CNN']
      #         N = len(active['SNR'][s])
      #
      #         PC['Bayes_factor'][s] = PC_para['Bayes_factor'][...,0]
      #         PC['A_0'][s] = PC_para['parameter'][:,:,0,0]
      #         PC['A'][s] = PC_para['parameter'][:,:,1,0]
      #         PC['p_mass'][s] = PC_para['posterior_mass']
      #         PC['reliability'][s] = np.zeros((N,f_max))
      #         for n in range(N):
      #             for f in np.where(~np.isnan(PC['A'][s][n,:]))[0]:
      #                 PC['reliability'][s][n,f],_,_ = get_reliability(firingstats['trial_map'][n,...],firingstats['map'][n,...],PC_para['parameter'][n,...],f)


      # return active, PC
      ### get SNR dependence of detected neurons
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.05) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other = (cluster.stats['r_values'] > 0) & idx_pm
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.5) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other_certain = (cluster.stats['r_values'] > 0) & idx_pm
      r_thr = 0
      SNR_arr = np.linspace(1,10,10)
      nROI = np.zeros(SNR_arr.shape + (nSes,2))
      nPC = np.zeros(SNR_arr.shape + (nSes,2))
      plt.figure(figsize=(7,2.5))
      width = 0.4
      ax = plt.axes([0.1,0.2,0.25,0.65])
      for i,SNR_thr in enumerate(SNR_arr):

          for s in range(nSes):
              idx_active = (self.eval_data[s]['SNR'] > SNR_thr) & (self.eval_data[s]['r_values'] > r_thr) & (self.eval_data[s]['CNN'] > CNN_thr)
              nROI[i,s,0] = (idx_active).sum()

              morphed_A0_thr = A0_thr-PC['reliability'][s]/2
              idx_fields = (PC['A_0'][s]>morphed_A0_thr) & \
                          (PC['A'][s]>A_thr) & \
                          (PC['p_mass'][s]>pmass_thr) & \
                          (PC['Bayes_factor'][s]>Bayes_thr) & \
                          (PC['reliability'][s]>rel_thr)

              nPC[i,s,0] = (idx_active & np.any(idx_fields,1)).sum()
              # print('active neurons / place cells in session %d: %d / %d'%(s+1,nAct,nPC))

          # nROI[i,:,0] = ((cluster.stats['SNR']>SNR_thr) & idx_other).sum(0)
          # nROI[i,:,1] = ((cluster.stats['SNR']>SNR_thr) & idx_other_certain).sum(0)

          # nPC[i,:,0] = ((cluster.stats['SNR']>SNR_thr) & idx_other & cluster.status[...,2]).sum(0)
          # nPC[i,:,1] = ((cluster.stats['SNR']>SNR_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
          ax.scatter(SNR_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nROI[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
          ax.scatter(SNR_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nPC[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

      ax.plot(SNR_arr,nROI[:,self.cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')#,label='neurons ($p_m\geq0.05$)')
      # ax.plot(SNR_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')# ($p_m\geq0.95$)')
      ax.plot(SNR_arr,nPC[:,self.cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
      # ax.plot(SNR_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')

      ax.set_ylim([0,2700])
      ax.set_xlabel('$\Theta_{SNR}$')
      ax.set_ylabel('# neurons')
      ax.spines['top'].set_visible(False)

      ax2 = ax.twinx()
      # ax2.plot(SNR_arr,nPC[...,0].mean(1)/nROI[...,0].mean(1),'k:')
      ax2.plot(SNR_arr,(nPC[:,self.cluster.sessions['bool'],0]/nROI[:,self.cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
      ax2.plot([3,3],[0,1],'k--')
      ax2.set_ylim([0,0.54])
      # ax2.set_ylabel('PC fraction')
      ax2.set_yticklabels([])
      ax2.spines['top'].set_visible(False)
      ax.legend(fontsize=10,bbox_to_anchor=[0.3,1.15],loc='upper left')
      # plt.tight_layout()
      # plt.show(block=False)

      # if sv:
          # pl_dat.save_fig('neuronNumbers_SNR')


      ### get r_val dependence of detected neurons
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.05) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other = (cluster.stats['SNR'] > 2) & idx_pm
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.5) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other_certain = (cluster.stats['SNR'] > 2) & idx_pm

      # idx_other = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.5)
      # idx_other_certain = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.95)
      SNR_thr = 3
      r_arr = np.linspace(-0.4,1,8)
      nROI = np.zeros(r_arr.shape + (nSes,2))
      nPC = np.zeros(r_arr.shape + (nSes,2))
      # plt.figure(figsize=(4,2.5))
      width = 0.08
      ax = plt.axes([0.375,0.2,0.25,0.65])
      for i,r_thr in enumerate(r_arr):
          for s in range(nSes):
              idx_active = (self.eval_data[s]['SNR'] > SNR_thr) & (self.eval_data[s]['r_values'] > r_thr) & (self.eval_data[s]['CNN'] > CNN_thr)
              nROI[i,s,0] = (idx_active).sum()

              morphed_A0_thr = A0_thr-PC['reliability'][s]/2
              idx_fields = (PC['A_0'][s]>morphed_A0_thr) & \
                          (PC['A'][s]>A_thr) & \
                          (PC['p_mass'][s]>pmass_thr) & \
                          (PC['Bayes_factor'][s]>Bayes_thr) & \
                          (PC['reliability'][s]>rel_thr)

              nPC[i,s,0] = (idx_active & np.any(idx_fields,1)).sum()

          # nROI[i,:,0] = ((cluster.stats['r_values']>r_thr) & idx_other).sum(0)
          # nROI[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain).sum(0)

          # nPC[i,:,0] = ((cluster.stats['r_values']>r_thr) & idx_other & cluster.status[...,2]).sum(0)
          # nPC[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
          ax.scatter(r_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nROI[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
          ax.scatter(r_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nPC[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

      ax.plot(r_arr,nROI[:,self.cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')
      # ax.plot(r_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5)
      ax.plot(r_arr,nPC[:,self.cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
      # ax.plot(r_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5)

      ax.set_ylim([0,2700])
      ax.set_xlabel('$\Theta_{r}$')
      # ax.set_ylabel('# neurons')
      ax.set_yticklabels([])
      # ax.legend(fontsize=10)
      ax.spines['top'].set_visible(False)

      ax2 = ax.twinx()
      # ax2.plot(r_arr,nPC[...,0].mean(1)/nROI[...,0].mean(1),'k:')
      ax2.plot(r_arr,(nPC[:,self.cluster.sessions['bool'],0]/nROI[:,self.cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
      ax2.plot([0,0],[0,1],'k--')
      ax2.set_ylim([0,0.54])
      ax2.set_yticklabels([])
      # ax2.set_ylabel('PC fraction')
      ax2.spines['top'].set_visible(False)
      plt.tight_layout()
      plt.show(block=False)
      # return
      ### get pm dependence of detected neurons
      idx_other = (self.cluster.stats['SNR'] > 3) & (cluster.stats['r_values']>0)
      # idx_other_certain = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.95)
      pm_arr = np.linspace(0,1,11)
      nROI = np.zeros(pm_arr.shape + (nSes,2))
      nPC = np.zeros(pm_arr.shape + (nSes,2))
      # plt.figure(figsize=(4,2.5))
      width = 0.04
      ax = plt.axes([0.65,0.2,0.25,0.65])
      for i,pm_thr in enumerate(pm_arr):
          # idx_pm = cluster.stats['match_score'][...,0]>pm_thr
          idx_pm = ((self.cluster.stats['match_score'][...,0]-self.cluster.stats['match_score'][...,1])>pm_thr) | (self.cluster.stats['match_score'][...,0]>0.95)
          # idx_pm = (cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>pm_thr
          nROI[i,:,0] = (idx_pm & idx_other).sum(0)
          # nROI[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain).sum(0)

          nPC[i,:,0] = (idx_pm & idx_other & self.cluster.status[...,2]).sum(0)
          # nPC[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
          ax.scatter(pm_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nROI[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
          ax.scatter(pm_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nPC[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

      ax.plot(pm_arr,nROI[:,self.cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')
      # ax.plot(pm_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5)
      ax.plot(pm_arr,nPC[:,self.cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
      # ax.plot(pm_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'r^',markersize=4,markeredgewidth=0.5)

      ax.set_ylim([0,2700])
      ax.set_xlabel('$\Theta_{p_m}$')
      # ax.set_ylabel('# neurons')
      ax.set_yticklabels([])
      # ax.legend(fontsize=10)
      ax.spines['top'].set_visible(False)

      ax2 = ax.twinx()
      ax2.plot(pm_arr,(nPC[:,self.cluster.sessions['bool'],0]/nROI[:,self.cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
      # ax2.plot(pm_arr,nPC[...,1].mean(1)/nROI[...,1].mean(1),'k-')
      ax2.plot([0.5,0.5],[0,1],'k--')
      ax2.set_ylim([0,0.54])
      ax2.set_ylabel('PC fraction')
      ax2.spines['top'].set_visible(False)
      plt.tight_layout()
      plt.show(block=False)

      if sv:
          pl_dat.save_fig('neuronNumbers_test')


      print('whats with MI of place cells, only?')
      ### get SNR dependence of detected neurons
      idx_other = (self.cluster.stats['r_values'] > 0) & (self.cluster.stats['match_score'][...,0]>0.05)
      idx_other_certain = (self.cluster.stats['r_values'] > 0) & (self.cluster.stats['match_score'][...,0]>0.95)
      SNR_arr = np.linspace(2,10,5)
      MI = np.zeros(SNR_arr.shape + (2,))
      # nPC = np.zeros(SNR_arr.shape + (nSes,2))
      plt.figure(figsize=(4,2.5))
      width = 0.6
      ax = plt.axes([0.2,0.2,0.65,0.65])
      for i,SNR_thr in enumerate(SNR_arr):
          idx = (self.cluster.stats['SNR'] >= (SNR_thr-0.5)) & (self.cluster.stats['SNR'] < (SNR_thr+0.5)) & idx_other
          MI[i,0] = self.cluster.stats['MI_value'][idx].mean(0)
          idx = (self.cluster.stats['SNR'] >= (SNR_thr-0.5)) & (self.cluster.stats['SNR'] < (SNR_thr+0.5)) & idx_other_certain
          MI[i,1] = self.cluster.stats['MI_value'][idx].mean(0)

          ax.boxplot(self.cluster.stats['MI_value'][idx],positions=[SNR_thr],widths=width,whis=[5,95],notch=True,bootstrap=100,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

      ax.plot(SNR_arr,MI[...,1],'k^',markersize=5,label='neurons')

      ax.set_ylim([0,0.6])
      ax.set_xlabel('$\Theta_{SNR}$')
      ax.set_ylabel('MI')
      ax.legend(fontsize=10)

      plt.tight_layout()
      plt.show(block=False)

      # if sv:
          # pl_dat.save_fig('MI_SNR')


  def plot_detection(self,ax,A1,A2,bounds=None,lw=0.7,ls='solid'):
    dims = (512,512)

    if bounds is None:
      bounds = [[0,dims[0]],[0,dims[1]]]

    A1_norm = normalize_sparse_array(A1)
    A2_norm = normalize_sparse_array(A2)

    ax.imshow(A1_norm.sum(1).reshape(dims)+A2_norm.sum(1).reshape(dims),origin='lower')

    CM = com(A1,dims[0],dims[1])
    idx_bounds = (CM[:,0] > bounds[0][0]) & (CM[:,0] < bounds[0][1]) & (CM[:,1] > bounds[1][0]) & (CM[:,1] < bounds[1][1])
    [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.5], colors='w', linewidths=0.5) for a in A1[:,idx_bounds].T]

    CM = com(A2,dims[0],dims[1])
    idx_bounds = (CM[:,0] > bounds[0][0]) & (CM[:,0] < bounds[0][1]) & (CM[:,1] > bounds[1][0]) & (CM[:,1] < bounds[1][1])
    [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.5], colors='tab:orange', linewidths=lw,linestyles=ls) for a in A2[:,idx_bounds].T]


    #plt.figure(figsize=(5,4))
    #ax = plt.subplot(111)
    #ax.imshow(self.dataOut['A'].sum(1).reshape(self.cluster.meta['dims']),origin='lower')

    #[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['active_good']].T]
    #[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['silent_good']].T]

    ##[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=1) for a in self.dataIn['A'][:,self.idxes['in']['silent']].T]
    #plt.tight_layout()
    #plt.show(block=False)
    #pathFigure = pathcat([pathSession,'find_silent_after.%s'%(ext)]);
    #plt.savefig(pathFigure,format=ext,dpi=300)
    #print('Figure saved as %s'%pathFigure)


# def compute_event_exceptionality(traces, robust_std=False, N=5, sigma_factor=3.):
#     """
#     Define a metric and order components according to the probability of some "exceptional events" (like a spike).

#     Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
#     The function first estimates the noise distribution by considering the dispersion around the mode.
#     This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
#     Then, the probability of having N consecutive events is estimated.
#     This probability is used to order the components.

#     Args:
#         traces: ndarray
#             Fluorescence traces

#         N: int
#             N number of consecutive events

#         sigma_factor: float
#             multiplicative factor for noise estimate (added for backwards compatibility)

#     Returns:
#         fitness: ndarray
#             value estimate of the quality of components (the lesser the better)

#         erfc: ndarray
#             probability at each time step of observing the N consequtive actual trace values given the distribution of noise

#         noise_est: ndarray
#             the components ordered according to the fitness
#     """

#     T = np.shape(traces)[-1]

#     md = find_modes(traces,axis=1)
#     ff1 = traces - md[:,None]

#     # only consider values under the mode to determine the noise standard deviation
#     ff1 = -ff1 * (ff1 < 0)
#     if robust_std:

#         # compute 25 percentile
#         ff1 = np.sort(ff1, axis=1)
#         ff1[ff1 == 0] = np.nan
#         Ns = np.round(np.sum(ff1 > 0,1) * .5)
#         iqr_h = np.zeros(traces.shape[0])

#         for idx, _ in enumerate(ff1):
#             iqr_h[idx] = ff1[idx, -Ns[idx]]

#         # approximate standard deviation as iqr/1.349
#         sd_r = 2 * iqr_h / 1.349

#     else:
#         Ns = np.sum(ff1 > 0, -1)
#         sd_r = np.sqrt(old_div(np.sum(ff1**2, -1), Ns))

#     # compute z value
#     z = old_div((traces - md[:,None]), (sigma_factor * sd_r[:,None]))

#     # probability of observing values larger or equal to z given normal
#     # distribution with mean md and std sd_r
#     #erf = 1 - norm.cdf(z)

#     # use logarithm so that multiplication becomes sum
#     #erf = np.log(erf)
#     # compute with this numerically stable function
#     erf = scipy.special.log_ndtr(-z)

#     # moving sum
#     erfc = np.cumsum(erf, 1)
#     erfc[:, N:] -= erfc[:, :-N]

#     # select the maximum value of such probability for each trace
#     fitness = np.min(erfc, 1)

#     return fitness, erfc, sd_r, md


if __name__=='__main__':
    sr = silence_redetection()
    print(sr.cluster.paths)
