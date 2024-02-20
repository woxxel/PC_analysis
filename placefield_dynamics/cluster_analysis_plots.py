import os, cv2, time, itertools

from tqdm import *
import numpy as np
from matplotlib import pyplot as plt, colors, image as mpimg
from matplotlib_scalebar.scalebar import ScaleBar
import scipy as sp
import scipy.stats as sstats
from collections import Counter

from multiprocessing import get_context
from caiman.utils.utils import load_dict_from_hdf5

from .cluster_analysis import cluster_analysis

from .placefield_detection import get_firingrate, compute_serial_matrix

from .utils import get_ICPI, get_dp, gauss_smooth, add_number, bootstrap_data, com

from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

class cluster_analysis_plots(cluster_analysis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_plots()

    def setup_plots(self,sv_ext='png'):
        
        nSes = self.data['nSes']
        self.pl_dat = plot_dat(self.data['mouse'],self.paths['figures'],nSes,self.data,sv_ext=sv_ext)


    def plot_turnover_mechanism(self,mode='act',sv=False,N_bs=10):
        print('### plot cell activity statistics ###')
        s = 10

        # session_bool = np.pad(self.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(self.status['sessions'][:],(0,0),constant_values=False)

        # s_bool = np.zeros(self.data['nSes'],'bool')
        # s_bool[17:87] = True
        # s_bool[~self.status['sessions']] = False
        s_bool = self.status['sessions']
        state_label = 'alpha' if (mode=='act') else 'beta'
        status_act = self.status['activity'][self.status['clusters'],:,1]
        status_act = status_act[:,s_bool]
        # status_act = status_act[:,session_bool]
        status_PC = self.status['activity'][self.status['clusters'],:,2]
        status_PC = status_PC[:,s_bool]
        nC_good,nSes_good = status_act.shape
        nSes_max = np.where(s_bool)[0][-1]


        active_neurons = status_act.mean(0)
        silent_neurons = (~status_act).mean(0)
        print('active neurons: %.3g +/- %.3g'%(active_neurons.mean()*100,active_neurons.std()*100))
        print('silent neurons: %.3g +/- %.3g'%(silent_neurons.mean()*100,silent_neurons.std()*100))

        coding_neurons = status_PC.sum(0)/status_act.sum(0)
        ncoding_neurons = ((~status_PC) & status_act).sum(0)/status_act.sum(0)
        print('coding neurons: %.3g +/- %.3g'%(coding_neurons.mean()*100,coding_neurons.std()*100))
        print('non-coding neurons: %.3g +/- %.3g'%(ncoding_neurons.mean()*100,coding_neurons.std()*100))


        p_act = np.count_nonzero(status_act)/(nC_good*nSes_good)
        p_PC = np.count_nonzero(status_PC)/np.count_nonzero(status_act)
        # print(p_PC)
        rnd_var_act = np.random.random(status_act.shape)
        rnd_var_PC = np.random.random(status_PC.shape)
        status_act_test = np.zeros(status_act.shape,'bool')
        status_act_test_rnd = np.zeros(status_act.shape,'bool')
        status_PC_test = np.zeros(status_PC.shape,'bool')
        status_PC_test_rnd = np.zeros(status_PC.shape,'bool')
        for c in range(nC_good):

            # status_act_test[c,:] = rnd_var_act[c,:] < (np.count_nonzero(status_act[c,:])/nSes_good)
            nC_act = status_act[c,:].sum()
            status_act_test[c,np.random.choice(nSes_good,nC_act,replace=False)] = True
            status_act_test_rnd[c,:] = rnd_var_act[c,:] < p_act
            # status_PC_test[c,status_act_test[c,:]] = rnd_var_PC[c,status_act_test[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
            # print(status_act[c,:])
            # status_PC_test[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
            status_PC_test[c,np.where(status_act[c,:])[0][np.random.choice(nC_act,status_PC[c,:].sum(),replace=False)]] = True
            status_PC_test_rnd[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < p_PC


        status = status_act if mode=='act' else status_PC
        status_test = status_act_test if mode=='act' else status_PC_test

        fig = plt.figure(figsize=(7,5),dpi=self.pl_dat.sv_opt['dpi'])

        ax_sketch = plt.axes([0.04,0.875,0.25,0.1])
        self.pl_dat.add_number(fig,ax_sketch,order=1,offset=[-40,10])
        if mode=='act':
            pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/sketches/neural_network_active.png'
        else:
            pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/sketches/neural_network_PC.png'

        if os.path.exists(pic_path):
            ax_sketch.axis('off')
            im = mpimg.imread(pic_path)
            ax_sketch.imshow(im)
            ax_sketch.set_xlim([0,im.shape[1]])

        if sv:   ## enable, when saving
            ### plot contours of two adjacent sessions
            # load data from both sessions
            pathLoad = os.path.join(self.paths['sessions'][10],self.paths['fileNameCNMF'])
            ld = load_dict_from_hdf5(pathLoad)
            A1 = ld['A']#.toarray().reshape(self.params['dims'][0],self.params['dims'][1],-1)
            # Cn = A1.sum(1).reshape(self.params['dims'])
            Cn = ld['Cn'].transpose()
            Cn -= Cn.min()
            Cn /= Cn.max()

            pathLoad = os.path.join(self.paths['sessions'][s+1],self.paths['fileNameCNMF'])
            ld = load_dict_from_hdf5(pathLoad)
            A2 = ld['A']

            # adjust to same reference frame
            x_grid, y_grid = np.meshgrid(np.arange(0., self.params['dims'][0]).astype(np.float32), np.arange(0., self.params['dims'][1]).astype(np.float32))
            x_remap = (x_grid - \
            self.alignment['shift'][s+1,0] + self.alignment['shift'][s,0] + \
            self.alignment['flow'][s+1,0,:,:] - self.alignment['flow'][s,0,:,:]).astype('float32')
            y_remap = (y_grid - \
            self.alignment['shift'][s+1,1] + self.alignment['shift'][s,1] + \
            self.alignment['flow'][s+1,1,:,:] - self.alignment['flow'][s,1,:,:]).astype('float32')

            ax_ROI = plt.axes([0.04,0.48,0.25,0.375])
            # pl_dat.add_number(fig,ax_ROI,order=1,offset=[-25,25])
            # plot background, based on first sessions
            ax_ROI.imshow(Cn,origin='lower',clim=[0,1],cmap='viridis')

            # plot contours occuring in first and in second session, only, and...
            # plot contours occuring in both sessions (taken from first session)

            twilight = plt.get_cmap('hsv')
            cNorm = colors.Normalize(vmin=0,vmax=100)
            scalarMap = plt.cm.ScalarMappable(norm=cNorm,cmap=twilight)

            idx = 1 if mode=='act' else 2
            idx_s1 = self.status['activity'][:,s,idx] & (~self.status['activity'][:,s+1,idx])
            idx_s2 = self.status['activity'][:,s+1,idx] & (~self.status['activity'][:,s,idx])
            idx_s12 = self.status['activity'][:,s+1,idx] & (self.status['activity'][:,s,idx])

            n_s1 = self.matching['IDs'][idx_s1,s].astype('int')
            n_s2 = self.matching['IDs'][idx_s2,s+1].astype('int')
            n_s12 = self.matching['IDs'][idx_s12,s].astype('int')

            A_tmp = sp.sparse.hstack([sp.sparse.csc_matrix(cv2.remap(img.reshape(self.params['dims']), x_remap,y_remap, cv2.INTER_CUBIC).reshape(-1,1)) for img in A2[:,n_s2].toarray().T])

            if mode=='act':
                [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['dashed']) for a in A1[:,n_s1].T]
                [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['solid']) for a in A1[:,n_s12].T]
                [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['dotted']) for a in A_tmp.T]
            elif mode=='PC':
                # print(np.where(idx_s1)[0])
                # print(n_s1)
                for c,n in zip(np.where(idx_s1)[0],n_s1):
                    a = A1[:,n]
                    f = np.where(self.fields['status'][c,s,:]>2)[0][0]
                    colVal = scalarMap.to_rgba(self.fields['location'][c,s,f,0])
                    ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['dashed'])
                # print(np.where(idx_s2)[0])
                # print(n_s2)
                for i,(c,n) in enumerate(zip(np.where(idx_s2)[0],n_s2)):
                    a = A_tmp[:,i]
                    f = np.where(self.fields['status'][c,s+1,:]>2)[0][0]
                    colVal = scalarMap.to_rgba(self.fields['location'][c,s+1,f,0])
                    ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['dotted'])
                for c,n in zip(np.where(idx_s12)[0],n_s12):
                    a = A1[:,n]
                    f = np.where(self.fields['status'][c,s,:]>2)[0][0]
                    colVal = scalarMap.to_rgba(self.fields['location'][c,s,f,0])
                    ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['solid'])

            if mode=='PC':
                # cbaxes = plt.axes([0.285,0.75,0.01,0.225])
                cbaxes = plt.axes([0.04,0.47,0.15,0.0125])
                cb = fig.colorbar(scalarMap,cax = cbaxes,orientation='horizontal')
                # cb.set_label('location')
                cb.set_label('location',fontsize=8,rotation='horizontal',ha='left',va='center')#,labelpad=0,y=-0.5)
                # print(cb.ax.__dict__.keys())
                cbaxes.xaxis.set_label_coords(1.075,0.4)

            ax_ROI.plot(np.NaN,np.NaN,'k-',label='$\\%s_{s_1}^+ \cap \\%s_{s_2}^+$'%(state_label,state_label))
            ax_ROI.plot(np.NaN,np.NaN,'k--',label='$\\%s_{s_1}^+$'%state_label)
            ax_ROI.plot(np.NaN,np.NaN,'k:',label='$\\%s_{s_2}^+$'%state_label)
            ax_ROI.legend(fontsize=10,bbox_to_anchor=[1.2,1.1],loc='upper right',handlelength=1)

            sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
            ax_ROI.add_artist(sbar)
            ax_ROI.set_xticks([])
            ax_ROI.set_yticks([])
        # plt.show()
        # return
        ### plot distribution of active sessions/neuron
        ax = plt.axes([0.45,0.85,0.175,0.1])
        self.pl_dat.add_number(fig,ax,order=2,offset=[-225,25])
        if mode=='act':
            ax.plot([0,self.data['nSes']],[nC_good,nC_good],'k--',linewidth=0.5)
            ax.plot(np.where(self.status['sessions'])[0],self.status['activity'][:,self.status['sessions'],1].sum(0),'k.',markersize=1)
            ax.set_ylim([0,1000])
            ax.set_ylabel('# neurons')
        else:
            # ax.plot([0,self.data['nSes']],[nC_good,nC_good],'k--',linewidth=0.5)
            ax.plot(np.where(self.status['sessions'])[0],self.status['activity'][:,self.status['sessions'],2].sum(0),'k.',markersize=1)
            ax.set_ylim([0,750])
            ax.set_ylabel('# PC')
        ax.set_xlabel('session')
        self.pl_dat.remove_frame(ax,['top','right'])



        ax = plt.axes([0.45,0.58,0.175,0.15])
        self.pl_dat.add_number(fig,ax,order=3,offset=[-200,50])

        # plt.hist(self.status['activity'][...,1:3].sum(1),pl_dat.h_edges,color=[[0.6,0.6,0.6],'k'],width=0.4,label=['# sessions active','# sessions coding']);
        ax.axhline(self.status['sessions'].sum(),color='r',linestyle='--',zorder=0)
        ax.hist(status.sum(1),self.pl_dat.h_edges,color='k',width=1,label='emp. data');
        # ax.hist(status_test.sum(1),pl_dat.h_edges,color='tab:red',alpha=0.7,width=0.8,label='rnd. data');
        if mode=='act':
            ax.hist((status_act_test_rnd).sum(1),self.pl_dat.h_edges,color=[0.5,0.5,0.5],alpha=0.5,width=1)
            res = sstats.ks_2samp(status_act.sum(1),status_act_test_rnd.sum(1))
            print(res)
        elif mode=='PC':
            ax.hist((status_PC_test_rnd).sum(1),self.pl_dat.h_edges,color=[1,0.5,0.5],alpha=0.5,width=1)
            res = sstats.ks_2samp(status_PC.sum(1),status_PC_test_rnd.sum(1))
            print(res)

        ax.set_xlabel('$N_{\\%s^+}$'%state_label)
        ax.set_ylabel('# neurons')
        # ax.legend(fontsize=10,loc='upper right')
        ax.set_xlim([-0.5,nSes_good+0.5])
        if mode=='act':
            ax.set_ylim([0,300])
        elif mode =='PC':
            ax.set_ylim([0,500])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        ### calculating ICPI
        status_alt = np.zeros_like(self.status['activity'][...,1],'int')

        IPI_test = np.zeros(self.data['nSes'])
        # for c in range(self.data['nC']):
        for c in np.where(self.status['clusters'])[0]:
            s0 = 0
            inAct = False
            for s in np.where(s_bool)[0]:
                if inAct:
                    if ~self.status['activity'][c,s,1]:
                        La = s_bool[s0:s].sum()
                        status_alt[c,s0:s] = La
                        IPI_test[La] += 1
                        # print(s_bool[s:s0].sum())
                        inAct=False
                else:
                    if self.status['activity'][c,s,1]:
                        # print('getting active')
                        s0 = s
                        inAct = True
            if inAct:
                La = s_bool[s0:s+1].sum()
                status_alt[c,s0:s+1] = La
                IPI_test[La] += 1

        status_alt[:,~s_bool] = 0
        # print(IPI_test)
        ### obtain inter-coding intervals
        ICI = np.zeros((nSes_good,2))    # inter-coding-interval
        IPI = np.zeros((nSes_good,2))    # inter-pause-interval (= coding duration)

        # print(status.shape)
        # return

        t_start = time.time()
        ICI[:,0] = get_ICPI(status,mode='ICI')
        print('time taken: %.2f'%(time.time()-t_start))
        t_start = time.time()
        ICI[:,1] = get_ICPI(status_test,mode='ICI')

        IPI[:,0] = get_ICPI(~status,mode='IPI')
        IPI[:,1] = get_ICPI(~status_test,mode='IPI')

        # print(IPI[:,0])

        IPI_bs = np.zeros((nSes_good,N_bs))
        IPI_bs_test = np.zeros((nSes_good,N_bs))
        ICI_bs = np.zeros((nSes_good,N_bs))
        ICI_bs_test = np.zeros((nSes_good,N_bs))
        for i in range(N_bs):
            IPI_bs[:,i] = get_ICPI(~status[np.random.randint(0,nC_good,nC_good),:],mode='IPI')
            IPI_bs_test[:,i] = get_ICPI(~status_test[np.random.randint(0,nC_good,nC_good),:],mode='IPI')

            ICI_bs[:,i] = get_ICPI(status[np.random.randint(0,nC_good,nC_good),:],mode='ICI')
            ICI_bs_test[:,i] = get_ICPI(status_test[np.random.randint(0,nC_good,nC_good),:],mode='ICI')
        
        ICI_summed =  ICI*np.arange(nSes_good)[:,np.newaxis]
        IPI_summed =  IPI*np.arange(nSes_good)[:,np.newaxis]

        ## end calculating ICPIs

        pval_IPI = np.zeros(nSes_good)*np.NaN
        pval_ICI = np.zeros(nSes_good)*np.NaN
        for s in range(nSes_good):
            # print('ttest (s=%d)'%(s+1))
            # print(np.nanmean(IPI_bs[s,:]),np.nanstd(IPI_bs[s,:]))
            # print(np.nanmean(IPI_bs_test[s,:]),np.nanstd(IPI_bs_test[s,:]))
            # res = sstats.ttest_ind(IPI_bs[s,:],IPI_bs_test[s,:])
            # print(res)
            res = sstats.ttest_ind_from_stats(np.nanmean(IPI_bs[s,:]),np.nanstd(IPI_bs[s,:]),N_bs,np.nanmean(IPI_bs_test[s,:]),np.nanstd(IPI_bs_test[s,:]),N_bs,equal_var=True)
            pval_IPI[s] = res.pvalue

            res = sstats.ttest_ind_from_stats(np.nanmean(ICI_bs[s,:]),np.nanstd(ICI_bs[s,:]),N_bs,np.nanmean(ICI_bs_test[s,:]),np.nanstd(ICI_bs_test[s,:]),N_bs,equal_var=True)
            pval_ICI[s] = res.pvalue

        print('time taken: %.2f'%(time.time()-t_start))
        IPI_bs[IPI_bs==0] = np.NaN
        # print(IPI_bs)

        ICI[ICI==0] = np.NaN
        IPI[IPI==0] = np.NaN

        ax = plt.axes([0.75,0.11,0.225,0.25])
        self.pl_dat.add_number(fig,ax,order=7)
        # ax.loglog(IPI[:,0],'k-',label='IPI')
        self.pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(IPI_bs,1),np.nanstd(IPI_bs,1),col='k',lw=0.5,label='$I_{\\%s^+}$'%state_label)
        self.pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(IPI_bs_test,1),np.nanstd(IPI_bs_test,1),col='tab:red',lw=0.5)
        self.pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(ICI_bs,1),np.nanstd(ICI_bs,1),col='k',ls=':',lw=0.5,label='$I_{\\%s^-}$'%state_label)
        self.pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good,nSes_good),np.nanmean(ICI_bs_test,1),np.nanstd(ICI_bs_test,1),col='tab:red',ls=':',lw=0.5)
        # ax.loglog(np.nanmean(IPI_bs,1)-np.nanstd(IPI_bs,1),'-',color=[0.1,0.5,0.5])
        # ax.loglog(np.nanmean(IPI_bs,1)+np.nanstd(IPI_bs,1),'-',color=[0.5,0.1,0.5])
        # ax.loglog(np.nanmean(IPI_bs,1),'k-',label='IPI')
        # ax.loglog(IPI.mean(0)+IPI.std(0),'-',color=[0.5,0.5,0.5])
        # ax.loglog(IPI[:,1],'tab:red')
        # ax.loglog(ICI[:,0],'k:',label='ICI')
        # ax.loglog(ICI[:,1],color='tab:red',linestyle=':')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.setp(ax,xlim=[0.9,np.maximum(105,nSes_max)],ylim=[1,10**5],
                 ylabel='# occurence',xlabel='$\mathcal{L}_{\\%s^+}$ / $\mathcal{L}_{\\%s^-}$ [sessions]'%(state_label,state_label))
        ax.legend(fontsize=10,loc='lower left')
        ax.spines[['top','right']].set_visible(False)

        # print(np.log10(pval))
        # ax = plt.axes([0.21,0.325,0.1,0.075])
        # ax.plot(np.log10(pval_IPI),'k',linewidth=0.5)
        # ax.plot(np.log10(pval_ICI),'k:',linewidth=0.5)
        # ax.plot([0,nSes_good],[-10,-10],'k--',linewidth=0.3)
        # ax.set_xscale('log')
        # ax.set_xlim([0.9,np.maximum(105,nSes_max)])
        # ax.set_xticks(np.logspace(0,2,3))
        # # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        # ax.set_ylim([-300,0])
        # ax.set_ylabel('$\log_{10}(p_{val})$',fontsize=7,rotation='horizontal',labelpad=-15,y=1.15)#,ha='center',va='center')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        
        # ax = plt.axes([0.45,0.325,0.15,0.15])
        # pl_dat.add_number(fig,ax,order=5)
        # ax.plot(IPI_summed[:,0]/np.nansum(IPI_summed[:,0]),'k-',label='$I_{\\%s^+}$'%(state_label))
        # ax.plot(IPI_summed[:,1]/np.nansum(IPI_summed[:,1]),'-',color='tab:red')
        # ax.plot(ICI_summed[:,0]/np.nansum(ICI_summed[:,0]),'k:',label='$I_{\\%s^-}$'%(state_label))
        # ax.plot(ICI_summed[:,1]/np.nansum(ICI_summed[:,1]),':',color='tab:red')
        # ax.set_xscale('log')
        # ax.set_xticklabels([])
        # ax.set_ylabel('$p_{\in \mathcal{L}_{\\%s^+} / \mathcal{L}_{\\%s^-}}$'%(state_label,state_label))
        # # ax.legend(fontsize=10)
        # ax.set_xlim([0.8,np.maximum(105,nSes_max)])
        # ax.set_xticks(np.logspace(0,2,3))
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.plot(IPI_summed[:,1]/np.nansum(IPI_summed[:,1]),'-',color='tab:red')
        # ax.plot(ICI_summed[:,1]/np.nansum(ICI_summed[:,1]),':',color='tab:red')
        # ax.set_yscale('log')

        ax = plt.axes([0.875,0.35,0.1,0.1])
        # ax.plot(IPI*np.arange(self.data['nSes'])/self.status['activity'][...,1].sum())

        ax.plot(range(1,nSes_good),np.nancumsum(IPI_summed[1:,0]/np.nansum(IPI_summed[1:,0])),'k-')
        ax.plot(range(1,nSes_good),np.nancumsum(IPI_summed[1:,1]/np.nansum(IPI_summed[1:,1])),'-',color='tab:red')
        ax.plot(range(1,nSes_good),np.nancumsum(ICI_summed[1:,0]/np.nansum(ICI_summed[1:,0])),'k:')
        ax.plot(range(1,nSes_good),np.nancumsum(ICI_summed[1:,1]/np.nansum(ICI_summed[1:,1])),':',color='tab:red')

        # ax.legend(fontsize=10)
        ax.set_xscale('log')
        ax.set_xlim([0.8,np.maximum(105,nSes_max)])
        ax.set_xticks(np.logspace(0,2,3))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(9))
        # print(LogLocator().tick_values(1,100))
        # ax.set_xlabel('$\mathcal{L}_{\\%s^+}$ / $\mathcal{L}_{\\%s^-}$'%(state_label,state_label))
        ax.set_ylabel('$cdf_{\in \mathcal{L}_{\\%s^+} / \mathcal{L}_{\\%s^-}}$'%(state_label,state_label),fontsize=8,rotation='horizontal',labelpad=-15,y=1.15)#,ha='center',va='center')
        # ax.set_xlabel('cont.coding [ses.]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax = plt.subplot(4,4,4)
        # ax.plot(status.sum(1),IPI_stats[:,2],'k.',markersize=0.5)

        status_act = self.status['activity'][self.status['clusters'],:,1]
        status_PC = self.status['activity'][self.status['clusters'],:,2]
        # status_act = self.status['activity'][...,1]
        # status_PC = self.status['activity'][...,2]
        status = status_act if mode=='act' else status_PC
        status_dep = None if mode=='act' else status_act

        status[:,~s_bool] = False
        ds = 1
        dp_pos,p_pos = get_dp(status,status_dep=status_dep,status_session=s_bool,ds=ds,mode=mode)
        dp_neg,p_neg = get_dp(~status,status_dep=status_dep,status_session=s_bool,ds=ds,mode=mode)

        status_dep = None if mode=='act' else status_act_test
        dp_pos_test,p_pos_test = get_dp(status_test,status_dep=status_dep,ds=ds,mode=mode)
        dp_neg_test,p_neg_test = get_dp(~status_test,status_dep=status_dep,ds=ds,mode=mode)
        # dp_pos,p_pos = get_dp(status,status_act,status_dep=status_dep,status_session=s_bool,ds=ds,mode=mode)
        # dp_neg,p_neg = get_dp(~status,status_act,status_dep=status_dep,status_session=s_bool,ds=ds,mode=mode)
        #
        # status_dep = None if mode=='act' else status_act_test
        # dp_pos_test,p_pos_test = get_dp(status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)
        # dp_neg_test,p_neg_test = get_dp(~status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)


        ax = plt.axes([0.1,0.11,0.16,0.225])
        self.pl_dat.add_number(fig,ax,order=5)
        ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),p_pos+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0,alpha=0.6,label='$\\%s^+_s$'%(state_label))
        ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),p_pos_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0,zorder=1)
        ax.set_yticks(np.linspace(0,1,3))
        ax.set_xlabel('$N_{\\%s^+}$'%(state_label))
        ax.set_ylabel('$p(\\%s^+_{s+1} | \\%s^+_s)$'%(state_label,state_label))
        self.pl_dat.remove_frame(ax,['top','right'])



        res = sstats.ks_2samp(dp_pos,dp_pos_test)
        print('IPI')
        print(res)
        # print(np.nanmean(dp_pos),np.nanstd(dp_pos))
        # print(np.nanpercentile(dp_pos,[2.5,97.5]))
        # print(np.nanmean(dp_pos_test),np.nanstd(dp_pos_test))

        res = sstats.kruskal(dp_pos,dp_pos_test,nan_policy='omit')
        # print(res)

        res = sstats.ks_2samp(dp_neg,dp_neg_test)
        print('IAI')
        print(res)
        # print(np.nanmean(dp_neg),np.nanstd(dp_neg))
        # print(np.nanmean(dp_neg_test),np.nanstd(dp_neg_test))

        width=0.75
        ax = plt.axes([0.41,0.3,0.075,0.125])
        self.pl_dat.add_number(fig,ax,order=6)
        ax.plot([-0.5,1.5],[0,0],'--',color=[0.6,0.6,0.6],linewidth=0.5)
        bp = ax.boxplot(dp_pos[np.isfinite(dp_pos)],positions=[0],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        bp_test = ax.boxplot(dp_pos_test[np.isfinite(dp_pos_test)],positions=[1],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        for element in ['boxes','whiskers','means','medians','caps']:
            plt.setp(bp[element], color='k')
            plt.setp(bp_test[element], color='tab:red')
        # ax.bar(1,np.nanmean(dp_pos_test),facecolor='tab:red')
        # ax.errorbar(1,np.nanmean(dp_pos_test),np.abs(np.nanmean(dp_pos_test)-np.nanpercentile(dp_pos_test,[2.5,97.5]))[:,np.newaxis],ecolor='r')
        self.pl_dat.remove_frame(ax,['top','right','bottom'])
        ax.set_xticks([])
        ax.set_ylabel('$\left\langle \Delta p_{\\%s} \\right \\rangle$'%state_label)
        ax.set_ylim([-0.25,0.75])

        ax2 = plt.axes([0.41,0.11,0.075,0.175])
        ax2.hist(dp_pos,np.linspace(-1,1,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.hist(dp_pos_test,np.linspace(-1,1,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_pos,np.linspace(0,2,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_pos_test,np.linspace(0,2,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.set_xticks([])# status_dilate = sp.ndimage.morphology.binary_dilation(status,np.ones((1,3),'bool'))
        ax2.set_xlim([0,ax2.get_xlim()[1]*2])
        ax2.set_ylim([-0.5,1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax = ax2.twiny()
        ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),dp_pos+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0,alpha=0.6,label='$\\%s^+_s$'%(state_label))
        ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),dp_pos_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0,zorder=1)

        ax.set_yticks(np.linspace(-1,1,5))
        ax.set_xlim([-20,nSes_good])
        ax.set_ylim([-0.5,1])
        # ax.set_ylim([0,4])

        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_xlabel('$N_{\\%s^+}$'%(state_label),x=1,y=-0.1)
        ax2.set_ylabel('$\Delta p (\\%s^+_{s+1} | \\%s^+_s)$'%(state_label,state_label))#'$p_{\\alpha}^{\pm1}$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(fontsize=10,loc='upper left')

        ax = plt.axes([0.5,0.3,0.075,0.125])
        ax.plot([-0.5,1.5],[0,0],'--',color=[0.6,0.6,0.6],linewidth=0.5)
        bp = ax.boxplot(dp_neg[np.isfinite(dp_neg)],positions=[0],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        bp_test = ax.boxplot(dp_neg_test[np.isfinite(dp_neg_test)],positions=[1],widths=width,whis=[5,95],notch=True,bootstrap=100,showfliers=False)#,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        for element in ['boxes','whiskers','means','medians','caps']:
            plt.setp(bp[element], color='k')
            plt.setp(bp_test[element], color='tab:red')

        self.pl_dat.remove_frame(ax,['top','right','bottom'])
        ax.set_xticks([])
        ax.set_ylim([-0.25,0.75])
        ax.set_yticklabels([])

        ax2 = plt.axes([0.5,0.11,0.075,0.175])
        ax2.invert_xaxis()
        ax2.hist(dp_neg,np.linspace(-1,1,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.hist(dp_neg_test,np.linspace(-1,1,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp,np.linspace(0,2,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_test,np.linspace(0,2,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.set_xticks([])
        ax2.set_xlim([ax2.get_xlim()[0]*2,0])
        ax2.set_ylim([-0.5,1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax = ax2.twiny()
        ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),dp_neg+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0,alpha=0.6,label='$\\beta_s$')
        ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),dp_neg_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0,zorder=1)
        # ax.set_ylim([0,1])
        ax.set_xlim([0,nSes_good+20])
        ax.set_yticks([])
        # ax.set_yticks(np.linspace(0,1,3))
        ax.set_ylim([-0.5,1])
        # ax.set_ylim([0,4])
        ax.xaxis.tick_bottom()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('$\Delta p (\\%s^-_{s+1} | \\%s^-_s)$'%(state_label,state_label))
        # ax.set_xlabel('\t # sessions')
        # ax.set_ylabel('$p (\\alpha_{s\pm1} | \\alpha_s)$')#'$p_{\\alpha}^{\pm1}$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)




        # ax.legend(fontsize=10,loc='lower left')


        # ax = plt.axes([0.75,0.625,0.125,0.325])
        # status_dilate = sp.ndimage.morphology.binary_dilation(~status,np.ones((1,3),'bool'))
        # cont_score = 1-(status_dilate&(~status)).sum(1)/(2*status.sum(1))
        # ax = plt.axes([0.75,0.625,0.225,0.325])
        # ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),cont_score+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0)
        #
        # status_dilate_test = sp.ndimage.morphology.binary_dilation(status_test,np.ones((1,3),'bool'))
        # cont_score_test = 1-(status_dilate_test&(~status_test)).sum(1)/(2*status_test.sum(1))
        # ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),cont_score_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0)
        # ax.set_ylim([0,1])
        # ax.set_xlim([0,nSes_max])
        # ax.set_xlabel('# sessions active')
        # ax.set_ylabel('$p (\\alpha_{s\pm1} | \\alpha_s)$')#'$p_{\\alpha}^{\pm1}$')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)






        # ax.plot(ICI_stats[:,0]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,0]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,0],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,0],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,8)
        # ax.plot(ICI_stats[:,1]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,1]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,1],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,1],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,12)
        # ax.plot(ICI_stats[:,2]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,2]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,2],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,2],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,16)
        # ax.plot(ICI_stats[:,3]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,3]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,3],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,3],np.linspace(0,nSes,nSes+1))

        # print(ICI_stats)
        # print(IPI_stats)
        # print(IPI*np.arange(self.data['nSes']))
        # print((IPI*np.arange(self.data['nSes'])).sum())
        status_act = status_act[:,s_bool]
        status_PC = status_PC[:,s_bool]
        status = status[:,s_bool]
        recurr = np.zeros((nSes_good,nSes_good))*np.NaN
        N_active = status_act.sum(0)
        # session_bool = np.pad(s_bool[1:],(0,1),constant_values=False) & np.pad(s_bool[:],(0,0),constant_values=False)

        for s in range(nSes_good):#np.where(s_bool)[0]:
            overlap = status[status[:,s],:].sum(0).astype('float')
            N_ref = N_active if mode=='act' else status_act[status_PC[:,s],:].sum(0)
            recurr[s,1:nSes_good-s] = (overlap/N_ref)[s+1:]


        recurr_test = np.zeros((nSes_good,nSes_good))*np.NaN
        N_active_test = status_test.sum(0)
        tmp = []
        for s in range(nSes_good):
            # overlap_act_test = status_test[status_test[:,s],:].sum(0).astype('float')
            overlap_test = status_test[status_test[:,s],:].sum(0).astype('float')
            N_ref = N_active_test if mode=='act' else status_act_test[status_PC_test[:,s],:].sum(0)
            recurr_test[s,1:nSes_good-s] = (overlap_test/N_ref)[s+1:]
            if (~np.isnan(recurr_test[s,:])).sum()>1:
                tmp.append(recurr_test[s,~np.isnan(recurr_test[s,:])])

        # print(tmp)
        # res = sstats.f_oneway(*tmp)
        # print(res)
        # ax = plt.subplot(2,4,8)
        rec_mean = np.nanmean(np.nanmean(recurr,0))
        rec_var = np.sqrt(np.nansum(np.nanvar(recurr,0))/(recurr.shape[1]-1))

        print(rec_mean)
        print(rec_var)

        if mode=='act':
            ax_sketch = plt.axes([0.675,0.875,0.15,0.1])
            ax_sketch2 = plt.axes([0.85,0.875,0.15,0.1])
            # pl_dat.add_number(fig,ax_sketch,order=1,offset=[-40,10])
            pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/sketches/ds1.png'
            if os.path.exists(pic_path):
                ax_sketch.axis('off')
                im = mpimg.imread(pic_path)
                ax_sketch.imshow(im)
                ax_sketch.set_xlim([0,im.shape[1]])
            
            pic2_path = '/home/wollex/Data/Science/PhD/Thesis/pics/sketches/ds3.png'
            if os.path.exists(pic2_path):
                ax_sketch2.axis('off')
                im2 = mpimg.imread(pic2_path)
                ax_sketch2.imshow(im2)
                ax_sketch2.set_xlim([0,im2.shape[1]])

            ax = plt.axes([0.775,0.65,0.2,0.155])
            self.pl_dat.add_number(fig,ax,order=4,offset=[-250,250])
        else:
            ax = plt.axes([0.55,0.65,0.1,0.05])

            nAct = self.status['activity'][...,1].sum(1)
            nPC = self.status['activity'][...,2].sum(1)
            rate = nPC/nAct
            mean_r = np.zeros((self.data['nSes'],3))*np.NaN
            tmp = []
            print('get CI from bootstrapping')
            for i in range(1,self.data['nSes']):
                if np.any(nAct==i):
                    mean_r[i,0] = rate[nAct==i].mean()
                    mean_r[i,1:] = np.percentile(rate[nAct==i],[15.8,84.1])

            count = np.zeros(self.data['nSes'])
            for item in Counter(status_alt[self.status['activity'][...,2]]).items():
                count[item[0]] = item[1]

            La_sessions = IPI_test*np.arange(len(IPI_test))
            pb = np.nanmean(self.status['activity'][self.status['clusters'],:,2].sum(0)/self.status['activity'][self.status['clusters'],:,1].sum(0))
            ax.plot([0,80],[pb,pb],'k--')
            ax.plot(gauss_smooth(count[:len(IPI_test)]/La_sessions,1),label='$p(\\beta^+| \in \mathcal{L}_{\\alpha})$')
            self.pl_dat.plot_with_confidence(ax,range(self.data['nSes']),mean_r[:,0],mean_r[:,1:].T,col='r',label='$p(\\beta^+| \in N_{\\alpha})$')
            ax.set_xlim([0,nSes_good])
            ax.set_ylim([0,0.5])
            ax.set_ylabel('p',fontsize=8)
            ax.set_xlabel('$N_{\\alpha} / \mathcal{L}_{\\alpha}$',fontsize=8)
            ax.xaxis.set_label_coords(0.3,-0.6)
            ax.legend(fontsize=6,loc='lower right',bbox_to_anchor=[1.35,0.9],handlelength=1)
            self.pl_dat.remove_frame(ax,['top','right'])

            ax = plt.axes([0.775,0.65,0.2,0.275])
            self.pl_dat.add_number(fig,ax,order=4,offset=[-150,50])



        p = status.sum()/(nSes_good*nC_good)


        ax.axvline(p,color='k',linestyle='--')
        ax.text(10,p+0.05,'$p^{(0)}_{\\%s^+}$'%(state_label),fontsize=8)
        SD = 1
        # ax.plot([1,nSes_good],[rec_mean,rec_mean],'k--',linewidth=0.5)

        self.pl_dat.plot_with_confidence(ax,np.linspace(1,nSes_good,nSes_good),np.nanmean(recurr,0),SD*np.nanstd(recurr,0),col='k',ls='-',label='emp. data')
        self.pl_dat.plot_with_confidence(ax,np.linspace(1,nSes_good,nSes_good),np.nanmean(recurr_test,0),SD*np.nanstd(recurr_test,0),col='tab:red',ls='-',label='rnd. data')
        ax.set_ylim([0,1])
        ax.set_xlabel('$\Delta$ sessions')
        ax.set_ylabel('$p(\\%s^+_{s+\Delta s} | \\%s^+_s)$'%(state_label,state_label))#'p(recurr.)')
        ax.set_xlim([0,nSes_good])
        if mode=='act':
            ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.05,1.3],handlelength=1)
        else:
            ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.05,0.9],handlelength=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            if mode=='act':
                self.pl_dat.save_fig('act_dynamics')
            elif mode=='PC':
                self.pl_dat.save_fig('PC_dynamics')

        steps = min(nSes_good,40)
        dp_pos = np.zeros((steps,2))*np.NaN
        dp_neg = np.zeros((steps,2))*np.NaN
        dp_pos_test = np.zeros((steps,2))*np.NaN
        dp_neg_test = np.zeros((steps,2))*np.NaN
        # print(get_dp(status,status_act,status_dep=status_dep,ds=ds,mode=mode))
        for ds in range(1,steps):
            status_dep = None if mode=='act' else status_act_test

            dp,_ = get_dp(status,status_dep=status_dep,ds=ds,mode=mode)
            dp_pos[ds,:] = [np.nanmean(dp),np.nanstd(dp)]
            dp_test,_ = get_dp(status_test,status_dep=status_dep,ds=ds,mode=mode)
            dp_pos_test[ds,:] = [np.nanmean(dp_test),np.nanstd(dp_test)]
            # res = sstats.ttest_ind_from_stats(dp_pos[ds,0],dp_pos[ds,1],nC,dp_pos_test[ds,0],dp_pos_test[ds,1],nC,equal_var=True)

            dp = get_dp(~status,status_dep=status_dep,ds=ds,mode=mode)
            dp_neg[ds,:] = [np.nanmean(dp),np.nanstd(dp)]
            dp_test = get_dp(~status_test,status_dep=status_dep,ds=ds,mode=mode)
            dp_neg_test[ds,:] = [np.nanmean(dp_test),np.nanstd(dp_test)]


        plt.figure()
        ax = plt.subplot(211)
        self.pl_dat.plot_with_confidence(ax,range(steps),dp_pos[:,0],dp_pos[:,1],col='k',ls='-')
        self.pl_dat.plot_with_confidence(ax,range(steps),dp_pos_test[:,0],dp_pos_test[:,1],col='r',ls='-')
        # plt.plot(dp_pos,'k')
        # plt.plot(dp_pos_test,'r')

        ax = plt.subplot(212)
        self.pl_dat.plot_with_confidence(ax,range(steps),dp_neg[:,0],dp_neg[:,1],col='k',ls='--')
        self.pl_dat.plot_with_confidence(ax,range(steps),dp_neg_test[:,0],dp_neg_test[:,1],col='r',ls='--')
        # plt.plot(dp_neg,'k--')
        # plt.plot(dp_neg_test,'r--')
        plt.show(block=False)


        # plt.figure()
        # plt.subplot(121)
        # plt.plot(dp_pos,dp,'k.')
        # plt.subplot(122)
        # plt.plot(dp_pos_test,dp_test,'r.')
        # plt.show(block=False)
        #plt.figure()
        #plt.scatter(numbers[:,0]+0.5*np.random.rand(nC),numbers[:,1]+0.5*np.random.rand(nC),s=10,marker='.')
        #plt.show(block=False)

        # plt.figure()
        # plt.hist(dp_pos,np.linspace(-1,1,201),color='k',histtype='step',cumulative=True,density=True)
        # plt.hist(dp_pos_test,np.linspace(-1,1,201),color='r',histtype='step',cumulative=True,density=True)
        # plt.show(block=False)
        # plt.figure()
        # plt.hist(dp_neg,np.linspace(-1,1,201),color='k',histtype='step',cumulative=True,density=True)
        # plt.hist(dp_neg_test,np.linspace(-1,1,201),color='r',histtype='step',cumulative=True,density=True)
        # plt.show(block=False)


    def plot_matching_results(self,sv=False):
        

        plt.figure(figsize=(4,2))
        ax1 = plt.subplot(111)
        #plt.figure(figsize=(4,3))
        #ax1 = plt.axes([0.15, 0.5, 0.8, 0.45])
        #ax2 = plt.axes([0.15, 0.2, 0.8, 0.25])


        #active_time = np.zeros(self.data['nSes'])
        #for s in range(self.data['nSes']):
          #if self.status['sessions'][s]:
            #pathSession = pathcat([self.params['pathMouse'],'Session%02d'%(s+1)]);

            #for file in os.listdir(pathSession):
              #if file.endswith("aligned.mat"):
                #pathBH = os.path.join(pathSession, file)

            #f = h5py.File(pathBH,'r')
            #key_array = ['longrunperiod']

            #dataBH = {}
            #for key in key_array:
              #dataBH[key] = np.squeeze(f.get('alignedData/resampled/%s'%key).value)
            #f.close()

            #active_time[s] = dataBH['longrunperiod'].sum()/len(dataBH['longrunperiod']);

        #ax2.plot(t_ses[self.status['sessions']],active_time[self.status['sessions']],color='k')
        ##ax2.plot(t_measures(1:s_end),active_time,'k')
        #ax2.set_xlim([0,t_ses[-1]])
        #ax2.set_ylim([0,1])
        #ax2.set_xlabel('t [h]',fontsize=14)
        #ax2.set_ylabel('active time',fontsize=14)

        #ax1.plot(t_ses[self.status['sessions']],np.ones(self.status['sessions'].sum())*nC,color='k',linestyle=':',label='# neurons')
        t_ses = np.arange(self.data['nSes'])
        ax1.scatter(t_ses[self.status['sessions']],self.status['activity'][:,self.status['sessions'],1].sum(0), s=20,color='k',marker='o',facecolor='none',label='# active neurons')
        ax1.set_ylim([0,self.data['nC']*1.2])
        ax1.set_xlim([0,t_ses[-1]])
        ax1.legend(loc='upper right')

        ax1.scatter(t_ses[self.status['sessions']],self.status['activity'][:,self.status['sessions'],2].sum(0),s=20,color='k',marker='o',facecolors='k',label='# place cells')

        ax2 = ax1.twinx()
        ax2.plot(t_ses[self.status['sessions']],self.status['activity'][:,self.status['sessions'],2].sum(0)/self.status['activity'][:,self.status['sessions'],1].sum(0),'r')
        ax2.set_ylim([0,0.7])
        ax2.yaxis.label.set_color('red')
        ax2.tick_params(axis='y',colors='red')
        ax2.set_ylabel('fraction PCs')

        ax1.set_xlim([0,t_ses[-1]])
        ax1.set_xlabel('session s',fontsize=14)
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.show(block=False)

        #print(self.status['activity'][:,self.status['sessions'],2].sum(0)/self.status['activity'][:,self.status['sessions'],1].sum(0))
        if sv:
            self.pl_dat.save_fig('neuron_numbers')


    
    def plot_recurrence(self,sv=False,n_processes=4):

        #plt.figure()
        #ax1 = plt.axes([0.2,0.3,0.75,0.65])
        #ax1.plot([0,self.data['nSes']],[0.75,0.75],color='k',linestyle=':')
        t_ses = np.arange(self.data['nSes'])
        recurrence = {'active': {'all':               np.zeros((self.data['nSes'],self.data['nSes']))*np.NaN,
                                 'continuous':        np.zeros((self.data['nSes'],self.data['nSes']))*np.NaN,
                                 'overrepresentation':np.zeros((self.data['nSes'],self.data['nSes']))*np.NaN},
                      'coding': {'all':               np.zeros((self.data['nSes'],self.data['nSes']))*np.NaN,
                                 'ofactive':          np.zeros((self.data['nSes'],self.data['nSes']))*np.NaN,
                                 'continuous':        np.zeros((self.data['nSes'],self.data['nSes']))*np.NaN,
                                 'overrepresentation':np.zeros((self.data['nSes'],self.data['nSes']))*np.NaN}}

        N = {'active': self.status['activity'][:,:,1].sum(0),
             'coding': self.status['activity'][:,:,2].sum(0)}
        L=10#00

        #for s in tqdm(range(self.data['nSes'])):#min(30,self.data['nSes'])):
        if n_processes>1:
            pool = get_context("spawn").Pool(n_processes)
            res = pool.starmap(get_overlap,zip(range(self.data['nSes']),itertools.repeat((self.status['activity'],N,L))))
        # print(res)
        for (s,r) in enumerate(res):
            for pop in r.keys():
                for key in r[pop].keys():
                    recurrence[pop][key][s,:] = r[pop][key]

        #for pop in recurrence.keys():
          #for key in recurrence[pop].keys():
            #recurrence[pop][key][:,~self.status['sessions']] = np.NaN
            #recurrence[pop][key][~self.status['sessions'],:] = np.NaN

        #print(recurrence['active']['all'])
        #for s in tqdm(range(self.data['nSes'])):#min(30,self.data['nSes'])):

        #recurrence['active']['all'][s,np.where(~self.status['sessions'][s:])[0]] = np.NaN
        #start_recurr = np.zeros(self.data['nSes'])*np.NaN
        #for s in range(self.data['nSes']-1):
          #if self.status['sessions'][s] and self.status['sessions'][s+1]:
            #start_recurr[s] = self.status['activity'][self.status['activity'][:,s,2],s+1,2].sum()/self.status['activity'][:,s,2].sum()

        #plt.figure()
        #plt.plot(pl_dat.n_edges,start_recurr)#recurrence['active']['all'][:,1])
        #plt.show(block=False)

        f,axs = plt.subplots(2,2,figsize=(10,4))

        axs[1][0].axhline(0,color=[0.8,0.8,0.8],linestyle='--')
        axs[1][1].axhline(0,color=[0.8,0.8,0.8],linestyle='--')

        axs[0][0].scatter(self.pl_dat.n_edges,recurrence['active']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        axs[0][0].scatter(self.pl_dat.n_edges,recurrence['active']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')

        axs[0][1].scatter(self.pl_dat.n_edges,recurrence['coding']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        axs[0][1].scatter(self.pl_dat.n_edges,recurrence['coding']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')

        for s in range(self.data['nSes']):
            axs[0][0].scatter(self.pl_dat.n_edges,recurrence['active']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
            axs[0][0].scatter(self.pl_dat.n_edges,recurrence['active']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

            axs[0][1].scatter(self.pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
            axs[0][1].scatter(self.pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

            axs[1][0].scatter(self.pl_dat.n_edges,recurrence['active']['overrepresentation'][s,:],5,color=[0.8,0.8,0.8],marker='o')
            axs[1][1].scatter(self.pl_dat.n_edges,recurrence['coding']['overrepresentation'][s,:],5,color=[0.8,0.8,0.8],marker='o')

        axs[0][0].plot(self.pl_dat.n_edges,np.nanmean(recurrence['active']['all'],0),color='k')

        axs[0][0].legend(loc='lower right',fontsize=12)

        axs[1][0].plot(self.pl_dat.n_edges,np.nanmean(recurrence['active']['overrepresentation'],0),color='k')
        axs[0][1].plot(self.pl_dat.n_edges,np.nanmean(recurrence['coding']['all'],0),color='k')
        axs[1][1].plot(self.pl_dat.n_edges,np.nanmean(recurrence['coding']['overrepresentation'],0),color='k')

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
            self.pl_dat.save_fig('ROI_stability')

        plt.figure(figsize=(5,2.5))
        ax = plt.subplot(111)
        #ax.scatter(pl_dat.n_edges,recurrence['active']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        #ax.scatter(pl_dat.n_edges,recurrence['active']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
        self.pl_dat.plot_with_confidence(ax,self.pl_dat.n_edges-1,np.nanmean(recurrence['active']['all'],0),1.96*np.nanstd(recurrence['active']['all'],0),col='k',ls='-',label='recurrence of active cells')
        #for s in range(self.data['nSes']):
          #ax.scatter(pl_dat.n_edges,recurrence['active']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #ax.scatter(pl_dat.n_edges,recurrence['active']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')
        #ax.plot(pl_dat.n_edges-1,np.nanmean(recurrence['active']['all'],0),color='k')
        ax.legend(loc='upper right',fontsize=10)
        ax.set_xlim([0,t_ses[-1]])
        ax.set_ylim([0,1])
        ax.set_ylabel('fraction',fontsize=14)
        ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig('ROI_stability_activity')

        plt.figure(figsize=(5,2.5))
        ax = plt.subplot(111)
        #ax.scatter(pl_dat.n_edges,recurrence['coding']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        #ax.scatter(pl_dat.n_edges,recurrence['coding']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
        self.pl_dat.plot_with_confidence(ax,self.pl_dat.n_edges-1,np.nanmean(recurrence['coding']['ofactive'],0),1.0*np.nanstd(recurrence['coding']['ofactive'],0),col='k',ls='-',label='recurrence of place cells (of active)')
        ax.plot(self.pl_dat.n_edges-1,np.nanmean(recurrence['coding']['all'],0),'k--',label='recurrence of place cells')
        #for s in range(self.data['nSes']):
          #ax.scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #ax.scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
          #axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')
        #ax.plot(pl_dat.n_edges,np.nanmean(recurrence['coding']['all'],0),color='k')
        ax.legend(loc='upper right',fontsize=10)
        ax.set_xlim([0,t_ses[-1]])
        ax.set_ylim([0,1])
        ax.set_ylabel('fraction',fontsize=14)
        ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig('ROI_stability_PC')


    def plot_stats(self,sv=False):
        print('plotting general statistics of PC and nPCs')

        nSes = self.data['nSes']
        mask_PC = (~self.status['activity'][...,2])
        mask_active = ~(self.status['activity'][...,1]&(~self.status['activity'][...,2]))

        fr_key = 'firingrate'#'firingrate_adapt'#firingrate_adapt'
        ### stats of all (PC & nPC) cells
        plt.figure(figsize=(4,3),dpi=self.pl_dat.sv_opt['dpi'])

        key_arr = ['SNR',fr_key,'Isec_value','MI_value']

        for i,key in enumerate(key_arr):
            print(key)
            ## firingrate
            dat_nPC = np.ma.array(self.stats[key], mask=mask_active, fill_value=np.NaN)
            dat_PC = np.ma.array(self.stats[key], mask=mask_PC, fill_value=np.NaN)

            dat_PC_mean = np.zeros(nSes)*np.NaN
            dat_PC_CI = np.zeros((2,nSes))*np.NaN
            dat_nPC_mean = np.zeros(nSes)*np.NaN
            dat_nPC_CI = np.zeros((2,nSes))*np.NaN
            for s in np.where(self.status['sessions'])[0]:
                dat_PC_s = dat_PC[:,s].compressed()
                # print(dat_PC_s)
                if len(dat_PC_s):
                    dat_PC_mean[s] = np.mean(dat_PC_s)
                    dat_PC_CI[:,s] = np.percentile(dat_PC_s,q=[32.5,67.5])
                
                dat_nPC_s = dat_nPC[:,s].compressed()
                if len(dat_PC_s):
                    dat_nPC_mean[s] = np.mean(dat_nPC_s)
                    dat_nPC_CI[:,s] = np.percentile(dat_nPC_s,q=[32.5,67.5])

            ax = plt.subplot(2,2,i+1)
            self.pl_dat.plot_with_confidence(ax,range(nSes),dat_nPC_mean,dat_nPC_CI,col='k',ls='-',label=None)
            self.pl_dat.plot_with_confidence(ax,range(nSes),dat_PC_mean,dat_PC_CI,col='tab:blue',ls='-',label=None)


            # dat_bs_nPC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_nPC,N_bs)
            # dat_bs_PC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_PC,N_bs)
            # dat_bs_nPC[0][~self.status['sessions']] = np.NaN
            # dat_bs_PC[0][~self.status['sessions']] = np.NaN
            #
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_nPC[0],dat_bs_nPC[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label=None)
            ax.set_ylabel(key)
        ax = plt.subplot(221)
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xticklabels([])

        ax = plt.subplot(222)
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xticklabels([])
        ax.set_ylabel('$\\bar{\\nu}$')

        if nSes > 20:
            s = 20
        else:
            s = 10
        ax = plt.axes([0.8,0.65,0.15,0.075])
        dat_nPC = np.ma.array(self.stats[fr_key], mask=mask_active, fill_value=np.NaN)
        dat_PC = np.ma.array(self.stats[fr_key], mask=mask_PC, fill_value=np.NaN)
        # ax.hist(dat_nPC[:,s][~mask_active[:,s]],np.linspace(0,0.3,21),density=True,facecolor='k',alpha=0.5)
        # ax.hist(dat_PC[:,s][~mask_PC[:,s]],np.linspace(0,0.3,21),density=True,facecolor='tab:blue',alpha=0.5)
        ax.hist(dat_nPC[:,s][~mask_active[:,s]],np.logspace(-2.5,0,21),density=True,facecolor='k',alpha=0.5)
        ax.hist(dat_PC[:,s][~mask_PC[:,s]],np.logspace(-2.5,0,21),density=True,facecolor='tab:blue',alpha=0.5)
        # ax.set_ylim([0,200])
        # ax.set_xticks()
        ax.set_xscale('log')
        ax.set_xlabel('$\\nu$')
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax = plt.subplot(223)
        # ax.set_ylabel('$r_{value}$')
        ax.set_ylabel('$I/sec$')
        ax.set_xlabel('session')

        ax = plt.subplot(224)
        ax.set_ylabel('MI')
        ax.set_xlabel('session')
        ax.set_ylim([0,ax.get_ylim()[1]])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig('neuronStats_nPCvPC')

        # return
        ### stats of PCs
        plt.figure(figsize=(4,3),dpi=self.pl_dat.sv_opt['dpi'])

        mask_fields = self.fields['status']<3

        ax = plt.subplot(2,2,1)
        nPC = self.status['activity'][...,2].sum(0).astype('float')
        nPC[~self.status['sessions']] = np.NaN
        ax.plot(nPC,'tab:blue')
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_ylabel('# PC')

        ax2 = ax.twinx()
        dat = np.ma.array(self.fields['nModes'], mask=mask_PC)
        ax2.plot(dat.mean(0),'k-')
        ax2.set_ylim([1,1.3])

        key_arr = ['width','amplitude','reliability']

        ## field width
        for i,key in enumerate(key_arr):
            print(key)
            if len(self.fields[key].shape)==4:
                dat = np.ma.array(self.fields[key][...,0], mask=mask_fields, fill_value=np.NaN)
            else:
                dat = np.ma.array(self.fields[key], mask=mask_fields, fill_value=np.NaN)

            ax = plt.subplot(2,2,i+2)#axes([0.1,0.6,0.35,0.35])
            dat_mean = np.zeros(nSes)*np.NaN
            dat_CI = np.zeros((4,nSes))*np.NaN
            for s in np.where(self.status['sessions'])[0]:
                dat_s = dat[:,s,:].compressed()
                if len(dat_s):
                    dat_mean[s] = np.mean(dat_s)
                    dat_CI[:,s] = np.percentile(dat_s,q=[2.5,32.5,67.5,97.5])
                # ax.boxplot(dat_s,positions=[s],widths=0.4,whis=[5,95],notch=True,bootstrap=100,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

            # dat_bs = bootstrap_data(lambda x : (np.mean(x,(0,2)),0),dat,N_bs)
            # dat_bs[0][~self.status['sessions']] = np.NaN
            # dat = dat[mask_fields]#[dat.mask] = np.NaN
            # dat[mask_fields] = np.NaN

            # ax.plot(width.mean((0,2)),'k')
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs[0],dat_bs[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat.mean((0,2)),np.percentile(dat.std((0,2))),col='k',ls='-',label=None)

            self.pl_dat.plot_with_confidence(ax,range(nSes),dat_mean,dat_CI[[0,3],:],col='k',ls='-',label=None)
            self.pl_dat.plot_with_confidence(ax,range(nSes),dat_mean,dat_CI[[1,2],:],col='k',ls='-',label=None)
            ax.set_ylim([0,ax.get_ylim()[1]])
            ax.set_ylabel(key)

        ax = plt.subplot(222)
        ax.set_ylabel('$\sigma$')
        ax.set_xticklabels([])

        ax = plt.subplot(223)
        ax.set_ylabel('$A$')
        ax.set_xlabel('session')

        ax = plt.subplot(224)
        ax.set_ylabel('reliability')
        ax.set_xlabel('session')

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig('neuronStats_PC')


    def plot_PC_stats(self,sv=False,N_bs=10):
        
        print('### plot place cell statistics ###')

        fig = plt.figure(figsize=(7,4),dpi=self.pl_dat.sv_opt['dpi'])

        nSes = self.data['nSes']
        nbin = self.data['nbin']

        if nSes > 70:
            s = 70
        else:
            s = 10
        pathLoad = os.path.join(self.paths['sessions'][s+1],self.paths['fileNameCNMF'])
        # ld = loadmat(pathLoad)
        
        ld = load_dict_from_hdf5(pathLoad)
        A = ld['A']#.toarray().reshape(self.params['dims'][0],self.params['dims'][1],-1)
        # Cn = A.sum(1).reshape(self.params['dims'])
        Cn = ld['Cn'].transpose()
        Cn -= Cn.min()
        Cn /= Cn.max()

        # adjust to same reference frame
        # x_grid, y_grid = np.meshgrid(np.arange(0., self.params['dims'][0]).astype(np.float32), np.arange(0., self.params['dims'][1]).astype(np.float32))
        # x_remap = (x_grid - \
        #             self.alignment['shift'][s+1,0] + self.alignment['shift'][s,0] + \
        #             self.alignment['flow'][s+1,:,:,0] - self.alignment['flow'][s,:,:,0]).astype('float32')
        # y_remap = (y_grid - \
        #             self.alignment['shift'][s+1,1] + self.alignment['shift'][s,1] + \
        #             self.alignment['flow'][s+1,:,:,1] - self.alignment['flow'][s,:,:,1]).astype('float32')

        ax_ROI = plt.axes([0.05,0.45,0.3,0.5])
        add_number(fig,ax_ROI,order=1,offset=[-50,25])
        # plot background, based on first sessions
        ax_ROI.imshow(Cn,origin='lower',clim=[0,1],cmap='viridis')

        # plot contours occuring in first and in second session, only, and...
        # plot contours occuring in both sessions (taken from first session)
        idx_act = self.status['activity'][:,s,1] & (~self.status['activity'][:,s,2])
        idx_PC = self.status['activity'][:,s,2]
        c_arr_PC = np.where(idx_PC)[0]

        n_act = self.matching['IDs'][idx_act,s].astype('int')
        n_PC = self.matching['IDs'][idx_PC,s].astype('int')

        twilight = plt.get_cmap('hsv')
        cNorm = colors.Normalize(vmin=0,vmax=100)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm,cmap=twilight)


        if sv:   ## enable, when saving
            [ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=[0.3], linestyles=['dotted']) for a in A[:,n_act].T]

            for c,n in zip(np.where(idx_PC)[0],n_PC):
                a = A[:,n]
                f = np.where(self.fields['status'][c,s,:]>2)[0][0]
                colVal = scalarMap.to_rgba(self.fields['location'][c,s,f,0])
                ax_ROI.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors=[colVal], linewidths=[0.5], linestyles=['solid'])

        cbaxes = plt.axes([0.345,0.75,0.01,0.2])
        cb = fig.colorbar(scalarMap,cax = cbaxes,orientation='vertical')
        cb.set_label('PF location',fontsize=8)

        ax_ROI.plot(np.NaN,np.NaN,'k-',label='PC')
        # ax_ROI.plot(np.NaN,np.NaN,'k--',label='$\\alpha_{s_1}$')
        ax_ROI.plot(np.NaN,np.NaN,'k:',label='nPC')
        # ax_ROI.legend(fontsize=10,bbox_to_anchor=[1.2,1.1],loc='upper right',handlelength=1)

        sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        ax_ROI.add_artist(sbar)
        ax_ROI.set_xticks([])
        ax_ROI.set_yticks([])

        ax = plt.axes([0.525,0.1,0.375,0.225])
        add_number(fig,ax,order=6)
        fields = np.zeros((nbin,nSes))
        for i,s in enumerate(np.where(self.status['sessions'])[0]):
            idx_PC = np.where(self.fields['status'][:,s,:]>=3)
            # fields[s,:] = np.nansum(self.fields['p_x'][:,s,:,:],1).sum(0)
            fields[:,s] = np.nansum(self.fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
        fields /= fields.sum(0)
        fields = gauss_smooth(fields,(2,0))

        im = ax.imshow(fields,origin='lower',aspect='auto',cmap='hot')#,clim=[0,1])
        ax.set_xlim([-0.5,nSes-0.5])

        cbaxes = plt.axes([0.92,0.15,0.01,0.175])
        h_cb = plt.colorbar(im,cax=cbaxes)
        h_cb.set_label('place field \ndensity',fontsize=8)
        h_cb.set_ticks([])

        ax.set_ylim([0,100])
        ax.set_xlabel('session')
        ax.set_ylabel('position')

        # idxes = [range(0,15),range(15,40),range(40,87)]
        # # idxes = [range(0,5),range(5,10),range(10,15)]
        # for (i,idx) in enumerate(idxes):
        #     # print(idx)
        #     ax = plt.axes([0.5,0.475-0.175*i,0.475,0.15])
        #     # ax = plt.subplot(len(idxes),1,i+1)
        #     fields = np.nansum(self.fields['p_x'][:,idx,:,:],2).sum(1).sum(0)
        #     fields /= fields.sum()
        #
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['GT'],width=1,facecolor=[0.8,1,0.8],edgecolor='none')
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['RW'],width=1,facecolor=[1,0.8,0.8],edgecolor='none')
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['PC'],width=1,facecolor=[0.7,0.7,1],edgecolor='none')
        #     #ax.bar(pl_dat.bin_edges,fields)
        #     # ax.hist(self.fields['location'][:,idx,0,0].flatten(),pl_dat.bin_edges-0.5,facecolor='k',width=0.8,density=True,label='Session %d-%d'%(idx[0]+1,idx[-1]+1))
        #
        #     idx_PC = np.where(self.fields['status']>=3)
        #     idx_fields = np.where((idx_PC[1] >= idx[0]) & (idx_PC[1] <=idx[-1]))[0]
        #     cov = self.fields['p_x'][idx_PC[0][idx_fields],idx_PC[1][idx_fields],idx_PC[2][idx_fields],:].sum(0)
        #     ax.bar(pl_dat.bin_edges,cov/cov.sum(),facecolor='k',width=0.9,label='Session %d-%d'%(idx[0]+1,idx[-1]+1))
        #     ax.set_xlim([0,L_track])
        #     ax.set_ylim([0,0.04])#np.nanmax(fields)*1.2])
        #
        #     if i==1:
        #         ax.set_ylabel('% of PC')
        #     else:
        #         ax.set_yticks([])
        #     if not (i==2):
        #         ax.set_xticks([])
        #     ax.legend(fontsize=10,loc='upper right')
        # ax.set_xlabel('position [bins]')


        # print('plot fmap corr vs distance for 1. all PC, 2. all active')
        # ax = plt.axes([0.1,0.1,0.25,0.25])
        # D_ROIs_PC =  sp.spatial.distance.pdist(self.matching['com'][c_arr_PC,s,:]))
        # ax.hist(D_ROIs[mat_PC].flat,np.linspace(0,700,201))

        nsteps = 51
        d_arr = np.linspace(0,50,nsteps)
        mean_corr = np.zeros((nsteps,nSes,2))*np.NaN

        for s in tqdm(np.where(self.status['sessions'])[0]):#range(10,15)):
            D_ROIs = sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.matching['com'][:,s,:]))
            np.fill_diagonal(D_ROIs,np.NaN)

            idx_PC = self.status['activity'][:,s,2]
            # print(idx_PC.sum())
            if idx_PC.sum()>1:
                mat_PC = idx_PC[:,np.newaxis] & idx_PC[:,np.newaxis].T
                D_PCs = D_ROIs[idx_PC,:]
                D_PCs = D_PCs[:,idx_PC]
                NN = np.nanargmin(D_PCs,1)

            C = np.corrcoef(self.stats['firingmap'][:,s,:])
            np.fill_diagonal(C,np.NaN)

            for i in range(nsteps-1):
                idx = (D_ROIs>d_arr[i]) & (D_ROIs<=d_arr[i+1])
                if idx_PC.sum()>0:
                    mean_corr[i,s,0] = np.mean(C[idx & mat_PC])
                mean_corr[i,s,1] = np.mean(C[idx])

        dat_bs_PC = bootstrap_data(lambda x : (np.nanmean(x,0),0),mean_corr[...,0].T,N_bs)
        dat_bs = bootstrap_data(lambda x : (np.nanmean(x,0),0),mean_corr[...,1].T,N_bs)

        ax = plt.axes([0.1,0.125,0.25,0.2])
        add_number(fig,ax,order=2)

        ax.plot(D_ROIs,C,'.',markerfacecolor=[0.6,0.6,0.6],markersize=0.5,markeredgewidth=0)
        # ax.plot(D_PCs[range(n_PC),NN],C[range(n_PC),NN],'g.',markersize=1,markeredgewidth=0)
        ax.plot([0,50],[0,0],'r:',linewidth=0.75)
        # ax.plot(d_arr,np.nanmean(mean_corr[...,0],1),'r-',linewidth=1)
        self.pl_dat.plot_with_confidence(ax,d_arr,dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label='place cells')
        self.pl_dat.plot_with_confidence(ax,d_arr,dat_bs[0],dat_bs[1],col='k',ls='--',label='others')

        # ax.plot(d_arr,np.nanmean(mean_corr[...,1],1),'r--',linewidth=1)
        ax.set_xlim([0,50])
        ax.set_ylim([-0.25,1])
        ax.set_xlabel('d [$\mu$m]')
        ax.set_ylabel('$c_{map(\\nu)}$')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.2,1.2],handlelength=1)

        mask_PC = (~self.status['activity'][...,2])
        mask_active = ~(self.status['activity'][...,1]&(~self.status['activity'][...,2]))


        fr_key = 'oof_firingrate_adapt'#'firingrate_adapt'
        # if not (fr_key in self.stats.keys()):
            # fr_key = 'firingrate'
        ### stats of all (PC & nPC) cells

        key_arr = ['rate','SNR']#,'r_values','MI_value']

        for i,key in enumerate(key_arr):
            ## firingrate
            if key=='SNR':
                mask_PC = (~self.status['activity'][...,2])
                # mask_active = ~(self.status['activity'][...,1]&(~self.status['activity'][...,2]))
                # dat_nPC = np.ma.array(self.stats[key], mask=mask_active, fill_value=np.NaN)
                dat_PC = np.ma.array(self.stats[key], mask=mask_PC, fill_value=np.NaN)
            else:
                mask_PC = ~self.status_fields
            #     # mask_active = ~(self.status['activity'][...,1][...,np.newaxis] & (~self.status_fields))
                # dat_nPC = np.ma.array(self.fields['baseline'][...,0], mask=mask_active, fill_value=np.NaN)
                dat_PC = np.ma.array(self.fields['amplitude'][...,0]/self.fields['baseline'][...,0], mask=mask_PC, fill_value=np.NaN)
                # dat_PC = np.ma.array(self.fields['baseline'][...,0], mask=mask_PC, fill_value=np.NaN)


            dat_PC_mean = np.zeros(nSes)*np.NaN
            dat_PC_CI = np.zeros((2,nSes))*np.NaN
            # dat_nPC_mean = np.zeros(nSes)*np.NaN
            # dat_nPC_CI = np.zeros((2,nSes))*np.NaN
            for s in np.where(self.status['sessions'])[0]:
                dat_PC_s = dat_PC[:,s].compressed()
                if len(dat_PC_s):
                  dat_PC_mean[s] = np.mean(dat_PC_s)
                  dat_PC_CI[:,s] = np.percentile(dat_PC_s,q=[32.5,67.5])#,q=[2.5,97.5])#
                # dat_nPC_s = dat_nPC[:,s].compressed()
                # dat_nPC_mean[s] = np.mean(dat_nPC_s)
                # dat_nPC_CI[:,s] = np.percentile(dat_nPC_s,q=[32.5,67.5])#,q=[2.5,97.5])#

            # ax = plt.axes([0.525+0.2*i,0.775,0.175,0.2])#subplot(2,2,i+1)
            ax = plt.axes([0.525,0.4+0.2*i,0.375,0.125])#subplot(2,2,i+1)
            add_number(fig,ax,order=5-i,offset=[-150,25])
            self.pl_dat.plot_with_confidence(ax,range(nSes),dat_PC_mean,dat_PC_CI,col='tab:blue',ls='-',label='place cells')
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_nPC_mean,dat_nPC_CI,col='k',ls='--',label='others')
            ax.set_xlim([-0.5,nSes-0.5])
            self.pl_dat.remove_frame(ax,['top','right'])
            # dat_bs_nPC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_nPC,N_bs)
            # dat_bs_PC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_PC,N_bs)
            # dat_bs_nPC[0][~self.status['sessions']] = np.NaN
            # dat_bs_PC[0][~self.status['sessions']] = np.NaN
            #
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_nPC[0],dat_bs_nPC[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label=None)
            # ax.set_ylabel(key)
            if i==1:
                ax.set_ylabel('SNR')
            else:
                # ax.set_ylabel('$\\bar{\\nu}$')
                ax.set_ylabel('$A/A_0$')
            ax.set_ylim([0,ax.get_ylim()[1]])
            ax.set_xticklabels([])

        ax = plt.axes([0.525,0.8,0.375,0.15])
        add_number(fig,ax,order=3,offset=[-150,25])
        ax.plot(np.where(self.status['sessions'])[0],self.status['activity'][:,self.status['sessions'],1].sum(0),'o',color='k',markersize=2,label='# active neurons')
        ax.plot(np.where(self.status['sessions'])[0],self.status['activity'][:,self.status['sessions'],2].sum(0),'o',color='tab:blue',markersize=2,label='# place cells')
        ax.set_ylim([0,np.max(self.status['activity'].sum(axis=0)[...,1])*1.2])
        ax.set_xlim([-0.5,nSes-0.5])
        ax.set_xticklabels([])
        ax.set_ylabel('# neurons',fontsize=8)
        self.pl_dat.remove_frame(ax,['top','right'])
        # ax.legend(loc='upper right')
        # ax.set_xlabel('session s',fontsize=14)


        ax2 = ax.twinx()
        ax2.plot(np.where(self.status['sessions'])[0],self.status['activity'][:,self.status['sessions'],2].sum(0)/self.status['activity'][:,self.status['sessions'],1].sum(0),'--',color='tab:blue',linewidth=0.5)
        ax2.set_ylim([0,0.5])
        ax2.yaxis.label.set_color('tab:blue')
        ax2.tick_params(axis='y',colors='tab:blue')
        ax2.set_ylabel('PC fraction',fontsize=8)
        self.pl_dat.remove_frame(ax2,['top','right'])

        plt.tight_layout()
        plt.show(block=False)


        # ax = plt.subplot(221)
        # ax.set_xticklabels([])

        # ax = plt.subplot(222)
        # ax.set_ylim([0,ax.get_ylim()[1]])
        # ax.set_xticklabels([])
        # ax.set_ylabel('$\\bar{\\nu}$')


        plt.tight_layout()

        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig('PC_statistics')

#      overrepr = occupancy(:,1:para.nbin)./(sum(nROI(:,3:5),2)/para.nbin);



    def plot_firingmaps(self):

        nSes = self.data['nSes']
        nbin = self.data['nbin']
        print('### plot firingmap over sessions and over time ###')
        # if nSes>65:
            # s_ref = 50
        # else:
        s_ref = 10
        n_plots = 5
        n_plots_half = (n_plots-1)/2
        # ordered = False

        # if ordered:
            # print('aligned order')
        idxes_tmp = np.where(self.status_fields[:,s_ref,:])
        idxes = idxes_tmp[0]
        sort_idx = np.argsort(self.fields['location'][idxes_tmp[0],s_ref,idxes_tmp[1],0])

        # idxes = np.where(self.status['activity'][:,s_ref,2])[0]
        # sort_idx = np.argsort(np.nanmin(self.fields['location'][self.status['activity'][:,s_ref,2],s_ref,:,0],-1))
        sort_idx_ref = idxes[sort_idx]
        nID_ref = len(sort_idx_ref)
        # else:
            # print('non-aligned order')

        width=0.11
        fig = plt.figure(figsize=(7,5),dpi=self.pl_dat.sv_opt['dpi'])

        ax = plt.axes([0.75,0.05,0.225,0.275])
        pic_path = '/home/wollex/Data/Science/PhD/Thesis/pics/others/status_sketch.png'
        ax.axis('off')
        if os.path.exists(pic_path):
            im = mpimg.imread(pic_path)
            ax.imshow(im)
            ax.set_xlim([0,im.shape[1]])
            self.pl_dat.add_number(fig,ax,order=4,offset=[-75,50])


        ax = plt.axes([0.1,0.525,width,0.4])
        self.pl_dat.add_number(fig,ax,order=1)
        ax = plt.axes([0.1,0.08,width,0.4])
        self.pl_dat.add_number(fig,ax,order=2)
        for (i,s) in enumerate(range(int(s_ref-n_plots_half),int(s_ref+n_plots_half)+1)):
            ax = plt.axes([0.1+i*width,0.525,width,0.4])
            # ax = plt.subplot(2,n_plots+1,i+1)
            idxes_tmp = np.where(self.status_fields[:,s,:])
            idxes = idxes_tmp[0]
            sort_idx = np.argsort(self.fields['location'][idxes_tmp[0],s,idxes_tmp[1],0])
            # idxes = np.where(self.status['activity'][:,s,2])[0]
            # sort_idx = np.argsort(np.nanmin(self.fields['location'][self.status['activity'][:,s,2],s,:,0],-1))
            sort_idx = idxes[sort_idx]
            nID = len(sort_idx)

            firingmap = self.stats['firingmap'][sort_idx,s,:]
            firingmap = gauss_smooth(firingmap,[0,3])
            firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,5])


            title_str = "s"
            ds = s-s_ref
            if ds<0:
                title_str += "%d"%ds
            elif ds>0:
                title_str += "+%d"%ds

            ax.set_title(title_str)

            # ax.plot([self.params['zone_idx']['reward'][0],self.params['zone_idx']['reward'][0]],[1,nID],color='g',linestyle=':',linewidth=3)
            # ax.plot([self.params['zone_idx']['reward'][1],self.params['zone_idx']['reward'][1]],[1,nID],color='g',linestyle=':',linewidth=3)
            if i == 0:
                #ax.plot([self.params['zone_idx']['gate'][0],self.params['zone_idx']['gate'][0]],[1,nID],color='r',linestyle=':',linewidth=3)
                #ax.plot([self.params['zone_idx']['gate'][1],self.params['zone_idx']['gate'][1]],[1,nID],color='r',linestyle=':',linewidth=3)
                #ax.set_xticks(np.linspace(0,nbin,3))
                #ax.set_xticklabels(np.linspace(0,nbin,3))
                ax.set_ylabel('Neuron ID')
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xlim([0,nbin])
            ax.set_ylim([nID,0])

            ax = plt.axes([0.1+i*width,0.08,width,0.4])
            # ax = plt.subplot(2,n_plots+1,i+2+n_plots)
            # if not ordered:

            firingmap = self.stats['firingmap'][sort_idx_ref,s,:]
            firingmap = gauss_smooth(firingmap,[0,3])
            firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            im = ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,5])

            ax.set_xlim([0,nbin])
            if i == 0:
                ax.set_ylabel('Neuron ID')
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            ax.set_ylim([nID_ref,0])

            if i==n_plots_half:
                ax.set_xlabel('Location [bin]')

        cbaxes = plt.axes([0.67,0.725,0.01,0.2])
        cb = fig.colorbar(im,cax = cbaxes,orientation='vertical')
        # cb.set_ticks([0,1])
        # cb.set_ticklabels(['low','high'])
        cb.set_label('$\\nu$',fontsize=10)

        ax = plt.axes([0.825,0.5,0.125,0.45])
        self.pl_dat.add_number(fig,ax,order=3,offset=[-125,30])
        idx_strong_PC = np.where((self.status['activity'][...,2].sum(1)>10) & (self.status['activity'][...,1].sum(1)<70))[0]
        idx_PC = np.random.choice(idx_strong_PC)    ## 28,1081
        print(idx_PC)
        firingmap = self.stats['firingmap'][idx_PC,...]
        firingmap = gauss_smooth(firingmap,[0,3])
        firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
        # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
        ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,5])
        ax.barh(range(nSes),-(self.status['activity'][idx_PC,:,2]*10.),left=-5,facecolor='r')
        # idx_coding = np.where(self.status[idx_PC,:,2])[0]
        # ax.plot(-np.ones_like(idx_coding)*10,idx_coding,'ro')
        ax.set_xlim([-10,nbin])
        ax.set_ylim([nSes,0])
        ax.set_ylabel('Session')
        ax.set_xlabel('Location [bins]')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #plt.set_cmap('jet')
        plt.tight_layout()
        plt.show(block=False)

        # if sv:
            # self.pl_dat.save_fig('PC_mapDynamics')

    def plot_stability_dynamics(self, n_processes = 4, reprocess = False, N_bs=10, sv = False):

        SD = 1.96
        nSes = self.data['nSes']
        nbin = self.data['nbin']
        L_track = 100


        s_bool = np.ones(nSes,'bool')
        s_bool[~self.status['sessions']] = False

        recurrence = {'active': {'all':               np.zeros((nSes,nSes))*np.NaN,
                                 'continuous':        np.zeros((nSes,nSes))*np.NaN,
                                 'overrepresentation':np.zeros((nSes,nSes))*np.NaN},
                      'coding': {'all':               np.zeros((nSes,nSes))*np.NaN,
                                 'ofactive':          np.zeros((nSes,nSes))*np.NaN,
                                 'continuous':        np.zeros((nSes,nSes))*np.NaN,
                                 'overrepresentation':np.zeros((nSes,nSes))*np.NaN}}

        N = {'active': self.status['activity'][:,:,1].sum(0),
             'coding': self.status['activity'][:,:,2].sum(0)}
        L=1#00

        #for s in tqdm(range(nSes)):#min(30,nSes)):
        if n_processes>1:
            pool = get_context("spawn").Pool(n_processes)
            res = pool.starmap(get_overlap,zip(range(nSes),itertools.repeat((self.status['activity'],N,L))))
        for (s,r) in enumerate(res):
            for pop in r.keys():
                for key in r[pop].keys():
                    recurrence[pop][key][s,:] = r[pop][key]


        ### ds = 0
        plt0 = True
        if plt0:
            p_shift = np.zeros(nbin)
            for s in np.where(s_bool)[0]:
                idx_field = np.where(self.status_fields[:,s,:])
                for c,f in zip(idx_field[0],idx_field[1]):
                    roll = round((-self.fields['location'][c,s,f,0]+nbin/2)/L_track*nbin)
                    p_shift += np.roll(self.fields['p_x'][c,s,f,:],roll)
            p_shift /= p_shift.sum()

            PC_idx = np.where(self.status['activity'][...,2])
            N_data = len(PC_idx[0])
            print('N data: %d'%N_data)

            p_ds0,p_cov = fit_shift_model(p_shift)

        ### ds > 0
        p = {'all':     {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'cont':    {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'mix':     {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'discont': {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'silent_mix':  {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN},
             'silent':  {'mean': np.zeros((nSes,4))*np.NaN,
                         'CI':   np.zeros((nSes,2,4))*np.NaN,
                         'std':  np.zeros((nSes,4))*np.NaN}}

        t_start = time.time()
        s1_shifts,s2_shifts,f1,f2 = np.unravel_index(self.compare['pointer'].col,(nSes,nSes,self.params['field_count_max'],self.params['field_count_max']))
        c_shifts = self.compare['pointer'].row

        celltype = 'all'
        if celltype == 'all':
            idx_celltype = self.status['activity'][c_shifts,s1_shifts,2]
        if celltype == 'gate':
            idx_celltype = self.status['activity'][c_shifts,s1_shifts,3]
        if celltype == 'reward':
            idx_celltype = self.status['activity'][c_shifts,s1_shifts,4]

        idx_celltype = idx_celltype & s_bool[s1_shifts] & s_bool[s2_shifts]

        if (not('stability' in vars(self).keys())) | reprocess:

            if n_processes>1:
                pool = get_context("spawn").Pool(n_processes)
                # pool = mp.Pool(n_processes)
                res = pool.starmap(get_shift_distr,zip(range(1,nSes),itertools.repeat(self.compare),itertools.repeat((nSes,nbin,N_bs,idx_celltype))))
                pool.close()
            else:
                res = []
                for ds in range(1,nSes):
                    res.append(get_shift_distr(ds,self.compare,(nSes,nbin,N_bs,idx_celltype)))

            for (ds,r) in enumerate(res):
                for pop in r.keys():
                    for key in r[pop].keys():
                        p[pop][key][ds,...] = r[pop][key]


            self.stability = p
        else:
            p = self.stability
        t_end = time.time()
        print('done - time: %5.3g'%(t_end-t_start))

        fig = plt.figure(figsize=(7,4),dpi=self.pl_dat.sv_opt['dpi'])

        ax_distr = plt.axes([0.075,0.11,0.35,0.325])
        self.pl_dat.add_number(fig,ax_distr,order=2,offset=[-100,50])

        for j,ds in tqdm(enumerate([1,5,10,20,40])):#min(nSes,30)):

            Ds = s2_shifts-s1_shifts
            idx_ds = np.where((Ds==ds) & idx_celltype & s_bool[s1_shifts] & s_bool[s2_shifts])[0]
            N_data = len(idx_ds)
            cdf_shifts_ds = np.zeros((N_data,nbin))

            idx_shifts = self.compare['pointer'].data[idx_ds].astype('int')-1
            shifts_distr = self.compare['shifts_distr'][idx_shifts,:].toarray()
            # for i,_ in enumerate(idx_ds):
            #     roll = round(-shifts[i]+L_track/2).astype('int')
            #     cdf_shifts_ds[i,:] = np.cumsum(np.roll(shifts_distr[i,:],roll))
            #     cdf_shifts_ds[i,:] = np.roll(cdf_shifts_ds[i,:],-roll)

            _, _, _, shift_distr = bootstrap_shifts(fit_shift_model,shifts_distr,N_bs,nbin)

            s1_ds = s1_shifts[idx_ds]
            s2_ds = s2_shifts[idx_ds]
            c_ds = self.compare['pointer'].row[idx_ds]

            idxes = self.compare['inter_coding'][idx_ds,1]==1

            CI = np.percentile(shift_distr,[5,95],0)
            ax_distr.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),shift_distr.mean(0),color=[0.2*j,0.2*j,0.2*j],linewidth=0.5,label='$\Delta$ s = %d'%ds)
            # ax_distr.errorbar(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),shift_distr.mean(0),shift_distr.mean(0)-CI[0,:],CI[1,:]-shift_distr.mean(0),fmt='none',ecolor=[1,0.,0.],elinewidth=0.5)

            # ax_distr.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),F_shifts(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),cluster.stability['all']['mean'][ds,0],cluster.stability['all']['mean'][ds,1],cluster.stability['all']['mean'][ds,2],cluster.stability['all']['mean'][ds,3]),'g',linewidth=2)
        self.pl_dat.remove_frame(ax_distr,['top','right'])

        dx_arr = np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin)
        ax_distr.plot(dx_arr,p_shift,'k--',linewidth=0.5)
        ax_distr.set_xlim([-L_track/2,L_track/2])
        ax_distr.set_ylim([0,0.065])
        ax_distr.set_xlabel('field shift $\Delta \\theta$ [bin]')
        ax_distr.set_ylabel('$\\left \\langle p(\Delta \\theta) \\right \\rangle$')
        ax_distr.set_yticks([])
        ax_distr.legend(loc='upper left',fontsize=8, handlelength=1,bbox_to_anchor=[0.05,1.1])

        N_data = np.zeros(nSes)*np.NaN

        D_KS = np.zeros(nSes)*np.NaN
        N_stable = np.zeros(nSes)*np.NaN
        N_total = np.zeros(nSes)*np.NaN     ### number of PCs which could be stable
        # fig = plt.figure()
        p_rec_alt = np.zeros(nSes)*np.NaN

        for ds in range(1,nSes):#min(nSes,30)):
            Ds = s2_shifts-s1_shifts
            idx_ds = np.where((Ds==ds) & s_bool[s1_shifts] & s_bool[s2_shifts])[0]
            N_data[ds] = len(idx_ds)

            idx_shifts = self.compare['pointer'].data[idx_ds].astype('int')-1
            shifts = self.compare['shifts'][idx_shifts]
            N_stable[ds] = (np.abs(shifts)<(SD*self.stability['all']['mean'][0,2])).sum()
            shifts_distr = self.compare['shifts_distr'][idx_shifts,:].toarray().sum(0)
            shifts_distr /= shifts_distr.sum()

            session_bool = np.pad(self.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(self.status['sessions'][:],(0,0),constant_values=False)
            N_total[ds] = self.status_fields[:,session_bool,:].sum()
            # if ds < 20:
            #     plt.subplot(5,4,ds)
            #     plt.plot(dx_arr,np.cumsum(shifts_distr),'k')
            #     plt.plot(dx_arr,np.cumsum(fun_distr),'r')
            #     plt.title('$\Delta s=%d$'%ds)
            fun_distr = F_shifts(dx_arr,self.stability['all']['mean'][ds,0],self.stability['all']['mean'][ds,1],self.stability['all']['mean'][ds,2],self.stability['all']['mean'][ds,3])
            
            
            D_KS[ds] = np.abs(np.cumsum(shifts_distr)-np.cumsum(fun_distr)).max()

            p_rec_alt[ds] = N_stable[ds]/N_data[ds]
        # plt.show(block=False)
        
        # plt.figure(fig_test.number)
        ax_p1 = plt.axes([0.05,0.825,0.175,0.1])
        ax_p2 = plt.axes([0.05,0.675,0.175,0.1])
        ax_shift1 = plt.axes([0.275,0.825,0.175,0.1])
        ax_shift2 = plt.axes([0.275,0.675,0.175,0.1])
        self.pl_dat.add_number(fig,ax_p1,order=1,offset=[-50,25])
        try:
            c = 5
            p1 = self.fields['p_x'][c,10,0,:]
            p2 = self.fields['p_x'][c,11,0,:]
            ax_p1.plot(p1,color='tab:orange',label='$p(\\theta_s$)')
            ax_p1.plot(p2,color='tab:blue',label='$p(\\theta_{s+\Delta s})$')
            ax_p1.set_xticklabels([])
            ax_p1.legend(fontsize=8,handlelength=1,loc='upper right',bbox_to_anchor=[1.2,1.6])
            self.pl_dat.remove_frame(ax_p1,['top','left','right'])
            ax_p1.set_yticks([])

            _,dp = periodic_distr_distance(p1,p2,nbin,nbin,N_bs=10000,mode='bootstrap')
            ax_shift1.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),dp,'k',label='$p(\Delta \\theta)$')
            ax_shift1.set_xticklabels([])
            ax_shift1.legend(fontsize=8,handlelength=1,loc='upper right',bbox_to_anchor=[1.2,1.6])
            self.pl_dat.remove_frame(ax_shift1,['top','left','right'])
            ax_shift1.set_yticks([])

            p1 = self.fields['p_x'][248,34,1,:]
            p2 = self.fields['p_x'][248,79,0,:]
            ax_p2.plot(p1,color='tab:orange')
            ax_p2.plot(p2,color='tab:blue')
            ax_p2.set_yticks([])
            _,dp = periodic_distr_distance(p1,p2,nbin,nbin,N_bs=10000,mode='bootstrap')
            ax_shift2.plot(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),dp,'k')
            ax_shift2.set_xlabel('field shift $\Delta \\theta$')
            self.pl_dat.remove_frame(ax_shift2,['top','left','right'])
            ax_shift2.set_yticks([])

            ax_p2.set_xlabel('position')
            self.pl_dat.remove_frame(ax_p2,['top','left','right'])
        except:
            pass

        ax_img = plt.axes([0.3,0.3,0.15,0.15])

        x_arr = np.linspace(-49.5,49.5,nbin)
        r = 0.3
        sig=5
        y_arr = F_shifts(x_arr,1-r,r,sig,0)
        #print(y_arr)
        ax_img.fill_between(x_arr,y_arr,color='tab:blue')
        ax_img.fill_between(x_arr,(1-r)/nbin,color='tab:red')
        # ax_img.fill_between([-sig*SD,sig*SD],(1-r)/nbin,0,color='tab:blue',alpha=0.5,facecolor='tab:blue',lw=0)
        # ax_img.fill_betweenx([0,(1-r)/nbin],-sig*SD,sig*SD,color='tab:red')
        plt.plot([-sig*SD,-sig*SD],[0,4*(1-r)/nbin],':',color='tab:blue')
        plt.plot([sig*SD,sig*SD],[0,4*(1-r)/nbin],':',color='tab:blue')

        # img = mpimg.imread('/home/wollex/Data/Science/PhD/Thesis/pics/others/shifthist_theory_0.3.png')
        # ax_img.imshow(img)
        self.pl_dat.remove_frame(ax_img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # x_lim = np.where(cluster.status['sessions'])[0][-1] - np.where(cluster.status['sessions'])[0][0] + 1
        x_lim = np.where(s_bool)[0][-1] - np.where(s_bool)[0][0] + 1
        ax_D = plt.axes([0.6,0.8,0.375,0.13])
        ax_D.plot(range(1,nSes+1),D_KS,'k')
        ax_D.set_xlim([0,x_lim])
        ax_D.set_ylabel('$D_{KS}$')
        ax_D.yaxis.set_label_coords(-0.15,0.5)
        ax_D.set_xticklabels([])
        ax_D.set_ylim([0,0.2])

        self.pl_dat.add_number(fig,ax_D,order=3)

        ax_mu = plt.axes([0.6,0.635,0.375,0.13])
        ax_sigma = plt.axes([0.6,0.46,0.375,0.13])
        ax_r = plt.axes([0.6,0.285,0.375,0.13])

        ax_sigma.plot([0,nSes],[p_ds0[2],p_ds0[2]],linestyle='--',color=[0.6,0.6,0.6])
        ax_sigma.text(10,p_ds0[2]+1,'$\sigma_0$',fontsize=8)
        ax_mu.plot([0,nSes],[0,0],linestyle=':',color=[0.6,0.6,0.6])

        sig_theta = self.stability['all']['mean'][0,2]
        r_random = 2*SD*self.stability['all']['mean'][0,2]/nbin
        ax_r.plot([1,nSes],[r_random,r_random],'--',color='tab:blue',linewidth=0.5)
        ax_r.plot([0,nSes],[0.5,0.5],linestyle=':',color=[0.6,0.6,0.6])

        # pl_dat.plot_with_confidence(ax_mu,range(1,nSes+1),p['all']['mean'][:,3],p['all']['mean'][:,3]+np.array([[-1],[1]])*p['all']['std'][:,3]*SD,'k','-')
        # pl_dat.plot_with_confidence(ax_sigma,range(1,nSes+1),p['all']['mean'][:,2],p['all']['mean'][:,2]+np.array([[-1],[1]])*p['all']['std'][:,2]*SD,'k','-')
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p['all']['mean'][:,1],p['all']['mean'][:,1]+np.array([[-1],[1]])*p['all']['std'][:,1]*SD,'k','-')
        print(p['all']['CI'].shape)
        self.pl_dat.plot_with_confidence(ax_mu,range(1,nSes+1),p['all']['mean'][:,3],p['all']['CI'][...,3].T,'k','-')
        self.pl_dat.plot_with_confidence(ax_sigma,range(1,nSes+1),p['all']['mean'][:,2],p['all']['CI'][...,2].T,'k','-')
        self.pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p['all']['mean'][:,1],p['all']['CI'][...,1].T,'k','-')

        p_corr = np.minimum(1,p['all']['mean'][:,1]+(1-p['all']['mean'][:,1])*(2*SD*p['all']['mean'][0,2]/nbin))
        p_SD = np.sqrt((1-2*SD*p['all']['mean'][0,2]/nbin)**2*p['all']['std'][:,1]**2 + ((1-p['all']['mean'][:,1])*2*SD/nbin)**2 * p['all']['std'][0,2]**2)
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p_corr,p_SD,'tab:blue','-')
        ax_r.plot(range(nSes),p_rec_alt,'-',color='tab:blue')

        # ax_r.plot(range(1,nSes+1),p_corr,'k--')
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p_corr,p_corr+np.array([[-1],[1]])*p['all']['std'][:,1]*SD,'k','-',label='stable place fields (of rec. place cell)')


        ax_mu.set_xlim([0,x_lim])
        ax_mu.set_ylim([-10,10])
        ax_mu.set_xticklabels([])
        ax_mu.set_ylabel('$\mu_{\Delta \\theta}$')
        ax_mu.yaxis.set_label_coords(-0.15,0.5)
        ax_sigma.set_xlim([0,x_lim])
        ax_sigma.set_ylim([0,10])
        ax_sigma.set_xticklabels([])
        ax_sigma.set_ylabel('$\sigma_{\Delta \\theta}$')
        ax_sigma.yaxis.set_label_coords(-0.15,0.5)
        ax_r.set_xlim([0,x_lim])
        ax_r.set_ylim([0.0,1])
        ax_r.set_yticks(np.linspace(0,1,3))
        ax_r.set_yticklabels(np.linspace(0,1,3))
        ax_r.set_xticklabels([])
        # ax_r.set_ylabel('$p(\\gamma_{\Delta s})$')
        ax_r.set_ylabel('$p_{\\gamma}$')
        ax_r.yaxis.set_label_coords(-0.15,0.5)
        self.pl_dat.remove_frame(ax_D,['top','right'])
        self.pl_dat.remove_frame(ax_mu,['top','right'])
        self.pl_dat.remove_frame(ax_sigma,['top','right'])
        self.pl_dat.remove_frame(ax_r,['top','right'])
        # axs[0][1].set_ylim([0,1])

        ax_N = plt.axes([0.6,0.11,0.375,0.13])
        ax_N.plot(N_data,'k',label='total')
        ax_N.plot(N_stable,'tab:blue',label='stable')
        ax_N.set_xlabel('session difference $\Delta s$')
        ax_N.set_xlim([0,x_lim])
        ax_N.set_ylabel('$N_{shifts}$')
        ax_N.yaxis.set_label_coords(-0.15,0.5)
        self.pl_dat.remove_frame(ax_N,['top','right'])
        ax_N.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.0,1.3])
        # print(N_stable/N_total)
        plt.tight_layout()
        plt.show(block=False)

        # plt.figure()
        # plt.plot(range(1,nSes+1),N_stable/N_total,'k--',linewidth=0.5)
        # plt.yscale('log')
        # plt.show(block=False)


        def plot_shift_distr(p,p_std,p_ds0):
            nSes = p.shape[0]
            f,axs = plt.subplots(2,2,figsize=(6,4),sharex=True)
            axs[1][0].plot([0,nSes],[p_ds0[2],p_ds0[2]],linestyle='--',color=[0.6,0.6,0.6])
            axs[1][1].plot([0,nSes],[0,0],linestyle='--',color=[0.6,0.6,0.6])
            for i in range(4):
                self.pl_dat.plot_with_confidence(axs[int(np.floor(i/2))][i%2],range(nSes),p[:,i],p[:,i]+np.array([[-1],[1]])*p_std[:,i]*SD,'k','--')
            #axs[0][1].set_yscale('log')

            axs[0][1].yaxis.set_label_position("right")
            axs[0][1].yaxis.tick_right()

            axs[1][1].yaxis.set_label_position("right")
            axs[1][1].yaxis.tick_right()

            axs[0][0].set_xlim([0,max(20,nSes/2)])
            axs[0][0].set_ylim([0,1])
            axs[0][1].set_ylim([0,1])
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


        # plot_shift_distr(p['all']['mean'],p['all']['std'],p_ds0)
        #for key in p.keys():
          #plot_shift_distr(p[key]['mean'],p[key]['std'],p_ds0)
        if sv:
            self.pl_dat.save_fig('stability_dynamics')

        return
        # plt.figure(figsize=(4,4))
        # ax = plt.axes([0.15,0.725,0.8,0.225])
        # pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,np.nanmean(recurrence['active']['all'],0),SD*np.nanstd(recurrence['active']['all'],0),col='k',ls='-',label='recurrence of active cells')
        # ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])
        # ax.set_xlim([0,t_ses[-1]])
        # ax.set_ylim([0,1.1])
        # ax.set_xticklabels([])
        # ax.set_ylabel('fraction',fontsize=14)
        # #ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # ax = plt.axes([0.15,0.425,0.8,0.225])
        # pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,np.nanmean(recurrence['coding']['ofactive'],0),1.0*np.nanstd(recurrence['coding']['ofactive'],0),col='k',ls='-',label='place cell recurrence (of rec. active)')
        # #ax.plot(pl_dat.n_edges-1,np.nanmean(recurrence['coding']['all'],0),'b--',label='recurrence of place cells')
        # ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])
        # ax.set_xlim([0,t_ses[-1]])
        # ax.set_ylim([0,1.1])
        # ax.set_xticklabels([])
        # ax.set_ylabel('fraction',fontsize=14)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # ax = plt.axes([0.15,0.125,0.8,0.225])
        # p_corr = np.minimum(1,p['all']['mean'][:,1]+(1-p['all']['mean'][:,1])*(2*SD*p['all']['mean'][:,2]/nbin))
        # pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,p_corr,p_corr+np.array([[-1],[1]])*p['all']['std'][:,1]*SD,'k','-',label='stable place fields (of rec. place cell)')
        # ax.set_xlim([0,t_ses[-1]])
        # ax.set_ylim([0,1.1])
        # ax.set_ylabel('fraction',fontsize=14)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        # ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])
        #
        # plt.tight_layout()
        # plt.show(block=False)

        # if sv:
        #     pl_dat.save_fig('stability_dynamics_hierarchy')

        plt.figure(figsize=(4,2))
        ax = plt.subplot(111)
        pl_dat.plot_with_confidence(ax,pl_dat.n_edges-1,p['all']['mean'][:,2],p['all']['mean'][:,2]+np.array([[-1],[1]])*p['all']['std'][:,2]*SD,'k','--')
        ax.set_xlim([0,40])
        ax.set_ylim([0,12])
        ax.set_xlabel('session diff. $\Delta$ s')
        ax.set_ylabel('$\sigma$ [bins]',fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        #ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])

        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('stability_dynamics_width')

        #plot_shift_distr(p['cont']['mean'],p['cont']['std'],p_ds0)
        #plot_shift_distr(p['silent']['mean'],p['silent']['std'],p_ds0)

        #if sv:
          #pl_dat.save_fig('stability_dynamics_cont')
        ##plot_shift_distr(p['mix']['mean'],p['mix']['std'],p_ds0)
        #plot_shift_distr(p['discont']['mean'],p['discont']['std'],p_ds0)
        #if sv:
          #pl_dat.save_fig('stability_dynamics_disc')

        #f,axs = plt.subplots(2,2,figsize=(6,4),sharex=True)
        plt.figure(figsize=(4,2.5))
        ax = plt.subplot(111)
        pl_dat.plot_with_confidence(ax,range(nSes),cluster.stability['cont']['mean'][:,1],cluster.stability['cont']['mean'][:,1]+np.array([[-1],[1]])*cluster.stability['cont']['std'][:,1],'b','--',label='coding')
        pl_dat.plot_with_confidence(ax,range(nSes),cluster.stability['discont']['mean'][:,1],cluster.stability['discont']['mean'][:,1]+np.array([[-1],[1]])*cluster.stability['discont']['std'][:,1],'r','--',label='no coding')
        #pl_dat.plot_with_confidence(ax,range(nSes),cluster.stability['silent']['mean'][:,1],cluster.stability['silent']['mean'][:,1]+np.array([[-1],[1]])*cluster.stability['silent']['std'][:,1]*SD,'g','--',label='silent')
        #ax.set_yscale('log')
        ax.set_ylim([0,1.1])
        ax.set_xlim([0,20])
        ax.set_xlabel('$\Delta$ s',fontsize=14)
        ax.set_ylabel('$r_{stable}$',fontsize=14)
        plt.tight_layout()
        plt.legend(loc='lower right',fontsize=10)
        #plt.title('cont')
        plt.show(block=False)
        if sv:
            pl_dat.save_fig('stability_dynamics_cont_vs_disc')



        maxSes = 6
        print('what are those stable cells coding for?')
        plt.figure(figsize=(5,2.5))

        col_arr = [[0.5,0.5,1],[0.5,0.5,0.5],[1,0.5,0.5],[0.5,1,0.5]]
        label_arr = ['continuous','mixed','non-coding','silent']
        key_arr = ['cont','mix','discont','silent']

        w_bar = 0.2
        nKey = len(key_arr)
        offset_bar = ((nKey+1)%2)*w_bar/2 + (nKey//2 - 1)*w_bar

        for i,key in enumerate(key_arr):

            plt.bar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
            plt.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,cluster.stability[key]['mean'][:maxSes,1],cluster.stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')

        plt.xlabel('session difference $\Delta s$',fontsize=14)
        plt.ylabel('$\%$ stable fields',fontsize=14)
        plt.ylim([0,1.1])
        plt.legend(loc='upper right',ncol=2)
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('intercoding_state')

    def plot_PC_choice(self, sv = False):

        s = 10
        idx_PCs = self.status['activity'][:,:,2]
        idx_fields = np.where(self.status_fields)
        plt.figure(figsize=(4,2.5))
        plt.scatter(self.stats['MI_p_value'][idx_fields[0],idx_fields[1]],self.fields['Bayes_factor'][self.status_fields],color='r',s=5)

        idx_nfields = np.where(~self.status_fields)
        plt.scatter(self.stats['MI_p_value'][idx_nfields[0],idx_nfields[1]],self.fields['Bayes_factor'][~self.status_fields],color=[0.6,0.6,0.6],s=3)
        plt.xlabel('p-value (mutual information)',fontsize=14)
        plt.ylabel('log($Z_{PC}$) - log($Z_{nPC}$)',fontsize=14)
        plt.ylim([-10,200])
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig('PC_choice_s=%d'%s)

    def plot_neuron_movement(self):

        nDisp = 500
        plt.figure()
        ax = plt.subplot(111)#,projection='3d')

        for n in range(nDisp):
            sc = ax.plot(self.matching['com'][n,::5,0],self.matching['com'][n,::5,1],'k-')#,c=range(0,nSes,5))#,cmap='jet')#,cluster.matching['com'][n,::5,2]
        #plt.colorbar(sc)
       
        plt.show(block=False)

    def plot_network_dynamics(self):
        print('### plot dynamics of whole network ###')

        plt.figure()
        for i,s in enumerate(range(0,10)):

            pathLoad = os.path.join(self.paths['sessions'][s],self.paths['fileNameCNMF'])
            # ld = loadmat(pathLoad,variable_names=['S'])
            if os.path.exists(pathLoad):
                ld = load_dict_from_hdf5(pathLoad)
                n_arr = self.matching['IDs'][self.status['activity'][:,s,1],s].astype('int')
        
                print(n_arr)

                if n_arr.size > 0:

                    S = ld['S'][n_arr,:]
                    _,_,S_thr = get_firingrate(S)

                    S_mean = gauss_smooth(S_thr.mean(0),5)

                    S_ft = np.fft.fft(S_mean)
                    # print(S.shape)
                    frequencies = np.arange(8989//2)/600
                    plt.subplot(5,2,i+1)
                    plt.plot(frequencies,S_ft[:8989//2])
                    plt.ylim([0,50])
        plt.show(block=False)
    
    def plot_coding_stats(self, sv = False):


        ### get place field max firing rate
        #for c in range(cluster.params['nC']):
          #for s in range(cluster.data['nSes']):

        print('test field width as well')
        print('test peak firing rate as well')

        nSes = self.data['nSes']

        s_bool = np.zeros(nSes,'bool')
        # s_bool[17:87] = True
        s_bool[0:15] = True
        

        s1,s2,f1,f2 = np.unravel_index(self.compare['pointer'].col,(self.data['nSes'],self.data['nSes'],self.params['field_count_max'],self.params['field_count_max']))
        idx_ds1 = np.where((s2-s1 == 1) & s_bool[s1] & s_bool[s2])[0]

        c_ds1 = self.compare['pointer'].row[idx_ds1]
        s1_ds1 = s1[idx_ds1]
        f1_ds1 = f1[idx_ds1]
        idx_shifts_ds1 = self.compare['pointer'].data[idx_ds1].astype('int')-1
        shifts_ds1 = self.compare['shifts'][idx_shifts_ds1]

        idx_stable_ds1 = np.where(np.abs(shifts_ds1) < 6)[0]
        idx_relocate_ds1 = np.where(np.abs(shifts_ds1) > 12)[0]

        c_stable = c_ds1[idx_stable_ds1]
        s1_stable = s1_ds1[idx_stable_ds1]
        f_stable = f1_ds1[idx_stable_ds1]
        rel_stable = self.fields['reliability'][c_stable,s1_stable,f_stable]
        Isec_stable = self.stats['Isec_value'][c_stable,s1_stable]
        fr_stable = self.stats['firingrate'][c_stable,s1_stable]

        c_relocate = c_ds1[idx_relocate_ds1]
        s1_relocate = s1_ds1[idx_relocate_ds1]
        f_relocate = f1_ds1[idx_relocate_ds1]
        Isec_relocate = self.stats['Isec_value'][c_relocate,s1_relocate]
        fr_relocate = self.stats['firingrate'][c_relocate,s1_relocate]

        idx_loosePC = np.where(np.diff(self.status['activity'][...,2].astype('int'),1)==-1)
        Isec_instable = self.stats['Isec_value'][idx_loosePC]
        fr_instable = self.stats['firingrate'][idx_loosePC]

        idx_nPC = np.where(self.status['activity'][...,1] & ~self.status['activity'][...,2])
        #rel_instable = np.nanmax(cluster.fields['reliability'][idx_loosePC[0],idx_loosePC[1],:],-1)
        Isec_nPC = self.stats['Isec_value'][idx_nPC]
        fr_nPC = self.stats['firingrate'][idx_nPC]


        col_stable = [0,0.5,0]
        plt.figure(figsize=(7,2.5))
        ax = plt.subplot(142)
        rel_relocate = self.fields['reliability'][c_relocate,s1_relocate,f_relocate]
        rel_instable = np.nanmax(self.fields['reliability'][idx_loosePC[0],idx_loosePC[1],:],-1)
        ax.hist(rel_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable)
        ax.hist(rel_relocate,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='b')
        ax.hist(rel_instable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r')
        #plt.hist(rel_nPC,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':')
        #rel_all = cluster.fields['reliability']
        #rel_all[~cluster.status_fields] = np.NaN
        #rel_all = rel_all[cluster.status['activity'][...,2],...]
        #ax.hist(rel_all.flat,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        ax.set_xlabel('reliability [%]',fontsize=14)
        ax.set_xlim([0,1])
        ax.set_yticks([])

        ax = plt.subplot(143)
        MI_nPC = self.stats['MI_value'][idx_nPC]
        MI_stable = self.stats['MI_value'][c_stable,s1_stable]
        MI_instable = self.stats['MI_value'][idx_loosePC]
        MI_relocate = self.stats['MI_value'][c_relocate,s1_relocate]

        # MI_nPC = cluster.stats['Isec_value'][idx_nPC]
        # MI_stable = cluster.stats['Isec_value'][c_stable,s1_stable]
        # MI_instable = cluster.stats['Isec_value'][idx_loosePC]
        # MI_relocate = cluster.stats['Isec_value'][c_relocate,s1_relocate]

        plt.hist(MI_nPC,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':',label='nPC')
        plt.hist(MI_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        plt.hist(MI_instable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        ax.set_xlabel('MI [bits]',fontsize=14)
        ax.set_xlim([0,1])
        ax.set_yticks([])
        ax.legend(fontsize=10,loc='lower right')

        ax = plt.subplot(144)
        # key = 'oof_firingrate_adapt'
        key = 'firingrate'
        nu_nPC = self.stats[key][idx_nPC]
        nu_stable = self.stats[key][c_stable,s1_stable]
        nu_instable = self.stats[key][idx_loosePC]
        nu_relocate = self.stats[key][c_relocate,s1_relocate]
        plt.hist(nu_nPC,np.linspace(0,0.3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':',label='nPC')
        plt.hist(nu_stable,np.linspace(0,0.3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        plt.hist(nu_instable,np.linspace(0,0.3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        ax.set_xlabel('$\\nu$ [Hz]')
        ax.set_xlim([0,0.3])
        ax.set_yticks([])
        ax.legend(fontsize=10,loc='lower right')

        #ax = plt.subplot(133)
        #maxrate_stable = cluster.fields['max_rate'][c_stable,s1_stable,f_stable]
        ##idx_loosePC = np.where(np.diff(cluster.status['activity'][...,2].astype('int'),1)==-1)
        #maxrate_instable = np.nanmax(cluster.fields['max_rate'][idx_loosePC[0],idx_loosePC[1],:],-1)
        #plt.hist(maxrate_stable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        #plt.hist(maxrate_instable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        #ax.set_xlabel('$\\nu_{max}$',fontsize=14)
        #ax.set_xlim([0,20])

        ax = plt.subplot(141)
        width_stable = self.fields['width'][c_stable,s1_stable,f_stable,0]
        #idx_loosePC = np.where(np.diff(cluster.status['activity'][...,2].astype('int'),1)==-1)
        width_instable = np.nanmax(self.fields['width'][idx_loosePC[0],idx_loosePC[1],:,0],-1)
        plt.hist(width_stable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        plt.hist(width_instable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        ax.set_xlabel('$\sigma$ [bins]',fontsize=14)
        ax.set_xlim([0,20])
        ax.set_yticks(np.linspace(0,1,3))
        ax.set_ylabel('cdf',fontsize=14)

        # ax = plt.subplot(144)
        # A_stable = cluster.fields['amplitude'][c_stable,s1_stable,f_stable,0]
        # #idx_loosePC = np.where(np.diff(cluster.status['activity'][...,2].astype('int'),1)==-1)
        # A_instable = np.nanmax(cluster.fields['amplitude'][idx_loosePC[0],idx_loosePC[1],:,0],-1)
        # plt.hist(A_stable,np.linspace(0,40,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        # plt.hist(A_instable,np.linspace(0,40,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        # ax.set_xlabel('$\sigma$ [bins]',fontsize=14)
        # ax.set_xlim([0,40])
        # ax.set_yticks(np.linspace(0,1,3))
        # ax.set_ylabel('cdf',fontsize=14)

        #plt.subplot(224)
        #plt.hist(Isec_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable)
        #plt.hist(Isec_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r')
        #MI_all = cluster.stats['Isec_value']
        #MI_all = MI_all[cluster.status['activity'][...,2]]
        #plt.hist(MI_all.flat,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        #plt.xlabel('I/sec')

        #ax = plt.subplot(131)
        ##a,b = bootstrap_data(lambda x : (np.cumsum(np.histogram(x,np.linspace(0,3,51))[0])/len(x),np.NaN),fr_stable,1000)
        ##pl_dat.plot_with_confidence(ax,np.linspace(0,3,51)[:-1],a,b,col='k',ls='-')
        #plt.hist(fr_nPC,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':',label='nPC')
        #plt.hist(fr_stable,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        ##plt.hist(fr_relocate,np.linspace(0,3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='b')
        #plt.hist(fr_instable,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')

        #fr_all = cluster.stats['firingrate']
        #fr_all = fr_all[cluster.status['activity'][...,2]]
        ##a,b = bootstrap_data(lambda x : (np.cumsum(np.histogram(x,np.linspace(0,3,51))[0])/len(x),np.NaN),fr_all,1000)
        ##pl_dat.plot_with_confidence(ax,np.linspace(0,3,51)[:-1],a,b,col='r',ls='-')
        ##ax.hist(fr_all.flat,np.linspace(0,3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        #ax.set_xlabel('activity [Hz]',fontsize=14)
        #ax.set_ylabel('cdf',fontsize=14)
        #ax.set_xlim([0,5])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig('codingChange_stats')


    def plot_neuron_remapping(self):

        nSes = self.data['nSes']
        nbin = self.data['nbin']
    
        fig = plt.figure()
        ax = plt.subplot(111)

        fields = np.zeros((nbin,nSes))
        for i,s in enumerate(np.where(self.status['sessions'])[0]):
            idx_PC = np.where(self.fields['status'][:,s,:]>=3)
            # fields[s,:] = np.nansum(self.fields['p_x'][:,s,:,:],1).sum(0)
            fields[:,s] = np.nansum(self.fields['p_x'][idx_PC[0],s,idx_PC[1],:],0)
        fields /= fields.max(0)
            # fields[:,s] /= fields[:,s].sum()

        fields = gauss_smooth(fields,(2,2))
        im = ax.imshow(fields,origin='lower',cmap='jet')#,clim=[0,1])
        plt.colorbar(im)
        ax.set_xlabel('session')
        ax.set_ylabel('position [bins]')

        plt.show(block=False)


        # s = 10
        ds = 1
        block_size = 10
        plt.figure(figsize=(7,5),dpi=self.pl_dat.sv_opt['dpi'])
       
        for s in np.where(self.status['sessions'])[0][:-ds]:#range(5,15):
            if (s%block_size)==0:
                if (s//block_size)>0:
                   
                    ax = plt.subplot(3,4,s//block_size)
                    remapping /= remapping.max()/2
                    ax.imshow(remapping,origin='lower',clim=[0,1],cmap='hot')
                    ax.text(5,90,'Sessions %d-%d'%(s-block_size,s),color='w',fontsize=8)
                    # plt.colorbar()
            remapping = np.zeros((nbin,nbin))

            for c in np.where(self.status['clusters'])[0]:
                if self.status['activity'][c,s,2] & self.status['activity'][c,s+ds,2]:
                    for f in np.where(self.fields['status'][c,s,:])[0]:
                        for ff in np.where(self.fields['status'][c,s+ds,:])[0]:
                            remapping[int(self.fields['location'][c,s,f,0]),:] += self.fields['p_x'][c,s+ds,ff,:]


        plt.show(block=False)
        # print(remapping.sum(1))
    #print(np.where(cluster.compare['inter_coding'][:,1]==0)[0])
    #print('search for cases, where the neuron loses its  coding ability -> lower MI / lower fr / ...?')
        
    def plot_neuron_examples(self, sv = False):
        print('### SNR & CNN examples ###')
        if True:
            plt.figure(figsize=(7,5),dpi=self.pl_dat.sv_opt['dpi'])

            nSteps = 11
            SNR_arr = np.linspace(1,11,nSteps)

            margin = 18

            ax = plt.axes([0.1,0.1,0.45,0.85])
            t_arr = np.linspace(0,8989/15,8989)

            s = 1
            pathLoad = os.path.join(self.paths['sessions'][s+1],self.paths['fileNameCNMF'])
            if os.path.exists(pathLoad):
                ld = load_dict_from_hdf5(pathLoad)
                # ld = loadmat(pathLoad,variable_names=['C','A','SNR','CNN'],squeeze_me=True)
                
                offset = 0
                for i in tqdm(range(nSteps-1)):
                    # idx_SNR = np.where((cluster.stats['SNR'][:,s] >= SNR_arr[i]) & (cluster.stats['SNR'][:,s] < SNR_arr[i+1]))
                    idx_SNR = np.where((ld['SNR_comp'] >= SNR_arr[i]) & (ld['SNR_comp'] < SNR_arr[i+1]))
                    n_idx = len(idx_SNR[0])
                    if n_idx > 0:
                        for j in np.random.choice(n_idx,min(n_idx,3),replace=False):
                            # c = idx_SNR[0][j]
                            # n = int(cluster.matching['IDs'][c,s])
                            n = idx_SNR[0][j]
                            C = ld['C'][n,:]/ld['C'][n,:].max()
                            ax.plot(t_arr,-C+offset,linewidth=0.5)
                            # ax.text(600,offset,'%.2f'%cluster.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                            offset += 1#= (nSteps-i)

                    offset += 1
                ax.set_yticks(np.linspace(1,offset-3,nSteps))
                ax.set_yticklabels(['$\\approx %d$'%i for i in SNR_arr])
                ax.set_ylabel('SNR',rotation='horizontal',labelpad=-20,y=1.)
                ax.set_xlabel('time [s]')
                ax.set_ylim([offset-1,-1])
                ax.set_xlim([0,600])
                self.pl_dat.remove_frame(ax,['top','right'])

                nSteps = 9
                CNN_arr = np.linspace(0.,1.,nSteps)
                acom = com(ld['A'],512,512)
                for i in tqdm(range(nSteps-1)):
                    # idx_CNN = np.where((cluster.stats['CNN'][:,s] >= CNN_arr[i]) & (cluster.stats['CNN'][:,s] < CNN_arr[i+1]))
                    idx_CNN = np.where((ld['cnn_preds'] >= CNN_arr[i]) & (ld['cnn_preds'] < CNN_arr[i+1]) & ((ld['A']>0).sum(0)>50) & np.all(acom>10,1) & np.all(acom<500,1))
                    n_idx = len(idx_CNN[0])
                    # print(idx_CNN)
                    if n_idx > 0:
                        for j in np.random.choice(n_idx,min(n_idx,1),replace=False):
                            # c = idx_CNN[0][j]
                            # n = int(cluster.matching['IDs'][c,s])
                            n = idx_CNN[1][j]
                            A = ld['A'][:,n].reshape(512,512).toarray()
                            a_com = com(A.reshape(-1,1),512,512)
                            ax = plt.axes([0.6+(i//(nSteps//2))*0.175,0.75-(i%(nSteps//2))*0.23,0.15,0.21])
                            if i==(nSteps-2):
                                sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
                                ax.add_artist(sbar)
                            A /= A.max()
                            A[A<0.001] = np.NaN
                            ax.imshow(A,cmap='viridis',origin='lower')
                            ax.contour(A, levels=[0.3,0.6,0.9], colors='w', linewidths=[0.5], linestyles=['dotted','dashed','solid'])

                            (x_ref,y_ref) = a_com[0]# print(x_ref,y_ref)
                            x_lims = [x_ref-margin,x_ref+margin]
                            y_lims = [y_ref-margin,y_ref+margin]
                            # ax.plot(t_arr,C+nSteps-offset)
                            # ax.text(600,nSteps-offset,'%.2f'%cluster.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                            ax.set_xlim(x_lims)
                            ax.set_ylim(y_lims)
                            # ax.text(x_ref,y_ref+5,'$CNN = %.3f$'%cluster.stats['CNN'][c,s],fontsize=8)
                            ax.text(x_ref+2,y_ref+12,'$%.3f$'%ld['cnn_preds'][n],fontsize=8)
                            self.pl_dat.remove_frame(ax)
                            ax.set_xticks([])
                            ax.set_yticks([])
                plt.tight_layout()
                plt.show(block=False)

            if sv:
                self.pl_dat.save_fig('neuron_stat_examples')

        if True:
            s = 1
            margin = 20
            nSteps = 9
            pathLoad = os.path.join(self.paths['sessions'][s],self.paths['fileNameCNMF'])
            if os.path.exists(pathLoad):
                ld1 = load_dict_from_hdf5(pathLoad)
            
                pathLoad = os.path.join(self.paths['sessions'][s+1],self.paths['fileNameCNMF'])
                if os.path.exists(pathLoad):
                    ld2 = load_dict_from_hdf5(pathLoad)
            
                    # pathLoad = pathcat([cluster.params['pathMouse'],'Session%02d/results_redetect.mat'%(s)])
                    # ld1 = loadmat(pathLoad,variable_names=['A'])
                    # pathLoad = pathcat([cluster.params['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
                    # ld2 = loadmat(pathLoad,variable_names=['A'])

                    x_grid, y_grid = np.meshgrid(np.arange(0., self.params['dims'][0]).astype(np.float32), np.arange(0., self.params['dims'][1]).astype(np.float32))
                    x_remap = (x_grid - \
                                self.alignment['shift'][s-1,0] + self.alignment['shift'][s,0] + \
                                self.alignment['flow'][s-1,0,:,:] - self.alignment['flow'][s,0,:,:]).astype('float32')
                    y_remap = (y_grid - \
                                self.alignment['shift'][s-1,1] + self.alignment['shift'][s,1] + \
                                self.alignment['flow'][s-1,1,:,:] - self.alignment['flow'][s,1,:,:]).astype('float32')

                    plt.figure(figsize=(2,4),dpi=self.pl_dat.sv_opt['dpi'])
                    p_arr = np.linspace(0,1,nSteps)
                    for i in tqdm(range(nSteps-1)):
                        idx_p = np.where((self.matching['score'][:,s,0] >= p_arr[i]) & (self.matching['score'][:,s,0] < p_arr[i+1]) & (self.status['activity'][:,s-1,1]))
                        n_idx = len(idx_p[0])
                        if n_idx > 0:
                            c = np.random.choice(idx_p[0])
                            # s = idx_SNR[1][j]
                            n1 = int(self.matching['IDs'][c,s-1])
                            n2 = int(self.matching['IDs'][c,s])


                            ax = plt.axes([0.05+(i//(nSteps//2))*0.45,0.75-(i%(nSteps//2))*0.23,0.4,0.2])
                            # ax = plt.axes([0.7,0.8-0.2*]])
                            # for j in np.random.choice(n_idx,min(n_idx,3),replace=False):
                            # offset += 1#= (nSteps-i)
                            A1 = ld1['A'][:,n1].reshape(512,512).toarray()
                            A1 = cv2.remap(A1, x_remap,y_remap, cv2.INTER_CUBIC)
                            A2 = ld2['A'][:,n2].reshape(512,512).toarray()

                            a_com = com(A2.reshape(-1,1),512,512)

                            ax.contour(A1/A1.max(), levels=[0.3,0.6,0.9], colors='k', linewidths=[0.5], linestyles=['dotted','dashed','solid'])
                            ax.contour(A2/A2.max(), levels=[0.3,0.6,0.9], colors='r', linewidths=[0.5], linestyles=['dotted','dashed','solid'])
                            if i==(nSteps-2):
                                sbar = ScaleBar(530.68/512 *10**(-6),location='lower right',box_alpha=0)
                                ax.add_artist(sbar)

                            (x_ref,y_ref) = a_com[0]# print(x_ref,y_ref)
                            x_lims = [x_ref-margin,x_ref+margin]
                            y_lims = [y_ref-margin,y_ref+margin]
                            # ax.plot(t_arr,C+nSteps-offset)
                            # ax.text(600,nSteps-offset,'%.2f'%cluster.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                            ax.set_xlim(x_lims)
                            ax.set_ylim(y_lims)
                            ax.text(x_ref+2,y_ref+8,'$%.2f$'%self.matching['score'][c,s,0],fontsize=8)
                            self.pl_dat.remove_frame(ax)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # offset += 1
                    plt.tight_layout()
                    plt.show(block=False)

            if sv:
                self.pl_dat.save_fig('neuron_matches_examples')

    def plot_pv_correlations(self):
        print('## plot population vector correlations etc')

        nSes = self.data['nSes']
        nC = self.data['nC']

        fmap = np.ma.masked_invalid(self.stats['firingmap'][self.status['clusters'],:,:])
        print(fmap.shape)

        if False:
            di = 3

            for ds in [1,2,3,5,10,20]:
                session_bool = np.where(np.pad(cluster.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(cluster.status['sessions'][:],(0,0),constant_values=False))[0]
                s_corr = np.zeros(nSes)*np.NaN
                plt.figure(figsize=(3,5))
                ax = plt.axes([0.1,0.6,0.85,0.35])
                for s in tqdm(np.where(session_bool)[0]):
                    corr = np.zeros(nbin)
                    for i in range(nbin):

                        idx = np.zeros(nbin,'bool')
                        idx[max(0,i-di):min(nbin+1,i+di)] = True

                        idx_cells = cluster.status['activity'][cluster.status['clusters'],s,2]
                        corr[i] = np.ma.corrcoef(fmap[idx_cells,s,:][:,idx].mean(-1),fmap[idx_cells,s+ds,:][:,idx].mean(-1))[0,1]


                    if s in [10,20,40,60]:
                        ax.plot(corr)

                    s_corr[s] = corr.mean()

                ax.set_ylim([-0.25,0.75])
                ax = plt.axes([0.1,0.15,0.85,0.35])
                ax.plot(gauss_smooth(s_corr,1,mode='constant'))
                ax.set_ylim([-0.25,0.75])
                plt.title('ds=%d'%ds)
                plt.show(block=False)

        if True:

            fmap = gauss_smooth(self.stats['firingmap'],(0,0,2))
            corr = np.zeros((nC,nSes,nSes))*np.NaN
            for ds in tqdm(range(1,30)):
                session_bool = np.where(np.pad(self.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(self.status['sessions'][:],(0,0),constant_values=False))[0]
                for s in np.where(session_bool)[0]:
                    for n in np.where(self.status['activity'][:,s,2] & self.status['activity'][:,s+ds,2])[0]:
                        corr[n,s,ds] = np.corrcoef(fmap[n,s,:],fmap[n,s+ds,:])[0,1]

            plt.figure(figsize=(7,5),dpi=self.pl_dat.sv_opt['dpi'])
            plt.subplot(121)
            im = plt.imshow(np.nanmean(corr,0),clim=[0,0.5])
            plt.colorbar(im)

            plt.subplot(122)
            plt.plot(np.nanmean(np.nanmean(corr,0),0))
            plt.ylim([0,1])
            plt.show(block=False)
    
    def plot_example_draw(self, sv = False):

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
            self.pl_dat.save_fig('example_draw')

    def plot_corr_pairs(self):

        print('### plot within session correlated pairs ###')

        print('neurons whos place field is activated conjointly')
        print('kinda difficult - cant find anything obvious on first sight')

        nC = self.data['nC']

        plt.figure(figsize=(7,5),dpi=self.pl_dat.sv_opt['dpi'])
        ## find trials

        self.fields['trial_act']

        high_corr = np.zeros(nC)
        for i,s in enumerate(range(0,20),1):
            idx = np.where(self.status_fields)

            idx_s = idx[1] == s
            c = idx[0][idx_s]
            f = idx[2][idx_s]
            trials = self.fields['trial_act'][c,s,f,:self.behavior['trial_ct'][s]]

            trial_corr = np.corrcoef(trials)
            trial_corr[np.tril_indices_from(trial_corr)] = np.NaN

            # idx[0][idx_s]# np.fill_diagonal(trial_corr,np.NaN)
            # print(trial_corr)
            # print(trial_corr.shape)
            idx_high_corr = np.where(trial_corr > 0.5)
            # print(cluster.stats.keys())
            for c1,c2 in zip(idx_high_corr[0],idx_high_corr[1]):
                # print('c: %d, %d'%(c1,c2))
                # print(cluster.matching['com'][c1,s,:])
                # print(cluster.matching['com'][c2,s,:])
                # print(np.linalg.norm(cluster.matching['com'][c1,s,:]-cluster.matching['com'][c2,s,:]))
                # if np.linalg.norm(cluster.matching['com'][c1,s,:]-cluster.matching['com'][c2,s,:])>10:
                high_corr[c1] += 1
                high_corr[c2] += 1

            # plt.subplot(5,4,i)
            # plt.hist(trial_corr.flat,np.linspace(-1,1,51))
        high_corr[high_corr==0] = np.NaN
        plt.hist(high_corr,np.linspace(0,400,51))
        plt.show(block=False)

    def plot_sdep_stability(self, reprocess = False, sv =  False):

        print('get session-dependent stability')

        nSes = self.data['nSes']

        s_bool = np.zeros(nSes,'bool')
        s_bool[17:87] = True
        # s_bool[:] = True
        s_bool[~self.status['sessions']] = False

        xlim = np.where(s_bool)[0][-1] - np.where(s_bool)[0][0]+1

        act_stab_thr = [0.1,0.9]
        r_stab_thr = [0.1,0.5]

        ds = 2
        if (not ('act_stability_temp' in self.stats.keys())) | reprocess:
            self.stats['act_stability_temp'] = get_act_stability_temp(self,ds=ds)
        if (not ('act_stability' in self.stats.keys())) | reprocess:
            self.stats['act_stability'] = get_act_stability(self,s_bool)
        if (not ('field_stability_temp' in self.stats.keys())) | reprocess:
            self.stats['field_stability_temp'] = get_field_stability_temp(self,SD=1.96,ds=ds)

        if (not ('field_stability' in self.stats.keys())):
            self.stats['field_stability'] = get_field_stability(self,SD=1.96)

        # act_clusters = cluster.status['clusters']
        act_clusters = np.any(self.status['activity'][:,s_bool,1],1)
        r_stab = gauss_smooth(self.stats['field_stability_temp'],(0,1))#[act_clusters,:]
        # act_stab = cluster.stats['act_stability_temp'][cluster.status['clusters'],:,1]
        # act_stab = act_stab[cluster.status['clusters'],:]
        act_stab = self.stats['act_stability_temp'][...,1]#[cluster.status['clusters'],:,1]

        nC = self.status['activity'].shape[0]#cluster.status['clusters'].sum()
        nSes_good = s_bool.sum()


        status = self.status['activity'][...,1]#[cluster.status['clusters'],:,1]
        status_dep = None

        dp_pos,p_pos = get_dp(status,status_dep=status_dep,status_session=s_bool,ds=1)

        dp_pos_temp = np.zeros((nC,nSes))*np.NaN
        p_pos_temp = np.zeros((nC,nSes))*np.NaN
        t_start = time.time()
        for s in range(nSes):#np.where(s_bool)[0]:
            s_bool_tmp = np.copy(s_bool)
            s_bool_tmp = np.copy(self.status['sessions'])
            s_bool_tmp[:s] = False
            s_bool_tmp[s+ds:] = False
            dp_pos_temp[:,s],p_pos_temp[:,s] = get_dp(status,status_dep=status_dep,status_session=s_bool_tmp,ds=1)

        # act_stab = p_pos_temp
        # p_pos_temp = act_stab


        fig = plt.figure(figsize=(7,5),dpi=300)

        locmin = LogLocator(base=10.0,subs=(0,1),numticks=8)
        locmaj = LogLocator(base=100.0,numticks=8)

        ax = plt.axes([0.1,0.85,0.125,0.08])
        self.pl_dat.add_number(fig,ax,order=1)
        ax.hist(p_pos[act_clusters],np.linspace(0,1.,21),color='k',label='$r^{\infty}_{\\alpha^+}$')
        ax.set_xlabel('$r^{\infty}_{\\alpha^+}$')
        self.pl_dat.remove_frame(ax,['top','right'])
        ax.set_yscale('log')
        ax.set_ylim([0.7,3000])
        ax.set_ylabel('count')
        # ax.set_yticks([0,1000])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax = plt.axes([0.3,0.85,0.125,0.08])
        ax.hist(np.nanmax(act_stab[act_clusters,:][:,s_bool],1),np.linspace(0,1,21),color='k',label='$max(r^%d_{\\alpha^+})$'%ds)
        ax.set_yscale('log')
        ax.set_ylim([0.7,3000])
        ylim = ax.get_ylim()[1]
        ax.plot([act_stab_thr[0],act_stab_thr[0]],[1,ylim],'--',color='tab:blue',linewidth=0.75)
        ax.plot([act_stab_thr[1],act_stab_thr[1]],[1,ylim],'--',color='tab:red',linewidth=0.75)
        ax.text(x=act_stab_thr[0]-0.05,y=ylim*1.3,s='low',fontsize=6)
        ax.text(x=act_stab_thr[1]-0.05,y=ylim*1.3,s='high',fontsize=6)
        ax.set_xlabel('$max_s(r^%d_{\\alpha^+})$'%ds)
        # ax.legend(fontsize=8,loc='upper right')
        self.pl_dat.remove_frame(ax,['top','right'])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())



        ax = plt.axes([0.6,0.85,0.125,0.08])
        self.pl_dat.add_number(fig,ax,order=5)
        ax.hist(self.stats['field_stability' ][act_clusters],np.linspace(0,1.,21),color='k',label='$r^{\infty}_{\\alpha}$')
        ax.set_xlabel('$r^{\infty}_{\\gamma^+}$')
        self.pl_dat.remove_frame(ax,['top','right'])
        ax.set_ylabel('count')
        ax.set_yscale('log')
        ax.set_ylim([0.7,3000])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax = plt.axes([0.8,0.85,0.125,0.08])
        ax.hist(np.nanmax(r_stab[act_clusters,:][:,s_bool],1),np.linspace(0,1.,21),color='k',label='$max(r^%d_{\\gamma^+})$'%ds)
        print(np.histogram(np.nanmax(r_stab[act_clusters,:][:,s_bool],1),np.linspace(0,1.,21)))
        ax.set_yscale('log')
        ax.set_ylim([0.7,3000])
        ylim = ax.get_ylim()[1]
        ax.plot([r_stab_thr[0],r_stab_thr[0]],[1,ylim],'--',color='tab:blue',linewidth=0.75)
        ax.plot([r_stab_thr[1],r_stab_thr[1]],[1,ylim],'--',color='tab:red',linewidth=0.75)
        ax.text(x=r_stab_thr[0]-0.05,y=ylim*1.3,s='low',fontsize=6)
        ax.text(x=r_stab_thr[1]-0.05,y=ylim*1.3,s='high',fontsize=6)
        ax.set_xlabel('$max_s(r^%d_{\\gamma^+})$'%ds)
        self.pl_dat.remove_frame(ax,['top','right'])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())
        # ax.yaxis.set_minor_locator(MultipleLocator(2))

        # plt.show(block=False)
        # return
        # ax = plt.axes([0.4,0.85,0.1,0.125])
        # ax.hist(dp_pos,np.linspace(-1,1,21),color='k',alpha=0.5)
        # ax.hist(np.nanmax(dp_pos_temp[act_clusters,:][:,s_bool],1),np.linspace(-1,1,21),color='r',alpha=0.5)
        # pl_dat.remove_frame(ax,['top','right'])


        ax = plt.axes([0.1,0.575,0.175,0.125])
        self.pl_dat.add_number(fig,ax,order=2)

        Np = np.zeros((nSes,3))
        for s in range(nSes):
            Np[s,0] = (act_stab[self.status['activity'][:,s,1],s]<act_stab_thr[0]).sum()
            Np[s,2] = (act_stab[self.status['activity'][:,s,1],s]>act_stab_thr[1]).sum()
            Np[s,1] = self.status['activity'][:,s,1].sum() - Np[s,0] - Np[s,2]
            # Np = np.histogram(act_stab[])
        ax.bar(range(nSes),Np[:,0],width=1,color='tab:blue')
        ax.bar(range(nSes),Np[:,1],width=1,bottom=Np[:,:1].sum(1),alpha=0.5,color='k')
        ax.bar(range(nSes),Np[:,2],width=1,bottom=Np[:,:2].sum(1),color='tab:red')
        self.pl_dat.remove_frame(ax,['top','right'])
        ax.set_xlabel('session')
        ax.set_ylabel('neurons')


        ax = plt.axes([0.6,0.575,0.175,0.125])
        self.pl_dat.add_number(fig,ax,order=6)

        Np = np.zeros((nSes,3))
        for s in range(nSes):
            Np[s,0] = (r_stab[self.status['activity'][:,s,1],s]<r_stab_thr[0]).sum()
            Np[s,2] = (r_stab[self.status['activity'][:,s,1],s]>r_stab_thr[1]).sum()
            Np[s,1] = self.status['activity'][:,s,1].sum() - Np[s,0] - Np[s,2]
            # Np = np.histogram(act_stab[])
        ax.bar(range(nSes),Np[:,0],width=1,color='tab:blue')
        ax.bar(range(nSes),Np[:,1],width=1,bottom=Np[:,:1].sum(1),alpha=0.5,color='k')
        ax.bar(range(nSes),Np[:,2],width=1,bottom=Np[:,:2].sum(1),color='tab:red')
        self.pl_dat.remove_frame(ax,['top','right'])
        ax.set_xlabel('session')
        ax.set_ylabel('neurons')



        ax_extremes = plt.axes([0.4,0.575,0.05,0.125])
        # s_arr = np.arange(17,88,1)#[0,5,10,17,25,40,87,97,112]
        # n_int = len(s_arr)-1
        low_p = np.zeros(nSes)*np.NaN
        high_p = np.zeros(nSes)*np.NaN

        for i,s in enumerate(np.where(s_bool)[0]):
            # act_s_range = np.any(cluster.status['activity'][:,s_arr[i]:s_arr[i+1],1],1)
            c_act = self.status['activity'][:,s,1]
            p_pos_hist = act_stab[c_act,s]
            low_p[s] = (p_pos_hist<act_stab_thr[0]).sum()/c_act.sum()
            high_p[s] = (p_pos_hist>act_stab_thr[1]).sum()/c_act.sum()
        ax_extremes.set_ylim([0,0.5])
        ax_extremes.bar(0,np.nanmean(low_p),facecolor='tab:blue')
        ax_extremes.errorbar(0,np.nanmean(low_p),np.nanstd(low_p),color='k')
        ax_extremes.bar(1,np.nanmean(high_p),facecolor='tab:red')
        ax_extremes.errorbar(1,np.nanmean(high_p),np.nanstd(high_p),color='k')
        ax_extremes.set_xticks([0,1])
        ax_extremes.set_xticklabels(['low $r_{\\alpha^+}^%d$'%ds,'high $r_{\\alpha^+}^%d$'%ds],rotation=60,fontsize=8)
        self.pl_dat.remove_frame(ax_extremes,['top','right'])
        ax_extremes.set_ylabel('fraction')

        ax_extremes = plt.axes([0.9,0.575,0.05,0.125])
        # s_arr = np.arange(17,88,1)#[0,5,10,17,25,40,87,97,112]
        # n_int = len(s_arr)-1
        # color_act = iter(plt.cm.get_cmap('Greys')(np.linspace(0,1,n_int+1)))#s_bool.sum())))
        low_p = np.zeros(nSes)*np.NaN
        high_p = np.zeros(nSes)*np.NaN

        for i,s in enumerate(np.where(s_bool)[0]):
            # col = next(color_act)
            # act_s_range = np.any(cluster.status['activity'][:,s_arr[i]:s_arr[i+1],1],1)
            c_act = self.status['activity'][:,s,1]
            p_pos_hist = r_stab[c_act,s]
            low_p[s] = (p_pos_hist<r_stab_thr[0]).sum()/c_act.sum()
            high_p[s] = (p_pos_hist>r_stab_thr[1]).sum()/c_act.sum()
        ax_extremes.set_ylim([0,1])
        ax_extremes.bar(0,np.nanmean(low_p),facecolor='tab:blue')
        ax_extremes.errorbar(0,np.nanmean(low_p),np.nanstd(low_p),color='k')
        print(low_p)
        print(np.nanmean(low_p),np.nanstd(low_p))
        ax_extremes.bar(1,np.nanmean(high_p),facecolor='tab:red')
        ax_extremes.errorbar(1,np.nanmean(high_p),np.nanstd(high_p),color='k')
        print(np.nanmean(high_p),np.nanstd(high_p))
        ax_extremes.set_xticks([0,1])
        ax_extremes.set_xticklabels(['low $r_{\\gamma^+}^%d$'%ds,'high $r_{\\gamma^+}^%d$'%ds],rotation=60,fontsize=8)
        self.pl_dat.remove_frame(ax_extremes,['top','right'])
        ax_extremes.set_ylabel('fraction')

        status = self.status['activity'][...,1]
        status = status[:,s_bool]

        # print(status.sum(1))
        # print(status.shape)
        # print(r_stab.shape)
        # print(act_stab.shape)
        #
        # print((status[np.nanmax(act_stab[:,s_bool],1)>0.9,:].sum(1)))
        # print((status[np.nanmax(r_stab[:,s_bool],1)>0.9,:].sum(1)))

        ax_Na = plt.axes([0.1,0.3,0.35,0.15])
        self.pl_dat.add_number(fig,ax_Na,order=3)
        Na_distr = np.zeros((nSes_good,3))
        # Na_distr[:,0] = np.histogram(status[np.nanmax(act_stab[:,s_bool],1)<0.1,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,1] = np.histogram(status[(np.nanmax(act_stab[:,s_bool],1)>=act_stab_thr[0]) & (np.nanmax(act_stab[:,s_bool],1)<=act_stab_thr[1]),:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,2] = np.histogram(status[np.nanmax(act_stab[:,s_bool],1)>act_stab_thr[1],:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,0] = np.histogram(status[act_clusters,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,0] -= Na_distr[:,1:].sum(1)
        # print(Na_distr)
        # ax_Na.hist(status.sum(1),np.linspace(0,nSes_good,nSes_good+1),facecolor='g')
        ax_Na.bar(range(nSes_good),Na_distr[:,0],width=1,color='tab:blue')
        ax_Na.bar(range(nSes_good),Na_distr[:,1],width=1,bottom=Na_distr[:,:1].sum(1),alpha=0.5,color='k')
        ax_Na.bar(range(nSes_good),Na_distr[:,2],width=1,bottom=Na_distr[:,:2].sum(1),color='tab:red')
        ax_Na.set_xlabel('$N_{\\alpha^+}$')
        ax_Na.set_ylabel('count')
        self.pl_dat.remove_frame(ax_Na,['top','right'])
        ax_Na.set_ylim([0,350])
        ax_Na.set_xlim([0,xlim])

        ax_Na_inset = plt.axes([0.325,0.4,0.075,0.05])
        ax_Na_inset.plot(Na_distr[:,0]/Na_distr.sum(1),color='tab:blue',linewidth=0.5)
        ax_Na_inset.plot(Na_distr[:,2]/Na_distr.sum(1),color='tab:red',linewidth=0.5)
        ax_Na_inset.set_ylim([0,1])
        ax_Na_inset.set_xlabel('$N_{\\alpha^+}$',fontsize=8)
        ax_Na_inset.xaxis.set_label_coords(0.5,-0.8)
        ax_Na_inset.set_ylabel('fraction',fontsize=8)
        self.pl_dat.remove_frame(ax_Na_inset,['top','right'])

        # ax.legend(fontsize=8)

        status = self.status['activity'][...,1]
        status = status[:,s_bool]
        ax_Na = plt.axes([0.6,0.3,0.35,0.15])
        self.pl_dat.add_number(fig,ax_Na,order=7)
        Na_distr = np.zeros((nSes_good,3))
        # Na_distr[:,0] = np.histogram(status[np.nanmax(r_stab[:,s_bool],1)<0.1,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,1] = np.histogram(status[(np.nanmax(r_stab[:,s_bool],1)>=r_stab_thr[0]) & (np.nanmax(r_stab[:,s_bool],1)<=r_stab_thr[1]),:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,2] = np.histogram(status[np.nanmax(r_stab[:,s_bool],1)>r_stab_thr[1],:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,0] = np.histogram(status[act_clusters,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,0] -= Na_distr[:,1:].sum(1)
        # print(Na_distr)

        # ax_Na.hist(status.sum(1),np.linspace(0,nSes_good,nSes_good+1),facecolor='g')
        ax_Na.bar(range(nSes_good),Na_distr[:,0],width=1,color='tab:blue')
        ax_Na.bar(range(nSes_good),Na_distr[:,1],width=1,bottom=Na_distr[:,:1].sum(1),alpha=0.5,color='k')
        ax_Na.bar(range(nSes_good),Na_distr[:,2],width=1,bottom=Na_distr[:,:2].sum(1),color='tab:red')
        ax_Na.set_xlabel('$N_{\\alpha^+}$')
        ax_Na.set_ylabel('count')
        self.pl_dat.remove_frame(ax_Na,['top','right'])
        ax_Na.set_ylim([0,350])
        ax_Na.set_xlim([0,xlim])

        ax_Na_inset = plt.axes([0.75,0.4,0.075,0.05])
        ax_Na_inset.plot(Na_distr[:,0]/Na_distr.sum(1),color='tab:blue',linewidth=0.5)
        ax_Na_inset.plot(Na_distr[:,2]/Na_distr.sum(1),color='tab:red',linewidth=0.5)
        ax_Na_inset.set_ylim([0,1])
        ax_Na_inset.set_ylabel('fraction',fontsize=8)
        ax_Na_inset.set_xlabel('$N_{\\alpha^+}$',fontsize=8)
        ax_Na_inset.xaxis.set_label_coords(0.5,-0.8)
        self.pl_dat.remove_frame(ax_Na_inset,['top','right'])

        status = self.status['activity'][...,2]
        status = status[:,s_bool]
        Na_distr = np.zeros((nSes_good,3))
        # Na_distr[:,0] = np.histogram(status[np.nanmax(r_stab[:,s_bool],1)<0.1,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,1] = np.histogram(status[(np.nanmax(r_stab[:,s_bool],1)>=r_stab_thr[0]) & (np.nanmax(r_stab[:,s_bool],1)<=r_stab_thr[1]),:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,2] = np.histogram(status[np.nanmax(r_stab[:,s_bool],1)>r_stab_thr[1],:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,0] = np.histogram(status.sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:,0] -= Na_distr[:,1:].sum(1)

        ax_Na_inset = plt.axes([0.875,0.4,0.075,0.05])
        ax_Na_inset.plot(Na_distr[:,0]/Na_distr.sum(1),color='tab:blue',linewidth=0.5)
        ax_Na_inset.plot(Na_distr[:,2]/Na_distr.sum(1),color='tab:red',linewidth=0.5)
        ax_Na_inset.set_ylim([0,1])
        ax_Na_inset.set_xlabel('$N_{\\beta^+}$',fontsize=8)
        ax_Na_inset.xaxis.set_label_coords(0.5,-0.8)
        # ax_Na_inset.set_ylabel('fraction',fontsize=8)
        self.pl_dat.remove_frame(ax_Na_inset,['top','right'])

        # ax.legend(fontsize=8)


        ax_act = plt.axes([0.1,0.1,0.35,0.1])
        self.pl_dat.add_number(fig,ax_act,order=4)

        ax_stab = plt.axes([0.6,0.1,0.35,0.1])
        self.pl_dat.add_number(fig,ax_stab,order=8)

        color_t = iter(plt.cm.get_cmap('Greys')(np.linspace(1,0,5)))
        # for ds in [1,3,5]:
        print(ds)
        col = next(color_t)

        # act_stab = get_act_stability_temp(cluster,ds=ds)[...,1]
        # r_stab = gauss_smooth(get_field_stability_temp(cluster,SD=1.96,ds=ds),(0,1))

        p_pos_high = act_stab>act_stab_thr[1]
        p_pos_high_recurr = np.zeros((nSes,nSes))*np.NaN
        for s in np.where(s_bool)[0]:
            p_pos_high_recurr[s,:nSes-s] = p_pos_high[p_pos_high[:,s],s:].sum(0) / p_pos_high[:,s].sum()
            p_pos_high_recurr[s,np.where(~s_bool[s:])[0]] = np.NaN

        # ax_act.plot(np.nanmean(p_pos_high_recurr,0),color=col)
        self.pl_dat.plot_with_confidence(ax_act,np.arange(nSes),np.nanmean(p_pos_high_recurr,0),np.nanstd(p_pos_high_recurr,0),col=col)

        r_stab_high = r_stab>r_stab_thr[1]
        r_stab_high_recurr = np.zeros((nSes,nSes))*np.NaN
        for s in np.where(s_bool)[0]:
            r_stab_high_recurr[s,:nSes-s] = r_stab_high[r_stab_high[:,s],s:].sum(0) / r_stab_high[:,s].sum()
            r_stab_high_recurr[s,np.where(~s_bool[s:])[0]] = np.NaN
        self.pl_dat.plot_with_confidence(ax_stab,np.arange(nSes),np.nanmean(r_stab_high_recurr,0),np.nanstd(r_stab_high_recurr,0),col=col,label='$\delta s = %d$'%ds)


        for axx in [ax_act,ax_stab]:
            axx.set_ylim([0,1])
            axx.set_xlim([0,xlim])
            axx.set_ylabel('overlap')
            axx.set_xlabel('$\Delta sessions$')
            self.pl_dat.remove_frame(axx,['top','right'])
        ax_stab.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.1,1.3])


        # ax = plt.axes([0.6,0.6,0.35,0.25])
        # ax.plot(act_stab+0.05*np.random.rand(nC,nSes_good),act_stab+0.05*np.random.rand(nC,nSes_good),'k.',markersize=1,markeredgecolor='none')

        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig('individuals1')

        return
    

        status_La = np.zeros((nC,nSes,2),'int')
        status_Lb = np.zeros_like(cluster.status['activity'][...,2],'int')

        highCode_thr = 0.5
        IPI_La = np.zeros(nSes)
        IPI_Lb = np.zeros(nSes)
        La_highCode = np.zeros(nSes)

        IPI_La_start = np.zeros_like(cluster.status['activity'][...,1],'bool')
        IPI_Lb_start = np.zeros_like(cluster.status['activity'][...,2],'bool')

        idx_fields = np.where(cluster.status_fields)

        for c in range(nC):
            s0_act = 0
            s0_code = 0
            inAct = False
            inCode = False
            for s in np.where(cluster.status['sessions'])[0]:
                if inAct:
                    if ~cluster.status['activity'][c,s,1]:
                        La = cluster.status['sessions'][s0_act:s].sum()
                        status_La[c,s0_act:s,0] = La
                        status_La[c,s0_act:s,1] = cluster.status['activity'][c,s0_act:s,2].sum()
                        if (cluster.status['activity'][c,s0_act:s,2].sum() / La)>highCode_thr:
                            La_highCode[La] += 1
                        IPI_La[La] += 1
                        inAct=False
                else:
                    if cluster.status['activity'][c,s,1]:
                        s0_act = s
                        inAct = True
                        IPI_La_start[c,s] = True

                if inCode:
                    if ~cluster.status['activity'][c,s,2]:
                        Lb = cluster.status['sessions'][s0_code:s].sum()
                        status_Lb[c,s0_code:s] = Lb
                        IPI_Lb[Lb] += 1
                        inCode=False
                else:
                    if cluster.status['activity'][c,s,2]:
                        s0_code = s
                        inCode = True
                        IPI_Lb_start[c,s] = True

            if inAct:
                La = cluster.status['sessions'][s0_act:s+1].sum()
                status_La[c,s0_act:s+1,0] = La
                status_La[c,s0_act:s+1,1] = cluster.status['activity'][c,s0_act:s+1,2].sum()
                if (cluster.status['activity'][c,s0_act:s,2].sum() / La)>highCode_thr:
                    La_highCode[La] += 1
                IPI_La[La] += 1
            if inCode:
                Lb = cluster.status['sessions'][s0_code:s+1].sum()
                status_Lb[c,s0_code:s+1] = Lb
                IPI_Lb[Lb] += 1

        status_La[:,~cluster.status['sessions'],:] = 0
        status_Lb[:,~cluster.status['sessions']] = 0
        # L_code = status_La[cluster.status['activity'][...,2],0]


        status, status_dep = get_status_arr(cluster)

        # plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        #
        # s_arr = [0,5,10,17,25,40,87,97,112]
        # s_arr = np.arange(0,112,1)#[0,5,10,17,25,40,87,97,112]
        # n_int = len(s_arr)-1
        # ax = plt.subplot(221)
        # s_interval = 10
        # fstab_mean = np.zeros((n_int,2))
        # astab_mean = np.zeros((n_int,2))
        #
        # color_field = iter(plt.cm.get_cmap('rainbow')(np.linspace(0,1,n_int)))
        # color_act = iter(plt.cm.get_cmap('Greys')(np.linspace(0,1,n_int)))
        # for i in range(n_int):
        #     col = next(color_field)
        #     ax.hist(cluster.stats['field_stability'][cluster.status['clusters'],s_arr[i]:s_arr[i+1]].flat,np.linspace(0,1,51),cumulative=True,density=True,histtype='step',color=col)
        #
        #     fstab_mean[i,0] = np.nanmean(cluster.stats['field_stability'][cluster.status['clusters'],s_arr[i]:s_arr[i+1]])
        #     fstab_mean[i,1] = np.nanstd(cluster.stats['field_stability'][cluster.status['clusters'],s_arr[i]:s_arr[i+1]])
        #
        #
        #     col = next(color_act)
        #     ax.hist(cluster.stats['act_stability'][cluster.status['clusters'],s_arr[i]:s_arr[i+1]].flat,np.linspace(0,1,51),cumulative=True,density=True,histtype='step',color=col)
        #
        #     astab_mean[i,0] = np.nanmean(cluster.stats['act_stability'][cluster.status['clusters'],s_arr[i]:s_arr[i+1]])
        #     astab_mean[i,1] = np.nanstd(cluster.stats['act_stability'][cluster.status['clusters'],s_arr[i]:s_arr[i+1]])
        # ax = plt.subplot(222)
        # pl_dat.plot_with_confidence(ax,np.arange(n_int)-0.1,fstab_mean[:,0],fstab_mean[:,1],col='r')
        # pl_dat.plot_with_confidence(ax,np.arange(n_int)+0.1,astab_mean[:,0],astab_mean[:,1],col='k')
        # plt.show(block=False)


        fig = plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])

        ax = plt.subplot(331)
        ax.plot(cluster.status['activity'][...,2].sum(0),'k.',markersize=1)
        # ax.hist(np.nanmax(cluster.stats['field_stability'],1),np.linspace(0,2,101))

        ax = plt.subplot(332)
        idx_med = ((cluster.stats['field_stability'] > 0.5)[...,np.newaxis] & (cluster.status_fields))
        idx_high = ((cluster.stats['field_stability'] > 0.9)[...,np.newaxis] & (cluster.status_fields))
        ax.hist(cluster.fields['location'][idx_med,0],np.linspace(0,100,101),color='tab:blue',density=True,alpha=0.5)
        ax.hist(cluster.fields['location'][idx_high,0],np.linspace(0,100,101),color='tab:red',density=True,alpha=0.5)
        # ax.scatter(np.nanmax(field_stability,1)+0.02*np.random.rand(nC),cluster.stats['p_post_c']['code']['code'][:,1,0,0]+0.02*np.random.rand(nC),s=cluster.status['activity'][...,2].sum(1)/10,c='k',edgecolor='none')

        ax = plt.subplot(333)
        ax.plot(np.nanmean(cluster.stats['field_stability'],0),'r')
        ax.plot(np.nanmean(cluster.stats['act_stability'],0),'k')

        ax = plt.subplot(334)
        ax.plot((cluster.stats['field_stability']>0.5).sum(0)/cluster.status['activity'][...,2].sum(0),'r')
        ax.plot((cluster.stats['act_stability'][...,1]>0.5).sum(0)/cluster.status['activity'][...,1].sum(0),'k')

        ax = plt.subplot(335)
        for thr in np.linspace(0,1,51):
            ax.plot(thr,np.any(cluster.stats['field_stability']>thr,1).sum(),'k.',markersize=1)

        ax = plt.subplot(336)
        La_mean = np.zeros((nSes,3))*np.NaN
        Lb_mean = np.zeros((nSes,3))*np.NaN
        for s in np.where(cluster.status['sessions'])[0]:
            La_mean[s,0] = status_La[status_La[:,s,0]>0,s,0].mean()
            La_mean[s,1:] = np.percentile(status_La[status_La[:,s,0]>0,s,0],[5,95])

            Lb_mean[s,0] = status_Lb[status_Lb[:,s]>0,s].mean()
            Lb_mean[s,1:] = np.percentile(status_Lb[status_Lb[:,s]>0,s],[5,95])

        pl_dat.plot_with_confidence(ax,range(nSes),La_mean[:,0],La_mean[:,1:].T,col='k')
        pl_dat.plot_with_confidence(ax,range(nSes),Lb_mean[:,0],Lb_mean[:,1:].T,col='b')

        ax = plt.subplot(337)
        # print(stat)
        ax.plot(status['stable'][:,:,1].sum(0),'k',linewidth=0.5)
        # ax = plt.subplot(337)
        # ax.plot(status['stable'][:,:,2].sum(0),'b',linewidth=0.5)
        # ax.plot(status['stable'][:,:,3].sum(0),'r',linewidth=0.5)

        ax = plt.subplot(338)
        ax.plot(cluster.stats['p_post_s']['code']['stable'][:,1,0])

        ax = plt.subplot(339)
        ax.hist(status['stable'][cluster.status['clusters'],:,1].sum(1),np.linspace(0,100,101))



        plt.show(block=False)

        # return field_stability

    def plot_field_stats(self):

        nSes = self.data['nSes']
        s_bool = np.zeros(nSes,'bool')
        # s_bool[17:87] = True
        s_bool[0:15] = True

        idx_PF = self.status_fields & s_bool[np.newaxis,:,np.newaxis]

        plt.figure(figsize=(7,5),dpi=300)
        plt.subplot(221)
        plt.plot(self.stats['MI_value'][np.where(idx_PF)[0],np.where(idx_PF)[1]],self.stats['Isec_value'][np.where(idx_PF)[0],np.where(idx_PF)[1]],'k.',markersize=1)

        plt.subplot(223)
        plt.plot(self.fields['reliability'][idx_PF],self.stats['MI_value'][np.where(idx_PF)[0],np.where(idx_PF)[1]],'k.',markersize=1)
        plt.subplot(224)
        plt.plot(self.fields['reliability'][idx_PF],self.stats['Isec_value'][np.where(idx_PF)[0],np.where(idx_PF)[1]],'r.',markersize=1)

        plt.show(block=False)

    def plot_alt_pf_detection(self):

        print('### get alternative place field detection from place maps directly ###')

        nSes = self.data['nSes']
        nbin = self.data['nbin']
        nC = self.data['nC']

        ### smooth firingmap
        ### find, if some part is > 4*SD+baseline
        ### find if part is > 4 bins width
        ### center is place field! (highest only)
        SD = 3
        MI_thr = 1#0.1
        loc = np.zeros((nC,nSes))*np.NaN
        width = np.zeros((nC,nSes))*np.NaN
        fields = np.zeros((nSes,nbin))*np.NaN
        PC_ct = np.zeros(nSes)
        # plt.figure(figsize=(7,5),dpi=300)
        for s in np.where(self.status['sessions'])[0]:
            fmaps = np.copy(self.stats['firingmap'][:,s,:])
            baseline = np.percentile(fmaps,20,axis=1)
            fmap_thr = np.zeros(self.data['nC'])
            # SD =
            fmaps_base_subtracted = fmaps-baseline[:,np.newaxis]
            N = (fmaps_base_subtracted <= 0).sum(1)
            fmaps_base_subtracted *= -1.*(fmaps_base_subtracted<=0)
            noise = np.sqrt((fmaps_base_subtracted**2).sum(1)/(N*(1-2/np.pi)))
            fmap_thr = baseline + SD*noise
            # print(fmap_thr)
            fmaps = gauss_smooth(fmaps,(0,4))
            for c in np.where(self.status['activity'][:,s,1])[0]:
                surp = (fmaps[c,:] >= fmap_thr[c]).sum()
                if (self.stats['MI_p_value'][c,s]<MI_thr) & (surp > 4):
                    loc[c,s] = np.argmax(fmaps[c,:])
                    width[c,s] = surp
                    # print('this is placecell!')
                    PC_ct[s] += 1
            # print('session %d'%s)
            # print('Place cells detected: %d'%PC_ct)
            # print(loc[c,:])
            # plt.subplot(5,5,s+1)
            # plt.title('PCs: %d'%PC_ct)
            # plt.hist(loc[:,s],np.linspace(0,100,41))
            # print(np.histogram(loc[c,np.isfinite(loc[c,:])],np.linspace(0,100,101),density=True)[0])
            fields[s,:] = np.histogram(loc[:,s],np.linspace(0,100,101),density=True)[0]
        # plt.show(block=False)

        plt.figure(figsize=(7,5),dpi=300)
        plt.subplot(211)
        plt.plot(PC_ct,'k')
        plt.subplot(212)
        plt.imshow(fields,aspect='auto',origin='lower',cmap='hot',clim=[0,0.03])
        plt.show(block=False)

        plt.figure(figsize=(7,5),dpi=300)
        # s_arr = [0,17,40,87,97,107]
        s_arr = [0,5,10,15,20]
        n_int = len(s_arr)-1
        color_t = iter(plt.cm.get_cmap('Greys')(np.linspace(0,1,n_int+1)))
        for i in range(n_int):
            col = next(color_t)
            plt.plot(np.nanmean(fields[s_arr[i]:s_arr[i+1],:],0),color=col)
        plt.show(block=False)

    def plot_multi_modes(self, sv = False):

        print('### whats up with multiple peaks? ###')

        nSes = self.data['nSes']
        nbin = self.data['nbin']
        nC = self.data['nC']

        nFields = np.ma.masked_array(self.status_fields.sum(-1),mask=~self.status['activity'][...,2])
        idx = nFields>1
        nMultiMode = nFields.mean(0)

        dLoc = np.zeros((nC,nSes))*np.NaN
        corr = np.zeros((nC,nSes))*np.NaN
        overlap = np.zeros((nC,nSes))*np.NaN
        for (c,s) in zip(np.where(idx)[0],np.where(idx)[1]):
            # pass
            loc = self.fields['location'][c,s,self.status_fields[c,s,:],0]
            dLoc[c,s] = np.abs(np.mod(loc[1] - loc[0]+nbin/2,nbin)-nbin/2)#loc[1]-loc[0]

            idx_loc = np.where(self.status_fields[c,s,:])[0]

            corr[c,s] = np.corrcoef(self.fields['trial_act'][c,s,idx_loc[0],:self.behavior['trial_ct'][s]],self.fields['trial_act'][c,s,idx_loc[1],:self.behavior['trial_ct'][s]])[0,1]

            overlap[c,s] = (self.fields['trial_act'][c,s,idx_loc[0],:self.behavior['trial_ct'][s]] & self.fields['trial_act'][c,s,idx_loc[1],:self.behavior['trial_ct'][s]]).sum()

        fig = plt.figure(figsize=(7,5),dpi=self.pl_dat.sv_opt['dpi'])
        ax = plt.axes([0.1,0.75,0.35,0.175])
        self.pl_dat.add_number(fig,ax,order=1)
        ax.plot(nMultiMode,'k')
        ax.set_ylim([0.98,1.2])
        self.pl_dat.remove_frame(ax,['top','right'])
        ax.set_xlabel('session')
        ax.set_ylabel('$\left \langle \# fields \\right \\rangle$')

        ax = plt.axes([0.55,0.75,0.35,0.175])
        self.pl_dat.add_number(fig,ax,order=2,offset=[-50,50])

        ax.hist(self.fields['location'][nFields==1,:,0].flat,np.linspace(0,100,101),facecolor='k',density=True,label='1 field')
        ax.hist(self.fields['location'][idx,:,0].flat,np.linspace(0,100,101),facecolor='tab:orange',alpha=0.5,density=True,label='2 fields')
        self.pl_dat.remove_frame(ax,['top','right','left'])
        ax.set_yticks([])
        ax.set_xlabel('position [bins]')
        ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[0.85,1.2],handlelength=1)

        ax = plt.axes([0.65,0.1,0.3,0.42])
        self.pl_dat.add_number(fig,ax,order=5)
        ax.plot(dLoc[overlap==0],corr[overlap==0],'k.',markersize=1,zorder=10)
        ax.plot(dLoc[overlap>0],corr[overlap>0],'.',color='tab:red',markersize=1,zorder=12)
        ax.set_xlim([0,50])
        ax.set_ylim([-1,1])
        ax.set_yticks(np.linspace(-1,1,5))
        ax.set_xlabel('$\Delta \\theta [bins]$')
        ax.set_ylabel('$c_a$')

        ax2 = ax.twiny()
        ax2.hist(corr.flat,np.linspace(-1,1,51),orientation='horizontal',facecolor='tab:orange',alpha=0.5,zorder=0)
        ax2.set_xlim([0,ax2.get_xlim()[1]*4])
        ax2.set_xticks([])

        ax3 = ax.twinx()
        ax3.hist(dLoc.flat,np.linspace(0,50,51),orientation='vertical',facecolor='tab:orange',alpha=0.5,zorder=0)
        ax3.set_ylim([ax3.get_ylim()[1]*4,0])
        ax3.set_yticks([])

        ### plot "proper" 2-field
        idx = np.where(dLoc>10)
        i = np.random.randint(len(idx[0]))
        c = idx[0][i]
        s = idx[1][i]
        # c,s = [12,73]
        print(c,s)

        ax_fmap = plt.axes([0.1,0.4,0.35,0.15])
        self.pl_dat.add_number(fig,ax_fmap,order=3)
        ax_fmap.bar(np.linspace(1,nbin,nbin),gauss_smooth(self.stats['firingmap'][c,s,:],1),width=1,facecolor='k')
        ax_fmap.set_ylabel('$\\bar{\\nu}$')

        loc = self.fields['location'][c,s,self.status_fields[c,s,:],0]
        ax_trial = plt.axes([0.375,0.525,0.125,0.1])
        idx_loc = np.where(self.status_fields[c,s,:])[0]
        self.pl_dat.remove_frame(ax_fmap,['top','right'])

        col_arr = ['tab:green','tab:blue']
        for i,f in enumerate(idx_loc):
            ax_fmap.plot(loc[i],1,'v',color=col_arr[i],markersize=5)
            ax_trial.bar(range(self.behavior['trial_ct'][s]),self.fields['trial_act'][c,s,f,:self.behavior['trial_ct'][s]],bottom=i,color=col_arr[i],alpha=0.5)

        ax_fmap.arrow(x=loc.min(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.max()-loc.min(),dy=0,shape='full',color='tab:orange',width=0.02,head_width=0.4,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.arrow(x=loc.max(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.min()-loc.max(),dy=0,shape='full',color='tab:orange',width=0.02,head_width=0.4,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.text(loc.min()/2+loc.max()/2,ax_fmap.get_ylim()[1],'$\Delta \\theta$',color='tab:orange',fontsize=10,ha='center')

        self.pl_dat.remove_frame(ax_trial,['top','right','left'])
        ax_trial.set_yticks([])
        ax_trial.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_trial.set_xlabel('trial',fontsize=10)#,labelpad=-5,x=0.4)
        ax_trial.xaxis.set_label_coords(0.3,-0.3)
        ax_trial.text(-10,1.2,s='$c_a=%.2f$'%corr[c,s],fontsize=6)

        ### plot "improper" 2-field
        idx = np.where(dLoc<10)
        i = np.random.randint(len(idx[0]))
        c = idx[0][i]
        s = idx[1][i]
        # c,s = [490,44]
        print(c,s)

        ax_fmap = plt.axes([0.1,0.1,0.35,0.15])
        self.pl_dat.add_number(fig,ax_fmap,order=4)
        ax_fmap.bar(np.linspace(1,nbin,nbin),gauss_smooth(self.stats['firingmap'][c,s,:],1),width=1,facecolor='k')
        self.pl_dat.remove_frame(ax_fmap,['top','right'])
        ax_fmap.set_ylabel('$\\bar{\\nu}$')
        ax_fmap.set_xlabel('position [bins]')

        loc = self.fields['location'][c,s,self.status_fields[c,s,:],0]
        ax_trial = plt.axes([0.375,0.225,0.125,0.1])
        idx_loc = np.where(self.status_fields[c,s,:])[0]
        for i,f in enumerate(idx_loc):

            ax_fmap.plot(loc[i],1,'v',color=col_arr[i],markersize=5)
            ax_trial.bar(range(self.behavior['trial_ct'][s]),self.fields['trial_act'][c,s,f,:self.behavior['trial_ct'][s]],bottom=i,color=col_arr[i],alpha=0.5)

        ax_fmap.arrow(x=loc.min(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.max()-loc.min(),dy=0,shape='full',color='tab:orange',width=0.015,head_width=0.2,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.arrow(x=loc.max(),y=ax_fmap.get_ylim()[1]*0.95,dx=loc.min()-loc.max(),dy=0,shape='full',color='tab:orange',width=0.015,head_width=0.2,head_length=2,length_includes_head=True)#"$\Delta \\theta$",
        ax_fmap.text(loc.min()/2+loc.max()/2,ax_fmap.get_ylim()[1],'$\Delta \\theta$',color='tab:orange',fontsize=10,ha='center')
        self.pl_dat.remove_frame(ax_trial,['top','right','left'])
        ax_trial.set_yticks([])
        ax_trial.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_trial.set_xlabel('trial',fontsize=10)#,labelpad=-5,x=0.4)
        ax_trial.xaxis.set_label_coords(0.3,-0.3)
        ax_trial.text(-10,1.2,s='$c_a=%.2f$'%corr[c,s],fontsize=6)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig('multi_modes')

    def plot_coding_overlap(self):

        nSes = self.data['nSes']
        s_bool = np.zeros(nSes,'bool')
        # s_bool[17:87] = True
        s_bool[:] = True
        s_bool[~self.status['sessions']] = False

        t_ct_max = max(self.behavior['trial_ct'])

        nsteps = 4
        perc_act = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_coding = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_act_overlap = np.zeros((nsteps,nSes,t_ct_max))*np.NaN
        perc_coding_overlap = np.zeros((nsteps,nSes,t_ct_max))*np.NaN

        field_var = np.zeros((nSes,2))*np.NaN

        coding_fit = np.zeros((nsteps,nSes,2,3))*np.NaN
        coding_overlap_fit = np.zeros((nsteps,nSes,2,3))*np.NaN
        coding_overlap = np.zeros((nSes,2,2))*np.NaN

        fig = plt.figure(figsize=(7,5),dpi=300)

        # plt.show(block=False)
        # return
        ax_overlap = plt.axes([0.1,0.125,0.2,0.5])
        # pl_dat.add_number(fig,ax_fit_overlap,order=9)
        # ax_fit_coact = plt.axes([0.725,0.425,0.225,0.2])
        # pl_dat.add_number(fig,ax_fit_coact,order=6)
        j=0
        # color_t = plt.cm.rainbow(np.linspace(0,1,nsteps))

        dt = 5


        for s in tqdm(np.where(s_bool)[0][:-1]):

            if s_bool[s+1]:
                t_start = min(self.behavior['trial_ct'][s],dt)
                t_end = max(0,self.behavior['trial_ct'][s]-dt)
                coding_s1_start = np.any(self.fields['trial_act'][:,s,:,:t_start],-1) & self.status_fields[:,s,:]
                coding_s1_end = np.any(self.fields['trial_act'][:,s,:,t_end:],-1) & self.status_fields[:,s,:]

                ### get first dt trials and last dt trials
                t_start = self.behavior['trial_ct'][s+1]#min(cluster.behavior['trial_ct'][s+1],dt)
                t_end = 0#max(0,cluster.behavior['trial_ct'][s+1]-dt)
                coding_s2_start = np.any(self.fields['trial_act'][:,s+1,:,:t_start],-1) & self.status_fields[:,s+1,:]
                coding_s2_end = np.any(self.fields['trial_act'][:,s+1,:,t_end:],-1) & self.status_fields[:,s+1,:]

                coding_overlap[s,0,0] = coding_s2_start[coding_s1_start].sum()/coding_s1_start.sum()
                coding_overlap[s,0,1] = coding_s2_end[coding_s1_start].sum()/coding_s1_start.sum()
                coding_overlap[s,1,0] = coding_s2_start[coding_s1_end].sum()/coding_s1_end.sum()
                coding_overlap[s,1,1] = coding_s2_end[coding_s1_end].sum()/coding_s1_end.sum()


        ax_overlap.plot(np.random.rand(nSes)*0.5,coding_overlap[:,0,0],'k.',markersize=1)
        ax_overlap.errorbar(0.25,np.nanmean(coding_overlap[:,0,0]),np.nanstd(coding_overlap[:,0,0]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.plot(1+np.random.rand(nSes)*0.5,coding_overlap[:,1,0],'k.',markersize=1)
        ax_overlap.errorbar(1.25,np.nanmean(coding_overlap[:,1,0]),np.nanstd(coding_overlap[:,1,0]),fmt='r.',markersize=5,linestyle='none')

        ax_overlap.plot(2+np.random.rand(nSes)*0.5,coding_overlap[:,0,1],'k.',markersize=1)
        ax_overlap.errorbar(2.25,np.nanmean(coding_overlap[:,0,1]),np.nanstd(coding_overlap[:,0,1]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.plot(3+np.random.rand(nSes)*0.5,coding_overlap[:,1,1],'k.',markersize=1)
        ax_overlap.errorbar(3.25,np.nanmean(coding_overlap[:,1,1]),np.nanstd(coding_overlap[:,1,1]),fmt='r.',markersize=5,linestyle='none')
        ax_overlap.set_ylim([0,0.5])

        res = sstats.kruskal(coding_overlap[:,0,0],coding_overlap[:,1,0],nan_policy='omit')
        print(res)

        res = sstats.kruskal(coding_overlap[:,1,0],coding_overlap[:,1,1],nan_policy='omit')
        print(res)

        plt.show(block=False)
    

    def plot_XY(self):

        nSes = self.data['nSes']

        mode = 'PC'
        # s_bool = np.zeros(self.data['nSes'],'bool')
        # s_bool[17:87] = True
        # s_bool[~self.status['sessions']] = False
        s_bool = self.status['sessions']
        state_label = 'alpha' if (mode=='act') else 'beta'
        status_act = self.status['activity'][self.status['clusters'],:,1]
        status_act = status_act[:,s_bool]
        # status_act = status_act[:,session_bool]
        status_PC = self.status['activity'][self.status['clusters'],:,2]
        status_PC = status_PC[:,s_bool]
        nC_good,nSes_good = status_act.shape
        nSes_max = np.where(s_bool)[0][-1]


        active_neurons = status_act.mean(0)
        silent_neurons = (~status_act).mean(0)
        print('active neurons: %.3g +/- %.3g'%(active_neurons.mean()*100,active_neurons.std()*100))
        print('silent neurons: %.3g +/- %.3g'%(silent_neurons.mean()*100,silent_neurons.std()*100))

        coding_neurons = status_PC.sum(0)/status_act.sum(0)
        ncoding_neurons = ((~status_PC) & status_act).sum(0)/status_act.sum(0)
        print('coding neurons: %.3g +/- %.3g'%(coding_neurons.mean()*100,coding_neurons.std()*100))
        print('non-coding neurons: %.3g +/- %.3g'%(ncoding_neurons.mean()*100,coding_neurons.std()*100))


        p_act = np.count_nonzero(status_act)/(nC_good*nSes_good)
        p_PC = np.count_nonzero(status_PC)/np.count_nonzero(status_act)
        # print(p_PC)
        rnd_var_act = np.random.random(status_act.shape)
        rnd_var_PC = np.random.random(status_PC.shape)
        status_act_test = np.zeros(status_act.shape,'bool')
        status_act_test_rnd = np.zeros(status_act.shape,'bool')
        status_PC_test = np.zeros(status_PC.shape,'bool')
        status_PC_test_rnd = np.zeros(status_PC.shape,'bool')
        for c in range(nC_good):

            # status_act_test[c,:] = rnd_var_act[c,:] < (np.count_nonzero(status_act[c,:])/nSes_good)
            nC_act = status_act[c,:].sum()
            status_act_test[c,np.random.choice(nSes_good,nC_act,replace=False)] = True
            status_act_test_rnd[c,:] = rnd_var_act[c,:] < p_act

            status_PC_test[c,np.where(status_act[c,:])[0][np.random.choice(nC_act,status_PC[c,:].sum(),replace=False)]] = True
            status_PC_test_rnd[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < p_PC


        for mode in ['act','code']:

            fig = plt.figure(figsize=(3,2),dpi=self.pl_dat.sv_opt['dpi'])
            status = status_act if mode=='act' else status_PC
            status_test = status_act_test if mode=='act' else status_PC_test

            recurr = np.zeros((nSes_good,nSes_good))*np.NaN
            N_active = status_act.sum(0)

            for s in range(nSes_good):#np.where(s_bool)[0]:
                overlap = status[status[:,s],:].sum(0).astype('float')
                N_ref = N_active if mode=='act' else status_act[status_PC[:,s],:].sum(0)
                recurr[s,1:nSes_good-s] = (overlap/N_ref)[s+1:]


            recurr_test = np.zeros((nSes_good,nSes_good))*np.NaN
            N_active_test = status_test.sum(0)
            tmp = []
            for s in range(nSes_good):
                # overlap_act_test = status_test[status_test[:,s],:].sum(0).astype('float')
                overlap_test = status_test[status_test[:,s],:].sum(0).astype('float')
                N_ref = N_active_test if mode=='act' else status_act_test[status_PC_test[:,s],:].sum(0)
                recurr_test[s,1:nSes_good-s] = (overlap_test/N_ref)[s+1:]
                if (~np.isnan(recurr_test[s,:])).sum()>1:
                    tmp.append(recurr_test[s,~np.isnan(recurr_test[s,:])])

            rec_mean = np.nanmean(np.nanmean(recurr,0))
            rec_var = np.sqrt(np.nansum(np.nanvar(recurr,0))/(recurr.shape[1]-1))

            ax = plt.axes([0.2,0.3,0.75,0.65])
            # pl_dat.add_number(fig,ax,order=4,offset=[-250,250])

            p = status.sum()/(nSes_good*nC_good)


            # ax.plot([0,self.data['nSes']],[p,p],'k--')
            # ax.text(10,p+0.05,'$p^{(0)}_{\\%s^+}$'%(state_label),fontsize=8)
            SD = 1
            # ax.plot([1,nSes_good],[rec_mean,rec_mean],'k--',linewidth=0.5)
            recurr[:,0] = 1
            # ax.plot([0,1],[1,np.nanmean(recurr[:,0])],'-k')
            self.pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good-1,nSes_good),np.nanmean(recurr,0),SD*np.nanstd(recurr,0),col='k',ls='-',label='emp. data')
            # pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good-1,nSes_good),np.nanmean(recurr_test,0),SD*np.nanstd(recurr_test,0),col='tab:red',ls='-',label='rnd. data')
            ax.set_ylim([0,1.1])
            ax.set_xlabel('$\Delta$ sessions')
            if mode == 'act':
                ax.set_ylabel('act. recurr.')
            else:
                ax.set_ylabel('code recurr.')
            # ax.set_ylabel('$p(\\%s^+_{s+\Delta s} | \\%s^+_s)$'%(state_label,state_label))#'p(recurr.)')
            ax.set_xlim([0,nSes_good])
            ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[0.9,1],handlelength=1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.show(block=False)

            # if sv:
            #     pl_dat.save_fig('defense_%s_nornd_recurr'%mode)


        SD=1.96
        fig = plt.figure(figsize=(3,2),dpi=self.pl_dat.sv_opt['dpi'])
        ax = plt.axes([0.2,0.3,0.75,0.65])
        N_data = np.zeros(self.data['nSes'])*np.NaN

        D_KS = np.zeros(self.data['nSes'])*np.NaN
        N_stable = np.zeros(self.data['nSes'])*np.NaN
        N_total = np.zeros(self.data['nSes'])*np.NaN     ### number of PCs which could be stable
        # fig = plt.figure()
        p_rec_alt = np.zeros(self.data['nSes'])*np.NaN

        s1_shifts,s2_shifts,f1,f2 = np.unravel_index(self.compare['pointer'].col,(self.data['nSes'],nSes,self.params['field_count_max'],self.params['field_count_max']))
        c_shifts = self.compare['pointer'].row

        for ds in range(1,self.data['nSes']):#min(self.data['nSes'],30)):
            Ds = s2_shifts-s1_shifts
            idx_ds = np.where((Ds==ds) & s_bool[s1_shifts] & s_bool[s2_shifts])[0]
            N_data[ds] = len(idx_ds)

            idx_shifts = self.compare['pointer'].data[idx_ds].astype('int')-1
            shifts = self.compare['shifts'][idx_shifts]
            N_stable[ds] = (np.abs(shifts)<(SD*self.stability['all']['mean'][0,2])).sum()

            p_rec_alt[ds] = N_stable[ds]/N_data[ds]

        p_rec_alt[0] = 1
        ax.plot(range(self.data['nSes']),p_rec_alt,'-',color='k')
        # ax.plot(0,1,'ok')
        r_random = 2*SD*self.stability['all']['mean'][0,2]/100
        ax.plot([1,self.data['nSes']],[r_random,r_random],'--',color='tab:red',linewidth=1)
        ax.set_ylim([0,1.1])
        ax.set_xlim([0,nSes_good])
        ax.set_ylabel('place field recurr.',fontsize=12)
        ax.set_xlabel('$\Delta$ sessions')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('defense_pf_recurr')

        p_shift = np.zeros(nbin)
        for s in np.where(s_bool)[0]:
            idx_field = np.where(self.status_fields[:,s,:])
            for c,f in zip(idx_field[0],idx_field[1]):
                roll = round((-self.fields['location'][c,s,f,0]+self.data['nbin']/2)/L_track*self.data['nbin'])
                p_shift += np.roll(self.fields['p_x'][c,s,f,:],roll)
        p_shift /= p_shift.sum()

        PC_idx = np.where(self.status['activity'][...,2])
        N_data = len(PC_idx[0])

        p_ds0,p_cov = fit_shift_model(p_shift)

        p = self.stability
        fig = plt.figure(figsize=(3,1.5),dpi=300)
        ax = plt.axes([0.2,0.3,0.75,0.65])

        ax.plot([0,self.data['nSes']],[p_ds0[2],p_ds0[2]],linestyle='--',color=[0.6,0.6,0.6])
        ax.text(10,p_ds0[2]+1,'$\sigma_0$',fontsize=8)

        sig_theta = self.stability['all']['mean'][0,2]

        self.pl_dat.plot_with_confidence(ax,range(1,nSes+1),p['all']['mean'][:,2],p['all']['CI'][...,2].T,'k','-')

        ax.set_ylim([0,12])
        ax.set_xlim([0,nSes_good])
        ax.set_ylabel('$\sigma_{\Delta \\theta}$',fontsize=12)
        ax.set_xlabel('$\Delta$ sessions')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show(block=False)
        if sv:
            pl_dat.save_fig('defense_sig_shift')




















class plot_dat:

  def __init__(self,mouse,pathFigures,nSes,para,sv_suffix='',sv_ext='png',sv_dpi=300):
    self.pathFigures = pathFigures
    if not os.path.exists(self.pathFigures):
        os.mkdir(self.pathFigures)
    self.mouse = mouse

    L_track = 100
    nbin = para['nbin']

    self.sv_opt = {'suffix':sv_suffix,
                   'ext':sv_ext,
                   'dpi':sv_dpi}

    self.plt_presi = True;
    self.plot_pop = False;

    self.plot_arr = ['NRNG','GT','RW']
    self.col = ['b','g','r']
    self.col_fill = [[0.5,0.5,1],[0.5,1,0.5],[1,0.5,0.5]]

    self.h_edges = np.linspace(-0.5,nSes+0.5,nSes+2)
    self.n_edges = np.linspace(1,nSes,nSes)
    self.bin_edges = np.linspace(1,L_track,nbin)

    # self.bars = {}
    # self.bars['PC'] = np.zeros(nbin)
    # self.bars['PC'][para['zone_mask']['others']] = 1

    # self.bars['GT'] = np.zeros(nbin);

    # if np.count_nonzero(para['zone_mask']['gate'])>1:
    #   self.bars['GT'][para['zone_mask']['gate']] = 1

    # self.bars['RW'] = np.zeros(nbin);
    # self.bars['RW'][para['zone_mask']['reward']] = 1


    ### build blue-red colormap
    #n = 51;   ## must be an even number
    #cm = ones(3,n);
    #cm(1,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## red
    #cm(2,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## green
    #cm(2,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## green
    #cm(3,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## blue

  def add_number(self,fig,ax,order=1,offset=None):

      # offset = [-175,50] if offset is None else offset
      offset = [-150,50] if offset is None else offset
      offset = np.multiply(offset,self.sv_opt['dpi']/300)
      pos = fig.transFigure.transform(plt.get(ax,'position'))
      x = pos[0,0]+offset[0]
      y = pos[1,1]+offset[1]
      ax.text(x=x,y=y,s='%s)'%chr(96+order),ha='center',va='center',transform=None,weight='bold',fontsize=14)


  def remove_frame(self,ax,positions=None):

    if positions is None:
      positions = ['left','right','top','bottom']

    for p in positions:
      ax.spines[p].set_visible(False)

    # if 'left' in positions:
      # ax.set_yticks([])

    # if 'bottom' in positions:
      # ax.set_xticks([])


  def plot_with_confidence(self,ax,x_data,y_data,CI,col='k',ls='-',lw=1,label=None):

    col_fill = np.minimum(np.array(colors.to_rgb(col))+np.ones(3)*0.3,1)
    if len(CI.shape) > 1:
      ax.fill_between(x_data,CI[0,:],CI[1,:],color=col_fill,alpha=0.2)
    else:
      ax.fill_between(x_data,y_data-CI,y_data+CI,color=col_fill,alpha=0.2)
    ax.plot(x_data,y_data,color=col,linestyle=ls,linewidth=lw,label=label)



  def save_fig(self,fig_name,fig_pos=None):
    path = os.path.join(self.pathFigures,'m%s_%s%s.%s'%(self.mouse,fig_name,self.sv_opt['suffix'],self.sv_opt['ext']));
    plt.savefig(path,format=self.sv_opt['ext'],dpi=self.sv_opt['dpi'])
    print('Figure saved as %s'%path)


  

def get_overlap(s,inVars):

  status,N,L = inVars
  nC,nSes = status.shape[:2]

  recurr = {'active': {'all':                np.zeros(nSes)*np.NaN,
                       'continuous':         np.zeros(nSes)*np.NaN,
                       'overrepresentation': np.zeros(nSes)*np.NaN},
            'coding': {'all':                np.zeros(nSes)*np.NaN,
                       'ofactive':           np.zeros(nSes)*np.NaN,
                       'continuous':         np.zeros(nSes)*np.NaN,
                       'overrepresentation': np.zeros(nSes)*np.NaN}}
  if N['active'][s] == 0:
    return recurr
  overlap_act = status[status[:,s,1],:,1].sum(0)
  overlap_PC = status[status[:,s,2],:,2].sum(0)

  recurr['active']['all'][1:(nSes-s)] = overlap_act[s+1:]/N['active'][s+1:]

  recurr['coding']['all'][1:(nSes-s)] = overlap_PC[s+1:]/N['coding'][s+1:]
  for (i,s1) in enumerate(range(s+1,nSes)):
    recurr['coding']['ofactive'][i+1] = overlap_PC[s1]/status[status[:,s,2],s1,1].sum()

  #print(recurr['active']['all'])
  rand_pull_act = np.zeros((nSes-s,L))*np.NaN
  rand_pull_PC = np.zeros((nSes-s,L))*np.NaN

  for s2 in range(s+1,nSes):
    if (N['active'][s]==0) or (N['active'][s2]==0):
      continue
    rand_pull_act[s2-s,:] = (np.random.choice(nC,(L,N['active'][s]))<N['active'][s2]).sum(1)

    offset = N['active'][s] - overlap_act[s2]
    randchoice_1 = np.random.choice(N['active'][s],(L,N['coding'][s]))
    randchoice_2 = np.random.choice(np.arange(offset,offset+N['active'][s2]),(L,N['coding'][s2]))
    for l in range(L):
      rand_pull_PC[s2-s,l] = np.isin(randchoice_1[l,:],randchoice_2[l,:]).sum()


    ### find continuously coding neurons
    recurr['active']['continuous'][s2-s] = (status[:,s:s2+1,1].sum(1)==(s2-s+1)).sum()/N['active'][s2]#(ic_vals[idx_ds,2] == 1).sum()/N['active'][s2]
    recurr['coding']['continuous'][s2-s] = (status[:,s:s2+1,2].sum(1)==(s2-s+1)).sum()/N['coding'][s2]#(ic_vals[idx_ds,2] == 1).sum()/N['active'][s2]

  recurr['active']['overrepresentation'][:nSes-s] = (overlap_act[s:]-np.nanmean(rand_pull_act,1))/np.nanstd(rand_pull_act,1)
  recurr['coding']['overrepresentation'][:nSes-s] = (overlap_PC[s:]-np.nanmean(rand_pull_PC,1))/np.nanstd(rand_pull_PC,1)
  return recurr

def bootstrap_shifts(fun,shifts,N_bs,nbin):

  L_track = 100
  N_data = len(shifts)
  if N_data == 0:
    return np.zeros(4)*np.NaN,np.zeros((2,4))*np.NaN,np.zeros(4)*np.NaN,np.zeros((2,nbin))*np.NaN

  samples = np.random.randint(0,N_data,(N_bs,N_data))
  # sample_randval = np.random.rand(N_bs,N_data)
  shift_distr_bs = np.zeros((N_bs,nbin))
  par = np.zeros((N_bs,4))*np.NaN
  for i in range(N_bs):
    shift_distr_bs[i,:] = shifts[samples[i,:],:].sum(0)
    shift_distr_bs[i,:] /= shift_distr_bs[i,:].sum()
    par[i,:],p_cov = fun(shift_distr_bs[i,:])
  p = np.nanmean(par,0)
  p_CI = np.percentile(par,[2.5,97.5],0)
  p_std = np.nanstd(par,0)

  return p, p_CI, p_std, shift_distr_bs


def get_shift_distr(ds,compare,para):

  nSes,nbin,N_bs,idx_celltype = para
  L_track=100
  p = {'all':{},
        'cont':{},
        'mix':{},
        'discont':{},
        'silent_mix':{},
        'silent':{}}

  s1_shifts,s2_shifts,f1,f2 = np.unravel_index(compare['pointer'].col,(nSes,nSes,5,5))
  #print(idx_celltype)
  Ds = s2_shifts-s1_shifts
  idx_ds = np.where((Ds==ds) & idx_celltype)[0]
  N_data = len(idx_ds)

  idx_shifts = compare['pointer'].data[idx_ds].astype('int')-1
  shifts = compare['shifts'][idx_shifts]
  shifts_distr = compare['shifts_distr'][idx_shifts,:].toarray()
  
  for pop in p.keys():
    if pop == 'all':
      idxes = np.ones(N_data,'bool')
    elif pop=='cont':
      idxes = compare['inter_coding'][idx_ds,1]==1
    elif pop=='mix':
      idxes = ((compare['inter_coding'][idx_ds,1]>0) & (compare['inter_coding'][idx_ds,1]<1)) & (compare['inter_active'][idx_ds,1]==1)
    elif pop=='discont':
      idxes = (compare['inter_coding'][idx_ds,1]==0) & (compare['inter_active'][idx_ds,1]==1)
    elif pop=='silent_mix':
      idxes =(compare['inter_active'][idx_ds,1]>0) & (compare['inter_active'][idx_ds,1]<1)
    elif pop=='silent':
      idxes = compare['inter_active'][idx_ds,1]==0

    # p[pop]['mean'], p[pop]['std'], _ = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[idxes,:],N_bs,nbin)
    p[pop]['mean'], p[pop]['CI'], p[pop]['std'], _ = bootstrap_shifts(fit_shift_model,shifts_distr[idxes,:],N_bs,nbin)
  return p

## fitting functions and options
F_shifts = lambda x,A0,A,sig,theta : A/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-theta)**2/(2*sig**2)) + A0/len(x)     ## gaussian + linear offset

def fit_shift_model(data):
  p_bounds = ([0,0,0,-10],[1,1,50,10])
  # shift_hist = np.histogram(data,np.linspace(-50,50,101),density=True)[0]
  # shift_hist[0] = shift_hist[1]
  # shift_hist /= shift_hist.sum()
  try:
    # return curve_fit(F_shifts,np.linspace(-49.5,49.5,100),shift_hist,bounds=p_bounds)
    return curve_fit(F_shifts,np.linspace(-49.5,49.5,100),data,bounds=p_bounds)
  except:
    return np.zeros(4)*np.NaN, np.NaN


def get_field_stability(cluster,SD=1.96,s_bool=None):

    nbin = cluster.data['nbin']

    sig_theta = cluster.stability['all']['mean'][0,2]
    stab_thr = SD*sig_theta

    s_bool = cluster.status['sessions'] if s_bool is None else s_bool

    field_stability = np.zeros(cluster.data['nC'])*np.NaN
    # idx_fields = np.where(cluster.status_fields & cluster.status['sessions'][np.newaxis,:,np.newaxis])
    idx_fields = np.where(cluster.status_fields & s_bool[np.newaxis,:,np.newaxis])

    for c in np.where(cluster.status['clusters'])[0]:#[:10]

        c_fields = (idx_fields[0] == c)
        fields_ref = cluster.fields['location'][c,idx_fields[1][c_fields],idx_fields[2][c_fields],0]

        count_hit = 0
        # count_ref = cluster.status['activity'][c,:,2].sum()
        if cluster.status['activity'][c,s_bool,2].sum()>1:
            for s in np.where(cluster.status['activity'][c,:,1] & s_bool)[0]:
                if cluster.status['activity'][c,s,2]:
                    fields_compare = cluster.fields['location'][c,s,cluster.status_fields[c,s,:],0]
                    count_ref = len(fields_ref)-len(fields_compare)
                    d = np.abs(np.mod(fields_ref[np.newaxis,:]-fields_compare[:,np.newaxis]+nbin/2,nbin)-nbin/2)
                    # count_hit += (np.sum(d < stab_thr)-len(fields_compare))/(count_ref-1) if count_ref > 1 else np.NaN
                    count_hit += (np.sum(d < stab_thr)-len(fields_compare))/count_ref if count_ref > 0 else np.NaN
        # N_norm = cluster.status['activity'][c,:,1].sum()
        N_norm = s_bool.sum()
        if N_norm > 0:
            field_stability[c] = count_hit / N_norm#count_ref# - count_miss / count_ref

    return field_stability


def get_act_stability_temp(cluster,status_act=None,ds=3):

    act_stability = np.zeros((cluster.data['nC'],cluster.data['nSes'],2))*np.NaN
    # ds = ds//2

    if status_act is None:
        status_act = cluster.status['activity'][...,1]

    # print(ds)
    for c in np.where(cluster.status['clusters'])[0]:#[:10]

        for s in np.where(cluster.status['sessions'])[0][:-1]:
            s_min = max(0,s-ds)
            s_max = min(cluster.data['nSes']-1,s+ds+1)

            count_act = status_act[c,s_min:s_max].sum()
            count_act_possible = cluster.status['sessions'][s_min:s_max].sum()
            count_act_recurr = 0
            count_act_recurr_possible = 0

            for s2 in range(s_min,s_max):
                if cluster.status['sessions'][s2]:
                    if cluster.status['sessions'][s2+1]:
                        count_act_recurr_possible += 1
                        if status_act[c,s2]:
                            count_act_recurr += status_act[c,s2+1]

            # if cluster.status['activity'][c,s,1]:
            act_stability[c,s,0] = count_act/count_act_possible
            act_stability[c,s,1] = count_act_recurr/count_act_recurr_possible if count_act_recurr_possible>0 else np.NaN
            # else:
                # act_stability[c,s,:] = 0
            # print('--- neuron %d @ s%d: ---'%(c,s))
            # print(act_stability[c,s,:])
            # print('counts: %d/%d'%(count_act,count_act_possible))
            # print(cluster.status['activity'][c,s_min:s_max,1])
            # print(cluster.status['sessions'][s_min:s_max])
    return act_stability


def get_act_stability(cluster,s_bool):

    act_stability = np.zeros((cluster.data['nC'],3))*np.NaN

    for c in np.where(cluster.status['clusters'])[0]:#[:10]

        count_act = cluster.status['activity'][c,s_bool,1].sum()
        count_act_possible = s_bool.sum()
        count_act_recurr = 0
        count_act_recurr_possible = 0

        for s in np.where(s_bool)[0][:-1]:

            if cluster.status['sessions'][s+1]:
                count_act_recurr_possible += 1
                if cluster.status['activity'][c,s,1]:
                    count_act_recurr += cluster.status['activity'][c,s+1,1]

        act_stability[c,0] = count_act/count_act_possible
        act_stability[c,1] = count_act_recurr/count_act_recurr_possible if count_act_recurr_possible>0 else np.NaN
        act_stability[c,2] = act_stability[c,1] - (count_act)/count_act_possible

        # print('--- neuron %d : ---'%c)
        # print(act_stability[c,:])
        # print('counts: %d/%d'%(count_act,count_act_possible))
        # print('counts (recurr): %d/%d'%(count_act_recurr,count_act_recurr_possible))
        # print(cluster.status['activity'][c,s_bool,1])
        # print(cluster.status['sessions'][s_min:s_max])
    return act_stability

def get_field_stability_temp(cluster,SD=1.96,ds=3):

    # nC = cluster.data['nC']
    # nSes = cluster.data['nSes']
    nbin = cluster.data['nbin']
    # nC,nSes = cluster.status['activity'].shape[:2]
    sig_theta = cluster.stability['all']['mean'][0,2]
    stab_thr = SD*sig_theta
    # nbin = 100
    # ds = ds//2
    print(ds)
    field_stability = np.zeros((cluster.data['nC'],cluster.data['nSes']))*np.NaN
    # act_stability = np.zeros((nC,nSes))*np.NaN
    idx_fields = np.where(cluster.status_fields & cluster.status['sessions'][np.newaxis,:,np.newaxis])

    for c in np.where(cluster.status['clusters'])[0]:

        c_fields = (idx_fields[0] == c)

        for s in np.where(cluster.status['sessions'])[0][:-1]:

            field_stability[c,s] = 0

            if cluster.status['activity'][c,s,2]:
                s_min = max(0,s-ds)
                s_max = min(cluster.data['nSes']-1,s+ds+1)
                if cluster.status['activity'][c,s_min:s_max,2].sum()>1:
                    s_fields = (idx_fields[1]>=s_min) & (idx_fields[1]<s_max)
                    fields_ref = cluster.fields['location'][c,idx_fields[1][c_fields&s_fields],idx_fields[2][c_fields&s_fields],0]

                    fields_compare = cluster.fields['location'][c,s,cluster.status_fields[c,s,:],0]
                    count_ref = len(fields_ref)-len(fields_compare)
                    d = np.abs(np.mod(fields_ref[np.newaxis,:]-fields_compare[:,np.newaxis]+nbin/2,nbin)-nbin/2)

                    field_stability[c,s] += (np.sum(d < stab_thr)-len(fields_compare))/count_ref# if count_ref > 0 else np.NaN
                    # count_hit = 0

            # count_ref = cluster.status['activity'][c,s_min:s_max,2].sum()
            # act_stability[c,s] = cluster.status['activity'][c,s_min:s_max,1].sum()/cluster.status['sessions'][s_min:s_max].sum()


            # if cluster.status['activity'][c,s_min:s_max,2].sum()>1:
            #     for s2 in range(s_min,s_max):#np.where(cluster.status['activity'][c,:,1])[0]:
            #         if cluster.status['activity'][c,s2,2]:
            #             fields_compare = cluster.fields['location'][c,s2,cluster.status_fields[c,s2,:],0]
            #             count_ref = len(fields_ref)-len(fields_compare)
            #             d = np.abs(np.mod(fields_ref[np.newaxis,:]-fields_compare[:,np.newaxis]+nbin/2,nbin)-nbin/2)
            #             # count_hit += (np.sum(d < stab_thr)-len(fields_compare))/(count_ref-1)
            #             count_hit += (np.sum(d < stab_thr)-len(fields_compare))/count_ref if count_ref > 0 else np.NaN

            # N_norm = cluster.status['activity'][c,s_min:s_max,1].sum()
            # N_norm = cluster.status['sessions'][s_min:s_max].sum()
            # if N_norm > 0:
                # field_stability[c,s] = count_hit / N_norm#count_ref# - count_miss / count_ref
            # print(field_stability[c,s])

    return field_stability