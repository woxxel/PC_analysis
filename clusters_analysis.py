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
from .build_clusters import cluster
from .utils import get_ICPI, get_dp
from .utils import gauss_smooth, add_number, bootstrap_data

class cluster_analysis(cluster):

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
        # D_ROIs_PC = sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.matching['com'][c_arr_PC,s,:]))
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