import cv2, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sstats
import scipy as sp
from scipy import ndimage
from scipy.stats.distributions import t
from matplotlib import animation
from matplotlib import colors, rc
from mpl_toolkits.mplot3d import Axes3D
from tqdm import *

from scipy.io import loadmat
from utils import fit_plane, z_from_point_normal_plane, rotation_matrix, bootstrap_data, calculate_img_correlation, get_shift_and_flow, pathcat, gauss_smooth, com


class clusterplots:

    def __init__(self,cluster):
        self.cluster = cluster

        self.nC, self.nSes = self.cluster.status.shape[:2]
        self.reprocess = True

    def find_overlaps(self,s,reprocess=None):

        if not (reprocess is None):
            self.reprocess = reprocess

        if self.reprocess:
            pathLoad = pathcat([self.cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
            data = loadmat(pathLoad,variable_names=['A','C'])
            self.D_ROIs = sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.cluster.stats['com'][:,s,:]))

            idx_close = self.D_ROIs<10

            self.corrcoef = np.zeros((self.nC,self.nC))*np.NaN
            self.Acorr = np.zeros((self.nC,self.nC))*np.NaN

            for c in tqdm(np.where(self.cluster.status[:,s,1])[0]):
                n = int(self.cluster.IDs['neuronID'][c,s,1])
                for cc in np.where(idx_close[c,:])[0]:
                    nn = int(self.cluster.IDs['neuronID'][cc,s,1])
                    self.corrcoef[c,cc] = np.corrcoef(data['C'][n,:],data['C'][nn,:])[0,1]
                    self.Acorr[c,cc],_ = calculate_img_correlation(data['A'][:,n],data['A'][:,nn],shift=False)
            self.reprocess=False

        np.fill_diagonal(self.corrcoef,np.NaN)
        np.fill_diagonal(self.Acorr,np.NaN)

        idx_overlap = np.where((self.Acorr>0.2) & (self.corrcoef>0.5))
        for (i,j) in zip(idx_overlap[0],idx_overlap[1]):
            print('neurons: %d,%d, \t Acorr: %.3f, Ccorr: %.3f, \t SNR: %.2f,%.2f'%(i,j,self.Acorr[i,j],self.corrcoef[i,j],self.cluster.stats['SNR'][i,s],self.cluster.stats['SNR'][j,s]))
        print(len(idx_overlap[0]))
        # print(idx_overlap)

    def plot_fmap_sessions(self,c_arr):

        nbin = 100
        margin = 20

        w_arr = [0.5,1,2]

        cmap = plt.get_cmap('rainbow')
        cNorm = colors.Normalize(vmin=0,vmax=self.nSes)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm,cmap=cmap)

        fig = plt.figure(figsize=(7,5),dpi=300)
        # ax = plt.axes([0.1,0.1,0.5,0.85])

        # x_grid, y_grid = np.meshgrid(np.arange(0., self.cluster.meta['dims'][0]).astype(np.float32), np.arange(0., self.cluster.meta['dims'][1]).astype(np.float32))
        #
        # set_ref = True
        # for s in tqdm(np.where(self.cluster.sessions['bool'])[0]):#range(11)):#
        #     n_arr = self.cluster.IDs['neuronID'][c_arr,s,1]
        #     if np.any(~np.isnan(n_arr)):
        #         pathLoad = pathcat([self.cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
        #         ld = loadmat(pathLoad,variable_names='A')
        #
        #         x_remap = (x_grid - \
        #                     self.cluster.sessions['shift'][s,0] + self.cluster.sessions['shift'][0,0] + \
        #                     self.cluster.sessions['flow_field'][s,:,:,0] - self.cluster.sessions['flow_field'][0,:,:,0]).astype('float32')
        #         y_remap = (y_grid - \
        #                     self.cluster.sessions['shift'][s,1] + self.cluster.sessions['shift'][0,1] + \
        #                     self.cluster.sessions['flow_field'][s,:,:,1] - self.cluster.sessions['flow_field'][0,:,:,1]).astype('float32')
        #
        #         for i,(n,c) in enumerate(zip(n_arr,c_arr)):
        #             if ~np.isnan(n):
        #                 a_tmp = ld['A'][:,n].reshape(self.cluster.meta['dims']).toarray()
        #                 a = cv2.remap(a_tmp, x_remap,y_remap, cv2.INTER_CUBIC)
        #                 # a = ld['A'][:,n]
        #                 if set_ref:
        #                     # print(a.reshape(-1,1).shape)
        #                     a_com = com(a.reshape(-1,1),512,512)
        #                     # print(a_com)
        #                     (x_ref,y_ref) = a_com[0]# print(x_ref,y_ref)
        #                     x_lims = [x_ref-margin,x_ref+margin]
        #                     y_lims = [y_ref-margin,y_ref+margin]
        #                     set_ref = False
        #                 # print(c,n)
        #                 # print(com(a,512,512))
        #                 colVal = scalarMap.to_rgba(s)
        #                 ax.contour(a/a.max(), levels=[0.3], colors=[colVal], linewidths=[w_arr[i]], linestyles=['solid'])
        # ax.set_xlim(x_lims)
        # ax.set_ylim(y_lims)

        cbaxes = plt.axes([0.625,0.75,0.01,0.225])
        cb = fig.colorbar(scalarMap,cax = cbaxes,orientation='vertical')

        for i,c in enumerate(c_arr):
            ax = plt.axes([0.7+0.125*i,0.08,0.1,0.87])
            # idx_strong_PC = np.where(cluster.status[...,2].sum(1)>20)[0]
            # idx_PC = 28#np.random.choice(idx_strong_PC)    ## 28,1081
            # print(idx_PC)
            firingmap = self.cluster.stats['firingmap'][c,...]
            firingmap = gauss_smooth(firingmap,[0,4])
            firingmap = firingmap - np.nanmin(firingmap,1)[:,np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            ax.imshow(firingmap,aspect='auto',origin='upper',cmap='jet',clim=[0,3])
            ax.barh(range(self.nSes),-(self.cluster.status[c,:,2]*10.),left=-5,facecolor='r')
            ax.barh(range(self.nSes),-((~self.cluster.status[c,:,1])*10.),left=-5,facecolor=[0.6,0.6,0.6])
            ax.barh(range(self.nSes),-((~self.cluster.sessions['bool'])*10.),left=-5,facecolor='k')
            # idx_coding = np.where(cluster.status[idx_PC,:,2])[0]
            # ax.plot(-np.ones_like(idx_coding)*10,idx_coding,'ro')
            ax.set_xlim([-10,nbin])
            ax.set_ylim([self.nSes,-0.5])
            ax.set_ylabel('Session')
            ax.set_xlabel('Location [bins]')
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        #plt.set_cmap('jet')
        plt.tight_layout()
        plt.show(block=False)
