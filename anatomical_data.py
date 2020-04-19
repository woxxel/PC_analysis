import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import pathcat, get_nPaths

def get_ROI_positions(basePath,mouse,s):
  
  dims = (512,512)
  pathMouse = pathcat([basePath,mouse])
  nSes,paths = get_nPaths(pathMouse,'Session')
  
  pathSession = paths[s]
  pathROI = pathcat([pathSession,'results_OnACID.mat'])
  A = sio.loadmat(pathROI)
  A = A['A']
  
  Coor = np.matrix([np.outer(np.ones(dims[1]), np.arange(dims[0])).ravel(),np.outer(np.arange(dims[1]), np.ones(dims[0])).ravel()], dtype=A.dtype)
  #Coor = np.matrix([np.outer(np.arange(dims[1]), np.ones(dims[0])).ravel(),np.outer(np.ones(dims[1]), np.arange(dims[0])).ravel()], dtype=A.dtype)
  Anorm = sp.sparse.vstack([a/a.sum() for a in A.T]).T;
  cm = (Coor * Anorm).T


def plot_ROI_movement(clusters,n=np.array([0])):
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  
  nCells = n.shape[0]
  nSes = clusters['com'].shape[1]
  
  print(nCells)
  print(nSes)
  
  for n in range(30):#range(nCells):
    plt.plot(clusters['com'][n,:,0]+clusters['session_shift'][:,1],clusters['com'][n,:,1]+clusters['session_shift'][:,0],range(nSes),'k')
    #plt.plot(clusters['com'][n,:,0],clusters['com'][n,:,1],range(nSes),'k')
  plt.xlim([0,512])
  plt.ylim([0,512])
  plt.show(block=False)