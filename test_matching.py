import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from utils import get_nFolder, pathcat

import cv2

def test_matching(basePath,mouse):
  
  pathMouse = pathcat([basePath,mouse])
  nSes = get_nFolder(pathMouse,'Session')
  
  pathSession = pathcat([pathMouse,'Session01'])
  pathLoad = pathcat([pathSession,'results_OnACID.mat'])
    
  ld = sio.loadmat(pathLoad)
  template1 = ld['Cn']
  dims = template1.shape
  
  x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32))
  
  for s in range(15,17):
    pathSession = pathcat([pathMouse,'Session%02d'%(s+1)])
    pathLoad = pathcat([pathSession,'results_OnACID.mat'])
      
    ld = sio.loadmat(pathLoad)
    template2 = ld['Cn']
  
    C = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(template1) * np.fft.fft2(np.rot90(template2,2)))))
    
    max_pos = np.where(C==np.max(C))
    x_shift = (max_pos[1] - (dims[1]/2-1)).astype(int)
    y_shift = (max_pos[0] - (dims[0]/2-1)).astype(int)
    
    x_remap = (x_grid - x_shift).astype(np.float32)
    y_remap = (y_grid - y_shift).astype(np.float32)
    template2 = cv2.remap(template2.astype(np.float32), x_remap, y_remap, cv2.INTER_NEAREST)
    
    
    print('Session %02d'%(s+1))
    print('max correlation: %5.3f'%C.max())
    print('shift: %d,%d'%(x_shift,y_shift))
    
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow(template1)
    plt.subplot(222)
    plt.imshow(template2)
    plt.subplot(223)
    plt.imshow(template2+template1)
    plt.subplot(224)
    plt.imshow(C)
    plt.show()
    
    #template1 = np.copy(template2)
    
    