import time
import numpy as np
import matplotlib.pyplot as plt
from utils import fdr_control

def find_PCs(PCs,alpha,Bayes_thr,A_ratio_thr,p_mass_thr,para):
  
  ### input
  ## alpha  -   control of false positives
  ##
  t_start = time.time()
  
  nC, nSes, tmp = PCs['status'].shape
  nROI = PCs['status'][:,:,1].sum(0)
  PCs['status'][:,:,2:] = False
  
  PCs['MI']['p_value'][PCs['MI']['p_value']==0.001] = 10**(-10)    ## need this - can't get any lower than 0.001 with 1000 shuffles...
  
  PC_mask = np.zeros((nC,nSes)).astype('bool')
  for s in range(nSes):
    PC_mask[:,s] = fdr_control(PCs['MI']['p_value'][:,s],alpha)
  
  A_ratio = PCs['fields']['parameter'][:,:,:,1,0]/PCs['fields']['parameter'][:,:,:,0,0]
  Bayes = PCs['Bayes']['factor'][:,:,0] - PCs['Bayes']['factor'][:,:,1]
  for c in range(nC):
    for s in range(nSes):
      if PCs['status'][c,s,1]:
        
        if PC_mask[c,s] & (Bayes[c,s] > Bayes_thr):
          
          for f in range(PCs['fields']['nModes'][c,s]):
            if (A_ratio[c,s,f] >= A_ratio_thr) & (PCs['fields']['posterior_mass'][c,s,f] >= p_mass_thr):
              if para['zone_idx']['reward'][0] <= PCs['fields']['parameter'][c,s,f,3,0] <= para['zone_idx']['reward'][-1]:
                PCs['fields']['status'][c,s,f] = 4;
              elif para['zone_idx']['gate'][0] <= PCs['fields']['parameter'][c,s,f,3,0] <= para['zone_idx']['gate'][-1]:
                PCs['fields']['status'][c,s,f] = 3;
              else:
                PCs['fields']['status'][c,s,f] = 2;
              PCs['status'][c,s,PCs['fields']['status'][c,s,f]] = True;
          
          PCs['fields']['nModes'][c,s] = np.count_nonzero(PCs['fields']['status'][c,s,:])
  
  t_end = time.time()
  print('PC-characterization done. Time taken: %7.5f'%(t_end-t_start))
  return PCs