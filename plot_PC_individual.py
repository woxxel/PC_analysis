import numpy as np
import matplotlib.pyplot as plt


def plot_single_PC(basePath,mouse,PCs,c,s):
  
  x_arr = range(100)
  
  print('mutual information: %5.3f, \t p_value=%5.3f'%(PCs['MI']['value'][c,s],PCs['MI']['p_value'][c,s]))
  print('Bayes: %5.3f'%PCs['Bayes']['factor'][c,s,0])
  plt.figure()
  
  ax1 = plt.subplot(311)
  ax1.bar(x_arr,PCs['firingmap'][c,s,0,:],facecolor='b')
  ax1.errorbar(x_arr,PCs['firingmap'][c,s,0,:],PCs['firingmap'][c,s,2:,:],ecolor='r',fmt='none')
  
  ax2 = plt.subplot(312)
  for f in range(PCs['fields']['nModes'][c,s]):
    print(f)
    ax2.bar(x_arr,PCs['fields']['p_x'][c,s,f,1,:])
  
  plt.show(block=False)