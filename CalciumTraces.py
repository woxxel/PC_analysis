import sys, scipy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

sys.path.append('/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Programs/PC_analysis')
from utils import pathcat, find_modes

def CalciumTraces(basePath,mouse,s,n=None):
  
  f = 15  ## Hz
  
  pathMouse = pathcat([basePath,mouse])
  pathSession = pathcat([pathMouse,'Session%02d'%s])
  
  ## do I need noise level for either method? (rerun redetection??)
  
  pathResults = pathcat([pathSession,'results_redetect.mat'])
  ld = loadmat(pathResults,variable_names=['C','S'])
  C = ld['C']
  S = ld['S']
  print(C.shape)
  ## plot results
  nCells,T = S.shape
  t_arr = np.linspace(0,T/f,T)
  
  
  N_spikes_C = np.zeros(nCells)
  fitness, erfc, sd, md = compute_event_exceptionality(C)
  data_thr_C = md+2*sd
  N_spikes_C = np.floor(np.maximum(0,C / data_thr_C[:,np.newaxis])).sum(1)
  
  data_thr_S = np.zeros(nCells)
  N_spikes_S = np.zeros(nCells)*np.NaN
  for nn in range(nCells):
    #print(nn)
    try:
      fitness, erfc, sd, md = compute_event_exceptionality(S[nn:nn+1,S[nn,:]>0])
      data_thr_S[nn] = md+2*sd
      N_spikes_S[nn] = np.floor(np.maximum(0,S[nn,:] / data_thr_S[nn])).sum()
    except:
      1
  return N_spikes_C, N_spikes_S
  
  plt.figure()
  ax1 = plt.subplot(221)
  ax1.plot(t_arr,C[n,:],'k')
  ax1.plot(t_arr,np.ones(T)*data_thr_C[n],'g')
  
  #print(N_spikes_S.sum())
  ax2 = plt.subplot(223,sharex=ax1)
  ax2.plot(t_arr,S[n,:],'r')
  ax2.plot(t_arr,np.ones(T)*data_thr_S[n],'g')
  plt.show(block=False)
  
  plt.figure()
  plt.scatter(N_spikes_C,N_spikes_S)
  plt.show(block=False)
  
  #plt.subplot(311)
  
  



def compute_event_exceptionality(traces, robust_std=False, N=5, sigma_factor=3.):
    """
    Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Args:
        traces: ndarray
            Fluorescence traces

        N: int
            N number of consecutive events

        sigma_factor: float
            multiplicative factor for noise estimate (added for backwards compatibility)

    Returns:
        fitness: ndarray
            value estimate of the quality of components (the lesser the better)

        erfc: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise

        noise_est: ndarray
            the components ordered according to the fitness
    """

    T = np.shape(traces)[-1]
    
    md = find_modes(traces,axis=1)
    ff1 = traces - md[:,None]

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:

        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0,1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, -1)
        sd_r = np.sqrt(np.sum(ff1**2, -1)/ Ns)

    # compute z value
    z = (traces - md[:,None])/(sigma_factor * sd_r[:,None])

    # probability of observing values larger or equal to z given normal
    # distribution with mean md and std sd_r
    #erf = 1 - norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    #erf = np.log(erf)
    # compute with this numerically stable function
    erf = scipy.special.log_ndtr(-z)

    # moving sum
    erfc = np.cumsum(erf, 1)
    erfc[:, N:] -= erfc[:, :-N]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    return fitness, erfc, sd_r, md