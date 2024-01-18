
import numpy as np


def gmean(X,axis=1,nanflag=False):

  if nanflag:
    return np.exp(np.nansum(np.log(X),axis)/(~np.isnan(X)).sum(axis))
  else:
    return np.exp(np.sum(np.log(X),axis)/X.shape[axis])
