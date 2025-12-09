import numpy as np
from collections import Counter


def get_ICPI(status,mode='ICI'):
    pad = 1
    ISI = np.zeros(status.shape[1])
    for stat in status:
        stat = np.pad(stat,pad_width=(pad,pad),constant_values=True)
        if np.any(stat):
            dCoding = np.diff(np.where(stat)[0])
            dCoding_cleaned = dCoding[dCoding>1]
            if len(dCoding_cleaned)>0:
                for key,val in Counter(dCoding_cleaned).items():
                    if key<status.shape[1]:
                        ISI[key-1] += val
    return ISI



def get_dp(status,status_dep=None,status_session=None,ds=1,mode='act'):

    if status_session is None:
        status_session = np.ones(status.shape[1],'bool')
    if status_dep is None:
        status_dep = np.ones_like(status,'bool')
        status_dep[:,~status_session] = False
    # before = np.pad((status[:,:-ds]&status_dep[:,:-ds]),pad_width=((0,0),(ds,0)),constant_values=False)
    # if mode=='act':
    #     cont_score = (before&status)[:,status_session].sum(1)/before[:,status_session].sum(1)
    #     p_tmp = status[:,status_session].sum(1)/status_dep[:,status_session].sum(1)
    #     dp = cont_score-p_tmp
    # elif mode=='PC':
    #     cont_score = (before&status&status_dep)[:,status_session].sum(1)/(before&status_dep)[:,status_session].sum(1)
    #     p_tmp = ((before&status_dep)[:,status_session].sum(1)-1)/(status_dep[:,ds:]&status_dep[:,:-ds]).sum(1)
    #     dp = cont_score-p_tmp

    session_bool = np.pad(status_session[ds:],(0,ds),constant_values=False) & np.pad(status_session,(0,0),constant_values=False)

    reactivation = np.zeros_like(status,'bool')
    reactivation_possibilities = np.zeros_like(status,'bool')
    activity_instances = np.zeros_like(status,'bool')

    cont_score = np.zeros_like(status,'bool')

    status_complete = status & status_dep
    # if mode=='act':
    #     for s in np.where(session_bool)[0]:
            # reactivation[:,s] = status_complete[:,s] & status_complete[:,s+ds]
        # reactivation_possibilities = status_complete[:,session_bool].sum(1)
        # cont_score = reactivation.sum(1)/reactivation_possibilities
        # p_tmp = status_complete[:,session_bool].sum(1)/status_dep[:,session_bool].sum(1)
        # dp = cont_score-p_tmp
    # if mode=='PC':
    for s in np.where(session_bool)[0]:
        reactivation[:,s] = status_complete[:,s] & status_complete[:,s+ds]
        reactivation_possibilities[:,s] = status_complete[:,s] & status_dep[:,s+ds]

        activity_instances[:,s] = (status_dep[:,s] & status_dep[:,s+ds])

    cont_score = reactivation.sum(1)/reactivation_possibilities.sum(1)
    p_tmp = reactivation_possibilities.sum(1) / activity_instances.sum(1)
    dp = cont_score-p_tmp

    return dp,cont_score
