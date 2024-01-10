import numpy as np

def get_session_specifics(mouse,nSes):

  ## put in here as well position around gate and position around reward, probability of reward reception, etc
  #t_measures = [];
    t_measures = np.NaN
    session_data = {'RW_pos':   np.NaN,
                    'GT_pos':   np.NaN,
                    'delay':    0,               ## required dwelltime at reward location
                    'p_RW':     np.ones(nSes)}   ## probability to receive reward

    rw1 = [50,70]
    rw2 = [75,95]
    rw3 = [25,45]

    gt1 = [15,35]
    gt2 = [65,85]

    if mouse in ['231']:
        session_data['RW_pos'] = np.ones((nSes,2))*rw1
        session_data['RW_pos'][10:20,:] = rw2
        session_data['RW_pos'][20:30,:] = rw3

        session_data['GT_pos'] = np.ones((nSes,2))*np.NaN

        session_data['delay'] = np.ones(nSes)*2

    elif mouse in ['232']:
        session_data['RW_pos'] = np.ones((nSes,2))*rw1
        session_data['RW_pos'][72:82,:] = rw2
        # session_data['RW_pos'][82:92,:] = rw3
        # session_data['RW_pos'][94,:] = rw2

        session_data['GT_pos'] = np.ones((nSes,2))*np.NaN

        session_data['delay'] = np.ones(nSes)*2
        session_data['delay'][:10] = 0
        session_data['delay'][10:23] = 1
        session_data['delay'][23:41] = 1.5

        # session_data['delay'][96] = 0

    elif mouse in ['236']:
        session_data['RW_pos'] = np.ones((nSes,2))*rw1

        session_data['GT_pos'] = np.ones((nSes,2))*np.NaN

        session_data['delay'] = np.ones(nSes)*0
        session_data['delay'][11:] = 1


    elif mouse in ["839","840","841","842","879","882","884","886"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2

        session_data['GT_pos'] = np.ones((nSes,2))*gt1

        session_data['delay'] = np.ones(nSes)*0

    elif mouse in ["34","35"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2
        session_data['RW_pos'][15:,:] = rw3

        session_data['GT_pos'] = np.ones((nSes,2))*gt1

        session_data['delay'] = np.ones(nSes)*0

    elif mouse in ["243"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw1
        session_data['RW_pos'][31:41,:] = rw2
        session_data['RW_pos'][41:51,:] = rw3

        session_data['GT_pos'] = np.ones((nSes,2))*np.NaN

        session_data['delay'] = np.ones(nSes)*2
        session_data['delay'][10] = 0
        session_data['delay'][4] = 0    #not sure


    elif mouse in ["245"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw1
        session_data['RW_pos'][22:32,:] = rw2
        session_data['RW_pos'][32:42,:] = rw3

        session_data['GT_pos'] = np.ones((nSes,2))*np.NaN

        session_data['delay'] = np.ones(nSes)*2
        session_data['delay'][1] = 0    #not sure
        session_data['delay'][8] = 0

    elif mouse in ["246"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw1
        session_data['RW_pos'][10:20,:] = rw2
        session_data['RW_pos'][20:30,:] = rw3

        session_data['GT_pos'] = np.ones((nSes,2))*np.NaN

        session_data['delay'] = np.ones(nSes)*2
        session_data['delay'][49] = 0    #not sure
        session_data['delay'][52] = 0

    elif mouse in ["762"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw1
        session_data['RW_pos'][87:97,:] = rw2
        session_data['RW_pos'][97:108,:] = rw3
        session_data['RW_pos'][109,:] = rw2

        session_data['GT_pos'] = np.ones((nSes,2))*np.NaN

        session_data['delay'] = np.ones(nSes)*2
        session_data['delay'][:11] = 0    #not sure
        session_data['delay'][11:31] = 1
        session_data['delay'][31:52] = 1.5
        session_data['delay'][111] = 0

    elif mouse in ["756","757","758"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2
        session_data['RW_pos'][15:,:] = rw3 #not sure, what's after s20

        session_data['GT_pos'] = np.ones((nSes,2))*gt1

        session_data['delay'] = np.ones(nSes)*0


    elif mouse in ["918shKO","931wt","943shKO"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2
        session_data['RW_pos'][15:,:] = rw3

        session_data['GT_pos'] = np.ones((nSes,2))*gt1

        session_data['delay'] = np.ones(nSes)*0

        session_data['p_RW'][20:] = 1/3 ## reliably every 3rd time!

    elif mouse in ["65"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2
        session_data['RW_pos'][24:26,:] = np.NaN

        session_data['GT_pos'] = np.ones((nSes,2))*gt1
        session_data['GT_pos'][20:22,:] = np.NaN
        session_data['GT_pos'][28:30,:] = gt2
        session_data['GT_pos'][32:39,:] = gt2
        session_data['GT_pos'][42:44,:] = gt1 ## and gt2

        session_data['delay'] = np.ones(nSes)*0

        session_data['p_RW'][16:18] = 1/2
        session_data['p_RW'][38] = 1/2

    elif mouse in ["66"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2
        session_data['RW_pos'][4,:] = np.NaN
        session_data['RW_pos'][25:27,:] = np.NaN
        session_data['RW_pos'][36,:] = np.NaN

        session_data['GT_pos'] = np.ones((nSes,2))*gt1
        session_data['GT_pos'][4,:] = np.NaN
        session_data['GT_pos'][21:23,:] = np.NaN
        session_data['GT_pos'][29:31,:] = gt2
        session_data['GT_pos'][33:40,:] = gt2
        session_data['GT_pos'][36,:] = np.NaN
        session_data['GT_pos'][43:45,:] = gt1 ## and gt2

        session_data['delay'] = np.ones(nSes)*0

        session_data['p_RW'][16:18] = 1/2
        session_data['p_RW'][29] = 1/2
        session_data['p_RW'][39] = 1/2

    elif mouse in ["72"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2
        session_data['RW_pos'][24:26,:] = np.NaN
        session_data['RW_pos'][28:30,:] = rw3
        session_data['RW_pos'][32:39,:] = rw3
        session_data['RW_pos'][42:44,:] = rw2 ## & rw3

        session_data['GT_pos'] = np.ones((nSes,2))*gt1
        session_data['GT_pos'][20:22,:] = np.NaN

        session_data['delay'] = np.ones(nSes)*0

        session_data['p_RW'][2] = 1/2
        session_data['p_RW'][16:18] = 1/2
        session_data['p_RW'][29] = 1/2
        session_data['p_RW'][38] = 1/2


    elif mouse in ["549","551"]:
        session_data['RW_pos'] = np.ones((nSes,2))*rw2
        session_data['RW_pos'][8,:] = np.NaN
        session_data['RW_pos'][15:20,:] = rw3

        session_data['GT_pos'] = np.ones((nSes,2))*gt1

        session_data['delay'] = np.ones(nSes)*0


    return session_data
