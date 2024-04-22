from tqdm import *
import numpy as np
from .cluster_analysis import cluster


class multi_cluster:
    def __init__(self):
        self.cMice = {}
        self.set_sessions()

    def set_sessions(self,start_ses=None):
        self.sessions = {'34':   {'total':22,
                             'order':range(1,23),
                             'analyze':[3,15],
                             'steady':  [3,15]},
                    '35':   {'total':22,
                             'order':range(1,23),
                             'analyze':[3,15],
                             'steady':  [3,64]},
                    '65':   {'total':44,
                             'order':range(1,45),
                             'analyze':[1,20],
                             'steady':  [3,20]}, ## maybe until 28
                    '66':   {'total':45,
                             'order':range(1,46),
                             'analyze':[1,21],
                             'steady':  [3,20]}, ## maybe until 29
                    '72':   {'total':44,
                             'order':range(1,45),
                             'analyze':[1,20],
                             'steady':  [3,20]}, ## maybe until 28
                    '243':  {'total':   71,
                             'order':[range(69,72),range(66,69),range(57,60),range(63,66),range(60,63),range(54,57),range(51,54),range(1,51)],
                             'analyze': [1,71],
                             'steady':  [53,71]},
                    '245':  {'total':73,
                             'order':[range(68,74),range(65,68),range(62,65),range(1,53),range(56,62),range(53,56)],
                             'analyze':[1,73],
                             'steady':  [45,64]},
                    '246':  {'total':63,
                             'order':[range(1,32),range(32,43),range(46,49),range(61,64),range(58,61),range(55,58),range(49,55),range(43,46)],       ## achtung!! between 32 & 32 there are 10 measurements missing over 150h
                             'analyze':[1,60],
                             'steady':  [33,60]},
                    '756':  {'total':30,    ## super huge gap between 22 & 23
                             'order':range(1,31),
                             'analyze':[1,20],
                             'steady':  [3,15]},
                    '757':  {'total':28,    ## super huge gap between 20 & 21
                             'order':range(1,29),
                             'analyze':[1,20],
                             'steady':  [3,15]},
                    '758':  {'total':28,    ## super huge gap between 20 & 21
                             'order':range(1,29),
                             'analyze':[1,20],
                             'steady':  [3,20]},
                    '839':  {'total':24,    ## cpp injections starting at 20
                             'order':range(1,25),
                             'analyze':[3,20],
                             'steady':[3,20]},
                    '840':  {'total':25,    ## cpp injections starting at 20
                             'order':range(1,25),
                             'analyze':[1,18],
                             'steady':  [3,18]},
                    '841':  {'total':np.NaN,
                             'analyze':[np.NaN,np.NaN]},
                    '842':  {'total':np.NaN,
                             'analyze':[np.NaN,np.NaN]},
                    '879':  {'total':15,
                             'order':range(1,16),
                             'analyze':[1,15],
                             'steady':  [3,15]},
                    '882':  {'total':np.NaN,
                             'analyze':[np.NaN,np.NaN]},
                    '884':  {'total':24,    ## cpp injections starting at 20
                             'order':range(1,25),
                             'analyze':[3,16],
                             'steady':[3,16]},
                    '886':  {'total':24,    ## saline injections starting at 22
                             'order':range(1,25),
                             'analyze':[3,19]}, ## dont know yet
                    '549':  {'total':29,    ## super huge gap between 20 & 21, kainate acid @ s21
                             'order':range(1,30),
                             'analyze':[3,20],
                             'steady':  [3,20]}, #bad matching?!
                    '551':  {'total':28,    ## super huge gap between 20 & 21, kainate acid @ s21
                             'order':range(1,29),
                             'analyze':[3,20],
                             'steady':  [3,20]},
                    '918shKO':  {'total':28,    # RW change at s16
                                 'order':range(1,29),
                             'analyze':[3,15],
                             'steady':  [3,15]},
                    '931wt':    {'total':28,    # RW change at s16
                                 'order':range(1,29),
                             'analyze':[3,15],
                             'steady':  [3,15]},
                    '943shKO':  {'total':28,    # RW change at s16
                                 'order':range(1,29),
                             'analyze':[3,15],
                             'steady':  [3,15]},
                    '231':  {'total':   87,
                             'order':   range(1,88),   ## RW change at s11, s21, s31
                             'analyze': [1,87],
                             'steady':  [33,87]},
                    '232':  {'total':   74,
                             'order':   range(1,75),   ## RW change at s73, s83, s93,s94,s95
                             'analyze': [18,72],
                             'steady':  [18,72]},
                    '236':  {'total':28,
                             'order':range(1,29),
                             'analyze':[3,28],
                             'steady':  [3,28]},
                    '762':  {'total':   112,
                             'order':   range(1,113),
                             'analyze': [1,112],
                             'steady':  [17,87]},
                    }
        if not (start_ses is None):
            for mouse in self.cMice.keys():
                self.sessions[mouse]['analyze'][0] = start_ses



    def load_mice(self,mice,load=False,reload=False,session_start=None,suffix=''):

        for mouse in mice:
            if (not (mouse in self.cMice.keys())) | reload:

                if mouse in ['34','35','65','66','72','243','244','756','757','758','839','840','841','842','879','882','884','886']:
                    basePath = '/media/wollex/Analyze_AS3/Data'
                elif mouse in ['549','551']:
                    basePath = '/media/wollex/Analyze_AS1/others'
                elif mouse in ['918shKO','931wt','943shKO']:
                    basePath = '/media/wollex/Analyze_AS1/Shank'
                elif mouse in ['231','232','236','245','246','762']:
                    basePath = '/media/wollex/Analyze_AS1/linstop'
                else:
                    print('Mouse %s does not exist'% mouse)

                self.cMice[mouse] = cluster(basePath,mouse,self.sessions[mouse]['total'],session_order=self.sessions[mouse]['order'],dataSet='redetect',suffix=suffix)
                if not load:
                    self.cMice[mouse].run_complete(sessions=self.sessions[mouse]['analyze'],n_processes=10,reprocess=True)
                else:
                    self.cMice[mouse].load()

        # if load:
            # self.update_all(which=['sessions','status','compare'],session_start=session_start)

    def update_all(self,mice=None,which=None,SNR_thr=2,rval_thr=0,Bayes_thr=10,rel_thr=0.1,A_thr=3,A0_thr=1,Arate_thr=2,pm_thr=0.3,alpha=1,nCluster_thr=2,session_start=None,session_end=None,sd_r=-1,steady=False):

        if mice is None:
            mice = self.cMice.keys()

        progress = tqdm(mice,leave=True)


        for mouse in tqdm(mice):
            progress.set_description('updating mouse %s...'%mouse)
            if 'sessions' in which:
                if steady:
                    ses = self.sessions[mouse]['steady']
                else:
                    ses = self.sessions[mouse]['analyze']

                if not (session_start is None):
                    ses[0] = session_start
                if ((not (session_end is None)) & (not (session_end==-1))):
                    ses[1] = session_end
                elif session_end==-1:
                    ses[1] = self.sessions[mouse]['total']
                print(mouse)
                print(ses)
                self.cMice[mouse].classify_sessions(ses)
                # self.cMice[mouse].classify_sessions()
            if 'behavior' in which:
                self.cMice[mouse].get_behavior()
            if 'stats' in which:
                self.cMice[mouse].get_stats(n_processes=4)
            elif 'firingrate' in which:
                _,_ = self.cMice[mouse].recalc_firingrate(sd_r)

            if 'status' in which:
                self.cMice[mouse].update_status(SNR_thr=SNR_thr,rval_thr=rval_thr,Bayes_thr=Bayes_thr,reliability_thr=rel_thr,A_thr=A_thr,A0_thr=A0_thr,Arate_thr=Arate_thr,MI_thr=0,pm_thr=pm_thr,alpha=alpha,nCluster_thr=nCluster_thr)
            if 'compare' in which:
                self.cMice[mouse].compareSessions(reprocess=True,n_processes=10)
            if 'transition' in which:
                # if not ('stability' in self.cMice[mouse].__dict__):
                    # plot_PC_analysis(self.cMice[mouse],plot_arr=[6],N_bs=100,n_processes=10,sv=False,reprocess=True)
                self.cMice[mouse].get_transition_prob()
                self.cMice[mouse].get_locTransition_prob()
            if 'save' in which:
                self.cMice[mouse].save()