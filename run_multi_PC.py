import time
from get_fields_quick import *
from build_PC_cluster import *
from utils import get_nPaths,pathcat

def run_detection_folders(basePath,nP):
  for mouseName in os.listdir(basePath):
    run_detection(basePath,mouseName,nP)


def run_detection(basePath,mouse,nP,s_start=1,s_end=None,nSession=None,session_order=None,rerun=False):

  pathMouse = pathcat([basePath,mouse])
  if nSession is None:
      nSession,_ = get_nPaths(pathMouse,'Session')

  cMouse = cluster(basePath,mouse,nSession,dataSet='redetect',session_order=session_order,suffix='')
  cMouse.process_sessions(n_processes=nP,reprocess=True)
  cMouse.get_IDs()
  cMouse.get_stats(n_processes=nP)

  # cMouse.load([True,True,True,False,False])
  cMouse.update_status(complete=False,SNR_thr=3,rval_thr=0.5,reliability_thr=0.1,Bayes_thr=10,nCluster_thr=2)
  # return cMouse
  s_end = nSession+1 if s_end is None else s_end+1
  print(s_start,s_end)
  print('run detection on %d sessions for mouse %s'%(nSession,mouse))
  for s in range(s_start,s_end):

    # try:
    IDs = cMouse.IDs['neuronID'][cMouse.status[:,s-1,1],s-1,1].astype('int')
    s_range = np.unique(cMouse.IDs['neuronID'][:,s-1,2])
    s0 = np.unique(cMouse.IDs['neuronID'][:,s-1,2])[0]
    if np.isfinite(s0):
        s0 = int(s0)
        print('Now processing session %d'%s0)
        dPC = detect_PC(basePath,mouse,s0,nP,nbin=100,plt_bool=False,sv_bool=False,suffix='')
        dPC.run_detection(f_max=1,rerun=rerun,dataSet='redetect',assignment=IDs)
    else:
        print('Session not matched')
    # except:
        # print('File not found')
        # raise('Interrupted')

      #print('###--- something went wrong in session %d of mouse %s! ---###'%(s,mouse))


nP = 10
run_detection('/media/wollex/Analyze_AS1/others','551',nP,2)

#run_detection_folders('/media/wollex/Analyze_AS3/Data',nP)
