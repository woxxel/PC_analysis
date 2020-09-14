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
      nSession,tmp = get_nPaths(pathMouse,'Session')

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
    s0 = int(np.unique(cMouse.IDs['neuronID'][:,s-1,2])[0])
    print('Now processing session %d'%s0)
    dPC = detect_PC(basePath,mouse,s0,nP,nbin=100,plt_bool=False,sv_bool=False,suffix='')
    dPC.run_detection(f_max=1,rerun=rerun,dataSet='redetect',assignment=IDs)
    # except:
        # print('File not found')
        # raise('Interrupted')

      #print('###--- something went wrong in session %d of mouse %s! ---###'%(s,mouse))


nP = 12
# run_detection('/media/wollex/Analyze_AS3/Data','756',nP,1,20)
# run_detection('/media/wollex/Analyze_AS3/Data','757',nP,1,20)
# run_detection('/media/wollex/Analyze_AS3/Data','758',nP,1,20)

# run_detection('/media/wollex/Analyze_AS1/linstop','762',nP,69,87)
# run_detection('/media/wollex/Analyze_AS1/linstop','762',nP,68,87)

# run_detection('/media/wollex/Analyze_AS3/Data','879',nP)
# run_detection('/media/wollex/Analyze_AS3/Data','840',nP)

# run_detection('/media/wollex/Analyze_AS3/Data','65',nP,1,20)
# run_detection('/media/wollex/Analyze_AS3/Data','66',nP,1,20)
# run_detection('/media/wollex/Analyze_AS3/Data','72',nP,1,20)

# run_detection('/media/wollex/Analyze_AS1/linstop','762',nP,88)
# run_detection('/media/wollex/Analyze_AS1/linstop','246',nP,session_order=[range(1,32),range(32,43),range(46,49),range(61,64),range(58,61),range(55,58),range(49,55),range(43,46)])
# run_detection('/media/wollex/Analyze_AS1/linstop','246',nP,session_order=[range(1,32),range(32,43),range(46,49),range(61,64),range(58,61),range(55,58),range(49,55),range(43,46)],s_start=42)
# run_detection('/media/wollex/Analyze_AS3/Data','243',nP,session_order=[range(69,72),range(66,69),range(57,60),range(63,66),range(60,63),range(54,57),range(51,54),range(1,51)])
# run_detection('/media/wollex/Analyze_AS1/linstop','245',nP,session_order=[range(68,74),range(65,68),range(62,65),range(1,53),range(56,62),range(53,56)])

# run_detection('/media/wollex/Analyze_AS1/linstop','232',nP)

# run_detection('/media/wollex/Analyze_AS3/Data','34',nP)
# run_detection('/media/wollex/Analyze_AS3/Data','35',nP)

run_detection('/media/wollex/Analyze_AS1/linstop','231',nP,nSession=87)
run_detection('/media/wollex/Analyze_AS3/Data','65',nP,21)
run_detection('/media/wollex/Analyze_AS3/Data','66',nP,21)
run_detection('/media/wollex/Analyze_AS3/Data','72',nP,21)


#
#
# run_detection('/media/wollex/Analyze_AS3/Data','65',nP)
# run_detection('/media/wollex/Analyze_AS3/Data','66',nP)
# run_detection('/media/wollex/Analyze_AS3/Data','72',nP)
#
# run_detection('/media/wollex/Analyze_AS3/Data','884',nP)

#run_detection_folders('/media/wollex/Analyze_AS3/Data',nP)
