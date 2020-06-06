import time
from get_fields import *

from utils import get_nPaths,pathcat

def run_detection_folders(basePath,nP):
  for mouseName in os.listdir(basePath):
    run_detection(basePath,mouseName,nP)


def run_detection(basePath,mouse,nP,s_start=1,s_end=None,rerun=False):
  
  pathMouse = pathcat([basePath,mouse])
  nSession,tmp = get_nPaths(pathMouse,'Session')
  s_end = nSession+1 if s_end is None else s_end+1
  print(s_start,s_end)
  print('run detection on %d sessions for mouse %s'%(nSession,mouse))
  for s in range(s_start,s_end):
    #try:
    dPC = detect_PC(basePath,mouse,s,nP,plt_bool=False,sv_bool=False)
    dPC.run_detection(rerun=rerun,dataSet='redetect')
    #except:
      #print('###--- something went wrong in session %d of mouse %s! ---###'%(s,mouse))


nP = 12
#run_detection_folders('/media/wollex/Analyze_AS3/Data',nP)

#run_detection('/media/wollex/Analyze_AS3/Data','879',nP,7)
#run_detection('/media/wollex/Analyze_AS3/Data','34',nP)
#run_detection('/media/wollex/Analyze_AS3/Data','35',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS3/Data','65',nP,38)
#run_detection('/media/wollex/Analyze_AS3/Data','66',nP,41,41)
#run_detection('/media/wollex/Analyze_AS3/Data','72',nP)

run_detection('/media/wollex/Analyze_AS1/linstop','762',nP,8,8)
#run_detection('/media/wollex/Analyze_AS1/Shank','918shKO',nP)
#run_detection('/media/wollex/Analyze_AS1/Shank','931wt',nP,22)
#run_detection('/media/wollex/Analyze_AS1/Shank','943shKO',nP)
#time.sleep(2000)
#run_detection('/media/wollex/Analyze_AS1/others','549',nP,22)
#run_detection('/media/wollex/Analyze_AS1/others','756',nP)
#run_detection('/media/wollex/Analyze_AS1/others','757',nP)
#run_detection('/media/wollex/Analyze_AS1/others','758',nP)
#run_detection('/media/wollex/Analyze_AS1/others','551',nP)

#run_detection('/media/wollex/Analyze_AS1/linstop','245',nP,58)
#run_detection_folders('/media/wollex/Analyze_AS1/Shank',nP,25,rerun=True)
#run_detection('/media/wollex/Analyze_AS1/Shank','918shKO',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS1/Shank','931wt',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS1/Shank','943shKO',nP,rerun=True)

#run_detection('/media/wollex/Analyze_AS1/linstop','231',nP,42)
#run_detection('/media/wollex/Analyze_AS1/linstop','232',nP,18,30)
#run_detection('/media/wollex/Analyze_AS1/linstop','236',nP)
#run_detection('/media/wollex/Analyze_AS1/linstop','246',nP)



