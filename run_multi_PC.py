from get_fields import *

from utils import get_nPaths,pathcat

def run_detection_folders(basePath,nP):
  for mouseName in os.listdir(basePath):
    run_detection(basePath,mouseName,nP)


def run_detection(basePath,mouse,nP,s_start=1,rerun=False):
  
  pathMouse = pathcat([basePath,mouse])
  nSession,tmp = get_nPaths(pathMouse,'Session')
  print('run detection on %d sessions for mouse %s'%(nSession,mouse))
  for s in range(s_start,nSession+1):
    #try:
    dPC = detect_PC(basePath,mouse,s,nP,plt_bool=False,sv_bool=False)
    dPC.run_detection(rerun=rerun)
    #except:
      #print('###--- something went wrong in session %d of mouse %s! ---###'%(s,mouse))

nP = 12
#run_detection_folders('/media/wollex/Analyze_AS3/Data',nP)

#run_detection('/media/wollex/Analyze_AS3/Data','34',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS3/Data','35',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS3/Data','65',nP,32,rerun=True)
#run_detection('/media/wollex/Analyze_AS3/Data','66',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS3/Data','72',nP,rerun=True)

#run_detection('/media/wollex/Analyze_AS1/linstop','762',nP)
#run_detection('/media/wollex/Analyze_AS1/Shank','918shKO',nP)
run_detection('/media/wollex/Analyze_AS1/Shank','931wt',nP,17)
run_detection('/media/wollex/Analyze_AS1/Shank','948shKO',nP)

#run_detection('/media/wollex/Analyze_AS1/others','549',nP)
#run_detection('/media/wollex/Analyze_AS1/others','756',nP)
#run_detection('/media/wollex/Analyze_AS1/others','757',nP)
#run_detection('/media/wollex/Analyze_AS1/others','758',nP)
#run_detection('/media/wollex/Analyze_AS1/others','551',nP)

#run_detection('/media/wollex/Analyze_AS1/linstop','245',nP,58)
#run_detection_folders('/media/wollex/Analyze_AS1/Shank',nP,25,rerun=True)
#run_detection('/media/wollex/Analyze_AS1/Shank','918shKO',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS1/Shank','931wt',nP,rerun=True)
#run_detection('/media/wollex/Analyze_AS1/Shank','943shKO',nP,rerun=True)

#run_detection('/media/wollex/Analyze_AS1/linstop','231',nP)
#run_detection('/media/wollex/Analyze_AS1/linstop','232',nP)
#run_detection('/media/wollex/Analyze_AS1/linstop','236',nP)
#run_detection('/media/wollex/Analyze_AS1/linstop','246',nP)



