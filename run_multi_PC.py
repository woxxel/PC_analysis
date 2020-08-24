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
    try:
        dPC = detect_PC(basePath,mouse,s,nP,nbin=100,plt_bool=False,sv_bool=False)
        dPC.run_detection(f_max=1,rerun=rerun,dataSet='redetect')
    except OSError:
        print('File not found')
        # raise('Interrupted')

      #print('###--- something went wrong in session %d of mouse %s! ---###'%(s,mouse))


nP = 12

run_detection('/media/wollex/Analyze_AS3/Data','756',nP)
run_detection('/media/wollex/Analyze_AS3/Data','757',nP)
run_detection('/media/wollex/Analyze_AS3/Data','758',nP)

run_detection('/media/wollex/Analyze_AS3/Data','34',nP)
run_detection('/media/wollex/Analyze_AS3/Data','35',nP)

run_detection('/media/wollex/Analyze_AS3/Data','65',nP)
run_detection('/media/wollex/Analyze_AS3/Data','66',nP)
run_detection('/media/wollex/Analyze_AS3/Data','72',nP)


run_detection('/media/wollex/Analyze_AS3/Data','839',nP)
run_detection('/media/wollex/Analyze_AS3/Data','840',nP)
run_detection('/media/wollex/Analyze_AS3/Data','840',nP)
# run_detection('/media/wollex/Analyze_AS3/Data','884',nP)

# run_detection('/media/wollex/Analyze_AS3/Data','243',nP,52)
# run_detection('/media/wollex/Analyze_AS1/linstop','245',nP,50)
# run_detection('/media/wollex/Analyze_AS1/linstop','246',nP)
#run_detection_folders('/media/wollex/Analyze_AS3/Data',nP)
