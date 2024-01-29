import os, sys
from placefield_dynamics.neuron_matching import *

_, datapath, result_files, suffix, dataset, mouse, cpus = sys.argv

mousePath = os.path.join(datapath,dataset,mouse)

paths = [os.path.join(mousePath,sessionPath,result_files) for sessionPath in os.listdir(mousePath) if 'Session' in sessionPath]
paths.sort()

match = matching(mousePath,paths,fileName_results=result_files,suffix=suffix)
match.run_matching(p_thr=[0.3,0.05])