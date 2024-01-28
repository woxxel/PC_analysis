import os, sys, shutil
from placefield_dynamics.silence_redetection import *

print('IMPORT ANALYSIS PACKAGE AS A WHOLE AS MODULE AND MOVE BATCH SUBMISSION SCRIPTS OUT OF MODULE!!')

_, datapath_in, datapath_out, dataset, mouse, s, session_name, cpus = sys.argv
n_processes = int(cpus)

print('Input variables:',datapath_in, datapath_out, dataset, mouse, s, session_name, n_processes)

print('Starting up silence redetection')
sr = silence_redetection(dataset=dataset,mouse=mouse,
                        data_pre=datapath_out,source=datapath_in)

path_tmp=os.path.join(os.environ['TMP_LOCAL'],dataset,mouse,session_name)
print(path_tmp)

sr.process_session(s=s,n_processes=8,
    path_tmp=path_tmp,
    ssh_alias=None,plt_bool=True
)

print("DONE! Removing temporary session folder")
shutil.rmtree(path_tmp)


