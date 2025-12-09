import os, sys, shutil
from turnover_dynamics.silence_redetection import silence_redetection

print('Input variables:',sys.argv)
_, datapath_in, datapath_out, dataset, mouse, s, session_name, cpus = sys.argv
n_processes = int(cpus)


print('Starting up silence redetection')
sr = silence_redetection(dataset=dataset,mouse=mouse,
                        path_detected=datapath_out,path_images=datapath_in)

sr.process_session(s=int(s),n_processes=8,
    path_tmp=os.environ['TMP_LOCAL'],
    ssh_alias=None,plt_bool=True
)

# print("DONE! Removing temporary session folder")
# shutil.rmtree(path_tmp)