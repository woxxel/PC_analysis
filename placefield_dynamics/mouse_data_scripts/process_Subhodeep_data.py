'''
    This file contains several scripts to run data analysis on 
    the data from Subhodeep on the GWDG cluster.

    The files are first detected on the cluster, then processing scripts are 
    launched remotely to process data on slurm.
'''

import os

from .utils.connections import *


def run_neuron_detection(path_source, path_target,ssh_conn='hpc-sofja',ssh_config_file_name='id_ed25519_GWDG'):
    """
        Run neuron detection on the GWDG cluster

        Both, path_source and path_target should be paths to folders on the
        GWDG cluster

        The source contains data stored in one folder per mouse, with subfolders
        for each day, containing possibly multiple sessions (up to 3?). They are
        named by their date of recording (YYYYMMDD) and the number of session per 
        day, thus alphabetical sorting provides proper ordering.

        The target folder will contain one folder per recording session, named 
        'SessionXX' with XX being the number of the (overall) session.

    """

    cpus = 8
    mouse = os.path.basename(path_source)

    path_code = '~/program_code/PC_analysis'
    submit_file = f"{path_code}/sbatch_submit.sh"


    ## setting up connection to server
    username = 'schmidt124'
    proxyServerName = 'login.gwdg.de'
    serverName = 'login-dbn02.hpc.gwdg.de'
    
    ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'

    client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)
    sftp_client = client.open_sftp()

    neuron_detection_script = f'{path_code}/neuron_detection_Subhodeep.py'
    ## first, write the neuron detection script to the server

    _, stdout, stderr = client.exec_command(f"""cat > {neuron_detection_script} <<- EOF
import os, sys, shutil, copy
from placefield_dynamics.neuron_detection import *

## obtain input parameters
_, path_source, path_target, cpus = sys.argv
n_processes = int(cpus)

## first, copy over the data to the local tmp folder
tmp_file = os.path.join(os.environ['TMP_LOCAL'],os.path.basename(path_source))
print('copying ', path_source, ' to ', tmp_file)

shutil.copy(path_source, tmp_file)


params = copy.deepcopy(CaImAn_params)
params['dxy']: 512./520.
params['fr']: 15.26
params['decay_time']: 1.75                  

path_to_motion_correct = motion_correct(tmp_file,params,n_processes=n_processes)
print('motion_correct:',path_to_motion_correct)
path_to_neuron_detection = neuron_detection(path_to_motion_correct,params,n_processes=n_processes,save_type='mat')
print('neuron_detection:',path_to_neuron_detection)
                                            
resultFile_name = os.path.split(path_to_neuron_detection)[-1]
os.makedirs(path_target,exist_ok=True)
print('copying to:',os.path.join(path_target,resultFile_name))
shutil.copy(path_to_neuron_detection,os.path.join(path_target,resultFile_name))
print('Finished processing!')
""")

    ## first, search all 'day folders' in the mouse source path
    _, stdout, stderr = client.exec_command(f"ls -d {os.path.join(path_source,'*/')}")
    directories = str(stdout.read(), encoding='utf-8').splitlines()

    s=1
    for dir in directories:

        _, stdout, stderr = client.exec_command(f"ls {os.path.join(dir,'*.tif')}")
        recordings = str(stdout.read(), encoding='utf-8').splitlines()

        ss = 0
        for path_recording in recordings:

            ss += 1

            path_results = os.path.join(path_target,f"Session{s:02d}")
            print(f"{path_recording=}, {path_results=}")
            
            ## make sure, path_results exists
            try:
                sftp_client.mkdir(path_results)
            except:
                pass

            ## writing textfile with date and per-date-session number to results folder
            date_file = os.path.basename(path_recording).split('_')[0]
            date_file += f'_{ss}'
            date_file = os.path.join(path_results,date_file)

            _, stdout, stderr = client.exec_command(f"touch {date_file}")

            ## write bash script to run neuron detection
            _, stdout, stderr = client.exec_command(f"""cat > {submit_file} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s}_detect
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c {cpus}
#SBATCH -t 04:00:00
#SBATCH -o {path_results}/log_neuron_detection.log
#SBATCH -e {path_results}/log_neuron_detection_error.log
#SBATCH --mem=64000

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 {neuron_detection_script} {path_recording} {path_results} {cpus}
EOF
""")
            ## requires absolute path to command, as $PATH is not set for client login
            _, stdout, stderr = client.exec_command(f'/usr/local/slurm/current/install/bin/sbatch {submit_file}')
            # print(stdout.read(),stderr.read())
            # break

            s+=1
        # break

# module use /usr/users/cidbn_sw/sw/modules
# module load cidbn_caiman-1.9.10_py-3.9
# source activate caiman-1.9.10_py-3.9



def run_neuron_matching(path_source,path_target=None,ssh_config_file_name='id_ed25519_GWDG'):
    """
        Run neuron matching on the GWDG cluster

        Both, path_source and path_target should be paths to mouse folders on the
        GWDG cluster, containing SessionXY folders with neuron detection results.
    """

    if path_target is None:
        path_target = path_source

    cpus = 8
    mouse = os.path.basename(path_source)

    path_code = '~/program_code/PC_analysis'
    submit_file = f"{path_code}/sbatch_submit.sh"


    ## setting up connection to server
    username = 'schmidt124'
    proxyServerName = 'login.gwdg.de'
    serverName = 'login-dbn02.hpc.gwdg.de'
    
    ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'

    client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)

    neuron_matching_script = f'{path_code}/neuron_matching_Subhodeep.py'
    ## first, write the neuron detection script to the server

    print(neuron_matching_script)
    print(submit_file)

    _, stdout, stderr = client.exec_command(f"""cat > {neuron_matching_script} <<- EOF
import os, sys
from placefield_dynamics.neuron_matching import *

_, path_source, path_target = sys.argv

match = matching(path_source,paths=None,fileName_results=path_target,suffix='')
match.run_matching(p_thr=[0.3,0.05])
""")


    ## write bash script to run neuron detection
    _, stdout, stderr = client.exec_command(f"""cat > {submit_file} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}_match
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c {cpus}
#SBATCH -t 04:00:00
#SBATCH -o {path_target}/log_neuron_matching.log
#SBATCH -e {path_target}/log_neuron_matching_error.log
#SBATCH --mem=64000

python3 {neuron_matching_script} {path_source} OnACID_results
EOF
""")
    
    _, stdout, stderr = client.exec_command(f'/usr/local/slurm/current/install/bin/sbatch {submit_file}')
    