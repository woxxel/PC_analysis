'''
    This file contains several scripts to run data analysis on 
    the data from Subhodeep on the GWDG cluster.

    The files are first detected on the cluster, then processing scripts are 
    launched remotely to process data on slurm.
'''

import os

from .utils.connections import *


def run_neuron_detection_Subhodeep(path_source, path_target=None,specific_sessions=[],ssh_config_file_name='id_ed25519_GWDG'):
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

    if path_target is None:
        path_target = path_source


    ## setting up connection to server
    username = 'schmidt124'
    proxyServerName = 'login.gwdg.de'
    serverName = 'login-dbn02.hpc.gwdg.de'
    
    ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'

    client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)
    sftp_client = client.open_sftp()

    neuron_detection_script = f'{path_code}/neuron_detection_Subhodeep.py'
    ## first, write the neuron detection script to the server

# params['dxy']: 512./520.
# params['fr']: 15.26
# params['decay_time']: 1.75                  

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

## enables possibility to change parameters
params = copy.deepcopy(CaImAn_params)


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

            if len(specific_sessions):
                if not (s in specific_sessions):
                    s+=1
                    continue

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



def run_neuron_detection(path_source, path_target=None,specific_sessions=[],resultFile_name='results_CaImAn',suffix='',save_type='hdf5',ssh_config_file_name='id_ed25519_GWDG'):
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

    if suffix:
        suffix = f'_{suffix}' if not suffix.startswith('_') else suffix
    
    path_code = '~/program_code/PC_analysis'
    submit_file = f"{path_code}/sbatch_submit.sh"

    if path_target is None:
        path_target = path_source

    ## setting up connection to server
    username = 'schmidt124'
    proxyServerName = 'login.gwdg.de'
    serverName = 'login-dbn02.hpc.gwdg.de'
    
    ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'

    client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)
    sftp_client = client.open_sftp()

    neuron_detection_script = f'{path_code}/run_neuron_detection.py'
    ## first, write the neuron detection script to the server

# params['dxy']: 512./520.
# params['fr']: 15.26
# params['decay_time']: 1.75                  

    _, stdout, stderr = client.exec_command(f"""cat > {neuron_detection_script} <<- EOF
import os, sys, shutil, copy
from placefield_dynamics.neuron_detection import *

## obtain input parameters
_, path_source, path_target, cpus = sys.argv
n_processes = int(cpus)

path_tmp = os.environ['TMP_LOCAL']

## first, check how images are present:
## 1. as image files in /images folder
## 2. as .h5 file in session folder
## 3. as .tif file in session folder
## goal is to have a memmapped .tif file in the session folder

h5Files = [file for file in os.listdir(path_source) if file.endswith('.h5')]
tifFiles = [file for file in os.listdir(path_source) if file.endswith('.tif')]

if len(tifFiles):
    path_to_stack = os.path.join(path_tmp,tifFiles[0])
    print('copying ', tifFiles[0], ' to ', path_to_stack)
    shutil.copy(os.path.join(path_source,tifFiles[0]), path_to_stack)

elif os.path.isdir(os.path.join(path_source,'images')):
    print('stacking single tifs into tif-stack')
    path_to_stack = make_stack_from_single_tifs(
        os.path.join(path_source,'images/'),
        path_tmp,
        data_type='float16',clean_after_stacking=False
    )
elif len(h5Files):
    print('stacking data from .h5 file into tif stack')
    path_to_stack = make_stack_from_h5(
        os.path.join(path_source,h5Files[0]),
        path_tmp,
        data_type='float16',clean_after_stacking=False
    )
else:
    assert False, "No image files found in source folder!"

fileName = os.path.splitext(os.path.basename(path_to_stack))[0]
resultFile_name = '{resultFile_name}_%s{suffix}' % fileName
print(f'finally, saving file to %s'%resultFile_name)

## enables possibility to change parameters
params = copy.deepcopy(CaImAn_params)

path_to_motion_correct = motion_correct(path_to_stack,params,n_processes=n_processes)
print('motion_correct:',path_to_motion_correct)
path_to_neuron_detection = neuron_detection(path_to_motion_correct,params,saveName=resultFile_name,n_processes=n_processes,save_type='{save_type}')
print('neuron_detection:',path_to_neuron_detection)
                                            
os.makedirs(path_target,exist_ok=True)
resultFile_name = os.path.split(path_to_neuron_detection)[-1]   ## required to get extension right
path_resultFile = os.path.join(path_target,resultFile_name)
print('copying to:',path_resultFile)
shutil.copy(path_to_neuron_detection,path_resultFile)
print('Finished processing!')
""")

    ## first, search all session folders in the mouse folder
    _, stdout, stderr = client.exec_command(f"ls -d {os.path.join(path_source,'Session*/ | xargs -n 1 basename')}")
    sessions = str(stdout.read(), encoding='utf-8').splitlines()

    for s,session in enumerate(sessions,start=1):
        sessionName = session
        pathSession_source = os.path.join(path_source,sessionName)
        pathSession_target = os.path.join(path_target,sessionName)
        # continue

        if len(specific_sessions):
            if not (s in specific_sessions):
                continue
        
        print(f'{s}: {pathSession_source=} ,{pathSession_target=}')

        ## make sure, path_results exists
        try:
            sftp_client.mkdir(pathSession_target)
        except:
            pass

        # ## writing textfile with date and per-date-session number to results folder
        # date_file = os.path.basename(path_recording).split('_')[0]
        # date_file += f'_{ss}'
        # date_file = os.path.join(path_results,date_file)
        # _, stdout, stderr = client.exec_command(f"touch {date_file}")

        ## write bash script to run neuron detection
        _, stdout, stderr = client.exec_command(f"""cat > {submit_file} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s}_detect
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c {cpus}
#SBATCH -t 04:00:00
#SBATCH -o {pathSession_target}/log_neuron_detection.log
#SBATCH -e {pathSession_target}/log_neuron_detection_error.log
#SBATCH --mem=64000

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 {neuron_detection_script} {pathSession_source} {pathSession_target} {cpus}
EOF
""")
        ## requires absolute path to command, as $PATH is not set for client login
        _, stdout, stderr = client.exec_command(f'/usr/local/slurm/current/install/bin/sbatch {submit_file}')
        # print(stdout.read(),stderr.read())
        # break

    # break



def run_neuron_matching(pathMouse,fileName='results_CaImAn*',suffix='',fileType='.hdf5',ssh_config_file_name='id_ed25519_GWDG'):
    """
        Run neuron matching on the GWDG cluster

        Both, path_source and path_target should be paths to mouse folders on the
        GWDG cluster, containing SessionXY folders with neuron detection results.
    """

    cpus = 8
    mouse = os.path.basename(pathMouse)

    path_code = '~/program_code/PC_analysis'
    submit_file = f"{path_code}/sbatch_submit.sh"


    ## setting up connection to server
    username = 'schmidt124'
    proxyServerName = 'login.gwdg.de'
    serverName = 'login-dbn02.hpc.gwdg.de'
    
    ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'

    client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)

    neuron_matching_script = f'{path_code}/run_neuron_matching.py'
    ## first, write the neuron detection script to the server

    _, stdout, stderr = client.exec_command(f"""cat > {neuron_matching_script} <<- EOF
import os, sys
from placefield_dynamics.neuron_matching import *

assert len(sys.argv) == 3, "Need to provide two arguments, pathMouse and fileName as arguments! Currently given: %s"%str(sys.argv)
_, pathMouse, fileName = sys.argv

print(pathMouse,fileName,'{suffix}')

_, pathsResults = set_paths_default(pathMouse=pathMouse,fileName_in=fileName,fileType='{fileType}',suffix='{suffix}')
                     
print(pathsResults)
                                            
match = matching(pathMouse,paths=pathsResults,suffix='{suffix}',matlab={fileType.endswith('mat')})
match.run_matching(p_thr=[0.3,0.05])
""")


    ## write bash script to run neuron detection
    _, stdout, stderr = client.exec_command(f"""cat > {submit_file} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}_match
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c {cpus}
#SBATCH -t 02:00:00
#SBATCH -o {pathMouse}/log_neuron_matching.log
#SBATCH -e {pathMouse}/log_neuron_matching_error.log
#SBATCH --mem=32000

python3 {neuron_matching_script} {pathMouse} {fileName}
EOF
""")
    
    _, stdout, stderr = client.exec_command(f'/usr/local/slurm/current/install/bin/sbatch {submit_file}')
    print(stdout.read(),stderr.read())
    


def run_neuron_redetection(pathMouse,resultFiles='results_CaImAn*',specific_sessions=[],ssh_config_file_name='id_ed25519_GWDG'):
    mouse = os.path.basename(pathMouse)
    cpus = 8
    
    path_code = '~/program_code/PC_analysis'
    submit_file = f"{path_code}/sbatch_submit.sh"


    ## setting up connection to server
    username = 'schmidt124'
    proxyServerName = 'login.gwdg.de'
    serverName = 'login-dbn02.hpc.gwdg.de'
    
    ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'

    client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)

    neuron_redetection_script = f'{path_code}/neuron_redetection_Subhodeep.py'

    _, stdout, stderr = client.exec_command(f"""cat > {neuron_redetection_script} <<- EOF
import os, sys
from placefield_dynamics.silence_redetection import *

_, pathMouse, resultFiles, s = sys.argv

pathsImages = sorted([os.path.join(pathMouse,dir,fname) for dir in os.listdir(pathMouse) if (os.path.isdir(os.path.join(pathMouse,dir)) and dir.startswith('202')) for fname in os.listdir(os.path.join(pathMouse,dir)) if os.path.splitext(fname)[-1]=='.tif'])

pathsSession = sorted([os.path.join(pathMouse,session) for session in os.listdir(pathMouse) if session.startswith('Session')])

pathsResults = [os.path.join(pathMouse,session,fname) for session in pathsSession for fname in os.listdir(os.path.join(pathMouse,session)) if (fname.startswith(resultFiles[:-1]) and not ('redetect' in fname))]

print(pathsSession)
print(pathsResults)
print(pathsImages)

sr = silence_redetection(pathsSession,pathsResults,pathsImages,matlab=True)
sr.process_session(int(s),ssh_alias=None)
""")

    _, stdout, stderr = client.exec_command(f"ls -d {os.path.join(pathMouse,'Session*/')}")
    directories = str(stdout.read(), encoding='utf-8').splitlines()

    for s,dir in enumerate(directories):

        if len(specific_sessions):
            if not ((s+1) in specific_sessions):
                s+=1
                continue
        # print(s,dir)
        # print(f'python3 {neuron_redetection_script} {pathMouse} {resultFiles} {s}')

        ## write bash script to run neuron detection
        _, stdout, stderr = client.exec_command(f"""cat > {submit_file} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s+1}_redetection
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c {cpus}
#SBATCH -t 2:00:00
#SBATCH -o {dir}/log_neuron_redetection.log
#SBATCH -e {dir}/log_neuron_redetection_error.log
#SBATCH --mem=64000

python3 {neuron_redetection_script} {pathMouse} {resultFiles} {s}
EOF
""")
        
        _, stdout, stderr = client.exec_command(f'/usr/local/slurm/current/install/bin/sbatch {submit_file}')
        print(stdout.read(),stderr.read())

        # break
