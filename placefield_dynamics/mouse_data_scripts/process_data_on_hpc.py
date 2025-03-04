"""
    This file contains several scripts to run data analysis on 
    the data from Subhodeep on the GWDG cluster.

    The files are first detected on the cluster, then processing scripts are 
    launched remotely to process data on slurm.
"""

import os, time
from pathlib import Path
from .utils.connections import *


def run_neuron_detection_Subhodeep(
    path_source,
    path_target=None,
    specific_sessions=[],
    resultFile_name="results_CaImAn",
    suffix="",
    save_type="mat",
    hpc="sofja",
):
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

    client, path_code, batch_params = set_hpc_params(hpc)

    if path_target is None:
        path_target = path_source

    # sftp_client = client.open_sftp()

    neuron_detection_script = f"{path_code}/neuron_detection.py"
    ## first, write the neuron detection script to the server

    # params['dxy']: 512./520.
    # params['fr']: 15.26
    # params['decay_time']: 1.75

    _, stdout, stderr = client.exec_command(
        f"""cat > {neuron_detection_script} <<- EOF
import os, sys, shutil, copy
from placefield_dynamics.neuron_detection import *

## obtain input parameters
_, path_source, path_target, cpus = sys.argv
n_processes = int(cpus)

## first, copy over the data to the local tmp folder
tmp_file = os.path.join(os.environ['TMP_LOCAL'],os.path.basename(path_source))
print('copying ', path_source, ' to ', tmp_file)

shutil.copy(path_source, tmp_file)

fileName = os.path.splitext(os.path.basename(path_source))[0]
resultFile_name = '{resultFile_name}_%s{suffix}' % fileName
print(f'finally, saving file to %s'%resultFile_name)

## enables possibility to change parameters
params = copy.deepcopy(CaImAn_params)
params['fr'] = 30.05

path_to_motion_correct = motion_correct(tmp_file,params,n_processes=n_processes)
print('motion_correct:',path_to_motion_correct)
path_to_neuron_detection = neuron_detection(path_to_motion_correct,params,saveName=resultFile_name,n_processes=n_processes,save_type='{save_type}')
print('neuron_detection:',path_to_neuron_detection)
                                            
resultFile_name = os.path.split(path_to_neuron_detection)[-1]
os.makedirs(path_target,exist_ok=True)
path_resultFile = os.path.join(path_target,resultFile_name)
print('copying to:',path_resultFile)
shutil.copy(path_to_neuron_detection,path_resultFile)

print('Finished processing!')
"""
    )

    ## first, search all 'day folders' in the mouse source path
    _, stdout, stderr = client.exec_command(f"ls -d {os.path.join(path_source,'*/')}")
    directories = str(stdout.read(), encoding="utf-8").splitlines()
    s = 1
    for dir in directories:

        _, stdout, stderr = client.exec_command(f"ls {os.path.join(dir,'*.tif')}")
        recordings = str(stdout.read(), encoding="utf-8").splitlines()

        ss = 0
        for path_recording in recordings:

            ss += 1

            if len(specific_sessions):
                if not (s in specific_sessions):
                    s += 1
                    continue

            # path_results = dir[:-1]
            path_results = Path(path_target) / Path(dir).stem
            # path_results = os.path.join(path_target,f"Session{s:02d}")
            print(f"{path_recording=}, {path_results=}")
            # break

            # ## make sure, path_results exists
            # try:
            #     sftp_client.mkdir(path_results)
            # except:
            #     pass
            base_name = os.path.splitext(os.path.basename(path_recording))[0]

            ## writing textfile with date and per-date-session number to results folder
            # date_file = os.path.basename(path_recording).split('_')[0]
            # date_file += f'_{ss}'
            # date_file = os.path.join(path_results,date_file)

            # _, stdout, stderr = client.exec_command(f"touch {date_file}")

            print(
                f"writing log to: {path_results}/log_neuron_detection_{base_name}.log"
            )
            ## write bash script to run neuron detection
            _, stdout, stderr = client.exec_command(
                f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s}_detect
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 04:00:00
#SBATCH -o {path_results}/log_neuron_detection_{base_name}.log
#SBATCH -e {path_results}/log_neuron_detection_{base_name}.log
#SBATCH --mem=64000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 {neuron_detection_script} {path_recording} {path_results} {cpus}
EOF
"""
            )
            ## requires absolute path to command, as $PATH is not set for client login
            _, stdout, stderr = client.exec_command(
                f"/usr/local/slurm/current/install/bin/sbatch {batch_params['submit_file']}"
            )
            print(stdout.read(), stderr.read())
            # break

            s += 1
        # break


def run_neuron_detection(
    path_source,
    path_target=None,
    specific_sessions=[],
    resultFile_name="results_CaImAn",
    suffix="",
    save_type="hdf5",
    hpc="sofja",
):
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

    client, path_code, batch_params = set_hpc_params(hpc)

    if suffix:
        suffix = f"_{suffix}" if not suffix.startswith("_") else suffix

    if path_target is None:
        path_target = path_source

    sftp_client = client.open_sftp()

    neuron_detection_script = f"{path_code}/run_neuron_detection.py"
    ## first, write the neuron detection script to the server

    # params['dxy']: 512./520.
    # params['fr']: 15.26
    # params['decay_time']: 1.75
    _, stdout, stderr = client.exec_command(
        f"""cat > {neuron_detection_script} <<- EOF
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
"""
    )

    ## first, search all session folders in the mouse folder
    _, stdout, stderr = client.exec_command(
        f"ls -d {os.path.join(path_source,'Session*/ | xargs -n 1 basename')}"
    )
    sessions = str(stdout.read(), encoding="utf-8").splitlines()
    print(sessions)

    for s, session in enumerate(sessions, start=1):
        sessionName = session
        pathSession_source = os.path.join(path_source, sessionName)
        pathSession_target = os.path.join(path_target, sessionName)
        # continue

        if len(specific_sessions):
            if not (s in specific_sessions):
                continue

        print(f"{s}: {pathSession_source=} ,{pathSession_target=}")

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
        _, stdout, stderr = client.exec_command(
            f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s}_detect
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 04:00:00
#SBATCH -o {pathSession_target}/log_neuron_detection.log
#SBATCH -e {pathSession_target}/log_neuron_detection.log
#SBATCH --mem=64000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 {neuron_detection_script} {pathSession_source} {pathSession_target} {cpus}
EOF
"""
        )
        ## requires absolute path to command, as $PATH is not set for client login
        _, stdout, stderr = client.exec_command(
            f"/usr/local/slurm/current/install/bin/sbatch {batch_params['submit_file']}"
        )
        client.exec_command(f"sleep 1 && rm {batch_params['submit_file']}")
        time.sleep(1)
        # print(stdout.read(),stderr.read())
        # break

    # break


# -p medium
# -A scc_user
# (oder scc_users ...)


def run_neuron_matching(
    pathMouse,
    fileName="results_CaImAn*",
    suffix="",
    exclude="xyzabc",
    fileType=".hdf5",
    hpc="sofja",
    subhodeep=False,
):
    """
    Run neuron matching on the GWDG cluster

    Both, path_source and path_target should be paths to mouse folders on the
    GWDG cluster, containing SessionXY folders with neuron detection results.
    """

    cpus = 8
    mouse = os.path.basename(pathMouse)

    if not suffix.startswith("_") and not (suffix == ""):
        suffix = "_" + suffix

    client, path_code, batch_params = set_hpc_params(hpc)

    neuron_matching_script = f"{path_code}/run_neuron_matching.py"
    ## first, write the neuron detection script to the server

    if subhodeep:
        construct_paths = f"""
# pathsResults = sorted([os.path.join(pathMouse,path,file) for path in os.listdir(pathMouse) if os.path.isdir(os.path.join(pathMouse,path)) for file in os.listdir(os.path.join(pathMouse,path)) if (file.startswith('{fileName[:-1]}') and file.endswith('{fileType}') and not ('{exclude}' in file))])

pathsResults = [path if os.path.exists(path) else False for path in sorted([os.path.join(pathMouse,path,'{fileName[:-1]}_' + os.path.splitext(file)[0] + '{suffix}.mat') for path in os.listdir(pathMouse) if os.path.isdir(os.path.join(pathMouse,path)) for file in os.listdir(os.path.join(pathMouse,path)) if (file.endswith('.tif') and not ('combined' in file))])]
"""
    else:
        construct_paths = f"""
_, pathsResults = set_paths_default(pathMouse=pathMouse,fileName_in=fileName,fileType='{fileType}',suffix='{suffix}',exclude='{exclude}')
"""

    _, stdout, stderr = client.exec_command(
        f"""cat > {neuron_matching_script} <<- EOF
import os, sys
from placefield_dynamics.neuron_matching import *

assert len(sys.argv) == 3, "Need to provide two arguments, pathMouse and fileName as arguments! Currently given: %s"%str(sys.argv)
_, pathMouse, fileName = sys.argv

print(pathMouse,fileName,'{suffix}')
{construct_paths}
print(pathsResults)
                                            
match = matching(pathMouse,paths=pathsResults,suffix='{suffix}',matlab={fileType.endswith('mat')})
match.run_matching(p_thr=[0.3,0.05])
"""
    )

    ## write bash script to run neuron detection
    _, stdout, stderr = client.exec_command(
        f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}_match
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 02:00:00
#SBATCH -o {pathMouse}/log_neuron_matching.log
#SBATCH -e {pathMouse}/log_neuron_matching.log
#SBATCH --mem=32000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9

python3 {neuron_matching_script} {pathMouse} {fileName}
EOF
"""
    )

    _, stdout, stderr = client.exec_command(
        f"/usr/local/slurm/current/install/bin/sbatch {batch_params['submit_file']}"
    )
    print(stdout.read(), stderr.read())


def run_neuron_redetection(
    pathMouse,
    pathMouse_ref=None,
    resultFiles="results_CaImAn*",
    specific_sessions=[],
    hpc="sofja",
    subhodeep=False,
):

    mouse = os.path.basename(pathMouse)
    cpus = 8

    client, path_code, batch_params = set_hpc_params(hpc)

    if pathMouse_ref is None:
        pathMouse_ref = pathMouse

    neuron_redetection_script = f"{path_code}/neuron_redetection.py"

    #     if subhodeep:
    #         construct_paths = f"""
    # pathsResults = sorted([os.path.join(pathMouse,path,file) for path in os.listdir(pathMouse) if os.path.isdir(os.path.join(pathMouse,path)) for file in os.listdir(os.path.join(pathMouse,path)) if (file.startswith('{resultFiles[:-1]}') and file.endswith('.mat') and not ('redetected' in file))])
    # pathsSessions = pathsResults ## actually not really needed

    # session, file = os.path.split(pathsResults[s])
    # basename = os.path.splitext(file.split('{resultFiles[:-1]+'_'}')[-1])[0]
    # pathImage = os.path.join(session,(basename+'.tif'))
    # """
    #     else:
    construct_paths = """
pathsSessions, pathsResults = set_paths_default(pathMouse,fileName_in=resultFiles,exclude='redetected')

thisSession = os.path.split(pathsSessions[s])[-1]
try:
    pathImage = [file for file in os.listdir(pathsSessions[s]) if file.endswith('.tif')][0]
    pathImage = os.path.join(pathMouse,thisSession,pathImage)
except:
    thisSession = os.path.split(pathsSessions[s])[-1]
    pathImage = [file for file in os.listdir(os.path.join(pathMouse_ref,thisSession)) if (file=='images' or file.endswith('.tif'))][0]
    pathImage = os.path.join(pathMouse_ref,thisSession,pathImage)
"""

    _, stdout, stderr = client.exec_command(
        f"""cat > {neuron_redetection_script} <<- EOF
import os, sys
from placefield_dynamics.silence_redetection import *

_, pathMouse, pathMouse_ref, resultFiles, s = sys.argv

s = int(s)
{construct_paths}

print(pathsSessions)
print(pathsResults)
print(pathImage)

sr = silence_redetection(pathsSessions,pathsResults)
sr.process_session(s,path_tmp=os.environ['TMP_LOCAL'],pathImage=pathImage,ssh_alias=None)
"""
    )

    _, stdout, stderr = client.exec_command(
        f"ls -d {os.path.join(pathMouse,'Session*/')}"
    )
    directories = str(stdout.read(), encoding="utf-8").splitlines()

    for s, dir in enumerate(directories):

        if len(specific_sessions):
            if not ((s + 1) in specific_sessions):
                s += 1
                continue

        print(f"{s}: {dir=}")
        # os.path.splitext(os.path.basename(resultFiles[s]))[0].removeprefix(resultFiles[:-1])

        ## write bash script to run neuron detection
        _, stdout, stderr = client.exec_command(
            f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s+1}_redetection
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 2:00:00
#SBATCH -o {dir}/log_neuron_redetection.log
#SBATCH -e {dir}/log_neuron_redetection.log
#SBATCH --mem=32000

python3 {neuron_redetection_script} {pathMouse} {pathMouse_ref} {resultFiles} {s}
EOF
"""
        )

        _, stdout, stderr = client.exec_command(
            f"/usr/local/slurm/current/install/bin/sbatch {batch_params['submit_file']}"
        )
        print(stdout.read(), stderr.read())
        client.exec_command(f"sleep 1 && rm {batch_params['submit_file']}")
        time.sleep(1)

        # break


def run_neuron_redetection_Subhodeep(
    pathMouse, resultFiles="results_CaImAn*", specific_sessions=[], hpc="sofja"
):

    mouse = os.path.basename(pathMouse)
    cpus = 8

    client, path_code, batch_params = set_hpc_params(hpc)

    neuron_redetection_script = f"{path_code}/neuron_redetection_subhodeep.py"

    construct_paths = f"""
pathsResults = sorted([os.path.join(pathMouse,path,file) for path in os.listdir(pathMouse) if os.path.isdir(os.path.join(pathMouse,path)) for file in os.listdir(os.path.join(pathMouse,path)) if (file.startswith('{resultFiles[:-1]}') and file.endswith('.mat') and not ('redetected' in file))])
pathsSessions = [os.path.split(path)[0] for path in pathsResults] ## actually not really needed

session, file = os.path.split(pathsResults[s])
basename = os.path.splitext(file.split('{resultFiles[:-1]+'_'}')[-1])[0]
pathImage = os.path.join(session,(basename+'.tif'))
"""

    _, stdout, stderr = client.exec_command(
        f"""cat > {neuron_redetection_script} <<- EOF
import os, sys
from placefield_dynamics.silence_redetection import *

_, pathMouse, s = sys.argv

s = int(s)
{construct_paths}

print(pathsSessions)
print(pathsResults)
print(pathImage)

sr = silence_redetection(pathsSessions,pathsResults,matlab=True,params_in=dict(fr=30.05))
sr.process_session(s,path_tmp=os.environ['TMP_LOCAL'],pathImage=pathImage,ssh_alias=None)
"""
    )
    # sr = silence_redetection(pathsSessions,pathsResults,matlab=True,params_in=dict(fr=15.))

    ## first, search all 'day folders' in the mouse source path
    _, stdout, stderr = client.exec_command(f"ls -d {os.path.join(pathMouse,'*/')}")
    directories = str(stdout.read(), encoding="utf-8").splitlines()
    s = 1
    for dir in directories:
        foldername = os.path.split(dir[:-1])[-1]

        if not foldername.startswith("202"):
            continue

        _, stdout, stderr = client.exec_command(f"ls {os.path.join(dir,'*.tif')}")
        recordings = str(stdout.read(), encoding="utf-8").splitlines()

        # ss = 0
        for path_recording in recordings:

            basename = os.path.splitext(os.path.split(path_recording)[-1])[0]
            if "combined" in basename:
                continue

            if len(specific_sessions):
                if not (s in specific_sessions):
                    s += 1
                    continue

            resultsName = resultFiles[:-1] + "_" + basename + ".mat"
            resultsPath = os.path.join(dir, resultsName)

            _, stdout, stderr = client.exec_command(
                f'test -e {resultsPath} && echo "File exists" || echo "File does not exist"'
            )
            output = stdout.read().decode().strip()

            if not ("File exists" in output):
                print(
                    f"original CNMF {resultsName} not yet present - process this first!"
                )
                s += 1
                break
            resultsName = resultFiles[:-1] + "_" + basename + "_redetected.mat"
            resultsPath = os.path.join(dir, resultsName)

            _, stdout, stderr = client.exec_command(
                f'test -e {resultsPath} && echo "File exists" || echo "File does not exist"'
            )
            output = stdout.read().decode().strip()

            # Check the result
            if "File exists" in output:
                print(f"file {resultsName} already exists!")
                s += 1
                continue

            # print(f'{s}: {dir=}, {resultsPath=}')
            # resultsPath = os.path.join(dir,resultsName)
            # print(s,path_recording,resultsName)

            print(f"processing session {s} in dir {dir} with basename {basename}")

            ## write bash script to run neuron detection
            # print(f'writing log to: {dir}log_neuron_redetection_{basename}.log')
            _, stdout, stderr = client.exec_command(
                f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s}_redetect
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 04:00:00
#SBATCH -o {dir}log_neuron_redetection_{basename}.log
#SBATCH -e {dir}log_neuron_redetection_{basename}.log
#SBATCH --mem=64000

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 {neuron_redetection_script} {pathMouse} {s-1}
EOF
"""
            )

            _, stdout, stderr = client.exec_command(
                f"/usr/local/slurm/current/install/bin/sbatch {batch_params['submit_file']}"
            )
            print(stdout.read(), stderr.read())
            client.exec_command(f"sleep 2 && rm {batch_params['submit_file']}")
            s += 1
            time.sleep(1)


def run_placecell_detection(
    path_mouse,
    result_files="results_CaImAn*",
    suffix="",
    exclude="xyzabc",
    file_type=".hdf5",
    specific_sessions=[],
    hpc="sofja",
):

    if not suffix.startswith("_") and not (suffix == ""):
        suffix = "_" + suffix

    mouse = os.path.basename(path_mouse)
    cpus = 64

    client, path_code, batch_params = set_hpc_params(hpc)

    placecell_detection_script = f"{path_code}/placecell_detection.py"

    construct_paths = f"""
_, path_results = set_paths_default(pathMouse=path_mouse,fileName_in='{result_files}',fileType='{file_type}',suffix='{suffix}',exclude='{exclude}')
"""

    _, stdout, stderr = client.exec_command(
        f"""cat > {placecell_detection_script} <<- EOF
import os, sys
from pathlib import Path
from placefield_dynamics.neuron_matching.utils import set_paths_default
from placefield_dynamics import placefield_detection

assert len(sys.argv) == 3, "Need to provide two arguments, pathMouse and fileName as arguments! Currently given: %s"%str(sys.argv)
_, path_mouse, s = sys.argv

print(path_mouse,'{suffix}')
{construct_paths}

path_results = Path(path_results[int(s)])
path_session = path_results.parent
print(path_results)

ps = placefield_detection.process_session(plot_it=False)
results = ps.process_input_from_file(
    path_data = path_results,
    path_behavior = path_session / 'aligned_behavior.pkl',
    path_results = path_session / 'placefield_detection{suffix}.hdf5',
    mode_place_cell_detection=['peak','information'],
    mode_place_field_detection=['bayesian'],
    nP={cpus},
)
"""
    )

    _, stdout, stderr = client.exec_command(
        f"ls -d {os.path.join(path_mouse,'Session*/')}"
    )
    directories = str(stdout.read(), encoding="utf-8").splitlines()

    for s, dir in enumerate(directories):

        if len(specific_sessions):
            if not ((s + 1) in specific_sessions):
                s += 1
                continue

        # results_path = Path(dir) / f"placefield_detection{suffix}.hdf5"
        # # print(results_path)
        # _, stdout, stderr = client.exec_command(
        #     f'test -e {results_path} && echo "File exists" || echo "File does not exist"'
        # )
        # output = stdout.read().decode().strip()

        # # Check the result
        # if "File exists" in output:
        #     print(f"Results file already exists!")
        #     s += 1
        #     continue

        print(f"{s}: {dir=}")

        ## write bash script to run neuron detection
        _, stdout, stderr = client.exec_command(
            f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J m{mouse}s{s+1}_PC_detection
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 2:00:00
#SBATCH -o {dir}/log_PC_detection.log
#SBATCH -e {dir}/log_PC_detection.log
#SBATCH --mem=64000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9

python3 {placecell_detection_script} {path_mouse} {s}
EOF
"""
        )
        # source activate caiman-1.9.10_py-3.9

        _, stdout, stderr = client.exec_command(
            f"/usr/local/slurm/current/install/bin/sbatch {batch_params['submit_file']}"
        )
        print(stdout.read(), stderr.read())
        client.exec_command(f"sleep 0.5 && rm {batch_params['submit_file']}")
        time.sleep(0.5)
