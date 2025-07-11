
from pathlib import Path
from .utils.connections import *

def run_stacking_on_hpc(basePath = '/usr/users/cidbn1/neurodyn/AlzheimerMice_Hayashi/',logPath=None,specific_mouse=None,max_parallel_jobs=2,hpc='sofja'):

    ## this won't work, once write permissions are removed from the server
    
    cpus = 4

    client, path_code, batch_params = set_hpc_params(hpc)
    
    # copy the stacking script to the server
    stacking_script = f'{path_code}/run_stacking.py'
    with open(Path(__file__).parent / 'stack_tifs_script.py','r') as f_open:
        lines = f_open.readlines()

    _, stdout, stderr = client.exec_command(f"""cat > {stacking_script} <<- EOF
{("").join(lines)}
EOF
""")

    ## now, loop over all mice to process the stacking
    if specific_mouse is None:
        _, stdout, stderr = client.exec_command(f"ls {basePath}")
        mice = str(stdout.read(), encoding='utf-8').splitlines()
        pathsMouse = sorted([Path(basePath, mouse) for mouse in mice if mouse[:2].isdigit()])
    else:
        pathsMouse = [Path(basePath) / specific_mouse]
    
    for pathMouse in pathsMouse:

        if logPath is None:
            logPathMouse = Path(pathMouse) / 'logs'
        else:
            logPathMouse = Path(logPath) / 'logs'
        ## for each mouse, run sessions as slurm_array
        _, stdout, stderr = client.exec_command(f"ls -d {pathMouse / 'Session*/'} | xargs -n 1 basename")
        pathsSession = str(stdout.read(), encoding='utf-8').splitlines()
        # pathsSession = sorted([path for path in pathMouse.iterdir() if path.stem.startswith('Session')])pathMouse / 'logs'
        nSession = len(pathsSession)
        if nSession==0:
            continue
        # print(pathsSession,nSession)

        print(f"Processing {pathMouse.stem} with {nSession} sessions...")
        # return

        _, stdout, stderr = client.exec_command(f"mkdir -p {logPathMouse}")

        _, stdout, stderr = client.exec_command(f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J stacking_tiffs_m{pathMouse.stem}
#SBATCH -a 1-{nSession}%{max_parallel_jobs}
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 00:30:00
#SBATCH -o {logPathMouse}/stacking_m{pathMouse.stem}_%a.log
#SBATCH -e {logPathMouse}/stacking_m{pathMouse.stem}_%a.log
#SBATCH --mem=16000

python3 {stacking_script} {pathMouse}
EOF
""")
        _, stdout, stderr = client.exec_command(f"/usr/local/slurm/current/install/bin/sbatch {batch_params['submit_file']}")
        print(stdout.read(),stderr.read())





    