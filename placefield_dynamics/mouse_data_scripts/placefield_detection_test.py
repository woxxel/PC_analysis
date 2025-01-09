import os, time
import numpy as np
from pathlib import Path
from .utils.connections import *


def run_placefield_detection_test(
    path_session,
    path_target,
    n_neurons=100,
    n_per_batch=10,
    nbin=40,
    hpc="sofja",
):

    cpus = 64

    client, path_code, batch_params = set_hpc_params(hpc)

    placecell_detection_script = f"{path_code}/placecell_detection.py"

    _, stdout, stderr = client.exec_command(
        f"""cat > {placecell_detection_script} <<- EOF
import os, sys, pickle
from pathlib import Path
from placefield_dynamics.placefield_detection.surrogate_data import SurrogateData
from placefield_dynamics.placefield_detection.process_session import process_session

from placefield_dynamics.placefield_detection.utils import prepare_behavior_from_file

assert len(sys.argv) == 3, "Need to provide two arguments, path_session and n_neurons as arguments! Currently given: %s"%str(sys.argv)
_, path_session, n_neurons = sys.argv

# suffix = os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
suffix = 'n=' + n_neurons + '_' + os.environ['SLURM_ARRAY_TASK_ID']
print('suffix:',suffix)

n_neurons = int(n_neurons)
track = dict(
    nbin = {nbin},
    length = 100,
)
place_field_parameter = dict(
    field_probabilities = [0.3,0.8],

    A0 = [0,2],
    A = [0.5,10],
    sigma = [1,5],
    theta = [0,track['nbin']],

    reliability = [0.3,1], 
)

path_behavior = Path(path_session) / 'aligned_behavior.pkl'
behavior = prepare_behavior_from_file(path_behavior,nbin=track['nbin'],f=15.)

surrogate_data = SurrogateData(
    n_neurons,track,
    place_field_parameter,behavior
)

surrogate_data.generate_activity_all(nP={cpus})

data = dict(
    tuning_curve_parameter=surrogate_data.tuning_curve_parameter,
    behavior=behavior,
    activity=surrogate_data.activity,
    field_activation=surrogate_data.field_activation,
)

with open(Path('{path_target}') / ('surrogate_data_'+suffix+'.pkl'),'wb') as f:
    pickle.dump(data,f)

ps = process_session(plot_it=False)
results = ps.process_input(
    behavior,
    surrogate_data.activity,
    path_results = Path('{path_target}') / ('surrogate_placefield_detection_' + suffix + '.hdf5'),
    mode_place_cell_detection=['peak','information'],
    mode_place_field_detection=['bayesian'],
    nP={cpus},
)
"""
    )

    n_repeat = n_neurons // n_per_batch
    print(f"Running {n_repeat} batches of {n_per_batch} neurons each...")
    ## write bash script to run neuron detection
    _, stdout, stderr = client.exec_command(
        f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J test_PC_detection
#SBATCH -a 1-{n_repeat}%20
#SBATCH -A {batch_params['A']}
#SBATCH -p {batch_params['p']}
#SBATCH -c {cpus}
#SBATCH -t 6:00:00
#SBATCH -o log_test_PC_detection_%A_%a.log
#SBATCH -e log_test_PC_detection_%A_%a.log
#SBATCH --mem=64000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9

python3 {placecell_detection_script} {path_session} {n_per_batch}
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
