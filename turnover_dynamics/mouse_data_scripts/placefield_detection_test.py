import os, time
import numpy as np
from pathlib import Path
from .utils.connections import *


def run_place_selectivity_inference_test(
    path_session,
    path_target,
    n_neurons=100,
    n_per_batch=10,
    nbin=40,
    hpc="sofja",
    cpus=64,
):

    client, path_code, batch_params = set_hpc_params(
        home_dir=Path("/mnt/vast-standard/home/schmidt124/u23010"),
        hpc=hpc
    )
    print(client)
    print(path_code)
    print(batch_params)

    placecell_detection_script = path_code / "placecell_detection.py"
    print(placecell_detection_script)

    _, stdout, stderr = client.exec_command(
        f"""cat > {placecell_detection_script} <<- EOF
import os, sys, pickle
from pathlib import Path
from turnover_dynamics.place_selectivity_inference.surrogate_data import SurrogateData
from turnover_dynamics.place_selectivity_inference.process_session import process_session

from turnover_dynamics.place_selectivity_inference.utils import prepare_behavior_from_file, save_data

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
    field_probabilities = [0.2,0.7],

    A0 = [0,2],
    A = [0.5,10],
    sigma = [1,5],
    theta = [0,track['nbin']],

    reliability = [0.2,1], 
)

path_behavior = Path(path_session) / 'aligned_behavior.pkl'
behavior = prepare_behavior_from_file(path_behavior,nbin=track['nbin'],f=15.)

surrogate_data = SurrogateData(
    None,
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

save_data(data, Path('{path_target}') / ('surrogate_data_'+suffix+'.hdf5'))

#with open(Path('{path_target}') / ('surrogate_data_'+suffix+'.hdf5'),'wb') as f:
#    pickle.dump(data,f)

ps = process_session(plot_it=False)
results = ps.from_input(
    behavior,
    surrogate_data.activity,
    path_results = Path('{path_target}') / ('surrogate_place_selectivity_inference_' + suffix + '.hdf5'),
    mode_place_cell_detection=['peak','information','stability'],
    mode_place_field_detection=['bayesian','threshold'],
    nP={cpus},
)
"""
    )

    print(stdout.read(), stderr.read())

    n_repeat = n_neurons // n_per_batch
    print(f"Running {n_repeat} batches of {n_per_batch} neurons each...")
    # SBATCH -A {batch_params['A']}
    ## write bash script to run neuron detection
    _, stdout, stderr = client.exec_command(
        f"""cat > {batch_params['submit_file']} <<- EOF
#!/bin/bash -l
#SBATCH -J test_PC_detection
#SBATCH -p {batch_params['p']}
#SBATCH -a 1-{n_repeat}%20
#SBATCH -c {cpus}
#SBATCH -t 6:00:00
#SBATCH -o log_test_PC_detection_%A_%a.log
#SBATCH -e log_test_PC_detection_%A_%a.log
#SBATCH --mem=64000

conda activate inference

python {placecell_detection_script} {path_session} {n_per_batch}
EOF
"""
    )
    # source activate caiman-1.9.10_py-3.9

    _, stdout, stderr = client.exec_command(
        f"/opt/slurm/el8/25.05.3/install/bin/sbatch {batch_params['submit_file']}"
    )
    print(stdout.read(), stderr.read())
    client.exec_command(f"sleep 0.5 && rm {batch_params['submit_file']}")
    time.sleep(0.5)
