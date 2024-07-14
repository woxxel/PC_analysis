#!/bin/bash
#SBATCH -J m$ID58s$1_detect
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /scratch/users/caiman_share/ID58/Session01/log_neuron_detection.log
#SBATCH -e /scratch/users/caiman_share/ID58/Session01/log_neuron_detection_error.log
#SBATCH --mem=64000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9
source activate caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 ./neuron_detection_hpc_wrapper.py $datapath_in $datapath_out $dataset $mouse $session_name $cpus
            