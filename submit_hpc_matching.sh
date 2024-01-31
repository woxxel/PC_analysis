#!/bin/bash

cpus=4
datapath='/usr/users/cidbn1/placefields'
dataset="AlzheimerMice_Hayashi"
# dataset="Shank2Mice_Hayashi"

SUBMIT_FILE="./sbatch_submit.sh"

read -p "Which CaImAn result files should be processed? " result_files
read -p "Which suffix should the matched files contain? " suffix

mice=$(find $datapath/$dataset/* -maxdepth 0 -type d -exec basename {} \;)
# echo "Found mice in dataset $dataset: $mice"
# read -p 'Which mouse should be processed? ' mouse

for mouse in $mice
do

  # if test -f $datapath/$dataset/$mouse/matching/neuron_registration_.pkl; then
  #   echo "$session_name already processed - skipping"
  #   continue
  # fi

  ## writing sbatch submission commands to bash-file
  cat > $SUBMIT_FILE <<- EOF
#!/bin/bash
#SBATCH -A all
#SBATCH -p medium
#SBATCH -c $cpus
#SBATCH -t 02:00:00
#SBATCH -o $datapath/$dataset/$mouse/log_matching.out
#SBATCH --mem=20000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9
source activate caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 ./neuron_matching_hpc_wrapper.py $datapath $result_files $suffix $dataset $mouse $cpus
EOF

  sbatch $SUBMIT_FILE
  rm $SUBMIT_FILE

done
