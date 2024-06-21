#!/bin/bash

cpus=4
datapath='/usr/users/cidbn1/placefields'
#dataset="AlzheimerMice_Hayashi"
# dataset="Shank2Mice_Hayashi"

SUBMIT_FILE="./sbatch_submit.sh"

datasets=$(find $datapath/* -maxdepth 0 -type d -exec basename {} \;)
echo "Found datasets: $datasets"
read -p "Which dataset should be processed? (hit enter to use default: AlzheimerMice_Hayashi) " dataset
if [[ -z $dataset ]]; then
  dataset="AlzheimerMice_Hayashi"
  echo "Using default dataset '$dataset'"
fi

read -p "Which CaImAn result files should be processed? (hit enter to use default: OnACID_results) " result_files
if [[ -z $result_files ]]; then
  result_files="OnACID_results"
  echo "Using default result files: '$result_files'"
fi

read -p "Which suffix should the matched files contain? (hit enter to use None) " suffix

mice=$(find $datapath/$dataset/* -maxdepth 0 -type d -exec basename {} \;)
echo "Found mice in dataset $dataset: $mice"
read -p 'Which mouse should be processed? (hit enter to process all): ' mouse

if [[ -n $mouse ]]; then
  mice=($mouse)
fi

for mouse in $mice
do
  echo "Processing mouse $mouse"

  # if test -f $datapath/$dataset/$mouse/matching/neuron_registration_.pkl; then
  #   echo "$session_name already processed - skipping"
  #   continue
  # fi

  ## writing sbatch submission commands to bash-file
  cat > $SUBMIT_FILE <<- EOF
#!/bin/bash
#SBATCH -J m${mouse}_match
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c $cpus
#SBATCH -t 02:00:00
#SBATCH -o $datapath/$dataset/$mouse/log_matching.log
#SBATCH -e $datapath/$dataset/$mouse/log_matching_error.log
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
