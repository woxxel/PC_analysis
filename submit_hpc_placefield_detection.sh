#!/bin/bash

cpus=12
datapath='/usr/users/cidbn1/placefields'  # same in/out datapath, as it uses processed file (CNMF-result), already
dataset="AlzheimerMice_Hayashi"
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



read -p "Which CaImAn result files should be processed? " result_files
read -p "Which suffix should the result files contain? " suffix

mice=$(find $datapath/$dataset/* -maxdepth 0 -type d -exec basename {} \;)
echo "Found mice in dataset $dataset: $mice"
read -p 'Which mouse should be processed? ' mouse

# for mouse in $mice
# do

  ## getting all sessions of $mouse to loop through
  session_names=$(find $datapath/$dataset/$mouse/Session* -maxdepth 0 -type d -exec basename {} \;)

  # s=1
  for session_name in $session_names
  do

    # if test -f $datapath/$dataset/$mouse/$session_name/placefields.pkl; then
    #   # echo "$session_name already processed - skipping"
    #   continue
    # fi

    echo "Processing mouse $mouse, $session_name"

    ## writing sbatch submission commands to bash-file
    cat > $SUBMIT_FILE <<- EOF
#!/bin/bash
#SBATCH -A cidbn_legacy
#SBATCH -p cidbn
#SBATCH -c $cpus
#SBATCH -t 01:00:00
#SBATCH -o $datapath/$dataset/$mouse/$session_name/log_placefield_detection.out
#SBATCH --mem=8000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9
source activate caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 ./placefield_detection_hpc_wrapper.py $datapath $result_files $suffix $dataset $mouse $session_name $cpus
EOF

    sbatch $SUBMIT_FILE
    rm $SUBMIT_FILE

  done
# done