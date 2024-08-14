#!/bin/bash

# usage: bash run_adapt.sh <DATASET> <TRIAL> <CONFIG_DIR>
# Run small datasets WIKI / REDDIT on ganges

export CUDA_DEVICE_ORDER=PCI_BUS_ID;
gpus=(1 3)
#echo "Enter a list of device numbers separated by commas:"
#IFS=',' read -ra gpus
#echo -n 'Schedule tasks on GPUs: '
#for num in "${gpus[@]}";
#do
#    echo -n "$num "
#done
#printf "\n\n"

datasets=("${1}")
# neighbors=(2 3 4)
# neighbors=(1 2 5 10 20 50 100)
runs=5

trial="${2}"
trial_dir="$(date +%Y-%m-%d)"_"${trial}"

log_dir=log/"${trial_dir}"
mkdir -p "${log_dir}"

mkdir -p trialspace
src_dir=trialspace/"${trial_dir}"
rm -rf "${src_dir}" || true
cp -r src "${src_dir}"

#config_dir=trial/"${trial}"
config_dir="${3}"/"${trial}"
mapfile -t configs < <(ls "${config_dir}")

# Loop through each dataset and neighbor
for config in "${configs[@]}"
do
  for dataset in "${datasets[@]}"
  do
    # for neighbor in "${neighbors[@]}"
    # do
        for i in $(seq 1 $runs)
        do
            # schedule GPU
            scheduled=false
            while [ "$scheduled" = false ];
            do
                for gpu in "${gpus[@]}";
                do
                    if [ "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; then
                    # if [ "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader | wc -l)" -le 2 ]; then
                        scheduled=true
                        break
                    fi
                done
                if [ "$scheduled" = false ]; then
                    echo No available GPU right now, sleep for 120 sec
                    sleep 120
                fi
            done

            python -u "${src_dir}"/train.py --config "${config_dir}"/"${config}" \
                                              --data "${dataset}" \
                                              --gpu "${gpu}" \
            | tee "${log_dir}"/"${dataset}"_"${config%.*}"_"${i}".out &
            sleep 10
      done
      # collect output
    # done
  done
done
