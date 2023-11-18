#!/bin/bash

# usage: bash run_adapt.sh <DATASET> <TRIAL> <CONFIG_DIR>
# Run small datasets WIKI / REDDIT on ganges

export CUDA_DEVICE_ORDER=PCI_BUS_ID;
gpus=(0)
#echo "Enter a list of device numbers separated by commas:"
#IFS=',' read -ra gpus
#echo -n 'Schedule tasks on GPUs: '
#for num in "${gpus[@]}";
#do
#    echo -n "$num "
#done
#printf "\n\n"

datasets=("${1}")
orders=("chorno")
runs=2

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

# Loop through each dataset and order
for config in "${configs[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for order in "${orders[@]}"
    do
        for i in $(seq 1 $runs)
        do
            # schedule GPU
            scheduled=false
            while [ "$scheduled" = false ];
            do
                for gpu in "${gpus[@]}";
                do
                    if [ "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; then
                        scheduled=true
                        break
                    fi
                done
                if [ "$scheduled" = false ]; then
                    echo No available GPU right now, sleep for 10 sec
                    sleep 10
                fi
            done

            python -u "${src_dir}"/train.py --config "${config_dir}"/"${config}" \
                                              --data "${dataset}" \
                                              --override_order "${order}" \
                                              --gpu "${gpu}" \
                                              --pure_gpu \
                                              --profile \
            | tee "${log_dir}"/"${order}"_"${dataset}"_"${config%.*}"_"${i}".out &
            sleep 10
      done
      # collect output
    done
  done
done
wait
python -u src/collect_logs.py --trial "${trial}" \
                              --log_dir "${log_dir}" \
                              --config_dir "${3}"\
      | tee -a "${log_dir}"/result