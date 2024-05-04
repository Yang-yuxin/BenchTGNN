alpha=${1}
beta=${2}
python src/gen_syn_dataset.py --data syn_"${alpha}"_"${beta}" --alpha ${alpha} --beta ${beta}
python src/preprocess.py --data syn_"${alpha}"_"${beta}"
python src/analyze_graph_recurrence_analysis.py --data syn_"${alpha}"_"${beta}"  --file_path figures/data/syn_"${alpha}"_"${beta}"_recur_time.pkl --bins 1000 --replace
