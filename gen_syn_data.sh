# python src/gen_syn_dataset.py --data syn1
# python src/preprocess.py --data syn1
# python src/analyze_graph_recurrence_analysis.py --data syn1  --file_path figures/data/syn1_recur_time.pkl --bins 1000 --replace

# alphas=(0.9 0.99 0.999 1)
alphas=(0.6 0.7 0.8 0.9 1)
betas=(0.4 0.5 0.6 0.7 0.8 0.9)
for alpha in "${alphas[@]}"
do
    for beta in "${betas[@]}"
    do
        python src/gen_syn_dataset.py --data syn_"${alpha}"_"${beta}" --alpha ${alpha} --beta ${beta}
        python src/preprocess.py --data syn_"${alpha}"_"${beta}"
        python src/analyze_graph_recurrence_analysis.py --data syn_"${alpha}"_"${beta}"  --file_path figures/data/syn_"${alpha}"_"${beta}"_recur_time.pkl --bins 1000 --replace
    done
done
