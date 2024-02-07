dataset="${1}"
# cp DATA_BenchTemp/benchtemp_datasets/${dataset}/ml_${dataset}.csv DATA/${dataset}/edges.csv
# python src/npy2pt.py --npy DATA_BenchTemp/benchtemp_datasets/${dataset}/ml_${dataset}_node.npy --pt DATA/${dataset}/node_features_pad.pt
if [ ! -d "DATA/${dataset}" ]; then
  mkdir DATA/${dataset}
fi
cp DATA_BenchTemp/benchtemp_datasets/${dataset}/ml_${dataset}.csv DATA/${dataset}/edges.csv &&  \
python src/gen_graph.py --data ${dataset} && \
python src/preprocess_benchtemp.py --data ${dataset}
