#!/bin/bash
# make sure all datasets are downloaded and placed at DATA\${dataset}\edges.csv
datasets=("WIKI" "REDDIT" "LASTFM" "MOOC" "UCI" "CollegeMsg" "Flights")
for dataset in "${datasets[@]}"
do
    python src/gen_graph.py --data ${dataset} && \
    python src/preprocess.py --data ${dataset}
done