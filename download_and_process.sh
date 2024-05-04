#!/bin/bash
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv  
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/ext_full.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_full.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_train.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/ext_full.npz
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_full.npz
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_train.npz
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
wget -P ./DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/edges.csv                 
wget -P ./DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/ext_full.npz           
wget -P ./DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/int_full.npz          
wget -P ./DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/int_train.npz        
wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/edges.csv   
wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/ext_full.npz
wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/int_full.npz
wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/int_train.npz
# make sure UCI, CollegeMsg and Flights dataset are downloaded and placed at DATA\${dataset}\edges.csv
datasets=("WIKI" "REDDIT" "LASTFM" "MOOC" "UCI" "CollegeMsg" "Flights")
for dataset in "${datasets[@]}"
do
    python src/gen_graph.py --data ${dataset} && \
    python src/preprocess.py --data ${dataset}
done