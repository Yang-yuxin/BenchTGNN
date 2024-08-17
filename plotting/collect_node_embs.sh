datasets=("WIKI" "REDDIT" "uci" "CollegeMsg" "LASTFM" "mooc" "Flights")
for dataset in "${datasets[@]}"
do
    python plotting/run_all_node.py --config_dir config/node --data ${dataset} --excel_file 'model_configs_detailed.xlsx'
done