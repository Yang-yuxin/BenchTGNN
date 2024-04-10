python src/collect_logs_all.py --export all_mrrs_0318.pkl
python src/collect_logs_all.py --export all_time_0318.pkl --target time
python src/plot_spl_strategy.py --layer 1 --pkl_path all_mrrs_0314.pkl --fontsize 32 --save_legends
python src/plot_spl_strategy.py --layer 2 --no_title --pkl_path all_mrrs_0314.pkl --fontsize 32