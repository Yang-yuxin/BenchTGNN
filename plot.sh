# python src/collect_logs_all.py --export all_mrrs_0313.pkl
python src/plot_spl_strategy.py --layer 1 --pkl_path all_mrrs_0313.pkl --fontsize 32 --save_legends
python src/plot_spl_strategy.py --layer 2 --no_title --pkl_path all_mrrs_0313.pkl --fontsize 32