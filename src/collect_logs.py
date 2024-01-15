import os

import numpy as np
import argparse

# PATH = 'log/danny20230811_adapt_sampler'
# PATH = 'log/danny20230814'
# PATH = 'log/danny20230817_mixer_sampler'
# PATH = 'log/danny20230817_mixer_frequency'
# PATH = 'log/danny20230817_mixer_fourier'
# PATH = 'log/danny20230818_align_dimension_64'
# PATH = 'log/danny20230818_align_dimension_100'
# PATH = 'log/danny20230818_align_dimension_100'
# PATH = 'log/danny20230822_selfnorm_0'

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='trial name')
parser.add_argument('--log_dir', type=str, help='log file directory')
parser.add_argument('--config_dir', type=str, default='trial', help='config file directory')
parser.add_argument('--runs', type=int, default=5, help='trial name')
args = parser.parse_args()

log_dir = args.log_dir
config_dir = args.config_dir + '/{}'.format(args.trial)

orders = ['chorno', 'gradient']
datasets = ['WIKI', 'REDDIT', 'Flight', 'MovieLens', 'sGDELT', 'GDELT', 'MOOC', 'LASTFM']
configs = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
configs.sort()

def get_test_mrr(file_path, results):
    try:
        with open(file_path, 'r') as f:
            mrr = None
            for lines in f:
                if lines.startswith('\ttest AP'):
                    mrr = float(lines.strip('\n').split(':')[-1])
            if mrr is not None:
                results.append(mrr)
    except FileNotFoundError:
        pass

all_mrrs = []
all_stds = []
for order in orders:
    for dataset in datasets:
        for config in configs:
            config_name = '.'.join(config.split('.')[:-1])
            mrrs = list()
            for i in range(1, args.runs+1):
                get_test_mrr(log_dir + '/{}_{}_{}_{}.out'.format(order, dataset, config_name, i), mrrs)

            if len(mrrs) > 0:
                mrrs = np.array(mrrs)
                print('{}_{}_{}:{:.4f}+-{:.4f}'.format(order, dataset, config_name, np.mean(mrrs), np.std(mrrs)))
                all_mrrs.append(np.mean(mrrs))
                all_stds.append(np.std(mrrs))
                print(mrrs)
                print()
print(all_mrrs)
print(all_stds)