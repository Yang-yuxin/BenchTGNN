import os

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='trial name')
parser.add_argument('--log_dir', type=str, help='log file directory')
parser.add_argument('--num_scope', type=int, default=25, help='trial name')
parser.add_argument('--num_neighbor', type=int, default=10, help='trial name')
parser.add_argument('--runs', type=int, default=5, help='trial name')
args = parser.parse_args()

log_dir = args.log_dir
config_dir = 'config_train' + '/{}'.format(args.trial)

datasets = ['WIKI', 'REDDIT', 'Flight', 'MovieLens', 'sGDELT', 'GDELT']
configs = [f for f in os.listdir(config_dir) if f.endswith('.yml')]


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


for dataset in datasets:
    for config in configs:
        config_name = '.'.join(config.split('.')[:-1])
        mrrs = list()
        for i in range(1, args.runs+1):
            get_test_mrr(log_dir + '/{}_{}_{}_{}_{}.out'.format(args.num_scope, args.num_neighbor,
                                                                dataset, config_name, i), mrrs)

        if len(mrrs) > 0:
            mrrs = np.array(mrrs)
            print('{}_{}_{}_{}:{:.4f}+-{:.4f}'.format(args.num_scope, args.num_neighbor,
                                                   dataset, config_name, np.mean(mrrs), np.std(mrrs)))
            print(mrrs)
            print()
