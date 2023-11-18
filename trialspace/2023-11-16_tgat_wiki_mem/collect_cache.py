import os
from collections import OrderedDict

import re

import numpy as np
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, help='log file directory')
args = parser.parse_args()

model = 'tgat'
datasets = ['wiki', 'reddit', 'movielens', 'gdelt']
cache_ratios = [0.1, 0.2, 0.3]

cache_dict = OrderedDict({'epoch': [i for i in range(200)]})
for dataset in datasets:
    for cache_ratio in cache_ratios:
        cache_dict[dataset+'_{}'.format(cache_ratio)] = []
        cache_dict['o'+dataset+'_{}'.format(cache_ratio)] = []

for dataset in datasets:
    for cache_ratio in cache_ratios:
        path = os.path.join(args.log_dir, '{}_{}_{}'.format(model, dataset, cache_ratio))
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('\tcache hit rate'):
                    hit_rates = re.findall(r"\d+\.\d+", line)
                    cache_dict[dataset+'_{}'.format(cache_ratio)].append(hit_rates[0])
                    cache_dict['o'+dataset+'_{}'.format(cache_ratio)].append(hit_rates[1])

df = pd.DataFrame(cache_dict)
df.to_csv(os.path.join(args.log_dir, 'cache.csv'), index=False)
