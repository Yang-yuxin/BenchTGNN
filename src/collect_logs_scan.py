import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='trial name')
parser.add_argument('--log_dir', type=str, default='log', help='log file directory')
parser.add_argument('--num_scope', type=int, default=25, help='trial name')
parser.add_argument('--num_neighbor', type=int, default=10, help='trial name')
parser.add_argument('--runs', type=int, default=5, help='trial name')
args = parser.parse_args()

log_dir = args.log_dir
config_dir = 'config' + '/{}'.format(args.trial)

scans = ['5', '10', '20', '50', '100']
datasets = ['WIKI', 'REDDIT', 'Flights', 'LASTFM', 'mooc', 'uci', 'CollegeMsg']
aggrs = ['GraphMixer', 'TGAT']
samplings = ['re', 'uni']
memorys = ['gru', 'embed']
# configs = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
green_colors = ['#98FB98', '#6B8E23', '#008080', '#00441b']  # Shades of green
red_colors = ['#FF9999', '#fb6a4a', '#cb181d', '#990000']  # Shades of red
colors = green_colors + red_colors

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

def get_best_epoch(file_path):
    try:
        with open(file_path, 'r') as f:
            epoch = None
            for lines in f:
                if lines.startswith('Loading'):
                    epoch = int(lines.split(' ')[4])
        return epoch
    except FileNotFoundError:
        print(file_path)
        pass

all_data = {}
for dataset in datasets:
    all_data[dataset] = {}
    for scan in scans:
        all_data[dataset][scan] = {}
        for aggr in aggrs:
            all_data[dataset][scan][aggr] = {}
            for sampling in samplings:
                all_data[dataset][scan][aggr][sampling] = {}
                for memory in memorys:
                    all_data[dataset][scan][aggr][sampling][memory] = []

for root, dirs, files in os.walk(log_dir):
    try:
        if not(root.split('-')[1] == '01' and int(root.split('_')[0].split('-')[2]) >= 25 and 'scan' in root):
            continue
        for file in files:
            path = os.path.join(root, file)
            if 'result' in file:
                continue
            scan = root.split('_')[-1]
            dataset, aggr, sampling, memory = file.split('_')[0:4]
            all_data[dataset][scan][aggr][sampling][memory].append(get_best_epoch(path, ))
    except IndexError:
        pass
for dataset in datasets:
    df = pd.DataFrame()
    for aggr in aggrs:
        for sampling in samplings:
            for memory in memorys:
                records = []
                for scan in scans:
                    record = all_data[dataset][scan][aggr][sampling][memory]
                    if None not in record and len(record) == 5:
                        print(f"{dataset}\t{scan}\t{aggr}\t{sampling}\t{memory}\t{np.mean(record)}")
                        records.append(np.mean(record))
                if len(records) == 5:
                    print(records)
                    df[f"{aggr}+{sampling}+{memory}"] = records
    print(df)
    if df.empty:
        continue
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(df.columns):
        plt.plot(np.arange(5), df[col], label=col, color=colors[i])
    plt.title(f'{dataset} Best Epoch with Different Neighbor Count')
    x_labels = [5, 10, 20, 50, 100]
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
    plt.legend()
    plt.savefig(f'{dataset}_best_epoch_plot.png')

# green_colors = ['#98FB98', '#6B8E23', '#008080', '#00441b']  # Shades of green
# red_colors = ['#FF9999', '#fb6a4a', '#cb181d', '#990000']  # Shades of red
# colors = green_colors + red_colors


# print(all_data)
# for dataset in datasets:
#     for config in configs:
#         config_name = '.'.join(config.split('.')[:-1])
#         mrrs = list()
#         for i in range(1, args.runs+1):
#             get_test_mrr(log_dir + '/{}_{}_{}_{}_{}.out'.format(args.num_scope, args.num_neighbor,
#                                                                 dataset, config_name, i), mrrs)

#         if len(mrrs) > 0:
#             mrrs = np.array(mrrs)
#             print('{}_{}_{}_{}:{:.4f}+-{:.4f}'.format(args.num_scope, args.num_neighbor,
#                                                    dataset, config_name, np.mean(mrrs), np.std(mrrs)))
#             print(mrrs)
#             print()
