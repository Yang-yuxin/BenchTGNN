import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='trial name')
parser.add_argument('--log_dir', type=str, default='log', help='log file directory')
parser.add_argument('--all_in_one', type=bool, default=True)
parser.add_argument('--export', type=str, default='')
parser.add_argument('--target', type=str, default='mrr', choices=['mrr', 'epoch', 'time'])
parser.add_argument('--num_scope', type=int, default=25, help='trial name')
parser.add_argument('--num_neighbor', type=int, default=10, help='trial name')
parser.add_argument('--runs', type=int, default=5, help='trial name')
args = parser.parse_args()
log_dir = args.log_dir
config_dir = 'config' + '/{}'.format(args.trial)


# scans = ['5', '10', '20', '50', '100']
scans = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '20', '50', '100', '5x5', '5x10', '10x5', '10x10', '20x20', '30x30', '40x40']
datasets = ['WIKI', 'REDDIT', 'Flights', 'LASTFM', 'mooc', 'uci', 'CollegeMsg']
aggrs = ['TGAT', 'GraphMixer']
samplings = ['re', 'uni']
memorys = ['gru', 'embed', '']
# configs = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
green_colors = ['#98FB98', '#6B8E23', '#008080', '#00441b']  # Shades of green
red_colors = ['#FF9999', '#fb6a4a', '#cb181d', '#990000']  # Shades of red
colors = green_colors + red_colors


            

def get_test_mrr(file_path):
    try:
        with open(file_path, 'r') as f:
            mrr = None
            for lines in f:
                if lines.startswith('\ttest AP'):
                    mrr = float(lines.strip('\n').split(':')[-1])
                    return mrr
    except FileNotFoundError:
        import pdb; pdb.set_trace()
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
        import pdb; pdb.set_trace()
        pass

def get_epoch_time(file_path):
    try:
        with open(file_path, 'r') as f:
            t = 0
            nt = 0
            for lines in f:
                if lines.startswith('	train time:'):
                    t += float(lines.split(':')[1].split('s')[0])
                    nt+=1
            if nt == 0:
                print(file_path)
                return None
            print(file_path, t, nt)
            return t / nt
        pass
    except FileNotFoundError:
        import pdb; pdb.set_trace()
        pass

all_data = {}


for dataset in datasets:
    all_data[dataset] = {}
all_results = []
all_stds = []
for root, dirs, files in os.walk(log_dir):
    try:
        if (int(root.split('-')[1]) >= 3 and int(root.split('_')[0].split('-')[2]) >= 19 and 'scan' in root):
            print(root)
        else:
            continue
        for file in files:
            path = os.path.join(root, file)
            if 'result' in file:
                continue
            # scan = root.split('_')[-1] if 'mixer' not in root else root.split('_')[-2]
            if 'mixer' not in root and 'none' not in root:
                scan = root.split('_')[-1]
            elif 'none' not in root:
                scan = root.split('_')[-2]
            else:
                scan = root.split('_')[-2]

            scan, dataset = file.split('_')[0:2]
            if args.target == 'mrr':
                result = get_test_mrr(path)
            elif args.target == 'time':
                result = get_epoch_time(path)
            else:
                result = get_best_epoch(path)
            # if len(results) > 0:
            #     results = np.array(results)
            #     # print('{}_{}_{}_{}_{}:{:.4f}+-{:.4f}'.format(dataset, scan, aggr, sampling, memory, np.mean(mrrs), np.std(mrrs)))
            #     all_results.append(np.mean(results))
            #     all_stds.append(np.std(results))
            if result is None:
                continue
            try:
                all_data[dataset][scan] = result
            except KeyError:
                import pdb; pdb.set_trace()
    except IndexError:
        pass
# for dataset in datasets:
#     for aggr in aggrs:
#         for sampling in samplings:
#             for memory in memorys:
#                 means = []
#                 stds = []
#                 for scan in scans:
#                     results = all_data[dataset][scan][aggr][sampling][memory]
#                     results = [x for x in results if x is not None]
#                     if len(results) > args.runs:
#                         results = results[:args.runs]
#                     all_data[dataset][scan][aggr][sampling][memory] = results

# save data
with open(args.export, 'wb') as f:
    pickle.dump(all_data, f)

