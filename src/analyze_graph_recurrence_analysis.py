import argparse
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path as osp
import pickle
from tqdm import tqdm
import random
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns

# plt.rcParams['font.family'] = 'Times New Roman'

def get_recurrence_quantification(src, istime=False, min_ts=None, step=None, recur_matrix=None, color_dict=None):
    neighs, ts, eid  = ext_full_indices[src], ext_full_ts[src], ext_full_eid[src]
    intervals = []
    if not istime:
        recur_matrix = np.zeros((len(neighs), len(neighs)))
    else:
        pass
    for i in range(len(neighs)):
        for j in range(i+1, len(neighs)):
            if (neighs[j] == neighs[i]):
                idxi, idxj = (i, j) if not istime else (int((ts[i] - min_ts) // step), int((ts[j] - min_ts) // step))
                try:
                    recur_matrix[idxi][idxj] += 1
                except IndexError:
                    import pdb; pdb.set_trace()
    if not istime:
        return recur_matrix, np.sum(recur_matrix)

def tsort_original(i, indices, ts, eid):
    if not len(indices):
        return
    try:
        sidx = np.argsort(ts)
        indices = np.array(indices)[sidx]
        ts = np.array(ts)[sidx]
        eid = np.array(eid)[sidx]
    except TypeError:
        import pdb; pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--all_data', action='store_true')
parser.add_argument('--add_reverse', default=False, action='store_true')
parser.add_argument('--use_real_time', action='store_true')
parser.add_argument('--bins', type=int, default=-1)
parser.add_argument('--file_path', type=str, default='')
parser.add_argument('--replace', action='store_true')
args = parser.parse_args()
print(args)

datasets = ['WIKI', 'REDDIT', 'Flights', 'LASTFM', 'mooc', 'uci', 'CollegeMsg']
show_datasets = {
    'WIKI': 'Wikipedia',
    'REDDIT': 'REDDIT',
    'Flights': 'Flights',
    'LASTFM': 'LASTFM',
    'mooc': 'MOOC',
    'uci': 'UCI',
    'CollegeMsg': 'CollegeMsg'
}
if not args.all_data:
    # assert args.data in datasets
    datasets = [args.data,]
else:
    datasets = ['WIKI', 'uci']

total_matrices = []
for data in datasets:
    df = pd.read_csv('DATA/{}/edges.csv'.format(data))
    s = 'src' if 'src' in df.columns else 'u'
    d = 'dst' if 'dst' in df.columns else 'i'
    t = 'time' if 'time' in df.columns else 'ts'
    num_nodes = max(int(df[s].max()), int(df[d].max())) + 1
    print('num_nodes: ', num_nodes)
    # BINS = 1000 if data in ['uci', 'CollegeMsg'] else 10000
    BINS = int(min(10 ** int(np.log10(len(df)) - 1), 1e4)) if (args.bins == -1) else args.bins

    # file_path = f'figures/data/{data}_recur_time.pkl'
    file_path = args.file_path
    if osp.exists(file_path) and not args.replace:
        print('loading...')
        with open(file_path, 'rb') as f:
            all_data = pickle.load(f)
            ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid = all_data['indptr'], \
            all_data['indices'], all_data['ts'], all_data['eid']
            recur_matrix = all_data['recur']
            min_ts, step = all_data['min_ts'], all_data['step']
    else:
        assert not args.all_data, 'Prepare data!'
        print('sorting and analyzing...')
        ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
        ext_full_indices = [[] for _ in range(num_nodes)]
        ext_full_ts = [[] for _ in range(num_nodes)]
        ext_full_eid = [[] for _ in range(num_nodes)]

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            src = int(row[s])
            dst = int(row[d])
            
            ext_full_indices[src].append(dst)
            ext_full_ts[src].append(row[t])
            ext_full_eid[src].append(idx)
            
            if args.add_reverse:
                ext_full_indices[dst].append(src)
                ext_full_ts[dst].append(row[t])
                ext_full_eid[dst].append(idx)

        for i in tqdm(range(num_nodes)):
            ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])
        tmp = np.array(list(itertools.chain(*ext_full_ts)))
        min_ts = np.min(tmp)
        max_ts = np.max(tmp)
        print(f'Bins: {BINS}\t Time span: {min_ts} - {max_ts}')
        time_bins, step = np.linspace(min_ts, max_ts, BINS+1, retstep=True)
        recur_matrices = []
        color_dict = {}
        recur_matrix = np.zeros((BINS+1, BINS+1))
        for i in tqdm(range(num_nodes)):
            tsort_original(i, ext_full_indices[i], ext_full_ts[i], ext_full_eid[i])
            get_recurrence_quantification(i, True, min_ts, step, recur_matrix, color_dict)
        with open(args.file_path, 'wb') as f:
            all_data = {
                'indptr': ext_full_indptr,
                'indices': ext_full_indices,
                'ts': ext_full_ts,
                'eid': ext_full_eid,
                'recur': recur_matrix,
                'min_ts': min_ts,
                'step': step
            }
            pickle.dump(all_data, f)

#     total_matrix = recur_matrix

#     ma, mi, me = np.max(total_matrix[total_matrix > 1e-6]), np.min(total_matrix[total_matrix > 1e-6]), np.mean(total_matrix[total_matrix > 1e-6])
#     p5 = np.percentile(total_matrix[total_matrix > 1e-6], 5)
#     p95 = np.percentile(total_matrix[total_matrix > 1e-6], 95)
#     print(ma, mi, me, p5, p95)
#     scalar = p95 - p5

#     total_matrix[(total_matrix > p95)] = p95 - 0.1
#     total_matrix[(total_matrix < p5) & (total_matrix > 1e-6)] = p5 + 0.1

#     total_matrices.append(total_matrix / scalar)

# fig, axes = plt.subplots(1, len(datasets), sharex=True, sharey=True)
# cbar_ax = fig.add_axes([.91, .3, .03, .4])
# print('Plotting heatmap...')
# for i, ax in enumerate(axes.flat):
#     # sns.heatmap(total_matrices[i], vmin=0, vmax=1, ax=ax, cbar=(i==0), xticklabels=int(BINS/5), 
#     #             yticklabels=int(BINS/5),square=True, cbar_ax=None if i else cbar_ax)
#     im = ax.matshow(total_matrices[i], vmin=0, vmax=1, cmap='rocket')
#     ax.set_title(f'{show_datasets[datasets[i]]}')
# fig.colorbar(im, cax=cbar_ax,)
# print('Saving...')
# fig.tight_layout()
# plt.savefig(f'figures/recurrence/all_recurrence_analysis.png')