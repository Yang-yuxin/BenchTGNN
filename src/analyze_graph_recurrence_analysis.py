import argparse
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import random
from scipy.optimize import minimize
from scipy.stats import norm


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')
parser.add_argument('--total', action='store_true')
parser.add_argument('--use_real_time', action='store_true')
parser.add_argument('--bins', type=int, default=-1)
args = parser.parse_args()

print(args)

df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
s = 'src' if 'src' in df.columns else 'u'
d = 'dst' if 'dst' in df.columns else 'i'
t = 'time' if 'time' in df.columns else 'ts'
num_nodes = max(int(df[s].max()), int(df[d].max())) + 1
print('num_nodes: ', num_nodes)
# import pdb; pdb.set_trace()
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

# BINS = 1000 if args.data in ['uci', 'CollegeMsg'] else 10000
BINS = int(min(10 ** int(np.log10(len(df)) - 1), 1e4)) if (args.bins == -1) else args.bins

tmp = np.array(list(itertools.chain(*ext_full_ts)))
min_ts = np.min(tmp)
max_ts = np.max(tmp)
print(f'Bins: {BINS}\t Time span: {min_ts} - {max_ts}')
time_bins, step = np.linspace(min_ts, max_ts, BINS+1, retstep=True)
# if args.data == 'WIKI':
#     for i in range(len(ext_full_ts)):
#         random.shuffle(ext_full_ts[i])

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

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
                    # if neighs[j] not in color_dict.keys():
                        # color_dict[neighs[j]] = (len(color_dict.keys())+1) * 10
                    #     recur_matrix[idxi][idxj] += (len(color_dict.keys())+1) * 10
                    # else:
                    #     recur_matrix[idxi][idxj] = color_dict[neighs[j]]
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

print('sorting and analyzing...')
recur_matrices = []
color_dict = {}
recur_matrix = np.zeros((BINS+1, BINS+1))
for i in tqdm(range(num_nodes)):
    tsort_original(i, ext_full_indices[i], ext_full_ts[i], ext_full_eid[i])
    # import pdb; pdb.set_trace()
    if args.use_real_time:
        get_recurrence_quantification(i, args.use_real_time, min_ts, step, recur_matrix, color_dict)
    else:
        recur_matrix, s_recur = get_recurrence_quantification(i, args.use_real_time, min_ts, step, color_dict)
        recur_matrices.append(recur_matrix)
    if args.total:
        continue
    if s_recur > 20 and i > 4:
        plt.matshow(recur_matrices[-1])
        break

if args.use_real_time:
    total_matrix = recur_matrix
else:
    most_neighs = max(i.shape[0] for i in recur_matrices)
    total_matrix = np.zeros((most_neighs, most_neighs))
    for i in recur_matrices:
        total_matrix[:i.shape[0], :i.shape[1]] += i
ma, mi, me = np.max(total_matrix[total_matrix > 1e-6]), np.min(total_matrix[total_matrix > 1e-6]), np.mean(total_matrix[total_matrix > 1e-6])
p5 = np.percentile(total_matrix[total_matrix > 1e-6], 5)
p95 = np.percentile(total_matrix[total_matrix > 1e-6], 95)
print([np.percentile(total_matrix[total_matrix > 1e-6], i) for i in range(5, 100, 5)])
print(ma, mi, me, p5, p95)
scalar = p95 - p5

total_matrix[(total_matrix > p95)] = p95 - 0.1
total_matrix[(total_matrix < p5) & (total_matrix > 1e-6)] = p5 + 0.1
    # for i in range(total_matrix.shape[0]):
    #     for j in range(total_matrix.shape[1]):
    #         if total_matrix[i][j] == 0:
    #             continue
    #         if total_matrix[i][j] > p95:
    #             total_matrix[i][j] = p95 - 0.1
    #         if total_matrix[i][j] < p5:
    #             total_matrix[i][j] = p5 + 0.1
import pdb; pdb.set_trace()
total_matrix /= scalar
plt.matshow(total_matrix)
if args.total and args.use_real_time:
    plt.suptitle(f'{args.data} recurrence distribution of all nodes using real time')
elif args.total:
    plt.suptitle(f'{args.data} recurrence distribution of all nodes')
else:
    plt.suptitle(f'{args.data} recurrence distribution of node {i}')
# plt.tight_layout()
plt.savefig(f'{args.data}_total_recurrence_analysis_{args.use_real_time}.png')
# import pdb; pdb.set_trace()