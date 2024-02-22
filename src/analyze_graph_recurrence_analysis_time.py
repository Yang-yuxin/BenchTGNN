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

BINS = 100

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')

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

# if args.data == 'WIKI':
#     for i in range(len(ext_full_ts)):
#         random.shuffle(ext_full_ts[i])

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

def get_inter_activity_time(src):
    neighs, ts, eid  = ext_full_indices[src], ext_full_ts[src], ext_full_eid[src]
    intervals = []
    i = 0
    while i < len(neighs):
        ni = neighs[i]
        j = i
        while j < len(neighs)-1:
            nj = neighs[j]
            nk = neighs[j+1]
            if (ni == nj and ni != nk):
                intervals.append(ts[j] - ts[i])
                break
            elif j == len(neighs) - 2:
                intervals.append(ts[j+1] - ts[i])
                break
            else:
                j += 1
            
        i = j + 1
    return intervals

def get_activity_density(src):
    neighs, ts, eid  = ext_full_indices[src], ext_full_ts[src], ext_full_eid[src]
    intervals = []
    i = 0
    while i < len(neighs):
        ni = neighs[i]
        j = i
        l = 0
        while j < len(neighs)-1:
            nj = neighs[j]
            nk = neighs[j+1]
            if (ni == nj and ni != nk):
                intervals.append(j - i)
                break
            elif j == len(neighs) - 2:
                intervals.append(j + 1 - i)
                break
            else:
                j += 1
        i = j + 1
    return intervals
    
def get_recurrence_quantification(src):
    neighs, ts, eid  = ext_full_indices[src], ext_full_ts[src], ext_full_eid[src]
    intervals = []
    recur_matrix = np.zeros((len(neighs), len(neighs)))
    for i in range(len(neighs)):
        for j in range(i, len(neighs)):
            if (neighs[j] == neighs[i]):
                recur_matrix[i][j] = i + 100
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
for i in tqdm(range(num_nodes)):
    tsort_original(i, ext_full_indices[i], ext_full_ts[i], ext_full_eid[i])
    recur_matrix, s_recur = get_recurrence_quantification(i)
    recur_matrices.append(recur_matrix)
    if s_recur > 20:
        plt.matshow(recur_matrices[-1])
        break



plt.suptitle(f'{args.data} recurrence distribution of node {i}')
# plt.tight_layout()
plt.savefig(f'{args.data}_recurrence_analysis.png')
# import pdb; pdb.set_trace()