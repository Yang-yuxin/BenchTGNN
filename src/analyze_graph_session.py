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

# BINS = 100

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

# Preprocessing
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
t_inter_act = []
for i in tqdm(range(num_nodes)):
    tsort_original(i, ext_full_indices[i], ext_full_ts[i], ext_full_eid[i])
    t_inter_act += get_activity_density(i)

t_inter_act = np.array(t_inter_act)
p1 = np.percentile(t_inter_act, 1)
p99 = np.percentile(t_inter_act, 99)

# Filtering the array to remove the smallest and largest 1%
filtered_data = t_inter_act[(t_inter_act > p1) & (t_inter_act < p99)]
# import pdb; pdb.set_trace()
if (len(filtered_data) < 0.5 * len(t_inter_act)) or (min(filtered_data) == max(filtered_data)):
    filtered_data = t_inter_act
x = pd.Series(filtered_data)
plt.subplot(211)
BINS = np.arange(min(x), max(x))
# hist, bins = np.histogram(x, BINS, density=True)
hist, bins, _ = plt.hist(x, BINS, density=True, stacked=True)
xlabel = f'neighbor count'
plt.xlabel(xlabel)
plt.subplot(212)
# plt.hist(x_log, BINS, density=True, stacked=True)
plt.bar(range(len(hist)), np.cumsum(hist))
xlabel = f'cumulative neighbor count'
# plt.xlabel(xlabel)
# plt.legend()
plt.suptitle(f'{args.data} session length distribution')
plt.tight_layout()
plt.savefig(f'figures/session/{args.data}_session_activity_density.pdf')
