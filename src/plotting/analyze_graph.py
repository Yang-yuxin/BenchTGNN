import argparse
import itertools
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import random

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
# import pdb; pdb.set_trace()

def isbetween(a, b, c):
    if a > b and c > a:
        return True
    if a < b and c < a:
        return True
    return False

def get_in_between(src):
    between = 0
    possible = 0
    neighs, ts, eid  = ext_full_indices[src], ext_full_ts[src], ext_full_eid[src]
    last_dict = {}
    for i in range(len(neighs)):
        if neighs[i] in last_dict.keys():
            last_dict[neighs[i]] = max(ts[i], last_dict[neighs[i]])
        else:
            last_dict[neighs[i]] = ts[i]
    last = np.zeros_like(ts)
    between = np.zeros_like(neighs)
    possible = np.zeros_like(neighs)
    for i in range(len(neighs)):
        last[i] = last_dict[neighs[i]]
    # if src == 12:
    #     import pdb; pdb.set_trace()
    for i in range(len(neighs)):
        u = neighs[i]
        ti = ts[i]
        between += np.multiply(np.isclose(neighs, u), np.multiply(ts <= ti, ti <= last[i]))
        possible += np.multiply(ts <= ti, ti <= last[i])
    return between, possible

all_crit = np.zeros(num_nodes)
all_bi = np.zeros(num_nodes)
all_pi = np.zeros(num_nodes)
for i in range(num_nodes):
    bi, pi = get_in_between(i)
    all_bi[i] = sum(bi)
    all_pi[i] = sum(pi)
    if sum(pi) == 0:
        continue
    all_crit[i] = sum(bi) / sum(pi)
import pdb; pdb.set_trace()


ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')

def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]


for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

print('saving...')

# np.savez('DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr,
#          indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)
