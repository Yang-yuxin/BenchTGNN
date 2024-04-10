import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os
import os.path as osp

TGAP = 10 # time gap
TSIGMA = 0.5
TEMP = 50

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')
parser.add_argument('--N', type=int, help='node count', default=10000)
parser.add_argument('--E', type=int, help='edge count', default=100000)
parser.add_argument('--featN', type=int, help='node feature dimension', default=10)
parser.add_argument('--featE', type=int, help='edge feature dimension', default=10)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--train_ratio', type=float, default=0.7)

args = parser.parse_args()

num_nodes = args.N
print('num_nodes: ', num_nodes)
split = [args.train_ratio, args.train_ratio+(1.0-args.train_ratio)/2]
split = [int(a * args.E) for a in split]
ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

mu, sigma = 3, 1.2 # mean and standard deviation
CCs = np.random.normal(mu, sigma, args.N)
CCs[CCs < 0.1] = 0.1
CCs /= np.sum(CCs)
pdf_CC = np.cumsum(CCs)
edge_counter = 0

def sample_decay(hist_neighs, hist_ts, ts, alpha):
    try:
        assert (ts >= hist_ts).all()
    except AssertionError:
        import pdb; pdb.set_trace()

    probs = alpha ** ((ts - hist_ts) / TEMP)
    pdf_probs = np.cumsum(probs)
    pdf_probs /= pdf_probs[-1]
    dart = np.random.uniform()
    # if len(hist_ts) > 3:
    #     print(hist_ts)
    #     print(pdf_probs)
    #     print(dart)
    #     print(np.searchsorted(pdf_probs, dart))
    #     import pdb; pdb.set_trace()
    return hist_neighs[np.searchsorted(pdf_probs, dart)]

# 1. find src with CC
# 2. if src doesn't have neighbors, find dst with CC
# 3. if src has neighbors, with prob beta sample from historical neighbors, (1-beta) sample dst with CC
# 4. sample from historical neighbors: p(j) prop to alpha^(t-t^j)
for it in range(args.T):
    t = (it+1) * TGAP
    n = args.E // args.T
    darts = np.random.uniform(size=n)
    src_nodes = np.searchsorted(pdf_CC, darts)
    src_times = np.random.uniform(t, TSIGMA, n)
    src_times[src_times < it * TGAP] = it*TGAP
    src_times[src_times > (it+2) * TGAP] = (it+2)*TGAP
    mask_no_neighbors = np.array([len(ext_full_indices[s])==0 for s in src_nodes])
    darts_new = np.random.uniform(size=len(src_nodes))
    mask_darts = np.array([darts_new>args.beta])
    mask = np.logical_or(mask_no_neighbors, mask_darts).squeeze() # new:
    # import pdb; pdb.set_trace()
    darts = np.random.uniform(size=mask.sum())
    dst_partial_nodes = np.searchsorted(pdf_CC, darts)
    left_src_nodes = src_nodes[~mask] # historical
    left_src_times = src_times[~mask]
    append_info = []
    for ie in range(len(left_src_nodes)):
        src_node = left_src_nodes[ie]
        dst_node = sample_decay(ext_full_indices[src_node], ext_full_ts[src_node], left_src_times[ie], args.alpha)
        append_info.append([src_node, dst_node, left_src_times[ie]])
    for info in append_info:
        ext_full_indices[info[0]].append(info[1])
        ext_full_ts[info[0]].append(info[2])
        ext_full_eid[info[0]].append(edge_counter)
        # append({'':edge_counter, 'src':src_node, 'dst': dst_node, 'time':left_src_times[ie]}, ignore_index=True)
        edge_counter += 1
    for idx_s, s in enumerate(src_nodes[mask]):
        ext_full_indices[s].append(dst_partial_nodes[idx_s])
        ext_full_ts[s].append(src_times[idx_s])
        ext_full_eid[s].append(edge_counter)
        # '':edge_counter, 'src':s, 'dst': dst_partial_nodes[idx_s], 'time':src_times[idx_s]}, ignore_index=True)
        edge_counter += 1
    

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))
ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))




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

src_nodes = []
for i in range(1, args.N+1):
    src_nodes += [i for _ in range(ext_full_indptr[i]-ext_full_indptr[i-1])]
edge_dict = {
    # '': ext_full_eid,
    'src': src_nodes,
    'dst': ext_full_indices,
    'time': ext_full_ts,

}
edge_df = pd.DataFrame(edge_dict)
edge_df = edge_df.sort_values('time')
edge_df['ext_roll']=[0 for _ in range(split[0])] + [1 for _ in range(split[1]-split[0])] + [2 for _ in range(args.E - split[1])]
print('saving...')
if not osp.exists('DATA/{}/'.format(args.data)):
    os.makedirs('DATA/{}/'.format(args.data))
else:
    pass
edge_df.to_csv('DATA/{}/edges.csv'.format(args.data))
np.savez('DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr,
         indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)
