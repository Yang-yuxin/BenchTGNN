import argparse
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='WIKI')
parser.add_argument('--clip_root_set', action='store_true', help='clip root set to 1M, train:test:val = 60%:20%:20%')
parser.add_argument('--clip_deg', type=int, default=32000, help='maximum number of neighbors per node in the adjacency matrix')
parser.add_argument('--num_val_neg_dst', type=int, default=9)
parser.add_argument('--num_test_neg_dst', type=int, default=49)
args = parser.parse_args()

is_bipartite = True if args.data in [
    'WIKI', 'REDDIT', 'MOOC', 'LASTFM', 'Taobao', 'sTaobao', 'MovieLens'] else False

df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
if 'src' in df.columns:
    src = torch.as_tensor(df['src'].to_numpy(dtype=np.int64))
    dst = torch.as_tensor(df['dst'].to_numpy(dtype=np.int64))
    time = torch.as_tensor(df['time'].to_numpy(dtype=np.float32))
else:
    src = torch.as_tensor(df['u'].to_numpy(dtype=np.int64))
    dst = torch.as_tensor(df['i'].to_numpy(dtype=np.int64))
    time = torch.as_tensor(df['ts'].to_numpy(dtype=np.float32))
pt_edge = dict()

# neg dst set
if is_bipartite:
    pt_edge['neg_dst'] = torch.unique(dst)
else:
    pt_edge['neg_dst'] = torch.arange(max(src.max(), dst.max()) + 1)

# idx split
if 'ext_roll' in df.columns:
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
else:
    train_edge_end = int(df.shape[0]*0.7)
    val_edge_end = int(df.shape[0]*0.85)

if args.clip_root_set:
    max_num_set = int(1e6)
    if src.shape[0] > max_num_set:
        src = src[-max_num_set:]
        dst = dst[-max_num_set:]
        time = time[-max_num_set:]
        train_edge_end = int(max_num_set * 0.7)
        val_edge_end = train_edge_end + int(max_num_set * 0.15)

pt_edge['train_src'] = src[:train_edge_end]
pt_edge['train_dst'] = dst[:train_edge_end]
pt_edge['train_time'] = time[:train_edge_end]
pt_edge['val_src'] = src[train_edge_end:val_edge_end]
pt_edge['val_dst'] = dst[train_edge_end:val_edge_end]
pt_edge['val_time'] = time[train_edge_end:val_edge_end]
pt_edge['test_src'] = src[val_edge_end:]
pt_edge['test_dst'] = dst[val_edge_end:]
pt_edge['test_time'] = time[val_edge_end:]

# fixed negative node for val/test
val_neg_idx = torch.randint(low=0, high=pt_edge['neg_dst'].shape[0],
                             size=(args.num_val_neg_dst * pt_edge['val_src'].shape[0],))
test_neg_idx = torch.randint(low=0, high=pt_edge['neg_dst'].shape[0],
                             size=(args.num_test_neg_dst * pt_edge['test_src'].shape[0],))
pt_edge['val_neg_dst'] = pt_edge['neg_dst'][val_neg_idx]
pt_edge['test_neg_dst'] = pt_edge['neg_dst'][test_neg_idx]

torch.save(pt_edge, 'DATA/{}/edges.pt'.format(args.data))

# process features (add dummy node/edge)
if os.path.isfile('DATA/{}/node_features.pt'.format(args.data)):
    if not os.path.isfile('DATA/{}/node_features_pad.pt'.format(args.data)):
        feat = torch.load('DATA/{}/node_features.pt'.format(args.data))
        feat = torch.cat([feat, torch.zeros_like(feat[0]).unsqueeze(0)], dim=0)
        torch.save(feat, 'DATA/{}/node_features_pad.pt'.format(args.data))
if os.path.isfile('DATA/{}/edge_features.pt'.format(args.data)):
    if not os.path.isfile('DATA/{}/edge_features_pad.pt'.format(args.data)):
        feat = torch.load('DATA/{}/edge_features.pt'.format(args.data))
        feat = torch.cat([feat, torch.zeros_like(feat[0]).unsqueeze(0)], dim=0)
        torch.save(feat, 'DATA/{}/edge_features_pad.pt'.format(args.data))

# add randomized features for LASTFM, MOOC, and superuser
if not os.path.isfile('DATA/{}/node_features.pt'.format(args.data)) and args.data in ['LASTFM', 'MOOC', 'superuser']:
    if not os.path.isfile('DATA/{}/node_features_pad.pt'.format(args.data)) or True:
        feat = torch.randn(max(src.max(), dst.max()), 128)
        feat = torch.cat([feat, torch.zeros_like(feat[0]).unsqueeze(0)], dim=0)
        torch.save(feat, 'DATA/{}/node_features_pad.pt'.format(args.data))

# process graph
g = np.load('DATA/{}/ext_full.npz'.format(args.data))
indptr = g['indptr']
degptr = np.diff(indptr, prepend=0)
indices = g['indices']
eid = g['eid']
ts = g['ts']
import pdb; pdb.set_trace()

keep = np.ones_like(indices, dtype=np.bool_)
for i in tqdm(range(indptr.shape[0] - 1)):
    if degptr[i + 1] > args.clip_deg:
        to_clip = degptr[i + 1] - args.clip_deg
        clip_idx = np.random.choice(degptr[i + 1], to_clip, replace=False) + indptr[i]
        keep[clip_idx] = False
        degptr[i + 1] = args.clip_deg
indptr = np.cumsum(degptr)
indices = indices[keep]
eid = eid[keep]
ts = ts[keep]

np.savez('DATA/{}/ext_full_clipped.npz'.format(args.data), indptr=indptr,
         indices=indices, ts=ts, eid=eid)


