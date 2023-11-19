import torch
import pdb
import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils import DenseTemporalBlock
from dataloader import DataLoader
import argparse
from utils import *
from temporal_sampling import sample_with_pad
import nfft

def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X

data = ['WIKI', 'REDDIT']
num_nodes = 100
root_path = 'DATA'
num_sample = 50
type_sample='recent'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
window_size = 1



fig, axss = plt.subplots(2, len(data), figsize=(15, 10))
for d, axs in zip(data, axss):
    g, edges, nfeat, efeat = load_data(d, root_path)
    device = 'cuda'
    cached_ratio = 0.3
    cache = False
    pure_gpu = False

    g = [el.to(device) for el in g]
    pt = torch.load('DATA/{}/edges.pt'.format(d))
    ft = torch.load('DATA/{}/edge_features.pt'.format(d))
    
    dummy_nid = nfeat.shape[0] - 1 if nfeat is not None else g[0].shape[0] - 1
    dummy_eid = efeat.shape[0] - 1 if efeat is not None else g[1].max() + 1
    root_nid = []
    dst_nid = []
    self_feats = []
    root_ts = []
    tmp = 0
    while (len(root_nid) < num_nodes):
        tmp += 1
        if torch.where(pt['train_src'] == tmp)[0].shape[0] > (num_sample+1):
            root_nid.append(tmp)
            l = torch.where(pt['train_src'] == tmp)[0].shape[0]
            ind_t = np.random.randint(num_sample+1, l)
            root_ts.append(pt['train_time'][torch.where(pt['train_src'] == tmp)[0]][ind_t])
            dst_nid.append(pt['train_dst'][torch.where(pt['train_src'] == tmp)[0]][ind_t])
            self_feats.append(ft[torch.where(pt['train_src'] == tmp)[0][ind_t], :])

    root_nid = torch.tensor(root_nid).to(device)
    root_ts = torch.tensor(root_ts).to(device)
    dst_nid = torch.tensor(dst_nid).to(device)
    self_feats = torch.tensor(np.array(self_feats)).to(device)
    # import pdb; pdb.set_trace()
    neigh_nids, neigh_eids, neigh_ts = sample_with_pad(
                root_nid, root_ts,
                g[0], g[1], g[2], g[3],
                num_sample, type_sample,
                dummy_nid, dummy_eid
            )
    block = DenseTemporalBlock(root_nid, root_ts, neigh_nids, neigh_eids, neigh_ts,
                                       dummy_nid, dummy_eid)
    block.slice_input_edge_features(ft)
    xs = []
    ys = []
    x_magnitude = []
    x_phase = []
    for j in range(num_nodes):
    # ind_nid = np.random.randint(0, len(root_nid)-1)
        ind_nid = j
        nid = root_nid[ind_nid]
        neigh_nid, neigh_feat, neigh_t, self_feat = neigh_nids[ind_nid], block.neighbor_edge_feature[ind_nid, :, :], neigh_ts[ind_nid, :], self_feats[ind_nid, :]
        dist = []
        for i in range(1, num_sample+1, 1):
            smooth_feat = torch.mean(neigh_feat[i-window_size:i], 0) if i>=5 else torch.mean(neigh_feat[0:i], 0)
            dist.append(torch.norm(smooth_feat).cpu())
        # x = [i for i in range(1, num_sample, 1)]
        n = np.arange(len(dist))
        sr = 1
        ts = 1.0/sr
        T = len(dist)/sr
        x = n/T 
        y = dist - np.mean(dist)
        xs.append(x)
        y = nfft.nfft(neigh_t.cpu(), y)
        ys.append(y)
        x_magnitude.append(np.abs(y))
        x_phase.append(np.angle(y))
    ys = np.array(ys)
    x_magnitude = np.array(x_magnitude)
    x_phase = np.array(x_phase)
    # for j in range(num_nodes):
        # axs.plot(xs[j], ys[j], marker='o', markersize=0.5)
    axs[0].stem(x, np.mean(x_magnitude, 0))
    axs[1].stem(x, np.mean(x_phase, 0))
    # path_config = 'config_train/tgat_wiki/TGAT.yml'
    # config = yaml.safe_load(open(path_config, 'r'))
    # train_loader = DataLoader(g, config['scope'][0]['neighbor'],
    #                       edges['train_src'], edges['train_dst'], edges['train_time'], edges['neg_dst'],
    #                       nfeat, efeat, config['train'][0]['batch_size'],
    #                       sampler=None,
    #                       device=device, mode='train',
    #                       type_sample=config['scope'][0]['strategy'],
    #                       order=config['train'][0]['order'],
    #                       # edge_deg=edges['train_deg'],
    #                       cached_ratio=cached_ratio, enable_cache=cache, pure_gpu=pure_gpu)
    # blocks = train_loader.get_blocks(log_cache_hit_miss=False)

fig.tight_layout()
plt.savefig('feat_dist_50_nfft_norm.png', dpi=300)