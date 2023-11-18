import argparse
import time
from typing import List

import torch

from utils import load_data, DenseTemporalBlock
from dataloader import DataLoader
from temporal_sampling import sample_with_pad


def test_sample():
    g, edge_split, nfeat, efeat = load_data('WIKI')

    batch_index = torch.randperm(len(edge_split['train_src']))[:1000]
    root_idx = edge_split['train_src'][batch_index].to('cuda')
    root_ts = edge_split['train_time'][batch_index].to('cuda')

    dummy_nid = g[0].shape[0] - 1
    dummy_eid = g[2].max().item() + 1

    tik = time.time()
    neigh_nid, neigh_eid, neigh_ts = sample_with_pad(
        root_idx, root_ts,
        g[0].to('cuda'), g[1].to('cuda'), g[2].to('cuda'), g[3].to('cuda'),
        10, 'uniform', dummy_nid, dummy_eid
    )
    print(f'time: {time.time()-tik:.3f}')

    # breakpoint()

    # Test 1: Every sampled neighbor should have a timestamp strictly less than the target timestamp
    ts = neigh_ts - root_ts.unsqueeze(1).repeat(1, neigh_ts.shape[1])
    assert (ts[neigh_nid != dummy_nid] >= 0).sum() == 0

    # breakpoint()


def test_loader():
    g, pt_edge, nfeat, efeat = load_data('WIKI')
    loader = DataLoader(g, [10, 10],
                        pt_edge['train_src'], pt_edge['train_dst'], pt_edge['train_time'], pt_edge['neg_dst'],
                        nfeat, efeat,
                        type_sample='uniform', unique_frontier=True, batch_size=600)
    batch: List[DenseTemporalBlock] = next(iter(loader))
    print(batch[0].neighbor_eid, batch[0].neighbor_eid.shape)
    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='sample')
    args = parser.parse_args()
    locals()["test_" + args.case]()
