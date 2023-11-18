import argparse
import time
from typing import List

import torch
import yaml

from utils import load_data, DenseTemporalBlock
from dataloader import DataLoader
from temporal_sampling import sample_with_pad
from model import EdgeBank
from sklearn.metrics import average_precision_score

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

@torch.no_grad()
def eval(model, dataloader):
    model.eval()
    if dataloader.sampler is not None:
        dataloader.sampler.eval()
    aps = list()
    mrrs = list()
    while not dataloader.epoch_end:
        blocks = dataloader.get_blocks()
        pred_pos, pred_neg = model(blocks)
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).cpu()
        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
        # print(sum(y_pred), sum(y_true))
        aps.append(average_precision_score(y_true, y_pred))
        mrrs.append(torch.reciprocal(
            torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(blocks[-1].num_neg_dst, -1), dim=0) + 1).type(
            torch.float))
    dataloader.reset()
    ap = float(torch.tensor(aps).mean())
    mrr = float(torch.cat(mrrs).mean())
    return ap, mrr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='sample')
    args = parser.parse_args()
    # locals()["test_" + args.case]()
    data_name = 'WIKI'
    root_path = 'DATA'
    g, edges, nfeat, efeat = load_data(data_name, root_path)
    model = EdgeBank()
    sampler = None
    device = 'cuda'
    config = yaml.safe_load(open('config_train/tgat_wiki/TGAT.yml', 'r'))
    config['scope'][0]['layer'] = 1
    config['scope'][0]['neighbor'] = [1000, 1]
    val_loader = DataLoader(g, config['scope'][0]['neighbor'],
                        edges['val_src'], edges['val_dst'], edges['val_time'], edges['neg_dst'],
                        nfeat, efeat, config['eval'][0]['batch_size'],
                        sampler=sampler,
                        device=device, mode='val',
                        eval_neg_dst_nid=edges['val_neg_dst'],
                        type_sample=config['scope'][0]['strategy'],
                        enable_cache=False, pure_gpu=True)
    ap, mrr = eval(model, val_loader)
    print(f'AP: {ap}, mrr: {mrr}')
