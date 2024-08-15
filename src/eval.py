import os
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--root_path', type=str, default='DATA', help='dataset root path')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default="0", help='which GPU to use')
parser.add_argument('--eval_neg_samples', type=int, default=49, help='how many negative samples to use at inference.')
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--ind_ratio', type=float, default=0.1)
parser.add_argument('--cached_ratio', type=float, default=0.3, help='the ratio of gpu cached edge feature')
parser.add_argument('--cache', action='store_true', help='cache edge features on device')
parser.add_argument('--pure_gpu', action='store_true', help='put all edge features on device, disable cache')
parser.add_argument('--print_cache_hit_rate', action='store_true', help='print cache hit rate each epoch. Note: this will slowdown performance.')

parser.add_argument('--tb_log_prefix', type=str, default='', help='prefix for the tb logging data.')

parser.add_argument('--profile', action='store_true', help='whether to profile.')
parser.add_argument('--profile_prefix', default='log_profile/', help='prefix for the profiling data.')

parser.add_argument('--no_time', action='store_true', help='do not record time (avoid extra cuda synchronization cost).')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import globals

import yaml
from utils import *
from model import TGNN
from dataloader import DataLoader
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

if not args.no_time:
    globals.timer.set_enable()

@torch.no_grad()
def eval(model, dataloader):
    model.eval()
    aps = list()
    mrrs = list()
    while not dataloader.epoch_end:
        blocks, messages = dataloader.get_blocks()
        pred_pos, pred_neg = model(blocks, messages)
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred))
        mrrs.append(torch.reciprocal(
            torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(blocks[-1].num_neg_dst, -1), dim=0) + 1).type(
            torch.float))
    dataloader.reset()
    ap = float(torch.tensor(aps).mean())
    mrr = float(torch.cat(mrrs).mean())
    return ap, mrr


config = yaml.safe_load(open(args.config, 'r'))

"""Data"""
g, edges, nfeat, efeat = load_data(args.data, args.root_path)
if efeat is not None and efeat.dtype == torch.bool:
    efeat = efeat.to(torch.int8)
if nfeat is not None and nfeat.dtype == torch.bool:
    nfeat = nfeat.to(torch.int8)
dim_edge_feat = efeat.shape[-1] if efeat is not None else 0
dim_node_feat = nfeat.shape[-1] if nfeat is not None else 0
n_node = nfeat.shape[0] if nfeat is not None else g[0].shape[0]

"""Model"""
path_model = args.model_path
device = 'cuda'
params = []
model = TGNN(args, config, device, n_node,dim_node_feat, dim_edge_feat).to(device)
params.append({
    'params': model.parameters(),
    'lr': config['train'][0]['lr']
})
optimizer = torch.optim.Adam(params)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
train_edge_end = len(edges['train_src'])
val_edge_end = len(edges['train_src']) + len(edges['val_src'])

"""Loader"""
test_loader = DataLoader(g, config['scope'][0]['neighbor'],
                        edges['test_src'], edges['test_dst'], edges['test_time'], edges['neg_dst'],
                        nfeat, efeat, train_edge_end, val_edge_end, config['eval'][0]['batch_size'],
                        device=device, mode='test', ind=args.inductive, ind_ratio=args.ind_ratio,
                        eval_neg_dst_nid=edges['test_neg_dst'],
                        type_sample=config['scope'][0]['strategy'],
                        memory=config['gnn'][0]['memory_type'],
                        enable_cache=False, pure_gpu=args.pure_gpu)
masked_nodes = test_loader.inductive_mask
train_loader = DataLoader(g, config['scope'][0]['neighbor'],
                          edges['train_src'], edges['train_dst'], edges['train_time'], edges['neg_dst'],
                          nfeat, efeat, train_edge_end, val_edge_end, config['train'][0]['batch_size'],
                          device=device, mode='train', ind=args.inductive, inductive_mask=masked_nodes,
                          type_sample=config['scope'][0]['strategy'],
                          order=config['train'][0]['order'],
                          memory=config['gnn'][0]['memory_type'],
                          # edge_deg=edges['train_deg'],
                          cached_ratio=args.cached_ratio, enable_cache=args.cache, pure_gpu=args.pure_gpu)
val_loader = DataLoader(g, config['scope'][0]['neighbor'],
                        edges['val_src'], edges['val_dst'], edges['val_time'], edges['neg_dst'],
                        nfeat, efeat, train_edge_end, val_edge_end, config['eval'][0]['batch_size'],
                        device=device, mode='val', ind=args.inductive, inductive_mask=masked_nodes,
                        eval_neg_dst_nid=edges['val_neg_dst'],
                        type_sample=config['scope'][0]['strategy'],
                        memory=config['gnn'][0]['memory_type'],
                        enable_cache=False, pure_gpu=args.pure_gpu)

print('Loading model at path {}...'.format(path_model))
param_dict = torch.load(path_model)
model.load_state_dict(param_dict['model'])
ap, mrr = eval(model, test_loader)
print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, mrr))
