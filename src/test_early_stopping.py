import os
import os.path as osp
import argparse
import torch
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
import time


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--root_path', type=str, default='DATA', help='dataset root path')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default="0", help='which GPU to use')
parser.add_argument('--eval_neg_samples', type=int, default=49, help='how many negative samples to use at inference.')

parser.add_argument('--cached_ratio', type=float, default=0.3, help='the ratio of gpu cached edge feature')
parser.add_argument('--cache', action='store_true', help='cache edge features on device')
parser.add_argument('--pure_gpu', action='store_true', help='put all edge features on device, disable cache')
parser.add_argument('--print_cache_hit_rate', action='store_true', help='print cache hit rate each epoch. Note: this will slowdown performance.')

parser.add_argument('--tb_log_prefix', type=str, default='', help='prefix for the tb logging data.')

parser.add_argument('--profile', action='store_true', help='whether to profile.')
parser.add_argument('--profile_prefix', default='log_profile/', help='prefix for the profiling data.')

parser.add_argument('--edge_feature_access_fn', default='', help='prefix to store the edge feature access pattern per epoch')

parser.add_argument('--override_epoch', type=int, default=0, help='override epoch in config.')
parser.add_argument('--override_valepoch', type=int, default=-1, help='override eval epoch in config.') # todel
parser.add_argument('--override_lr', type=float, default=-1, help='override learning rate in config.')
parser.add_argument('--override_order', type=str, default='', help='override training order in config.')
parser.add_argument('--override_scope', type=int, default=0, help='override sampling scope in config.')
parser.add_argument('--override_neighbor', type=int, default=0, help='override sampling neighbors in config.')

parser.add_argument('--gradient_option', type=str, default='none', choices=["none", "unbiased"])

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
from modules.memory import GRUMemory

if not args.no_time:
    globals.timer.set_enable()

torch.autograd.set_detect_anomaly(True)

@torch.no_grad()
def eval(model, dataloader):
    model.eval()
    aps = list()
    mrrs = list()
    while not dataloader.epoch_end:
        blocks, msgs = dataloader.get_blocks()
        pred_pos, pred_neg = model(blocks, msgs)
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred))
        mrrs.append(torch.reciprocal(
            torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(blocks[-1].num_neg_dst, -1), dim=0) + 1).type(
            torch.float))
    dataloader.reset()
    # import pdb; pdb.set_trace()
    ap = float(torch.tensor(aps).mean())
    mrr = float(torch.cat(mrrs).mean())
    return ap, mrr, aps, mrrs


config = yaml.safe_load(open(args.config, 'r'))

if args.override_epoch > 0:
    config['train'][0]['epoch'] = args.override_epoch
if args.override_valepoch > -1:  # todel
    config['eval'][0]['val_epoch'] = args.override_valepoch  # todel
if args.override_lr > 0:
    config['train'][0]['lr'] = args.override_lr
if args.override_order != '':
    config['train'][0]['order'] = args.override_order
if args.override_scope > 0:
    fanout = config['scope'][0]['neighbor']
    for i in range(len(fanout)):
        fanout[i] = args.override_scope

"""Logger"""
path_saver = 'models/{}_{}_{}.pkl'.format(args.data, args.config.split('/')[1].split('.')[0],
                                          time.strftime('%m-%d %H:%M:%S'))
path = os.path.dirname(path_saver)
os.makedirs(path, exist_ok=True)

if args.tb_log_prefix != '':
    tb_path_saver = 'log_tb/{}{}_{}_{}'.format(args.tb_log_prefix, args.data,
                                               args.config.split('/')[1].split('.')[0],
                                               time.strftime('%m-%d %H:%M:%S'))
    os.makedirs('log_tb/', exist_ok=True)
    writer = SummaryWriter(log_dir=tb_path_saver)

if args.edge_feature_access_fn != '':
    os.makedirs('../log_cache/', exist_ok=True)
    efeat_access_path_saver = 'log_cache/{}'.format(args.edge_feature_access_fn)

profile_path_saver = '{}{}_{}_{}'.format(args.profile_prefix, args.data,
                                         args.config.split('/')[1].split('.')[0],
                                         time.strftime('%m-%d %H:%M:%S'))
path = os.path.dirname(profile_path_saver)
os.makedirs(path, exist_ok=True)

"""Data"""
g, edges, nfeat, efeat = load_data(args.data, args.root_path)
if efeat is not None and efeat.dtype == torch.bool:
    efeat = efeat.to(torch.int8)
if nfeat is not None and nfeat.dtype == torch.bool:
    nfeat = nfeat.to(torch.int8)
dim_edge_feat = efeat.shape[-1] if efeat is not None else 0
dim_node_feat = nfeat.shape[-1] if nfeat is not None else 0
n_node = g[0].shape[0]

"""Model"""
device = 'cuda'
params = []
model = TGNN(config, device, n_node, dim_node_feat, dim_edge_feat)
df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

"""Loader"""
train_loader = DataLoader(g, config['scope'][0]['neighbor'],
                          edges['train_src'], edges['train_dst'], edges['train_time'], edges['neg_dst'],
                          nfeat, efeat, train_edge_end, val_edge_end, config['train'][0]['batch_size'],
                          device=device, mode='train',
                          type_sample=config['scope'][0]['strategy'],
                          order=config['train'][0]['order'],
                          memory=config['gnn'][0]['memory_type'],
                          # edge_deg=edges['train_deg'],
                          cached_ratio=args.cached_ratio, enable_cache=args.cache, pure_gpu=args.pure_gpu)
val_loader = DataLoader(g, config['scope'][0]['neighbor'],
                        edges['val_src'], edges['val_dst'], edges['val_time'], edges['neg_dst'],
                        nfeat, efeat, train_edge_end, val_edge_end, config['eval'][0]['batch_size'],
                        device=device, mode='val',
                        eval_neg_dst_nid=edges['val_neg_dst'],
                        type_sample=config['scope'][0]['strategy'],
                        memory=config['gnn'][0]['memory_type'],
                        enable_cache=False, pure_gpu=args.pure_gpu)
test_loader = DataLoader(g, config['scope'][0]['neighbor'],
                         edges['test_src'], edges['test_dst'], edges['test_time'], edges['neg_dst'],
                         nfeat, efeat, train_edge_end, val_edge_end, config['eval'][0]['batch_size'],
                         device=device, mode='test',
                         eval_neg_dst_nid=edges['test_neg_dst'],
                         type_sample=config['scope'][0]['strategy'],
                         memory=config['gnn'][0]['memory_type'],
                         enable_cache=False, pure_gpu=args.pure_gpu)

mrrs = []
model_dir = 'models/WIKI_TGAT_re_gru_01-08 07:16:17'
for root, dirs, files in os.walk(model_dir):
    sorted_files = []
    for file in files:
        try:
            file = int(file.split('.')[0])
        except:
            continue
        sorted_files.append(file)
    sorted_files.sort()
    print(sorted_files)
    for file in sorted_files:
        param_dict = torch.load(osp.join(model_dir, str(file) + '.pkl'))
        model.load_state_dict(param_dict['model'])
        if isinstance(model.memory, GRUMemory):
            model.memory.__init_memory__(True)
            _ = eval(model, train_loader)
            _ = eval(model, val_loader)
        ap, mrr, _, _ = eval(model, test_loader)
        mrrs.append(mrr)
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, mrr))
    param_dict = torch.load(osp.join(model_dir, 'best.pkl'))
    model.load_state_dict(param_dict['model'])
    if isinstance(model.memory, GRUMemory):
        model.memory.__init_memory__(True)
        _ = eval(model, train_loader)
        _ = eval(model, val_loader)
    ap, mrr, _, _ = eval(model, test_loader)
    mrrs.append(mrr)
    print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, mrr))

print(mrrs)