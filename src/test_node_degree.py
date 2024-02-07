import os
import re
from datetime import datetime
import os.path as osp
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import time
import globals
import yaml
import pickle
from utils import *
from model import TGNN
from dataloader import DataLoader
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
from modules.memory import GRUMemory


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--root_path', type=str, default='DATA', help='dataset root path')
parser.add_argument('--model_path', type=str, default='models', help='trained models root path')
parser.add_argument('--config', type=str, default='config', help='path to config file')
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
bin_size = 10

if not args.no_time:
    globals.timer.set_enable()

torch.autograd.set_detect_anomaly(True)

def parse_datetime(dir_name):
    # Extract the date and time part from the directory name
    match = re.search(r'(\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})', dir_name)
    if match:
        date_str, time_str = match.groups()
        # Parse the datetime (assuming the year is 2024)
        return datetime.strptime(f'2024-{date_str} {time_str}', '%Y-%m-%d %H:%M:%S')
    else:
        return None

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
    ap = float(torch.tensor(aps).mean())
    mrr = float(torch.cat(mrrs).mean())
    return ap, mrr, aps, mrrs

@torch.no_grad()
def eval_degree(model, dataloader, n_neg_dst):
    model.eval()
    aps = list()
    mrrs = list()
    real_neighs = list()
    while not dataloader.epoch_end:
        blocks, msgs = dataloader.get_blocks()
        # import pdb; pdb.set_trace()
        pred_pos, pred_neg = model(blocks, msgs)
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred))
        mrrs.append(torch.reciprocal(
            torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(blocks[-1].num_neg_dst, -1), dim=0) + 1).type(
            torch.float))
        real_neigh_count = torch.sum(blocks[0].neighbor_nid == dataloader.dummy_nid, 1).reshape(-1, (n_neg_dst+2))
        real_neigh_count = torch.sum(real_neigh_count, 1)
        real_neighs.append(real_neigh_count)
    dataloader.reset()
    ap = float(torch.tensor(aps).mean())
    mrr = float(torch.cat(mrrs).mean())
    return ap, mrr, (aps, mrrs), real_neighs

scans = ['5', '10', '20', '50', '100']
# datasets = ['WIKI', 'REDDIT', 'Flights', 'LASTFM', 'mooc', 'uci', 'CollegeMsg']
datasets = ['Flights', 'LASTFM', 'mooc']
aggrs = ['GraphMixer', 'TGAT']
samplings = ['re', 'uni']
memorys = ['gru', 'embed']

model_dir = args.model_path
config_dir = args.config
device = 'cuda'

all_data = {}
for dataset in datasets:
    all_data[dataset] = {}
    for scan in scans:
        all_data[dataset][scan] = {}
        for aggr in aggrs:
            all_data[dataset][scan][aggr] = {}
            for sampling in samplings:
                all_data[dataset][scan][aggr][sampling] = {}
                for memory in memorys:
                    all_data[dataset][scan][aggr][sampling][memory] = []

for root, dirs, files in os.walk(model_dir):
    try:
        timegroup = parse_datetime(root)
        try:
            if not(timegroup.month == 1 and timegroup.day >= 25):
                continue
        except AttributeError:
            continue
        for file in files:
            path = os.path.join(root, file)
            if 'best' not in file:
                continue
            dataset, aggr, sampling, memory = root.split('/')[1].split('_')[0:4]
            # print(dataset, aggr, sampling, memory)
            if not dataset in datasets or not memory in memorys:
                continue
            param_dict = torch.load(os.path.join(path))
            # for key in param_dict['model'].keys():
            #     print(param_dict['model'][key].shape)
            all_data[dataset][scan][aggr][sampling][memory].append((root, path))
    except IndexError:
        pass

# print(all_data)

for dataset in datasets:
    for aggr in aggrs:
        for sampling in samplings:
            for memory in memorys:
                for scan in scans:
                    all_data[dataset][scan][aggr][sampling][memory] = \
                    sorted(all_data[dataset][scan][aggr][sampling][memory], key=lambda d: parse_datetime(d[0]))
for dataset in datasets:
    for aggr in aggrs:
        for sampling in samplings:
            for memory in memorys:
                for scan in scans:
                    if len(all_data[dataset][scan][aggr][sampling][memory]) > 25:
                        all_data[dataset][scan][aggr][sampling][memory] = all_data[dataset][scan][aggr][sampling][memory][-25:]
                        print(dataset, aggr, sampling, memory)
                    elif len(all_data[dataset][scan][aggr][sampling][memory]) < 25:
                        all_data[dataset][scan][aggr][sampling][memory] = []
                    else:
                        print(dataset, aggr, sampling, memory)
for dataset in datasets:
    for aggr in aggrs:
        for sampling in samplings:
            for memory in memorys:
                for i, scan in enumerate(scans):
                    all_data[dataset][scan][aggr][sampling][memory] = \
                    all_data[dataset][scans[-1]][aggr][sampling][memory][i*5:(i+1)*5]

for root, dirs, files in os.walk(config_dir):
    try:
        if not('scan' in root):
            continue
        scan = root.split('_')[-1]
        if scan == '200':
            continue
        for file in files:
            path = os.path.join(root, file)
            config = yaml.safe_load(open(path, 'r'))
    except IndexError:
        pass
for dataset in datasets:
    """Data"""
    g, edges, nfeat, efeat = load_data(dataset, args.root_path)
    if efeat is not None and efeat.dtype == torch.bool:
        efeat = efeat.to(torch.int8)
    if nfeat is not None and nfeat.dtype == torch.bool:
        nfeat = nfeat.to(torch.int8)
    dim_edge_feat = efeat.shape[-1] if efeat is not None else 0
    dim_node_feat = nfeat.shape[-1] if nfeat is not None else 0
    n_node = g[0].shape[0]
    train_edge_end = len(edges['train_src'])
    val_edge_end = len(edges['train_src']) + len(edges['val_src'])
    for aggr in aggrs:
        for sampling in samplings:
            for memory in memorys:
                fig, axes = plt.subplots(1, len(scans), figsize=(20,4))  # 1 row, 5 columns
                for i_s, scan in enumerate(scans):
                    tmp_file = f'degree_analysis/{dataset}_scan{scan}_{aggr}_{sampling}_{memory}.pkl'
                    if os.path.isfile(tmp_file):
                        with open(tmp_file, 'rb') as file:
                            histogram = pickle.load(file)
                    else:
                        path = osp.join(config_dir, f'scan_{scan}', f'{aggr}_{sampling}_{memory}.yml')
                        config = yaml.safe_load(open(path, 'r'))
                        """Model"""
                        model = TGNN(config, device, n_node, dim_node_feat, dim_edge_feat)
                        """Loader"""
                        train_loader = DataLoader(g, config['scope'][0]['neighbor'],
                                                edges['train_src'], edges['train_dst'], edges['train_time'], edges['neg_dst'],
                                                nfeat, efeat, train_edge_end, val_edge_end, config['train'][0]['batch_size'],
                                                device=device, mode='train',
                                                type_sample=config['scope'][0]['strategy'],
                                                order=config['train'][0]['order'],
                                                memory=config['gnn'][0]['memory_type'],
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
                        if not len(all_data[dataset][scan][aggr][sampling][memory]):
                            continue
                        all_info = {'mrr': []}
                        histogram = {}
                        for rep in range(1):
                            param_dict = torch.load(osp.join(all_data[dataset][scan][aggr][sampling][memory][rep][1]))
                            model.load_state_dict(param_dict['model'])
                            if isinstance(model.memory, GRUMemory):
                                model.memory.__init_memory__(True)
                                _ = eval(model, train_loader)
                                _ = eval(model, val_loader)
                            ap, mrr, (aps, mrrs), real_neigh = eval_degree(model, test_loader, args.eval_neg_samples)
                            for i_mrr, i_real_neigh in zip(mrrs, real_neigh):
                                for i in range(len(i_mrr)):
                                    b = i_real_neigh[i].item() // bin_size
                                    if b not in histogram.keys():
                                        histogram[b] = []
                                    histogram[b].append(i_mrr[i].item())
                            # import pdb; pdb.set_trace()
                            # all_info['mrr'].append(mrr)
                        # import pdb; pdb.set_trace()
                        for k in histogram.keys():
                            histogram[k] = np.mean(histogram[k])
                        with open(tmp_file, 'wb') as file:
                            pickle.dump(histogram, file)
                    bins = [int(_) for _ in histogram.keys()]
                    bins = sorted(bins) * bin_size
                    values = [histogram[k] for k in bins]
                    axes[i_s].plot(bins, values, label=f'scan{scan}_{aggr}_{sampling}_{memory}')
                    axes[i_s].set_title(f'scan {scan}')
                    axes[i_s].legend()
                plt.savefig(f'{dataset}_node_degree_plot.png')
                plt.close()