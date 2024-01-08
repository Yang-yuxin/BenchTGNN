import os
import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
import pandas as pd

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
    ap = float(torch.tensor(aps).mean())
    mrr = float(torch.cat(mrrs).mean())
    return ap, mrr


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
# path_saver = 'models/{}_{}_{}.pkl'.format(args.data, args.config.split('/')[1].split('.')[0],
#                                           time.strftime('%m-%d %H:%M:%S'))

time_str = time.strftime('%m-%d %H:%M:%S')
path_saver_prefix = 'models/{}_{}_{}/'.format(args.data, args.config.split('/')[-1].split('.')[0], time_str)


path = os.path.dirname(path_saver_prefix)
os.makedirs(path, exist_ok=True)


if args.tb_log_prefix != '':
    tb_path_saver = 'log_tb/{}{}_{}_{}'.format(args.tb_log_prefix, args.data,
                                               args.config.split('/')[-1].split('.')[0],
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
params.append({
    'params': model.parameters(),
    'lr': config['train'][0]['lr']
})
optimizer = torch.optim.Adam(params)
if model.memory is not None:
    model.memory.to_device()
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
train_edge_end = len(edges['train_src'])
val_edge_end = len(edges['train_src']) + len(edges['val_src'])

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

if args.edge_feature_access_fn != '':
    efeat_access_freq = list()
# import pdb; pdb.set_trace()
best_ap = 0
best_e = 0
use_memory = (config['gnn'][0]['memory_type'] != 'none')
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=50, warmup=50, active=20, skip_first=100, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path_saver)
) if args.profile else nullcontext() as profiler:
    for e in range(config['train'][0]['epoch']):
        print('Epoch {:d}:'.format(e))
        if config['gnn'][0]['memory_type'] == 'gru':
            model.memory.__init_memory__(True)
            # import pdb; pdb.set_trace()

        if args.edge_feature_access_fn != '':
            efeat_access_freq.append(torch.zeros(efeat.shape[0], dtype=torch.int32, device='cpu'))

        # training
        model.train()
        total_loss = 0

        if args.no_time: t_s = time.time()
        if args.print_cache_hit_rate:
            hit_count = 0
            miss_count = 0
        aps = []
        mrrs = []
        while not train_loader.epoch_end:
            blocks, messages = train_loader.get_blocks(log_cache_hit_miss=args.print_cache_hit_rate)
            if args.print_cache_hit_rate:
                for block in blocks:
                    hit_count += block.cache_hit_count
                    miss_count += block.cache_miss_count

            if args.edge_feature_access_fn != '':
                with torch.no_grad():
                    for b in blocks:
                        access = b.neighbor_eid.flatten().detach().cpu()
                        value = torch.ones_like(access, dtype=torch.int32)
                        efeat_access_freq[-1].put_(access, value, accumulate=True)

            globals.timer.start_train()
            optimizer.zero_grad()
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #     with record_function("model_prop_forward"):        
            #         pred_pos, pred_neg = model(blocks, messages)
            #     loss_pos = criterion(pred_pos, torch.ones_like(pred_pos))
            #     loss = loss_pos.mean()
            #     loss += criterion(pred_neg, torch.zeros_like(pred_neg)).mean()
            #     with record_function("loss_backward"):
            #         loss.backward()
            #     with record_function("optimizer_step"):
            #         optimizer.step()
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))    
            pred_pos, pred_neg = model(blocks, messages)
            loss_pos = criterion(pred_pos, torch.ones_like(pred_pos))
            
            loss = loss_pos.mean()
            loss += criterion(pred_neg, torch.zeros_like(pred_neg)).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                aps.append(average_precision_score(y_true, y_pred))
                mrrs.append(torch.reciprocal(
                torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(blocks[-1].num_neg_dst, -1), dim=0) + 1).type(
                torch.float))
            if isinstance(model.memory, GRUMemory):
                model.memory.detach_memory()
            globals.timer.end_train()
            

            if args.profile:
                profiler.step()
                
            with torch.no_grad():
                total_loss += float(loss) * config['train'][0]['batch_size']
                if config['train'][0]['order'].startswith('gradient'):
                    weights = torch.special.expit(pred_pos)
                    train_loader.update_gradient(blocks[-1].gradient_idx, weights)
        train_ap = float(torch.tensor(aps).mean())
        train_mrr = float(torch.cat(mrrs).mean())
        # print(ap, mrr)
        if args.print_cache_hit_rate:
            oracle_cache_hit_rate = train_loader.reset(log_cache_hit_miss=True)
        else:
            train_loader.reset()
        if args.no_time:
            torch.cuda.synchronize()
            t_rough = time.time() - t_s

        ap = mrr = 0.
        time_val = 0.
        if e >= config['eval'][0]['val_epoch']:
            globals.timer.start_val()
            ap, mrr = eval(model, val_loader)
            globals.timer.end_val()
            if ap > best_ap:
                best_e = e
                best_ap = ap
                param_dict = {'model': model.state_dict()}
                # if config['gnn'][0]['memory_type'] == 'gru':
                #     param_dict['memory'] = model.memory.memory
                torch.save(param_dict, path_saver_prefix + 'best.pkl')
            if e % config['train'][0]['save_every'] == 0:
                torch.save(param_dict, path_saver_prefix + '{}.pkl'.format(e))

        if args.tb_log_prefix != '':
            writer.add_scalar(tag='Loss/Train', scalar_value=total_loss, global_step=e)
            writer.add_scalar(tag='AP/Val', scalar_value=ap, global_step=e)
            writer.add_scalar(tag='MRR/Val', scalar_value=mrr, global_step=e)
            writer.add_scalar(tag='MRR/Train', scalar_value=train_mrr, global_step=e)
            writer.add_scalar(tag='AP/Train', scalar_value=train_ap, global_step=e)
        print('\ttrain loss:{:.4f} train ap:{:4f} train mrr:{:4f} val ap:{:4f}  val mrr:{:4f}'.format(total_loss, train_ap, train_mrr, ap, mrr))
        if args.no_time:
            print('\trough train time: {:.2f}s'.format(t_rough))
        if args.print_cache_hit_rate:
            print('\tcache hit rate: {:.2f}%  oracle hit rate: {:.2f}%'.format(hit_count / (hit_count + miss_count) * 100, oracle_cache_hit_rate * 100))
        else:
            globals.timer.print(prefix='\t')
            globals.timer.reset()

if args.tb_log_prefix != '':
    writer.close()

if args.edge_feature_access_fn != '':
    torch.save(efeat_access_freq, efeat_access_path_saver)

print('Loading model at epoch {} with val AP {:4f}...'.format(best_e, best_ap))
param_dict = torch.load(path_saver_prefix + 'best.pkl')
model.load_state_dict(param_dict['model'])
if isinstance(model.memory, GRUMemory):
    model.memory.__init_memory__(True)
    _ = eval(model, train_loader)
    _ = eval(model, val_loader)
ap, mrr = eval(model, test_loader)
print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, mrr))
