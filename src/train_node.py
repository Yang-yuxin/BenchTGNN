import argparse
import os
import hashlib

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--root_path', type=str, default='DATA', help='dataset root path')
parser.add_argument('--config', type=str, default='', help='path to config file')
parser.add_argument('--batch_size', type=int, default=600)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model', type=str, default='', help='name of stored model to load')
parser.add_argument('--cached_ratio', type=float, default=0.3, help='the ratio of gpu cached edge feature')
parser.add_argument('--cache', action='store_true', help='cache edge features on device')
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--ind_ratio', type=float, default=0.1)
parser.add_argument('--print_cache_hit_rate', action='store_true', help='print cache hit rate each epoch. Note: this will slowdown performance.')
parser.add_argument('--posneg', default=False, action='store_true', help='for positive negative detection, whether to sample negative nodes')
parser.add_argument('--pure_gpu', action='store_true', help='put all edge features on device, disable cache')

args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.data == 'WIKI' or args.data == 'REDDIT':
    args.posneg = True

import torch
import time
import random
import yaml
import numpy as np
import pandas as pd
from modules import *
from utils import *
from model import TGNN
from tqdm import tqdm
from modules.memory import GRUMemory
from modules.linear import NodeClassificationModel
from modules.sampler import NegLinkSampler
from dataloader import DataLoader
from sklearn.metrics import average_precision_score, f1_score

config = yaml.safe_load(open(args.config, 'r'))
device = 'cuda'

ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
role = ldf['ext_roll'].values
labels = ldf['label'].values.astype(np.int64)

g, edges, nfeat, efeat = load_data(args.data, args.root_path)
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
"""Combined Loader"""
# Combine the source, destination, time, and negative destination arrays from the three loaders
combined_src = torch.cat([edges['train_src'], edges['val_src'], edges['test_src']], dim=0)
combined_dst = torch.cat([edges['train_dst'], edges['val_dst'], edges['test_dst']], dim=0)
combined_time = torch.cat([edges['train_time'], edges['val_time'], edges['test_time']], dim=0)
combined_neg_dst = torch.cat([edges['neg_dst'], edges['neg_dst'], edges['neg_dst']], dim=0)

# Combine the negative destination arrays for validation and test (if needed)
combined_val_test_neg_dst = torch.cat([edges['val_neg_dst'], edges['test_neg_dst']], dim=0)

# Create the combined dataloader
combined_loader = DataLoader(
    g,
    config['scope'][0]['neighbor'],
    combined_src,
    combined_dst,
    combined_time,
    combined_neg_dst,
    nfeat,
    efeat,
    train_edge_end,
    val_edge_end,
    config['train'][0]['batch_size'],  # Use the largest batch size among train, val, and test
    device=device,
    mode='all', 
    ind=args.inductive,
    eval_neg_dst_nid=combined_val_test_neg_dst,
    type_sample=config['scope'][0]['strategy'],
    memory=config['gnn'][0]['memory_type'],
    enable_cache=args.cache,  # Enable or disable cache as needed
    pure_gpu=args.pure_gpu
)


emb_file_name = hashlib.md5(str(torch.load(args.model, map_location=torch.device('cpu'))).encode('utf-8')).hexdigest() + '.pt'
if not os.path.isdir('embs'):
    os.mkdir('embs')
if not os.path.isfile('embs/' + emb_file_name):
    print('Generating temporal embeddings..')

    node_feats, edge_feats = load_feat(args.data)
    g, df = load_graph(args.data)
    train_param, eval_param, sample_param, gnn_param = parse_config(args.config)


    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    n_node = node_feats.shape[0] if node_feats else g[0].shape[0]

    model = TGNN(args, config, device, n_node, gnn_dim_node, gnn_dim_edge)
    model.load_state_dict(torch.load(args.model)['model'])
    if isinstance(model.memory, GRUMemory):
        model.memory.__init_memory__(True)
    creterion = torch.nn.BCEWithLogitsLoss()
    if args.pure_gpu:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()

    processed_edge_id = 0

    def forward_model_to(t):
        global processed_edge_id
        if processed_edge_id >= len(df):
            return
        clk = 0
        while df.time[processed_edge_id] < t:
            clk += 1
            if clk > 1:
                import pdb; pdb.set_trace()
            loader  = combined_loader
            if processed_edge_id < train_edge_end:
                model.train()
                batch_size = train_param['batch_size']
            else:
                model.eval()
                batch_size = eval_param['batch_size']
            blocks, messages = loader.get_blocks(log_cache_hit_miss=args.print_cache_hit_rate, mode='trans')
            with torch.no_grad():
                pred_pos, pred_neg = model(blocks, messages)
            processed_edge_id += batch_size
            if processed_edge_id >= len(df):
                return

    def get_node_emb(root_nodes, ts):
        forward_model_to(ts[-1])
        # if combined_loader.start > 157474*0.7:
        #     import pdb; pdb.set_trace()
        blocks = combined_loader.get_emb(torch.tensor(root_nodes).cuda().long(), 
                                                   torch.tensor(ts).cuda(), 
                                                   log_cache_hit_miss=args.print_cache_hit_rate)
        with torch.no_grad():
            ret = model.aggregate_messages(blocks)
        return ret.detach().cpu()

    emb = list()
    # import pdb; pdb.set_trace()
    for _, rows in tqdm(ldf.groupby(ldf.index // args.batch_size)):
        emb.append(get_node_emb(rows.node.values.astype(np.int32), rows.time.values.astype(np.float32)))
    emb = torch.cat(emb, dim=0)
    torch.save(emb, 'embs/' + emb_file_name)
    print('Saved to embs/' + emb_file_name)
else:
    print('Loading temporal embeddings from embs/' + emb_file_name)
    emb = torch.load('embs/' + emb_file_name)

import pdb; pdb.set_trace()
model = NodeClassificationModel(emb.shape[1], args.dim, labels.max() + 1).cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
labels = torch.from_numpy(labels).type(torch.int32)
role = torch.from_numpy(role).type(torch.int32)
emb = emb

class NodeEmbMinibatch():

    def __init__(self, emb, role, label, batch_size):
        self.role = role
        self.label = label
        self.batch_size = batch_size
        self.train_emb = emb[role == 0]
        self.val_emb = emb[role == 1]
        self.test_emb = emb[role == 2]
        self.train_label = label[role == 0]
        self.val_label = label[role == 1]
        self.test_label = label[role == 2]
        self.mode = 0
        self.s_idx = 0

    def shuffle(self):
        perm = torch.randperm(self.train_emb.shape[0])
        self.train_emb = self.train_emb[perm]
        self.train_label = self.train_label[perm]

    def set_mode(self, mode):
        if mode == 'train':
            self.mode = 0
        elif mode == 'val':
            self.mode = 1
        elif mode == 'test':
            self.mode = 2
        self.s_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == 0:
            emb = self.train_emb
            label = self.train_label
        elif self.mode == 1:
            emb = self.val_emb
            label = self.val_label
        else:
            emb = self.test_emb
            label = self.test_label
        if self.s_idx >= emb.shape[0]:
            raise StopIteration
        else:
            end = min(self.s_idx + self.batch_size, emb.shape[0])
            curr_emb = emb[self.s_idx:end]
            curr_label = label[self.s_idx:end]
            self.s_idx += self.batch_size
            return curr_emb.cuda(), curr_label.cuda()

if args.posneg:
    role = role[labels == 1]
    emb_neg = emb[labels == 0].cuda()
    emb = emb[labels == 1]
    labels = torch.ones(emb.shape[0], dtype=torch.int64).cuda()
    labels_neg = torch.zeros(emb_neg.shape[0], dtype=torch.int64).cuda()
    neg_node_sampler = NegLinkSampler(emb_neg.shape[0])

minibatch = NodeEmbMinibatch(emb, role, labels, args.batch_size)
if not os.path.isdir('models'):
    os.mkdir('models')
save_path = 'models/node_' + args.model.split('/')[-1]
best_e = 0
best_acc = 0
for e in range(args.epoch):
    minibatch.set_mode('train')
    minibatch.shuffle()
    model.train()
    for emb, label in minibatch:
        optimizer.zero_grad()
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        loss = loss_fn(pred, label.long())
        loss.backward()
        optimizer.step()
    minibatch.set_mode('val')
    model.eval()
    accs = list()
    with torch.no_grad():
        for emb, label in minibatch:
            if args.posneg:
                neg_idx = neg_node_sampler.sample(emb.shape[0])
                emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
                label = torch.cat([label, labels_neg[neg_idx]], dim=0)
            pred = model(emb)
            if args.posneg:
                acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
            else:
                acc = f1_score(label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro")
            accs.append(acc)
        acc = float(torch.tensor(accs).mean())
    print('Epoch: {}\tVal acc: {:.4f}'.format(e, acc))
    if acc > best_acc:
        best_e = e
        best_acc = acc
        torch.save(model.state_dict(), save_path)
print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(save_path))
minibatch.set_mode('test')
model.eval()
accs = list()
with torch.no_grad():
    for emb, label in minibatch:
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        if args.posneg:
            acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
        else:
            acc = f1_score(label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro")
        accs.append(acc)
    acc = float(torch.tensor(accs).mean())
print('Testing acc: {:.4f}'.format(acc))