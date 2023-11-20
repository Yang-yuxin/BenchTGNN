from typing import List

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from layers import *
from utils import DenseTemporalBlock, Categorical
from modules.time_encoder import LearnableTimeEncoder, FixedFrequencyEncoder, FixedTimeEncoder
from modules.aggregator import TransformerAggregator, MixerAggregator
from modules.memory import GRUMemory, EmbeddingTableMemory
from copy import deepcopy, copy

class TGNN(torch.nn.Module):

    def __init__(self, config, device, n_nodes, dim_node_feat, dim_edge_feat):
        super(TGNN, self).__init__()

        sampler_config = config['sample'][0]
        gnn_config = config['gnn'][0]
        train_config = config['train'][0]
        self.device = device

        # Time encoding
        time_encoder_type = gnn_config['time_enc']
        if time_encoder_type == 'learnable':
            dim_time = gnn_config['dim_time']
            self.time_encoder = LearnableTimeEncoder(dim_time)
        elif time_encoder_type == 'fixed':
            dim_time = gnn_config['dim_time']
            self.time_encoder = FixedTimeEncoder(dim_time)
        elif time_encoder_type == 'none':
            self.time_encoder = None
        else:
            raise NotImplementedError
        if sampler_config['type'] == 'none':
            num_neighbors = config['scope'][0]['neighbor'][0]
        else:
            num_neighbors = sampler_config['neighbor']

        # Memory
        memory_type = gnn_config['memory_type']
        if memory_type == 'gru':
            dim_memory = gnn_config['dim_memory']
            dim_time = gnn_config['dim_time']
            dim_msg = dim_memory * 2 + dim_edge_feat + dim_time
            self.memory = GRUMemory(device, n_nodes, dim_memory, dim_msg, dim_time, 
                                    gnn_config['msg_reducer'], time_encoder_type)
            dim_memory = gnn_config['dim_memory']
        elif memory_type == 'embedding':
            dim_memory = gnn_config['dim_memory']
            self.memory = EmbeddingTableMemory(device, n_nodes, dim_memory)
            dim_memory = gnn_config['dim_memory']
        elif memory_type == 'none':
            self.memory = None
            dim_memory = 0
        else:
            raise NotImplementedError
            

        self.layers = torch.nn.ModuleList()
        if gnn_config['arch'] == 'transformer':
            self.layers.append(TransformerAggregator(dim_node_feat, dim_edge_feat, 
                                                     gnn_config['dim_time'], dim_memory, gnn_config['att_head'], 
                                                     gnn_config['dim_out'], train_config['dropout'], 
                                                     att_clamp=sampler_config.get('att_clamp', 10),))
        elif gnn_config['arch'] == 'mixer':
            self.layers.append(MixerAggregator(num_neighbors, dim_node_feat, dim_edge_feat,
                                               gnn_config['dim_time'], dim_memory, 
                                               gnn_config['dim_out'], train_config['dropout'], ))
        else:
            raise NotImplementedError
        for i in range(1, gnn_config['layer']):
            if gnn_config['arch'] == 'transformer':
                self.layers.append(TransformerAggregator(gnn_config['dim_out'], dim_edge_feat,
                                                         gnn_config['dim_time'], dim_memory, gnn_config['att_head'], 
                                                         gnn_config['dim_out'], train_config['dropout']))
            elif gnn_config['arch'] == 'mixer':
                self.layers.append(MixerAggregator(num_neighbors, gnn_config['dim_out'],
                                                   dim_edge_feat, gnn_config['dim_time'], dim_memory,
                                                   gnn_config['dim_out'], train_config['dropout']))
        
        self.edge_predictor = EdgePredictor(gnn_config['dim_out'])
        self.to(device)
    
    def aggregate_messages_and_update_memory(self, blocks, pos_edge_feats):
        h_in = None
        for block, layer in zip(blocks, self.layers):
            if h_in is not None:
                block.slice_hidden_node_features(h_in)
            neighbor_node_feature = block.neighbor_node_feature.view(
                block.neighbor_node_feature.shape[0] * block.neighbor_node_feature.shape[1],
                block.neighbor_node_feature.shape[2]
            )
            neighbor_edge_feature = block.neighbor_edge_feature.view(
                block.neighbor_edge_feature.shape[0] * block.neighbor_edge_feature.shape[1],
                block.neighbor_edge_feature.shape[2]
            )
            zero_time_feat = self.time_encoder(torch.zeros(block.n, dtype=torch.float32, device=self.device))
            edge_time_feat = self.time_encoder((block.root_ts.unsqueeze(-1) - block.neighbor_ts).flatten()) \
            if self.time_encoder is not None else torch.tensor([])
            root_node_memory = self.memory.get_memory(block.root_nid).clone().detach() if self.memory is not None else torch.tensor([]).to(self.device)
            neighbor_node_memory = self.memory.get_memory(block.neighbor_nid).reshape(-1, self.memory.dim_memory).clone().detach() \
            if self.memory is not None else torch.tensor([]).clone().to(self.device)
            h_in = layer.forward(block.root_node_feature,
                                 neighbor_node_feature,
                                 neighbor_edge_feature,
                                 zero_time_feat,
                                 edge_time_feat,
                                 root_node_memory,
                                 neighbor_node_memory)
            if isinstance(self.memory, GRUMemory):
                block = blocks[-1]
                pos_src_nids = block.root_nid[:block.pos_dst_size]
                pos_dst_nids = block.root_nid[block.pos_dst_size:block.pos_dst_size * 2]
                assert (pos_edge_feats.shape[0] == pos_src_nids.shape[0] == pos_dst_nids.shape[0])
                pos_node_pairs = torch.concatenate([pos_src_nids, pos_dst_nids])
                pos_edge_times = block.root_ts[:block.pos_dst_size]
                self.memory.update_memory(pos_src_nids, None,
                                        pos_dst_nids, None,
                                        pos_edge_times,
                                        pos_edge_feats)
        #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #         with record_function("slice_feature"):  
        #             if h_in is not None:
        #                 block.slice_hidden_node_features(h_in)
        #             neighbor_node_feature = block.neighbor_node_feature.view(
        #                 block.neighbor_node_feature.shape[0] * block.neighbor_node_feature.shape[1],
        #                 block.neighbor_node_feature.shape[2]
        #             )
        #             neighbor_edge_feature = block.neighbor_edge_feature.view(
        #                 block.neighbor_edge_feature.shape[0] * block.neighbor_edge_feature.shape[1],
        #                 block.neighbor_edge_feature.shape[2]
        #             )
        #         with record_function("encode_time"):  
        #             zero_time_feat = self.time_encoder(torch.zeros(block.n, dtype=torch.float32, device=self.device))
        #             edge_time_feat = self.time_encoder((block.root_ts.unsqueeze(-1) - block.neighbor_ts).flatten()) \
        #             if self.time_encoder is not None else torch.tensor([])
        #         with record_function("encode_memory"):  
        #             root_node_memory = self.memory.get_memory(block.root_nid).clone().detach() if self.memory is not None else torch.tensor([]).to(self.device)
        #             neighbor_node_memory = self.memory.get_memory(block.neighbor_nid).reshape(-1, self.memory.dim_memory).clone().detach() \
        #             if self.memory is not None else torch.tensor([]).clone().to(self.device)
        #         with record_function("forward_pass"):  
        #             h_in = layer.forward(block.root_node_feature,
        #                          neighbor_node_feature,
        #                          neighbor_edge_feature,
        #                          zero_time_feat,
        #                          edge_time_feat,
        #                          root_node_memory,
        #                          neighbor_node_memory)
        #         with record_function("update_memory"):  
        # # time_encod_src = zero_time_feat
        # # time_encod_dst = edge_time_feat[:block.pos_dst_size]
        #             if self.memory is not None:
        #                 pos_src_nids = blocks[-1].root_nid[:block.pos_dst_size]
        #                 pos_dst_nids = blocks[-1].root_nid[block.pos_dst_size:block.pos_dst_size * 2]
        #                 assert (pos_edge_feats.shape[0] == pos_src_nids.shape[0] == pos_dst_nids.shape[0])
        #                 pos_node_pairs = torch.concatenate([pos_src_nids, pos_dst_nids])
        #                 pos_edge_times = blocks[-1].root_ts[:block.pos_dst_size]
        #                 self.memory.update_memory(pos_src_nids, None,
        #                                         pos_dst_nids, None,
        #                                         pos_edge_times,
        #                                         pos_edge_feats)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return h_in
    

    def forward(self, blocks, messages):
        h_in = self.aggregate_messages_and_update_memory(blocks, messages)


        return self.edge_predictor(h_in, blocks[-1].num_neg_dst)

