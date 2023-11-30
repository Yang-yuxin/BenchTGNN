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

        gnn_config = config['gnn'][0]
        train_config = config['train'][0]
        self.device = device
        self.n_nodes = n_nodes

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
        num_neighbors = config['scope'][0]['neighbor'][0]

        # Memory
        memory_type = gnn_config['memory_type']
        use_embedding_in_message = gnn_config['memory_update_use_embed']
        if memory_type == 'gru':
            dim_memory = gnn_config['dim_memory']
            dim_time = gnn_config['dim_time']
            dim_msg = dim_memory * 2 + dim_edge_feat + dim_time
            self.memory = GRUMemory(device, n_nodes, dim_memory, dim_node_feat, dim_msg, dim_time, 
                                    gnn_config['msg_reducer'], time_encoder_type, use_embedding_in_message)
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
                                                     gnn_config['dim_out'], train_config['dropout'],))
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
        if isinstance(self.memory, GRUMemory):
            block = blocks[-1]
            updated_memory, last_update = self.memory.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
        else:
            # TODO: embedding
            pass
        h_in = None
        # 
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
            # import pdb; pdb.set_trace()
            root_node_memory = self.memory.get_memory(block.root_nid, updated_memory) if self.memory is not None else torch.tensor([]).to(self.device)
            neighbor_node_memory = self.memory.get_memory(block.neighbor_nid, updated_memory).reshape(-1, self.memory.dim_memory) \
            if self.memory is not None else torch.tensor([]).to(self.device)
            h_in = layer.forward(block.root_node_feature,
                                 neighbor_node_feature,
                                 neighbor_edge_feature,
                                 zero_time_feat,
                                 edge_time_feat,
                                 root_node_memory,
                                 neighbor_node_memory) # TODO: time difference?
        if isinstance(self.memory, GRUMemory):
            block = blocks[-1]
            src_nids = block.root_nid[:block.pos_dst_size]
            dst_nids = block.root_nid[block.pos_dst_size:block.pos_dst_size*2]
            positives = torch.cat([src_nids, dst_nids])
            pos_edge_times = block.root_ts[:block.pos_dst_size]
            self.memory.update_memory(positives, self.memory.messages)
            assert torch.allclose(updated_memory[positives], self.memory.get_memory(positives, self.memory.memory), atol=1e-5), \
            "Something wrong in how the memory was updated"
            self.memory.clear_messages(positives)
            self.memory.get_raw_messages(src_nids, None, dst_nids, None, pos_edge_times, pos_edge_feats)
            self.memory.get_raw_messages(dst_nids, None, src_nids, None, pos_edge_times, pos_edge_feats)

        return h_in
    

    def forward(self, blocks, messages):
        h_in = self.aggregate_messages_and_update_memory(blocks, messages)
        return self.edge_predictor(h_in, blocks[-1].num_neg_dst)

