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
        num_neighbors = config['scope'][0]['neighbor']

        # Memory
        memory_type = gnn_config['memory_type']
        use_embedding_in_message = gnn_config['memory_update_use_embed']
        if memory_type == 'gru':
            dim_memory = gnn_config['dim_memory']
            dim_time = gnn_config['dim_time']
            dim_msg = dim_memory * 2 + dim_edge_feat
            self.memory = GRUMemory(device, n_nodes, dim_memory, dim_node_feat, dim_msg, dim_time, 
                                    gnn_config['msg_reducer'], time_encoder_type, use_embedding_in_message)
            dim_memory = gnn_config['dim_memory']
            if dim_node_feat > 0:
                self.memory_mapper = nn.Linear(dim_node_feat, dim_memory)
        elif memory_type == 'embedding':
            dim_memory = gnn_config['dim_memory']
            self.memory = EmbeddingTableMemory(device, n_nodes, dim_memory)
            dim_memory = gnn_config['dim_memory']
            if dim_node_feat > 0:
                self.memory_mapper = nn.Linear(dim_node_feat, dim_memory)
        elif memory_type == 'none':
            self.memory = None
            dim_memory = 0
        else:
            raise NotImplementedError
        
        dim_out = gnn_config['dim_out']
        # self.mlp_q0 = nn.Sequential(
        #     nn.Linear(dim_node_feat + dim_time + dim_memory, dim_out),
        #     nn.ReLU(),
        #     )
        # self.mlp_q = nn.Sequential(
        #     nn.Linear(dim_out + dim_time + dim_memory, dim_out),
        #     nn.ReLU(),
        #     )
        # self.mlp_k0 = nn.Sequential(
        #     nn.Linear(dim_node_feat + dim_edge_feat + dim_time + dim_memory, dim_out),
        #     nn.ReLU(),
        #     )
        # self.mlp_k = nn.Sequential(
        #     nn.Linear(dim_out + dim_edge_feat + dim_time + dim_memory, dim_out),
        #     nn.ReLU(),
        #     )
        self.layers = torch.nn.ModuleList()
        if gnn_config['arch'] == 'transformer':
            self.layers.append(TransformerAggregator(dim_node_feat, dim_time, dim_edge_feat, dim_memory, dim_out, gnn_config['att_head'], 
                                                    train_config['dropout'],))
        elif gnn_config['arch'] == 'mixer':
            if self.memory:
                self.layers.append(MixerAggregator(num_neighbors[-1], dim_memory, dim_edge_feat,
                                               gnn_config['dim_time'], dim_memory, 
                                               dim_out, train_config['dropout'], ))
            else:
                self.layers.append(MixerAggregator(num_neighbors[0], dim_node_feat, dim_edge_feat,
                                               gnn_config['dim_time'], dim_memory, 
                                               dim_out, train_config['dropout'], ))
        else:
            raise NotImplementedError
        for i in range(1, gnn_config['layer']):
            if gnn_config['arch'] == 'transformer':
                self.layers.append(TransformerAggregator(dim_out, dim_time, dim_edge_feat, dim_memory, dim_out, gnn_config['att_head'], 
                                                    train_config['dropout'],))
            elif gnn_config['arch'] == 'mixer':
                self.layers.append(MixerAggregator(num_neighbors[-i-1], dim_out,
                                                   dim_edge_feat, gnn_config['dim_time'], dim_memory,
                                                   dim_out, train_config['dropout']))
        self.edge_predictor = EdgePredictor(dim_out)
        self.to(device)
    
    def aggregate_messages_and_update_memory(self, blocks, pos_edge_feats):
        if isinstance(self.memory, GRUMemory):
            updated_memory, last_update = self.memory.get_updated_memory(torch.arange(self.n_nodes).to(self.device))
        elif isinstance(self.memory, EmbeddingTableMemory):
            updated_memory, last_update = self.memory.get_memory(torch.arange(self.n_nodes).to(self.device), self.memory.memory), \
            self.memory.get_last_update(torch.arange(self.n_nodes).to(self.device))
        else:
            updated_memory = None
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
            if self.memory is None:
                root_node_feature = block.root_node_feature
            else:
                root_node_memory = self.memory.get_memory(block.root_nid, updated_memory) if self.memory is not None else torch.tensor([]).to(self.device)
                neighbor_node_memory = self.memory.get_memory(block.neighbor_nid, updated_memory).reshape(-1, self.memory.dim_memory) \
                if self.memory is not None else torch.tensor([]).to(self.device)
                if h_in is None:
                    if neighbor_node_feature.shape[1] > 0:
                        neighbor_node_feature = neighbor_node_memory + self.memory_mapper(neighbor_node_feature)
                        root_node_feature = root_node_memory + self.memory_mapper(block.root_node_feature)
                    else:
                        neighbor_node_feature = neighbor_node_memory
                        root_node_feature = root_node_memory
                else:
                    root_node_feature = block.root_node_feature
                    
            zero_time_feat = self.time_encoder(torch.zeros(block.n, dtype=torch.float32, device=self.device))
            edge_time_feat = self.time_encoder((block.root_ts.unsqueeze(-1) - block.neighbor_ts).flatten()) \
            if self.time_encoder is not None else torch.tensor([])
            h_in = layer.forward(root_node_feature,
                                zero_time_feat,
                                neighbor_node_feature,
                                neighbor_edge_feature,
                                edge_time_feat)
        if isinstance(self.memory, GRUMemory):
            block = blocks[-1]
            src_nids = block.root_nid[:block.pos_dst_size]
            dst_nids = block.root_nid[block.pos_dst_size:block.pos_dst_size*2]
            positives = torch.cat([src_nids, dst_nids])
            pos_edge_times = block.root_ts[:2 * block.pos_dst_size]
            self.memory.update_memory(positives, root_node_memory[:2 * block.pos_dst_size])
            if not torch.allclose(updated_memory[positives], self.memory.get_memory(positives, self.memory.memory), atol=1e-5): 
                print("Something wrong in how the memory was updated")
                import pdb; pdb.set_trace()
            # self.memory.clear_mailbox(positives)
            self.memory.store_raw_messages(src_nids, None, dst_nids, None, pos_edge_times, pos_edge_feats)
            # self.memory.store_raw_messages(dst_nids, None, src_nids, None, pos_edge_times, pos_edge_feats)
            # import pdb; pdb.set_trace()

        return h_in
    

    def forward(self, blocks, messages):
        h_in = self.aggregate_messages_and_update_memory(blocks, messages)
        return self.edge_predictor(h_in, blocks[-1].num_neg_dst)

