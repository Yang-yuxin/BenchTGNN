from typing import List

import numpy as np
import torch
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
            self.memory = GRUMemory(n_nodes, dim_memory, dim_msg, dim_time, 
                                    gnn_config['msg_reducer'], time_encoder_type)
            dim_memory = gnn_config['dim_memory']
        elif memory_type == 'embedding':
            dim_memory = gnn_config['dim_memory']
            self.memory = EmbeddingTableMemory(n_nodes, dim_memory)
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
                                                     att_clamp=sampler_config.get('att_clamp', 10),
                                                     save_h_neigh_grad=sampler_config['type'] == 'adapt'))
        elif gnn_config['arch'] == 'mixer':
            self.layers.append(MixerAggregator(num_neighbors, dim_node_feat, dim_edge_feat,
                                               gnn_config['dim_time'], dim_memory, 
                                               gnn_config['dim_out'], train_config['dropout'], 
                                               save_h_neigh_grad=sampler_config['type'] == 'adapt'))
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

        # time_encod_src = zero_time_feat
        # time_encod_dst = edge_time_feat[:block.pos_dst_size]
        if self.memory is not None:
            pos_src_nids = blocks[-1].root_nid[:block.pos_dst_size]
            pos_dst_nids = blocks[-1].root_nid[block.pos_dst_size:block.pos_dst_size * 2]
            assert (pos_edge_feats.shape[0] == pos_src_nids.shape[0] == pos_dst_nids.shape[0])
            pos_node_pairs = torch.concatenate([pos_src_nids, pos_dst_nids])
            pos_edge_times = blocks[-1].root_ts[:block.pos_dst_size]
            self.memory.update_memory(pos_src_nids, None,
                                    pos_dst_nids, None,
                                    pos_edge_times,
                                    pos_edge_feats)

        return h_in
    

    def forward(self, blocks, messages):
        h_in = self.aggregate_messages_and_update_memory(blocks, messages)


        return self.edge_predictor(h_in, blocks[-1].num_neg_dst)

class AdaptSampler(torch.nn.Module):
    def __init__(self, config, dim_edge_feat, dim_node_feat):
        super(AdaptSampler, self).__init__()

        sampler_config = config['sample'][0]

        self.num_sample = sampler_config['neighbor']
        self.num_scope = config['scope'][0]['neighbor'][0]  # TODO: support non equal scope for different layers

        self.enable_identity_encode = sampler_config['identity_encode']

        # force different encodes have same dim
        dim_feat = sampler_config['dim_feat']
        dim_encode = sampler_config['dim_encode']
        dim_link_encode = dim_encode + dim_encode \
                          + (dim_feat if dim_node_feat > 0 else 0) \
                          + (dim_feat if dim_edge_feat > 0 else 0) \
                          + (self.num_scope if self.enable_identity_encode else 0)  # - dim_feat
        dim_root_encode = dim_encode + dim_encode + (dim_feat if dim_node_feat > 0 else 0)

        self.node_feat_encoder = FeedForward(dim_node_feat, out_dims=dim_feat, dropout=0., use_single_layer=True)
        self.edge_feat_encoder = FeedForward(dim_edge_feat, out_dims=dim_feat, dropout=0., use_single_layer=True)
        self.time_encoder = FixedTimeEncoder(dim_encode)
        self.frequency_encoder = FixedFrequencyEncoder(dim_encode, sampler_config.get('freq_encode', False))

        self.link_mapper = LinkMapper(dim_link_encode, dim_root_encode, self.num_scope,
                                      feat_norm=sampler_config.get('feat_norm', False),
                                      neigh_norm=sampler_config.get('neigh_norm', False),
                                      decoder_type=sampler_config.get('decoder', 'transformer'),
                                      enable_mixer=sampler_config.get('mixer', True),
                                      init_gain=sampler_config.get('init_gain', 1.0),
                                      unif_bias=sampler_config.get('unif_bias', False),
                                      )

        self.keep_dummy_freq = sampler_config.get('keep_dummy_freq', True)
        self.log_prob = None

    def get_log_prob(self):
        return self.log_prob

    def forward(self, block: DenseTemporalBlock, is_input_layer=False):
        """
        Take Dense Temporal Block as input, out put sampled block
        """

        """encode root node"""
        root_feat_encode = self.node_feat_encoder(block.root_node_feature.to(torch.float32))
        root_time_encode = self.time_encoder(torch.zeros(block.n, dtype=torch.float32, device=block.device))
        root_freq_encode = self.frequency_encoder(torch.ones(block.n, dtype=torch.float32, device=block.device))
        root_encode = torch.cat([root_feat_encode, root_time_encode, root_freq_encode], dim=1)

        """encode link"""
        # feat encode
        edge_feat = block.neighbor_edge_feature.flatten(0, 1)
        node_feat = block.neighbor_node_feature.flatten(0, 1)
        edge_feat_encode = self.edge_feat_encoder(edge_feat.to(torch.float))
        node_feat_encode = self.node_feat_encoder(node_feat.to(torch.float))
        # time_encode
        neigh_time = block.root_ts.unsqueeze(-1) - block.neighbor_ts
        time_encode = self.time_encoder(neigh_time.flatten())
        # frequency encode
        neigh_mask = block.neighbor_nid.unsqueeze(1) == block.neighbor_nid.unsqueeze(2)
        neigh_count = neigh_mask.sum(dim=2)
        if not self.keep_dummy_freq:
            neigh_count[block.neighbor_nid == block.dummy_nid] = 0  # TODO: Do not set dummy node to zero
        neigh_freq = neigh_count / self.num_scope  # boost learning
        freq_encode = self.frequency_encoder(neigh_freq.flatten())
        # identity encode
        encode_list = [node_feat_encode, edge_feat_encode, time_encode, freq_encode]
        if self.enable_identity_encode:
            iden_encode = neigh_mask.to(torch.float).flatten(0, 1)
            encode_list.append(iden_encode)
        link_encode = torch.cat(encode_list, dim=1).view(block.n, block.b, -1)

        """compute prob"""
        probs = self.link_mapper(link_encode, root_encode)

        """sample neighbor"""
        # offset = 0.5
        # policy = Categorical(probs=probs*(1-offset)+offset)  # very bad
        policy = Categorical(probs=probs)
        action = policy.sample((self.num_sample,), replacement=False)
        if is_input_layer and self.training:
            # TODO: add some randomness
            self.log_prob = policy.log_prob(action).transpose(0, 1)
        action = action.transpose(0, 1)  # size (n, b') with replacement

        """update block"""
        block.update(action)

        # logging
        # hist, bin_edges = np.histogram(action.cpu().numpy(), bins=25)
        # print(hist)

        return block


# class ATGNN(TGNN):
#     """
#     Deprecated.
#     """
#
#     def __init__(self, config, dim_node_feat, dim_edge_feat):
#         super(ATGNN, self).__init__(config, dim_node_feat, dim_edge_feat)
#
#         self.num_sample = config['sample'][0]['neighbor']
#         self.num_scope = config['scope'][0]['neighbor'][0]
#
#         dim_time_encode = config['gnn'][0]['dim_time']
#         self.time_encoder = FixedTimeEncoder(dim_time_encode)
#
#         self.log_prob = None
#
#     def sample_loss(self):
#         return self.layers[0].sample_loss(self.log_prob)
#
#     def sample_neighbors(self, blocks: List[DenseTemporalBlock], link_mapper: LinkMapper):
#         blocks.reverse()
#
#         root_idx = torch.arange(blocks[0].n, device=blocks[0].device)
#         for i, block in enumerate(blocks):  # follow the top-down manner
#             # slice data from block
#             assert block.local_neigh_idx is None, 'Support Non-Unique Frontier Only'
#             neigh_ts = block.neighbor_ts[root_idx]
#             neigh_edge_feat = block.neighbor_edge_feature[root_idx]
#
#             # link encoding, size (n, b, d)
#             edge_feat = neigh_edge_feat.view(-1, block.neighbor_edge_feature.shape[-1])
#             time_encode = self.time_encoder((block.root_ts[root_idx].unsqueeze(-1) - neigh_ts).flatten())
#             link_encode = torch.cat([edge_feat, time_encode], dim=1)
#             link_encode = link_encode.view(-1, self.num_scope, link_encode.shape[-1])
#
#             # # prob mapping
#             probs = link_mapper(link_encode)
#             # policy = Categorical(probs=torch.ones_like(neigh_ts))
#             policy = Categorical(probs=probs)
#             action = policy.sample((self.num_sample,), replacement=False)
#             if i == len(blocks) - 1:  # input layer
#                 self.log_prob = policy.log_prob(action).transpose(0, 1)  # save log_prob or add some randomness?
#             action = action.transpose(0, 1)  # size (n, b') with replacement
#
#             # update current block data
#             block.root_nid = block.root_nid[root_idx]
#             block.root_ts = block.root_ts[root_idx]
#             block.neighbor_nid = torch.gather(block.neighbor_nid[root_idx], dim=1, index=action)
#             block.neighbor_eid = torch.gather(block.neighbor_eid[root_idx], dim=1, index=action)
#             block.neighbor_ts = torch.gather(block.neighbor_ts[root_idx], dim=1, index=action)
#             if block.root_node_feature is not None:
#                 block.root_node_feature = block.root_node_feature[root_idx]
#             if block.neighbor_node_feature is not None:
#                 idx = action.unsqueeze(-1).expand(action.size(0), action.size(1), block.neighbor_node_feature.size(-1))
#                 block.neighbor_node_feature = torch.gather(block.neighbor_node_feature[root_idx], dim=1, index=idx)
#             if block.neighbor_edge_feature is not None:
#                 idx = action.unsqueeze(-1).expand(action.size(0), action.size(1), block.neighbor_edge_feature.size(-1))
#                 block.neighbor_edge_feature = torch.gather(block.neighbor_edge_feature[root_idx], dim=1, index=idx)
#
#             # calculate root_idx for next frontier
#             offset = root_idx * block.b
#             sampled_idx = (action + offset.unsqueeze(1)).view(-1)
#             root_idx = torch.cat([sampled_idx, root_idx + block.n * block.b])
#
#             # update block size
#             block.n, block.b = action.shape[0], action.shape[1]
#
#         blocks.reverse()
#         return blocks
#
#     def forward(self, blocks, link_mapper=None):
#         blocks = self.sample_neighbors(blocks, link_mapper)
#         return super().forward(blocks)

