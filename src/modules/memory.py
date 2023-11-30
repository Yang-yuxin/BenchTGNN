import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from collections import defaultdict
from modules.time_encoder import LearnableTimeEncoder, FixedTimeEncoder
# from torch.profiler import profile, record_function, ProfilerActivity

class Memory(nn.Module):
    def __init__(self, n_nodes, dim_memory, device):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.dim_memory = dim_memory
        self.messages = defaultdict(list)
        self.device = device
        self.__init_memory__()

    def get_memory(self, nids, memory):
        return memory[nids, :]
    
    def set_memory(self, nids, memory):
        self.memory[nids, :] = memory

    def get_last_update(self, nids):
        return self.last_update[nids]
    
    def __init_memory__(self):
        self.memory = nn.Parameter(nn.init.orthogonal_(torch.randn(self.n_nodes, self.dim_memory)), requires_grad=False).to(self.device)
        self.messages = defaultdict(list)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes),
                                            requires_grad=False).to(self.device)
    

    def set_last_update(self, nids, last_update):
        assert len(nids) == last_update.shape[0]
        assert (self.last_update[nids] <= last_update).all()
        self.last_update[nids] = last_update

    def update_memory(self):
        pass


class GRUMemory(Memory):
    def __init__(self, device, n_nodes, dim_memory, dim_node_feat, dim_msg, 
    dim_time, msg_reducer_type, time_encoder_type, use_embedding_in_message):
        super(GRUMemory, self).__init__(n_nodes, dim_memory, device)
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory.requires_grad = False
        self.memory_updater = nn.GRUCell(input_size=dim_msg,
                                     hidden_size=dim_memory)
        self.message_reducer = None
        if msg_reducer_type == 'mean':
            self.message_reducer = MeanMemoryMessageReducer() 
        elif msg_reducer_type == 'last':
            self.message_reducer = LastMemoryMessageReducer()
        else:
            raise NotImplementedError
        
        if time_encoder_type == 'fixed':
            self.time_encoder = FixedTimeEncoder(dim_time)
        elif time_encoder_type == 'learnable':
            self.time_encoder = LearnableTimeEncoder(dim_time)
        elif time_encoder_type == 'none':
            self.time_encoder = None
        else:
            raise NotImplementedError
        
        self.use_embedding_in_message = use_embedding_in_message
        if use_embedding_in_message:
            self.linear = nn.Linear(dim_memory + dim_node_feat, dim_memory)
        else:
            self.linear = nn.Linear(dim_memory + dim_node_feat, dim_memory)
    
    def store_raw_messages(self, src_nodes,
                         src_node_embeddings, 
                         dst_nodes, 
                         dst_node_embeddings,
                         edge_times,
                         edge_features):
        """
        Store messages in the self.messages[src_nodes]
        """
        # TODO: implement the version that uses src and dst node embeddings
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("get_memory"):  
        #         src_memory = self.get_memory(src_nodes)
        #         dst_memory = self.get_memory(dst_nodes)
        #     with record_function("get_time_encoding"):  
        #         source_time_delta = edge_times - self.get_last_update(src_nodes)
        #         source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(src_nodes), -1)
        #     with record_function("concatenation"):
        #         src_message = torch.cat([src_memory, dst_memory, edge_features,
        #                         source_time_delta_encoding],
        #                         dim=1)
        #     messages = defaultdict(list)
        #     with record_function("get_unique_msgs"):  
        #         unique_sources = torch.unique(src_nodes)
        #     with record_function("appending_msgs"):  
        #         for i in range(len(src_nodes)):
        #             messages[src_nodes[i].item()].append((src_message[i], edge_times[i]))
        # print(prof.key_averages().table(sort_by="cpu_time_total",))
        src_memory = self.get_memory(src_nodes, self.memory) 
        if self.use_embedding_in_message:
            src_memory = self.linear(torch.cat([src_node_embeddings, src_memory], 1))
        dst_memory = self.get_memory(dst_nodes, self.memory) 
        if self.use_embedding_in_message:
            dst_memory = self.linear(torch.cat([dst_node_embeddings, dst_memory], 1))
        
        source_time_delta = edge_times - self.get_last_update(src_nodes)
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(src_nodes), -1)
        src_message = torch.cat([src_memory, dst_memory, edge_features,
                                source_time_delta_encoding],
                                dim=1)
        unique_sources = torch.unique(src_nodes)
        for i in range(len(src_nodes)):
            self.messages[src_nodes[i].item()].append((src_message[i], edge_times[i])) # LOWWWWWWWWWW efficiency!!!
        return unique_sources, messages
        # unique_sources, inv = torch.unique(nid, return_inverse=True)
        # perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        # perm = inv.new_empty(unique_sources.size(0)).scatter_(0, inv, perm)
        # nid = nid[perm]
        # mail = mail[perm]
        # mail_ts = mail_ts[perm]
        # return unique_sources, messages

    def update_memory(self, nodes, messages):
        unique_nids, unique_messages, unique_timestamps = \
            self.message_reducer.aggregate(nodes, messages)
        if not len(unique_messages):
            return
        memory = self.get_memory(unique_nids, self.memory)
        updated_memory = self.memory_updater(unique_messages, memory)
        self.set_memory(unique_nids, updated_memory)
        self.set_last_update(unique_nids, unique_timestamps)
    
    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []

    def get_updated_memory(self, nodes, messages):
        # get updated memory of src, src_neigh, dst, dst_neigh, neg, neg_neigh
        unique_nids, unique_messages, unique_timestamps = \
            self.message_reducer.aggregate(nodes, messages)
        if len(unique_nids) <= 0:
            return self.memory.data.clone(), self.last_update.data.clone()
        updated_memory = self.memory.data.clone()
        updated_memory[unique_nids] = self.memory_updater(unique_messages, self.get_memory(unique_nids, self.memory))
        return updated_memory, self.last_update.data.clone()

    def detach_memory(self):
        self.memory.detach_()
        # Detach all stored messages
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.messages[k] = new_node_messages


class EmbeddingTableMemory(Memory):
    def __init__(self, device, n_nodes, dim_memory, dim_msg=0):
        super(EmbeddingTableMemory, self).__init__(n_nodes, dim_memory, device)
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory.requires_grad = True
    
    
class MeanMemoryMessageReducer(nn.Module):
    def __init__(self):
        super(MeanMemoryMessageReducer, self).__init__()
    
    def aggregate(self, unique_nids, messages):
        """
        Reduce [M nodes, N messages] to one message per node.
        """
        unique_messages = []
        unique_timestamps = []
        for nid in unique_nids:
            nid_item = nid.item()
            assert len(messages[nid_item]) > 0
            unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[nid_item]]), dim=0))
            unique_timestamps.append(messages[nid_item][-1][1])
        unique_messages = torch.stack(unique_messages) if len(unique_nids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(unique_nids) > 0 else []
        return unique_nids, unique_messages, unique_timestamps

class LastMemoryMessageReducer(nn.Module):
    def __init__(self):
        super(LastMemoryMessageReducer, self).__init__()
    
    def aggregate(self, nodes, messages):
        unique_nids = torch.unique(torch.tensor(nodes))
        to_update_nids = []
        unique_messages = []
        unique_timestamps = []
        for nid in unique_nids:
            nid_item = nid.item() if isinstance(nid, torch.Tensor) else nid
            if (len(messages[nid_item]) > 0):
                unique_timestamps.append(messages[nid_item][-1][1])
                unique_messages.append(messages[nid_item][-1][0])
                to_update_nids.append(nid)
        if len(to_update_nids) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)
        else:
            unique_messages = []
            unique_timestamps = []
        return to_update_nids, unique_messages, unique_timestamps
