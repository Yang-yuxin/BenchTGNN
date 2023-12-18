import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from collections import defaultdict
from modules.time_encoder import LearnableTimeEncoder, FixedTimeEncoder
# from torch.profiler import profile, record_function, ProfilerActivity

class Memory(nn.Module):
    def __init__(self, n_nodes, dim_memory, dim_msg, device):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.dim_memory = dim_memory
        # self.messages = defaultdict(list)
        self.dim_msg = dim_msg
        self.device = device
        self.mailbox = torch.zeros((n_nodes, dim_msg), dtype=torch.float32).to(self.device)
        self.mailbox_ts = torch.zeros((n_nodes), dtype=torch.float32).to(self.device)
        self.last_update = torch.zeros(n_nodes).to(self.device)
       
        
    def get_memory(self, nids, memory):
        return memory[nids, :]
    
    def set_memory(self, nids, memory):
        self.memory[nids, :] = memory

    def get_last_update(self, nids):
        return self.last_update[nids]
    
    def to_device(self):
        self.memory.cuda()
    
    def __init_memory__(self, isgru):
        if isgru:
            # self.memory = nn.Parameter(torch.zeros(self.n_nodes, self.dim_memory), requires_grad=False).to(self.device)
            torch.nn.init.normal_(self.memory)
            # self.memory.fill_(0)
        # self.memory.fill_(0)
        # self.memory = nn.Embedding(self.n_nodes, self.dim_memory).to(self.device)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.last_update.fill_(0)
    

    def set_last_update(self, nids, last_update):
        assert len(nids) == last_update.shape[0]
        assert (self.last_update[nids] <= last_update).all()
        self.last_update[nids] = last_update

    def update_memory(self):
        pass


class GRUMemory(Memory):
    def __init__(self, device, n_nodes, dim_memory, dim_node_feat, dim_msg, 
    dim_time, msg_reducer_type, time_encoder_type, use_embedding_in_message):
        super(GRUMemory, self).__init__(n_nodes, dim_memory, dim_msg, device)
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory = nn.Parameter(torch.zeros(self.n_nodes, self.dim_memory), requires_grad=False).to(device)
        self.memory_updater = nn.GRUCell(input_size=dim_msg,
                                     hidden_size=dim_memory)
        self.message_reducer_type = msg_reducer_type
        
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
        torch.nn.init.normal_(self.memory)
        self.__init_memory__(True)

    
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
        src_memory = self.get_memory(src_nodes, self.memory) 
        if self.use_embedding_in_message:
            src_memory = self.linear(torch.cat([src_node_embeddings, src_memory], 1))
        dst_memory = self.get_memory(dst_nodes, self.memory) 
        if self.use_embedding_in_message:
            dst_memory = self.linear(torch.cat([dst_node_embeddings, dst_memory], 1))
        
        # source_time_delta = edge_times - self.get_last_update(src_nodes)
        # source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(src_nodes), -1)
        # src_message = torch.cat([src_memory, dst_memory, edge_features,
        #                         source_time_delta_encoding],
        #                         dim=1)
        # source_time_delta = edge_times - self.get_last_update(src_nodes)
        # source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(src_nodes), -1)
        src_message = torch.cat([src_memory, dst_memory, edge_features], dim=1)
        dst_message = torch.cat([dst_memory, src_memory, edge_features], dim=1)
        nid = torch.cat([src_nodes.unsqueeze(1), dst_nodes.unsqueeze(1)], dim=1).reshape(-1)
        mail = torch.cat([src_message, dst_message], dim=1).reshape(-1, src_message.shape[1])
        # tgn mailbox
        unique_sources, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(unique_sources.size(0)).scatter_(0, inv, perm)
        nid = nid[perm]
        mail = mail[perm]
        mail_ts = edge_times[perm]
        if self.message_reducer_type == 'last':
            idx_to_update = nid
            self.mailbox[nid.long()] = mail
            self.mailbox_ts[nid.long()] = mail_ts
        else:
            raise NotImplementedError

    def update_memory(self, nodes):
        unique_nids = torch.unique(nodes)
        to_update_nids = unique_nids
        memory = self.get_memory(to_update_nids, self.memory)
        updated_memory = self.memory_updater(self.mailbox[to_update_nids, :], memory)
        self.set_memory(to_update_nids, updated_memory)
        self.set_last_update(to_update_nids, self.mailbox_ts[to_update_nids])

    


    def get_updated_memory(self, nodes):
        unique_nids, inv = torch.unique(nodes, return_inverse=True)
        to_update_nids = unique_nids
        # import pdb; pdb.set_trace()
        memory = self.get_memory(to_update_nids, self.memory)
        updated_memory = self.memory.data.clone()
        with torch.no_grad():
            updated_memory[to_update_nids] = self.memory_updater(self.mailbox[to_update_nids, :], updated_memory[to_update_nids, :])
        return updated_memory[nodes, :], self.last_update[nodes].data.clone()

    def detach_memory(self):
        self.memory.detach_()
        self.mailbox.detach_()
    

class EmbeddingTableMemory(Memory):    
    def __init__(self, device, n_nodes, dim_memory, dim_msg=0):
        super(EmbeddingTableMemory, self).__init__(n_nodes, dim_memory, dim_msg, device)
        # self.memory = nn.Embedding(self.n_nodes, self.dim_memory).to(self.device)
        # Treat memory as parameter so that it is saved and loaded together with the model
        # self.memory.requires_grad = True
        # print(self.memory.requires_grad)
        self.memory = nn.Parameter(torch.zeros(self.n_nodes, self.dim_memory), requires_grad=True)
        torch.nn.init.normal_(self.memory)

    def backup_memory(self):
        return self.memory.clone()
    
    def restore_memory(self, memory):
        self.memory = memory.clone()
    
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
