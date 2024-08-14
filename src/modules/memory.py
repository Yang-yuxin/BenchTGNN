import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from collections import defaultdict
from modules.time_encoder import LearnableTimeEncoder, FixedTimeEncoder
# from torch.profiler import profile, record_function, ProfilerActivity

class Memory(nn.Module):
    def __init__(self, n_nodes, dim_memory, dim_msg, len_msg, device):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.dim_memory = dim_memory
        self.dim_msg = dim_msg
        self.len_msg = len_msg
        self.device = device
        self.mailbox = torch.zeros((n_nodes, len_msg, dim_msg), dtype=torch.float32).to(self.device)
        self.mailbox_mask = torch.zeros((n_nodes, len_msg), dtype=torch.bool).to(self.device)
        self.mailbox_pointer = torch.zeros((n_nodes), dtype=torch.int32).to(self.device)
        self.mailbox_ts = torch.zeros((n_nodes), dtype=torch.float32).to(self.device)
        self.last_update = torch.zeros(n_nodes).to(self.device)
       
    def get_memory(self, nids, memory=None):
        if memory == None:
            memory = self.memory
        return memory[nids, :]
    
    def set_memory(self, nids, memory):
        self.memory[nids, :] = memory

    def get_last_update(self, nids):
        return self.last_update[nids]
    
    def to_device(self):
        self.memory.cuda()
    
    def __init_memory__(self, isgru):
        if isgru:
            torch.nn.init.zeros_(self.memory)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.last_update.fill_(0)
    
    def set_last_update(self, nids, last_update):
        assert len(nids) == last_update.shape[0]
        try:
            assert (self.last_update[nids] <= last_update).all()
        except AssertionError:
            import pdb; pdb.set_trace()
        self.last_update[nids] = last_update

    def update_memory(self):
        pass


class GRUMemory(Memory):
    def __init__(self, device, n_nodes, dim_memory, dim_node_feat, dim_msg, len_msg, 
    dim_time, msg_reducer_type, time_encoder_type, use_embedding_in_message):
        super(GRUMemory, self).__init__(n_nodes, dim_memory, dim_msg, len_msg, device)
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
        
        if edge_features is not None:
            src_message = torch.cat([src_memory, dst_memory, edge_features], dim=1)
            dst_message = torch.cat([dst_memory, src_memory, edge_features], dim=1)
        else:
            src_message = torch.cat([src_memory, dst_memory], dim=1)
            dst_message = torch.cat([dst_memory, src_memory], dim=1)
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
            self.mailbox[nid.long()] = mail.unsqueeze(1)
            self.mailbox_ts[nid.long()] = mail_ts
        else:
            raise NotImplementedError

    def update_memory(self, nodes, memory):
        unique_nids, inv = torch.unique(nodes, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(unique_nids.size(0)).scatter_(0, inv, perm)
        self.set_memory(unique_nids, memory[perm])
        self.set_last_update(unique_nids, self.mailbox_ts[unique_nids])

    def get_updated_memory(self, nodes):
        unique_nids, inv = torch.unique(nodes, return_inverse=True)
        to_update_nids = unique_nids
        updated_memory = self.memory.data.clone().squeeze()
        updated_memory[to_update_nids] = self.memory_updater(self.mailbox[to_update_nids, :].squeeze(), updated_memory[to_update_nids, :])
        return updated_memory[nodes, :], self.last_update[nodes].data.clone()

    def detach_memory(self):
        self.memory.detach_()
        self.mailbox.detach_()
    

class EmbeddingTableMemory(Memory):    
    def __init__(self, device, n_nodes, dim_memory, dim_msg=0, len_msg=1, init_trick=False):
        if init_trick:
            dim_msg = dim_memory
        super(EmbeddingTableMemory, self).__init__(n_nodes, dim_memory, dim_msg, len_msg, device)
        self.memory = nn.Parameter(torch.zeros(self.n_nodes, self.dim_memory), requires_grad=True)
        self.inductive_mask = torch.ones(n_nodes).to(self.device)
        self.cutoff = min(3, self.len_msg)
        torch.nn.init.normal_(self.memory)

    def backup_memory(self):
        return self.memory.clone()
    
    def restore_memory(self, memory):
        self.memory = memory.clone()
    
    def update_memory(self, nodes):
        update_mask = torch.zeros(self.n_nodes, dtype=torch.bool).to(self.device)
        update_mask[nodes] = True
        update_mask = update_mask.logical_and(self.inductive_mask)
        cutoff_mask = (torch.sum(self.mailbox_mask[nodes], dim=1) > self.cutoff).to(self.device)
        update_mask = update_mask.logical_and(cutoff_mask)
        if update_mask.sum() == 0:
            return
        new_node_memory = torch.sum(self.mailbox[update_mask], dim=1) / torch.sum(self.mailbox_mask[update_mask], dim=1)
        self.memory[update_mask] = new_node_memory
        self.inductive_mask[update_mask] = False

    def store_raw_messages(self, src_nids, dst_nids):
        num_neighs = dst_nids.shape[1]
        msgs = self.memory[dst_nids.view(dst_nids.shape[0]*dst_nids.shape[1])].reshape(-1, num_neighs, self.memory.shape[1])
        msgs = torch.mean(msgs, dim=1)
        self.mailbox[src_nids, self.mailbox_pointer[src_nids]] = msgs
        self.mailbox_mask[src_nids, self.mailbox_pointer[src_nids]] = True
        self.mailbox_pointer[src_nids] = (self.mailbox_pointer[src_nids]+1) % self.len_msg
        # self.mailbox[dst_nids, self.mailbox_pointer[dst_nids]] = self.memory[src_nids]
        # self.mailbox_mask[dst_nids, self.mailbox_pointer[dst_nids]] = True
        # self.mailbox_pointer[dst_nids] = (self.mailbox_pointer[dst_nids]+1) % self.len_msg

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
