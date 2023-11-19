import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from collections import defaultdict
from modules.time_encoder import LearnableTimeEncoder, FixedTimeEncoder

class Memory(nn.Module):
    def __init__(self, n_nodes, dim_memory):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.dim_memory = dim_memory
        self.memory = nn.Parameter(torch.zeros(n_nodes, dim_memory), requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes),
                                    requires_grad=False)
        self.new_memory = None
        self.update_nids = None
    
    def get_memory(self, nids):
        return self.memory[nids, :]
    
    def set_memory(self, nids, memory):
        self.memory[nids, :] = memory

    def get_last_update(self, nids):
        return self.last_update[nids]
    
    # def update_memory_from_stored(self):
    #     self.memory[self.update_nids, :] = self.new_memory
    #     self.update_nids = None
    #     self.new_memory = None
    
    # def store_memory_for_update(self, nids, new_memory):
    #     assert len(nids) == new_memory.shape[0]
    #     assert self.memory.shape[1] == new_memory.shape[1]
    #     self.new_memory = new_memory
    #     self.update_nids = nids

    def set_last_update(self, nids, last_update):
        assert len(nids) == last_update.shape[0]
        self.last_update[nids] = last_update

    def update_memory(self):
        pass


class GRUMemory(Memory):
    def __init__(self, n_nodes, dim_memory, dim_msg, dim_time, msg_reducer_type, time_encoder_type):
        super(GRUMemory, self).__init__(n_nodes, dim_memory)
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory.requires_grad = False
        self.memory_updater = nn.GRUCell(input_size=dim_msg,
                                     hidden_size=dim_memory)
        self.messages = None
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
    
    def get_raw_messages(self, src_nodes,
                         src_node_embeddings, 
                         dst_nodes, 
                         dst_node_embeddings,
                         edge_times,
                         edge_features):
        # TODO: implement the version that uses src and dst node embeddings
        src_memory = self.get_memory(src_nodes)
        dst_memory = self.get_memory(dst_nodes)
        source_time_delta = edge_times - self.get_last_update(src_nodes)
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(src_nodes), -1)
        src_message = torch.cat([src_memory, dst_memory, edge_features,
                                source_time_delta_encoding],
                                dim=1)
        messages = defaultdict(list)
        unique_sources = torch.unique(src_nodes)

        for i in range(len(src_nodes)):
            messages[src_nodes[i].item()].append((src_message[i], edge_times[i]))
        return unique_sources, messages
            

    def update_memory(self, src_nids, src_embeddings, dst_nids, dst_embeddings, edge_times, edge_features):
        unique_sources, source_id_to_messages = self.get_raw_messages(src_nids,
                                                                    src_embeddings,
                                                                    dst_nids,
                                                                    dst_embeddings,
                                                                    edge_times, edge_features)
        
        unique_destinations, destination_id_to_messages = self.get_raw_messages(dst_nids,
                                                                              dst_embeddings,
                                                                              src_nids,
                                                                              src_embeddings,
                                                                              edge_times, edge_features)
        for nodes, messages in zip([unique_sources, unique_destinations], [source_id_to_messages, destination_id_to_messages]):
            unique_nids, unique_messages, unique_timestamps = \
            self.message_reducer.aggregate(nodes, messages)
            self.update_memory_by_reduced_messages(unique_nids, unique_messages, unique_timestamps)
            

    def update_memory_by_reduced_messages(self, unique_nids, unique_messages, unique_timestamps):
        if (len(unique_nids) == 0): return
        memory = self.get_memory(unique_nids)
        updated_memory = self.memory_updater(unique_messages, memory)
        self.set_memory(unique_nids, updated_memory)
        # assert torch.allclose(memory, self.get_memory(unique_nids), atol=1e-5), \
        #   "Something wrong in how the memory was updated"
        self.set_last_update(unique_nids, unique_timestamps)

    def detach_memory(self):
        self.memory.detach_()


class EmbeddingTableMemory(Memory):
    def __init__(self, n_nodes, dim_memory, dim_msg=0):
        super(EmbeddingTableMemory, self).__init__(n_nodes, dim_memory)
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
    
    def aggregate(self, unique_nids, messages):
        unique_messages = []
        unique_timestamps = []
        for nid in unique_nids:
            assert len(messages[nid]) > 0
            unique_messages.append(messages[nid][-1][0])
            unique_timestamps.append(messages[nid][-1][1])
        unique_messages = torch.stack(unique_messages) if len(unique_nids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(unique_nids) > 0 else []
        return unique_nids, unique_messages, unique_timestamps
