import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

class TransformerAggregator(nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, dim_memory, num_head, dim_out,
                 dropout=0.0):
        super(TransformerAggregator, self).__init__()

        self.h_v = None
        self.h_exp_a = None
        self.h_neigh = None

        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_memory = dim_memory
        self.num_head = num_head
        self.dim_out = dim_out

        self.dropout = nn.Dropout(dropout)

        self.att_dropout = nn.Dropout(dropout)
        self.att_act = nn.LeakyReLU(0.2)

        self.w_q = nn.Linear(dim_node_feat + dim_time + dim_memory, dim_out)
        self.w_k = nn.Linear(dim_node_feat + dim_edge_feat + dim_time + dim_memory, dim_out)
        self.w_v = nn.Linear(dim_node_feat + dim_edge_feat + dim_time + dim_memory, dim_out)
        self.w_out = nn.Linear(dim_node_feat + dim_out, dim_out)

        self.layer_norm = nn.LayerNorm(dim_out)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, 
                root_node_feature, 
                neighbor_node_feature, 
                neighbor_edge_feature,
                zero_time_feat,
                edge_time_feat,
                root_node_memory,
                neighbor_node_memory):
        
        h_q = self.w_q(torch.cat([root_node_feature, zero_time_feat, root_node_memory], dim=1))
        h_k = self.w_k(torch.cat([neighbor_node_feature, neighbor_edge_feature, edge_time_feat, neighbor_node_memory], dim=1))
        h_v = self.w_v(torch.cat([neighbor_node_feature, neighbor_edge_feature, edge_time_feat, neighbor_node_memory], dim=1))

        h_q = h_q.view((h_q.shape[0], 1, self.num_head, -1))
        h_k = h_k.view((h_q.shape[0], -1, self.num_head, h_q.shape[-1]))
        h_v = h_v.view((h_q.shape[0], -1, self.num_head, h_q.shape[-1]))

        h_att = self.att_act(torch.sum(h_q * h_k, dim=3))
        h_att = F.softmax(h_att, dim=1).unsqueeze(-1)
        h_neigh = (h_v * h_att).sum(dim=1)
        h_neigh = h_neigh.view(h_v.shape[0], -1)
        h_out = self.w_out(torch.cat([h_neigh, root_node_feature], dim=1)) # residual
        h_out = self.layer_norm(nn.functional.relu(self.dropout(h_out)))
        return h_out


class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """

    def __init__(self, dims, expansion_factor=1., dropout=0., use_single_layer=False,
                 out_dims=0, use_act=True):
        super().__init__()

        self.h_v = None
        self.h_neigh = None

        self.use_single_layer = use_single_layer
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.use_act = use_act

        out_dims = dims if out_dims == 0 else out_dims

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, out_dims)
            self.detached_linear_0 = nn.Linear(dims, out_dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.detached_linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), out_dims)

        self.reset_parameters()

    def reset_parameters(self, init_type='model', gain=1.0):
        if init_type == 'model':
            self.linear_0.reset_parameters()
            if not self.use_single_layer:
                self.linear_1.reset_parameters()
        elif init_type == 'sampler':
            init.xavier_uniform_(self.linear_0.weight, gain=gain)
            init.zeros_(self.linear_0.bias)
            if not self.use_single_layer:
                init.xavier_uniform_(self.linear_1.weight, gain=gain)
                init.zeros_(self.linear_1.bias)
        elif init_type == 'model_zero':
            init.kaiming_uniform_(self.linear_0.weight, a=math.sqrt(5))
            init.zeros_(self.linear_0.bias)
            if not self.use_single_layer:
                init.kaiming_uniform_(self.linear_1.weight, a=math.sqrt(5))
                init.zeros_(self.linear_1.bias)
        else:
            raise NotImplementedError

    def sample_loss(self, log_prob):
        grad_h_neigh = self.h_neigh.grad.detach()

        self.detached_linear_0.load_state_dict(self.linear_0.state_dict())
        for para in self.detached_linear_0.parameters():
            para.requires_grad = False
        h_neigh = self.detached_linear_0(log_prob.unsqueeze(1) * self.h_v.detach())

        batch_loss = torch.bmm(grad_h_neigh.view(grad_h_neigh.shape[0], 1, -1),
                               h_neigh.view(h_neigh.shape[0], -1, 1))

        # none negative node, bad performance
        # batch_size = batch_loss.shape[0] // 3 * 2
        # batch_loss = batch_loss[:batch_size]

        return batch_loss

    def forward(self, x):
        if x.shape[-1] == 0:
            return x

        x = self.linear_0(x)

        if self.use_act:
            x = F.gelu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        if not self.use_single_layer:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """

    def __init__(self, num_neighbor, dim_feat,
                 token_expansion_factor=0.5,
                 channel_expansion_factor=4.,
                 dropout=0.,):
        super().__init__()

        self.token_layernorm = nn.LayerNorm(dim_feat)
        self.token_forward = FeedForward(num_neighbor, token_expansion_factor, dropout,)

        self.channel_layernorm = nn.LayerNorm(dim_feat)
        self.channel_forward = FeedForward(dim_feat, channel_expansion_factor, dropout)

    def reset_parameters(self, init_type='model', gain=1.0):
        self.token_layernorm.reset_parameters()
        self.token_forward.reset_parameters(init_type, gain)

        self.channel_layernorm.reset_parameters()
        self.channel_forward.reset_parameters(init_type, gain)

    def sample_loss(self, log_prob):
        return self.token_forward.sample_loss(log_prob)

    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x

    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class MixerAggregator(nn.Module):

    def __init__(self, num_neighbor, dim_node_feat, dim_edge_feat, dim_time, dim_memory, dim_out, dropout=0.0,):
        super(MixerAggregator, self).__init__()

        self.num_neighbor = num_neighbor
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_memory = dim_memory
        self.dim_out = dim_out

        self.mixer = MixerBlock(num_neighbor, dim_node_feat + dim_edge_feat + dim_time + dim_memory,
                                dropout=dropout,)
        self.layer_norm = nn.LayerNorm(dim_node_feat + dim_edge_feat + dim_time + dim_memory)
        self.mlp_out = nn.Linear(dim_node_feat + dim_edge_feat + dim_time + dim_memory, dim_out)

    def sample_loss(self, log_prob):
        return self.mixer.sample_loss(log_prob)

    def forward(self, 
                root_node_feature, 
                neighbor_node_feature, 
                neighbor_edge_feature,
                zero_time_feat,
                edge_time_feat,
                root_node_memory,
                neighbor_node_memory):

        feats = torch.cat([neighbor_node_feature, neighbor_edge_feature, edge_time_feat, neighbor_node_memory], dim=1)
        feats = feats.view(-1, self.num_neighbor, feats.shape[-1])

        feats = self.mixer(feats)

        h_out = self.layer_norm(feats)
        h_out = torch.mean(h_out, dim=1)
        h_out = self.mlp_out(h_out)

        return h_out
