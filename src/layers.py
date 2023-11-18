import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch_geometric.nn import inits



class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


class SelfNorm(torch.nn.Module):
    def __init__(self, num_scope, dropout=0., eps=1e-5):
        super(SelfNorm, self).__init__()

        self.norm = nn.LayerNorm(num_scope)
        self.linear = nn.Linear(num_scope, 1)
        self.dropout = nn.Dropout(p=dropout)

        # self.weight = nn.Parameter(torch.tensor([0.1]))
        # self.bias = nn.Parameter(torch.tensor([0.]))
        # self.weight = torch.tensor([1e-1])
        # self.bias = torch.tensor([0.])
        # self.eps = eps

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.linear.reset_parameters()

        # self.weight = nn.Parameter(torch.tensor([1.]))
        # self.bias = nn.Parameter(torch.tensor([0.]))


    def forward(self, x, neigh_mask):
        d = x.unsqueeze(dim=1).repeat(1, 25, 1)
        r = torch.zeros_like(d)  # (1800, 25, 25)
        r[neigh_mask] = d[neigh_mask]

        # neigh_count = neigh_mask.sum(dim=2)
        # mean = r.sum(dim=2) / neigh_count
        # var = torch.square(r - mean.unsqueeze(dim=2)).sum(dim=2) / neigh_count
        # x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # x = self.weight * x_norm + self.bias

        r = self.norm(r)
        x = self.linear(r).squeeze(-1)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class AttentionDecoder(torch.nn.Module):

    def __init__(self, dim_neigh_encode, dim_root_encode, att_type, dim_embed=100):
        super(AttentionDecoder, self).__init__()

        att_type = att_type.lower()
        assert att_type in ['transformer', 'gat_v1', 'gat_v2']
        self.att_type = att_type
        self.dim_embed = dim_embed
        self.dim_neigh_encode = dim_neigh_encode
        self.dim_root_encode = dim_root_encode

        self.w_q = nn.Linear(dim_root_encode, dim_embed, bias=True)  # bias is important for attention
        self.w_k = nn.Linear(dim_neigh_encode, dim_embed, bias=True)
        self.att_act = nn.LeakyReLU(0.2)
        self.att = nn.Parameter(torch.empty(dim_embed, 1))

        self.reset_parameters()

    def reset_parameters(self, gain=1.0, bias=False):
        """
        For weight init, xavier is better than kaiming for attention, linear vise versa
        """
        nn.init.xavier_uniform_(self.w_q.weight, gain)
        nn.init.xavier_uniform_(self.w_q.weight, gain)
        nn.init.xavier_uniform_(self.att, gain)

        # nn.init.kaiming_uniform_(self.w_q.weight, 0.2, nonlinearity='leaky_relu')
        # nn.init.kaiming_uniform_(self.w_q.weight, 0.2, nonlinearity='leaky_relu')
        # nn.init.kaiming_uniform_(self.att, 0.2, nonlinearity='leaky_relu')

        """
        For bias init, zero is better for mixer+att, uniform is better for att only
        """
        if bias:
            bound = 1 / math.sqrt(self.dim_root_encode)
            init.uniform_(self.w_q.bias, -bound, bound)
            bound = 1 / math.sqrt(self.dim_neigh_encode)
            init.uniform_(self.w_k.bias, -bound, bound)
        else:
            nn.init.zeros_(self.w_q.bias)
            nn.init.zeros_(self.w_k.bias)

    def forward(self, neigh_encode, root_encode):
        h_q = self.w_q(root_encode).view(root_encode.shape[0], 1, self.dim_embed)
        h_k = self.w_k(neigh_encode).view(root_encode.shape[0], -1, self.dim_embed)
        h = h_q + h_k

        if self.att_type == 'transformer':  # good for mixer
            alpha = self.att_act(torch.sum(h_q * h_k, dim=-1))
        elif self.att_type == 'gat_v1':
            alpha = self.att_act(h @ self.att)
        elif self.att_type == 'gat_v2':  # good for tgat
            alpha = self.att_act(h) @ self.att
        else:
            raise NotImplementedError

        return F.softmax(alpha, dim=1)


class LinkMapper(torch.nn.Module):
    """
    Mapping Link Encoding to Sample Probability
    (num_roots, num_scope, dim) -> (num_roots, num_scope)
    """

    def __init__(self, link_encode_dims, root_encode_dims, num_scope,
                 feat_norm=False, neigh_norm=False, dropout=0., decoder_type='transformer',
                 enable_mixer=True, init_gain=1.0, unif_bias=False):
        super(LinkMapper, self).__init__()

        self.decoder_type = decoder_type.lower()

        if enable_mixer:
            self.mixer = MixerBlock(num_scope, link_encode_dims, dropout=dropout,
                                    token_expansion_factor=0.2, channel_expansion_factor=0.2)
        if self.decoder_type == 'linear':
            self.decoder = nn.Sequential(
                FeedForward(link_encode_dims, out_dims=1, dropout=dropout, use_single_layer=True,
                            use_act=True),
                nn.Softmax(dim=1),
            )
        else:
            self.decoder = AttentionDecoder(link_encode_dims, root_encode_dims, self.decoder_type, dim_embed=100)

        if feat_norm:
            self.feat_norm = nn.LayerNorm(link_encode_dims)
            self.feat_norm_root = nn.LayerNorm(root_encode_dims)
        self.neigh_norm = neigh_norm

        self.reset_parameters(init_gain, unif_bias)

    def reset_parameters(self, gain, bias):
        if hasattr(self, 'mixer'):
            self.mixer.reset_parameters(init_type='model')
        if self.decoder_type == 'linear':
            self.decoder[0].reset_parameters(init_type='model')
        else:
            self.decoder.reset_parameters(gain=gain, bias=bias)

    def forward(self, x, x_root):
        if hasattr(self, 'feat_norm'):
            x = self.feat_norm(x)
            x_root = self.feat_norm_root(x_root)

        if hasattr(self, 'mixer'):
            x = self.mixer(x)  # significant improvement in GraphMixer

        if self.neigh_norm:
            x = F.normalize(x, p=2, dim=1)  # works alright

        if self.decoder_type == 'linear':
            x = self.decoder(x)
        else:
            x = self.decoder(x, x_root)

        """
        Trials:
        x = F.gelu(x)  # bad performance, very slow convergence
        x = self.feat_norm(x)  # useless

        x = x + self.self_norm(x, *args)  # bad
        x = self.neigh_norm(x)  # very bad performance
        x = F.normalize(x, p=2, dim=1)  # bad at here

        x = F.softmax(x, dim=1)  # softmax is important
        x = F.relu(x) + 10e-10  # very bad
        x = F.dropout(x, p=0.1, training=self.training) + 1e-10  # pretty bad
        
        x = self.decoder(x, x.mean(dim=1)) # a little bit worse 
        """
        return x.squeeze()

if __name__ == '__main__':
    ff = FeedForward(10, 1, 0, False, 3, True, True)
    input = torch.tensor(np.random.random((3, 10))).float()
    output = ff(input)
    print(output)