import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class LearnableTimeEncoder(torch.nn.Module):

    def __init__(self, dim):
        super(LearnableTimeEncoder, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(self.dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class FixedTimeEncoder(torch.nn.Module):

    def __init__(self, dim):
        super(FixedTimeEncoder, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class FixedFrequencyEncoder(torch.nn.Module):

    def __init__(self, dim, encode_type='sin'):
        super(FixedFrequencyEncoder, self).__init__()

        self.dim = dim
        assert encode_type in ['sin', 'fourier', 'poly']
        self.encode_type = encode_type

    @torch.no_grad()
    def forward(self, freqs):
        device = freqs.device
        if self.encode_type == 'sin':  # sinusoidal_encoding
            div_term = torch.exp(
                torch.arange(0., self.dim, 2, device=device) * -(torch.log(torch.tensor(10000.0)) / self.dim))
            encoded = torch.zeros(freqs.shape[0], self.dim, device=device)
            encoded[:, 0::2] = torch.sin(freqs.unsqueeze(-1) * div_term)
            encoded[:, 1::2] = torch.cos(freqs.unsqueeze(-1) * div_term)
        elif self.encode_type == 'poly':  # polynomial_encoding
            powers = torch.arange(self.dim + 1, device=device).unsqueeze(0)
            encoded = torch.pow(freqs.unsqueeze(-1), powers)
        elif self.encode_type == 'fourier':  # fourier_encoding
            signal = torch.sin(2 * torch.pi * freqs.unsqueeze(-1) * torch.arange(self.dim, device=device))
            spectrum = torch.fft.fft(signal)
            encoded = spectrum.real
        else:
            raise NotImplementedError
        return encoded