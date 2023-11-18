import torch
import matplotlib.pyplot as plt

data = ['WIKI', 'REDDIT']
num_bins = 100

fig, axss = plt.subplots(len(data), 3, figsize=(15, len(data) *5))
for d, axs in zip(data, axss):
    pt = torch.load('DATA/{}/edges.pt'.format(d))

    src_deg = pt['train_src_deg']
    x = torch.arange(src_deg.min(), src_deg.max(), (src_deg.max() - src_deg.min()) / num_bins)
    src_deg = torch.histc(src_deg.float(), bins=num_bins)
    axs[0].plot(x[:src_deg.shape[0]], src_deg.int())
    axs[0].set_title('{} Src Deg'.format(d))

    dst_deg = pt['train_dst_deg']
    x = torch.arange(dst_deg.min(), dst_deg.max(), (dst_deg.max() - dst_deg.min()) / num_bins)
    dst_deg = torch.histc(dst_deg.float(), bins=num_bins)
    axs[1].plot(x[:dst_deg.shape[0]], dst_deg.int())
    axs[1].set_title('{} Dst Deg'.format(d))

    tot_deg = pt['train_deg']
    x = torch.arange(tot_deg.min(), tot_deg.max(), (tot_deg.max() - tot_deg.min()) / num_bins)
    tot_deg = torch.histc(tot_deg.float(), bins=num_bins)
    axs[2].plot(x[:tot_deg.shape[0]], tot_deg.int())
    axs[2].set_title('{} Total Deg'.format(d))

fig.tight_layout()
plt.savefig('train_deg.png', dpi=300)