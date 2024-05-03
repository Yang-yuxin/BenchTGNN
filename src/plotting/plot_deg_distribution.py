import torch
import matplotlib.pyplot as plt

data = ['LASTFM', 'mooc', 'WIKI', 'REDDIT', 'Flights', 'CollegeMsg', 'uci']
num_bins = 100

fig, axss = plt.subplots(len(data), 1, figsize=(15, len(data) *5))
for d, axs in zip(data, axss):
    pt = torch.load('DATA/{}/edges.pt'.format(d))
    import pdb; pdb.set_trace()
    src_deg = pt['train_src']
    x = torch.arange(src_deg.min(), src_deg.max(), (src_deg.max() - src_deg.min()) / num_bins)
    src_deg = torch.histc(src_deg.float(), bins=num_bins)
    axs.plot(x[:src_deg.shape[0]], src_deg.int(), color='b')
    axs.set_title('{} Src Deg'.format(d))

    dst_deg = pt['train_dst']
    x = torch.arange(dst_deg.min(), dst_deg.max(), (dst_deg.max() - dst_deg.min()) / num_bins)
    dst_deg = torch.histc(dst_deg.float(), bins=num_bins)
    axs.plot(x[:dst_deg.shape[0]], dst_deg.int(), color='r')
    axs.set_title('{} Dst Deg'.format(d))

    print(d, dst_deg.shape[0])
    # tot_deg = pt['train_deg']
    # x = torch.arange(tot_deg.min(), tot_deg.max(), (tot_deg.max() - tot_deg.mins()) / num_bins)
    # tot_deg = torch.histc(tot_deg.float(), bins=num_bins)
    # axs.plot(x[:tot_deg.shape[0]], tot_deg.int())
    # axs.set_title('{} Total Deg'.format(d))

fig.tight_layout()
plt.savefig(f'datasets_train_deg.png', dpi=300)