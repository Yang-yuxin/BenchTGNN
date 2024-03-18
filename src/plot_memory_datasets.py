import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import pickle

# Set the font globally
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.family'] = 'Times New Roman'


parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='trial name')
parser.add_argument('--log_dir', type=str, default='log', help='log file directory')
parser.add_argument('--pkl_path', type=str, default='')
parser.add_argument('--target', type=str, default='mrr', choices=['mrr', 'epoch'])
parser.add_argument('--num_scope', type=int, default=25, help='trial name')
parser.add_argument('--num_neighbor', type=int, default=10, help='trial name')
parser.add_argument('--runs', type=int, default=5, help='trial name')
parser.add_argument('--fontsize', type=int, default=32, help='font size')
parser.add_argument('--no_title', action='store_true')
parser.add_argument('--save_legends', action='store_true')

args = parser.parse_args()
log_dir = args.log_dir
config_dir = 'config' + '/{}'.format(args.trial)
# Optionally, you can set the font size as well
plt.rcParams['font.size'] = args.fontsize
scans = ['5', '10', '20', '50', '100']
datasets = ['WIKI', 'REDDIT', 'Flights', 'LASTFM', 'mooc', 'uci', 'CollegeMsg']
show_datasets = {
    'WIKI': 'Wikipedia',
    'REDDIT': 'REDDIT',
    'Flights': 'Flights',
    'LASTFM': 'LASTFM',
    'mooc': 'MOOC',
    'uci': 'UCI',
    'CollegeMsg': 'CollegeMsg'
}
aggrs = ['TGAT', 'GraphMixer']
show_aggrs = {
    'TGAT': 'atten',
    'GraphMixer': 'mixer'
}
samplings = ['re', 'uni',]
show_samplings = {
    're': 'MR',
    'uni': 'uni'
}
memorys = ['gru', 'embed', '']
show_memorys = {
    'gru': 'RNN',
    'embed': 'emb',
    '': 'None'
}
# configs = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
# green_colors = ['#98FB98', '#DAF7A6', '#2FB617', '#6B8E23', '#008080', '#00441b']  # Shades of green
# red_colors = ['#FF9999', '#FFC300', '#fb6a4a', '#cb181d', '#990000', '#581845']  # Shades of red
# red_colors = ['#FF5733', 
#             '#FFC300',
#             '#DAF7A6',
#             '#C70039',
#             '#900C3F',
#             ]

# colors = [
# '#ff8a65',
# '#ffd54f',
# '#aed581',
# '#4db6ac',
# '#4fc3f7',
# '#7986cb'
# ]

colors = [
    '#e74c3c',
    '#e67e22',
    '#f1c40f',
    '#2ecc71',
    '#3498db',
    '#008080',
    '#34495e',
    '#000080'
]
            

all_data = {}

# load data
if os.path.exists(args.pkl_path):
    with open(args.pkl_path, 'rb') as f:
        all_data = pickle.load(f)
else:
    raise NotImplementedError

fig, ax=plt.subplots(1, 1, figsize=(14, 8))
handles = []
# ax = axs[0]
for i, dataset in enumerate(datasets):
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    df_all = pd.DataFrame()
    aggr = 'TGAT'
    sampling = 're'
    points = [list(), list()]
    for sampling in samplings:
        for aggr in aggrs:    
            for scan in scans:
                points[0].append(all_data[dataset][scan][aggr][sampling]['gru'])
                points[1].append(all_data[dataset][scan][aggr][sampling]['embed'])
                if len(points[0][-1]) == args.runs:
                    points[0][-1] = np.mean(points[0][-1])
                    points[1][-1] =  np.mean(points[1][-1])
                    lbl = f'{show_datasets[dataset]}' 
                    format= 'o'
                else:
                    print(dataset, scan, aggr)
    # import pdb; pdb.set_trace()
    handles.append(ax.scatter(x=points[0], y=points[1], s=150, marker=format,label=lbl, color=colors[i]))

# title_str = f'{show_samplings[sampling]}-*-{show_aggrs[aggr]}'
title_str = f'1-layer, 5-100'
if not args.no_title:
    ax.set_title(title_str, x=0.5, y=1.05)
ax.set_xlabel(f'MRR (RNN)')
ax.set_xlim(0, 1)
ax.set_ylabel(f'MRR (emb)')
ax.set_ylim(0, 1)
ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes,color='r')
plt.legend(frameon=False, bbox_to_anchor=(1.6, 0.5), loc='center right', borderaxespad=0, handlelength=0., fontsize='small')
plt.tight_layout()
plt.savefig(f'figures/mrr_memory_strategy.pdf',bbox_inches='tight')

if args.save_legends:
    # Step 1: Create dummy figure and axes
    fig, ax = plt.subplots()
    # Step 2: Define your handles and labels
    
    # handles = []
    # for i,col in enumerate(df_mean.columns):
    #     handles.append(mpatches.Patch(color=colors[i], label=col))
        
    # Step 3: Create the legend
    legend = ax.legend(handles=handles, loc='center', ncol=1)

    # Step 4: Remove the axes
    ax.axis('off')
    
    # Step 5: Save the legend to a PDF file
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('figures/legend_2.pdf',bbox_inches='tight')
