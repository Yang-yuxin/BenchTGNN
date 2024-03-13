import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import openpyxl
import pickle

# Set the font globally
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.family'] = 'Times New Roman'
# Optionally, you can set the font size as well
plt.rcParams['font.size'] = 12

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, help='trial name')
parser.add_argument('--log_dir', type=str, default='log', help='log file directory')
parser.add_argument('--pkl_path', type=str, default='')
parser.add_argument('--target', type=str, default='mrr', choices=['mrr', 'epoch'])
parser.add_argument('--num_scope', type=int, default=25, help='trial name')
parser.add_argument('--num_neighbor', type=int, default=10, help='trial name')
parser.add_argument('--runs', type=int, default=5, help='trial name')
args = parser.parse_args()
log_dir = args.log_dir
config_dir = 'config' + '/{}'.format(args.trial)


scans = ['5', '10', '20', '50', '100']
# scans = ['5', '10', '20', '50', '100', '5x5', '5x10', '10x5', '10x10']
datasets = ['WIKI', 'REDDIT', 'Flights', 'LASTFM', 'mooc', 'uci', 'CollegeMsg']
aggrs = ['TGAT', 'GraphMixer']
show_aggrs = {
    'TGAT': 'Attention',
    'GraphMixer': 'MLP-Mixer'
}
samplings = ['re', 'uni',]
memorys = ['gru', 'embed', '']
show_memorys = {
    'gru': 'RNN',
    'embed': 'Embedding',
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
colors = [
'#ff8a65',
'#ffd54f',
'#aed581',
'#4db6ac',
'#4fc3f7',
'#7986cb'
]


            

def get_test_mrr(file_path):
    try:
        with open(file_path, 'r') as f:
            mrr = None
            for lines in f:
                if lines.startswith('\ttest AP'):
                    mrr = float(lines.strip('\n').split(':')[-1])
                    return mrr
    except FileNotFoundError:
        import pdb; pdb.set_trace()
        pass

def get_best_epoch(file_path):
    try:
        with open(file_path, 'r') as f:
            epoch = None
            for lines in f:
                if lines.startswith('Loading'):
                    epoch = int(lines.split(' ')[4])
                    return epoch
    except FileNotFoundError:
        import pdb; pdb.set_trace()
        pass

all_data = {}

# load data
if os.path.exists(args.pkl_path):
    with open(args.pkl_path, 'rb') as f:
        all_data = pickle.load(f)
else:
    raise NotImplementedError

for dataset in datasets:
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    df_all = pd.DataFrame()
    for aggr in aggrs:
        for memory in memorys:
            means = []
            stds = []
            for scan in scans:
                results = np.array(all_data[dataset][scan][aggr]['re'][memory]) - np.array(all_data[dataset][scan][aggr]['uni'][memory])
                if len(results) == args.runs:
                    # print(f"{dataset}\t{scan}\t{aggr}\t{sampling}\t{memory}\t{np.mean(results)}")
                    means.append(np.mean(results))
                    stds.append(np.std(results))
                else:
                    import pdb; pdb.set_trace()
            if len(means) == len(scans):
                df_mean[f"{show_aggrs[aggr]}+{show_memorys[memory]}"] = means
                df_std[f"{show_aggrs[aggr]}+{show_memorys[memory]}"] = stds
                df_all[f"{show_aggrs[aggr]}+{show_memorys[memory]}"] = [(str(round(mean, 4)) + '+-' + str(round(std, 4))) for (mean, std) in zip(means, stds)]
    if df_mean.empty:
        continue
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(df_mean.columns):
        # plt.plot(np.arange(len(scans)), df_mean[col], label=col, color=colors[i])
        try:
            plt.errorbar(x=range(len(scans)), y=df_mean[col], yerr=df_std[col], fmt='-o', capsize=5, label=col, color=colors[i])
        except IndexError:
            import pdb; pdb.set_trace()
    if dataset == 'mooc':
        plt.title(f'{dataset.upper()}')
    else:
        plt.title(f'{dataset}')
    x_labels = scans
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
    plt.legend()
    df_all = df_all.T
    df_all.columns = scans
    
    plt.savefig(f'figures/{dataset}_mrr_spl_strategy.png')

