import benchtemp as bt
import torch
import os
import os.path as osp
import numpy as np
import argparse
import pdb
from utils import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name', default='WIKI')
    parser.add_argument('--root_path', type=str, help='dataset root path', default='DATA/')
    args = parser.parse_args()
    if args.data in ['Flight', 'GDELT', 'MovieLens', 'REDDIT', 'sGDELT', 'WIKI']:
        g, edges, nfeat, efeat = load_data(args.data, args.root_path)
        if efeat is not None and efeat.dtype == torch.bool:
            efeat = efeat.to(torch.int8)
        if nfeat is not None and nfeat.dtype == torch.bool:
            nfeat = nfeat.to(torch.int8)
        dim_edge_feat = efeat.shape[-1] if efeat is not None else 0
        dim_node_feat = nfeat.shape[-1] if nfeat is not None else 0
        
    elif args.data in ['MOOC', 'lastfm', 'UNtrade', 'USLegis']:
        data = bt.lp.DataLoader(dataset_path=osp.join(args.root_path, args.data) + '/', dataset_name=args.data)
        node_features, edge_features, full_data, train_data, val_data, test_data, \
        new_node_val_data, new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, new_new_node_test_data, unseen_nodes_num = data.load()
        dim_edge_feat = edge_features.shape[-1] if edge_features is not None else 0
        dim_node_feat = node_features.shape[-1] if node_features is not None else 0
        # pdb.set_trace()
    print('Node feature dimension is {}'.format(dim_node_feat))
    print('Edge feature dimension is {}'.format(dim_edge_feat))
    # pdb.set_trace()
    