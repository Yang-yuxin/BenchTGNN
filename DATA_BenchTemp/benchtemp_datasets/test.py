import numpy as np
import csv

rows = []
for dataset in ['mooc', 'lastfm', 'uci', 'CollegeMsg', 'Flights', 'UNtrade', 'UNvote', 'USLegis',
'CanParl', 'Contacts', 'DGraphFin', 'enron', 'SocialEvo', 'taobao','TaobaoLarge', 'wikipedia', 
'YoutubeRedditLarge', 'YoutubeReddit']:
    edges = np.load(f'{dataset}/ml_{dataset}.npy')
    print(dataset)
    print(f"\t", f'edge num = {edges.shape[0]}')
    has_efeat = ~(np.sum(edges, 1) == 0).all()
    print(f"\t", f'has_efeat = {has_efeat}')
    if has_efeat:
        print(f"\t", f'efeat size = {edges.shape[1]}')
    nodes = np.load(f'{dataset}/ml_{dataset}_node.npy')
    print(f"\t", f'node num = {nodes.shape[0]}')
    has_nfeat = ~(np.sum(abs(nodes), 1) == 0).all()
    print(f"\t", f'has_nfeat = {has_nfeat}')
    if has_nfeat:
        print(f"\t", f'nfeat size = {nodes.shape[1]}')
    row = [dataset, edges.shape[0], nodes.shape[0],]
    if has_efeat:
        row.append(edges.shape[1])
    else:
        row.append(0)
    if has_nfeat:
        row.append(nodes.shape[1])
    else:
        row.append(0)
    rows.append(row)

file_name = 'statistics.csv'
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write each row to the CSV
    for row in rows:
        writer.writerow(row)
