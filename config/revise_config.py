import ruamel.yaml
import os

yaml = ruamel.yaml.YAML()
for root, dirs, files in os.walk(".", topdown=False):
    for file in files:
        # print(os.path.join(root,file))
        if 'scan' in root and 'x' in root:
            neigh = root.split('_')[1].split('x')
            neigh = [int(_) for _ in neigh]
            config_path = os.path.join(root,file)
            config = yaml.load(open(config_path, 'r'))
            # import pdb; pdb.set_trace()
            if 'save_every' in config['train'][0].keys():
                del config['train'][0]['save_every']
            config['train'][0]['early_stop'] = 5
            if 'embed' in file:
                config['train'][0]['order'] = 'uniform_random'
            if config['scope'][0]['layer'] == 1:
                config['scope'][0]['layer'] = 2
                config['scope'][0]['neighbor'] = neigh
            config['gnn'][0]['layer'] = 2
            with open(config_path, 'w') as outfile:
                yaml.dump(config, outfile)