import ruamel.yaml
import os

yaml = ruamel.yaml.YAML()
for root, dirs, files in os.walk(".", topdown=False):
    for file in files:
        # print(os.path.join(root,file))
        if 'scan' in root:
            config_path = os.path.join(root,file)
            config = yaml.load(open(config_path, 'r'))
            if 'save_every' in config['train'][0].keys():
                del config['train'][0]['save_every']
            config['train'][0]['early_stop'] = 5
            if 'uni' in file:
                config['train'][0]['order'] = 'uniform_random'
            with open(config_path, 'w') as outfile:
                yaml.dump(config, outfile)