import os
import torch
import pandas as pd

def extract_config_details(config):
    """Extract specific details from the nested config dictionary."""
    details = {}
    # Extracting specific keys from the config dictionary
    details['train_order'] = config.get('train', [{}])[0].get('order', None)
    details['train_epoch'] = config.get('train', [{}])[0].get('epoch', None)
    details['train_batch_size'] = config.get('train', [{}])[0].get('batch_size', None)
    details['train_lr'] = config.get('train', [{}])[0].get('lr', None)
    details['train_dropout'] = config.get('train', [{}])[0].get('dropout', None)
    details['train_early_stop'] = config.get('train', [{}])[0].get('early_stop', None)

    details['eval_batch_size'] = config.get('eval', [{}])[0].get('batch_size', None)
    details['eval_val_epoch'] = config.get('eval', [{}])[0].get('val_epoch', None)

    details['scope_layer'] = config.get('scope', [{}])[0].get('layer', None)
    details['scope_neighbor'] = config.get('scope', [{}])[0].get('neighbor', None)
    details['scope_strategy'] = config.get('scope', [{}])[0].get('strategy', None)

    details['gnn_arch'] = config.get('gnn', [{}])[0].get('arch', None)
    details['gnn_layer'] = config.get('gnn', [{}])[0].get('layer', None)
    details['gnn_att_head'] = config.get('gnn', [{}])[0].get('att_head', None)
    details['gnn_time_enc'] = config.get('gnn', [{}])[0].get('time_enc', None)
    details['gnn_dim_time'] = config.get('gnn', [{}])[0].get('dim_time', None)
    details['gnn_dim_out'] = config.get('gnn', [{}])[0].get('dim_out', None)
    details['gnn_memory_type'] = config.get('gnn', [{}])[0].get('memory_type', None)
    details['gnn_init_trick'] = config.get('gnn', [{}])[0].get('init_trick', None)
    details['gnn_dim_memory'] = config.get('gnn', [{}])[0].get('dim_memory', None)
    details['gnn_msg_reducer'] = config.get('gnn', [{}])[0].get('msg_reducer', None)
    details['gnn_memory_update_use_embed'] = config.get('gnn', [{}])[0].get('memory_update_use_embed', None)

    return details

def get_saved_configs(models_dir):
    config_data = []

    for root, dirs, files in os.walk(models_dir):
        if 'best.pkl' in files:
            pkl_path = os.path.join(root, 'best.pkl')
            try:
                saved_data = torch.load(pkl_path, map_location=torch.device('cpu'))
            except Exception as e:
                if (type(e) == EOFError):    # the exception type
                    print('EOF at:', pkl_path)
                elif (type(e) == RuntimeError):
                    print('Runtime at:', pkl_path)
                else:
                    print(type(e))
                    import pdb; pdb.set_trace()
                continue
            config = saved_data.get('config', None)
            if config is not None:
                # Extract the relevant details from the config dictionary
                config_details = extract_config_details(config)
                dataset_name = root.split('_')[0].split('/')[-1]
                if dataset_name not in ['WIKI', 'REDDIT', 'mooc', 'LASTFM', 'GDELT', 'superuser', 'uci', 'CollegeMsg', 'Flights']:
                    continue
                config_details['dataset'] = dataset_name
                # Add the directory name as a key in the dictionary
                config_details['model_directory'] = os.path.basename(root)
                config_data.append(config_details)

    return config_data

def save_configs_to_excel(config_data, output_file):
    # Create a DataFrame from the list of config detail dictionaries
    df = pd.DataFrame(config_data)
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)

# Example usage
models_dir = 'models/'
output_file = 'model_configs_detailed.xlsx'
config_data = get_saved_configs(models_dir)
save_configs_to_excel(config_data, output_file)
