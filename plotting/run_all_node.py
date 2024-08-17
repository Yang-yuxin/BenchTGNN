import os
import pandas as pd
import yaml
import argparse

def flatten_config(config):
    """Flatten the nested config dictionary for easier comparison."""
    flattened_config = {}
    flattened_config['train_order'] = config.get('train', [{}])[0].get('order', None)
    flattened_config['train_epoch'] = config.get('train', [{}])[0].get('epoch', None)
    flattened_config['train_batch_size'] = config.get('train', [{}])[0].get('batch_size', None)
    flattened_config['train_lr'] = config.get('train', [{}])[0].get('lr', None)
    flattened_config['train_dropout'] = config.get('train', [{}])[0].get('dropout', None)
    flattened_config['train_early_stop'] = config.get('train', [{}])[0].get('early_stop', None)

    flattened_config['eval_batch_size'] = config.get('eval', [{}])[0].get('batch_size', None)
    flattened_config['eval_val_epoch'] = config.get('eval', [{}])[0].get('val_epoch', None)

    flattened_config['scope_layer'] = config.get('scope', [{}])[0].get('layer', None)
    flattened_config['scope_neighbor'] = config.get('scope', [{}])[0].get('neighbor', None)
    flattened_config['scope_strategy'] = config.get('scope', [{}])[0].get('strategy', None)

    flattened_config['gnn_arch'] = config.get('gnn', [{}])[0].get('arch', None)
    flattened_config['gnn_layer'] = config.get('gnn', [{}])[0].get('layer', None)
    flattened_config['gnn_att_head'] = config.get('gnn', [{}])[0].get('att_head', None)
    flattened_config['gnn_time_enc'] = config.get('gnn', [{}])[0].get('time_enc', None)
    flattened_config['gnn_dim_time'] = config.get('gnn', [{}])[0].get('dim_time', None)
    flattened_config['gnn_dim_out'] = config.get('gnn', [{}])[0].get('dim_out', None)
    flattened_config['gnn_memory_type'] = config.get('gnn', [{}])[0].get('memory_type', None)
    flattened_config['gnn_init_trick'] = config.get('gnn', [{}])[0].get('init_trick', None)
    flattened_config['gnn_dim_memory'] = config.get('gnn', [{}])[0].get('dim_memory', None)
    flattened_config['gnn_msg_reducer'] = config.get('gnn', [{}])[0].get('msg_reducer', None)
    flattened_config['gnn_memory_update_use_embed'] = config.get('gnn', [{}])[0].get('memory_update_use_embed', None)

    return flattened_config

def find_matching_directory(config, data, excel_file):
    """Find the matching directory based on the provided config."""
    # Load the Excel file
    df = pd.read_excel(excel_file)

    # Flatten the provided config
    flat_config = flatten_config(config)

    # import pdb; pdb.set
    # Compare the flattened config with each row in the DataFrame
    match = df.loc[
        (df['train_order'] == flat_config['train_order']) &
        # (df['train_epoch'] == flat_config['train_epoch']) &
        (df['train_batch_size'] == flat_config['train_batch_size']) &
        # (df['train_lr'] == flat_config['train_lr']) &
        # (df['train_dropout'] == flat_config['train_dropout']) &
        # (df['train_early_stop'] == flat_config['train_early_stop']) &
        # (df['eval_batch_size'] == flat_config['eval_batch_size']) &
        # (df['eval_val_epoch'] == flat_config['eval_val_epoch']) &
        (df['scope_layer'] == flat_config['scope_layer']) &
        (df['scope_neighbor'] == str(flat_config['scope_neighbor'])) &
        (df['scope_strategy'] == flat_config['scope_strategy']) &
        (df['gnn_arch'] == flat_config['gnn_arch']) &
        # (df['gnn_layer'] == flat_config['gnn_layer']) &
        (df['gnn_att_head'] == flat_config['gnn_att_head']) &
        (df['gnn_time_enc'] == flat_config['gnn_time_enc']) &
        (df['gnn_dim_time'] == flat_config['gnn_dim_time']) &
        (df['gnn_dim_out'] == flat_config['gnn_dim_out']) &
        (df['gnn_memory_type'] == flat_config['gnn_memory_type']) &
        # (df['gnn_init_trick'] == flat_config['gnn_init_trick']) &
        (df['gnn_dim_memory'] == flat_config['gnn_dim_memory']) &
        (df['gnn_msg_reducer'] == flat_config['gnn_msg_reducer']) &
        (df['gnn_memory_update_use_embed'] == flat_config['gnn_memory_update_use_embed']) &
        (df['dataset'] == data)
    ]
    

    if not match.empty:
        return match['model_directory'].values[0]
    else:
        return "No such model saved."

def process_config_directory(config_dir, data, excel_file):
    """Process all YAML configuration files in the given directory."""
    results = {}

    for file_name in os.listdir(config_dir):
        if file_name.endswith('.yml') or file_name.endswith('.yaml'):
            config_path = os.path.join(config_dir, file_name)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            directory_name = find_matching_directory(config, data, excel_file)
            results[file_name] = directory_name

    return results

def main():
    parser = argparse.ArgumentParser(description="Find matching directories for config files.")
    parser.add_argument('--config_dir', type=str, required=True, help="Directory containing YAML config files.")
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--excel_file', type=str, required=True, help="Excel file with model configuration data.")
    
    args = parser.parse_args()

    results = process_config_directory(args.config_dir, args.data, args.excel_file)

    for config_file, directory in results.items():
        print(f"Config file: {config_file} -> Directory: {directory}")

if __name__ == "__main__":
    main()
