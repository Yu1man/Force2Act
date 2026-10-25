import argparse

import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hdf5-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None)
    return parser.parse_args()


def apply_cli_overrides(config, args):
    training_cfg = config.setdefault('training', {})
    dataset_cfg = config.setdefault('dataset', {})
    wandb_cfg = config.setdefault('wandb', {})
    if args.max_steps is not None:
        training_cfg['max_steps'] = args.max_steps
    if args.batch_size is not None:
        training_cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        training_cfg['learning_rate'] = args.lr
    if args.seed is not None:
        training_cfg['seed'] = args.seed
    if args.output_dir is not None:
        training_cfg['output_dir'] = args.output_dir
    if args.hdf5_path is not None:
        dataset_cfg['hdf5_path'] = args.hdf5_path
    if args.wandb_name is not None:
        wandb_cfg['name'] = args.wandb_name
    if args.no_wandb:
        wandb_cfg['enabled'] = False
    return config
