import torch
from torch.utils.data import DataLoader

from force2act.data.dataset import EpisodeDataset
from force2act.data.hdf5 import load_episodes
from force2act.data.split import get_train_val_names


def load_train_val_episodes(config):
    dataset_cfg = config['dataset']
    hdf5_path = dataset_cfg['hdf5_path']
    state_keys = dataset_cfg['state_keys']
    camera_keys = dataset_cfg['camera_keys']
    force_keys = dataset_cfg.get('force_keys')
    train_names, val_names = get_train_val_names(hdf5_path)
    train_episodes = load_episodes(hdf5_path=hdf5_path, demo_names=train_names, state_keys=state_keys, camera_keys=camera_keys, force_keys=force_keys)
    val_episodes = load_episodes(hdf5_path=hdf5_path, demo_names=val_names, state_keys=state_keys, camera_keys=camera_keys, force_keys=force_keys)
    return (train_episodes, val_episodes)


def build_dataloaders(config, train_dataset, val_dataset):
    training_cfg = config['training']
    batch_size = int(training_cfg['batch_size'])
    num_workers = int(training_cfg.get('num_workers', 0))
    seed = int(training_cfg.get('seed', 42))
    generator = torch.Generator()
    generator.manual_seed(seed)
    common_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': torch.cuda.is_available(), 'persistent_workers': num_workers > 0}
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, generator=generator, **common_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_args)
    if len(train_loader) == 0:
        raise RuntimeError('Training DataLoader is empty. Reduce batch_size.')
    if len(val_loader) == 0:
        raise RuntimeError('Validation DataLoader is empty.')
    return (train_loader, val_loader)
