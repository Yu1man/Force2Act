import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from force2act.data.dataset import EpisodeDataset
from force2act.data.normalization import fit_normalizer
from force2act.models.factory import build_model
from force2act.training.utils.checkpoint import normalizer_from_state_dict, save_checkpoint
from force2act.training.utils.config import save_config
from force2act.training.utils.data import build_dataloaders, load_train_val_episodes
from force2act.training.utils.steps import infinite_loader, train_step, validate
from force2act.training.utils.tracking import init_wandb


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config, args):
    training_cfg = config['training']
    dataset_cfg = config['dataset']
    validation_cfg = config.get('validation', {})
    seed = int(training_cfg.get('seed', 42))
    seed_everything(seed)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable.')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    output_dir = Path(training_cfg.get('output_dir', 'outputs/default'))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / 'config.yaml')
    checkpoint = None
    if args.resume is not None:
        print(f'Loading checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    print('Loading episodes...')
    train_episodes, val_episodes = load_train_val_episodes(config)
    print(f'Train episodes: {len(train_episodes)}')
    print(f'Val episodes: {len(val_episodes)}')
    if checkpoint is not None:
        normalizer = normalizer_from_state_dict(checkpoint['normalizer'])
    else:
        normalizer = fit_normalizer(train_episodes)
    normalizer.to(device)
    history = int(dataset_cfg['history'])
    action_horizon = int(dataset_cfg['action_horizon'])
    train_dataset = EpisodeDataset(train_episodes, history, action_horizon)
    val_dataset = EpisodeDataset(val_episodes, history, action_horizon)
    print(f'Train windows: {len(train_dataset)}')
    print(f'Val windows: {len(val_dataset)}')
    train_loader, val_loader = build_dataloaders(config, train_dataset, val_dataset)
    train_iterator = infinite_loader(train_loader)
    model = build_model(config).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError('Model has no trainable parameters.')
    num_trainable = sum((parameter.numel() for parameter in trainable_parameters))
    print(f'Trainable parameters: {num_trainable:,}')
    learning_rate = float(training_cfg['learning_rate'])
    weight_decay = float(training_cfg.get('weight_decay', 0.0001))
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    max_steps = int(training_cfg['max_steps'])
    log_interval = int(training_cfg.get('log_interval', 20))
    val_interval = int(training_cfg.get('val_interval', 1000))
    checkpoint_interval = int(training_cfg.get('checkpoint_interval', 1000))
    grad_clip_norm = training_cfg.get('grad_clip_norm')
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)
    max_val_batches = validation_cfg.get('max_batches')
    use_force = bool(dataset_cfg.get('use_force', True))
    if log_interval <= 0:
        raise ValueError('log_interval must be > 0')
    if val_interval <= 0:
        raise ValueError('val_interval must be > 0')
    if checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be > 0')
    start_step = 0
    best_val_loss = float('inf')
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = int(checkpoint.get('step', 0))
        best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
        print(f'Resuming from step {start_step}')
    if start_step >= max_steps:
        print('Checkpoint has already reached max_steps.')
        return
    resume_run_id = None
    if checkpoint is not None:
        resume_run_id = checkpoint.get('wandb_run_id')
    run = init_wandb(config=config, output_dir=output_dir, resume_run_id=resume_run_id)
    wandb_run_id = run.id if run is not None else None
    print(f'Training from step {start_step} → {max_steps}')
    print(f'Validation every {val_interval} steps')
    running_loss = 0.0
    running_count = 0
    last_completed_step = start_step
    progress = tqdm(total=max_steps, initial=start_step, desc='Training', dynamic_ncols=True)
    try:
        for step in range(start_step + 1, max_steps + 1):
            batch = next(train_iterator)
            train_loss = train_step(model=model, batch=batch, normalizer=normalizer, optimizer=optimizer, device=device, use_force=use_force, grad_clip_norm=grad_clip_norm)
            last_completed_step = step
            running_loss += train_loss
            running_count += 1
            metrics = {}
            if step % log_interval == 0 or step == max_steps:
                mean_train_loss = running_loss / running_count
                metrics['train/loss'] = mean_train_loss
                metrics['train/learning_rate'] = optimizer.param_groups[0]['lr']
                running_loss = 0.0
                running_count = 0
            validation_due = step % val_interval == 0 or step == max_steps
            if validation_due:
                val_loss = validate(model=model, val_loader=val_loader, normalizer=normalizer, device=device, use_force=use_force, max_batches=max_val_batches)
                metrics['val/loss'] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(path=output_dir / 'best.pt', model=model, optimizer=optimizer, normalizer=normalizer, config=config, step=step, best_val_loss=best_val_loss, wandb_run_id=wandb_run_id)
                    print(f'\nNew best validation loss: {best_val_loss:.6f}')
            if step % checkpoint_interval == 0:
                save_checkpoint(path=output_dir / 'latest.pt', model=model, optimizer=optimizer, normalizer=normalizer, config=config, step=step, best_val_loss=best_val_loss, wandb_run_id=wandb_run_id)
            if run is not None and metrics:
                run.log(metrics, step=step)
            progress.set_postfix(train=f'{train_loss:.4f}', best_val='N/A' if best_val_loss == float('inf') else f'{best_val_loss:.4f}')
            progress.update(1)
    except KeyboardInterrupt:
        print('\nTraining interrupted. Saving latest checkpoint...')
        save_checkpoint(path=output_dir / 'latest.pt', model=model, optimizer=optimizer, normalizer=normalizer, config=config, step=last_completed_step, best_val_loss=best_val_loss, wandb_run_id=wandb_run_id)
        return
    finally:
        progress.close()
        if run is not None:
            run.finish()
    save_checkpoint(path=output_dir / 'final.pt', model=model, optimizer=optimizer, normalizer=normalizer, config=config, step=max_steps, best_val_loss=best_val_loss, wandb_run_id=wandb_run_id)
    save_checkpoint(path=output_dir / 'latest.pt', model=model, optimizer=optimizer, normalizer=normalizer, config=config, step=max_steps, best_val_loss=best_val_loss, wandb_run_id=wandb_run_id)
    print('Training finished.')
    print(f'Best validation loss: ', f'{best_val_loss:.6f}')
