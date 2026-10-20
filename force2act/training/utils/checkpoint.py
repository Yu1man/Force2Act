from pathlib import Path

import torch

from force2act.data.normalization import ModalityStats, StandardNormalizer


def normalizer_state_dict(normalizer):
    state = {}
    for name, stats in normalizer.stats.items():
        state[name] = {'mean': stats.mean.detach().cpu(), 'std': stats.std.detach().cpu()}
    return state


def normalizer_from_state_dict(state):
    stats = {}
    for name, values in state.items():
        stats[name] = ModalityStats(mean=values['mean'], std=values['std'])
    return StandardNormalizer(stats)


def save_checkpoint(path, model, optimizer, normalizer, config, step, best_val_loss, wandb_run_id=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'normalizer': normalizer_state_dict(normalizer), 'config': config, 'step': int(step), 'best_val_loss': float(best_val_loss), 'wandb_run_id': wandb_run_id}, path)
