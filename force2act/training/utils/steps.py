from itertools import islice

import torch
from tqdm.auto import tqdm

from force2act.models.policy.flow_matching import flow_matching_loss
from force2act.training.utils.batch import prepare_batch


def train_step(model, batch, normalizer, optimizer, device, use_force, grad_clip_norm=None):
    model.train()
    processed_batch = prepare_batch(batch=batch, normalizer=normalizer, device=device, use_force=use_force)
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_loss(model=model, actions=processed_batch['actions'], images=processed_batch['images'], state=processed_batch['state'], force=processed_batch['force'], action_mask=processed_batch['action_mask'])
    if not torch.isfinite(loss):
        raise RuntimeError(f'Non-finite training loss: {loss.item()}')
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    return loss.item()


def validate(model, val_loader, normalizer, device, use_force, max_batches=None):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_batches = 0
    if max_batches is None:
        iterator = val_loader
        total = len(val_loader)
    else:
        max_batches = min(int(max_batches), len(val_loader))
        iterator = islice(val_loader, max_batches)
        total = max_batches
    try:
        with torch.no_grad():
            for batch in tqdm(iterator, total=total, desc='Validation', leave=False, dynamic_ncols=True):
                processed_batch = prepare_batch(batch=batch, normalizer=normalizer, device=device, use_force=use_force)
                loss = flow_matching_loss(model=model, actions=processed_batch['actions'], images=processed_batch['images'], state=processed_batch['state'], force=processed_batch['force'], action_mask=processed_batch['action_mask'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite validation loss.')
                total_loss += loss.item()
                num_batches += 1
    finally:
        model.train(was_training)
    if num_batches == 0:
        raise RuntimeError('Validation processed zero batches.')
    return total_loss / num_batches


def infinite_loader(loader):
    while True:
        yield from loader
