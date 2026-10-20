from force2act.data.image_preprocessing import preprocess_images


def prepare_batch(batch, normalizer, device, use_force=True):
    images = batch['images'].to(device, non_blocking=True)
    images = preprocess_images(images)
    state = batch['state'].to(device, non_blocking=True)
    state = normalizer.normalize(state, 'state')
    actions = batch['actions'].to(device, non_blocking=True)
    actions = normalizer.normalize(actions, 'actions')
    action_mask = batch['action_mask'].to(device, non_blocking=True)
    if use_force:
        force = batch['force'].to(device, non_blocking=True)
        force = normalizer.normalize(force, 'force')
    else:
        force = None
    return {'images': images, 'state': state, 'force': force, 'actions': actions, 'action_mask': action_mask}
