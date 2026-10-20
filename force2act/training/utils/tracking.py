def init_wandb(config, output_dir, resume_run_id=None):
    wandb_cfg = config.get('wandb', {})
    if not wandb_cfg.get('enabled', True):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError('W&B is enabled but wandb is not installed.\nRun: python -m pip install wandb') from exc
    kwargs = {'project': wandb_cfg.get('project', 'Force2Act'), 'name': wandb_cfg.get('name'), 'config': config, 'dir': str(output_dir)}
    entity = wandb_cfg.get('entity')
    if entity:
        kwargs['entity'] = entity
    if resume_run_id is not None:
        kwargs['id'] = resume_run_id
        kwargs['resume'] = 'allow'
    return wandb.init(**kwargs)
