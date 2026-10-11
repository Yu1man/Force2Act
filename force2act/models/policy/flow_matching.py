import torch

def flow_matching_loss(
    model,
    actions,
    images,
    state,
    action_mask,
    force = None,
):
    if actions.ndim != 3:
        raise ValueError(f"Actions has {actions.ndim} dimensions, expected 3")

    if action_mask.shape != actions.shape[:2]:
        raise ValueError(f"Action mask has feature dimension of {action_mask.shape}, expected {actions.shape[:2]}")

    if action_mask.dtype != torch.bool:
        raise ValueError(f"Action mask has dtype of {action_mask.dtype}, expected {torch.bool}")

    noise = torch.randn_like(actions)   # [B, T, D]

    batch = actions.shape[0]

    t = torch.rand(batch, device=actions.device, dtype=actions.dtype)   # [B]

    x_t = noise * (1 - t[:, None, None]) + actions * t[:, None, None]

    target_velocity = actions - noise

    pred_velocity = model(
        x_t,
        t,
        images,
        state,
        force
    )

    squared_error = (pred_velocity - target_velocity) ** 2  # [B, T, D]

    action_mask = action_mask.to(actions.device)
    valid_error = squared_error[action_mask]

    if valid_error.numel() == 0:
        raise ValueError("action_mask contains no valid actions")

    loss = valid_error.mean()

    return loss
