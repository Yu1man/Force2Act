from force2act.models.model.cross_attention import CrossAttentionForce2Act
from force2act.models.model.force_adaln import ForceAdaLNForce2Act
from force2act.models.model.fsra import FSRAForce2Act

def build_model(config):
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    action_dim = dataset_cfg["action_dim"]

    architecture = model_cfg.get("architecture", "cross_attn")

    history = dataset_cfg["history"]

    image_history = dataset_cfg.get("image_history", history)
    state_history = dataset_cfg.get("state_history", history)
    force_history = dataset_cfg.get("force_history", history)

    state_dim = model_cfg["state_dim"]

    max_cams = len(dataset_cfg["camera_keys"])
    max_h = model_cfg["max_feature_height"]
    max_w = model_cfg["max_feature_width"]

    action_horizon = dataset_cfg["action_horizon"]

    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]
    num_heads = model_cfg["num_heads"]


    # Vision Encoder
    backbone_name = model_cfg["backbone_name"]
    pretrained = model_cfg["pretrained"]
    freeze_backbone = model_cfg["freeze_backbone"]

    # Optional model settings
    fourier_dim = model_cfg["fourier_dim"]
    frequency_scale = model_cfg["frequency_scale"]
    dropout = model_cfg["dropout"]
    mlp_ratio = model_cfg["mlp_ratio"]

    dataset_use_force = dataset_cfg["use_force"]

    force_architectures = {
        "cross_attn",
        "force_adaln",
        "fsra",
    }

    if architecture == "no_force":
        model_use_force = False

    elif architecture in force_architectures:
        model_use_force = True

    else:
        raise ValueError(
            f"Unknown model architecture: {architecture}"
        )

    if model_use_force and not dataset_use_force:
        raise ValueError(
            f"{architecture} requires dataset.use_force=True"
        )

    if not model_use_force and dataset_use_force:
        raise ValueError(
            "no_force requires dataset.use_force=False"
        )

    force_dim = (
        model_cfg["force_dim"]
        if model_use_force
        else None
    )


    if architecture == "no_force":
        return CrossAttentionForce2Act(
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            dropout = dropout,

            backbone_name = backbone_name,
            pretrained = pretrained,
            freeze_backbone = freeze_backbone,

            action_dim = action_dim,
            state_dim = state_dim,
            force_dim = force_dim,

            fourier_dim = fourier_dim,
            frequency_scale = frequency_scale,

            use_force = False,

            mlp_ratio = mlp_ratio,

            max_cams = max_cams,
            max_h = max_h,
            max_w = max_w,

            max_image_history = image_history,
            max_state_history = state_history,
            max_force_history = force_history,

            max_action_horizon = action_horizon,

            num_layers = num_layers
        )

    elif architecture == "cross_attn":
        return CrossAttentionForce2Act(
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            dropout = dropout,

            backbone_name = backbone_name,
            pretrained = pretrained,
            freeze_backbone = freeze_backbone,

            action_dim = action_dim,
            state_dim = state_dim,
            force_dim = force_dim,

            fourier_dim = fourier_dim,
            frequency_scale = frequency_scale,

            use_force = True,

            mlp_ratio = mlp_ratio,

            max_cams = max_cams,
            max_h = max_h,
            max_w = max_w,

            max_image_history = image_history,
            max_state_history = state_history,
            max_force_history = force_history,

            max_action_horizon = action_horizon,

            num_layers = num_layers
        )

    elif architecture == "force_adaln":
        return ForceAdaLNForce2Act(
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            dropout = dropout,

            backbone_name = backbone_name,
            pretrained = pretrained,
            freeze_backbone = freeze_backbone,

            action_dim = action_dim,
            state_dim = state_dim,
            force_dim = force_dim,

            fourier_dim = fourier_dim,
            frequency_scale = frequency_scale,

            mlp_ratio = mlp_ratio,

            max_cams = max_cams,
            max_h = max_h,
            max_w = max_w,

            max_image_history = image_history,
            max_state_history = state_history,
            max_force_history = force_history,

            max_action_horizon = action_horizon,

            num_layers = num_layers
        )

    elif architecture == "fsra":
        return FSRAForce2Act(
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            dropout = dropout,

            backbone_name = backbone_name,
            pretrained = pretrained,
            freeze_backbone = freeze_backbone,

            action_dim = action_dim,
            state_dim = state_dim,
            force_dim = force_dim,

            fourier_dim = fourier_dim,
            frequency_scale = frequency_scale,

            mlp_ratio = mlp_ratio,

            max_cams = max_cams,
            max_h = max_h,
            max_w = max_w,

            max_image_history = image_history,
            max_state_history = state_history,
            max_force_history = force_history,

            max_action_horizon = action_horizon,

            num_layers = num_layers
        )

    raise ValueError(f"Unknown model architecture: {architecture}")
