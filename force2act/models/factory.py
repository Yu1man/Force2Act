from force2act.models.cross_attention import CrossAttentionForce2Act

def build_model(config):
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    action_dim = dataset_cfg["action_dim"]

    architecture = model_cfg.get("architecture", "cross_attn")

    history = dataset_cfg["history"]

    image_history = dataset_cfg.get("image_history", history)
    state_history = dataset_cfg.get("state_history", history)
    force_history = dataset_cfg.get("force_history", history)

    use_force = dataset_cfg["use_force"]

    state_dim = model_cfg["state_dim"]

    if use_force:
        force_dim = model_cfg["force_dim"]
    else:
        force_dim = None


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


    if architecture == "cross_attn":
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

            use_force = use_force,

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

if __name__ == "__main__":
    import yaml

    with open(
        "configs/test.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    print(type(model))
    print(model.use_force)