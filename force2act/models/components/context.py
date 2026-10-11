import torch
import torch.nn as nn

from force2act.models.components.encoders import (
    VisionEncoder,
    StateEncoder,
    ForceEncoder
)

from force2act.models.components.embeddings import LearnedPositionalEmbedding

# Context Encoder
class ContextEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        backbone_name: str,
        pretrained: bool,
        freeze_backbone: bool,

        state_dim:int,
        force_dim: int | None ,

        use_force: bool,

        max_cams: int,
        max_h: int,
        max_w: int,

        max_image_history: int,
        max_state_history: int,
        max_force_history: int,

    ):
        super().__init__()

        self.vision_encoder = VisionEncoder(hidden_dim, backbone_name, pretrained, freeze_backbone, max_cams, max_image_history, max_h, max_w)

        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.state_pos = LearnedPositionalEmbedding(hidden_dim, max_state_history)

        self.use_force = use_force

        if self.use_force:
            if force_dim is None:
                raise ValueError("force_dim must be provided when use_force=True")

            self.force_encoder = ForceEncoder(force_dim, hidden_dim)
            self.force_pos = LearnedPositionalEmbedding(hidden_dim, max_force_history)

        self.layer_norm = nn.LayerNorm(hidden_dim)


    def forward(self, images: torch.Tensor, state: torch.Tensor, force:torch.Tensor | None = None) -> torch.Tensor:
        """
        inputs:
            images: [B, N_cam, T_img, C, H, W]
            state:  [B, T_state, state_dim]
            force:  [B, T_force, force_dim] or None

        output:
            context: [B, N_context, hidden_dim]
        """
        if self.use_force and force is None:
            raise ValueError("force must be provided when use_force=True")

        if not self.use_force and force is not None:
            raise ValueError(f"Use_force has to be enabled for force use")


        image_feat = self.vision_encoder(images)

        state_feat = self.state_encoder(state)
        state_feat = self.state_pos(state_feat)

        if self.use_force:
            force_feat = self.force_encoder(force)
            force_feat = self.force_pos(force_feat)

            context = torch.cat([image_feat, state_feat, force_feat], dim = 1)  # [B, image state force token, hidden_dimension]

        else:
            context = torch.cat([image_feat, state_feat], dim = 1)

        context = self.layer_norm(context)

        return context