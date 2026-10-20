import torch
import torch.nn as nn

from force2act.models.components.blocks import (
    DiTBlock,
    FinalLayer
)

from force2act.models.components.encoders import (
    ActionEncoder,
    ForceEncoder
)

from force2act.models.components.context import ContextEncoder

from force2act.models.components.embeddings import (
    RandomFourierEmbedding,
    LearnedPositionalEmbedding
)


class ForceAdaLNForce2Act(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,

        backbone_name: str,
        pretrained: bool,
        freeze_backbone: bool,

        action_dim: int,
        state_dim: int,
        force_dim: int,

        fourier_dim: int,
        frequency_scale: float,

        mlp_ratio: int,

        max_cams: int,
        max_h: int,
        max_w: int,

        max_image_history: int,
        max_state_history: int,
        max_force_history: int,

        max_action_horizon: int,

        num_layers: int,

    ):
        super().__init__()

        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        self.action_pos_emb = LearnedPositionalEmbedding(hidden_dim, max_action_horizon)

        self.context_encoder = ContextEncoder(
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            state_dim=state_dim,
            force_dim=None,
            use_force=False,
            max_cams=max_cams,
            max_h=max_h,
            max_w=max_w,
            max_image_history=max_image_history,
            max_state_history=max_state_history,
            max_force_history=max_force_history,
        )

        self.fourier_emb = RandomFourierEmbedding(hidden_dim, fourier_dim, frequency_scale)

        self.force_encoder = ForceEncoder(
            force_dim,
            hidden_dim,
        )

        self.max_force_history = max_force_history

        self.force_pos_emb = LearnedPositionalEmbedding(
            hidden_dim,
            max_force_history,
        )

        self.force_condition_proj = nn.Sequential(
            nn.LayerNorm(max_force_history * hidden_dim),
            nn.Linear(max_force_history * hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim,
                    num_heads,
                    dropout,
                    mlp_ratio,
                )


                for _ in range(num_layers)
            ]
        )

        self.final_layer = FinalLayer(hidden_dim, action_dim)

    def forward(self, noisy_action: torch.Tensor,  time: torch.Tensor, images: torch.Tensor, state: torch.Tensor, force: torch.Tensor | None = None) -> torch.Tensor:
        if force is None:
            raise ValueError(
                "ForceAdaLNForce2Act requires force input"
            )

        if force.shape[1] != self.max_force_history:
            raise ValueError(
                f"Expected force history {self.max_force_history}, "
                f"got {force.shape[1]}"
            )

        action_emb = self.action_pos_emb(self.action_encoder(noisy_action))

        context_emb, _= self.context_encoder(images, state, None)

        time_emb = self.fourier_emb(time)

        force_emb = self.force_pos_emb(self.force_encoder(force))
        force_emb = self.force_condition_proj(force_emb.flatten(1))

        condition_input = torch.cat([time_emb, force_emb], dim=-1)

        modulate_emb = self.condition_fusion(condition_input)

        for block in self.blocks:
            action_emb = block(
                action_emb,
                context_emb,
                modulate_emb,
            )

        output = self.final_layer(action_emb, modulate_emb)

        return output

