import torch
import torch.nn as nn

import timm

from force2act.models.components.embeddings import LearnedSpatialPositionalEmbedding, LearnedPositionalEmbedding

# Vision Encoder
class VisionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim:int,
        backbone_name: str,
        pretrained: bool,
        freeze_backbone: bool,
        max_cams: int,
        max_history: int,
        max_h: int,
        max_w: int,
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.image_proj = nn.Conv2d(self.backbone.num_features, hidden_dim, 1)

        self.max_cams = max_cams
        self.max_history = max_history

        self.camera_pos_emb = nn.Parameter(torch.empty(1, max_cams, 1, 1, hidden_dim))  # [1, max_cams, 1, 1, hidden_dim]
        self.t_pos_emb = nn.Parameter(torch.empty(1, 1, max_history, 1, hidden_dim))    # [1, 1, max_history, 1, hidden_dim]

        nn.init.normal_(self.camera_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.t_pos_emb, mean=0.0, std=0.02)

        self.spatial_pos_emb = LearnedSpatialPositionalEmbedding(max_h, max_w, hidden_dim)


    def forward(self, images:torch.Tensor):
        """
        Input:  [B, N_cam, T_obs, C, H, W]
        """

        if images.ndim != 6:
            raise ValueError(f"Images ndim is {images.ndim}, expected 6")

        B, N, T, C, H, W = images.shape

        if C != 3:
            raise ValueError(f"Images channel is {images.shape[-3]}, expected 3")

        if N > self.max_cams:
            raise ValueError(f"Number of camera is {N}, max number of camera is {self.max_cams}")

        if T > self.max_history:
            raise ValueError(f"Number of history is {T}, max history is {self.max_history}")

        images = images.reshape(B*N*T, C, H, W)     # [BNT, C, H, W]

        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.forward_features(images) # [BNT, C_feat, H_feat, W_feat]
        else:
            features = self.backbone.forward_features(images) # [BNT, C_feat, H_feat, W_feat]


        features = self.image_proj(features)  # [BNT, hidden_dim, H_feat, W_feat]

        BNT, h_dim, Hf, Wf = features.shape

        features = features.reshape(BNT, h_dim, Hf * Wf)   # [BNT, hidden_dim, Hf*Wf]

        features = features.transpose(-1, -2)   # [BNT, spatial_tokens, D]

        features = features.reshape(B, N, T, Hf*Wf, h_dim)  # [B, N, T, spatial_tokens, D]

        features = features + self.camera_pos_emb[:, :N, :, :, :] + self.t_pos_emb[:, :, :T, :, :] + self.spatial_pos_emb(Hf, Wf)   # [B, N, T, spatial_tokens, D]

        features = features.reshape(B, N * T * Hf * Wf, h_dim)

        return features


    # Override backbone.eval during training
    def train(self, mode=True):
        super().train(mode)

        if self.freeze_backbone:
            self.backbone.eval()

        return self




# State (Proprioceptive) Encoder
class StateEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.state_encoder = nn.Sequential(                  # [B, T_obs, 9]
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state: torch.Tensor):
        """
        input: [B, T_obs, state_dim]
        """
        if state.ndim != 3:
            raise ValueError(f"State has {state.ndim} dimensions, expected 3")

        if state.shape[-1] != self.state_dim:
            raise ValueError(f"State has feature dimension of {state.shape[-1]}, expected {self.state_dim}")

        state_feat = self.state_encoder(state)   # [B, T_obs, hidden_dim]

        return state_feat


# Force Encoder
class ForceEncoder(nn.Module):
    def __init__(
        self,
        force_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.force_dim = force_dim
        self.hidden_dim = hidden_dim

        self.force_encoder = nn.Sequential(
            nn.Conv1d(force_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

    def forward(self, force: torch.Tensor):
        """
        input: [B, T_obs, force_dim]
        """
        if force.ndim != 3:
            raise ValueError(f"Force has  {force.ndim} dimension , expected 3")

        if force.shape[-1] != self.force_dim:
            raise ValueError(f"Force has feature dimension of {force.shape[-1]}, expected {self.force_dim}")

        force = force.transpose(-1, -2) # [B, 3, T_obs]

        force_feat = self.force_encoder(force)

        force_feat = force_feat.transpose(-1, -2)    # [B, T_obs, hidden_dim]

        return force_feat


# Action Encoder
class ActionEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, action: torch.Tensor):
        """
        input: [B, T_action, action_dim]
        """

        if action.ndim != 3:
            raise ValueError(f"Action has  {action.ndim} dimension , expected 3")

        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Action has feature dimension of {action.shape[-1]}, expected {self.action_dim}")

        action_feat = self.action_encoder(action)   # [B, T, D]

        return action_feat




if __name__ == "__main__":
    images = torch.randn(4, 2, 3, 3, 224, 224)

    encoder = VisionEncoder(256, "resnet18", True, True, 2, 3, 16, 16)
    output = encoder(images)


    print(output.shape)
    print(encoder.backbone.num_features)

    state = torch.randn(4, 3, 9)

    encoder = StateEncoder(9, 256)
    output = encoder(state)

    print(output.shape)

    encoder = ForceEncoder(force_dim=3, hidden_dim=256)

    for T in [1, 3, 5]:
        force = torch.randn(4, T, 3)
        output = encoder(force)
        print(T, output.shape)

    action = torch.randn(4, 16, 7)

    encoder = ActionEncoder(
        action_dim=7,
        hidden_dim=256,
    )

    output = encoder(action)

    print(output.shape)