import torch
import torch.nn as nn

import timm

from forcefm.models.components.embeddings import LearnedSpatialPositionalEmbedding

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




if __name__ == "__main__":
    images = torch.randn(4, 2, 3, 3, 224, 224)

    encoder = VisionEncoder(256, "resnet18", True, True, 2, 3, 16, 16)
    output = encoder(images)


    print(output.shape)
    print(encoder.backbone.num_features)