import torch
import torch.nn as nn

class LearnedSpatialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_h: int,
        max_w: int,
        hidden_dim: int
    ):
        super().__init__()

        self.max_h = max_h
        self.max_w = max_w

        self.row_embedding = nn.Parameter(torch.empty([1, 1, 1, max_h, 1, hidden_dim]))
        self.column_embedding = nn.Parameter(torch.empty([1, 1, 1, 1, max_w, hidden_dim]))

        nn.init.normal_(self.row_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.column_embedding, mean=0.0, std=0.02)


    def forward(self, height, width):
        if height > self.max_h:
            raise ValueError(f"Height {height} exceeded max height {self.max_h}")

        if width > self.max_w:
            raise ValueError(f"Width {width} exceeded max width {self.max_w}")

        row_pos_emb = self.row_embedding[:, :, :, :height, :, :]    # [1, 1, 1, H, 1, hidden_dim]
        column_pos_emb = self.column_embedding[:, :, :, :, :width, :]   # [1, 1, 1, 1, W, hidden_dim]

        spatial_pos_emb = row_pos_emb + column_pos_emb # [1, 1, 1, H, W, hidden_dim]

        B, N, T, H, W, h_dim = spatial_pos_emb.shape

        spatial_pos_emb = spatial_pos_emb.reshape(B, N, T, H*W, h_dim)  # [1, 1, 1, H*W, hidden_dim]

        return spatial_pos_emb


