import torch
import torch.nn as nn

# Image Spatial Positional Embedding
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


    def forward(self, height, width) -> torch.Tensor:
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


# Positional Embedding
class LearnedPositionalEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_length: int,
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.pos_emb = nn.Parameter(torch.empty(1, max_length, hidden_dim))    # [1, max_length, hidden_dim]
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, feat:torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError(f"Feature has dimension of {feat.ndim}, expected 3")

        B, T, D = feat.shape

        if D != self.hidden_dim:
            raise ValueError(f"Feature has hidden dimension of {D}, expected {self.hidden_dim}")

        if T > self.max_length:
            raise ValueError(f"Feature has length of {T}, exceeded max length {self.max_length}")

        feat_pos = feat + self.pos_emb[:, :T, :]    # [B, T, D]

        return feat_pos


# Time Embedding
class RandomFourierEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        fourier_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.fourier_dim = fourier_dim
        self.scale = scale

        if fourier_dim % 2 != 0:
            raise ValueError(f"Fourier dimension must be even, got {fourier_dim}")

        freq_dim = fourier_dim // 2

        frequencies = torch.randn(freq_dim) * scale

        self.register_buffer("frequencies", frequencies)

        self.time_mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        input: [B]
        output: [B, hidden_dim]
        """

        if time.ndim != 1:
            raise ValueError(f"Time has {time.ndim} dimensions, expected 1")

        angles = time[:, None] * self.frequencies[None, :]   # [B, freq_dim]

        sin_feat = torch.sin(angles) # [B, sin_freq_dim]
        cos_feat = torch.cos(angles) # [B, cos_freq_dim]

        fourier_feat = torch.cat([sin_feat, cos_feat], dim=-1) # [B, fourier_dim]

        fourier_feat = self.time_mlp(fourier_feat)  # [B, hidden_dim]

        return fourier_feat


if __name__ == "__main__":
    t = torch.rand(4)

    embedding = RandomFourierEmbedding(
        hidden_dim=256,
        fourier_dim=128,
    )

    output = embedding(t)

    print(t.shape)
    print(output.shape)






