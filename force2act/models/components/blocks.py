import torch
import torch.nn as nn

# Cross-Attention
class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        input:
            action  [B, T_action, D]
            context [B, N_context, D]

        output:
            refined_action [B, T, D]
        """

        if action.ndim != 3:
            raise ValueError(f"Action has {action.ndim} dimensions, expected 3")

        if context.ndim != 3:
            raise ValueError(f"Context has {context.ndim} dimensions, expected 3")

        if action.shape[-1] != self.hidden_dim:
            raise ValueError(f"Action has feature dimension of {action.shape[-1]}, expected {self.hidden_dim}")

        if context.shape[-1] != self.hidden_dim:
            raise ValueError(f"Context has feature dimension of {context.shape[-1]}, expected {self.hidden_dim}")

        output, _ = self.cross_attn(
            query=action,
            key=context,
            value=context,
            need_weights=False
        )

        return output



# Self-Attention
class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        input:
            action [B, T_action, D]

        output:
            refined_action [B, T, D]
        """

        if action.ndim != 3:
            raise ValueError(f"Action has {action.ndim} dimensions, expected 3")

        if action.shape[-1] != self.hidden_dim:
            raise ValueError(f"Action has feature dimension of {action.shape[-1]}, expected {self.hidden_dim}")

        output, _ = self.self_attn(
            query = action,
            key = action,
            value = action,
            need_weights = False
        )

        return output

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        mlp_ratio: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        input:
            action [B, T_action, D]

        output:
            refined_action [B, T, D]
        """

        if action.ndim != 3:
            raise ValueError(f"Action has {action.ndim} dimensions, expected 3")

        if action.shape[-1] != self.hidden_dim:
            raise ValueError(f"Action has feature dimension of {action.shape[-1]}, expected {self.hidden_dim}")

        output = self.feedforward(action)

        return output


# Modulate
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """
    x       [B, T ,D]
    shift   [B, D]
    scale   [B, D]
    """
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]



# DiT Block
class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout)
        self.self_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.ffn = FeedForward(hidden_dim, dropout, mlp_ratio)
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 9)
        )

        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)



    def forward(self, action: torch.Tensor, context: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:

        (cross_shift, cross_scale, cross_gate,
        self_shift, self_scale, self_gate,
        ffn_shift, ffn_scale, ffn_gate) = self.adaln_modulation(time_emb).chunk(9, dim=-1)

        action_cross = modulate(self.cross_norm(action), cross_shift, cross_scale)
        action_cross = self.cross_attn(action_cross, context)
        action_cross = action + action_cross * cross_gate[:, None, :]

        action_self = modulate(self.self_norm(action_cross), self_shift, self_scale)
        action_self = self.self_attn(action_self)
        action_self = action_cross + action_self * self_gate[:, None, :]

        action_ffn = modulate(self.ffn_norm(action_self), ffn_shift, ffn_scale)
        action_ffn = self.ffn(action_ffn)
        action_ffn = action_self + action_ffn * ffn_gate[:, None, :]

        return action_ffn


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

        self.output_proj = nn.Linear(hidden_dim, action_dim)

        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, action: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(time_emb).chunk(2, dim=-1)

        output = modulate(self.norm(action), shift, scale)

        output = self.output_proj(output)

        return output












if __name__ == "__main__":
    action_tokens = torch.randn(4, 16, 256)
    context_tokens = torch.randn(4, 300, 256)

    attention = CrossAttention(
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,
    )

    output = attention(action_tokens, context_tokens)

    print(output.shape)

    action_tokens = torch.randn(4, 16, 256)

    attn = SelfAttention(
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,
    )

    output = attn(action_tokens)

    print(output.shape)

    action = torch.randn(4, 16, 256)

    ffn = FeedForward(
        hidden_dim=256,
        dropout=0.0,
    )

    output = ffn(action)
    print(output.shape)

    x = torch.randn(4, 16, 256)
    shift = torch.zeros(4, 256)
    scale = torch.zeros(4, 256)

    y = modulate(x, shift, scale)

    print(torch.allclose(x, y))

    action = torch.randn(4, 16, 256)
    context = torch.randn(4, 300, 256)
    time_embedding = torch.randn(4, 256)

    block = DiTBlock(
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,
    )

    output = block(
        action,
        context,
        time_embedding
    )

    print(output.shape)
    print(torch.allclose(output, action))

    action = torch.randn(4, 16, 256)
    time_emb = torch.randn(4, 256)

    layer = FinalLayer(
        hidden_dim=256,
        action_dim=7,
    )

    output = layer(action, time_emb)

    print(output.shape)
    print(torch.allclose(output, torch.zeros_like(output)))




