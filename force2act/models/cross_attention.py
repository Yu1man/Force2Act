import torch
import torch.nn as nn

from force2act.models.components.blocks import (
    DiTBlock,
    FinalLayer
)

from force2act.models.components.encoders import ActionEncoder
from force2act.models.components.context import ContextEncoder

from force2act.models.components.embeddings import (
    RandomFourierEmbedding,
    LearnedPositionalEmbedding
)

from force2act.models.policy.flow_matching import flow_matching_loss, sample_action


class CrossAttentionForce2Act(nn.Module):
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
        force_dim: int | None ,

        fourier_dim: int,
        frequency_scale: float,

        use_force: bool,

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

        self.use_force = use_force

        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        self.action_pos_emb = LearnedPositionalEmbedding(hidden_dim, max_action_horizon)

        self.context_encoder = ContextEncoder(
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            state_dim=state_dim,
            force_dim=force_dim,
            use_force=use_force,
            max_cams=max_cams,
            max_h=max_h,
            max_w=max_w,
            max_image_history=max_image_history,
            max_state_history=max_state_history,
            max_force_history=max_force_history,
        )

        self.fourier_emb = RandomFourierEmbedding(hidden_dim, fourier_dim, frequency_scale)

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
        action_emb = self.action_pos_emb(self.action_encoder(noisy_action))

        context_emb = self.context_encoder(images, state, force)

        time_emb = self.fourier_emb(time)

        for block in self.blocks:
            action_emb = block(
                action_emb,
                context_emb,
                time_emb,
            )

        output = self.final_layer(action_emb, time_emb)

        return output


if __name__ == "__main__":
    B = 4

    noisy_action = torch.randn(B, 16, 7)
    time = torch.rand(B)

    images = torch.randn(B, 2, 3, 3, 224, 224)
    state = torch.randn(B, 3, 9)
    force = torch.randn(B, 3, 3)

    force_model = CrossAttentionForce2Act(
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,

        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=True,

        action_dim=7,
        state_dim=9,
        force_dim=3,

        fourier_dim=128,
        frequency_scale=1.0,

        use_force=True,

        mlp_ratio=4,

        max_cams=2,
        max_h=16,
        max_w=16,

        max_image_history=3,
        max_state_history=3,
        max_force_history=3,

        max_action_horizon=16,

        num_layers=2,
    )

    output = force_model(
        noisy_action,
        time,
        images,
        state,
        force,
    )

    print("Force model output shape:")
    print(output.shape)

    print("Force model zero output at init:")
    print(torch.allclose(output, torch.zeros_like(output)))



    no_force_model = CrossAttentionForce2Act(
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,

        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=True,

        action_dim=7,
        state_dim=9,
        force_dim=None,

        fourier_dim=128,
        frequency_scale=1.0,

        use_force=False,

        mlp_ratio=4,

        max_cams=2,
        max_h=16,
        max_w=16,

        max_image_history=3,
        max_state_history=3,
        max_force_history=3,

        max_action_horizon=16,

        num_layers=2,
    )

    output_no_force = no_force_model(
        noisy_action,
        time,
        images,
        state,
        None,
    )

    print("No-force model output shape:")
    print(output_no_force.shape)

    print("No-force model zero output at init:")
    print(
        torch.allclose(
            output_no_force,
            torch.zeros_like(output_no_force)
        )
    )



    try:
        no_force_model(
            noisy_action,
            time,
            images,
            state,
            force,
        )

        print("ERROR: no-force model accepted force input")

    except ValueError as e:
        print("Strict validation works:")
        print(e)

    B = 4

    actions = torch.randn(B, 16, 7)
    images = torch.randn(B, 2, 3, 3, 224, 224)
    state = torch.randn(B, 3, 9)
    force = torch.randn(B, 3, 3)

    action_mask = torch.ones(
        B,
        16,
        dtype=torch.bool,
    )

    loss = flow_matching_loss(
        model=force_model,
        actions=actions,
        images=images,
        state=state,
        action_mask=action_mask,
        force=force,
    )

    print("Loss:", loss.item())
    print("Finite:", torch.isfinite(loss).item())

    loss.backward()

    print("Backward succeeded")

    optimizer = torch.optim.AdamW(
        force_model.parameters(),
        lr=1e-4,
    )

    for step in range(2):
        optimizer.zero_grad()

        loss = flow_matching_loss(
            model=force_model,
            actions=actions,
            images=images,
            state=state,
            action_mask=action_mask,
            force=force,
        )

        loss.backward()

        print(f"\nStep {step + 1}")
        print("Loss:", loss.item())

        # Final output layer
        print(
            "Final projection grad:",
            force_model.final_layer.output_proj.weight.grad.abs().mean().item()
        )

        # Action encoder
        action_grad = force_model.action_encoder.action_encoder[0].weight.grad

        print(
            "Action encoder grad:",
            None if action_grad is None
            else action_grad.abs().mean().item()
        )

        # First DiT block AdaLN
        dit_grad = (
            force_model.blocks[0]
            .adaln_modulation[-1]
            .weight.grad
        )

        print(
            "DiT AdaLN grad:",
            None if dit_grad is None
            else dit_grad.abs().mean().item()
        )

        optimizer.step()

        force_model.eval()

    sampled_actions = sample_action(
        model=force_model,
        images=images,
        state=state,
        action_horizon=16,
        action_dim=7,
        num_steps=20,
        force=force,
    )

    print(sampled_actions.shape)
    print(torch.isfinite(sampled_actions).all())



