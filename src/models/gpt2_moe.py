import torch
from torch import nn, Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2Model
from .transformer import BackboneModel


class MoEGPT2(BackboneModel):
    """
    GPT2 with token-wise MoE in the MLP layers
    """

    def __init__(self,
                 x_dim: int,
                 n_positions: int,
                 y_dim: int = 1,
                 n_embd: int = 64,
                 n_layer: int = 6,
                 n_head: int = 4,
                 n_experts: int = 4,
                 expert_dim: int = 256,
                 **kwargs):

        # GPT2 config
        config = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * n_embd,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False
        )

        gpt2_model = GPT2Model(config)

        # Replace GPT2 MLPs with MoE MLPs
        for block in gpt2_model.h:
            # Use config values instead of trying to infer from Conv1D weights
            in_dim = config.n_embd
            out_dim = config.n_embd

            block.mlp = MoEFeedForward(
                in_dim=in_dim,
                out_dim=out_dim,
                n_experts=n_experts,
                expert_dim=expert_dim
            )

        super().__init__(gpt2_model, x_dim=x_dim, n_positions=n_positions, n_embd=n_embd, y_dim=y_dim, **kwargs)

        self.name = f"moe_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}_experts={n_experts}"


class MoEFeedForward(nn.Module):
    """
    Token-wise MoE block for GPT2 MLP
    """

    def __init__(self, in_dim: int, out_dim: int, n_experts: int = 4, expert_dim: int = 256):
        super().__init__()
        self.n_experts = n_experts

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, out_dim)
            ) for _ in range(n_experts)
        ])

        # Router
        self.router = nn.Linear(in_dim, n_experts)

    # def forward(self, x: Tensor) -> Tensor:
    #     # x shape: [batch, seq_len, hidden]
    #     B, S, H = x.shape
    #     x_flat = x.reshape(-1, H)  # flatten for Linear: [B*S, H]

    #     # Compute router gates
    #     logits = self.router(x_flat)                 # [B*S, n_experts]
    #     gates = torch.softmax(logits, dim=-1)       # [B*S, n_experts]

    #     # Compute expert outputs
    #     expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=-1)  # [B*S, H, E]

    #     # Weighted sum over experts
    #     output_flat = torch.einsum('bh e, b e -> bh', expert_outputs, gates)  # [B*S, H]

    #     # Reshape back to [B, S, H]
    #     output = output_flat.reshape(B, S, H)

    #     return output

    def forward(self, x: Tensor) -> Tensor:
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)  # [B*S, H]
        
        # Router gates
        logits = self.router(x_flat)  # [B*S, n_experts]
        gates = torch.softmax(logits, dim=-1)  # [B*S, n_experts]
        
        # Compute expert outputs - this is fine for dense MoE
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B*S, E, H]
        
        # Weighted combination
        output_flat = torch.einsum('beh,be->bh', expert_outputs, gates)  # [B*S, H]
        
        return output_flat.reshape(B, S, H)
