import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

class MoEMLP(nn.Module):
    """
    Dense / Soft MoE FFN
    - 所有 token 同时通过所有 experts
    - 使用 softmax(gate(x)) 加权求和
    - 完全等价于你代码 1 的 MoEFeedForward
    """

    def __init__(self, in_dim, out_dim, n_experts=4, expert_dim=256):
        super().__init__()

        self.n_experts = n_experts

        # Experts: List of MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, out_dim)
            )
            for _ in range(n_experts)
        ])

        # Router: produce softmax weights for all experts
        self.router = nn.Linear(in_dim, n_experts)

    def forward(self, x):
        # x: [B, S, H]
        B, S, H = x.shape
        x_flat = x.reshape(B * S, H)     # [B*S, H]

        # router logits & softmax
        logits = self.router(x_flat)     # [B*S, E]
        gates = torch.softmax(logits, dim=-1)  # [B*S, E]

        # expert outputs: stack → [B*S, E, H]
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.experts],
            dim=1
        )

        # weighted sum via einsum: gate · expert_outputs
        output = torch.einsum("beh,be -> bh", expert_outputs, gates)

        return output.reshape(B, S, H)


class CustomGPT2Model(GPT2Model):
    def __init__(self, config, num_experts=4, expert_dim=256):
        super().__init__(config)

        in_dim = config.n_embd
        out_dim = config.n_embd

        for block in self.h:
            block.mlp = MoEMLP(
                in_dim=in_dim,
                out_dim=out_dim,
                n_experts=num_experts,
                expert_dim=expert_dim
            )

class TransformerModel_var_moe(nn.Module):
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        n_layer=12,
        n_head=4,
        num_experts=4,
        top_k=1,
    ):
        super(TransformerModel_var_moe, self).__init__()

        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.name = f"gpt2_moe_embd={n_embd}_layer={n_layer}_head={n_head}_experts={num_experts}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)

        # change backbone to MoE GPT2
        self._backbone = CustomGPT2Model(
            configuration,
            num_experts=num_experts,
        )

        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]
