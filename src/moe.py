import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

class MoEMLP(nn.Module):
    """
    MoE-based Feedforward layer replacing GPT2MLP.
    Supports configurable number of experts and top-k routing.
    """
    def __init__(self, n_embd, hidden_dim=None, num_experts=4, top_k=1):
        super().__init__()

        hidden_dim = hidden_dim or 4 * n_embd
        self.num_experts = num_experts
        self.top_k = top_k

        # experts: each is a standard MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_embd),
            ) for _ in range(num_experts)
        ])

        # gating network
        self.gate = nn.Linear(n_embd, num_experts)
        self.register_buffer("routing_counts", torch.zeros(num_experts)) #used to record routing statistics when evaluating

    def forward(self, x):
        """
        x: [batch, seq, n_embd]
        """
        B, S, D = x.shape

        # gating
        gate_logits = self.gate(x)                  # [B, S, num_experts]
        topk_vals, topk_idx = torch.topk(
            gate_logits, k=self.top_k, dim=-1
        )                                           # both [B, S, top_k]
        
        # ---- Routing Statistics (Eval only) ----
        if not self.training:
            with torch.no_grad():
                flat_idx = topk_idx.reshape(-1)
                bincount = torch.bincount(flat_idx, minlength=self.num_experts)
                self.routing_counts += bincount.float()


        gate_weights = torch.softmax(topk_vals, dim=-1)  # normalized routing weights

        outputs = torch.zeros_like(x)

        # top-k MoE dispatch
        for k in range(self.top_k):
            expert_idx = topk_idx[..., k]    # [B, S]
            weight = gate_weights[..., k].unsqueeze(-1)  # [B, S, 1]

            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id).unsqueeze(-1)  # [B, S, 1]
                if mask.sum() == 0:
                    continue

                expert_output = self.experts[expert_id](x)      # [B, S, D]

                outputs += weight * expert_output * mask

        return outputs




class CustomGPT2Model(GPT2Model):
    """
    GPT2Model with MoE-based MLP in every transformer block.
    """
    def __init__(self, config, num_experts=12, top_k=1):
        super().__init__(config)

        n_embd = config.n_embd
        hidden_dim = 4 * n_embd

        # change mlp to MoE MLP
        for block in self.h:
            block.mlp = MoEMLP(
                n_embd=n_embd,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
            )

        self.num_experts = num_experts
        self.top_k = top_k

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
            top_k=top_k,
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
