import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .utils import autocast_ctx


class FourBranchGatedMIL(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc_go = nn.Linear(cfg.go_dim, cfg.hidden_dim)
        self.enc_tf = nn.Linear(cfg.tf_dim, cfg.hidden_dim)
        self.enc_kegg = nn.Linear(cfg.kegg_dim, cfg.hidden_dim)
        self.enc_react = nn.Linear(cfg.reactome_dim, cfg.hidden_dim)

        self.ln_go = nn.LayerNorm(cfg.hidden_dim)
        self.ln_tf = nn.LayerNorm(cfg.hidden_dim)
        self.ln_kegg = nn.LayerNorm(cfg.hidden_dim)
        self.ln_react = nn.LayerNorm(cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.gate = nn.Sequential(
            nn.Linear(4 * cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.hidden_dim, 4),
        )

        self.att_v = nn.Linear(cfg.hidden_dim, cfg.num_heads * cfg.att_dim)
        self.att_u = nn.Linear(cfg.hidden_dim, cfg.num_heads * cfg.att_dim)

        self.att_w = nn.Parameter(torch.empty(cfg.num_heads, cfg.att_dim))
        self.att_b = nn.Parameter(torch.zeros(cfg.num_heads))
        nn.init.xavier_uniform_(self.att_w)

        self.head_gate_w = nn.Parameter(torch.empty(cfg.num_heads, cfg.hidden_dim))
        self.head_gate_b = nn.Parameter(torch.zeros(cfg.num_heads))
        nn.init.xavier_uniform_(self.head_gate_w)

        self.classifier = nn.Linear(3 * cfg.hidden_dim, cfg.num_classes)

    def branch_embed(self, x: torch.Tensor, enc: nn.Linear, ln: nn.LayerNorm) -> torch.Tensor:
        h = F.relu(enc(x))
        h = ln(h)
        h = self.drop(h)
        return h

    def encode_cells(
        self,
        go: torch.Tensor,
        tf: torch.Tensor,
        kegg: torch.Tensor,
        reactome: torch.Tensor,
    ) -> torch.Tensor:
        h_go = self.branch_embed(go, self.enc_go, self.ln_go)
        h_tf = self.branch_embed(tf, self.enc_tf, self.ln_tf)
        h_kegg = self.branch_embed(kegg, self.enc_kegg, self.ln_kegg)
        h_react = self.branch_embed(reactome, self.enc_react, self.ln_react)

        h_cat = torch.cat([h_go, h_tf, h_kegg, h_react], dim=1)
        gate_logits = self.gate(h_cat)
        alpha = torch.softmax(gate_logits, dim=1)

        h = (
            alpha[:, 0:1] * h_go
            + alpha[:, 1:2] * h_tf
            + alpha[:, 2:3] * h_kegg
            + alpha[:, 3:4] * h_react
        )
        return h

    def attention_logits(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        heads = self.cfg.num_heads
        d = self.cfg.att_dim

        v = self.att_v(h).view(n, heads, d)
        u = self.att_u(h).view(n, heads, d)

        a_v = torch.tanh(v)
        a_u = torch.sigmoid(u)
        g = a_v * a_u

        logits = (g * self.att_w.unsqueeze(0)).sum(dim=-1) + self.att_b.unsqueeze(0)
        return logits

    def combine_heads(self, z_heads: torch.Tensor) -> torch.Tensor:
        scores = (z_heads * self.head_gate_w).sum(dim=-1) + self.head_gate_b
        beta = torch.softmax(scores, dim=0)
        z_att = (beta.unsqueeze(1) * z_heads).sum(dim=0)
        return z_att


def forward_bag_all_cells(
    model: FourBranchGatedMIL,
    go_cpu: torch.Tensor,
    tf_cpu: torch.Tensor,
    kegg_cpu: torch.Tensor,
    reactome_cpu: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
):
    with autocast_ctx(device, enabled=use_amp):
        if int(chunk_size) <= 0:
            go = go_cpu.to(device, non_blocking=True)
            tf = tf_cpu.to(device, non_blocking=True)
            kegg = kegg_cpu.to(device, non_blocking=True)
            reactome = reactome_cpu.to(device, non_blocking=True)

            h = model.encode_cells(go, tf, kegg, reactome)
            attn_logits = model.attention_logits(h)
            w_heads = torch.softmax(attn_logits, dim=0)

            z_heads = (w_heads.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=0)
            z_att = model.combine_heads(z_heads)
            z_mean = torch.mean(h, dim=0)
            z_std = torch.std(h, dim=0, unbiased=False)
            z = torch.cat([z_att, z_mean, z_std], dim=0)

            logits = model.classifier(z)
            w_mean = w_heads.mean(dim=1)
            return logits, w_mean

        n = int(go_cpu.shape[0])
        hs = []
        attn_parts = []
        for s in range(0, n, int(chunk_size)):
            e = min(s + int(chunk_size), n)
            go = go_cpu[s:e].to(device, non_blocking=True)
            tf = tf_cpu[s:e].to(device, non_blocking=True)
            kegg = kegg_cpu[s:e].to(device, non_blocking=True)
            reactome = reactome_cpu[s:e].to(device, non_blocking=True)

            h = model.encode_cells(go, tf, kegg, reactome)
            attn_logits = model.attention_logits(h)
            hs.append(h)
            attn_parts.append(attn_logits)

        h_all = torch.cat(hs, dim=0)
        attn_all = torch.cat(attn_parts, dim=0)

        w_heads = torch.softmax(attn_all, dim=0)
        z_heads = (w_heads.unsqueeze(-1) * h_all.unsqueeze(1)).sum(dim=0)
        z_att = model.combine_heads(z_heads)
        z_mean = torch.mean(h_all, dim=0)
        z_std = torch.std(h_all, dim=0, unbiased=False)
        z = torch.cat([z_att, z_mean, z_std], dim=0)

        logits = model.classifier(z)
        w_mean = w_heads.mean(dim=1)
        return logits, w_mean
