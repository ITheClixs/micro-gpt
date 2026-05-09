"""A compact decoder-only Transformer for micro-GPT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MicroGPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def apply_rope(x):
    batch, heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)
    )
    angles = torch.einsum("t,d->td", positions, inv_freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x, return_attention=False):
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        y = self.proj(y)
        if return_attention:
            return y, attention
        return y


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = 8 * ((hidden_dim + 7) // 8)
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        if input_ids.size(1) > self.config.block_size:
            raise ValueError("Input sequence length exceeds block_size.")
        x = self.drop(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        return MicroGPTOutput(logits=logits, loss=loss, hidden_states=hidden)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size:]
            logits = self(context).logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(1, sorted_indices, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        if was_training:
            self.train()
        return input_ids

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def attention_probe(model, input_ids):
    with torch.no_grad():
        x = model.token_embedding(input_ids)
        block = model.blocks[0]
        _, attention = block.attn(block.attn_norm(x), return_attention=True)
    return attention.mean(dim=(0, 1)).cpu()
