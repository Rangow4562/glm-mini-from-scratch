"""A small GLM-style language model built from scratch.

Three ingredients, at educational scale:

1. Hybrid attention in a ``linear, linear, linear, sparse`` rhythm. Linear
   layers use O(length) positive-feature attention with a prefix-sum state;
   every fourth layer gathers a local window plus strided anchor tokens and
   runs true softmax attention over only those sources.
2. A top-k routed mixture of experts plus one always-on shared expert in
   every block, dispatched with a readable per-expert loop.
3. Several residual streams per layer with learned input mixing and output
   routing, a simplified stand-in for manifold-constrained hyper-connections.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260
    dim: int = 192
    layers: int = 12
    heads: int = 6
    expert_hidden: int = 384
    experts: int = 8
    top_k: int = 2
    streams: int = 4
    sparse_window: int = 32
    sparse_stride: int = 32
    max_sequence_length: int = 192

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def apply_rotary(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to [batch, time, heads, head_dim]."""
    if q.shape[-1] % 2:
        raise ValueError("rotary embeddings require an even head dimension")
    length, head_dim = q.shape[1], q.shape[-1]
    positions = torch.arange(length, device=q.device, dtype=torch.float32)
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, :, None, :].to(q.dtype)
    sin = angles.sin()[None, :, None, :].to(q.dtype)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class LinearAttention(nn.Module):
    """Causal linear attention: positive ELU features plus prefix sums, O(length)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        q, k = apply_rotary(q, k)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        v = v.view(batch, length, self.heads, self.head_dim)
        state = torch.einsum("bthd,bthe->bthde", k, v).cumsum(dim=1)
        key_sums = k.cumsum(dim=1)
        numerator = torch.einsum("bthd,bthde->bthe", q, state)
        denominator = torch.einsum("bthd,bthd->bth", q, key_sums).unsqueeze(-1)
        output = numerator / denominator.clamp_min(1e-6)
        return self.out(output.reshape(batch, length, dim))


class SparseAttention(nn.Module):
    """Gathered causal attention over a local window plus strided anchor tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.window = config.sparse_window
        self.stride = config.sparse_stride
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def _gather_plan(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        plan: list[list[int]] = []
        for position in range(length):
            anchors = range(0, position + 1, self.stride)
            local = range(max(0, position - self.window + 1), position + 1)
            plan.append(sorted(set(anchors) | set(local)))
        width = max(map(len, plan))
        indices = torch.zeros(length, width, dtype=torch.long, device=device)
        valid = torch.zeros(length, width, dtype=torch.bool, device=device)
        for row, sources in enumerate(plan):
            indices[row, : len(sources)] = torch.tensor(sources, device=device)
            valid[row, : len(sources)] = True
        return indices, valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        q, k = apply_rotary(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        indices, valid = self._gather_plan(length, x.device)
        keys = k[:, :, indices, :]
        values = v[:, :, indices, :]
        scores = torch.einsum("bhtd,bhtkd->bhtk", q, keys) * self.head_dim ** -0.5
        scores = scores.masked_fill(~valid[None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        output = torch.einsum("bhtk,bhtkd->bhtd", weights, values)
        return self.out(output.transpose(1, 2).reshape(batch, length, dim))


class Expert(nn.Module):
    """SwiGLU feedforward with no biases."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class SparseMoE(nn.Module):
    """Top-k routed experts plus one always-on shared expert."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = nn.Linear(config.dim, config.experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(config.dim, config.expert_hidden) for _ in range(config.experts)
        )
        self.shared = Expert(config.dim, config.expert_hidden)
        self.top_k = config.top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        top_values, top_indices = self.router(flat).topk(self.top_k, dim=-1)
        weights = torch.softmax(top_values.float(), dim=-1).to(flat.dtype)
        output = self.shared(flat)
        usage = torch.zeros(len(self.experts), device=x.device, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(top_indices == expert_id)
            if positions.numel() == 0:
                continue
            contribution = expert(flat[positions]) * weights[positions, slots, None].to(output.dtype)
            output.index_add_(0, positions, contribution)
            usage[expert_id] = positions.numel()
        usage /= max(1, flat.shape[0] * self.top_k)
        return output.view(shape), usage


class HybridBlock(nn.Module):
    """Pre-norm block: hybrid attention residual, then routed MoE residual."""

    def __init__(self, config: ModelConfig, sparse: bool):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.moe_norm = RMSNorm(config.dim)
        self.attention = SparseAttention(config) if sparse else LinearAttention(config)
        self.moe = SparseMoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attention_norm(x))
        experts, usage = self.moe(self.moe_norm(x))
        return x + experts, usage


class HyperConnection(nn.Module):
    """Several residual streams: learned mixing into the block, learned routing back."""

    def __init__(self, config: ModelConfig, block: HybridBlock):
        super().__init__()
        self.block = block
        self.input_logits = nn.Parameter(torch.linspace(0.1, -0.1, config.streams))
        self.output_logits = nn.Parameter(torch.linspace(-0.1, 0.1, config.streams))

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = torch.einsum("s,btsd->btd", torch.softmax(self.input_logits, dim=0), streams)
        transformed, usage = self.block(mixed)
        routed = torch.softmax(self.output_logits, dim=0)
        streams = streams + transformed.unsqueeze(2) * routed[None, None, :, None]
        return streams, usage


class GLMMiniFromScratch(nn.Module):
    """The full model: tokens -> streams -> hybrid blocks -> tied LM head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.heads:
            raise ValueError("dim must be divisible by heads")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            HyperConnection(config, HybridBlock(config, sparse=(index + 1) % 4 == 0))
            for index in range(config.layers)
        )
        self.final_norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_embeddings(self, embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network from precomputed embeddings (alternative modalities enter here)."""
        if embedded.ndim != 3 or embedded.shape[-1] != self.config.dim:
            raise ValueError("embedded input must have shape [batch, time, dim]")
        if embedded.shape[1] > self.config.max_sequence_length:
            raise ValueError("input exceeds max_sequence_length")
        streams = embedded.unsqueeze(2).expand(-1, -1, self.config.streams, -1).contiguous()
        usages = []
        for layer in self.layers:
            streams, usage = layer(streams)
            usages.append(usage)
        hidden = self.final_norm(streams.mean(dim=2))
        return self.output(hidden), torch.stack(usages)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embedding(input_ids))

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        routed = sum(
            sum(parameter.numel() for expert in layer.block.moe.experts for parameter in expert.parameters())
            for layer in self.layers
        )
        # Active estimate: everything plus top_k of the routed expert pool per layer.
        active = total - routed + routed * self.config.top_k // self.config.experts
        return {"total": total, "active_per_token_estimate": active}
