"""Model invariants: layer rhythm, shapes, causality, MoE usage, parameter counts."""
import pytest
import torch

from glm_mini import ModelConfig
from glm_mini.model import GLMMiniFromScratch, LinearAttention, SparseAttention


def tiny_config() -> ModelConfig:
    return ModelConfig(
        dim=32, heads=4, layers=4, expert_hidden=64,
        experts=4, top_k=2, streams=2, max_sequence_length=64,
    )


def test_sparse_rhythm_every_fourth_layer():
    model = GLMMiniFromScratch(tiny_config())
    for index, layer in enumerate(model.layers):
        expected = SparseAttention if (index + 1) % 4 == 0 else LinearAttention
        assert isinstance(layer.block.attention, expected)


def test_forward_shapes_and_usage_normalization():
    model = GLMMiniFromScratch(tiny_config()).eval()
    inputs = torch.randint(4, 260, (2, 16))
    with torch.no_grad():
        logits, usage = model(inputs)
    assert logits.shape == (2, 16, 260)
    assert usage.shape == (4, 4)
    assert torch.allclose(usage.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_prefix_causality():
    model = GLMMiniFromScratch(tiny_config()).eval()
    torch.manual_seed(0)
    inputs = torch.randint(4, 260, (1, 24))
    with torch.no_grad():
        short_logits, _ = model(inputs[:, :12])
        long_logits, _ = model(inputs)
    assert torch.allclose(short_logits, long_logits[:, :12], atol=1e-4)


def test_embeddings_are_tied():
    model = GLMMiniFromScratch(tiny_config())
    assert model.output.weight is model.embedding.weight


def test_parameter_counts_match_and_active_is_smaller():
    config = tiny_config()
    model = GLMMiniFromScratch(config)
    counts = model.parameter_counts()
    assert counts["total"] == sum(parameter.numel() for parameter in model.parameters())
    routed = sum(
        parameter.numel()
        for layer in model.layers
        for expert in layer.block.moe.experts
        for parameter in expert.parameters()
    )
    expected_active = counts["total"] - routed + routed * config.top_k // config.experts
    assert counts["active_per_token_estimate"] == expected_active
    assert counts["active_per_token_estimate"] < counts["total"]


def test_input_validation():
    model = GLMMiniFromScratch(tiny_config())
    with pytest.raises(ValueError):
        GLMMiniFromScratch(ModelConfig(dim=33, heads=4, layers=1))
    with pytest.raises(ValueError):
        model.forward_embeddings(torch.randn(1, 8, 16))
    with pytest.raises(ValueError):
        model.forward_embeddings(torch.randn(1, 65, 32))
