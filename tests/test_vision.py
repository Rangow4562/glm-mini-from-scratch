"""Vision path invariants: shapes, token ordering, labels, gradient flow, overfit smoke."""
import torch
import torch.nn.functional as F

from glm_mini import (
    ByteTokenizer,
    GLMMiniFromScratch,
    MiniVisionConfig,
    ModelConfig,
    VisionLanguageModel,
    answer_only_labels,
    build_multimodal_input_ids,
)
from glm_mini.vision import ImageTokenIds, MiniGLMVisionEncoder, RGBPatchEmbedding


def language_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=263, dim=32, layers=2, heads=4, expert_hidden=48,
        experts=4, top_k=2, streams=2, max_sequence_length=96,
    )


def vision_config() -> MiniVisionConfig:
    return MiniVisionConfig(
        image_size=32, patch_size=4, hidden_size=24, depth=2, heads=4,
        intermediate_size=48, spatial_merge_size=2, projection_intermediate_size=64,
    )


def wrapper() -> VisionLanguageModel:
    torch.manual_seed(3)
    return VisionLanguageModel(GLMMiniFromScratch(language_config()), vision_config()).eval()


def test_patch_embedding_shapes():
    encoder = RGBPatchEmbedding(vision_config())
    images = torch.rand(2, 3, 32, 32)
    tokens, grid = encoder(images)
    assert tokens.shape == (2, 64, 24)
    assert grid == (8, 8)


def test_visual_token_count():
    model = wrapper()
    assert model.visual_token_count == 16
    assert model.inserted_token_count == 18


def test_multimodal_input_ids_ordering():
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("Digit: ", bos=True)
    ids = build_multimodal_input_ids(torch.tensor([prompt]), visual_token_count=16)
    assert ids.shape == (1, 8 + 18)
    token_ids = ImageTokenIds()
    assert ids[0, 0].item() == token_ids.bos
    assert ids[0, 1].item() == token_ids.start
    assert ids[0, 2:18].tolist() == [token_ids.placeholder] * 16
    assert ids[0, 18].item() == token_ids.end
    assert ids[0, 19:].tolist() == prompt[1:]


def test_embed_multimodal_replaces_placeholders():
    model = wrapper()
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("Digit: ", bos=True)
    ids = build_multimodal_input_ids(torch.tensor([prompt]), visual_token_count=model.visual_token_count)
    images = torch.rand(1, 3, 32, 32)
    embedded = model.embed_multimodal(images, ids)
    assert embedded.shape == (1, ids.shape[1], 32)
    # Visual positions must differ from the raw placeholder embedding.
    assert not torch.allclose(embedded[0, 2], model.language_model.embedding.weight[261])


def test_forward_shapes_and_labels():
    model = wrapper()
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("Digit: ", bos=True)
    rows = [prompt + tokenizer.encode(str(digit)) for digit in range(2)]
    tokens = torch.tensor(rows, dtype=torch.long)
    images = torch.rand(2, 3, 32, 32)
    logits, usage = model(images, tokens[:, :-1])
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 263
    labels = answer_only_labels(tokens, prompt_length=8, visual_token_count=model.visual_token_count)
    assert labels.shape == logits.shape[:2]
    unmasked = (labels != -100)
    assert unmasked.sum(dim=1).tolist() == [1, 1]


def test_vision_gradients_flow():
    model = wrapper().train()
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("Digit: ", bos=True)
    tokens = torch.tensor([prompt + tokenizer.encode("7")], dtype=torch.long)
    images = torch.rand(1, 3, 32, 32)
    logits, _ = model(images, tokens[:, :-1])
    loss = logits[:, -1].sum()
    loss.backward()
    assert model.vision_encoder.patch_embedding.proj.weight.grad is not None
    assert model.language_model.embedding.weight.grad is not None


def test_encoder_standalone_shapes():
    encoder = MiniGLMVisionEncoder(vision_config(), 32)
    outputs = encoder(torch.rand(3, 3, 32, 32))
    assert outputs.shape == (3, 16, 32)


def test_tiny_two_example_overfit_smoke():
    model = wrapper().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("Digit: ", bos=True)
    tokens = torch.tensor([prompt + tokenizer.encode("7")] * 2, dtype=torch.long)
    images = torch.rand(2, 3, 32, 32)
    first_loss = None
    last_loss = None
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images, tokens[:, :-1])
        targets = answer_only_labels(tokens, prompt_length=8, visual_token_count=model.visual_token_count)
        loss = F.cross_entropy(
            logits.reshape(-1, 263), targets.reshape(-1), ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        first_loss = first_loss if first_loss is not None else float(loss.detach())
        last_loss = float(loss.detach())
    assert last_loss < first_loss
