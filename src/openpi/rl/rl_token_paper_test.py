import dataclasses
import pathlib

import pytest
import safetensors.torch
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.rl.image_embeddings import extract_image_embeddings
from openpi.rl.rl_token import RLTokenAutoencoder
from openpi.rl.rl_token import RLTokenConfig
from openpi.rl.rl_token import reconstruction_loss
import openpi.training.config as _config


class _SimpleObservation:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.mark.manual
def test_rl_token_matches_paper_with_real_pi05_checkpoint() -> None:
    model_dir = pathlib.Path("/home/xspark-ai/project/openpi/checkpoint/pi05/pytorch/rtc_pi05")
    config = _config.get_config("pi05_full_base")
    backbone = PI0Pytorch(dataclasses.replace(config.model, dtype="float32", pytorch_compile_mode=None))
    safetensors.torch.load_model(backbone, model_dir / "model.safetensors", device="cpu")
    backbone.eval()

    batch_size = 1
    observation = _SimpleObservation(
        images={
            "base_0_rgb": torch.randn(batch_size, 224, 224, 3),
            "left_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3),
            "right_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        },
        state=torch.randn(batch_size, config.model.action_dim),
        tokenized_prompt=torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32),
        tokenized_prompt_mask=torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )

    with torch.no_grad():
        image_features = extract_image_embeddings(backbone, observation, train=False)

    assert image_features.tokens.shape == (1, 768, 2048)
    assert image_features.pad_masks.shape == (1, 768)

    rl_model = RLTokenAutoencoder(
        RLTokenConfig(
            hidden_dim=2048,
            rl_token_dim=2048,
            max_seq_len=768,
            encoder_layers=2,
            decoder_layers=2,
            num_heads=8,
        )
    )
    rl_token, recon = rl_model(image_features.tokens.to(torch.float32))
    loss = reconstruction_loss(image_features.tokens.to(torch.float32), recon, image_features.pad_masks)

    assert rl_token.shape == (1, 2048)
    assert recon.shape == (1, 768, 2048)
    assert torch.isfinite(loss)