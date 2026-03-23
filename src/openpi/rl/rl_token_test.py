import torch
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.rl.rl_token import RLTokenAutoencoder
from openpi.rl.rl_token import RLTokenConfig
from openpi.rl.rl_token import reconstruction_loss


def test_rl_token_autoencoder_shapes() -> None:
    model = RLTokenAutoencoder(RLTokenConfig(hidden_dim=16, rl_token_dim=8, max_seq_len=12, num_heads=4))
    prefix_tokens = torch.randn(2, 6, 16)

    rl_token, recon = model(prefix_tokens)

    assert rl_token.shape == (2, 8)
    assert recon.shape == (2, 6, 16)


def test_reconstruction_loss_respects_mask() -> None:
    target = torch.tensor([[[1.0], [10.0]], [[5.0], [8.0]]])
    recon = torch.tensor([[[3.0], [999.0]], [[123.0], [6.0]]])
    mask = torch.tensor([[True, False], [False, True]])

    loss = reconstruction_loss(target, recon, mask)

    assert torch.isclose(loss, torch.tensor(4.0))


def test_extract_prefix_features_runs_prefix_branch_only() -> None:
    class DummyBackbone:
        def forward(self, **kwargs):
            prefix_embs, suffix_embs = kwargs["inputs_embeds"]
            assert suffix_embs is None
            return (prefix_embs + 1.0, None), None

    model = object.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model._preprocess_observation = lambda observation, train=False: (
        [torch.zeros(2, 3, 4, 4)],
        [torch.ones(2, dtype=torch.bool)],
        torch.zeros(2, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 4),
    )
    prefix_embs = torch.randn(2, 5, 7)
    prefix_pad_masks = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    prefix_att_masks = torch.zeros(2, 5, dtype=torch.bool)
    model.embed_prefix = lambda images, img_masks, lang_tokens, lang_masks: (
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
    )
    model._prepare_attention_masks_4d = lambda att_masks: att_masks[:, None, :, :].to(torch.float32)
    model.paligemma_with_expert = DummyBackbone()

    features = model.extract_prefix_features(object(), train=False)

    assert torch.equal(features.tokens, prefix_embs + 1.0)
    assert torch.equal(features.pad_masks, prefix_pad_masks)
    assert torch.equal(features.att_masks, prefix_att_masks)