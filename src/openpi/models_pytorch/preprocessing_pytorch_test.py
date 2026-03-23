import torch

from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch


class DummyObservation:
    def __init__(self, images, image_masks, state):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = None
        self.tokenized_prompt_mask = None
        self.token_ar_mask = None
        self.token_loss_mask = None


def test_preprocess_observation_pytorch_converts_nhwc_to_nchw() -> None:
    batch_size = 2
    images = {
        "base_0_rgb": torch.randn(batch_size, 224, 224, 3),
        "left_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3),
        "right_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3),
    }
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool) for key in images}
    observation = DummyObservation(images=images, image_masks=image_masks, state=torch.randn(batch_size, 32))

    processed = preprocess_observation_pytorch(observation, train=False)

    assert processed.images["base_0_rgb"].shape == (batch_size, 3, 224, 224)
    assert processed.images["left_wrist_0_rgb"].shape == (batch_size, 3, 224, 224)
    assert processed.images["right_wrist_0_rgb"].shape == (batch_size, 3, 224, 224)