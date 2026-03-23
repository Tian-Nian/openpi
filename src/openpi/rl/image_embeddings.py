import dataclasses

import torch

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


@dataclasses.dataclass(frozen=True)
class ImageEmbeddingFeatures:
    tokens: torch.Tensor
    pad_masks: torch.Tensor


def extract_image_embeddings(model, observation, *, train: bool = False) -> ImageEmbeddingFeatures:
    processed = _preprocessing.preprocess_observation_pytorch(observation, train=train)

    image_embeddings = []
    image_pad_masks = []
    for image, image_mask in zip(processed.images.values(), processed.image_masks.values(), strict=True):
        embedding = model.paligemma_with_expert.embed_image(image)
        batch_size, num_tokens = embedding.shape[:2]
        image_embeddings.append(embedding)
        image_pad_masks.append(image_mask[:, None].expand(batch_size, num_tokens))

    return ImageEmbeddingFeatures(
        tokens=torch.cat(image_embeddings, dim=1),
        pad_masks=torch.cat(image_pad_masks, dim=1),
    )