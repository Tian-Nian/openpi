# RL Token Training

This document describes the paper-aligned RL token training path in this repository.

## Training Target

The paper-aligned setup uses image embeddings as the RL token input.

- Per camera image embedding: `(B, 256, 2048)`
- Three camera views combined: `(B, 768, 2048)`
- RL token encoder output: `(B, 2048)`
- Decoder reconstruction target: `(B, 768, 2048)`

The training script is [scripts/train_rl_token_pytorch.py](scripts/train_rl_token_pytorch.py).

## Data Reading Path

When `--config-name pi05_full_base` is used, the data path is defined in [src/openpi/training/config.py](src/openpi/training/config.py).

The effective pipeline is:

1. `get_config("pi05_full_base")` loads the training config.
2. `config.data.create(...)` builds a `LeRobotAlohaDataConfig`.
3. `create_data_loader(...)` in [src/openpi/training/data_loader.py](src/openpi/training/data_loader.py) creates the PyTorch dataloader.
4. `create_torch_dataset(...)` opens the LeRobot dataset with `repo_id="1118"`.
5. `repack_transforms` remap raw fields into openpi fields:
   - `observation.images.cam_high`
   - `observation.images.cam_left_wrist`
   - `observation.images.cam_right_wrist`
   - `observation.state`
   - `action`
   - `prompt`
6. `data_transforms` apply Aloha preprocessing and optional delta-action conversion.
7. `Normalize(...)` applies normalization stats for `state` and `actions`.
8. `model_transforms` tokenize prompt text and convert the sample into the model input structure.

The RL token training script does not consume actions for supervision. It only reuses the dataloader to obtain observations, then extracts image embeddings from the frozen pi05 backbone.

## Norm Stats

If the normalization statistics for `pi05_full_base` are not present yet, compute them first:

```bash
cd /home/xspark-ai/project/merge/RIL/openpi
./.venv/bin/python scripts/compute_norm_stats.py pi05_full_base
```

If you are only doing a debug smoke run and intentionally want to skip normalization stats, pass `--skip-norm-stats` to the RL token training script.

## Real Training Command

Use the real pi05 PyTorch checkpoint and the real data config:

```bash
cd /home/xspark-ai/project/merge/RIL/openpi
./.venv/bin/python scripts/train_rl_token_pytorch.py \
  --config-name pi05_full_base \
  --exp-name rl_token_pi05_image \
  --feature-source image_embeddings \
  --pretrained-model-path /home/xspark-ai/project/openpi/checkpoint/pi05/pytorch/rtc_pi05 \
  --device cuda \
  --batch-size 8 \
  --num-train-steps 10000 \
  --log-interval 20 \
  --save-interval 500
```

Notes:

1. `--feature-source image_embeddings` is the paper-aligned mode and is now the default.
2. `--pretrained-model-path` should point to the directory containing `model.safetensors`.
3. Outputs are saved under `./checkpoints/rl_token/<config-name>/<exp-name>/<step>/`.

## Debug Smoke Command

To verify the training loop without the real dataset:

```bash
cd /home/xspark-ai/project/merge/RIL/openpi
./.venv/bin/python scripts/train_rl_token_pytorch.py \
  --config-name debug \
  --exp-name rl_token_debug_image \
  --device cpu \
  --feature-source image_embeddings \
  --num-train-steps 1 \
  --log-interval 1 \
  --save-interval 1 \
  --skip-norm-stats
```

## Checkpoint Contents

Each saved step contains:

- `model.safetensors`: RL token autoencoder weights
- `optimizer.pt`: optimizer state
- `metadata.json`: training arguments and RL token config

The frozen pi05 backbone is not saved in these RL token checkpoints.