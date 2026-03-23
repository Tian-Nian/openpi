import dataclasses
import json
import logging
import pathlib
import time
from typing import Literal

import jax
import safetensors.torch
import torch
import tyro

import openpi.models.pi0_config as pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.rl import extract_image_embeddings
from openpi.rl import RLTokenAutoencoder
from openpi.rl import RLTokenConfig
from openpi.rl import reconstruction_loss
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str
    exp_name: str
    device: str = "cuda"
    feature_source: Literal["image_embeddings", "prefix_features"] = "image_embeddings"
    pretrained_model_path: str | None = None
    batch_size: int | None = None
    num_train_steps: int = 1000
    log_interval: int = 50
    save_interval: int = 500
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 8
    ff_dim: int | None = None
    rl_token_dim: int | None = None
    decoder_causal: bool = False
    output_root: str = "./checkpoints/rl_token"
    pytorch_training_precision: str = "float32"
    skip_norm_stats: bool = False


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _build_model_config(config: _config.TrainConfig, precision: str) -> pi0_config.Pi0Config:
    if isinstance(config.model, pi0_config.Pi0Config):
        return dataclasses.replace(config.model, dtype=precision, pytorch_compile_mode=None)
    return pi0_config.Pi0Config(
        dtype=precision,
        action_dim=config.model.action_dim,
        action_horizon=config.model.action_horizon,
        max_token_len=config.model.max_token_len,
        paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
        action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
        pi05=getattr(config.model, "pi05", False),
        pytorch_compile_mode=None,
    )


def _move_observation_to_device(observation, device: torch.device):
    return jax.tree.map(lambda x: x.to(device) if hasattr(x, "to") else x, observation)


def _prepare_run_config(args: Args) -> _config.TrainConfig:
    base_config = _config.get_config(args.config_name)
    batch_size = base_config.batch_size if args.batch_size is None else args.batch_size
    return dataclasses.replace(
        base_config,
        exp_name=args.exp_name,
        batch_size=batch_size,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        wandb_enabled=False,
        pytorch_training_precision=args.pytorch_training_precision,
    )


def _extract_training_features(base_model: PI0Pytorch, observation, feature_source: str):
    if feature_source == "image_embeddings":
        return extract_image_embeddings(base_model, observation, train=False)
    if feature_source == "prefix_features":
        return base_model.extract_prefix_features(observation, train=False)
    raise ValueError(f"Unsupported feature_source: {feature_source}")


def _build_output_dir(args: Args, run_config: _config.TrainConfig) -> pathlib.Path:
    return (pathlib.Path(args.output_root) / run_config.name / args.exp_name).resolve()


def _save_checkpoint(
    output_dir: pathlib.Path,
    step: int,
    model: RLTokenAutoencoder,
    optimizer: torch.optim.Optimizer,
    args: Args,
    rl_config: RLTokenConfig,
) -> None:
    ckpt_dir = output_dir / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, ckpt_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    metadata = {
        "step": step,
        "args": dataclasses.asdict(args),
        "rl_config": dataclasses.asdict(rl_config),
        "timestamp": time.time(),
    }
    with (ckpt_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)


def main() -> None:
    init_logging()
    args = tyro.cli(Args)
    device = _resolve_device(args.device)
    run_config = _prepare_run_config(args)
    output_dir = _build_output_dir(args, run_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = _data.create_data_loader(
        run_config,
        framework="pytorch",
        shuffle=True,
        skip_norm_stats=args.skip_norm_stats,
    )

    base_model = PI0Pytorch(_build_model_config(run_config, args.pytorch_training_precision)).to(device)
    pretrained_model_path = args.pretrained_model_path or run_config.pytorch_weight_path
    if pretrained_model_path is not None:
        model_path = pathlib.Path(pretrained_model_path) / "model.safetensors"
        safetensors.torch.load_model(base_model, model_path, device=str(device))
        logging.info("Loaded frozen backbone weights from %s", model_path)

    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    sample_observation, _ = next(iter(loader))
    sample_observation = _move_observation_to_device(sample_observation, device)
    with torch.no_grad():
        sample_features = _extract_training_features(base_model, sample_observation, args.feature_source)

    rl_config = RLTokenConfig(
        hidden_dim=sample_features.tokens.shape[-1],
        rl_token_dim=args.rl_token_dim,
        max_seq_len=sample_features.tokens.shape[1],
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        decoder_causal=args.decoder_causal,
    )
    rl_model = RLTokenAutoencoder(rl_config).to(device)
    optimizer = torch.optim.AdamW(rl_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logging.info(
        "Starting RL-token training: config=%s exp=%s device=%s feature_source=%s hidden_dim=%s seq_len=%s",
        run_config.name,
        args.exp_name,
        device,
        args.feature_source,
        rl_config.hidden_dim,
        rl_config.max_seq_len,
    )

    step = 0
    start_time = time.time()
    metrics: list[float] = []

    while step < args.num_train_steps:
        for observation, _actions in loader:
            if step >= args.num_train_steps:
                break

            observation = _move_observation_to_device(observation, device)
            with torch.no_grad():
                features = _extract_training_features(base_model, observation, args.feature_source)
                tokens = features.tokens.to(torch.float32)

            rl_token, recon = rl_model(tokens)
            loss = reconstruction_loss(tokens, recon, features.pad_masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(rl_model.parameters(), args.grad_clip)
            optimizer.step()

            metrics.append(loss.item())
            step += 1

            if step % args.log_interval == 0 or step == 1:
                avg_loss = sum(metrics) / len(metrics)
                elapsed = time.time() - start_time
                logging.info(
                    "step=%s loss=%.6f grad_norm=%.4f time=%.2fs",
                    step,
                    avg_loss,
                    float(grad_norm),
                    elapsed,
                )
                metrics.clear()
                start_time = time.time()

            if step % args.save_interval == 0 or step == args.num_train_steps:
                _save_checkpoint(output_dir, step, rl_model, optimizer, args, rl_config)
                logging.info("Saved checkpoint to %s", output_dir / str(step))


if __name__ == "__main__":
    main()