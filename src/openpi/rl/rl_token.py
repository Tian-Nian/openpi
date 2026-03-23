import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    hidden_dim: int
    rl_token_dim: int | None = None
    max_seq_len: int = 512
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 8
    ff_dim: int | None = None
    dropout: float = 0.1
    decoder_causal: bool = False

    def __post_init__(self) -> None:
        if self.rl_token_dim is None:
            object.__setattr__(self, "rl_token_dim", self.hidden_dim)
        if self.ff_dim is None:
            object.__setattr__(self, "ff_dim", self.hidden_dim * 2)


class RLTokenEncoder(nn.Module):
    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        self.rl_probe = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        if config.rl_token_dim != config.hidden_dim:
            self.out_proj = nn.Linear(config.hidden_dim, config.rl_token_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = prefix_tokens.shape[0]
        probe = self.rl_probe.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([prefix_tokens, probe], dim=1))
        rl_token = encoded[:, -1, :]
        return self.out_proj(rl_token)


class RLTokenDecoder(nn.Module):
    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        if config.rl_token_dim != config.hidden_dim:
            self.memory_proj = nn.Linear(config.rl_token_dim, config.hidden_dim)
        else:
            self.memory_proj = nn.Identity()
        self.query_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(config.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

    def _build_tgt_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if not self.config.decoder_causal:
            return None
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, rl_token: torch.Tensor, seq_len: int) -> torch.Tensor:
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.config.max_seq_len})")

        memory = self.memory_proj(rl_token)[:, None, :]
        queries = self.query_embed[:, :seq_len, :].expand(rl_token.shape[0], -1, -1)
        queries = self.query_norm(queries)
        tgt_mask = self._build_tgt_mask(seq_len, rl_token.device)
        return self.decoder(tgt=queries, memory=memory, tgt_mask=tgt_mask)


class RLTokenAutoencoder(nn.Module):
    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.encoder = RLTokenEncoder(config)
        self.decoder = RLTokenDecoder(config)

    def forward(self, prefix_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rl_token = self.encoder(prefix_tokens)
        recon = self.decoder(rl_token, prefix_tokens.shape[1])
        return rl_token, recon


def reconstruction_loss(
    target_tokens: torch.Tensor,
    recon_tokens: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    squared_error = (target_tokens - recon_tokens) ** 2
    if token_mask is None:
        return squared_error.mean()

    expanded_mask = token_mask.unsqueeze(-1).to(dtype=squared_error.dtype)
    valid = expanded_mask.sum() * target_tokens.shape[-1]
    if valid.item() == 0:
        raise ValueError("token_mask does not contain any valid tokens")
    return (squared_error * expanded_mask).sum() / valid