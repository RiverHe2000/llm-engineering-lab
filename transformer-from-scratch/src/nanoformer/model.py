"""The decoder-only Transformer language model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanoformer.cache import KVCache
from nanoformer.config import ModelConfig
from nanoformer.layers import RMSNorm, TransformerBlock, precompute_rope


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None


class Transformer(nn.Module):
    """Token embedding -> N pre-norm blocks -> RMSNorm -> (tied) LM head.

    Initialisation follows GPT-2/nanoGPT: N(0, 0.02) everywhere, with the two residual
    output projections per block scaled by ``1/sqrt(2 * n_layers)`` so that the residual
    stream's variance does not grow with depth at init.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Weight tying (Press & Wolf, 2017): halves embedding parameters and
            # regularises the output layer; standard in GPT-2 and most small LMs.
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.rope_cos: Tensor
        self.rope_sin: Tensor

        self.apply(self._init_weights)
        residual_scale = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("attn.wo.weight", "ffn.w_down.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # ----------------------------------------------------------------------------------
    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> ModelOutput:
        """Run the model on ``idx`` (``[B, T]`` token ids).

        * ``targets`` (``[B, T]``): next-token labels; positions equal to ``-1`` are ignored.
          When given, ``loss`` is the mean cross-entropy computed in float32.
        * ``cache``: when given, ``idx`` is treated as the continuation of the sequence
          already in the cache (positions ``cache.seq_len ...``) and the cache is advanced.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must be [batch, seq]; got shape {tuple(idx.shape)}")
        _, t = idx.shape
        start = 0 if cache is None else cache.seq_len
        if start + t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence of length {start + t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        cos = self.rope_cos[start : start + t]
        sin = self.rope_sin[start : start + t]

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.advance(t)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return ModelOutput(logits=logits, loss=loss)

    # ----------------------------------------------------------------------------------
    def num_params(self, exclude_embeddings: bool = True) -> int:
        """Parameter count; by default excludes the (input and, if untied, output) embeddings,
        which is the convention used when quoting model sizes like "124M"."""
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    @property
    def device(self) -> torch.device:
        return self.tok_emb.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_emb.weight.dtype

    def set_attention_impl(self, use_sdpa: bool) -> None:
        """Switch every attention layer between the fused kernel and the reference path."""
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            block.attn.use_sdpa = use_sdpa

    def estimate_flops_per_token(self) -> int:
        """Forward-pass FLOPs per token, ``2 * params`` + attention (Kaplan et al., 2020)."""
        cfg = self.cfg
        attn = 2 * cfg.n_layers * cfg.max_seq_len * cfg.d_model
        return 2 * self.num_params(exclude_embeddings=True) + attn
