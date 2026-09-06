"""Offline fixtures: two randomly initialised tiny causal LMs (Qwen2 and GPT-2 families,
i.e. both the modern Linear-based and the legacy Conv1D-based HF architectures) and a
word-level tokenizer built in memory. Nothing downloads."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from llmserve.config import Settings
from llmserve.engine import GenerationEngine

WORDS = [
    "the",
    "a",
    "of",
    "to",
    "in",
    "and",
    "on",
    "for",
    "at",
    "by",
    "with",
    "from",
    "profit",
    "loss",
    "revenue",
    "rose",
    "fell",
    "sharply",
    "slightly",
    "bank",
    "rates",
    "central",
    "raised",
    "cut",
    "today",
    "yesterday",
    "quarter",
    "year",
    "shares",
    "market",
    "stock",
    "bond",
    "growth",
    "decline",
    "strong",
    "weak",
    "report",
    "said",
    "announced",
    "expects",
    "guidance",
    "outlook",
    "risk",
    "capital",
    "australia",
    "sydney",
    "dollar",
    "percent",
    "million",
    "billion",
    "company",
    "board",
    "investors",
    "trading",
]
SPECIALS = ["<pad>", "<|endoftext|>", "<unk>"]


def build_vocab() -> dict[str, int]:
    return {w: i for i, w in enumerate(SPECIALS + WORDS)}


def make_tokenizer(with_pad: bool = True) -> PreTrainedTokenizerFast:
    core = Tokenizer(models.WordLevel(build_vocab(), unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    kwargs = {"pad_token": "<pad>"} if with_pad else {}
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core, eos_token="<|endoftext|>", unk_token="<unk>", **kwargs
    )
    tok.padding_side = "left"
    return tok


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    return make_tokenizer()


def build_model(family: str, tokenizer: PreTrainedTokenizerFast) -> nn.Module:
    vocab_size = len(build_vocab())
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    torch.manual_seed(0)
    model: nn.Module
    if family == "qwen2":
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=vocab_size,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=64,
                max_position_embeddings=128,
                eos_token_id=eos,
                pad_token_id=pad,
                bos_token_id=None,
                tie_word_embeddings=False,
            )
        )
    elif family == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_layer=2,
                n_embd=32,
                n_head=2,
                n_positions=128,
                eos_token_id=eos,
                bos_token_id=eos,
                pad_token_id=pad,
            )
        )
    else:
        raise ValueError(family)
    return model.eval()


@pytest.fixture(params=["qwen2", "gpt2"])
def model(request: pytest.FixtureRequest, tokenizer: PreTrainedTokenizerFast) -> nn.Module:
    family: str = request.param
    return build_model(family, tokenizer)


@pytest.fixture
def engine(model: nn.Module, tokenizer: PreTrainedTokenizerFast) -> GenerationEngine:
    return GenerationEngine(
        model, tokenizer, device="cpu", max_context=64, model_name="tiny", seed=0
    )


@pytest.fixture
def qwen_engine(tokenizer: PreTrainedTokenizerFast) -> GenerationEngine:
    return GenerationEngine(
        build_model("qwen2", tokenizer),
        tokenizer,
        device="cpu",
        max_context=64,
        model_name="tiny-qwen2",
        seed=0,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        model_name="tiny",
        device="cpu",
        max_batch_size=4,
        batch_window_ms=20.0,
        max_context=64,
        max_new_tokens_limit=16,
        max_prompt_chars=400,
        queue_maxsize=8,
        request_timeout_s=10.0,
    )
