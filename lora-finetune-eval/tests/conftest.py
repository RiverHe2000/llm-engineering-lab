"""Offline fixtures: a randomly initialised 23k-parameter DistilBERT, a word-level
tokenizer built in memory, and a synthetic sentiment dataset that a tiny model can learn
in seconds on CPU. No test touches the network."""

from __future__ import annotations

import random

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from torch import nn
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from loraeval.data import Example

POS = ["profit", "rose", "surge", "growth", "record", "strong", "gain", "up", "improved", "beat"]
NEG = ["loss", "fell", "decline", "drop", "weak", "cut", "down", "plunge", "missed", "worse"]
NEU = [
    "company",
    "said",
    "quarter",
    "report",
    "announced",
    "shares",
    "board",
    "today",
    "meeting",
    "plans",
]
FILL = ["the", "of", "in", "and", "to", "a", "for", "on"]
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


def build_vocab() -> dict[str, int]:
    words = SPECIALS + POS + NEG + NEU + FILL
    return {w: i for i, w in enumerate(words)}


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def vocab() -> dict[str, int]:
    return build_vocab()


@pytest.fixture(scope="session")
def fake_tokenizer(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _sentence(rng: random.Random, class_words: list[str]) -> str:
    words = rng.sample(class_words, k=rng.randint(3, 5)) + rng.choices(FILL, k=rng.randint(2, 4))
    rng.shuffle(words)
    return " ".join(words)


@pytest.fixture(scope="session")
def synthetic_examples() -> list[Example]:
    rng = random.Random(0)
    examples: list[Example] = []
    for label, words, n in ((0, NEG, 80), (1, NEU, 120), (2, POS, 100)):
        examples.extend(Example(_sentence(rng, words), label) for _ in range(n))
    rng.shuffle(examples)
    return examples


@pytest.fixture
def tiny_config(vocab: dict[str, int]) -> DistilBertConfig:
    return DistilBertConfig(
        vocab_size=len(vocab),
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=32,
        num_labels=3,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        pad_token_id=vocab["[PAD]"],
    )


@pytest.fixture
def tiny_model(tiny_config: DistilBertConfig) -> nn.Module:
    torch.manual_seed(0)
    model: nn.Module = DistilBertForSequenceClassification(tiny_config)
    return model
