from __future__ import annotations

from pathlib import Path

import pytest

from nanoformer.tokenizer import BPETokenizer, _merge_pair

CORPUS = (
    "The quick brown fox jumps over the lazy dog. The dog sleeps; the fox runs.\n"
    "Numbers like 2024 and 3.14 appear, as do symbols: #$%&!\n"
    "Unicode works too: 你好，世界! Ünïcödé — 🚀🚀\n"
) * 20


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=320)


def test_merge_pair_helper() -> None:
    assert _merge_pair([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert _merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]  # non-overlapping, left to right
    assert _merge_pair([], (1, 2), 9) == []


def test_vocab_layout(tok: BPETokenizer) -> None:
    assert tok.vocab_size == 320
    assert len(tok.merges) == 320 - 256 - 1
    assert tok.eos_id == 319
    assert tok.eos_token == "<|endoftext|>"
    assert tok.token_bytes(65) == b"A"
    assert tok.token_bytes(tok.eos_id) == b"<|endoftext|>"


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "The quick brown fox",
        "   leading and trailing   ",
        "tabs\tand\nnewlines\r\n",
        "你好，世界! Ünïcödé — 🚀",
        "never seen ZZZ qqq 9999 §§§",
        "",
    ],
)
def test_round_trip(tok: BPETokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_compresses_training_text(tok: BPETokenizer) -> None:
    n_bytes = len(CORPUS.encode("utf-8"))
    n_tokens = len(tok.encode(CORPUS))
    assert n_tokens < 0.6 * n_bytes


def test_common_words_become_single_tokens(tok: BPETokenizer) -> None:
    assert len(tok.encode(" the")) == 1


def test_special_tokens(tok: BPETokenizer) -> None:
    text = "hi<|endoftext|>there"
    ids = tok.encode(text)
    assert tok.eos_id in ids
    assert tok.decode(ids) == text
    plain = tok.encode(text, allow_special=False)
    assert tok.eos_id not in plain
    assert tok.decode(plain) == text
    assert tok.special_token_id("<|endoftext|>") == tok.eos_id


def test_training_is_deterministic() -> None:
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_save_load_round_trip(tok: BPETokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.encode(CORPUS[:500]) == tok.encode(CORPUS[:500])


def test_decode_invalid_utf8_uses_replacement_char(tok: BPETokenizer) -> None:
    # 0xE4 is the first byte of a 3-byte sequence; alone it is invalid.
    assert tok.decode([0xE4]) == "�"


def test_too_small_vocab_rejected() -> None:
    with pytest.raises(ValueError, match="too small"):
        BPETokenizer.train("abc", vocab_size=256)


def test_training_stops_early_when_nothing_left_to_merge() -> None:
    tok = BPETokenizer.train("ab", vocab_size=400)
    assert len(tok.merges) == 1  # only one pair ever existed
    assert tok.vocab_size == 258


def test_invalid_merges_and_duplicates_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ids"):
        BPETokenizer([(1, 999)])
    with pytest.raises(ValueError, match="unique"):
        BPETokenizer([], special_tokens=["<a>", "<a>"])


def test_no_special_tokens() -> None:
    tok = BPETokenizer([], special_tokens=[])
    assert tok.vocab_size == 256
    assert tok.encode("<|endoftext|>") == list(b"<|endoftext|>")
    with pytest.raises(ValueError):
        _ = tok.eos_id
