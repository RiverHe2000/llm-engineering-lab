"""Byte-level Byte-Pair Encoding (Sennrich et al., 2016; GPT-2 style), from scratch.

Vocabulary layout (fixed, so ids are stable across saves):

    0..255                      the 256 raw bytes
    256..256+n_merges-1         merge tokens, in the order they were learned
    256+n_merges..              special tokens (default: ``<|endoftext|>``)

Working on bytes rather than characters guarantees that *any* string can be encoded
(no "unknown" token) and that decoding is lossless up to invalid UTF-8 boundaries.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

Pair = tuple[int, int]

# A simplified GPT-2 pre-tokeniser: words with an optional leading space, numbers,
# punctuation runs, and whitespace. Merges never cross these boundaries, which keeps
# the vocabulary linguistically sensible ("ing", " the") instead of "e t".
DEFAULT_PATTERN = r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<|endoftext|>",)


def _merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Trainable byte-level BPE tokenizer with special-token support."""

    def __init__(
        self,
        merges: Sequence[Pair],
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        pattern: str = DEFAULT_PATTERN,
    ) -> None:
        self.merges: list[Pair] = [(int(a), int(b)) for a, b in merges]
        self.pattern = pattern
        self._word_re = re.compile(pattern)
        self._ranks: dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}
        self._merge_ids: dict[Pair, int] = {p: 256 + i for i, p in enumerate(self.merges)}

        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            if a not in self._vocab or b not in self._vocab:
                raise ValueError(f"merge {i} references unknown ids ({a}, {b})")
            self._vocab[256 + i] = self._vocab[a] + self._vocab[b]

        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special tokens must be unique")
        self.special_tokens: tuple[str, ...] = tuple(special_tokens)
        base = 256 + len(self.merges)
        self._special_to_id: dict[str, int] = {
            t: base + j for j, t in enumerate(self.special_tokens)
        }
        self._id_to_special: dict[int, str] = {v: k for k, v in self._special_to_id.items()}
        self._special_re: re.Pattern[str] | None = None
        if self.special_tokens:
            alternation = "|".join(
                re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_re = re.compile(f"({alternation})")
        self._cache: dict[str, list[int]] = {}

    # ----- properties ----------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eos_id(self) -> int:
        if not self.special_tokens:
            raise ValueError("tokenizer has no special tokens, so no eos id")
        return self._special_to_id[self.special_tokens[0]]

    @property
    def eos_token(self) -> str:
        return self.special_tokens[0]

    def special_token_id(self, token: str) -> int:
        return self._special_to_id[token]

    def token_bytes(self, token_id: int) -> bytes:
        if token_id in self._id_to_special:
            return self._id_to_special[token_id].encode("utf-8")
        return self._vocab[token_id]

    # ----- training ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        pattern: str = DEFAULT_PATTERN,
    ) -> BPETokenizer:
        """Learn ``vocab_size - 256 - len(special_tokens)`` merges from ``text``.

        Complexity is O(n_merges * unique_words); fine for corpora of a few MB, which is
        the scope of this project. Production tokenizers keep incremental pair counts.
        Ties are broken deterministically (highest count, then smallest pair) so the same
        corpus always yields the same vocabulary.
        """
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is too small for 256 bytes + "
                f"{len(special_tokens)} special tokens"
            )
        word_re = re.compile(pattern)
        word_freq: Counter[str] = Counter(word_re.findall(text))
        seqs: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in word_freq}

        merges: list[Pair] = []
        for i in range(n_merges):
            pair_counts: Counter[Pair] = Counter()
            for w, freq in word_freq.items():
                s = seqs[w]
                for a, b in pairwise(s):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = 256 + i
            merges.append(best)
            for w, s in seqs.items():
                if len(s) > 1:
                    seqs[w] = _merge_pair(s, best, new_id)
        return cls(merges, special_tokens, pattern)

    # ----- encoding ------------------------------------------------------------------
    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(pairwise(ids))
            best = min(pairs, key=lambda p: self._ranks.get(p, len(self._ranks)))
            if best not in self._ranks:
                break
            ids = _merge_pair(ids, best, self._merge_ids[best])
        self._cache[word] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for word in self._word_re.findall(text):
            out.extend(self._encode_word(word))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode ``text``. Special tokens appearing verbatim in the text are recognised
        only when ``allow_special`` is set; otherwise they are encoded as ordinary bytes
        (the safe default for untrusted user input)."""
        if not allow_special or self._special_re is None:
            return self._encode_plain(text)
        out: list[int] = []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            if chunk in self._special_to_id:
                out.append(self._special_to_id[chunk])
            else:
                out.extend(self._encode_plain(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            buf.extend(self.token_bytes(int(i)))
        return buf.decode("utf-8", errors="replace")

    # ----- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": list(self.special_tokens),
            "merges": [list(p) for p in self.merges],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=None), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BPETokenizer:
        merges = [(int(a), int(b)) for a, b in d["merges"]]
        return cls(merges, d.get("special_tokens", DEFAULT_SPECIAL_TOKENS), d["pattern"])

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)
