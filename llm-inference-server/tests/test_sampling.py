from __future__ import annotations

import math

import pytest
import torch

from llmserve.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    sample_token,
    top_k_filter,
    top_p_filter,
)


class TestParams:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": -0.1},
            {"top_k": -1},
            {"top_p": 0.0},
            {"top_p": 1.5},
            {"repetition_penalty": 0.5},
        ],
    )
    def test_invalid(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            SamplingParams(**kwargs)  # type: ignore[arg-type]

    def test_greedy_flag_and_equality(self) -> None:
        assert SamplingParams(temperature=0.0).is_greedy
        assert not SamplingParams().is_greedy
        assert SamplingParams(top_k=5) == SamplingParams(top_k=5)


def test_top_k() -> None:
    logits = torch.tensor([[0.1, 3.0, 2.0, -1.0, 2.5]])
    out = top_k_filter(logits, 2)
    assert (
        torch.isfinite(out).sum() == 2 and torch.isfinite(out[0, 1]) and torch.isfinite(out[0, 4])
    )
    assert torch.equal(top_k_filter(logits, 0), logits)
    assert torch.equal(top_k_filter(logits, 10), logits)


def test_top_p() -> None:
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()
    assert torch.isfinite(top_p_filter(logits, 0.75)).tolist() == [[True, True, False, False]]
    assert torch.isfinite(top_p_filter(logits, 0.01)).tolist() == [[True, False, False, False]]
    assert torch.equal(top_p_filter(logits, 1.0), logits)
    shuffled = torch.tensor([[math.log(0.05), math.log(0.5), math.log(0.15), math.log(0.3)]])
    assert torch.isfinite(top_p_filter(shuffled, 0.75)).tolist() == [[False, True, False, True]]


def test_repetition_penalty_pushes_seen_tokens_down() -> None:
    logits = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_repetition_penalty(logits, torch.tensor([0, 1, 1]), penalty=2.0)
    assert out[0] == pytest.approx(1.0)  # positive: divided
    assert out[1] == pytest.approx(-4.0)  # negative: multiplied
    assert out[2] == 1.0 and out[3] == 0.5  # untouched
    assert torch.equal(apply_repetition_penalty(logits, torch.tensor([0]), 1.0), logits)
    assert torch.equal(
        apply_repetition_penalty(logits, torch.tensor([], dtype=torch.long), 2.0), logits
    )


def test_sample_token_greedy_and_seeded() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0], [2.0, 0.0, 0.0]])
    assert sample_token(logits, SamplingParams(temperature=0.0)).tolist() == [1, 0]
    rand = torch.randn(3, 20)
    a = sample_token(rand, SamplingParams(), torch.Generator().manual_seed(1))
    b = sample_token(rand, SamplingParams(), torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and a.shape == (3,)
    peaked = torch.tensor([[10.0, 0.0, -10.0]])
    for _ in range(10):
        assert sample_token(peaked, SamplingParams(top_k=1)).item() == 0
    with pytest.raises(ValueError):
        sample_token(torch.zeros(3), SamplingParams())
