from __future__ import annotations

import math

import pytest
import torch

from nanoformer.generate import generate, sample_next_token, top_k_filter, top_p_filter
from nanoformer.model import Transformer


class TestFilters:
    def test_top_k_keeps_exactly_k(self) -> None:
        logits = torch.tensor([[0.1, 3.0, 2.0, -1.0, 2.5]])
        out = top_k_filter(logits, 2)
        assert torch.isfinite(out).sum() == 2
        assert torch.isfinite(out[0, 1]) and torch.isfinite(out[0, 4])
        assert torch.equal(top_k_filter(logits, 0), logits)
        assert torch.equal(top_k_filter(logits, 99), logits)

    def test_top_p_nucleus(self) -> None:
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
        logits = probs.log()
        kept = torch.isfinite(top_p_filter(logits, 0.75))
        assert kept.tolist() == [[True, True, False, False]]
        # The top-1 token is always kept, even when p is tiny.
        assert torch.isfinite(top_p_filter(logits, 0.01)).tolist() == [[True, False, False, False]]
        assert torch.equal(top_p_filter(logits, 1.0), logits)

    def test_top_p_unsorted_input_maps_back_to_original_positions(self) -> None:
        logits = torch.tensor([[math.log(0.05), math.log(0.5), math.log(0.15), math.log(0.3)]])
        kept = torch.isfinite(top_p_filter(logits, 0.75))
        assert kept.tolist() == [[False, True, False, True]]

    def test_top_p_invalid(self) -> None:
        with pytest.raises(ValueError):
            top_p_filter(torch.zeros(1, 3), 0.0)


class TestSampling:
    def test_greedy(self) -> None:
        logits = torch.tensor([[0.0, 5.0, 1.0], [2.0, 0.0, 0.0]])
        assert sample_next_token(logits, temperature=0.0).tolist() == [1, 0]

    def test_seeded_sampling_is_reproducible(self) -> None:
        logits = torch.randn(4, 20)
        a = sample_next_token(logits, generator=torch.Generator().manual_seed(1))
        b = sample_next_token(logits, generator=torch.Generator().manual_seed(1))
        assert torch.equal(a, b)

    def test_sampling_respects_filters(self) -> None:
        logits = torch.tensor([[10.0, 0.0, -10.0]])
        for _ in range(20):
            tok = sample_next_token(logits, temperature=1.0, top_k=1)
            assert tok.item() == 0

    def test_negative_temperature_rejected(self) -> None:
        with pytest.raises(ValueError):
            sample_next_token(torch.zeros(1, 3), temperature=-1.0)


class TestGenerate:
    def test_cached_and_uncached_greedy_agree(self, tiny_model: Transformer) -> None:
        prompt = torch.randint(0, 64, (2, 4))
        with_cache = generate(tiny_model, prompt, 10, temperature=0.0, use_cache=True)
        no_cache = generate(tiny_model, prompt, 10, temperature=0.0, use_cache=False)
        assert with_cache.shape == (2, 14)
        assert torch.equal(with_cache, no_cache)
        assert torch.equal(with_cache[:, :4], prompt)

    def test_seeded_sampling_reproducible(self, tiny_model: Transformer) -> None:
        prompt = torch.randint(0, 64, (1, 3))
        a = generate(tiny_model, prompt, 8, generator=torch.Generator().manual_seed(3))
        b = generate(tiny_model, prompt, 8, generator=torch.Generator().manual_seed(3))
        assert torch.equal(a, b)

    def test_eos_freezes_finished_rows(self, tiny_model: Transformer) -> None:
        prompt = torch.randint(0, 64, (3, 2))
        greedy_first = generate(tiny_model, prompt, 1, temperature=0.0)[:, -1]
        eos = int(greedy_first[0])
        out = generate(tiny_model, prompt, 6, temperature=0.0, eos_id=eos)
        # Row 0 emits eos immediately, so everything after is padded with eos.
        assert (out[0, 2:] == eos).all()
        assert out.shape[1] <= 8

    def test_stops_early_when_all_rows_finished(self, tiny_model: Transformer) -> None:
        prompt = torch.randint(0, 64, (1, 2))
        eos = int(generate(tiny_model, prompt, 1, temperature=0.0)[0, -1])
        out = generate(tiny_model, prompt, 20, temperature=0.0, eos_id=eos)
        assert out.shape == (1, 3)

    def test_zero_new_tokens_and_overflow(self, tiny_model: Transformer) -> None:
        prompt = torch.randint(0, 64, (1, 5))
        assert torch.equal(generate(tiny_model, prompt, 0), prompt)
        with pytest.raises(ValueError, match="exceeds"):
            generate(tiny_model, prompt, tiny_model.cfg.max_seq_len)
        with pytest.raises(ValueError):
            generate(tiny_model, prompt, -1)

    def test_training_mode_is_restored(self, tiny_model: Transformer) -> None:
        tiny_model.train()
        generate(tiny_model, torch.zeros(1, 1, dtype=torch.long), 2)
        assert tiny_model.training
