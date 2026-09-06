from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import PreTrainedTokenizerFast

from conftest import build_model, make_tokenizer
from llmserve.engine import ContextLengthError, GenerationEngine, GenerationRequest
from llmserve.sampling import SamplingParams

GREEDY = SamplingParams(temperature=0.0)


def greedy_reference(
    model: nn.Module, tokenizer: PreTrainedTokenizerFast, prompt: str, n: int
) -> list[int]:
    """Unbatched, uncached, unpadded greedy decoding: the oracle for the engine."""
    ids = tokenizer([prompt], padding=False, return_tensors="pt")["input_ids"]
    out: list[int] = []
    with torch.no_grad():
        for _ in range(n):
            nxt = int(model(input_ids=ids).logits[0, -1].argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


def req(rid: str, prompt: str, n: int, **kw: object) -> GenerationRequest:
    return GenerationRequest(rid, prompt, max_new_tokens=n, **kw)  # type: ignore[arg-type]


def test_batched_left_padded_greedy_matches_single_sequence(
    engine: GenerationEngine, model: nn.Module, tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = ["profit rose sharply", "the", "central bank raised rates again today"]
    results = engine.generate_batch(
        [req(f"r{i}", p, 6, sampling=GREEDY, stop_on_eos=False) for i, p in enumerate(prompts)]
    )
    for r, p in zip(results, prompts, strict=True):
        assert r.token_ids == greedy_reference(model, tokenizer, p, 6)
        assert r.finish_reason == "length" and r.completion_tokens == 6
        assert r.prompt_tokens == len(p.split())
        assert r.text == tokenizer.decode(r.token_ids, skip_special_tokens=True)
        assert r.latency_ms > 0
    assert engine.batches_run == 1


def test_rows_finishing_early_are_dropped_without_changing_others(
    engine: GenerationEngine, model: nn.Module, tokenizer: PreTrainedTokenizerFast
) -> None:
    """A: 2 tokens, B: 6 tokens, C: 4 tokens in one batch. B must be unaffected by A and C
    leaving the batch (exercises DynamicCache.batch_select_indices twice)."""
    prompts = {"a": "revenue fell", "b": "the bank said", "c": "shares rose"}
    lengths = {"a": 2, "b": 6, "c": 4}
    results = engine.generate_batch(
        [req(k, prompts[k], lengths[k], sampling=GREEDY, stop_on_eos=False) for k in prompts]
    )
    for r in results:
        ref = greedy_reference(model, tokenizer, prompts[r.request_id], lengths[r.request_id])
        assert r.token_ids == ref, r.request_id
        assert r.completion_tokens == lengths[r.request_id]


def test_eos_stops_generation(model: nn.Module, tokenizer: PreTrainedTokenizerFast) -> None:
    first = greedy_reference(model, tokenizer, "profit", 1)[0]
    eng = GenerationEngine(model, tokenizer, device="cpu", max_context=64, eos_token_id=first)
    stop = eng.generate_batch([req("s", "profit", 5, sampling=GREEDY)])[0]
    assert stop.finish_reason == "stop"
    assert stop.token_ids == [first] and stop.completion_tokens == 1
    assert stop.text == ""  # EOS is not part of the visible completion
    cont = eng.generate_batch([req("c", "profit", 5, sampling=GREEDY, stop_on_eos=False)])[0]
    assert cont.finish_reason == "length" and cont.completion_tokens == 5


def test_context_length_is_enforced(engine: GenerationEngine) -> None:
    long_prompt = " ".join(["profit"] * 70)  # truncated to max_context - 1 = 63 tokens
    with pytest.raises(ContextLengthError, match="exceeds max_context"):
        engine.generate_batch([req("x", long_prompt, 5, sampling=GREEDY)])
    ok = engine.generate_batch([req("x", long_prompt, 1, sampling=GREEDY, stop_on_eos=False)])[0]
    assert ok.prompt_tokens == 63 and ok.completion_tokens == 1


def test_per_request_seed_is_reproducible_and_batch_independent(
    qwen_engine: GenerationEngine,
) -> None:
    hot = SamplingParams(temperature=1.5, seed=7)
    alone = qwen_engine.generate_batch(
        [req("x", "the market", 12, sampling=hot, stop_on_eos=False)]
    )[0]
    batched = qwen_engine.generate_batch(
        [
            req("x", "the market", 12, sampling=hot, stop_on_eos=False),
            req(
                "y",
                "bank rates",
                12,
                sampling=SamplingParams(temperature=1.5, seed=3),
                stop_on_eos=False,
            ),
            req("x2", "the market", 12, sampling=hot, stop_on_eos=False),
        ]
    )
    by_id = {r.request_id: r for r in batched}
    assert by_id["x"].token_ids == alone.token_ids
    assert by_id["x2"].token_ids == alone.token_ids
    different_seed = qwen_engine.generate_batch(
        [
            req(
                "z",
                "the market",
                12,
                sampling=SamplingParams(temperature=1.5, seed=8),
                stop_on_eos=False,
            )
        ]
    )[0]
    assert different_seed.token_ids != alone.token_ids


def test_mixed_sampling_params_in_one_batch(
    engine: GenerationEngine, model: nn.Module, tokenizer: PreTrainedTokenizerFast
) -> None:
    results = engine.generate_batch(
        [
            req("greedy", "profit rose", 5, sampling=GREEDY, stop_on_eos=False),
            req(
                "sampled",
                "profit rose",
                5,
                sampling=SamplingParams(temperature=1.0, top_k=5, seed=1),
                stop_on_eos=False,
            ),
            req(
                "penalised",
                "profit rose",
                5,
                sampling=SamplingParams(temperature=0.0, repetition_penalty=1.5),
                stop_on_eos=False,
            ),
        ]
    )
    by_id = {r.request_id: r for r in results}
    assert by_id["greedy"].token_ids == greedy_reference(model, tokenizer, "profit rose", 5)
    assert all(len(r.token_ids) == 5 for r in results)


def test_input_validation(engine: GenerationEngine) -> None:
    with pytest.raises(ValueError, match="empty batch"):
        engine.generate_batch([])
    with pytest.raises(ValueError):
        GenerationRequest("a", "x", max_new_tokens=0)
    with pytest.raises(ValueError):
        GenerationRequest("a", "", max_new_tokens=1)


def test_count_tokens(engine: GenerationEngine) -> None:
    assert engine.count_tokens("profit rose sharply") == 3


def test_pad_token_falls_back_to_eos(tokenizer: PreTrainedTokenizerFast) -> None:
    tok = make_tokenizer(with_pad=False)
    assert tok.pad_token_id is None
    eng = GenerationEngine(build_model("qwen2", tokenizer), tok, device="cpu", max_context=64)
    assert eng.tokenizer.pad_token_id == tok.eos_token_id
    assert eng.eos_token_id == tok.eos_token_id
    out = eng.generate_batch(
        [
            req("a", "profit", 3, sampling=GREEDY, stop_on_eos=False),
            req("b", "the bank said", 3, sampling=GREEDY, stop_on_eos=False),
        ]
    )
    assert [r.completion_tokens for r in out] == [3, 3]
