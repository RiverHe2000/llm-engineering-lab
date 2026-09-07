"""Tests for the DPO loop: the reference-policy trick, the step-0 identity, and convergence.

Everything runs on CPU against a two-layer Qwen2 built from config with a 256-token vocabulary,
so a forty-step preference run costs under a second. No tokenizer is downloaded; the pairs are
constructed as token sequences directly, which is also the only way to build a set whose
preference is separable by construction.

Four tests carry the weight of the file.

`test_step_zero_loss_is_exactly_ln_two` is the acceptance test for the whole forward path. A
freshly attached LoRA adapter has ``B = 0``, so the policy and the reference are the same
function and every log-ratio cancels. The loss is then exactly ``ln 2`` -- not approximately --
and it stays exactly ``ln 2`` for every beta. Any error in the concatenation, the label masking,
the log-probability shift or the reference context would break the cancellation and move the
number.

`test_the_reference_reproduces_a_separately_measured_base_model` is the memory claim. The base
model's log-probabilities are recorded *before* an adapter is attached; the adapter is then
attached, deliberately moved, and the reference pass has to return those recorded numbers back,
bit for bit. That is what licences a DPO run to hold one set of weights instead of two.

`test_reward_accuracy_rises_from_a_tie_to_the_whole_batch` and
`test_the_preference_rule_generalises_to_held_out_pairs` are the statements that the loop
learns: on the training pairs the implicit reward goes from ranking nothing correctly to ranking
everything correctly, and on pairs it has never seen the accuracy climbs from roughly chance to
one while the margin opens.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.modeling.chat import ChatFormatter
from sftdpo.modeling.collate import IGNORE_INDEX, PairFeature, PreferenceCollator
from sftdpo.modeling.loader import build_tiny_model
from sftdpo.modeling.lora import attach_adapter, lora_config
from sftdpo.schemas import PreferencePair, Slice
from sftdpo.train.common import RunManifest, TrainConfig, TrainLog
from sftdpo.train.dpo import (
    PAIR_KEYS,
    REWARD_LOG_FILENAME,
    DPOEvaluation,
    DPOResult,
    RewardPoint,
    concatenated_batch,
    dpo_batch_output,
    encode_pairs,
    evaluate_dpo,
    least_squares_slope,
    pair_token_count,
    policy_logprobs,
    reference_logprobs,
    train_dpo,
)
from sftdpo.train.dpo_loss import DPOVariant, sequence_logprob
from sftdpo.train.sft import LOG_FILENAME, MANIFEST_FILENAME, SUMMARY_FILENAME

VOCAB_SIZE = 256
SEQ_BUDGET = 48

LN2_FLOAT32 = 0.6931471824645996
"""``-log sigmoid(0)`` evaluated in float32, which is what the forward pass produces.

Deliberately the float32 value rather than `math.log(2)`: the assertion is an exact equality,
and quoting the float64 constant would force a tolerance that could hide a real difference.
"""

PROPERTY_SETTINGS = settings(max_examples=50, deadline=None)


# --------------------------------------------------------------------------------------
# Fixtures and stand-ins
# --------------------------------------------------------------------------------------


class TinyTokenizer:
    """Whitespace tokeniser with stable ids inside the tiny model's vocabulary.

    CRC32 rather than `hash()`, whose salt changes between processes: a determinism test is
    worthless if the ids themselves move between runs.
    """

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None
    last_add_special_tokens: bool = True

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.last_add_special_tokens = add_special_tokens
        return [2 + zlib.crc32(word.encode("utf-8")) % (VOCAB_SIZE - 2) for word in text.split()]


def tiny_model(seed: int = 0) -> Any:
    """A two-layer Qwen2 small enough for a forty-step preference run to cost a second."""
    return build_tiny_model(
        seed=seed,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ_BUDGET * 2,
    )


def attach(model: Any, *, seed: int = 0, r: int = 4) -> Any:
    """Attach a LoRA adapter whose `A` matrices do not depend on the global RNG state.

    PEFT initialises `lora_A` from the global torch generator, so without the fork an adapter
    would depend on whichever tests ran first -- and two runs meant to be compared would start
    from different weights. That is not hypothetical: it is what made an early version of the
    accumulation test below disagree by eight per cent for reasons that had nothing to do with
    accumulation.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return attach_adapter(model, lora_config(r=r))


def adapted_model(seed: int = 0, r: int = 4) -> Any:
    """The tiny model with a freshly attached adapter, so `B` is still zero."""
    return attach(tiny_model(seed), seed=seed, r=r)


def move_adapter(model: Any, amount: float = 0.05) -> None:
    """Push every `lora_B` off zero, so the policy stops being its own reference."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.add_(torch.full_like(parameter, amount))


def make_pairs(count: int, *, completion_len: int = 4, start: int = 0) -> list[PairFeature]:
    """Preference pairs separable by construction: chosen and rejected tokens never overlap.

    Every chosen completion is drawn from the token band 60-79 and every rejected one from
    180-199, so there is a rule to learn that is not merely memorisation of these prompts --
    which is what makes the held-out test below meaningful.
    """
    pairs: list[PairFeature] = []
    for index in range(start, start + count):
        prompt = [10 + index * 3 + offset for offset in range(4)]
        chosen = [60 + (index + offset) % 20 for offset in range(completion_len)]
        rejected = [180 + (index + offset) % 20 for offset in range(completion_len)]
        pairs.append(PairFeature.from_shared(prompt, chosen, rejected))
    return pairs


def make_preference_pair(index: int) -> PreferencePair:
    """One mined pair, the shape the preference-mining stage produces."""
    return PreferencePair(
        example_id=f"ex-{index}",
        slice=Slice.CLEAN,
        prompt=f"note {index} growth",
        chosen='{"risk_profile": "growth"}',
        rejected="the client wants growth",
        chosen_reward=1.0,
        rejected_reward=0.2,
    )


def formatter() -> ChatFormatter:
    """A formatter with a one-word system turn, so pairs fit the tiny sequence budget."""
    return ChatFormatter(system="SYS")


def make_config(tmp_path: Path, **overrides: Any) -> TrainConfig:
    """A config that never writes to the repository and logs every step."""
    settings_: dict[str, Any] = {
        "learning_rate": 1e-2,
        "batch_size": 4,
        "epochs": 1,
        "warmup_ratio": 0.0,
        "eval_every_steps": 0,
        "log_every_steps": 1,
        "max_seq_length": SEQ_BUDGET,
        "save_best_adapter": False,
        "output_dir": tmp_path,
    }
    settings_.update(overrides)
    return TrainConfig(**settings_)


@pytest.fixture
def tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


@pytest.fixture
def collator() -> PreferenceCollator:
    """A fresh collator per test, so the collation counters cannot leak between them."""
    return PreferenceCollator(pad_token_id=0, max_length=SEQ_BUDGET)


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------


def test_features_pass_through_encoding_untouched(tokenizer: TinyTokenizer) -> None:
    pairs = make_pairs(3)
    assert encode_pairs(pairs, formatter=formatter(), tokenizer=tokenizer) == pairs


def test_preference_pairs_are_encoded_on_both_sides(tokenizer: TinyTokenizer) -> None:
    """Both completions are rendered through the same prompt the model will be sampled from."""
    pair = make_preference_pair(0)
    chat = formatter()
    [feature] = encode_pairs([pair], formatter=chat, tokenizer=tokenizer)
    prompt_ids = tokenizer.encode(chat.prompt_text(pair.prompt, tokenizer=tokenizer))
    assert feature.chosen.prompt_len == len(prompt_ids)
    assert feature.rejected.prompt_len == len(prompt_ids)
    assert feature.chosen.prompt_ids == feature.rejected.prompt_ids
    assert feature.chosen.completion_ids != feature.rejected.completion_ids


def test_encoding_refuses_an_item_it_does_not_understand(tokenizer: TinyTokenizer) -> None:
    with pytest.raises(TypeError, match="expected a PreferencePair or a PairFeature"):
        encode_pairs(["a pair"], formatter=formatter(), tokenizer=tokenizer)  # type: ignore[list-item]


def test_pair_token_count_covers_both_sides(collator: PreferenceCollator) -> None:
    """A DPO step consumes both completions, so a per-step token figure has to count both."""
    batch = collator(make_pairs(3, completion_len=5))
    assert pair_token_count(batch) == 3 * 5 * 2


def test_pair_token_count_ignores_padding(collator: PreferenceCollator) -> None:
    """Ragged pairs are padded to one width; the padding must not inflate the count."""
    pairs = make_pairs(2, completion_len=3) + make_pairs(2, completion_len=7, start=2)
    batch = collator(pairs)
    assert batch["chosen_input_ids"].shape[1] == 11
    assert pair_token_count(batch) == 2 * (3 + 3) + 2 * (7 + 7)


# --------------------------------------------------------------------------------------
# The concatenated forward pass
# --------------------------------------------------------------------------------------


def test_chosen_rows_come_first(collator: PreferenceCollator) -> None:
    """The row order is a contract: every `chunk(2)` downstream reads chosen from the front."""
    batch = collator(make_pairs(3))
    input_ids, attention_mask, labels = concatenated_batch(batch)
    assert torch.equal(input_ids[:3], batch["chosen_input_ids"])
    assert torch.equal(input_ids[3:], batch["rejected_input_ids"])
    assert torch.equal(attention_mask[:3], batch["chosen_attention_mask"])
    assert torch.equal(labels[3:], batch["rejected_labels"])


@pytest.mark.parametrize("pairs", [1, 2, 5])
def test_concatenation_doubles_the_batch_and_keeps_the_width(
    collator: PreferenceCollator, pairs: int
) -> None:
    batch = collator(make_pairs(pairs))
    width = batch["chosen_input_ids"].shape[1]
    for tensor in concatenated_batch(batch):
        assert tuple(tensor.shape) == (2 * pairs, width)


def test_chunking_recovers_the_two_halves(collator: PreferenceCollator) -> None:
    """Concatenate then chunk is the identity, which is what makes one forward pass legal."""
    batch = collator(make_pairs(4))
    input_ids, _, labels = concatenated_batch(batch)
    chosen_ids, rejected_ids = input_ids.chunk(2, dim=0)
    chosen_labels, rejected_labels = labels.chunk(2, dim=0)
    assert torch.equal(chosen_ids, batch["chosen_input_ids"])
    assert torch.equal(rejected_ids, batch["rejected_input_ids"])
    assert torch.equal(chosen_labels, batch["chosen_labels"])
    assert torch.equal(rejected_labels, batch["rejected_labels"])


def test_concatenated_batch_names_every_missing_key(collator: PreferenceCollator) -> None:
    """A partial batch is nearly always an SFT collator called by mistake; say so at once."""
    batch = collator(make_pairs(2))
    del batch["rejected_labels"]
    del batch["rejected_attention_mask"]
    with pytest.raises(KeyError, match="rejected_attention_mask"):
        concatenated_batch(batch)


def test_concatenated_batch_refuses_halves_of_different_widths(
    collator: PreferenceCollator,
) -> None:
    batch = collator(make_pairs(2))
    batch["rejected_input_ids"] = batch["rejected_input_ids"][:, :-1]
    with pytest.raises(ValueError, match="must be padded to one width"):
        concatenated_batch(batch)


def test_pair_keys_are_exactly_what_the_collator_emits(collator: PreferenceCollator) -> None:
    assert set(PAIR_KEYS) == set(collator(make_pairs(1)))


# --------------------------------------------------------------------------------------
# The reference policy: one model, adapter off
# --------------------------------------------------------------------------------------


def test_policy_log_probabilities_carry_gradients(collator: PreferenceCollator) -> None:
    model = adapted_model()
    chosen, rejected = policy_logprobs(model, collator(make_pairs(3)))
    assert chosen.requires_grad is True
    assert rejected.requires_grad is True
    assert tuple(chosen.shape) == (3,)


def test_reference_log_probabilities_do_not(collator: PreferenceCollator) -> None:
    """`torch.no_grad()` lives inside `reference_logprobs` so the two cannot be separated."""
    model = adapted_model()
    chosen, rejected = reference_logprobs(model, collator(make_pairs(3)))
    assert chosen.requires_grad is False
    assert rejected.requires_grad is False


def test_a_fresh_adapter_makes_policy_and_reference_identical(
    collator: PreferenceCollator,
) -> None:
    """LoRA initialises `B` to zero, so the two forward passes are the same function."""
    model = adapted_model()
    model.eval()
    batch = collator(make_pairs(4))
    policy_chosen, policy_rejected = policy_logprobs(model, batch)
    ref_chosen, ref_rejected = reference_logprobs(model, batch)
    assert torch.equal(policy_chosen.detach(), ref_chosen)
    assert torch.equal(policy_rejected.detach(), ref_rejected)


def test_the_reference_reproduces_a_separately_measured_base_model(
    collator: PreferenceCollator,
) -> None:
    """The claim that lets a DPO run hold one model instead of two.

    The base log-probabilities are recorded before any adapter exists. The adapter is then
    attached and deliberately moved off zero, and the reference pass has to hand those recorded
    numbers back bit for bit -- while the policy pass has visibly moved away from them. A
    reference that merely happened to be close would not license dropping the second copy.
    """
    base = tiny_model(3)
    base.eval()
    batch = collator(make_pairs(3))
    with torch.no_grad():
        base_chosen, base_rejected = policy_logprobs(base, batch)

    model = attach(base, seed=3)
    move_adapter(model)
    model.eval()
    policy_chosen, _ = policy_logprobs(model, batch)
    ref_chosen, ref_rejected = reference_logprobs(model, batch)

    assert float((policy_chosen.detach() - base_chosen).abs().max()) > 0.01
    assert torch.equal(ref_chosen, base_chosen)
    assert torch.equal(ref_rejected, base_rejected)


def test_a_moved_adapter_separates_policy_from_reference(collator: PreferenceCollator) -> None:
    model = adapted_model(4)
    move_adapter(model)
    model.eval()
    batch = collator(make_pairs(3))
    policy_chosen, _ = policy_logprobs(model, batch)
    ref_chosen, _ = reference_logprobs(model, batch)
    assert not torch.equal(policy_chosen.detach(), ref_chosen)


def test_reference_needs_an_adapter_to_disable(collator: PreferenceCollator) -> None:
    """Without an adapter the "reference" would be the policy and the loss would sit at ln 2."""
    with pytest.raises(TypeError, match="no disable_adapter"):
        reference_logprobs(tiny_model(), collator(make_pairs(2)))


def test_length_normalisation_divides_by_the_scored_token_count(
    collator: PreferenceCollator,
) -> None:
    model = adapted_model()
    model.eval()
    batch = collator(make_pairs(3, completion_len=6))
    summed, _ = policy_logprobs(model, batch)
    averaged, _ = policy_logprobs(model, batch, length_normalise=True)
    scored = (batch["chosen_labels"][:, 1:] != IGNORE_INDEX).sum(dim=-1)
    assert torch.equal(scored, torch.full((3,), 6))
    assert torch.allclose(averaged.detach() * scored, summed.detach(), atol=1e-5)


# --------------------------------------------------------------------------------------
# The step-0 identity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 1.0, 10.0])
def test_step_zero_loss_is_exactly_ln_two(collator: PreferenceCollator, beta: float) -> None:
    """The strongest correctness check available without a second implementation.

    Policy and reference coincide, so `h` is identically zero and the sigmoid loss is
    `-log sigma(0) = ln 2` for every beta. Exact equality, not `approx`: if the concatenation,
    the masking, the shift or the reference context were wrong, the two log-ratios would not
    cancel and this number would move.
    """
    model = adapted_model()
    model.eval()
    output = dpo_batch_output(model, collator(make_pairs(4)), beta=beta)
    assert float(output.loss.detach()) == LN2_FLOAT32


def test_step_zero_margins_are_exactly_zero(collator: PreferenceCollator) -> None:
    model = adapted_model()
    model.eval()
    output = dpo_batch_output(model, collator(make_pairs(4)), beta=0.1)
    assert float(output.reward_margins.abs().max()) == 0.0
    assert float(output.chosen_rewards.abs().max()) == 0.0
    assert float(output.rejected_rewards.abs().max()) == 0.0


def test_step_zero_reward_accuracy_is_zero_because_a_tie_is_a_miss(
    collator: PreferenceCollator,
) -> None:
    """A policy that cannot separate a pair has not learned it, so 0.0 and not 0.5."""
    model = adapted_model()
    model.eval()
    output = dpo_batch_output(model, collator(make_pairs(4)), beta=0.1)
    assert float(output.reward_accuracy) == 0.0


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.3, 0.45])
def test_cdpo_at_step_zero_is_ln_two_at_every_smoothing(
    collator: PreferenceCollator, label_smoothing: float
) -> None:
    """At a zero margin the smoothed mixture weights `-log sigma(0)` by `(1 - e) + e`."""
    model = adapted_model()
    model.eval()
    output = dpo_batch_output(
        model,
        collator(make_pairs(4)),
        beta=0.1,
        variant="cdpo",
        label_smoothing=label_smoothing,
    )
    assert float(output.loss.detach()) == pytest.approx(LN2_FLOAT32, abs=1e-6)


@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0])
def test_ipo_at_step_zero_is_the_squared_target_margin(
    collator: PreferenceCollator, beta: float
) -> None:
    """IPO regresses `h` onto `1 / (2 beta)`, so a zero margin costs that target squared."""
    model = adapted_model()
    model.eval()
    output = dpo_batch_output(model, collator(make_pairs(4)), beta=beta, variant="ipo")
    assert float(output.loss.detach()) == pytest.approx((1.0 / (2.0 * beta)) ** 2)


def test_the_step_zero_gradient_reaches_the_adapter_only(collator: PreferenceCollator) -> None:
    """The loss is flat in value at step 0 but not in gradient: `dL/dh` is `-beta / 2`."""
    model = adapted_model()
    output = dpo_batch_output(model, collator(make_pairs(4)), beta=0.1)
    output.loss.backward()
    moved = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and float(parameter.grad.abs().max()) > 0.0
    }
    assert moved
    assert all("lora_" in name for name in moved)


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def test_reward_point_serialises_under_the_names_a_dpo_run_is_read_by() -> None:
    point = RewardPoint(step=4, chosen=0.3, rejected=-0.2, margin=0.5, accuracy=0.75)
    assert point.as_dict() == {
        "step": 4,
        "reward_chosen": 0.3,
        "reward_rejected": -0.2,
        "reward_margin": 0.5,
        "reward_accuracy": 0.75,
    }


def test_reward_point_rejects_a_negative_step() -> None:
    with pytest.raises(ValueError, match="step must not be negative"):
        RewardPoint(step=-1, chosen=0.0, rejected=0.0, margin=0.0, accuracy=0.5)


@pytest.mark.parametrize("accuracy", [-0.1, 1.1])
def test_reward_point_rejects_an_impossible_accuracy(accuracy: float) -> None:
    with pytest.raises(ValueError, match=r"accuracy must lie in \[0, 1\]"):
        RewardPoint(step=1, chosen=0.0, rejected=0.0, margin=0.0, accuracy=accuracy)


def test_slope_of_a_straight_line_is_its_gradient() -> None:
    assert least_squares_slope([(0, 1.0), (1, 3.0), (2, 5.0)]) == pytest.approx(2.0)


def test_slope_of_a_single_point_is_zero() -> None:
    """One measurement is not a trend, and the report asks for the slope regardless."""
    assert least_squares_slope([(3, 1.0)]) == 0.0
    assert least_squares_slope([]) == 0.0


def test_slope_of_a_vertical_stack_is_zero() -> None:
    """Every point at the same step has no slope; returning nan would poison the summary."""
    assert least_squares_slope([(2, 1.0), (2, 5.0)]) == 0.0


@given(
    slope=st.floats(min_value=-5.0, max_value=5.0),
    intercept=st.floats(min_value=-10.0, max_value=10.0),
    count=st.integers(min_value=2, max_value=20),
)
@PROPERTY_SETTINGS
def test_slope_recovers_the_line_it_was_given(slope: float, intercept: float, count: int) -> None:
    points = [(step, slope * step + intercept) for step in range(count)]
    assert least_squares_slope(points) == pytest.approx(slope, abs=1e-9)


@given(
    values=st.lists(st.floats(-20.0, 20.0), min_size=2, max_size=15),
    shift=st.floats(-50.0, 50.0),
)
@PROPERTY_SETTINGS
def test_slope_is_unchanged_by_shifting_every_value(values: list[float], shift: float) -> None:
    """A trend is about differences, so a constant offset in the rewards must not move it."""
    points = list(enumerate(values))
    shifted = [(step, value + shift) for step, value in points]
    assert least_squares_slope(shifted) == pytest.approx(least_squares_slope(points), abs=1e-6)


def test_evaluation_serialises() -> None:
    evaluation = DPOEvaluation(loss=0.5, accuracy=0.75, margin=1.25, pairs=8)
    assert evaluation.as_dict() == {"loss": 0.5, "accuracy": 0.75, "margin": 1.25, "pairs": 8}


# --------------------------------------------------------------------------------------
# Held-out evaluation
# --------------------------------------------------------------------------------------


def test_evaluation_of_a_fresh_adapter_is_the_step_zero_identity(
    collator: PreferenceCollator,
) -> None:
    evaluation = evaluate_dpo(adapted_model(), make_pairs(8), collator, batch_size=4, beta=0.2)
    assert evaluation.loss == LN2_FLOAT32
    assert evaluation.accuracy == 0.0
    assert evaluation.margin == 0.0
    assert evaluation.pairs == 8


def test_evaluation_does_not_depend_on_the_batch_size(collator: PreferenceCollator) -> None:
    """Pair-weighted averaging is what makes two batchings of the same pairs agree."""
    model = adapted_model(1)
    move_adapter(model, 0.03)
    pairs = make_pairs(6, completion_len=3) + make_pairs(3, completion_len=6, start=6)
    at_one = evaluate_dpo(model, pairs, collator, batch_size=1)
    at_four = evaluate_dpo(model, pairs, collator, batch_size=4)
    assert at_one.loss == pytest.approx(at_four.loss, rel=1e-5)
    assert at_one.margin == pytest.approx(at_four.margin, rel=1e-5, abs=1e-9)
    assert at_one.accuracy == at_four.accuracy
    assert at_one.pairs == at_four.pairs == 9


def test_evaluation_restores_the_training_mode(collator: PreferenceCollator) -> None:
    model = adapted_model()
    model.train()
    evaluate_dpo(model, make_pairs(2), collator, batch_size=2)
    assert model.training is True
    model.eval()
    evaluate_dpo(model, make_pairs(2), collator, batch_size=2)
    assert model.training is False


def test_evaluation_refuses_an_empty_pair_set(collator: PreferenceCollator) -> None:
    with pytest.raises(ValueError, match="at least one preference pair"):
        evaluate_dpo(adapted_model(), [], collator, batch_size=2)


def test_evaluation_passes_the_variant_through(collator: PreferenceCollator) -> None:
    evaluation = evaluate_dpo(
        adapted_model(), make_pairs(4), collator, batch_size=2, beta=0.5, variant="ipo"
    )
    assert evaluation.loss == pytest.approx(1.0)


def test_evaluation_under_bf16_autocast_stays_finite(collator: PreferenceCollator) -> None:
    model = adapted_model(2)
    move_adapter(model, 0.02)
    evaluation = evaluate_dpo(model, make_pairs(4), collator, batch_size=2, bf16=True)
    assert math.isfinite(evaluation.loss)
    assert math.isfinite(evaluation.margin)


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


def test_the_run_starts_at_ln_two(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """The first logged loss is measured before the first update, so the identity survives."""
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(8),
        make_config(tmp_path, max_steps=3),
        beta=0.1,
        collator=collator,
    )
    assert result.log.records[0].loss == LN2_FLOAT32
    assert result.rewards[0].margin == 0.0
    assert result.rewards[0].accuracy == 0.0


def test_reward_accuracy_rises_from_a_tie_to_the_whole_batch(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """The headline claim of a DPO stage, on pairs that are separable by construction."""
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(8),
        make_config(tmp_path, max_steps=40),
        beta=0.1,
        collator=collator,
    )
    assert result.initial_accuracy == 0.0
    assert result.final_accuracy == 1.0
    assert result.accuracy_gain == 1.0
    tail = [point.accuracy for point in result.rewards[-10:]]
    assert sum(tail) / len(tail) == 1.0
    assert result.final_loss < LN2_FLOAT32


def test_the_reward_margin_increases_in_expectation(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """Increasing in expectation, not step by step: a batch of harder pairs dents the margin.

    Asserting step-wise monotonicity would produce a test that fails on a healthy run, so the
    claim is made three honest ways instead -- a positive least-squares trend, a second half
    that averages above the first, and a chosen reward that ends above the rejected one.
    """
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(8),
        make_config(tmp_path, max_steps=40),
        beta=0.1,
        collator=collator,
    )
    margins = [point.margin for point in result.rewards]
    first_half = margins[: len(margins) // 2]
    second_half = margins[len(margins) // 2 :]
    assert result.margin_slope > 0.0
    assert sum(second_half) / len(second_half) > sum(first_half) / len(first_half)
    assert margins[-1] > margins[0]
    assert result.rewards[-1].chosen > result.rewards[-1].rejected


def test_the_preference_rule_generalises_to_held_out_pairs(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """Accuracy on pairs the run never saw climbs from roughly chance towards one.

    The training pairs are separable by construction, so ranking them is not by itself evidence
    that anything was learned -- memorising eight prompts would do it. These eight pairs follow
    the same rule with prompts the run has never seen, and the implicit reward has to rank them
    too before the claim means anything.
    """
    held_out = make_pairs(8, start=40)
    short = adapted_model()
    train_dpo(
        short,
        tokenizer,
        make_pairs(8),
        make_config(tmp_path, max_steps=3),
        beta=0.1,
        collator=collator,
    )
    early = evaluate_dpo(short, held_out, collator, batch_size=4, beta=0.1)

    long = adapted_model()
    train_dpo(
        long,
        tokenizer,
        make_pairs(8),
        make_config(tmp_path, max_steps=40),
        beta=0.1,
        collator=collator,
    )
    late = evaluate_dpo(long, held_out, collator, batch_size=4, beta=0.1)

    assert late.accuracy == 1.0
    assert late.accuracy > early.accuracy or late.margin > early.margin
    assert late.margin > early.margin > 0.0
    assert late.loss < early.loss < LN2_FLOAT32


def test_two_runs_with_the_same_seed_are_identical(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Same seed, same pairs, same losses and same rewards -- to the last bit, on CPU."""
    pairs = make_pairs(6)

    def run() -> list[float]:
        result = train_dpo(
            adapted_model(),
            tokenizer,
            pairs,
            make_config(tmp_path, max_steps=6, batch_size=2),
            beta=0.1,
            collator=PreferenceCollator(pad_token_id=0, max_length=SEQ_BUDGET),
        )
        return [record.loss for record in result.log.records] + [
            point.margin for point in result.rewards
        ]

    assert run() == run()


def test_a_different_seed_changes_the_run(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    pairs = make_pairs(8)

    def run(seed: int) -> list[float]:
        result = train_dpo(
            adapted_model(),
            tokenizer,
            pairs,
            make_config(tmp_path, max_steps=4, batch_size=2, seed=seed),
            beta=0.1,
            collator=PreferenceCollator(pad_token_id=0, max_length=SEQ_BUDGET),
        )
        return [record.loss for record in result.log.records]

    assert run(0) != run(7)


def test_gradient_accumulation_equals_one_larger_batch(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Four micro-batches of two must produce the gradient of one batch of eight.

    The pairs are all the same length here so the two batchings pad to the same width, which
    makes the comparison exact rather than merely close: the accumulated gradient norm agrees
    bit for bit. The usual "divide each micro-batch mean by the number of micro-batches" recipe
    does not have that property as soon as a group is short.
    """

    def run(batch_size: int, accumulation: int) -> tuple[float, float]:
        result = train_dpo(
            adapted_model(2),
            tokenizer,
            make_pairs(8),
            make_config(
                tmp_path,
                max_steps=1,
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
            ),
            beta=0.1,
            collator=PreferenceCollator(pad_token_id=0, max_length=SEQ_BUDGET),
        )
        record = result.log.records[0]
        return record.loss, record.grad_norm

    assert run(8, 1) == run(2, 4)


def test_the_logged_rate_follows_the_configured_schedule(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    config = make_config(tmp_path, max_steps=10, warmup_ratio=0.3, learning_rate=1e-3)
    result = train_dpo(
        adapted_model(), tokenizer, make_pairs(4), config, beta=0.1, collator=collator
    )
    for record in result.log.records:
        assert record.lr == pytest.approx(config.lr_at(record.step - 1, 10))
    assert result.log.records[0].lr == pytest.approx(0.0)


def test_logging_honours_its_interval_and_always_records_the_last_step(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4),
        make_config(tmp_path, max_steps=7, log_every_steps=3),
        beta=0.1,
        collator=collator,
    )
    assert [record.step for record in result.log.records] == [3, 6, 7]
    assert [point.step for point in result.rewards] == [3, 6, 7]


def test_token_counts_cover_both_sides_and_accumulate(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4, completion_len=5),
        make_config(tmp_path, max_steps=3),
        beta=0.1,
        collator=collator,
    )
    per_step = 4 * 5 * 2
    assert result.supervised_tokens == 3 * per_step
    assert result.pairs_seen == 3 * 4
    assert result.log.curve("tokens") == [(1, per_step), (2, 2 * per_step), (3, 3 * per_step)]


def test_epochs_and_max_steps_both_control_the_length(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    pairs = make_pairs(8)
    by_epochs = train_dpo(
        adapted_model(),
        tokenizer,
        pairs,
        make_config(tmp_path, epochs=3),
        beta=0.1,
        collator=PreferenceCollator(pad_token_id=0, max_length=SEQ_BUDGET),
    )
    by_cap = train_dpo(
        adapted_model(),
        tokenizer,
        pairs,
        make_config(tmp_path, epochs=3, max_steps=2),
        beta=0.1,
        collator=PreferenceCollator(pad_token_id=0, max_length=SEQ_BUDGET),
    )
    assert by_epochs.steps == 6
    assert by_cap.steps == 2


def test_more_steps_than_the_data_supports_cycles_epochs(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(2),
        make_config(tmp_path, max_steps=5, batch_size=2),
        beta=0.1,
        collator=collator,
    )
    assert result.steps == 5
    assert result.pairs_seen == 10


def test_a_trainable_parameter_the_forward_pass_never_uses_is_tolerated(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """A tensor outside the graph gets no gradient, and the step still has to happen.

    The gradient normalisation walks every trainable parameter, so a `None` gradient there is
    the difference between a run that steps and a run that dies on an attribute error.
    """
    model = adapted_model()
    model.unused_head = torch.nn.Parameter(torch.zeros(4))
    result = train_dpo(
        model,
        tokenizer,
        make_pairs(4),
        make_config(tmp_path, max_steps=2),
        beta=0.1,
        collator=collator,
    )
    assert result.steps == 2
    assert model.unused_head.grad is None
    assert torch.equal(model.unused_head.detach(), torch.zeros(4))


def test_only_the_adapter_moves(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """The LoRA promise, checked end to end: the base weights come out bit-identical.

    It matters more here than in SFT. The reference policy *is* these frozen weights, so a run
    that nudged them would be optimising against a moving reference and every implicit reward
    in the log would be measured from a different origin at every step.
    """
    model = adapted_model()
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    train_dpo(
        model,
        tokenizer,
        make_pairs(8),
        make_config(tmp_path, max_steps=5),
        beta=0.1,
        collator=collator,
    )
    after = dict(model.named_parameters())
    assert frozen
    assert all(torch.equal(value, after[name].detach()) for name, value in frozen.items())


@pytest.mark.parametrize("variant", ["sigmoid", "ipo", "cdpo"])
def test_every_variant_trains(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path, variant: DPOVariant
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(8),
        make_config(tmp_path, max_steps=6),
        beta=0.1,
        variant=variant,
        label_smoothing=0.1 if variant == "cdpo" else 0.0,
        collator=collator,
    )
    assert result.variant == variant
    assert result.steps == 6
    assert not result.log.diverged
    assert result.rewards[-1].margin > 0.0


def test_training_refuses_an_empty_pair_set(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="at least one preference pair"):
        train_dpo(
            adapted_model(),
            tokenizer,
            [],
            make_config(tmp_path, max_steps=1),
            beta=0.1,
            collator=collator,
        )


def test_training_refuses_a_model_with_nothing_to_train(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """A frozen model would report a flat ln 2 forever: the policy never leaves the reference."""
    model = adapted_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="no parameter has requires_grad"):
        train_dpo(
            model,
            tokenizer,
            make_pairs(2),
            make_config(tmp_path, max_steps=1),
            beta=0.1,
            collator=collator,
        )


def test_training_refuses_a_model_without_an_adapter(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match="no disable_adapter"):
        train_dpo(
            tiny_model(),
            tokenizer,
            make_pairs(2),
            make_config(tmp_path, max_steps=1),
            beta=0.1,
            collator=collator,
        )


def test_bf16_training_stays_finite(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4),
        make_config(tmp_path, max_steps=3, bf16=True),
        beta=0.1,
        collator=collator,
    )
    assert not result.log.diverged
    assert all(math.isfinite(record.grad_norm) for record in result.log.records)
    assert all(math.isfinite(point.margin) for point in result.rewards)


def test_training_from_mined_preference_pairs_uses_the_chat_formatter(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """The real path: `PreferencePair`s straight from the mining stage, no explicit collator."""
    pairs = [make_preference_pair(index) for index in range(4)]
    result = train_dpo(
        adapted_model(),
        tokenizer,
        pairs,
        make_config(tmp_path, max_steps=2, batch_size=2),
        beta=0.1,
        formatter=formatter(),
    )
    assert result.steps == 2
    assert result.supervised_tokens > 0
    assert result.collation.seen == 4
    assert result.collation.dropped == 0


# --------------------------------------------------------------------------------------
# Artefacts, the manifest and the result object
# --------------------------------------------------------------------------------------


def test_a_tight_sequence_budget_is_reported_not_hidden(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Truncation counts sequences while the drop count counts pairs; both reach the result."""
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4, completion_len=8),
        make_config(tmp_path, max_steps=1),
        beta=0.1,
        collator=PreferenceCollator(pad_token_id=0, max_length=6),
    )
    assert result.collation.seen == 4
    assert result.collation.truncated == 8
    assert result.collation.dropped == 0


def test_the_final_adapter_is_written_when_saving_is_on(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """A DPO run has no validation split, so the weights at the end are the result."""
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4),
        make_config(tmp_path, max_steps=1, save_best_adapter=True),
        beta=0.1,
        collator=collator,
    )
    assert result.adapter_path == tmp_path / "final"
    assert (tmp_path / "final" / "adapter_config.json").exists()


def test_nothing_is_written_when_saving_is_off(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4),
        make_config(tmp_path, max_steps=1),
        beta=0.1,
        collator=collator,
    )
    assert result.adapter_path is None
    assert list(tmp_path.iterdir()) == []


def test_a_result_writes_both_logs_the_manifest_and_the_summary(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4),
        make_config(tmp_path, max_steps=2),
        beta=0.25,
        collator=collator,
    )
    directory = result.save(tmp_path / "artefacts")

    assert TrainLog.from_jsonl(directory / LOG_FILENAME).records == result.log.records
    assert RunManifest.from_json(directory / MANIFEST_FILENAME) == result.manifest
    rows = [
        json.loads(line)
        for line in (directory / REWARD_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[0]["reward_margin"] == 0.0
    summary = json.loads((directory / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["stage"] == "dpo"
    assert summary["beta"] == 0.25
    assert summary["variant"] == "sigmoid"
    assert summary["steps"] == 2
    assert summary["collation"]["seen"] == 8


def test_the_reward_log_uses_unix_newlines(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    """A log written here and hashed on Linux must be the same bytes."""
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(4),
        make_config(tmp_path, max_steps=2),
        beta=0.1,
        collator=collator,
    )
    directory = result.save(tmp_path / "artefacts")
    assert b"\r\n" not in (directory / REWARD_LOG_FILENAME).read_bytes()


def test_the_manifest_records_the_dpo_stage_and_its_inputs(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    config = make_config(tmp_path, max_steps=1)
    result = train_dpo(
        adapted_model(), tokenizer, make_pairs(4), config, beta=0.1, collator=collator
    )
    assert result.manifest.stage == "dpo"
    assert result.manifest.config == config
    assert result.manifest.model_name == "PeftModelForCausalLM"
    assert 0 < result.manifest.trainable_parameters < result.manifest.total_parameters
    assert result.manifest.data_content_hash


def test_the_data_hash_covers_the_rejected_side_too(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Two runs with the same chosen completions but different rejected ones are not the same run."""
    config = make_config(tmp_path, max_steps=1)
    baseline = make_pairs(4)
    swapped = [
        PairFeature.from_shared(
            pair.chosen.prompt_ids,
            pair.chosen.completion_ids,
            tuple(token + 1 for token in pair.rejected.completion_ids),
        )
        for pair in baseline
    ]

    def run(pairs: list[PairFeature]) -> str:
        result = train_dpo(
            adapted_model(),
            tokenizer,
            pairs,
            config,
            beta=0.1,
            collator=PreferenceCollator(pad_token_id=0, max_length=SEQ_BUDGET),
        )
        return result.manifest.data_content_hash

    assert run(baseline) != run(swapped)


def test_an_explicit_hash_and_model_name_win(
    tokenizer: TinyTokenizer, collator: PreferenceCollator, tmp_path: Path
) -> None:
    result = train_dpo(
        adapted_model(),
        tokenizer,
        make_pairs(2),
        make_config(tmp_path, max_steps=1, batch_size=2),
        beta=0.1,
        collator=collator,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        data_content_hash="corpus-deadbeef",
    )
    assert result.manifest.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert result.manifest.data_content_hash == "corpus-deadbeef"


def test_an_empty_result_reports_nan_rather_than_crashing(tmp_path: Path) -> None:
    """The report reads these before it knows whether the run produced anything."""
    empty = DPOResult(
        log=TrainLog(),
        rewards=(),
        steps=0,
        pairs_seen=0,
        supervised_tokens=0,
        beta=0.1,
        variant="sigmoid",
        manifest=RunManifest(
            stage="dpo",
            model_name="tiny",
            data_content_hash="abc",
            trainable_parameters=1,
            total_parameters=2,
            config=make_config(tmp_path),
        ),
        adapter_path=None,
        collation=PreferenceCollator(pad_token_id=0).report,
    )
    assert math.isnan(empty.final_loss)
    assert math.isnan(empty.initial_accuracy)
    assert math.isnan(empty.final_accuracy)
    assert math.isnan(empty.accuracy_gain)
    assert empty.margin_slope == 0.0
    assert empty.summary()["adapter_path"] is None


def test_a_saved_adapter_path_reaches_the_summary(tmp_path: Path) -> None:
    result = DPOResult(
        log=TrainLog(),
        rewards=(RewardPoint(step=1, chosen=0.1, rejected=-0.1, margin=0.2, accuracy=1.0),),
        steps=1,
        pairs_seen=4,
        supervised_tokens=32,
        beta=0.1,
        variant="ipo",
        manifest=RunManifest(
            stage="dpo",
            model_name="tiny",
            data_content_hash="abc",
            trainable_parameters=1,
            total_parameters=2,
            config=make_config(tmp_path),
        ),
        adapter_path=tmp_path / "final",
        collation=PreferenceCollator(pad_token_id=0).report,
    )
    summary = result.summary()
    assert summary["adapter_path"] == str(tmp_path / "final")
    assert summary["variant"] == "ipo"
    assert summary["final_reward_accuracy"] == 1.0
    assert summary["reward_margin_slope"] == 0.0


def test_sequence_logprob_reads_the_batch_the_collator_built(
    collator: PreferenceCollator,
) -> None:
    """The trainer and the loss module agree about which positions are scored.

    `policy_logprobs` is a thin wrapper over one forward pass and `sequence_logprob`; this pins
    that down against the same reduction computed from the model's logits by hand, so a future
    change to the wrapper cannot quietly start scoring the prompt.
    """
    model = adapted_model()
    model.eval()
    batch = collator(make_pairs(3))
    input_ids, attention_mask, labels = concatenated_batch(batch)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        expected = sequence_logprob(logits, labels)
        chosen, rejected = policy_logprobs(model, batch)
    assert torch.equal(torch.cat([chosen, rejected]), expected)
