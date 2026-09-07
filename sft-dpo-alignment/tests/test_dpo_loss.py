"""Tests for the DPO objective.

The objective is pure mathematics, so almost everything here is an identity rather than a
recorded number: `policy == reference` gives ln 2 for any beta, swapping the two responses
adds the margin to the loss, cDPO at zero smoothing is the original objective, and the
minimum of cDPO is the binary entropy of the smoothing rate. Identities catch a sign error
that a golden value chosen to match the implementation would happily enshrine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from torch import Tensor

from sftdpo.train.dpo_loss import IGNORE_INDEX, DPOOutput, dpo_loss, sequence_logprob

LN2 = math.log(2.0)
LN2_FLOAT32 = 0.6931471824645996
"""ln 2 rounded to float32 and read back as a double, verified on this machine."""

VOCAB = 4
TARGET = 1


def _build(
    token_probs: Sequence[Sequence[float | None]], vocab: int = VOCAB
) -> tuple[Tensor, Tensor]:
    """Build ``(logits, labels)`` with exactly the requested per-token probabilities.

    ``token_probs[b][i]`` is the probability the model should assign to the ``i``-th scored
    token of row ``b``, or ``None`` for a position that is masked out. The logits are the
    logarithms of a normalised distribution, and ``log_softmax`` of a log-probability vector
    is that vector back again, so the scored values are exact rather than approximate.

    The returned sequence has one more position than ``token_probs``: scored token ``i``
    lives at label position ``i + 1``, predicted by logit column ``i``. Building the fixture
    this way rather than by slicing keeps the test data honest about the shift.
    """
    batch = len(token_probs)
    width = len(token_probs[0])
    probs = torch.full((batch, width + 1, vocab), 1.0 / vocab, dtype=torch.float64)
    labels = torch.full((batch, width + 1), IGNORE_INDEX, dtype=torch.long)
    for b, row in enumerate(token_probs):
        for i, p in enumerate(row):
            if p is None:
                continue
            labels[b, i + 1] = TARGET
            probs[b, i] = (1.0 - p) / (vocab - 1)
            probs[b, i, TARGET] = p
    return torch.log(probs), labels


def _logps(*values: float) -> tuple[Tensor, ...]:
    """Four (or fewer) 1-D float64 log-probability tensors from plain numbers."""
    return tuple(torch.tensor([v], dtype=torch.float64) for v in values)


def _grads(
    chosen: float,
    rejected: float,
    *,
    ref_chosen: float = 0.0,
    ref_rejected: float = 0.0,
    beta: float = 0.1,
    variant: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[float, float, float, float]:
    """Gradients of the loss with respect to all four log-probabilities."""
    tensors = [
        torch.tensor([v], dtype=torch.float64, requires_grad=True)
        for v in (chosen, rejected, ref_chosen, ref_rejected)
    ]
    out = dpo_loss(
        *tensors,
        beta=beta,
        variant=variant,  # type: ignore[arg-type]
        label_smoothing=label_smoothing,
    )
    out.loss.backward()
    return tuple(float(t.grad) for t in tensors)  # type: ignore[return-value]


def _softplus(x: float) -> float:
    """-log sigma(-x) in plain Python, written the other stable way round to `logsigmoid`."""
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def _loss_at_logit(h: float, *, beta: float, variant: str, label_smoothing: float = 0.0) -> float:
    """Closed-form loss for a given h, derived independently of the module under test."""
    u = beta * h
    if variant == "ipo":
        return (h - 1.0 / (2.0 * beta)) ** 2
    return (1.0 - label_smoothing) * _softplus(-u) + label_smoothing * _softplus(u)


# ----- sequence_logprob: the shift ------------------------------------------------------


def test_certain_distribution_scores_zero() -> None:
    logits, labels = _build([[1.0, 1.0, 1.0]])
    assert float(sequence_logprob(logits, labels)) == pytest.approx(0.0, abs=1e-12)


def test_matches_hand_computed_log_softmax() -> None:
    logits, labels = _build([[0.5, 0.25, 0.125]])
    expected = math.log(0.5) + math.log(0.25) + math.log(0.125)
    assert float(sequence_logprob(logits, labels)) == pytest.approx(expected, abs=1e-12)


def test_matches_a_hand_written_log_softmax_on_raw_logits() -> None:
    """Raw integer logits, log-softmax re-derived in plain Python, shift applied by hand."""
    rows = [[0.0, 1.0, 2.0], [1.0, 0.0, -1.0], [5.0, 5.0, 5.0]]
    logits = torch.tensor([rows], dtype=torch.float64)
    labels = torch.tensor([[IGNORE_INDEX, 2, 0]], dtype=torch.long)

    def logp(row: list[float], token: int) -> float:
        return row[token] - math.log(sum(math.exp(x) for x in row))

    # Column 0 predicts label position 1; column 1 predicts label position 2; column 2 is
    # dropped by the shift, which is why its uniform row cannot appear in the total.
    expected = logp(rows[0], 2) + logp(rows[1], 0)
    assert float(sequence_logprob(logits, labels)) == pytest.approx(expected, abs=1e-12)


def test_returns_one_score_per_row() -> None:
    logits, labels = _build([[0.5, 0.5], [0.25, 0.25]])
    assert sequence_logprob(logits, labels).shape == (2,)


def test_first_label_is_never_scored() -> None:
    """Nothing in the sequence predicts position 0, so its label must not move the result."""
    logits, labels = _build([[0.5, 0.25]])
    before = float(sequence_logprob(logits, labels))
    labels[0, 0] = TARGET
    assert float(sequence_logprob(logits, labels)) == before


def test_last_logit_column_is_never_used() -> None:
    """The final column predicts a token past the end of the sequence."""
    logits, labels = _build([[0.5, 0.25]])
    before = float(sequence_logprob(logits, labels))
    logits[0, -1] = torch.tensor([100.0, -100.0, 0.0, 0.0], dtype=torch.float64)
    assert float(sequence_logprob(logits, labels)) == before


def test_unshifted_gather_would_give_a_different_answer() -> None:
    """Guards the off-by-one: the naive same-index gather must not agree by accident."""
    logits, labels = _build([[0.5, 0.25, 0.125]])
    scored = labels != IGNORE_INDEX
    naive = (
        torch.log_softmax(logits, dim=-1)
        .gather(2, labels.masked_fill(~scored, 0).unsqueeze(2))
        .squeeze(2)
        .masked_fill(~scored, 0.0)
        .sum(dim=-1)
    )
    assert float(naive) != pytest.approx(float(sequence_logprob(logits, labels)), abs=1e-6)


def test_gradient_reaches_the_predicting_columns_only() -> None:
    logits, labels = _build([[0.5, 0.25]])
    logits = logits.detach().requires_grad_(True)
    sequence_logprob(logits, labels).sum().backward()
    assert logits.grad is not None
    assert torch.any(logits.grad[:, :-1] != 0.0)
    assert torch.all(logits.grad[:, -1] == 0.0)


# ----- sequence_logprob: masking --------------------------------------------------------


def test_ignored_positions_contribute_nothing() -> None:
    logits, dense_labels = _build([[0.5, 0.75, 0.25]])
    masked_labels = dense_labels.clone()
    masked_labels[0, 2] = IGNORE_INDEX
    assert float(sequence_logprob(logits, dense_labels)) == pytest.approx(
        math.log(0.5) + math.log(0.75) + math.log(0.25), abs=1e-12
    )
    assert float(sequence_logprob(logits, masked_labels)) == pytest.approx(
        math.log(0.5) + math.log(0.25), abs=1e-12
    )


def test_ignored_positions_cannot_leak_nan_from_minus_inf_logits() -> None:
    """A padded column whose substitute token has probability zero gathers -inf.

    Multiplying by a zero mask would turn that into nan; masking with `masked_fill` cannot.
    """
    probs = torch.tensor(
        [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]]],
        dtype=torch.float64,
    )
    labels = torch.tensor([[IGNORE_INDEX, TARGET, IGNORE_INDEX]], dtype=torch.long)
    total = sequence_logprob(torch.log(probs), labels)
    assert torch.isfinite(total).all()
    assert float(total) == pytest.approx(math.log(0.5), abs=1e-12)


def test_fully_ignored_row_sums_to_zero() -> None:
    logits, labels = _build([[None, None], [0.5, 0.5]])
    assert float(sequence_logprob(logits, labels)[0]) == 0.0


def test_fully_ignored_row_averages_to_zero_rather_than_nan() -> None:
    logits, labels = _build([[None, None], [0.5, 0.5]])
    averaged = sequence_logprob(logits, labels, average=True)
    assert torch.isfinite(averaged).all()
    assert float(averaged[0]) == 0.0


def test_average_equals_sum_over_scored_count() -> None:
    logits, labels = _build([[0.5, 0.25, None], [0.125, 0.5, 0.5]])
    total = sequence_logprob(logits, labels)
    averaged = sequence_logprob(logits, labels, average=True)
    counts = torch.tensor([2.0, 3.0], dtype=torch.float64)
    assert torch.allclose(averaged, total / counts)


def test_rows_are_independent() -> None:
    logits, labels = _build([[0.5, 0.25], [0.125, 0.75]])
    batched = sequence_logprob(logits, labels)
    for i in range(2):
        alone = sequence_logprob(logits[i : i + 1], labels[i : i + 1])
        assert float(alone) == pytest.approx(float(batched[i]), abs=1e-12)


def test_mask_partition_is_additive() -> None:
    """Splitting the scored tokens across two calls must sum to the whole."""
    logits, labels = _build([[0.5, 0.25, 0.125, 0.75]])
    left = labels.clone()
    right = labels.clone()
    left[:, 3:] = IGNORE_INDEX
    right[:, :3] = IGNORE_INDEX
    whole = float(sequence_logprob(logits, labels))
    parts = float(sequence_logprob(logits, left)) + float(sequence_logprob(logits, right))
    assert parts == pytest.approx(whole, abs=1e-12)


def test_inputs_are_not_mutated() -> None:
    logits, labels = _build([[0.5, None, 0.25]])
    logits_before = logits.clone()
    labels_before = labels.clone()
    sequence_logprob(logits, labels, average=True)
    assert torch.equal(logits, logits_before)
    assert torch.equal(labels, labels_before)


# ----- sequence_logprob: dtype and invariances ------------------------------------------


@pytest.mark.parametrize(
    ("given_dtype", "expected_dtype"),
    [
        (torch.float64, torch.float64),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
def test_low_precision_logits_are_promoted(
    given_dtype: torch.dtype, expected_dtype: torch.dtype
) -> None:
    logits, labels = _build([[0.5, 0.25]])
    out = sequence_logprob(logits.to(given_dtype), labels)
    assert out.dtype == expected_dtype


def test_bfloat16_result_still_lands_near_the_float64_result() -> None:
    logits, labels = _build([[0.5, 0.25]])
    exact = float(sequence_logprob(logits, labels))
    lowered = float(sequence_logprob(logits.bfloat16(), labels))
    assert lowered == pytest.approx(exact, abs=0.05)


@settings(deadline=None, max_examples=50)
@given(shift=st.floats(min_value=-50.0, max_value=50.0))
def test_adding_a_constant_to_every_logit_changes_nothing(shift: float) -> None:
    """Softmax is shift-invariant, so the scores must be too."""
    logits, labels = _build([[0.5, 0.25, 0.125]])
    base = float(sequence_logprob(logits, labels))
    assert float(sequence_logprob(logits + shift, labels)) == pytest.approx(base, abs=1e-9)


@settings(deadline=None, max_examples=50)
@given(probs=st.lists(st.floats(min_value=1e-6, max_value=1.0 - 1e-6), min_size=1, max_size=6))
def test_log_probability_is_never_positive(probs: list[float]) -> None:
    logits, labels = _build([list(probs)])
    assert float(sequence_logprob(logits, labels)) <= 0.0
    assert float(sequence_logprob(logits, labels, average=True)) <= 0.0


def test_length_averaging_flips_the_ranking_length_bias() -> None:
    """The named length-bias test.

    Row 0 is a short response of two mediocre tokens; row 1 is a long response of six good
    ones. Summed, the short one wins purely because it has fewer negative terms to add up --
    that is the bias which teaches preference training to prefer terse answers. Averaged,
    the long response wins, which is what its per-token quality deserves.
    """
    short = [0.7, 0.7, None, None, None, None]
    long = [0.8] * 6
    logits, labels = _build([short, long])

    summed = sequence_logprob(logits, labels)
    averaged = sequence_logprob(logits, labels, average=True)

    assert float(summed[0]) > float(summed[1])
    assert float(averaged[0]) < float(averaged[1])
    assert float(summed[0]) == pytest.approx(2 * math.log(0.7), abs=1e-12)
    assert float(averaged[1]) == pytest.approx(math.log(0.8), abs=1e-12)


# ----- sequence_logprob: rejected input -------------------------------------------------


def test_rejects_two_dimensional_logits() -> None:
    with pytest.raises(ValueError, match="logits must be"):
        sequence_logprob(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long))


def test_rejects_three_dimensional_labels() -> None:
    with pytest.raises(ValueError, match="labels must be"):
        sequence_logprob(torch.zeros(2, 3, 4), torch.zeros(2, 3, 1, dtype=torch.long))


def test_rejects_mismatched_batch_or_sequence() -> None:
    with pytest.raises(ValueError, match="must agree on"):
        sequence_logprob(torch.zeros(2, 3, 4), torch.zeros(2, 5, dtype=torch.long))


def test_rejects_floating_point_labels() -> None:
    with pytest.raises(ValueError, match="integer token ids"):
        sequence_logprob(torch.zeros(1, 3, 4), torch.zeros(1, 3))


def test_rejects_boolean_labels() -> None:
    with pytest.raises(ValueError, match="integer token ids"):
        sequence_logprob(torch.zeros(1, 3, 4), torch.zeros(1, 3, dtype=torch.bool))


def test_rejects_a_single_position() -> None:
    with pytest.raises(ValueError, match="at least two positions"):
        sequence_logprob(torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.long))


def test_rejects_a_label_id_beyond_the_vocabulary() -> None:
    logits, labels = _build([[0.5, 0.5]])
    labels[0, 1] = VOCAB
    with pytest.raises(ValueError, match="must lie in"):
        sequence_logprob(logits, labels)


def test_rejects_a_negative_label_id_that_is_not_the_ignore_index() -> None:
    logits, labels = _build([[0.5, 0.5]])
    labels[0, 1] = -7
    with pytest.raises(ValueError, match="must lie in"):
        sequence_logprob(logits, labels)


# ----- dpo_loss: the ln 2 identity ------------------------------------------------------


@pytest.mark.parametrize("beta", [0.01, 0.05, 0.1, 0.5, 1.0, 10.0])
def test_policy_equal_to_reference_costs_ln_two(beta: float) -> None:
    """At initialisation pi == pi_ref, the margin is zero and sigma(0) = 1/2 for any beta."""
    chosen, rejected = _logps(-3.5, -9.25)
    out = dpo_loss(chosen, rejected, chosen, rejected, beta=beta)
    assert float(out.loss) == pytest.approx(LN2, abs=1e-12)
    assert float(out.reward_margin) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.25, 0.4])
def test_cdpo_at_a_zero_margin_is_also_ln_two(label_smoothing: float) -> None:
    """The smoothed mixture collapses to -log sigma(0) whatever the smoothing rate."""
    chosen, rejected = _logps(-2.0, -6.0)
    out = dpo_loss(
        chosen, rejected, chosen, rejected, variant="cdpo", label_smoothing=label_smoothing
    )
    assert float(out.loss) == pytest.approx(LN2, abs=1e-12)


def test_ln_two_matches_the_recorded_float32_constant() -> None:
    chosen, rejected = (torch.tensor([v], dtype=torch.float32) for v in (-1.0, -4.0))
    out = dpo_loss(chosen, rejected, chosen, rejected, beta=0.1)
    assert float(out.loss) == LN2_FLOAT32


def test_policy_equal_to_reference_scores_zero_accuracy() -> None:
    """Rewards tie at zero, and a tie is a miss."""
    chosen, rejected = _logps(-1.0, -2.0)
    out = dpo_loss(chosen, rejected, chosen, rejected)
    assert float(out.reward_accuracy) == 0.0


# ----- dpo_loss: gradients --------------------------------------------------------------


def test_gradient_pushes_the_chosen_log_probability_up() -> None:
    d_chosen, _, _, _ = _grads(-2.0, -3.0)
    assert d_chosen < 0.0


def test_gradient_pushes_the_rejected_log_probability_down() -> None:
    _, d_rejected, _, _ = _grads(-2.0, -3.0)
    assert d_rejected > 0.0


@pytest.mark.parametrize("beta", [0.05, 0.1, 1.0])
@pytest.mark.parametrize(("chosen", "rejected"), [(-2.0, -3.0), (-9.0, -1.0), (-4.0, -4.0)])
def test_gradient_signs_hold_everywhere_for_the_sigmoid_objective(
    beta: float, chosen: float, rejected: float
) -> None:
    """-sigma(-u) is negative for every finite u, so the direction never depends on the data."""
    d_chosen, d_rejected, _, _ = _grads(chosen, rejected, beta=beta)
    assert d_chosen < 0.0
    assert d_rejected > 0.0


def test_the_two_policy_gradients_are_equal_and_opposite() -> None:
    """The loss sees only the difference, so it can never move both in the same direction."""
    d_chosen, d_rejected, _, _ = _grads(-2.0, -5.0, beta=0.3)
    assert d_chosen == pytest.approx(-d_rejected, abs=1e-15)


def test_reference_gradients_mirror_the_policy_gradients() -> None:
    """pi_ref enters with the opposite sign; in a real run it is under no_grad anyway."""
    d_chosen, d_rejected, d_ref_chosen, d_ref_rejected = _grads(-2.0, -5.0, beta=0.3)
    assert d_ref_chosen == pytest.approx(-d_chosen, abs=1e-15)
    assert d_ref_rejected == pytest.approx(-d_rejected, abs=1e-15)


def test_gradient_magnitude_is_beta_times_sigmoid_of_the_negative_logit() -> None:
    beta = 0.4
    chosen, rejected = -2.0, -5.0
    d_chosen, _, _, _ = _grads(chosen, rejected, beta=beta)
    u = torch.tensor(-beta * (chosen - rejected), dtype=torch.float64)
    expected = -beta * float(torch.sigmoid(u))
    assert d_chosen == pytest.approx(expected, abs=1e-12)


# ----- dpo_loss: swapping chosen and rejected -------------------------------------------


@pytest.mark.parametrize("variant", ["sigmoid", "ipo", "cdpo"])
def test_swapping_the_pair_evaluates_the_loss_at_the_negated_logit(variant: str) -> None:
    """Swapping chosen and rejected maps h to -h, so the loss is L evaluated there."""
    beta = 0.25
    smoothing = 0.15 if variant == "cdpo" else 0.0
    chosen, rejected, ref_chosen, ref_rejected = _logps(-2.0, -6.0, -3.0, -5.0)

    original = dpo_loss(
        chosen,
        rejected,
        ref_chosen,
        ref_rejected,
        beta=beta,
        variant=variant,  # type: ignore[arg-type]
        label_smoothing=smoothing,
    )
    swapped = dpo_loss(
        rejected,
        chosen,
        ref_rejected,
        ref_chosen,
        beta=beta,
        variant=variant,  # type: ignore[arg-type]
        label_smoothing=smoothing,
    )

    h = float(original.reward_margins) / beta
    assert float(swapped.reward_margins) == pytest.approx(
        -float(original.reward_margins), abs=1e-12
    )
    assert float(original.loss) == pytest.approx(
        _loss_at_logit(h, beta=beta, variant=variant, label_smoothing=smoothing), abs=1e-12
    )
    assert float(swapped.loss) == pytest.approx(
        _loss_at_logit(-h, beta=beta, variant=variant, label_smoothing=smoothing), abs=1e-12
    )


def test_swapping_adds_the_reward_margin_to_the_sigmoid_loss() -> None:
    """log sigma(u) - log sigma(-u) = u exactly, so L(-u) = L(u) + u = L(u) + margin."""
    beta = 0.2
    chosen, rejected, ref_chosen, ref_rejected = _logps(-2.0, -6.0, -3.0, -5.0)
    original = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=beta)
    swapped = dpo_loss(rejected, chosen, ref_rejected, ref_chosen, beta=beta)
    assert float(swapped.loss) == pytest.approx(
        float(original.loss) + float(original.reward_margin), abs=1e-12
    )


def test_swapping_flips_the_reward_accuracy() -> None:
    chosen = torch.tensor([-1.0, -8.0, -2.0], dtype=torch.float64)
    rejected = torch.tensor([-5.0, -1.0, -9.0], dtype=torch.float64)
    ref = torch.zeros(3, dtype=torch.float64)
    forward = dpo_loss(chosen, rejected, ref, ref)
    backward = dpo_loss(rejected, chosen, ref, ref)
    assert float(forward.reward_accuracy) == pytest.approx(2.0 / 3.0)
    assert float(backward.reward_accuracy) == pytest.approx(1.0 - float(forward.reward_accuracy))


# ----- dpo_loss: numerical robustness ---------------------------------------------------


@pytest.mark.parametrize("variant", ["sigmoid", "ipo", "cdpo"])
@pytest.mark.parametrize("h", [-200.0, 200.0])
@pytest.mark.parametrize("beta", [0.1, 1.0])
def test_loss_is_finite_at_extreme_log_ratio_differences(
    variant: str, h: float, beta: float
) -> None:
    smoothing = 0.1 if variant == "cdpo" else 0.0
    chosen, rejected, ref = _logps(h, 0.0, 0.0)
    out = dpo_loss(
        chosen,
        rejected,
        ref,
        ref,
        beta=beta,
        variant=variant,  # type: ignore[arg-type]
        label_smoothing=smoothing,
    )
    assert torch.isfinite(out.loss).all()
    assert torch.isfinite(out.losses).all()
    assert torch.isfinite(out.reward_margins).all()


def test_extreme_negative_logit_stays_finite_where_naive_log_sigmoid_does_not() -> None:
    """The reason `logsigmoid` is not optional: sigmoid(-200) is exactly zero in float32."""
    u = torch.tensor(-200.0, dtype=torch.float32)
    assert not torch.isfinite(torch.log(torch.sigmoid(u)))

    chosen, rejected, ref = (torch.tensor([v], dtype=torch.float32) for v in (-200.0, 0.0, 0.0))
    out = dpo_loss(chosen, rejected, ref, ref, beta=1.0)
    assert torch.isfinite(out.loss).all()
    assert float(out.loss) == pytest.approx(200.0, abs=1e-3)


def test_extreme_negative_logit_produces_a_usable_gradient() -> None:
    d_chosen, d_rejected, _, _ = _grads(-200.0, 0.0, beta=1.0)
    assert math.isfinite(d_chosen)
    assert d_chosen == pytest.approx(-1.0, abs=1e-9)
    assert d_rejected == pytest.approx(1.0, abs=1e-9)


# ----- dpo_loss: cDPO -------------------------------------------------------------------


@pytest.mark.parametrize("h", [-30.0, -1.0, 0.0, 2.5, 30.0])
def test_cdpo_with_zero_smoothing_is_exactly_the_sigmoid_objective(h: float) -> None:
    chosen, rejected, ref = _logps(h, 0.0, 0.0)
    plain = dpo_loss(chosen, rejected, ref, ref, beta=0.3)
    smoothed = dpo_loss(chosen, rejected, ref, ref, beta=0.3, variant="cdpo", label_smoothing=0.0)
    assert float(smoothed.loss) == float(plain.loss)


@pytest.mark.parametrize("label_smoothing", [0.05, 0.1, 0.3, 0.45])
def test_cdpo_is_minimised_at_the_smoothing_logit(label_smoothing: float) -> None:
    """dL/du = sigma(u) - (1 - eps), so the optimum sits at u* = log((1 - eps) / eps)."""
    beta = 0.5
    h_star = math.log((1.0 - label_smoothing) / label_smoothing) / beta
    at_star = _loss_at_logit(h_star, beta=beta, variant="cdpo", label_smoothing=label_smoothing)
    for delta in (-1.0, -0.1, 0.1, 1.0):
        nearby = dpo_loss(
            *_logps(h_star + delta, 0.0, 0.0, 0.0),
            beta=beta,
            variant="cdpo",
            label_smoothing=label_smoothing,
        )
        assert float(nearby.loss) > at_star


@pytest.mark.parametrize("label_smoothing", [0.05, 0.1, 0.3, 0.45])
def test_cdpo_minimum_value_is_the_binary_entropy_of_the_smoothing_rate(
    label_smoothing: float,
) -> None:
    beta = 0.5
    eps = label_smoothing
    h_star = math.log((1.0 - eps) / eps) / beta
    entropy = -(1.0 - eps) * math.log(1.0 - eps) - eps * math.log(eps)
    out = dpo_loss(
        *_logps(h_star, 0.0, 0.0, 0.0),
        beta=beta,
        variant="cdpo",
        label_smoothing=eps,
    )
    assert float(out.loss) == pytest.approx(entropy, abs=1e-12)


def test_cdpo_gradient_reverses_beyond_the_target_margin() -> None:
    """Conservative DPO pulls an over-separated pair back; the plain objective never does."""
    beta, eps = 0.5, 0.2
    h_star = math.log((1.0 - eps) / eps) / beta
    below, _, _, _ = _grads(h_star - 1.0, 0.0, beta=beta, variant="cdpo", label_smoothing=eps)
    above, _, _, _ = _grads(h_star + 1.0, 0.0, beta=beta, variant="cdpo", label_smoothing=eps)
    assert below < 0.0
    assert above > 0.0

    plain_above, _, _, _ = _grads(h_star + 1.0, 0.0, beta=beta)
    assert plain_above < 0.0


# ----- dpo_loss: IPO --------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.05, 0.1, 1.0])
def test_ipo_is_minimised_at_a_finite_target_margin(beta: float) -> None:
    target = 1.0 / (2.0 * beta)
    at_target = dpo_loss(*_logps(target, 0.0, 0.0, 0.0), beta=beta, variant="ipo")
    assert float(at_target.loss) == pytest.approx(0.0, abs=1e-18)
    for delta in (-2.0, -0.5, 0.5, 2.0):
        off = dpo_loss(*_logps(target + delta, 0.0, 0.0, 0.0), beta=beta, variant="ipo")
        assert float(off.loss) == pytest.approx(delta**2, abs=1e-12)


@pytest.mark.parametrize("h", [-5.0, 0.0, 3.0, 40.0])
def test_ipo_matches_its_closed_form(h: float) -> None:
    beta = 0.2
    out = dpo_loss(*_logps(h, 0.0, 0.0, 0.0), beta=beta, variant="ipo")
    assert float(out.loss) == pytest.approx((h - 1.0 / (2.0 * beta)) ** 2, abs=1e-9)


def test_ipo_does_not_saturate_where_the_sigmoid_objective_does() -> None:
    """The point of IPO: an already-separated pair keeps producing gradient."""
    beta = 0.5
    near, far = 5.0, 50.0
    ipo_near, _, _, _ = _grads(near, 0.0, beta=beta, variant="ipo")
    ipo_far, _, _, _ = _grads(far, 0.0, beta=beta, variant="ipo")
    sig_near, _, _, _ = _grads(near, 0.0, beta=beta)
    sig_far, _, _, _ = _grads(far, 0.0, beta=beta)

    assert abs(ipo_far) > abs(ipo_near)
    assert abs(sig_far) < abs(sig_near)
    assert abs(sig_far) < 1e-9


# ----- dpo_loss: rewards and reporting --------------------------------------------------


def test_chosen_reward_is_beta_times_the_chosen_log_ratio() -> None:
    beta = 0.3
    out = dpo_loss(*_logps(-2.0, -5.0, -2.5, -4.0), beta=beta)
    assert float(out.chosen_rewards) == pytest.approx(beta * (-2.0 - -2.5), abs=1e-12)
    assert float(out.rejected_rewards) == pytest.approx(beta * (-5.0 - -4.0), abs=1e-12)


def test_reward_margin_is_the_difference_of_the_two_rewards() -> None:
    out = dpo_loss(*_logps(-2.0, -5.0, -2.5, -4.0), beta=0.3)
    assert float(out.reward_margins) == pytest.approx(
        float(out.chosen_rewards) - float(out.rejected_rewards), abs=1e-15
    )


def test_reward_margin_equals_beta_times_the_log_ratio_difference() -> None:
    beta = 0.7
    chosen, rejected, ref_chosen, ref_rejected = -2.0, -5.0, -2.5, -4.0
    out = dpo_loss(*_logps(chosen, rejected, ref_chosen, ref_rejected), beta=beta)
    h = (chosen - rejected) - (ref_chosen - ref_rejected)
    assert float(out.reward_margins) == pytest.approx(beta * h, abs=1e-12)


def test_rewards_are_detached_from_the_graph() -> None:
    """A logged tensor that still carries a graph pins every activation behind it."""
    tensors = [
        torch.tensor([v], dtype=torch.float64, requires_grad=True) for v in (-2.0, -5.0, -2.5, -4.0)
    ]
    out = dpo_loss(*tensors)
    assert out.loss.requires_grad
    assert not out.chosen_rewards.requires_grad
    assert not out.rejected_rewards.requires_grad
    assert not out.reward_margins.requires_grad
    assert not out.reward_accuracy.requires_grad


def test_reward_accuracy_counts_correctly_ranked_pairs() -> None:
    chosen = torch.tensor([-1.0, -8.0, -2.0, -0.5], dtype=torch.float64)
    rejected = torch.tensor([-5.0, -1.0, -9.0, -4.0], dtype=torch.float64)
    ref = torch.zeros(4, dtype=torch.float64)
    out = dpo_loss(chosen, rejected, ref, ref)
    assert float(out.reward_accuracy) == pytest.approx(0.75)


def test_reward_accuracy_treats_a_tie_as_a_miss() -> None:
    same = torch.tensor([-3.0, -3.0], dtype=torch.float64)
    ref = torch.zeros(2, dtype=torch.float64)
    out = dpo_loss(same, same.clone(), ref, ref)
    assert float(out.reward_accuracy) == 0.0


def test_loss_is_the_mean_of_the_per_pair_losses() -> None:
    chosen = torch.tensor([-1.0, -8.0, -2.0], dtype=torch.float64)
    rejected = torch.tensor([-5.0, -1.0, -9.0], dtype=torch.float64)
    ref = torch.zeros(3, dtype=torch.float64)
    out = dpo_loss(chosen, rejected, ref, ref)
    assert out.losses.shape == (3,)
    assert float(out.loss) == pytest.approx(float(out.losses.mean()), abs=1e-15)


def test_reward_margin_property_is_the_batch_mean() -> None:
    chosen = torch.tensor([-1.0, -8.0], dtype=torch.float64)
    rejected = torch.tensor([-5.0, -1.0], dtype=torch.float64)
    ref = torch.zeros(2, dtype=torch.float64)
    out = dpo_loss(chosen, rejected, ref, ref)
    assert float(out.reward_margin) == pytest.approx(float(out.reward_margins.mean()), abs=1e-15)


def test_metrics_are_plain_floats_under_the_expected_names() -> None:
    out = dpo_loss(*_logps(-2.0, -5.0, -2.5, -4.0))
    metrics = out.metrics()
    assert set(metrics) == {
        "loss",
        "reward_chosen",
        "reward_rejected",
        "reward_margin",
        "reward_accuracy",
    }
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics["reward_margin"] == pytest.approx(
        metrics["reward_chosen"] - metrics["reward_rejected"], abs=1e-12
    )


def test_zero_dimensional_inputs_are_accepted() -> None:
    scalars = tuple(torch.tensor(v, dtype=torch.float64) for v in (-2.0, -5.0, 0.0, 0.0))
    out = dpo_loss(*scalars)
    assert out.losses.ndim == 0
    assert isinstance(out, DPOOutput)


# ----- dpo_loss: rejected input ---------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, -0.1])
def test_rejects_a_non_positive_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="beta must be positive"):
        dpo_loss(*_logps(-1.0, -2.0, 0.0, 0.0), beta=beta)


def test_rejects_an_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        dpo_loss(*_logps(-1.0, -2.0, 0.0, 0.0), variant="hinge")  # type: ignore[arg-type]


@pytest.mark.parametrize("label_smoothing", [-0.01, 0.5, 0.9, 1.0])
def test_rejects_label_smoothing_outside_the_half_open_unit_half(label_smoothing: float) -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 0.5\)"):
        dpo_loss(
            *_logps(-1.0, -2.0, 0.0, 0.0),
            variant="cdpo",
            label_smoothing=label_smoothing,
        )


@pytest.mark.parametrize("variant", ["sigmoid", "ipo"])
def test_rejects_label_smoothing_for_a_variant_without_a_smoothing_term(variant: str) -> None:
    with pytest.raises(ValueError, match="only defined for variant='cdpo'"):
        dpo_loss(
            *_logps(-1.0, -2.0, 0.0, 0.0),
            variant=variant,  # type: ignore[arg-type]
            label_smoothing=0.1,
        )


def test_rejects_mismatched_input_shapes() -> None:
    with pytest.raises(ValueError, match="ref_rejected_logp has shape"):
        dpo_loss(
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(2),
        )


def test_rejects_integer_log_probabilities() -> None:
    with pytest.raises(ValueError, match="must be floating point"):
        dpo_loss(
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
        )


# ----- dpo_loss: properties -------------------------------------------------------------


@settings(deadline=None, max_examples=100)
@given(
    first=st.floats(min_value=-40.0, max_value=40.0),
    second=st.floats(min_value=-40.0, max_value=40.0),
    beta=st.floats(min_value=0.01, max_value=1.0),
)
def test_sigmoid_loss_is_monotonically_decreasing_in_the_reward_margin(
    first: float, second: float, beta: float
) -> None:
    """The whole point of the objective: a larger margin is never a worse loss."""
    low, high = sorted((first, second))
    assume(beta * (high - low) > 1e-6)
    ref = torch.zeros(1, dtype=torch.float64)
    losses = [
        float(dpo_loss(torch.tensor([h], dtype=torch.float64), ref, ref, ref, beta=beta).loss)
        for h in (low, high)
    ]
    assert losses[0] > losses[1]


@settings(deadline=None, max_examples=100)
@given(
    h=st.floats(min_value=-30.0, max_value=30.0),
    beta=st.floats(min_value=0.01, max_value=1.0),
)
def test_sigmoid_loss_is_strictly_positive(h: float, beta: float) -> None:
    ref = torch.zeros(1, dtype=torch.float64)
    out = dpo_loss(torch.tensor([h], dtype=torch.float64), ref, ref, ref, beta=beta)
    assert float(out.loss) > 0.0


@settings(deadline=None, max_examples=100)
@given(
    h=st.floats(min_value=-50.0, max_value=50.0),
    beta=st.floats(min_value=0.01, max_value=1.0),
)
def test_ipo_loss_is_non_negative(h: float, beta: float) -> None:
    ref = torch.zeros(1, dtype=torch.float64)
    out = dpo_loss(torch.tensor([h], dtype=torch.float64), ref, ref, ref, beta=beta, variant="ipo")
    assert float(out.loss) >= 0.0


@settings(deadline=None, max_examples=100)
@given(
    h=st.floats(min_value=-20.0, max_value=20.0),
    beta=st.floats(min_value=0.05, max_value=2.0),
    scale=st.floats(min_value=1.5, max_value=5.0),
)
def test_sigmoid_loss_depends_only_on_the_product_of_beta_and_h(
    h: float, beta: float, scale: float
) -> None:
    """Halving beta and doubling the log-ratio difference is the same objective value."""
    ref = torch.zeros(1, dtype=torch.float64)
    wide = dpo_loss(
        torch.tensor([h * scale], dtype=torch.float64), ref, ref, ref, beta=beta / scale
    )
    narrow = dpo_loss(torch.tensor([h], dtype=torch.float64), ref, ref, ref, beta=beta)
    assert float(wide.loss) == pytest.approx(float(narrow.loss), rel=1e-9, abs=1e-15)


@settings(deadline=None, max_examples=100)
@given(
    h=st.floats(min_value=-40.0, max_value=40.0),
    eps=st.floats(min_value=0.0, max_value=0.49),
)
def test_cdpo_matches_its_closed_form_everywhere(h: float, eps: float) -> None:
    beta = 0.4
    ref = torch.zeros(1, dtype=torch.float64)
    out = dpo_loss(
        torch.tensor([h], dtype=torch.float64),
        ref,
        ref,
        ref,
        beta=beta,
        variant="cdpo",
        label_smoothing=eps,
    )
    expected = -(1.0 - eps) * float(F.logsigmoid(torch.tensor(beta * h, dtype=torch.float64)))
    expected += -eps * float(F.logsigmoid(torch.tensor(-beta * h, dtype=torch.float64)))
    assert float(out.loss) == pytest.approx(expected, abs=1e-12)


# ----- the two functions together -------------------------------------------------------


def test_sequence_logprob_feeds_dpo_loss_to_the_ln_two_identity() -> None:
    """The integration a trainer actually performs, held to the same identity."""
    chosen_logits, chosen_labels = _build([[0.6, 0.4, 0.9]])
    rejected_logits, rejected_labels = _build([[0.2, 0.3, None]])

    policy_chosen = sequence_logprob(chosen_logits, chosen_labels)
    policy_rejected = sequence_logprob(rejected_logits, rejected_labels)

    out = dpo_loss(policy_chosen, policy_rejected, policy_chosen, policy_rejected, beta=0.1)
    assert float(out.loss) == pytest.approx(LN2, abs=1e-12)
    assert float(out.reward_margin) == pytest.approx(0.0, abs=1e-15)
    assert float(policy_chosen) > float(policy_rejected)


def test_a_policy_that_improved_on_the_reference_beats_ln_two() -> None:
    """A policy that raised the chosen sequence and lowered the rejected one must score better."""
    reference_logits, chosen_labels = _build([[0.4, 0.4, 0.4]])
    improved_logits, _ = _build([[0.8, 0.8, 0.8]])
    worsened_logits, rejected_labels = _build([[0.1, 0.1, 0.1]])

    ref_chosen = sequence_logprob(reference_logits, chosen_labels)
    ref_rejected = sequence_logprob(reference_logits, rejected_labels)
    policy_chosen = sequence_logprob(improved_logits, chosen_labels)
    policy_rejected = sequence_logprob(worsened_logits, rejected_labels)

    out = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert float(out.loss) < LN2
    assert float(out.reward_margin) > 0.0
    assert float(out.reward_accuracy) == 1.0


# --------------------------------------------------------------------------------------
# Chunking the vocabulary softmax
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunking_the_softmax_is_bit_identical(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    """Splitting the position axis must not change a single bit of the answer.

    Scoring label tokens is a softmax over a 151 936-entry vocabulary, and doing every
    position at once needs 6.95 GiB above its inputs for one DPO pair at 2 048 tokens --
    enough to kill a real run on a 12 GiB card, which is exactly what happened. Chunking is
    sound because each position's softmax reads only that position's row, so this test holds
    the inputs fixed and drives the chunk size down to one row at a time.

    Equality here is exact, not approximate. If a refactor ever makes chunking approximate,
    `assert torch.equal` is the line that says so; a tolerance would hide it.
    """
    torch.manual_seed(0)
    logits = torch.randn(3, 9, 64, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 64, (3, 9))
    labels[:, :3] = IGNORE_INDEX

    unchunked = sequence_logprob(logits, labels)
    unchunked.sum().backward()
    assert logits.grad is not None
    reference_grad = logits.grad.clone()

    logits.grad = None
    # One row of the vocabulary per chunk: the most fragmented split possible.
    monkeypatch.setattr("sftdpo.train.dpo_loss.LOGPROB_CHUNK_ELEMENTS", 64)
    chunked = sequence_logprob(logits, labels)
    chunked.sum().backward()

    assert torch.equal(chunked, unchunked)
    assert logits.grad is not None
    assert torch.equal(logits.grad, reference_grad)


def test_chunking_survives_a_ragged_final_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last chunk is shorter than the rest whenever the split does not divide evenly."""
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8))
    expected = sequence_logprob(logits, labels)

    # 14 rows to score, 96 // 32 = 3 rows per chunk: four full chunks and a remainder of two.
    monkeypatch.setattr("sftdpo.train.dpo_loss.LOGPROB_CHUNK_ELEMENTS", 96)
    assert torch.equal(sequence_logprob(logits, labels), expected)


def test_a_vocabulary_larger_than_the_budget_still_scores_one_row_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chunk size floors at one row, so an enormous vocabulary cannot divide to zero."""
    torch.manual_seed(2)
    logits = torch.randn(2, 5, 40)
    labels = torch.randint(0, 40, (2, 5))
    expected = sequence_logprob(logits, labels)

    monkeypatch.setattr("sftdpo.train.dpo_loss.LOGPROB_CHUNK_ELEMENTS", 1)
    assert torch.equal(sequence_logprob(logits, labels), expected)
