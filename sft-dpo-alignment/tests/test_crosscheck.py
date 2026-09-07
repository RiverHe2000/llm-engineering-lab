"""Tests for the cross-check against the official `trl` implementation.

The module under test exists to answer one question honestly: does the hand-written DPO
objective compute the same numbers as the reference implementation? These tests hold it to
that, and hold it just as firmly to the two ways it could stop being an answer -- by quietly
reporting "unavailable" as though it were agreement, or by quietly failing to drive TRL's loss
at all.

Three groups carry the weight.

*The comparison itself* runs TRL's own `_compute_loss` on logits that realise the batch's
log-probabilities exactly, and asserts agreement to float64 noise. That is the point of the
whole module, and the tests fail loudly if the two ever part company.

*The two documented differences of convention* are pinned with numbers rather than prose.
TRL's IPO divides by the completion length and mine does not, so the two disagree by whole
units at three tokens per completion and agree exactly once the normalisation is applied;
TRL 1.12.0's sigmoid loss ignores `label_smoothing` entirely, which is asserted directly
against TRL and is the reason cDPO is compared against `aot` instead.

*The failure modes* are exercised with a stubbed TRL, so they run whether or not the optional
extra is installed: a missing `_compute_loss`, a `_compute_loss` whose interface has moved, and
a genuine numerical disagreement all have to surface rather than pass.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from sftdpo.train import crosscheck as crosscheck_module
from sftdpo.train.crosscheck import (
    DEFAULT_TOLERANCE,
    KNOWN_CONVENTIONS,
    TRL_LOSS_TYPES,
    CrossCheckBatch,
    CrossCheckResult,
    TrlInterfaceError,
    _logit_for,
    _run_trl,
    _synthesise,
    crosscheck_against_trl,
    crosscheck_all,
    trl_available,
    trl_version,
)
from sftdpo.train.dpo_loss import IGNORE_INDEX, DPOVariant, dpo_loss, sequence_logprob

requires_trl = pytest.mark.skipif(
    not trl_available(),
    reason="the optional crosscheck extra (trl) is not installed",
)

PROPERTY_SETTINGS = settings(max_examples=15, deadline=None)

VOCAB = 8


def four_logprobs(*values: float) -> list[Tensor]:
    """Four one-element log-probability tensors, for a single hand-built pair."""
    return [torch.tensor([value], dtype=torch.float64) for value in values]


def a_result(**overrides: Any) -> CrossCheckResult:
    """A `CrossCheckResult` with plausible fields, for testing the object rather than the check."""
    fields: dict[str, Any] = {
        "status": "agree",
        "variant": "sigmoid",
        "trl_loss_type": "sigmoid",
        "beta": 0.1,
        "label_smoothing": 0.0,
        "pairs": 4,
        "tolerance": DEFAULT_TOLERANCE,
        "loss_difference": 1e-16,
        "chosen_reward_difference": 2e-16,
        "rejected_reward_difference": 3e-16,
        "normalised_loss_difference": None,
        "note": "",
        "reason": "",
        "trl_version": "1.12.0",
    }
    fields.update(overrides)
    return CrossCheckResult(**fields)


# --------------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------------


def test_availability_is_decided_by_the_import_machinery() -> None:
    """Nothing is executed to answer the question, so the answer cannot depend on import order."""
    assert trl_available() == (importlib.util.find_spec("trl") is not None)


def test_the_version_is_recorded_or_named_as_absent() -> None:
    """A cross-check is a statement about two implementations, and the other one has a version."""
    version = trl_version()
    assert version == "not installed" or version[0].isdigit()
    assert trl_available() == (version != "not installed")


def test_an_unreadable_version_is_named_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing distribution must not take down the report that quotes the comparison."""

    def absent(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("trl")

    monkeypatch.setattr(importlib.metadata, "version", absent)
    assert trl_version() == "not installed"


def test_an_absent_trl_is_reported_rather_than_papered_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crosscheck_module, "trl_available", lambda: False)
    result = crosscheck_against_trl(CrossCheckBatch.synthetic(2))
    assert result.status == "unavailable"
    assert result.available is False
    assert result.agrees is False
    assert result.pairs == 0
    assert math.isnan(result.loss_difference)
    assert math.isnan(result.max_difference)
    assert "crosscheck" in result.reason
    assert str(result).startswith("cross-check unavailable")


def test_an_unavailable_result_is_never_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure this module exists to prevent: a disabled check reported as a passing one."""
    monkeypatch.setattr(crosscheck_module, "trl_available", lambda: False)
    for variant in ("sigmoid", "ipo", "cdpo"):
        result = crosscheck_against_trl(CrossCheckBatch.synthetic(2), variant=variant)
        assert result.agrees is False
        assert result.trl_loss_type == TRL_LOSS_TYPES[variant]
        assert result.trl_version == "not installed"


# --------------------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------------------


def test_the_synthetic_batch_spans_both_signs_of_the_log_ratio() -> None:
    """A batch the policy already ranks correctly only exercises the saturating tail."""
    batch = CrossCheckBatch.synthetic(16, seed=0)
    h = (batch.policy_chosen - batch.policy_rejected) - (batch.ref_chosen - batch.ref_rejected)
    assert bool((h > 0).any())
    assert bool((h < 0).any())


def test_the_synthetic_batch_is_reproducible_from_its_seed() -> None:
    assert torch.equal(
        CrossCheckBatch.synthetic(4, seed=3).policy_chosen,
        CrossCheckBatch.synthetic(4, seed=3).policy_chosen,
    )
    assert not torch.equal(
        CrossCheckBatch.synthetic(4, seed=3).policy_chosen,
        CrossCheckBatch.synthetic(4, seed=4).policy_chosen,
    )


def test_the_synthetic_batch_leaves_the_global_generator_alone() -> None:
    """The generator is local, so building a batch cannot perturb a seeded test around it."""
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    CrossCheckBatch.synthetic(8, seed=11)
    assert torch.equal(torch.rand(3), expected)


def test_the_draw_is_in_float64_and_strictly_negative() -> None:
    batch = CrossCheckBatch.synthetic(8, seed=5, spread=6.0)
    assert batch.policy_chosen.dtype == torch.float64
    assert bool((batch.policy_chosen < 0.0).all())
    assert batch.size == 8


def test_the_width_covers_the_longest_completion_plus_a_prompt() -> None:
    batch = CrossCheckBatch.synthetic(3, tokens=4)
    assert batch.width == 5


@pytest.mark.parametrize(("kwargs", "message"), [({"size": 0}, "size"), ({"tokens": 0}, "tokens")])
def test_the_synthetic_batch_validates_its_shape(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CrossCheckBatch.synthetic(**kwargs)


def test_a_two_dimensional_tensor_is_refused() -> None:
    values = torch.full((2, 2), -1.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be one-dimensional"):
        CrossCheckBatch(values, values, values, values, (1, 1), (1, 1))


def test_misaligned_tensors_are_refused() -> None:
    two, three = four_logprobs(-1.0)[0].repeat(2), four_logprobs(-1.0)[0].repeat(3)
    with pytest.raises(ValueError, match="aligned per pair"):
        CrossCheckBatch(two, three, two, two, (1, 1), (1, 1))


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_a_non_negative_log_probability_is_refused(value: float) -> None:
    """Zero and above cannot be realised as logits, so the check would test the reconstruction."""
    bad, good = four_logprobs(value, -1.0)
    with pytest.raises(ValueError, match="must hold log-probabilities"):
        CrossCheckBatch(bad, good, good, good, (1,), (1,))


def test_token_counts_must_line_up_with_the_pairs() -> None:
    one = four_logprobs(-1.0)[0].repeat(2)
    with pytest.raises(ValueError, match="chosen_tokens has 1 entries"):
        CrossCheckBatch(one, one, one, one, (1,), (1, 1))


def test_token_counts_must_be_positive() -> None:
    one = four_logprobs(-1.0)[0].repeat(2)
    with pytest.raises(ValueError, match="must be positive"):
        CrossCheckBatch(one, one, one, one, (1, 0), (1, 1))


# --------------------------------------------------------------------------------------
# Realising log-probabilities as logits
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", [-0.01, -0.5, -2.0, -9.0, -30.0])
@pytest.mark.parametrize("vocab", [2, 8, 64])
def test_the_closed_form_logit_realises_its_target_exactly(target: float, vocab: int) -> None:
    """Solved analytically rather than optimised, so no residual can leak into the comparison."""
    logits = torch.zeros(vocab, dtype=torch.float64)
    logits[0] = _logit_for(target, vocab)
    assert float(torch.log_softmax(logits, dim=-1)[0]) == pytest.approx(target, abs=1e-12)


@pytest.mark.parametrize("tokens", [1, 2, 5])
def test_the_synthesised_logits_reduce_to_the_batch_they_came_from(tokens: int) -> None:
    """The precondition of the whole comparison: both sides really do read the same numbers."""
    batch = CrossCheckBatch.synthetic(6, seed=7, tokens=tokens)
    tensors = _synthesise(batch, vocab=VOCAB)
    chosen, rejected = sequence_logprob(tensors["logits"], tensors["labels"]).chunk(2, dim=0)
    assert torch.allclose(chosen, batch.policy_chosen, atol=1e-12)
    assert torch.allclose(rejected, batch.policy_rejected, atol=1e-12)


def test_every_position_of_a_completion_carries_a_different_score() -> None:
    """Uniform shares would let a mis-shift cancel; distinct ones make it change the total."""
    batch = CrossCheckBatch.synthetic(1, seed=2, tokens=4)
    tensors = _synthesise(batch, vocab=VOCAB)
    scored = tensors["logits"][0][tensors["logits"][0].abs().sum(dim=-1) > 0]
    values = sorted(float(row.max()) for row in scored)
    assert len(values) == 4
    assert len(set(values)) == 4


def test_only_the_completion_is_labelled() -> None:
    batch = CrossCheckBatch.synthetic(2, seed=1, tokens=3)
    tensors = _synthesise(batch, vocab=VOCAB)
    labels, mask = tensors["labels"], tensors["completion_mask"]
    assert bool((labels[mask == 0] == IGNORE_INDEX).all())
    assert bool((labels[mask == 1] != IGNORE_INDEX).all())
    assert int(mask.sum()) == 2 * 3 * 2


# --------------------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------------------


@requires_trl
def test_the_sigmoid_objective_agrees_with_trl() -> None:
    """The headline result: two independent implementations of one formula, to float64 noise."""
    result = crosscheck_against_trl(CrossCheckBatch.synthetic(8, seed=0), beta=0.1)
    assert result.status == "agree"
    assert result.agrees is True
    assert result.available is True
    assert result.loss_difference <= DEFAULT_TOLERANCE
    assert result.max_difference <= DEFAULT_TOLERANCE
    assert result.pairs == 8
    assert result.trl_loss_type == "sigmoid"
    assert result.trl_version == trl_version()


@requires_trl
def test_the_implicit_rewards_agree_as_well_as_the_loss() -> None:
    """The rewards are what a DPO run is read by, so a check on the loss alone is half a check."""
    result = crosscheck_against_trl(CrossCheckBatch.synthetic(8, seed=4, tokens=3), beta=0.25)
    assert result.chosen_reward_difference <= DEFAULT_TOLERANCE
    assert result.rejected_reward_difference <= DEFAULT_TOLERANCE
    assert result.reward_difference == max(
        result.chosen_reward_difference, result.rejected_reward_difference
    )


@requires_trl
@pytest.mark.parametrize("beta", [0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
def test_agreement_holds_across_beta(beta: float) -> None:
    """Beta scales the implicit reward, so a factor dropped on one side would show up here."""
    result = crosscheck_against_trl(CrossCheckBatch.synthetic(6, seed=2), beta=beta)
    assert result.status == "agree"


@requires_trl
@pytest.mark.parametrize("size", [1, 3, 12])
def test_agreement_does_not_depend_on_the_batch_size(size: int) -> None:
    result = crosscheck_against_trl(CrossCheckBatch.synthetic(size, seed=6))
    assert result.status == "agree"
    assert result.pairs == size


@requires_trl
@given(beta=st.floats(min_value=0.02, max_value=2.0))
@PROPERTY_SETTINGS
def test_agreement_is_a_property_of_the_formula_not_of_a_lucky_beta(beta: float) -> None:
    result = crosscheck_against_trl(CrossCheckBatch.synthetic(4, seed=8), beta=beta)
    assert result.max_difference <= DEFAULT_TOLERANCE


@requires_trl
@pytest.mark.parametrize("label_smoothing", [0.05, 0.2, 0.45])
def test_cdpo_agrees_with_trls_aot_loss(label_smoothing: float) -> None:
    """cDPO's expression survives in TRL's `aot`, whose batch sort is the identity on one pair."""
    result = crosscheck_against_trl(
        CrossCheckBatch.synthetic(5, seed=9),
        beta=0.1,
        variant="cdpo",
        label_smoothing=label_smoothing,
    )
    assert result.trl_loss_type == "aot"
    assert result.status == "agree"
    assert result.label_smoothing == label_smoothing


@requires_trl
def test_ipo_agrees_when_a_completion_is_a_single_token() -> None:
    """At one token the length normalisation is a division by one, so the two forms coincide."""
    result = crosscheck_against_trl(
        CrossCheckBatch.synthetic(6, seed=1, tokens=1), beta=0.1, variant="ipo"
    )
    assert result.status == "agree"
    assert result.normalised_loss_difference is not None
    assert result.normalised_loss_difference <= DEFAULT_TOLERANCE


@requires_trl
def test_crosscheck_all_reports_every_variant_in_order() -> None:
    results = crosscheck_all(CrossCheckBatch.synthetic(4, seed=5), beta=0.1, label_smoothing=0.2)
    assert [result.variant for result in results] == ["sigmoid", "ipo", "cdpo"]
    assert [result.label_smoothing for result in results] == [0.0, 0.0, 0.2]
    assert all(result.status == "agree" for result in results)


@requires_trl
def test_crosscheck_all_builds_its_own_batch_when_given_none() -> None:
    results = crosscheck_all()
    assert len(results) == 3
    assert all(result.pairs == 8 for result in results)


# --------------------------------------------------------------------------------------
# The two documented differences of convention
# --------------------------------------------------------------------------------------


@requires_trl
def test_trl_normalises_ipo_by_the_completion_length_and_this_package_does_not() -> None:
    """Finding one, measured rather than asserted.

    At three tokens per completion the two IPO losses differ by whole units -- TRL divides each
    side's log-ratio by that side's token count before squaring, a choice its own source
    attributes to correspondence with the IPO authors. Recomputing my loss on per-token averages
    removes the difference entirely, which is what makes this a convention rather than a bug.
    """
    result = crosscheck_against_trl(
        CrossCheckBatch.synthetic(6, seed=1, tokens=3), beta=0.1, variant="ipo"
    )
    assert result.status == "explained"
    assert result.agrees is False
    assert result.loss_difference > 1.0
    assert result.normalised_loss_difference is not None
    assert result.normalised_loss_difference <= DEFAULT_TOLERANCE
    assert result.note == KNOWN_CONVENTIONS["ipo"]
    assert result.chosen_reward_difference <= DEFAULT_TOLERANCE


@requires_trl
def test_trls_sigmoid_loss_ignores_label_smoothing_entirely() -> None:
    """Finding two, asserted directly against TRL rather than taken on trust.

    This is why cDPO is compared against `aot` and not against `sigmoid`: in TRL 1.12.0 the
    argument no longer reaches the sigmoid objective at all, so a comparison there would be
    checking my smoothed loss against an unsmoothed one and calling the gap a defect.
    """
    batch = CrossCheckBatch.synthetic(3, seed=2)
    tensors = _synthesise(batch, vocab=VOCAB)
    arguments: dict[str, Any] = {
        "row": 0,
        "pairs": batch.size,
        "ref_chosen": batch.ref_chosen,
        "ref_rejected": batch.ref_rejected,
        "beta": 0.1,
    }
    plain = _run_trl(tensors, loss_type="sigmoid", label_smoothing=0.0, **arguments)
    smoothed = _run_trl(tensors, loss_type="sigmoid", label_smoothing=0.3, **arguments)
    assert plain == smoothed

    # ... whereas the smoothing genuinely moves this package's cDPO, which is the whole point.
    logps = sequence_logprob(tensors["logits"], tensors["labels"])
    policy_chosen, policy_rejected = logps.chunk(2, dim=0)
    unsmoothed = dpo_loss(
        policy_chosen, policy_rejected, batch.ref_chosen, batch.ref_rejected, beta=0.1
    )
    conservative = dpo_loss(
        policy_chosen,
        policy_rejected,
        batch.ref_chosen,
        batch.ref_rejected,
        beta=0.1,
        variant="cdpo",
        label_smoothing=0.3,
    )
    assert float(conservative.loss) != float(unsmoothed.loss)


def test_the_known_conventions_are_documented_where_the_report_can_quote_them() -> None:
    assert set(KNOWN_CONVENTIONS) == {"ipo", "cdpo"}
    assert "length_normalise=True" in KNOWN_CONVENTIONS["ipo"]
    assert "aot" in KNOWN_CONVENTIONS["cdpo"]
    assert TRL_LOSS_TYPES == {"sigmoid": "sigmoid", "ipo": "ipo", "cdpo": "aot"}


# --------------------------------------------------------------------------------------
# Failure modes, exercised with a stubbed TRL so they run without the extra
# --------------------------------------------------------------------------------------


@pytest.fixture
def pretend_trl_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crosscheck_module, "trl_available", lambda: True)


@pytest.mark.parametrize(
    ("variant", "expected_note"),
    [
        ("sigmoid", ""),
        ("ipo", KNOWN_CONVENTIONS["ipo"]),
        ("cdpo", KNOWN_CONVENTIONS["cdpo"]),
    ],
)
def test_a_real_disagreement_is_reported_as_one(
    monkeypatch: pytest.MonkeyPatch,
    pretend_trl_is_installed: None,
    variant: DPOVariant,
    expected_note: str,
) -> None:
    """The outcome the module exists to make impossible to miss.

    TRL is replaced by a stub that returns numbers nobody's objective would produce. Nothing may
    round that away: the status has to be `disagree`, and the documented convention for the
    variant is attached so a reader can rule it out before opening the source.
    """

    def wrong(_tensors: dict[str, Tensor], **_kwargs: Any) -> tuple[float, float, float]:
        return (99.0, 42.0, -42.0)

    monkeypatch.setattr(crosscheck_module, "_run_trl", wrong)
    result = crosscheck_against_trl(
        CrossCheckBatch.synthetic(3, seed=1),
        variant=variant,
        label_smoothing=0.1 if variant == "cdpo" else 0.0,
    )
    assert result.status == "disagree"
    assert result.agrees is False
    assert result.available is True
    assert result.loss_difference > 1.0
    assert result.reward_difference > 1.0
    assert result.note == expected_note


def test_a_near_miss_outside_tolerance_still_disagrees(
    monkeypatch: pytest.MonkeyPatch, pretend_trl_is_installed: None
) -> None:
    """The tolerance is a threshold, not a rounding: 1e-6 is a formula difference, not noise."""
    batch = CrossCheckBatch.synthetic(2, seed=0)
    tensors = _synthesise(batch, vocab=VOCAB)
    losses, chosen, rejected = crosscheck_module._mine(
        tensors,
        batch,
        beta=0.1,
        variant="sigmoid",
        label_smoothing=0.0,
        length_normalise=False,
    )

    def nudged(_tensors: dict[str, Tensor], *, row: int, **_kwargs: Any) -> tuple[float, ...]:
        return (float(losses[row]) + 1e-6, float(chosen[row]), float(rejected[row]))

    monkeypatch.setattr(crosscheck_module, "_run_trl", nudged)
    result = crosscheck_against_trl(batch)
    assert result.status == "disagree"
    assert result.loss_difference == pytest.approx(1e-6, rel=1e-3)
    assert result.chosen_reward_difference <= DEFAULT_TOLERANCE


def test_a_looser_tolerance_accepts_what_a_tight_one_rejects(
    monkeypatch: pytest.MonkeyPatch, pretend_trl_is_installed: None
) -> None:
    batch = CrossCheckBatch.synthetic(2, seed=0)
    tensors = _synthesise(batch, vocab=VOCAB)
    losses, chosen, rejected = crosscheck_module._mine(
        tensors,
        batch,
        beta=0.1,
        variant="sigmoid",
        label_smoothing=0.0,
        length_normalise=False,
    )

    def nudged(_tensors: dict[str, Tensor], *, row: int, **_kwargs: Any) -> tuple[float, ...]:
        return (float(losses[row]) + 1e-6, float(chosen[row]), float(rejected[row]))

    monkeypatch.setattr(crosscheck_module, "_run_trl", nudged)
    assert crosscheck_against_trl(batch, tolerance=1e-4).status == "agree"


@requires_trl
def test_a_missing_compute_loss_raises_rather_than_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The day TRL moves its objective, the honest outcome is a loud failure.

    A skip here would leave a green suite whose cross-check has quietly stopped comparing
    anything, which is the failure mode this module was written to avoid.
    """
    from trl.trainer.dpo_trainer import DPOTrainer

    monkeypatch.delattr(DPOTrainer, "_compute_loss")
    with pytest.raises(TrlInterfaceError, match="has no _compute_loss"):
        crosscheck_against_trl(CrossCheckBatch.synthetic(2))


@requires_trl
@pytest.mark.parametrize("failure", [AttributeError, KeyError, TypeError])
def test_a_changed_trl_interface_raises_rather_than_skipping(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    """Every way a moved interface can fail is named, so none of them degrades into a pass."""
    from trl.trainer.dpo_trainer import DPOTrainer

    def moved(*_args: Any, **_kwargs: Any) -> None:
        raise failure("gradient_checkpointing_kwargs")

    monkeypatch.setattr(DPOTrainer, "_compute_loss", moved)
    with pytest.raises(TrlInterfaceError, match="could not be driven directly"):
        crosscheck_against_trl(CrossCheckBatch.synthetic(2))


# --------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tolerance", [0.0, -1e-9])
def test_a_non_positive_tolerance_is_refused(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance must be positive"):
        crosscheck_against_trl(CrossCheckBatch.synthetic(2), tolerance=tolerance)


@pytest.mark.parametrize("vocab", [0, 1])
def test_a_degenerate_vocabulary_is_refused(vocab: int) -> None:
    """With one token in the vocabulary every log-probability is zero and nothing is realisable."""
    with pytest.raises(ValueError, match="vocab must be at least 2"):
        crosscheck_against_trl(CrossCheckBatch.synthetic(2), vocab=vocab)


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        crosscheck_against_trl(CrossCheckBatch.synthetic(2), variant="hinge")  # type: ignore[arg-type]


def test_validation_happens_before_trl_is_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad argument is a bad argument whether or not the optional extra is installed."""
    monkeypatch.setattr(crosscheck_module, "trl_available", lambda: False)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        crosscheck_against_trl(CrossCheckBatch.synthetic(2), tolerance=0.0)


# --------------------------------------------------------------------------------------
# The result object
# --------------------------------------------------------------------------------------


def test_max_difference_is_the_worst_number_anywhere() -> None:
    result = a_result(
        loss_difference=3.0, chosen_reward_difference=5.0, rejected_reward_difference=1.0
    )
    assert result.reward_difference == 5.0
    assert result.max_difference == 5.0


def test_a_loss_difference_can_be_the_worst_number() -> None:
    result = a_result(
        loss_difference=7.0, chosen_reward_difference=1.0, rejected_reward_difference=2.0
    )
    assert result.max_difference == 7.0


def test_explained_is_not_reported_as_agreement() -> None:
    """Status "explained" claims something about the write-up; only "agree" claims arithmetic."""
    result = a_result(status="explained", note=KNOWN_CONVENTIONS["ipo"], loss_difference=19.5)
    assert result.available is True
    assert result.agrees is False


def test_the_result_serialises_every_field_the_report_reads() -> None:
    payload = a_result(status="explained", normalised_loss_difference=1e-15).as_dict()
    assert set(payload) == {
        "status",
        "variant",
        "trl_loss_type",
        "trl_version",
        "beta",
        "label_smoothing",
        "pairs",
        "tolerance",
        "loss_difference",
        "chosen_reward_difference",
        "rejected_reward_difference",
        "normalised_loss_difference",
        "note",
        "reason",
    }
    assert payload["status"] == "explained"
    assert payload["normalised_loss_difference"] == 1e-15


def test_the_string_form_quotes_the_versions_and_the_worst_difference() -> None:
    line = str(a_result(loss_difference=1.5e-16))
    assert "sigmoid" in line
    assert "1.12.0" in line
    assert "4 pairs" in line
    assert "agree" in line


def test_the_default_tolerance_is_tight_enough_to_separate_noise_from_a_formula() -> None:
    """Float64 noise on this arithmetic sits near 1e-16, several orders below the threshold."""
    assert DEFAULT_TOLERANCE == 1e-9


def test_a_rejected_reward_discrepancy_is_not_explained_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The length-normalisation convention explains a loss, never a reward.

    The `explained` branch checked the loss residual and the *chosen* reward and forgot the
    rejected one, so a real disagreement in the rejected implicit reward was reported as a
    documented convention with a zero exit code. That is the single outcome this module
    exists to prevent, so it gets its own test rather than a comment.
    """
    batch = CrossCheckBatch.synthetic(6, seed=1, tokens=3)
    honest = crosscheck_against_trl(batch, beta=0.1, variant="ipo")
    assert honest.status in {"agree", "explained"}

    original = crosscheck_module._run_trl

    def poisoned(*args: object, **kwargs: object) -> object:
        losses, chosen, rejected = original(*args, **kwargs)  # type: ignore[arg-type]
        return losses, chosen, rejected + 1.0

    monkeypatch.setattr(crosscheck_module, "_run_trl", poisoned)
    poisoned_result = crosscheck_against_trl(batch, beta=0.1, variant="ipo")

    assert poisoned_result.status == "disagree"
    assert not poisoned_result.agrees
