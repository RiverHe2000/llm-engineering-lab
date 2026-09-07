"""Tests for the evaluation harness: batched generation and the report built from it.

The two modules are tested together because they are two halves of one claim. `generate.py`
turns examples into completions and `metrics.py` turns completions into the numbers the
promotion gate reads, and the failure everyone should fear is not a wrong average -- it is a
harness that attaches the right completion to the wrong gold record. Several tests here exist
only to rule that out: batching must not change any completion, reordering the corpus must
not change what any example gets, and left padding must leave every row's own final real
token last.

The Qwen tokenizer is not available offline, so the model and tokenizer are stubs. That is not
a compromise: the logic worth testing here -- padding, length bucketing, token accounting,
scoring, grouping -- is all harness logic, and a real checkpoint would only make the test slow
and non-deterministic while testing the same lines.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.eval.generate import (
    completion_length,
    generate_samples,
    left_pad,
    length_sorted_batches,
    resolve_pad_token_id,
)
from sftdpo.eval.metrics import (
    EvalReport,
    ExampleOutcome,
    FieldCoverage,
    MetricBlock,
    ParseGap,
    evaluate,
)
from sftdpo.schemas import (
    AdviceRecord,
    Example,
    GenerationConfig,
    Sample,
    Slice,
    ViolationKind,
)
from sftdpo.verify.reward import lenient_verifier

# --------------------------------------------------------------------------------------
# The task fixtures
# --------------------------------------------------------------------------------------

GOLD_JSON: Final[dict[str, Any]] = {
    "client_name": "Ada Lovelace",
    "record_date": "2026-03-31",
    "risk_profile": "balanced",
    "objectives": ["retire at 60"],
    "recommendations": [{"product": "Growth Fund", "action": "buy", "amount": 25000.0}],
    "fees": {"advice_fee": 3300.0, "ongoing_fee_pct": 0.88},
    "review_months": 12,
    "flags": ["tfn missing"],
}

GOLD: Final = AdviceRecord.model_validate(GOLD_JSON)
CLEAN: Final = GOLD.model_dump_json()
PROSE: Final = f"Certainly, here is the record:\n```json\n{CLEAN}\n```"
JUNK: Final = "I am not able to produce that record."


def variant(**patch: Any) -> str:
    """The gold object as a completion, with top-level fields replaced."""
    value = deepcopy(GOLD_JSON)
    value.update(patch)
    return json.dumps(value)


WRONG_NAME: Final = variant(client_name="Someone Else")
EXTRA_KEY: Final = variant(adviser="Bo")


def example(example_id: str, slice_: Slice = Slice.CLEAN) -> Example:
    """One test example whose note carries its own id, so a stub can route on it."""
    return Example(
        example_id=example_id,
        split="test",
        slice=slice_,
        note=f"[{example_id}] Reviewed Ada Lovelace, balanced, review in 12 months.",
        gold=GOLD,
    )


def sample(example_id: str, text: str, model: str = "base") -> Sample:
    return Sample(example_id=example_id, model=model, text=text)


def report_for(completions: Mapping[str, str], slices: Mapping[str, Slice] | None = None) -> Any:
    """Evaluate a mapping of example id to completion, all on one slice unless told otherwise."""
    where = {} if slices is None else slices
    examples = [example(key, where.get(key, Slice.CLEAN)) for key in completions]
    samples = [sample(key, text) for key, text in completions.items()]
    return evaluate(samples, examples)


# --------------------------------------------------------------------------------------
# Metrics: the five headline numbers
# --------------------------------------------------------------------------------------


def test_a_perfect_completion_scores_one_everywhere() -> None:
    report = report_for({"ex-1": CLEAN})
    assert report.overall.n == 1
    assert report.overall.json_valid_rate == 1.0
    assert report.overall.schema_valid_rate == 1.0
    assert report.overall.mean_field_f1 == pytest.approx(1.0)
    assert report.overall.exact_match_rate == 1.0
    assert report.overall.mean_reward == pytest.approx(1.0)


def test_junk_scores_zero_and_names_the_cause() -> None:
    report = report_for({"ex-1": JUNK})
    assert report.overall.json_valid_rate == 0.0
    assert report.overall.mean_field_f1 == pytest.approx(0.0)
    assert report.outcomes[0].violation is ViolationKind.NO_JSON


def test_an_object_with_an_invented_key_parses_but_fails_the_schema() -> None:
    """`extra="forbid"` is the point: a plausible extra key is not schema compliance."""
    outcome = report_for({"ex-1": EXTRA_KEY}).outcomes[0]
    assert outcome.json_valid
    assert not outcome.schema_valid
    assert not outcome.success
    assert outcome.violation is ViolationKind.EXTRA_FIELD


def test_a_schema_valid_record_with_one_wrong_field_still_succeeds() -> None:
    """Success is schema validity, not exactness: downstream can consume and correct it."""
    outcome = report_for({"ex-1": WRONG_NAME}).outcomes[0]
    assert outcome.success
    assert not outcome.exact_match
    assert 0.0 < outcome.field_f1 < 1.0


# --------------------------------------------------------------------------------------
# Metrics: the strict-versus-lenient parse gap
# --------------------------------------------------------------------------------------


def test_prose_wrapped_json_is_invalid_strictly_and_valid_leniently() -> None:
    """ "Emits JSON" and "emits JSON if you allow a repair" are different claims."""
    outcome = report_for({"ex-1": PROSE}).outcomes[0]
    assert not outcome.json_valid
    assert outcome.lenient_json_valid
    assert outcome.repaired_only


def test_the_parse_gap_is_the_size_of_the_repair_step() -> None:
    report = report_for({"a": CLEAN, "b": PROSE, "c": PROSE, "d": JUNK})
    assert report.parse_gap.strict_rate == pytest.approx(0.25)
    assert report.parse_gap.lenient_rate == pytest.approx(0.75)
    assert report.parse_gap.gap == pytest.approx(0.5)
    assert report.parse_gap.repaired == 2


def test_the_parse_gap_closes_when_nothing_needed_repairing() -> None:
    report = report_for({"a": CLEAN, "b": WRONG_NAME})
    assert report.parse_gap.gap == 0.0
    assert report.parse_gap.repaired == 0


@settings(max_examples=30, deadline=None)
@given(texts=st.lists(st.sampled_from([CLEAN, PROSE, JUNK, EXTRA_KEY]), min_size=1, max_size=8))
def test_the_parse_gap_is_never_negative(texts: list[str]) -> None:
    """Lenient parsing is a superset of strict parsing, so the gap has a sign."""
    report = report_for({f"ex-{index}": text for index, text in enumerate(texts)})
    assert report.parse_gap.gap >= 0.0
    assert report.parse_gap.repaired <= report.parse_gap.n


def test_an_empty_parse_gap_reports_zeroes_rather_than_dividing_by_nothing() -> None:
    empty = ParseGap.over([])
    assert (empty.n, empty.strict_rate, empty.lenient_rate, empty.gap) == (0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------------------
# Metrics: the breakdowns that stop an average hiding a regression
# --------------------------------------------------------------------------------------


def test_per_slice_blocks_report_each_slice_separately() -> None:
    report = report_for(
        {"a": CLEAN, "b": JUNK, "c": CLEAN},
        {"a": Slice.CLEAN, "b": Slice.LONG_CONTEXT, "c": Slice.LONG_CONTEXT},
    )
    assert report.per_slice[Slice.CLEAN].schema_valid_rate == 1.0
    assert report.per_slice[Slice.LONG_CONTEXT].schema_valid_rate == pytest.approx(0.5)
    assert report.per_slice[Slice.LONG_CONTEXT].n == 2


def test_per_slice_blocks_follow_the_enum_order_not_the_sample_order() -> None:
    """A rendered table must not depend on which slice the first sample came from."""
    report = report_for(
        {"a": CLEAN, "b": CLEAN},
        {"a": Slice.MANY_ITEMS, "b": Slice.CLEAN},
    )
    assert list(report.per_slice) == [Slice.CLEAN, Slice.MANY_ITEMS]


def test_only_slices_that_occur_get_a_block() -> None:
    report = report_for({"a": CLEAN})
    assert list(report.per_slice) == [Slice.CLEAN]


def test_per_violation_blocks_group_by_the_headline_cause() -> None:
    report = report_for({"a": CLEAN, "b": JUNK, "c": EXTRA_KEY, "d": JUNK})
    assert report.per_violation[ViolationKind.NO_JSON].n == 2
    assert report.per_violation[ViolationKind.EXTRA_FIELD].n == 1
    # A clean completion has no violation, so it appears in no violation block.
    assert sum(block.n for block in report.per_violation.values()) == 3


def test_per_violation_blocks_follow_the_severity_order() -> None:
    report = report_for({"a": EXTRA_KEY, "b": JUNK})
    assert list(report.per_violation) == [ViolationKind.NO_JSON, ViolationKind.EXTRA_FIELD]


def test_an_empty_metric_block_is_zeroes_rather_than_a_hole_in_the_table() -> None:
    block = MetricBlock.over([])
    assert block.n == 0
    assert block.as_dict() == {
        "n": 0.0,
        "json_valid_rate": 0.0,
        "schema_valid_rate": 0.0,
        "mean_field_f1": 0.0,
        "exact_match_rate": 0.0,
        "mean_reward": 0.0,
    }


def test_metric_block_as_dict_widens_n_for_a_uniform_value_type() -> None:
    values = MetricBlock.over(report_for({"a": CLEAN, "b": JUNK}).outcomes).as_dict()
    assert values["n"] == 2.0
    assert isinstance(values["n"], float)
    assert values["schema_valid_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Metrics: the report as an object
# --------------------------------------------------------------------------------------


def test_the_report_keeps_the_sample_order() -> None:
    report = report_for({"z": CLEAN, "a": JUNK, "m": CLEAN})
    assert report.example_ids == ("z", "a", "m")
    assert report.n == 3


def test_outcome_for_finds_a_row_and_refuses_an_unknown_id() -> None:
    report = report_for({"a": CLEAN})
    assert report.outcome_for("a").example_id == "a"
    with pytest.raises(KeyError, match="no outcome for example"):
        report.outcome_for("b")


def test_the_paired_views_are_keyed_by_example_id() -> None:
    report = report_for({"a": CLEAN, "b": JUNK}, {"a": Slice.CLEAN, "b": Slice.DISTRACTOR})
    assert report.success_by_example() == {"a": 1.0, "b": 0.0}
    assert report.field_f1_by_example()["a"] == pytest.approx(1.0)
    assert report.slice_by_example() == {"a": Slice.CLEAN, "b": Slice.DISTRACTOR}


def test_as_dict_is_json_serialisable_and_drops_the_per_example_rows() -> None:
    summary = report_for({"a": CLEAN, "b": PROSE}).as_dict()
    assert "outcomes" not in summary
    assert summary["n"] == 2
    assert json.loads(json.dumps(summary))["parse_gap"]["gap"] == pytest.approx(0.5)


def test_the_report_is_frozen() -> None:
    report = report_for({"a": CLEAN})
    with pytest.raises(ValueError, match="frozen"):
        report.model = "other"


def test_the_model_name_is_inferred_when_the_samples_agree() -> None:
    report = evaluate([sample("a", CLEAN, model="sft")], [example("a")])
    assert report.model == "sft"


def test_a_report_refuses_to_mix_two_variants() -> None:
    """A report built from two models would later be compared against itself."""
    samples = [sample("a", CLEAN, model="base"), sample("b", CLEAN, model="dpo")]
    with pytest.raises(ValueError, match="cannot infer the model name"):
        evaluate(samples, [example("a"), example("b")])


def test_an_explicit_model_name_overrides_the_samples() -> None:
    report = evaluate([sample("a", CLEAN, model="base")], [example("a")], model="base-rerun")
    assert report.model == "base-rerun"


def test_evaluate_refuses_two_completions_for_one_example() -> None:
    """The report feeds a paired comparison, where every example counts exactly once."""
    samples = [sample("a", CLEAN), sample("a", JUNK)]
    with pytest.raises(ValueError, match="two samples for example"):
        evaluate(samples, [example("a")])


def test_evaluate_refuses_a_sample_with_no_example() -> None:
    with pytest.raises(ValueError, match="no example for sample"):
        evaluate([sample("ghost", CLEAN)], [example("a")])


def test_evaluate_refuses_a_corpus_with_a_duplicate_id() -> None:
    with pytest.raises(ValueError, match="duplicate example_id"):
        evaluate([sample("a", CLEAN)], [example("a"), example("a")])


def test_evaluate_accepts_an_empty_run_when_told_the_model_name() -> None:
    report = evaluate([], [], model="base")
    assert report.n == 0
    assert report.overall.n == 0
    assert report.per_slice == {}


def test_custom_verifiers_are_honoured() -> None:
    """Scoring the headline numbers leniently is a different measurement, and is available."""
    report = evaluate(
        [sample("a", PROSE)],
        [example("a")],
        strict=lenient_verifier(),
        lenient=lenient_verifier(),
    )
    assert report.outcomes[0].json_valid
    assert report.parse_gap.gap == 0.0


def test_example_outcome_success_tracks_schema_validity() -> None:
    row = ExampleOutcome(
        example_id="a",
        slice=Slice.CLEAN,
        json_valid=True,
        schema_valid=True,
        field_f1=0.5,
        exact_match=False,
        reward=0.7,
        lenient_json_valid=True,
    )
    assert row.success
    assert not row.repaired_only


# --------------------------------------------------------------------------------------
# Generation stubs
# --------------------------------------------------------------------------------------

PAD_ID: Final = 0
EOS_ID: Final = 1
BOS_ID: Final = 2
FIRST_TEXT_ID: Final = 3


class ByteTokenizer:
    """A deterministic byte-level stand-in for the Qwen tokenizer.

    Invertible and offline, which is all the harness needs: the properties under test are
    about padding, ordering and token accounting, none of which care what the vocabulary
    means. Ids below `FIRST_TEXT_ID` are the special tokens, so a decoded row shows
    immediately whether filler leaked into the text.
    """

    def __init__(self, *, pad_token_id: int | None = PAD_ID, eos_token_id: int | None = EOS_ID):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [byte + FIRST_TEXT_ID for byte in text.encode("utf-8")]
        return [BOS_ID, *ids] if add_special_tokens else ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        if not skip_special_tokens:
            return "".join(
                f"<{token}>" if token < FIRST_TEXT_ID else chr(token - FIRST_TEXT_ID)
                for token in ids
            )
        kept = bytes(token - FIRST_TEXT_ID for token in ids if token >= FIRST_TEXT_ID)
        return kept.decode("utf-8", errors="ignore")


class ScriptedModel:
    """A causal-LM stand-in that replies with a canned completion per example.

    It routes on the example id embedded in the note, so a completion can only be attached to
    the wrong example if the harness misaligns the batch -- which is exactly what the ordering
    tests are looking for. This base class is the most minimal thing the harness has to cope
    with: no train/eval switch and no way of saying which device it is on.
    """

    def __init__(self, tokenizer: ByteTokenizer, replies: Mapping[str, str]) -> None:
        self.tokenizer = tokenizer
        self.replies = dict(replies)
        self.calls: list[dict[str, Any]] = []

    def _reply_for(self, prompt: str) -> str:
        for example_id, reply in self.replies.items():
            if f"[{example_id}]" in prompt:
                return reply
        raise AssertionError(f"the stub saw a prompt it cannot route: {prompt[:60]!r}")

    def generate(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        pad_token_id: int,
    ) -> Any:
        self.calls.append(
            {
                "width": int(input_ids.shape[1]),
                "rows": int(input_ids.shape[0]),
                "do_sample": do_sample,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
            }
        )
        rows: list[list[int]] = []
        for ids_row, mask_row in zip(input_ids.tolist(), attention_mask.tolist(), strict=True):
            real = [token for token, keep in zip(ids_row, mask_row, strict=True) if keep == 1]
            reply = self._reply_for(self.tokenizer.decode(real, skip_special_tokens=True))
            produced = self.tokenizer.encode(reply, add_special_tokens=False)[:max_new_tokens]
            if len(produced) < max_new_tokens and self.tokenizer.eos_token_id is not None:
                produced.append(self.tokenizer.eos_token_id)
            rows.append(produced)
        width = max(len(row) for row in rows)
        squared = [row + [pad_token_id] * (width - len(row)) for row in rows]
        return torch.cat([input_ids, torch.tensor(squared, dtype=torch.long)], dim=1)


class EvalModel(ScriptedModel):
    """A model with a train/eval switch, which every `transformers` model has."""

    def __init__(self, tokenizer: ByteTokenizer, replies: Mapping[str, str]) -> None:
        super().__init__(tokenizer, replies)
        self.eval_calls = 0

    def eval(self) -> None:
        """Record that the harness took the model out of training mode."""
        self.eval_calls += 1


class DeviceModel(EvalModel):
    """The realistic case: the model knows which device it is on."""

    def __init__(self, tokenizer: ByteTokenizer, replies: Mapping[str, str]) -> None:
        super().__init__(tokenizer, replies)
        self.device = torch.device("cpu")


class ParameterModel(EvalModel):
    """A model that must be asked its parameters to find out where it lives."""

    def parameters(self) -> Any:
        return iter([torch.zeros(1)])


class HomelessModel(EvalModel):
    """A model that answers the placement question with nothing at all."""

    def parameters(self) -> Any:
        return iter([])


class EcholessModel(DeviceModel):
    """A model whose `generate` returns the prompt and nothing else."""

    def generate(self, **kwargs: Any) -> Any:
        return kwargs["input_ids"]


def scripted(replies: Mapping[str, str]) -> tuple[DeviceModel, ByteTokenizer]:
    tokenizer = ByteTokenizer()
    return DeviceModel(tokenizer, replies), tokenizer


# --------------------------------------------------------------------------------------
# Generation: padding and bucketing
# --------------------------------------------------------------------------------------


def test_resolve_pad_token_id_prefers_the_real_pad_token() -> None:
    assert resolve_pad_token_id(ByteTokenizer()) == PAD_ID


def test_resolve_pad_token_id_falls_back_to_the_stop_token() -> None:
    """Instruct tokenizers routinely ship without a pad token; reusing EOS is the remedy."""
    assert resolve_pad_token_id(ByteTokenizer(pad_token_id=None)) == EOS_ID


def test_resolve_pad_token_id_refuses_to_guess() -> None:
    bare = ByteTokenizer(pad_token_id=None, eos_token_id=None)
    with pytest.raises(ValueError, match="neither pad_token_id nor eos_token_id"):
        resolve_pad_token_id(bare)


def test_left_pad_leaves_every_rows_own_final_token_last() -> None:
    """The property batched decoder-only generation depends on."""
    input_ids, mask = left_pad([[7, 8, 9], [4], [5, 6]], PAD_ID)
    assert input_ids.shape == (3, 3)
    assert input_ids[:, -1].tolist() == [9, 4, 6]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1], [0, 1, 1]]
    assert input_ids.dtype == torch.long


@settings(max_examples=50, deadline=None)
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=6),
)
def test_left_pad_masks_exactly_the_filler(lengths: list[int]) -> None:
    sequences = [list(range(FIRST_TEXT_ID, FIRST_TEXT_ID + length)) for length in lengths]
    input_ids, mask = left_pad(sequences, PAD_ID)
    assert mask.sum(dim=1).tolist() == lengths
    assert input_ids.shape[1] == max(lengths)
    for row, length in zip(input_ids.tolist(), lengths, strict=True):
        assert row[-length:] == list(range(FIRST_TEXT_ID, FIRST_TEXT_ID + length))


def test_left_pad_refuses_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        left_pad([], PAD_ID)


def test_left_pad_refuses_an_empty_prompt() -> None:
    """An empty prompt gives the model no position to continue from."""
    with pytest.raises(ValueError, match="empty prompt"):
        left_pad([[1, 2], []], PAD_ID)


@settings(max_examples=50, deadline=None)
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=200), min_size=1, max_size=20),
    batch_size=st.integers(min_value=1, max_value=8),
)
def test_length_sorted_batches_partition_the_corpus(lengths: list[int], batch_size: int) -> None:
    """Every example is generated for exactly once, however the lengths fall."""
    batches = length_sorted_batches(lengths, batch_size)
    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == list(range(len(lengths)))
    assert all(len(batch) <= batch_size for batch in batches)


def test_length_sorted_batches_group_similar_lengths_and_break_ties_by_index() -> None:
    assert length_sorted_batches([9, 1, 9, 1, 5], 2) == [(1, 3), (4, 0), (2,)]


def test_length_sorted_batches_of_an_empty_corpus_is_empty() -> None:
    assert length_sorted_batches([], 4) == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_length_sorted_batches_rejects_a_useless_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        length_sorted_batches([1, 2], batch_size)


# --------------------------------------------------------------------------------------
# Generation: token accounting
# --------------------------------------------------------------------------------------


def test_completion_length_stops_at_the_stop_token() -> None:
    assert completion_length([9, 8, EOS_ID, 7], eos_token_id=EOS_ID, pad_token_id=PAD_ID) == 3


def test_completion_length_stops_at_the_filler() -> None:
    assert completion_length([9, 8, PAD_ID, PAD_ID], eos_token_id=EOS_ID, pad_token_id=PAD_ID) == 2


def test_completion_length_counts_a_row_that_ran_to_the_budget() -> None:
    assert completion_length([9, 8, 7], eos_token_id=EOS_ID, pad_token_id=PAD_ID) == 3


def test_completion_length_counts_a_shared_stop_and_pad_token_once() -> None:
    """When a tokenizer reuses EOS as its pad token, the stop token is output, not filler."""
    assert completion_length([9, EOS_ID, EOS_ID], eos_token_id=EOS_ID, pad_token_id=EOS_ID) == 2


def test_completion_length_without_special_tokens_counts_everything() -> None:
    assert completion_length([5, 6, 7], eos_token_id=None, pad_token_id=None) == 3


# --------------------------------------------------------------------------------------
# Generation: the batched loop
# --------------------------------------------------------------------------------------

# The stub tokenizer spends one token per byte, so the default budget of 320 would clip the
# longest reply and the tests would be measuring the budget rather than the harness.
ROOMY: Final = GenerationConfig(max_new_tokens=1024)


def test_generation_returns_one_sample_per_example_in_the_callers_order() -> None:
    replies = {"ex-a": CLEAN, "ex-b": JUNK, "ex-c": PROSE}
    model, tokenizer = scripted(replies)
    examples = [example(key) for key in replies]
    samples = generate_samples(
        model, tokenizer, examples, model_name="base", batch_size=2, config=ROOMY
    )
    assert [item.example_id for item in samples] == ["ex-a", "ex-b", "ex-c"]
    assert [item.text for item in samples] == [CLEAN, JUNK, PROSE]
    assert {item.model for item in samples} == {"base"}


def test_generation_records_the_token_accounting_and_the_cost() -> None:
    model, tokenizer = scripted({"ex-a": JUNK})
    [item] = generate_samples(model, tokenizer, [example("ex-a")], model_name="base")
    assert item.prompt_tokens > 0
    # The reply plus the stop token the stub emitted.
    assert item.completion_tokens == len(JUNK.encode("utf-8")) + 1
    assert item.latency_ms >= 0.0


def test_batching_does_not_change_a_single_completion() -> None:
    """If it did, the harness would be scoring padding rather than the model."""
    replies = {f"ex-{index}": CLEAN if index % 2 else JUNK for index in range(6)}
    examples = [example(key) for key in replies]
    results: list[list[str]] = []
    for batch_size in (1, 2, 5, 50):
        model, tokenizer = scripted(replies)
        samples = generate_samples(
            model, tokenizer, examples, model_name="base", batch_size=batch_size
        )
        results.append([item.text for item in samples])
    assert results[0] == results[1] == results[2] == results[3]


def test_reordering_the_corpus_does_not_change_what_an_example_gets() -> None:
    """Length bucketing reorders the work; it must not reorder the answers."""
    replies = {"short": JUNK, "long": PROSE, "middle": CLEAN}
    forward = [example(key) for key in replies]
    model, tokenizer = scripted(replies)
    straight = generate_samples(
        model, tokenizer, forward, model_name="base", batch_size=2, config=ROOMY
    )

    model, tokenizer = scripted(replies)
    reversed_samples = generate_samples(
        model, tokenizer, list(reversed(forward)), model_name="base", batch_size=2, config=ROOMY
    )
    assert {item.example_id: item.text for item in straight} == {
        item.example_id: item.text for item in reversed_samples
    }


def test_generation_puts_the_model_in_eval_mode_and_asks_for_greedy_decoding() -> None:
    model, tokenizer = scripted({"ex-a": CLEAN})
    generate_samples(model, tokenizer, [example("ex-a")], model_name="base")
    assert model.eval_calls == 1
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["num_beams"] == 1


def test_the_token_budget_truncates_a_long_reply() -> None:
    model, tokenizer = scripted({"ex-a": CLEAN})
    [item] = generate_samples(
        model,
        tokenizer,
        [example("ex-a")],
        model_name="base",
        config=GenerationConfig(max_new_tokens=12),
    )
    assert item.completion_tokens == 12
    assert item.text == CLEAN[:12]


def test_an_empty_corpus_produces_no_samples_and_no_generate_call() -> None:
    model, tokenizer = scripted({})
    assert generate_samples(model, tokenizer, [], model_name="base") == []
    assert model.calls == []


def test_generation_refuses_to_sample() -> None:
    """Any variance the decoder adds is variance the promotion gate has to pay for."""
    model, tokenizer = scripted({"ex-a": CLEAN})
    with pytest.raises(ValueError, match="decodes greedily"):
        generate_samples(
            model,
            tokenizer,
            [example("ex-a")],
            model_name="base",
            config=GenerationConfig(temperature=0.7),
        )


def test_generation_refuses_an_empty_token_budget() -> None:
    model, tokenizer = scripted({"ex-a": CLEAN})
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        generate_samples(
            model,
            tokenizer,
            [example("ex-a")],
            model_name="base",
            config=GenerationConfig(max_new_tokens=0),
        )


def test_generation_refuses_a_model_that_does_not_echo_the_prompt() -> None:
    tokenizer = ByteTokenizer()
    model = EcholessModel(tokenizer, {"ex-a": CLEAN})
    with pytest.raises(ValueError, match="expects the prompt to be echoed back"):
        generate_samples(model, tokenizer, [example("ex-a")], model_name="base")


def test_the_batch_is_placed_where_the_parameters_live() -> None:
    tokenizer = ByteTokenizer()
    model = ParameterModel(tokenizer, {"ex-a": CLEAN})
    [item] = generate_samples(model, tokenizer, [example("ex-a")], model_name="base")
    assert item.text == CLEAN


def test_a_model_with_no_switch_and_no_placement_hint_still_runs() -> None:
    """A bare callable wrapper has neither `eval` nor `device`, and must not be a crash."""
    tokenizer = ByteTokenizer()
    model = ScriptedModel(tokenizer, {"ex-a": CLEAN})
    [item] = generate_samples(model, tokenizer, [example("ex-a")], model_name="base")
    assert item.text == CLEAN


def test_a_model_with_no_parameters_falls_back_to_the_cpu() -> None:
    tokenizer = ByteTokenizer()
    model = HomelessModel(tokenizer, {"ex-a": CLEAN})
    [item] = generate_samples(model, tokenizer, [example("ex-a")], model_name="base")
    assert item.text == CLEAN


def test_an_explicit_device_overrides_the_model() -> None:
    model, tokenizer = scripted({"ex-a": CLEAN})
    [item] = generate_samples(
        model, tokenizer, [example("ex-a")], model_name="base", device=torch.device("cpu")
    )
    assert item.text == CLEAN


def test_a_tokenizer_without_a_pad_token_still_generates() -> None:
    """EOS doubles as the filler, and the stop token must still be counted exactly once."""
    tokenizer = ByteTokenizer(pad_token_id=None)
    model = DeviceModel(tokenizer, {"ex-a": JUNK, "ex-b": CLEAN})
    samples = generate_samples(
        model, tokenizer, [example("ex-a"), example("ex-b")], model_name="base", batch_size=2
    )
    assert [item.text for item in samples] == [JUNK, CLEAN]
    assert samples[0].completion_tokens == len(JUNK.encode("utf-8")) + 1


def test_generated_samples_feed_straight_into_a_report() -> None:
    """The end-to-end shape: generate, score, and read the numbers off the report."""
    replies = {"ex-a": CLEAN, "ex-b": PROSE, "ex-c": JUNK}
    model, tokenizer = scripted(replies)
    examples = [example(key) for key in replies]
    samples = generate_samples(
        model, tokenizer, examples, model_name="base", batch_size=2, config=ROOMY
    )
    report: EvalReport = evaluate(samples, examples)
    assert report.model == "base"
    assert report.overall.schema_valid_rate == pytest.approx(1 / 3)
    assert report.parse_gap.gap == pytest.approx(1 / 3)


# --------------------------------------------------------------------------------------
# Per-field coverage
#
# The block that answers a question no average over records can: not how much of each record
# was right, but which field the model stopped producing at all.
# --------------------------------------------------------------------------------------


def test_coverage_counts_omissions_and_hits_per_top_level_field() -> None:
    """One record right, one with `flags` emptied: recall halves, precision stays perfect."""
    report = report_for({"a": CLEAN, "b": variant(flags=[])})
    coverage = report.per_field["flags"]
    assert coverage.expected == 2
    assert coverage.emitted == 1
    assert coverage.correct == 1
    assert coverage.recall == pytest.approx(0.5)
    assert coverage.precision == pytest.approx(1.0)


def test_an_invented_member_costs_precision_without_costing_recall() -> None:
    report = report_for({"a": variant(flags=["tfn missing", "invented"])})
    coverage = report.per_field["flags"]
    assert coverage.recall == pytest.approx(1.0)
    assert coverage.precision == pytest.approx(0.5)


def test_coverage_is_keyed_by_the_top_level_field_not_the_path() -> None:
    report = report_for({"a": CLEAN})
    assert {"flags", "recommendations", "fees", "client_name"} <= set(report.per_field)
    assert "fees.advice_fee" not in report.per_field


def test_a_field_gold_never_asks_for_scores_full_recall_rather_than_zero() -> None:
    empty = FieldCoverage(field="flags")
    assert empty.recall == 1.0
    assert empty.precision == 1.0


def test_coverage_serialises_with_its_two_derived_rates() -> None:
    row = FieldCoverage(field="flags", expected=4, emitted=2, correct=1)
    assert row.as_dict() == {
        "field": "flags",
        "expected": 4,
        "emitted": 2,
        "correct": 1,
        "recall": 0.25,
        "precision": 0.5,
    }


def test_the_report_summary_carries_the_per_field_block() -> None:
    report = report_for({"a": CLEAN})
    assert report.as_dict()["per_field"]["flags"]["recall"] == 1.0


def test_an_unparseable_completion_emits_no_field_at_all() -> None:
    report = report_for({"a": JUNK})
    assert report.per_field
    assert all(row.emitted == 0 for row in report.per_field.values())
    assert all(row.recall == 0.0 for row in report.per_field.values())


def test_coverage_survives_a_round_trip_through_json() -> None:
    report = report_for({"a": CLEAN, "b": variant(flags=[])})
    restored = EvalReport.model_validate_json(report.model_dump_json())
    assert restored.per_field == report.per_field
