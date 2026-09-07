"""Tests for the pipeline driver.

The centrepiece is `smoke_run`: one module-scoped fixture that executes every stage of the
real experiment on the two-layer CI model in a few seconds, and a family of tests that assert
what it left on disk. That is the test which proves the pipeline is a pipeline -- nine stages
that hand artefacts to each other -- rather than nine functions that happen to live in one
file. Everything else here pins down a property the driver depends on: that the layout claims
distinct paths, that a stage's completeness check is not fooled by an empty checkpoint
directory, that the stand-in tokenizer is stable across processes, and that the report is
byte-stable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sftdpo.eval.metrics import EvalReport
from sftdpo.modeling.chat import ChatFormatter, common_prefix_length
from sftdpo.modeling.collate import encode_sft_example
from sftdpo.modeling.lora import reference_context, trainable_parameter_report
from sftdpo.pipeline import (
    COMPARISONS,
    EVAL_VARIANTS,
    STAGE_ORDER,
    HubProvider,
    PipelineConfig,
    PipelineError,
    PipelineResult,
    RunLayout,
    SmokeProvider,
    Stage,
    StageRecord,
    TinyTokenizer,
    _complete,
    _resolve_provider,
    build_report,
    config_digest,
    generate_replies,
    gold_fallback_pairs,
    measure_alignment_tax,
    publish_adapter,
    run_pipeline,
    smoke_config,
)
from sftdpo.schemas import Example, GenerationConfig, PreferencePair, Sample
from sftdpo.task.dataset import Dataset, build_dataset
from sftdpo.task.generate import render_prompt
from sftdpo.verify.reward import strict_verifier

TEXT = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=400)


# --------------------------------------------------------------------------------------
# The layout and the skip rule
# --------------------------------------------------------------------------------------


def test_stage_order_is_the_declaration_order() -> None:
    order = list(STAGE_ORDER)
    assert order == list(Stage)
    assert order.index(Stage.DATA) < order.index(Stage.SFT) < order.index(Stage.MINE)
    assert order.index(Stage.MINE) < order.index(Stage.DPO) < order.index(Stage.EVAL_DPO)
    assert order.index(Stage.EVAL_DPO) < order.index(Stage.COMPARE) < order.index(Stage.REPORT)


def test_every_declared_output_lives_under_the_run_root(tmp_path: Path) -> None:
    layout = RunLayout(root=tmp_path / "run")
    for stage in STAGE_ORDER:
        outputs = layout.outputs(stage)
        assert outputs, f"{stage} declares no artefacts, so it could never be skipped"
        for path in outputs:
            assert layout.root in path.parents


def test_no_two_stages_claim_the_same_artefact(tmp_path: Path) -> None:
    """Two stages sharing an output would make one of them skip on the other's work."""
    claimed: dict[Path, Stage] = {}
    layout = RunLayout(root=tmp_path)
    for stage in STAGE_ORDER:
        for path in layout.outputs(stage):
            assert path not in claimed, f"{path} is claimed by {claimed.get(path)} and {stage}"
            claimed[path] = stage


def test_complete_is_false_for_a_missing_file(tmp_path: Path) -> None:
    assert not _complete([tmp_path / "absent.json"])


def test_complete_is_false_for_an_empty_checkpoint_directory(tmp_path: Path) -> None:
    """An interrupted run leaves the directory and no weights; that is not a finished stage."""
    empty = tmp_path / "adapter"
    empty.mkdir()
    assert not _complete([empty])


def test_complete_is_true_only_when_every_artefact_is_there(tmp_path: Path) -> None:
    present = tmp_path / "a.json"
    present.write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights.safetensors").write_bytes(b"0")
    assert _complete([present, adapter])
    assert not _complete([present, adapter, tmp_path / "missing"])


def test_complete_is_false_for_no_artefacts_at_all() -> None:
    assert not _complete([])


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def test_evaluation_decodes_greedily_and_sampling_does_not(tmp_path: Path) -> None:
    config = PipelineConfig(run_dir=tmp_path)
    assert config.eval_config().greedy
    assert not config.sampling_config().greedy


def test_training_configs_point_at_the_run_directory(tmp_path: Path) -> None:
    config = PipelineConfig(run_dir=tmp_path, sft_epochs=2, dpo_epochs=5)
    assert config.sft_train_config().output_dir == config.layout.sft_dir
    assert config.dpo_train_config().output_dir == config.layout.dpo_dir
    assert config.sft_train_config().epochs == 2
    assert config.dpo_train_config().epochs == 5


def test_the_gold_fallback_is_off_by_default_and_on_only_for_the_smoke_run(
    tmp_path: Path,
) -> None:
    assert not PipelineConfig(run_dir=tmp_path).gold_pair_fallback
    assert smoke_config(tmp_path).gold_pair_fallback


def test_configuration_is_frozen(tmp_path: Path) -> None:
    config = PipelineConfig(run_dir=tmp_path)
    with pytest.raises(ValueError):
        config.seed = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_train", 0),
        ("sample_k", 0),
        ("temperature", 0.0),
        ("beta", 0.0),
        ("variant", "dpo"),
        ("json_floor", 1.5),
        ("alpha", 1.0),
    ],
)
def test_configuration_rejects_a_setting_that_cannot_work(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(run_dir=tmp_path, **{field: value})


def test_smoke_config_overrides_win_over_the_defaults(tmp_path: Path) -> None:
    assert smoke_config(tmp_path, seed=11).seed == 11


# --------------------------------------------------------------------------------------
# The stand-in tokenizer
# --------------------------------------------------------------------------------------


@given(text=TEXT)
def test_tokenizer_splits_into_one_token_per_chunk(text: str) -> None:
    tokenizer = TinyTokenizer(chunk=8)
    assert len(tokenizer.encode(text)) == math.ceil(len(text) / 8)


@given(text=TEXT)
def test_tokenizer_ids_are_inside_the_vocabulary(text: str) -> None:
    tokenizer = TinyTokenizer()
    assert all(2 <= token < tokenizer.vocab_size for token in tokenizer.encode(text))


@given(text=TEXT)
def test_tokenizer_ids_do_not_depend_on_the_instance(text: str) -> None:
    """The property the pipeline depends on: an adapter trained in one stage is used in the
    next, so the same text must tokenise the same way in a different process."""
    assert TinyTokenizer().encode(text) == TinyTokenizer().encode(text)


@given(text=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=64))
def test_tokenizer_round_trips_a_single_chunk(text: str) -> None:
    tokenizer = TinyTokenizer(chunk=64)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_renders_an_unseen_id_visibly() -> None:
    """A randomly initialised model emits ids from all over the vocabulary; decoding them to
    nothing would leave the verifier with no text to score."""
    assert TinyTokenizer().decode([9001]) == "<9001>"


def test_tokenizer_drops_the_stop_token() -> None:
    tokenizer = TinyTokenizer()
    assert tokenizer.decode([tokenizer.eos_token_id]) == ""
    assert tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False) == "<eos>"


def test_tokenizer_refuses_to_add_special_tokens() -> None:
    with pytest.raises(ValueError, match="no special tokens"):
        TinyTokenizer().encode("hello", add_special_tokens=True)


@pytest.mark.parametrize(("field", "value"), [("chunk", 0), ("vocab_size", 3)])
def test_tokenizer_rejects_a_degenerate_setting(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        TinyTokenizer(**{field: value})


def test_tokenizer_supports_completion_only_masking() -> None:
    """The one thing the pipeline actually asks of it: a supervised feature with a prompt
    boundary strictly inside the sequence."""
    tokenizer = TinyTokenizer()
    formatter = ChatFormatter()
    example = build_dataset(seed=5, n_train=1, n_val=0, n_test=0).train[0]
    feature = encode_sft_example(
        formatter, tokenizer, render_prompt(example.note), example.gold_json
    )
    assert 0 < feature.prompt_len < feature.total_len
    assert feature.completion_len >= 1


def test_tokenizer_prompt_ids_are_a_prefix_of_the_full_text(tmp_path: Path) -> None:
    """Fixed-width chunking may merge the last prompt chunk with the first answer chunk, so
    the shared prefix is allowed to be one token short and no shorter."""
    tokenizer = TinyTokenizer(chunk=16)
    prompt = "the adviser note goes here and runs on for a while"
    full = prompt + '{"client_name": "Rosa"}'
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(full)
    assert common_prefix_length(prompt_ids, full_ids) >= len(prompt_ids) - 1


# --------------------------------------------------------------------------------------
# The provider
# --------------------------------------------------------------------------------------


def test_smoke_provider_attaches_an_adapter_only_when_asked() -> None:
    provider = SmokeProvider()
    plain = provider.load("tiny")
    adapted = provider.load("tiny", trainable=True)
    assert not hasattr(plain.model, "disable_adapter")
    report = trainable_parameter_report(adapted.model)
    assert 0 < report.trainable < report.total
    assert report.percentage < 1.0


def test_smoke_provider_gives_dpo_a_reference_policy() -> None:
    """DPO's whole memory argument is that the reference is the policy with the adapter off."""
    adapted = SmokeProvider().load("tiny", trainable=True)
    with reference_context(adapted.model) as reference:
        assert reference is adapted.model


def test_smoke_provider_reloads_a_saved_adapter(tmp_path: Path) -> None:
    trained = SmokeProvider().load("tiny", trainable=True)
    trained.model.save_pretrained(str(tmp_path / "adapter"))
    reloaded = SmokeProvider().load("tiny", adapter=tmp_path / "adapter")
    assert hasattr(reloaded.model, "disable_adapter")


def test_smoke_provider_records_the_name_it_was_given() -> None:
    assert SmokeProvider().load("Qwen/whatever").info.name == "Qwen/whatever"


# --------------------------------------------------------------------------------------
# The gold-pair fallback
# --------------------------------------------------------------------------------------


def _examples(n: int = 2) -> tuple[Example, ...]:
    return build_dataset(seed=13, n_train=n, n_val=0, n_test=0).train


def _sample(example: Example, text: str, index: int = 0) -> Sample:
    return Sample(example_id=example.example_id, model="tiny", text=text, sample_index=index)


def test_gold_fallback_pairs_choose_gold_over_the_worst_sample() -> None:
    examples = _examples(1)
    example = examples[0]
    pairs = gold_fallback_pairs(
        examples,
        [_sample(example, "not json at all"), _sample(example, example.gold_json, 1)],
        strict_verifier(),
        min_margin=0.05,
        max_pairs_per_prompt=2,
    )
    assert len(pairs) == 1
    assert pairs[0].chosen == example.gold_json
    assert pairs[0].rejected == "not json at all"
    assert pairs[0].margin > 0.05


def test_gold_fallback_pairs_respect_the_cap() -> None:
    examples = _examples(1)
    example = examples[0]
    samples = [_sample(example, f"junk {index}", index) for index in range(5)]
    pairs = gold_fallback_pairs(
        examples, samples, strict_verifier(), min_margin=0.05, max_pairs_per_prompt=2
    )
    assert len(pairs) == 2


def test_gold_fallback_pairs_skip_blank_completions() -> None:
    examples = _examples(1)
    pairs = gold_fallback_pairs(
        examples,
        [_sample(examples[0], "   ")],
        strict_verifier(),
        min_margin=0.05,
        max_pairs_per_prompt=2,
    )
    assert pairs == []


def test_gold_fallback_pairs_never_pair_gold_with_itself() -> None:
    examples = _examples(1)
    example = examples[0]
    pairs = gold_fallback_pairs(
        examples,
        [_sample(example, example.gold_json)],
        strict_verifier(),
        min_margin=0.05,
        max_pairs_per_prompt=2,
    )
    assert pairs == []


def test_gold_fallback_pairs_respect_the_margin() -> None:
    """A margin nothing can clear yields nothing, rather than a pair with no signal in it."""
    examples = _examples(1)
    pairs = gold_fallback_pairs(
        examples,
        [_sample(examples[0], "rubbish")],
        strict_verifier(),
        min_margin=2.0,
        max_pairs_per_prompt=2,
    )
    assert pairs == []


def test_gold_fallback_pairs_ignore_samples_for_unknown_examples() -> None:
    examples = _examples(2)
    stray = Sample(example_id="not-in-the-corpus", model="tiny", text="junk")
    pairs = gold_fallback_pairs(
        examples, [stray], strict_verifier(), min_margin=0.05, max_pairs_per_prompt=2
    )
    assert pairs == []


# --------------------------------------------------------------------------------------
# Helpers around the run directory
# --------------------------------------------------------------------------------------


def test_stage_record_round_trips_through_json(tmp_path: Path) -> None:
    record = StageRecord(
        stage=Stage.MINE, status="ran", outputs=("pairs.jsonl",), summary={"pairs": 12}
    )
    path = record.to_json(tmp_path / "mine.json")
    assert StageRecord.from_json(path) == record


def test_publish_adapter_refuses_when_the_trainer_saved_nothing(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="saved no adapter"):
        publish_adapter(None, tmp_path / "adapter", Stage.SFT)


def test_publish_adapter_replaces_whatever_was_there(tmp_path: Path) -> None:
    source = tmp_path / "final"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"new")
    destination = tmp_path / "adapter"
    destination.mkdir()
    (destination / "stale.bin").write_bytes(b"old")

    publish_adapter(source, destination, Stage.SFT)
    assert (destination / "adapter_model.safetensors").read_bytes() == b"new"
    assert not (destination / "stale.bin").exists()


def test_a_stage_names_the_stage_that_should_have_produced_its_input(tmp_path: Path) -> None:
    config = smoke_config(tmp_path / "run")
    with pytest.raises(PipelineError, match="run the 'data' stage"):
        run_pipeline(config, provider=SmokeProvider(), stages=[Stage.SFT])


def test_a_missing_provider_defaults_to_the_real_checkpoint_loader(tmp_path: Path) -> None:
    """`_resolve_provider` must not be a mock-only path; the default is the hub loader."""
    config = PipelineConfig(run_dir=tmp_path, dtype="float32", device="cpu", lora_r=3)
    provider = _resolve_provider(config, None)
    assert isinstance(provider, HubProvider)
    assert (provider.dtype, provider.device, provider.lora_r) == ("float32", "cpu", 3)
    assert _resolve_provider(config, SmokeProvider()) != provider


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (Stage.EVAL_BASE, "there is nothing to evaluate"),
        (Stage.MINE, "nothing to sample from"),
    ],
)
def test_a_stage_refuses_an_empty_split(tmp_path: Path, stage: Stage, message: str) -> None:
    config = smoke_config(tmp_path / "run", n_val=0, eval_split="val", mine_split="val")
    build_dataset(seed=config.seed, n_train=2, n_val=0, n_test=2).save(config.layout.data_dir)
    with pytest.raises(PipelineError, match=message):
        run_pipeline(config, provider=SmokeProvider(), stages=[stage])


def test_the_dpo_stage_refuses_an_empty_pair_file(tmp_path: Path) -> None:
    config = smoke_config(tmp_path / "run")
    config.layout.pairs.parent.mkdir(parents=True, exist_ok=True)
    config.layout.pairs.write_text("", encoding="utf-8")
    config.layout.sft_adapter.mkdir(parents=True)
    (config.layout.sft_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PipelineError, match="nothing to optimise"):
        run_pipeline(config, provider=SmokeProvider(), stages=[Stage.DPO])


def test_the_comparison_stage_needs_every_variant(
    tmp_path: Path, smoke_run: PipelineResult
) -> None:
    config = smoke_config(tmp_path / "run")
    config.layout.root.mkdir(parents=True)
    source = smoke_run.layout.eval_report("base")
    config.layout.eval_report("base").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(PipelineError, match="every variant's evaluation report"):
        run_pipeline(config, provider=SmokeProvider(), stages=[Stage.COMPARE])


def test_requesting_stages_out_of_order_still_runs_them_in_order(tmp_path: Path) -> None:
    config = smoke_config(tmp_path / "run")
    result = run_pipeline(config, provider=SmokeProvider(), stages=[Stage.EVAL_BASE, Stage.DATA])
    assert result.stages == (Stage.DATA, Stage.EVAL_BASE)


# --------------------------------------------------------------------------------------
# The end-to-end run
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    """Every stage of the real experiment, on the tiny model, in a few seconds.

    Module-scoped because it is the only expensive thing in this file and because several
    tests need to inspect the same finished run directory.
    """
    root = tmp_path_factory.mktemp("smoke")
    return run_pipeline(smoke_config(root / "run"), provider=SmokeProvider())


def test_the_whole_pipeline_runs_every_stage(smoke_run: PipelineResult) -> None:
    assert smoke_run.stages == STAGE_ORDER
    assert smoke_run.ran == STAGE_ORDER
    assert smoke_run.skipped == ()


def test_every_declared_artefact_appears(smoke_run: PipelineResult) -> None:
    """The assertion that makes this a pipeline test rather than nine unit tests."""
    for stage in STAGE_ORDER:
        for path in smoke_run.layout.outputs(stage):
            assert path.exists(), f"{stage.value} did not produce {path}"
    assert _complete(smoke_run.artefacts)


def test_every_stage_writes_a_record(smoke_run: PipelineResult) -> None:
    for stage in STAGE_ORDER:
        record = StageRecord.from_json(smoke_run.layout.stage_record(stage))
        assert record.stage is stage
        assert record.status == "ran"
        assert record.library_versions["torch"] != "not installed"
        assert smoke_run.record(stage) == record


def test_asking_for_a_stage_that_did_not_run_raises(smoke_run: PipelineResult) -> None:
    partial = PipelineResult(
        config=smoke_run.config, layout=smoke_run.layout, results=smoke_run.results[:1]
    )
    with pytest.raises(KeyError, match="dpo"):
        partial.record(Stage.DPO)


def test_the_corpus_reloads_and_matches_its_manifest(smoke_run: PipelineResult) -> None:
    dataset = Dataset.load(smoke_run.layout.data_dir)
    assert len(dataset.test) == smoke_run.config.n_test
    stats = json.loads(smoke_run.layout.data_stats.read_text(encoding="utf-8"))
    assert stats["content_hash"] == dataset.content_hash()


def test_both_adapters_hold_weights(smoke_run: PipelineResult) -> None:
    for adapter in (smoke_run.layout.sft_adapter, smoke_run.layout.dpo_adapter):
        assert (adapter / "adapter_config.json").is_file()
        assert any(adapter.glob("adapter_model*"))


def test_the_training_manifests_cite_the_corpus(smoke_run: PipelineResult) -> None:
    corpus_hash = Dataset.load(smoke_run.layout.data_dir).content_hash()
    sft = json.loads((smoke_run.layout.sft_dir / "manifest.json").read_text(encoding="utf-8"))
    dpo = json.loads((smoke_run.layout.dpo_dir / "manifest.json").read_text(encoding="utf-8"))
    assert sft["stage"] == "sft"
    assert sft["data_content_hash"] == corpus_hash
    assert dpo["stage"] == "dpo"
    assert dpo["trainable_parameters"] > 0


def test_every_variant_was_evaluated_on_the_same_examples(smoke_run: PipelineResult) -> None:
    reports = [
        EvalReport.model_validate_json(
            smoke_run.layout.eval_report(variant).read_text(encoding="utf-8")
        )
        for variant in EVAL_VARIANTS
    ]
    ids = {report.example_ids for report in reports}
    assert len(ids) == 1, "the promotion gate pairs over example ids, so they must agree"
    assert len(reports[0].outcomes) == smoke_run.config.n_test
    assert len({report.model for report in reports}) == 3


def test_the_samples_and_pairs_are_readable(smoke_run: PipelineResult) -> None:
    lines = smoke_run.layout.samples.read_text(encoding="utf-8").splitlines()
    samples = [Sample.model_validate_json(line) for line in lines]
    assert len(samples) == smoke_run.config.n_train * smoke_run.config.sample_k

    pairs = [
        PreferencePair.model_validate_json(line)
        for line in smoke_run.layout.pairs.read_text(encoding="utf-8").splitlines()
    ]
    assert pairs, "the DPO stage would have had nothing to train on"
    assert all(pair.margin > 0 for pair in pairs)
    assert all(pair.chosen != pair.rejected for pair in pairs)


def test_the_mining_statistics_admit_the_fallback(smoke_run: PipelineResult) -> None:
    """The randomly initialised model produces nothing rankable, and the artefact says so."""
    stats = json.loads(smoke_run.layout.mining_stats.read_text(encoding="utf-8"))
    assert stats["pairs"] == 0
    assert stats["gold_fallback_used"] is True
    assert stats["pairs_written"] > 0


def test_every_comparison_is_written_twice(smoke_run: PipelineResult) -> None:
    decisions = smoke_run.record(Stage.COMPARE).summary["decisions"]
    for name, _, _ in COMPARISONS:
        markdown = smoke_run.layout.comparison(name, "md").read_text(encoding="utf-8")
        payload = json.loads(smoke_run.layout.comparison(name, "json").read_text(encoding="utf-8"))
        assert payload["decision"] == decisions[name]
        assert payload["decision"].upper() in markdown


def test_the_report_covers_every_variant_and_both_trainers(smoke_run: PipelineResult) -> None:
    text = smoke_run.layout.report.read_text(encoding="utf-8")
    for variant in EVAL_VARIANTS:
        assert f"| {variant} |" in text
    assert "Supervised fine-tuning" in text
    assert "Direct preference optimisation" in text
    assert "Preference mining" in text
    assert "Promotion gates" in text


def test_the_run_directory_records_the_settings_that_produced_it(
    smoke_run: PipelineResult,
) -> None:
    """A directory of numbers that cannot say how it was decoded is not evidence.

    The decoding batch sizes matter as much as the seed here: batched generation is not
    bit-identical to single-stream generation on a GPU, so two variants compared at
    different batch sizes were not compared fairly, and nothing else on disk would show it.
    """
    written = json.loads(smoke_run.layout.config.read_text(encoding="utf-8"))
    assert written == json.loads(smoke_run.config.model_dump_json())
    for setting in ("seed", "sample_batch_size", "eval_batch_size", "temperature", "top_p"):
        assert setting in written


def test_resuming_with_changed_settings_rewrites_the_config(tmp_path: Path) -> None:
    """The file describes what produced the artefacts now, not what was asked for first."""
    first = smoke_config(tmp_path / "run")
    run_pipeline(first, provider=SmokeProvider(), stages=[Stage.DATA])
    second = first.model_copy(update={"sample_batch_size": first.sample_batch_size + 5})
    run_pipeline(second, provider=SmokeProvider(), stages=[Stage.DATA])
    written = json.loads(first.layout.config.read_text(encoding="utf-8"))
    assert written["sample_batch_size"] == second.sample_batch_size


def test_resuming_with_changed_settings_rebuilds_rather_than_reusing(tmp_path: Path) -> None:
    """A stage whose artefacts were built under other settings is not reusable.

    Skipping on file presence alone is how a run directory ends up self-contradicting: the
    corpus on disk is still the one seed 0 produced while `config.json`, and every number
    attributed to it downstream, says seed 99. The stage record now carries a fingerprint of
    the configuration that produced it, and a mismatch re-runs the stage.
    """
    first = smoke_config(tmp_path / "run", seed=0, n_test=6)
    run_pipeline(first, provider=SmokeProvider(), stages=[Stage.DATA])
    rows = len((first.layout.data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines())
    assert rows == 6

    second = smoke_config(tmp_path / "run", seed=99, n_test=24)
    result = run_pipeline(second, provider=SmokeProvider(), stages=[Stage.DATA])

    assert result.ran == (Stage.DATA,)
    rebuilt = len((second.layout.data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines())
    assert rebuilt == 24


def test_an_unchanged_configuration_still_skips(tmp_path: Path) -> None:
    """The fingerprint must not defeat resumption, which is the point of the whole design."""
    config = smoke_config(tmp_path / "run")
    run_pipeline(config, provider=SmokeProvider(), stages=[Stage.DATA])
    again = run_pipeline(config, provider=SmokeProvider(), stages=[Stage.DATA])
    assert again.ran == ()
    assert again.skipped == (Stage.DATA,)


def test_the_fingerprint_ignores_where_the_run_is_written(tmp_path: Path) -> None:
    """Copying a finished run elsewhere does not change what produced it."""
    here = smoke_config(tmp_path / "a")
    there = smoke_config(tmp_path / "b")
    assert config_digest(here) == config_digest(there)
    assert config_digest(here) != config_digest(here.model_copy(update={"seed": here.seed + 1}))


def test_a_second_invocation_skips_everything(smoke_run: PipelineResult) -> None:
    again = run_pipeline(smoke_run.config, provider=SmokeProvider())
    assert again.ran == ()
    assert again.skipped == STAGE_ORDER


def test_a_skipped_stage_keeps_the_numbers_of_the_run_that_did_the_work(
    smoke_run: PipelineResult,
) -> None:
    again = run_pipeline(smoke_run.config, provider=SmokeProvider())
    assert again.record(Stage.SFT).summary == smoke_run.record(Stage.SFT).summary


def test_forcing_the_data_stage_reproduces_the_same_corpus(smoke_run: PipelineResult) -> None:
    """Determinism where it can be asserted exactly: the corpus is a pure function of a seed."""
    before = smoke_run.layout.data_stats.read_text(encoding="utf-8")
    run_pipeline(smoke_run.config, provider=SmokeProvider(), stages=[Stage.DATA], force=True)
    assert smoke_run.layout.data_stats.read_text(encoding="utf-8") == before


def test_the_report_is_byte_stable(smoke_run: PipelineResult) -> None:
    assert build_report(smoke_run.layout.root) == build_report(smoke_run.layout.root)
    assert build_report(smoke_run.layout.root) == smoke_run.layout.report.read_text(
        encoding="utf-8"
    )


def test_the_report_does_not_embed_an_absolute_path(smoke_run: PipelineResult) -> None:
    text = build_report(smoke_run.layout.root)
    assert str(smoke_run.layout.root) not in text
    assert smoke_run.layout.root.name in text


def test_alignment_tax_runs_on_the_tiny_model(tmp_path: Path, smoke_run: PipelineResult) -> None:
    config = smoke_config(tmp_path / "tax")
    tax = measure_alignment_tax(
        config,
        provider=SmokeProvider(),
        adapter=smoke_run.layout.dpo_adapter,
        baseline_adapter=None,
    )
    assert tax.n == 12
    assert 0.0 <= tax.before_rate <= 1.0
    assert 0.0 <= tax.after_rate <= 1.0
    assert tax.tax == max(0.0, tax.before_rate - tax.after_rate)
    assert "Alignment tax" in tax.to_markdown()


# --------------------------------------------------------------------------------------
# Decoding replies that have no gold record
# --------------------------------------------------------------------------------------


def test_generate_replies_returns_one_reply_per_instruction() -> None:
    loaded = SmokeProvider().load("tiny")
    replies = generate_replies(
        loaded.model,
        loaded.tokenizer,
        ["say one word", "list two things", "reply with 42"],
        formatter=ChatFormatter(tokenizer=loaded.tokenizer),
        config=GenerationConfig(max_new_tokens=4, temperature=0.0),
        batch_size=2,
    )
    assert len(replies) == 3
    assert all(isinstance(reply, str) for reply in replies)


def test_generate_replies_on_nothing_returns_nothing() -> None:
    loaded = SmokeProvider().load("tiny")
    assert (
        generate_replies(
            loaded.model,
            loaded.tokenizer,
            [],
            formatter=ChatFormatter(tokenizer=loaded.tokenizer),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        )
        == []
    )


# --------------------------------------------------------------------------------------
# The report on partial and broken directories
# --------------------------------------------------------------------------------------


def test_build_report_rejects_a_directory_that_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run directory"):
        build_report(tmp_path / "nowhere")


def test_build_report_on_an_empty_directory_says_it_found_nothing(tmp_path: Path) -> None:
    text = build_report(tmp_path)
    assert "No evaluation reports were found" in text
    assert text.startswith("# sft-dpo-alignment")


def test_build_report_ignores_a_file_that_is_not_a_report(tmp_path: Path) -> None:
    (tmp_path / "eval_broken.json").write_text("{}", encoding="utf-8")
    assert "No evaluation reports were found" in build_report(tmp_path)


def test_build_report_ignores_a_comparison_that_is_not_a_result(tmp_path: Path) -> None:
    (tmp_path / "compare_broken.json").write_text('{"decision": "promote"}', encoding="utf-8")
    assert "Promotion gates" not in build_report(tmp_path)


def test_build_report_orders_extra_variants_after_the_pipelines_own(
    tmp_path: Path, smoke_run: PipelineResult
) -> None:
    """A run may also evaluate a larger prompted model; it belongs after the three stages."""
    for variant in EVAL_VARIANTS:
        source = smoke_run.layout.eval_report(variant)
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    extra = smoke_run.layout.eval_report("base").read_text(encoding="utf-8")
    (tmp_path / "eval_big.json").write_text(extra, encoding="utf-8")

    text = build_report(tmp_path)
    positions = [text.index(f"| {name} |") for name in (*EVAL_VARIANTS, "big")]
    assert positions == sorted(positions)


def test_build_report_tolerates_a_corpus_summary_missing_a_split(tmp_path: Path) -> None:
    """`data_stats.json` may have been written by an older run; a table row is not worth a
    crash."""
    (tmp_path / "data_stats.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "total": 2,
                "content_hash": "a" * 64,
                "splits": {
                    "train": {
                        "n": 2,
                        "field_presence": {"objectives": 1.0, "recommendations": 1.0, "flags": 0.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    text = build_report(tmp_path)
    assert "| train | 2 |" in text
    assert "| val |" not in text


def test_build_report_omits_the_slice_table_when_no_report_has_slices(tmp_path: Path) -> None:
    bare = EvalReport(model="tiny")
    (tmp_path / "eval_base.json").write_text(bare.model_dump_json(), encoding="utf-8")
    text = build_report(tmp_path)
    assert "| base |" in text
    assert "Schema validity per slice" not in text


def test_build_report_renders_one_trainer_without_the_other(tmp_path: Path) -> None:
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft" / "summary.json").write_text('{"steps": 7}', encoding="utf-8")
    text = build_report(tmp_path)
    assert "Supervised fine-tuning" in text
    assert "Direct preference optimisation" not in text
    assert "| steps | 7 |" in text


def test_build_report_reads_a_high_saturation_rate_as_success(tmp_path: Path) -> None:
    """A finished pipeline yields no pairs; the report must not read that as a failure."""
    (tmp_path / "mining_stats.json").write_text(
        json.dumps({"prompts": 40, "pairs": 0, "saturation_rate": 0.9}), encoding="utf-8"
    )
    assert "finished pipeline" in build_report(tmp_path)


def test_build_report_always_ends_in_exactly_one_newline(tmp_path: Path) -> None:
    text = build_report(tmp_path)
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    payload=st.dictionaries(
        st.sampled_from(["steps", "pairs", "mean_margin", "diverged"]),
        st.one_of(st.integers(0, 10**6), st.floats(0, 1), st.booleans()),
        max_size=4,
    )
)
def test_summary_values_render_without_raising(tmp_path: Path, payload: dict[str, Any]) -> None:
    """The report renders whatever a stage summary carries, so it must survive every type."""
    (tmp_path / "mining_stats.json").write_text(json.dumps(payload), encoding="utf-8")
    assert isinstance(build_report(tmp_path), str)


def test_the_preference_stage_has_its_own_learning_rate(tmp_path: Path) -> None:
    """Sharing one rate between the two stages destroyed a real run.

    At the supervised 1e-4 the preference stage drove the rejected completions' implicit
    reward from +0.13 to -7.15 while the chosen reward stayed near zero. Reward accuracy hit
    1.0, the loss fell to 0.0014, the margin grew to 7.5 -- and strict JSON validity went
    0.981 to 0.000, because the policy had walked far enough from the reference to stop
    producing the format at all. DPO is a small correction to a model that already works, and
    its step size has to say so.
    """
    config = PipelineConfig(run_dir=tmp_path, learning_rate=1e-4)
    assert config.sft_train_config().learning_rate == 1e-4
    assert config.dpo_train_config().learning_rate == config.dpo_learning_rate
    assert config.dpo_learning_rate < config.learning_rate


def test_the_preference_learning_rate_is_overridable(tmp_path: Path) -> None:
    """It is a default, not a law: a different corpus wants a different rate."""
    config = PipelineConfig(run_dir=tmp_path, dpo_learning_rate=3e-6)
    assert config.dpo_train_config().learning_rate == 3e-6
