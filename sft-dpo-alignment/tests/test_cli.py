"""Tests for the command line.

The interface being tested is not only what the commands print: it is the exit codes, which
`scripts/run_experiments.sh` and CI both read. So every command is exercised for its status as
well as its output, and `eval compare --gate` -- the one command whose whole job is to fail --
is tested from both sides of the decision.

The four commands that need a model are driven through `main(..., provider=SmokeProvider())`,
which is why they can run here at all: the Qwen tokenizer is not available offline, so the
alternative would be to leave a quarter of the interface untested.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sftdpo.cli import (
    _adapter_label,
    _optional_path,
    _provider,
    _variant_label,
    build_parser,
    main,
)
from sftdpo.eval.metrics import EvalReport, ExampleOutcome, MetricBlock, ParseGap
from sftdpo.pipeline import STAGE_ORDER, HubProvider, SmokeProvider, Stage
from sftdpo.schemas import PreferencePair, Slice
from sftdpo.task.dataset import Dataset, build_dataset
from sftdpo.task.generate import render_prompt

HELP_PATHS = [
    ["--help"],
    ["data", "--help"],
    ["data", "build", "--help"],
    ["data", "stats", "--help"],
    ["verify", "--help"],
    ["verify", "check", "--help"],
    ["sft", "train", "--help"],
    ["prefs", "mine", "--help"],
    ["dpo", "train", "--help"],
    ["eval", "run", "--help"],
    ["eval", "compare", "--help"],
    ["eval", "tax", "--help"],
    ["crosscheck", "--help"],
    ["report", "--help"],
    ["pipeline", "run", "--help"],
    ["pipeline", "smoke", "--help"],
]

INCOMPLETE = [[], ["data"], ["verify"], ["sft"], ["prefs"], ["dpo"], ["eval"], ["pipeline"]]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A six-per-split corpus on disk, shared by every command that needs one."""
    directory = tmp_path_factory.mktemp("corpus") / "data"
    build_dataset(seed=2, n_train=6, n_val=6, n_test=6).save(directory)
    return directory


@pytest.fixture(scope="module")
def pairs_file(tmp_path_factory: pytest.TempPathFactory, corpus: Path) -> Path:
    """Two preference pairs, gold against rubbish, so `dpo train` has something to read."""
    path = tmp_path_factory.mktemp("pairs") / "pairs.jsonl"
    dataset = Dataset.load(corpus)
    rows = [
        PreferencePair(
            example_id=example.example_id,
            slice=example.slice,
            prompt=render_prompt(example.note),
            chosen=example.gold_json,
            rejected="that is not an advice record",
            chosen_reward=1.0,
            rejected_reward=0.0,
        )
        for example in dataset.train[:2]
    ]
    path.write_text(
        "".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8", newline="\n"
    )
    return path


def _report(model: str, successes: list[bool]) -> EvalReport:
    """An evaluation report over a fixed set of examples, with the outcomes dictated.

    Built directly rather than by decoding, because the comparison gate reads only the
    per-example indicators and the aggregate block, and a test of the gate should not depend
    on what a randomly initialised model happened to emit.
    """
    outcomes = tuple(
        ExampleOutcome(
            example_id=f"test-clean-{index:04d}",
            slice=Slice.CLEAN,
            json_valid=ok,
            schema_valid=ok,
            field_f1=1.0 if ok else 0.0,
            exact_match=ok,
            reward=1.0 if ok else 0.0,
            lenient_json_valid=ok,
        )
        for index, ok in enumerate(successes)
    )
    return EvalReport(
        model=model,
        overall=MetricBlock.over(outcomes),
        per_slice={Slice.CLEAN: MetricBlock.over(outcomes)},
        parse_gap=ParseGap.over(outcomes),
        outcomes=outcomes,
    )


def _write_report(path: Path, model: str, successes: list[bool]) -> Path:
    path.write_text(_report(model, successes).model_dump_json(indent=2), encoding="utf-8")
    return path


@pytest.fixture
def completions(tmp_path: Path, corpus: Path) -> Path:
    """A JSONL of completions: one perfect, one wrapped in prose, one hopeless."""
    examples = Dataset.load(corpus).test[:3]
    golds = [example.gold.model_dump(mode="json") for example in examples]
    rows: list[dict[str, object]] = [
        {"example_id": examples[0].example_id, "text": examples[0].gold_json, "gold": golds[0]},
        {
            "example_id": examples[1].example_id,
            "completion": f"Certainly! Here it is: {examples[1].gold_json}",
            "gold": golds[1],
        },
        {
            "example_id": examples[2].example_id,
            "text": "I could not read that note.",
            "gold": golds[2],
        },
    ]
    path = tmp_path / "completions.jsonl"
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# The parser itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", HELP_PATHS, ids=lambda argv: " ".join(argv))
def test_help_exits_zero_everywhere(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert raised.value.code == 0


@pytest.mark.parametrize("argv", INCOMPLETE, ids=lambda argv: " ".join(argv) or "(none)")
def test_a_command_without_its_subcommand_is_a_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert raised.value.code == 2


def test_an_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["definitely-not-a-command"])
    assert raised.value.code == 2


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert "sftdpo" in capsys.readouterr().out


def test_every_command_has_a_handler() -> None:
    """A subcommand with no handler would fail with an AttributeError instead of running."""
    parser = build_parser()
    for argv in (
        ["data", "stats", "somewhere"],
        ["verify", "check", "--file", "f"],
        ["crosscheck"],
        ["report", "somewhere"],
    ):
        assert callable(parser.parse_args(argv).handler)


def test_optional_path_treats_none_as_the_base_model() -> None:
    assert _optional_path(None) is None
    assert _optional_path("none") is None
    assert _optional_path("NONE") is None
    assert _optional_path("  ") is None
    assert _optional_path("runs/sft/adapter") == Path("runs/sft/adapter")


def test_an_adapter_is_labelled_by_the_stage_that_produced_it() -> None:
    assert _adapter_label(Path("runs/main/sft/adapter")) == "sft"
    assert _adapter_label(Path("runs/main/dpo/final")) == "dpo"
    assert _adapter_label(Path("checkpoints/my-experiment")) == "my-experiment"


def test_an_explicit_label_wins_over_the_derived_one() -> None:
    assert _variant_label("Qwen", Path("runs/sft/adapter"), "candidate-b") == "candidate-b"
    assert _variant_label("Qwen", None, None) == "Qwen"


def test_without_an_injected_provider_the_real_loader_is_used() -> None:
    """The provider argument is a test seam, not the only path through the commands."""
    args = build_parser().parse_args(
        ["sft", "train", "--model", "m", "--data", "d", "--out", "o", "--device", "cpu"]
    )
    resolved = _provider(args, None)
    assert isinstance(resolved, HubProvider)
    assert resolved.device == "cpu"
    assert _provider(args, SmokeProvider()) != resolved


def test_the_module_entry_point_wires_to_the_cli() -> None:
    import sftdpo.__main__ as entry

    assert entry.main is main


def test_python_dash_m_sftdpo_runs() -> None:
    """The one thing about `__main__.py` worth asserting: it turns the code into a status.

    Run as a real subprocess because that is the only way the `if __name__` guard is
    executed at all, and a package whose entry point does not start is a package nobody can
    use from the shell.
    """
    finished = subprocess.run(
        [sys.executable, "-m", "sftdpo", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0
    assert "sftdpo" in finished.stdout


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def test_data_build_writes_a_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "data"
    assert (
        main(
            [
                "data",
                "build",
                "--seed",
                "4",
                "--n-train",
                "6",
                "--n-val",
                "6",
                "--n-test",
                "6",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    dataset = Dataset.load(out)
    assert (len(dataset.train), len(dataset.val), len(dataset.test)) == (6, 6, 6)
    assert (out / "stats.json").is_file()
    assert dataset.content_hash() in capsys.readouterr().out


def test_data_build_is_reproducible(tmp_path: Path) -> None:
    """Same seed, same corpus: the claim every experiment record in this project rests on."""
    for name in ("first", "second"):
        assert (
            main(
                [
                    "data",
                    "build",
                    "--seed",
                    "9",
                    "--n-train",
                    "6",
                    "--n-val",
                    "0",
                    "--n-test",
                    "6",
                    "--out",
                    str(tmp_path / name),
                ]
            )
            == 0
        )
    left = (tmp_path / "first" / "manifest.json").read_text(encoding="utf-8")
    right = (tmp_path / "second" / "manifest.json").read_text(encoding="utf-8")
    assert left == right


def test_data_stats_prints_the_per_slice_breakdown(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "stats", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "long_context" in out
    assert "present: objectives" in out


def test_data_stats_can_emit_json(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "stats", str(corpus), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 18
    assert set(payload["splits"]) == {"train", "val", "test"}


def test_data_stats_on_a_missing_corpus_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "stats", str(tmp_path / "nowhere")]) == 1
    assert "sftdpo:" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------


def test_verify_check_prints_a_reward_breakdown(
    completions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["verify", "check", "--file", str(completions)]) == 0
    out = capsys.readouterr().out
    assert "3 completions" in out
    assert "mean reward" in out
    assert "unparseable" in out or "no_json" in out


def test_verify_check_is_never_stricter_when_lenient(
    completions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Lenient parsing accepts everything strict parsing does, so the score cannot fall."""
    assert main(["verify", "check", "--file", str(completions), "--json"]) == 0
    strict = json.loads(capsys.readouterr().out)["summary"]
    assert main(["verify", "check", "--file", str(completions), "--lenient", "--json"]) == 0
    lenient = json.loads(capsys.readouterr().out)["summary"]

    assert lenient["parse_rate"] >= strict["parse_rate"]
    assert lenient["mean_value"] >= strict["mean_value"]
    assert lenient["parse_rate"] > strict["parse_rate"], "the prose-wrapped row should recover"


def test_verify_check_resolves_gold_from_a_corpus(
    tmp_path: Path, corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example = Dataset.load(corpus).test[0]
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"example_id": example.example_id, "text": example.gold_json}) + "\n",
        encoding="utf-8",
    )
    assert main(["verify", "check", "--file", str(path), "--data", str(corpus)]) == 0
    assert "mean reward 1.0000" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"gold": {}}\n', "none of"),
        ('{"text": "x"}\n', "carries no 'gold' object"),
        ("not json at all\n", "not valid JSON"),
        ("[1, 2]\n", "not a JSON object"),
        ("\n\n", "no completions to score"),
    ],
)
def test_verify_check_refuses_a_row_it_cannot_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str, message: str
) -> None:
    """Skipping a bad row would change the denominator of every rate printed beside it."""
    path = tmp_path / "bad.jsonl"
    path.write_text(body, encoding="utf-8")
    assert main(["verify", "check", "--file", str(path)]) == 1
    assert message in capsys.readouterr().err


def test_verify_check_on_a_missing_file_exits_one(tmp_path: Path) -> None:
    assert main(["verify", "check", "--file", str(tmp_path / "absent.jsonl")]) == 1


# --------------------------------------------------------------------------------------
# eval compare: the gate
# --------------------------------------------------------------------------------------


def test_compare_without_gate_always_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The experiment script runs several comparisons under `set -e`; a HOLD is a result."""
    baseline = _write_report(tmp_path / "base.json", "base", [True] * 10)
    candidate = _write_report(tmp_path / "cand.json", "cand", [False] * 10)
    assert main(["eval", "compare", str(baseline), str(candidate)]) == 0
    assert "REJECT" in capsys.readouterr().out


def test_compare_with_gate_exits_non_zero_on_a_regression(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", "base", [True] * 10)
    candidate = _write_report(tmp_path / "cand.json", "cand", [False] * 10)
    assert main(["eval", "compare", str(baseline), str(candidate), "--gate"]) == 1


def test_compare_with_gate_exits_zero_on_a_promotion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_report(tmp_path / "base.json", "base", [False] * 16)
    candidate = _write_report(tmp_path / "cand.json", "cand", [True] * 16)
    assert main(["eval", "compare", str(baseline), str(candidate), "--gate"]) == 0
    assert "PROMOTE" in capsys.readouterr().out


def test_a_floor_blocks_an_improvement_that_is_still_not_deployable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Going from 0.00 to 0.50 is a large improvement and still useless in production."""
    baseline = _write_report(tmp_path / "base.json", "base", [False] * 16)
    candidate = _write_report(tmp_path / "cand.json", "cand", [True] * 8 + [False] * 8)
    argv = ["eval", "compare", str(baseline), str(candidate), "--json-floor", "0.95"]
    assert main([*argv, "--gate"]) == 1
    assert "below the floor" in capsys.readouterr().out


def test_a_floor_outside_the_unit_interval_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_report(tmp_path / "base.json", "base", [True] * 4)
    candidate = _write_report(tmp_path / "cand.json", "cand", [True] * 4)
    assert main(["eval", "compare", str(baseline), str(candidate), "--json-floor", "1.5"]) == 1
    assert "sftdpo:" in capsys.readouterr().err


def test_compare_writes_markdown_and_json(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", "base", [False] * 12)
    candidate = _write_report(tmp_path / "cand.json", "cand", [True] * 12)
    markdown = tmp_path / "compare.md"
    payload = tmp_path / "compare.json"
    assert (
        main(
            [
                "eval",
                "compare",
                str(baseline),
                str(candidate),
                "--out",
                str(markdown),
                "--json-out",
                str(payload),
            ]
        )
        == 0
    )
    assert "Promotion gate" in markdown.read_text(encoding="utf-8")
    assert json.loads(payload.read_text(encoding="utf-8"))["decision"] == "promote"


def test_compare_on_a_missing_report_exits_one(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", "base", [True] * 4)
    assert main(["eval", "compare", str(baseline), str(tmp_path / "absent.json")]) == 1


# --------------------------------------------------------------------------------------
# crosscheck
# --------------------------------------------------------------------------------------


def test_crosscheck_prints_the_maximum_absolute_difference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["crosscheck", "--tolerance", "1e-6", "--pairs", "4"]) == 0
    out = capsys.readouterr().out
    assert "maximum absolute difference" in out
    assert "sigmoid" in out


def test_crosscheck_can_emit_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["crosscheck", "--pairs", "2", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert [row["variant"] for row in results] == ["sigmoid", "ipo", "cdpo"]
    assert all(
        row["status"] in {"agree", "explained", "disagree", "unavailable"} for row in results
    )


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def test_report_prints_and_writes_the_same_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_report(tmp_path / "eval_base.json", "base", [False] * 4)
    out = tmp_path / "report.md"
    assert main(["report", str(tmp_path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert out.read_text(encoding="utf-8") in printed
    assert "| base |" in printed


def test_report_on_a_missing_directory_exits_one(tmp_path: Path) -> None:
    assert main(["report", str(tmp_path / "nowhere")]) == 1


# --------------------------------------------------------------------------------------
# The commands that need a model
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def provider() -> SmokeProvider:
    return SmokeProvider()


def test_sft_train_writes_logs_and_an_adapter(
    tmp_path: Path, corpus: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "sft"
    code = main(
        [
            "sft",
            "train",
            "--model",
            "tiny",
            "--data",
            str(corpus),
            "--out",
            str(out),
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--lora-r",
            "4",
            "--lora-alpha",
            "8",
        ],
        provider=provider,
    )
    assert code == 0
    assert (out / "summary.json").is_file()
    assert (out / "train_log.jsonl").is_file()
    assert (out / "adapter" / "adapter_config.json").is_file()
    printed = capsys.readouterr().out
    assert "sft: 2 steps" in printed
    assert "adapter written to" in printed


def test_sft_train_without_validation_data_still_saves(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    """No validation split means no best checkpoint, and the final weights are still the run."""
    directory = tmp_path / "data"
    build_dataset(seed=1, n_train=4, n_val=0, n_test=2).save(directory)
    out = tmp_path / "sft"
    code = main(
        [
            "sft",
            "train",
            "--model",
            "tiny",
            "--data",
            str(directory),
            "--out",
            str(out),
            "--max-steps",
            "1",
            "--batch-size",
            "2",
            "--log-every",
            "1",
            "--lora-r",
            "4",
            "--lora-alpha",
            "8",
        ],
        provider=provider,
    )
    assert code == 0
    assert (out / "adapter" / "adapter_config.json").is_file()
    printed = capsys.readouterr().out
    assert "best validation loss" not in printed
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["best_val_loss"] is None


def test_prefs_mine_writes_pairs_samples_and_statistics(
    tmp_path: Path, corpus: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "pairs.jsonl"
    samples = tmp_path / "samples.jsonl"
    stats = tmp_path / "mining.json"
    code = main(
        [
            "prefs",
            "mine",
            "--model",
            "tiny",
            "--data",
            str(corpus),
            "--split",
            "train",
            "--k",
            "2",
            "--out",
            str(out),
            "--samples-out",
            str(samples),
            "--stats-out",
            str(stats),
            "--max-new-tokens",
            "4",
            "--batch-size",
            "3",
        ],
        provider=provider,
    )
    assert code == 0
    assert len(samples.read_text(encoding="utf-8").splitlines()) == 12
    assert json.loads(stats.read_text(encoding="utf-8"))["prompts"] == 6
    assert "sampled 12 completions" in capsys.readouterr().out


def test_prefs_mine_says_so_when_it_finds_nothing(
    tmp_path: Path, corpus: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    """A random policy scores every completion the same, which is correctly no pairs at all."""
    out = tmp_path / "pairs.jsonl"
    assert (
        main(
            [
                "prefs",
                "mine",
                "--model",
                "tiny",
                "--data",
                str(corpus),
                "--k",
                "2",
                "--out",
                str(out),
                "--max-new-tokens",
                "4",
                "--batch-size",
                "3",
            ],
            provider=provider,
        )
        == 0
    )
    assert out.read_text(encoding="utf-8") == ""
    assert "no pairs" in capsys.readouterr().err


def test_prefs_mine_on_an_empty_split_exits_one(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "data"
    build_dataset(seed=1, n_train=2, n_val=0, n_test=2).save(directory)
    code = main(
        [
            "prefs",
            "mine",
            "--model",
            "tiny",
            "--data",
            str(directory),
            "--split",
            "val",
            "--k",
            "2",
            "--out",
            str(tmp_path / "pairs.jsonl"),
        ],
        provider=provider,
    )
    assert code == 1
    assert "'val' split is empty" in capsys.readouterr().err


def test_dpo_train_writes_logs_and_an_adapter(
    tmp_path: Path, pairs_file: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "dpo"
    code = main(
        [
            "dpo",
            "train",
            "--model",
            "tiny",
            "--adapter",
            "none",
            "--pairs",
            str(pairs_file),
            "--out",
            str(out),
            "--beta",
            "0.2",
            "--variant",
            "ipo",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--log-every",
            "1",
            "--lora-r",
            "4",
            "--lora-alpha",
            "8",
        ],
        provider=provider,
    )
    assert code == 0
    assert (out / "reward_log.jsonl").is_file()
    assert (out / "adapter" / "adapter_config.json").is_file()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["variant"] == "ipo"
    assert summary["beta"] == 0.2
    assert "dpo (ipo, beta 0.2)" in capsys.readouterr().out


def test_dpo_train_on_an_empty_pairs_file_exits_one(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "pairs.jsonl"
    empty.write_text("", encoding="utf-8")
    code = main(
        ["dpo", "train", "--model", "tiny", "--pairs", str(empty), "--out", str(tmp_path / "d")],
        provider=provider,
    )
    assert code == 1
    assert "no preference pairs" in capsys.readouterr().err


def test_eval_run_writes_a_report(
    tmp_path: Path, corpus: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "eval.json"
    code = main(
        [
            "eval",
            "run",
            "--model",
            "tiny",
            "--data",
            str(corpus),
            "--split",
            "test",
            "--out",
            str(out),
            "--batch-size",
            "3",
            "--max-new-tokens",
            "4",
        ],
        provider=provider,
    )
    assert code == 0
    report = EvalReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.n == 6
    assert report.model == "tiny"
    assert "parse gap" in capsys.readouterr().out


def test_eval_run_labels_the_variant_by_its_adapter(
    tmp_path: Path, corpus: Path, provider: SmokeProvider
) -> None:
    trained = provider.load("tiny", trainable=True)
    trained.model.save_pretrained(str(tmp_path / "sft" / "adapter"))
    out = tmp_path / "eval.json"
    assert (
        main(
            [
                "eval",
                "run",
                "--model",
                "tiny",
                "--adapter",
                str(tmp_path / "sft" / "adapter"),
                "--data",
                str(corpus),
                "--out",
                str(out),
                "--batch-size",
                "3",
                "--max-new-tokens",
                "4",
            ],
            provider=provider,
        )
        == 0
    )
    assert EvalReport.model_validate_json(out.read_text(encoding="utf-8")).model == "tiny+sft"


def test_eval_run_on_an_empty_split_exits_one(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "data"
    build_dataset(seed=1, n_train=2, n_val=0, n_test=2).save(directory)
    code = main(
        [
            "eval",
            "run",
            "--model",
            "tiny",
            "--data",
            str(directory),
            "--split",
            "val",
            "--out",
            str(tmp_path / "eval.json"),
        ],
        provider=provider,
    )
    assert code == 1
    assert "nothing to evaluate" in capsys.readouterr().err


def test_eval_tax_writes_the_paired_probe_comparison(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tax.json"
    code = main(
        [
            "eval",
            "tax",
            "--model",
            "tiny",
            "--adapter",
            "none",
            "--baseline-adapter",
            "none",
            "--out",
            str(out),
            "--max-new-tokens",
            "4",
            "--batch-size",
            "4",
        ],
        provider=provider,
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n"] == 12
    assert "Alignment tax" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------


def test_pipeline_smoke_runs_every_stage(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    assert main(["pipeline", "smoke", "--out", str(out)], provider=provider) == 0
    printed = capsys.readouterr().out
    assert f"{len(STAGE_ORDER)} stages ran" in printed
    assert (out / "report.md").is_file()


def test_pipeline_only_runs_the_named_stages(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code = main(
        ["pipeline", "smoke", "--out", str(out), "--only", Stage.DATA.value],
        provider=provider,
    )
    assert code == 0
    assert "1 stages ran, 0 skipped" in capsys.readouterr().out
    assert (out / "data" / "manifest.json").is_file()
    assert not (out / "report.md").exists()


def test_pipeline_skips_and_then_forces(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    argv = ["pipeline", "smoke", "--out", str(out), "--only", Stage.DATA.value]
    assert main(argv, provider=provider) == 0
    capsys.readouterr()

    assert main(argv, provider=provider) == 0
    assert "0 stages ran, 1 skipped" in capsys.readouterr().out

    assert main([*argv, "--force"], provider=provider) == 0
    assert "1 stages ran, 0 skipped" in capsys.readouterr().out


def test_pipeline_rejects_an_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["pipeline", "smoke", "--out", str(tmp_path), "--only", "not-a-stage"])
    assert raised.value.code == 2


def test_pipeline_run_reports_a_missing_upstream_stage(
    tmp_path: Path, provider: SmokeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "pipeline",
            "run",
            "--out",
            str(tmp_path / "run"),
            "--model",
            "tiny",
            "--only",
            Stage.SFT.value,
        ],
        provider=provider,
    )
    assert code == 1
    assert "run the 'data' stage" in capsys.readouterr().err
