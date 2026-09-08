"""The command line: every stage of the experiment, runnable on its own.

`argparse` rather than a dependency, because the whole point of this project is that the
interesting parts are written out. A CLI framework would be one more library between a reader
and the argument that changes the result.

Three conventions run through all of it.

*Exit codes are part of the interface.* 0 means the command did what it was asked; 1 means it
could not, or -- for `eval compare --gate` and `crosscheck` -- that what it found is a
failure; 2 is argparse's own usage error and is left alone. CI reads those codes, so they are
tested rather than assumed.

*Nothing prints what it did not compute.* Every command that produces a file also prints a
short summary to standard output, so `| tee` in `scripts/run_experiments.sh` captures a
readable record and the file stays machine-readable.

*Models arrive through a provider.* `main` takes an optional `ModelProvider`, which is how the
tests exercise `sft train`, `prefs mine`, `dpo train` and `eval run` without a checkpoint or a
tokenizer on disk. A console-script invocation passes nothing and gets `HubProvider`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from sftdpo import __version__
from sftdpo.eval.compare import ComparisonResult, Floors, compare, gate
from sftdpo.eval.generate import generate_samples
from sftdpo.eval.metrics import EvalReport, evaluate
from sftdpo.modeling.chat import ChatFormatter
from sftdpo.modeling.collate import DEFAULT_MAX_SEQ_LENGTH
from sftdpo.pipeline import (
    DEFAULT_MODEL,
    STAGE_ORDER,
    HubProvider,
    ModelProvider,
    PipelineConfig,
    PipelineError,
    SmokeProvider,
    Stage,
    build_report,
    measure_alignment_tax,
    publish_adapter,
    run_pipeline,
    smoke_config,
)
from sftdpo.prefs.pairs import mine_pairs
from sftdpo.prefs.sample import prompts_for_examples, sample_completions
from sftdpo.schemas import AdviceRecord, GenerationConfig, PreferencePair, Reward, Split
from sftdpo.task.dataset import Dataset, build_dataset
from sftdpo.train.common import TrainConfig
from sftdpo.train.crosscheck import CrossCheckBatch, crosscheck_all
from sftdpo.train.dpo import train_dpo
from sftdpo.train.sft import train_sft
from sftdpo.verify.reward import RewardBreakdown, Verifier, lenient_verifier, strict_verifier

__all__ = ["build_parser", "main"]

logger = logging.getLogger(__name__)

Handler = Callable[[argparse.Namespace, ModelProvider | None], int]

SPLITS: tuple[Split, ...] = ("train", "val", "test")
VARIANTS = ("sigmoid", "ipo", "cdpo")

_COMPLETION_KEYS = ("text", "completion", "output", "response")
"""Keys `verify check` will read a completion from, in order of preference.

Several because the file being scored is usually not one this package wrote: it is a dump
from whatever produced the completions, and refusing to read `"completion"` because the
package happens to call it `"text"` would make the command useless exactly when it is most
wanted.
"""


# --------------------------------------------------------------------------------------
# Argument helpers
# --------------------------------------------------------------------------------------


def _optional_path(value: str | None) -> Path | None:
    """Read a path argument that may be deliberately absent.

    `--adapter none` means the base model. The shell script passes it that way because an
    empty string is easy to lose in a variable expansion, and a literal `none` says what it
    means in the command that ends up in a log.
    """
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "none":
        return None
    return Path(text)


def _adapter_label(adapter: Path) -> str:
    """A short name for an adapter directory, for labelling an evaluation report.

    A run's adapters live at `<run>/sft/adapter` and `<run>/dpo/adapter`, so the leaf name is
    the same for both and the parent is what distinguishes them. Anything else is named by
    its own directory.
    """
    if adapter.name in {"adapter", "best", "final"} and adapter.parent.name:
        return adapter.parent.name
    return adapter.name


def _variant_label(model: str, adapter: Path | None, given: str | None) -> str:
    if given is not None:
        return given
    return model if adapter is None else f"{model}+{_adapter_label(adapter)}"


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None, help="cpu, cuda or auto (default: auto)")
    parser.add_argument("--dtype", default="bfloat16", help="weight dtype (default: bfloat16)")


def _add_lora_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.0, help="LoRA dropout (default: 0.0)"
    )


def _add_train_options(
    parser: argparse.ArgumentParser,
    *,
    lr: float = 1e-4,
    batch_size: int = 4,
    grad_accum: int = 1,
) -> None:
    """The options both trainers share, with defaults the caller sets per stage.

    The defaults are parameters rather than constants because the two stages must not share
    them. A preference step puts four sequences through the model where a supervised step
    puts one, and 1e-4 is an ordinary supervised rate and a destructive preference one: the
    first real DPO run inherited it and collapsed. `PipelineConfig` already keeps the two
    apart; the standalone `dpo train` command has to as well, or a reader following the
    README's per-stage commands reproduces the collapse.
    """
    parser.add_argument("--epochs", type=int, default=1, help="passes over the data (default: 1)")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="hard cap on optimiser steps; overrides epochs"
    )
    parser.add_argument("--lr", type=float, default=lr, help=f"peak learning rate (default: {lr})")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help=f"examples per forward pass (default: {batch_size})",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=grad_accum,
        help=f"micro-batches per optimiser step (default: {grad_accum})",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="token budget per example",
    )
    parser.add_argument("--log-every", type=int, default=10, help="logging interval in steps")
    parser.add_argument(
        "--eval-every", type=int, default=50, help="validation interval in steps; 0 disables"
    )
    parser.add_argument("--seed", type=int, default=0, help="run seed")
    parser.add_argument("--bf16", action="store_true", help="run under bf16 autocast")


def _train_config(args: argparse.Namespace, output_dir: Path) -> TrainConfig:
    """Build the frozen training configuration from parsed arguments."""
    return TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        seed=args.seed,
        eval_every_steps=args.eval_every,
        log_every_steps=args.log_every,
        max_seq_length=args.max_seq_length,
        save_best_adapter=True,
        output_dir=output_dir,
    )


def _provider(args: argparse.Namespace, injected: ModelProvider | None) -> ModelProvider:
    """The injected provider if a caller supplied one, otherwise the real checkpoint loader."""
    if injected is not None:
        return injected
    return HubProvider(
        dtype=getattr(args, "dtype", "bfloat16"),
        device=getattr(args, "device", None),
        lora_r=getattr(args, "lora_r", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0.0),
    )


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, payload: Any) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _rate(value: float) -> str:
    return f"{0.0 if value == 0 else value:.4f}"


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def _cmd_data_build(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Build the corpus and write it, with its statistics beside it."""
    del provider
    dataset = build_dataset(
        seed=args.seed, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test
    )
    out = Path(args.out)
    dataset.save(out)
    stats = dataset.stats()
    _write_json(out / "stats.json", stats.model_dump(mode="json"))
    print(f"wrote {stats.total} examples to {out}")
    print(f"content hash {stats.content_hash}")
    for name in SPLITS:
        print(f"  {name}: {len(dataset.examples_for(name))}")
    return 0


def _cmd_data_stats(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Print descriptive statistics for a corpus already on disk."""
    del provider
    stats = Dataset.load(Path(args.directory)).stats()
    if args.json:
        print(json.dumps(stats.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    print(f"seed {stats.seed}, {stats.total} examples, content hash {stats.content_hash}")
    for name in SPLITS:
        block = stats.splits[name]
        words = block.note_words
        length = "empty" if words is None else f"{words.p50:.0f} words median"
        print(f"\n{name}: n = {block.n} ({length})")
        for slice_name, count in sorted(block.per_slice.items()):
            print(f"  {slice_name:<16} {count}")
        for field_name, rate in sorted(block.field_presence.items()):
            print(f"  present: {field_name:<24} {_rate(rate)}")
    return 0


# --------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------


def _completion_of(row: dict[str, Any], line_number: int) -> str:
    for key in _COMPLETION_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(
        f"line {line_number} has none of {list(_COMPLETION_KEYS)}; there is no completion to score"
    )


def _gold_of(row: dict[str, Any], line_number: int, golds: dict[str, AdviceRecord]) -> AdviceRecord:
    raw = row.get("gold")
    if isinstance(raw, dict):
        return AdviceRecord.model_validate(raw)
    example_id = row.get("example_id")
    if isinstance(example_id, str) and example_id in golds:
        return golds[example_id]
    raise ValueError(
        f"line {line_number} carries no 'gold' object, and its example_id "
        f"{example_id!r} is not in the corpus supplied with --data"
    )


def _score_file(
    path: Path, verifier: Verifier, golds: dict[str, AdviceRecord]
) -> list[tuple[str, Reward]]:
    """Score every completion in a JSONL file against its gold record.

    Raises:
        ValueError: On a line that is not a JSON object, a line with no completion, or a line
            whose gold record can be found neither inline nor in the supplied corpus. Skipping
            such a line would quietly change the denominator of every rate printed below it.
    """
    scored: list[tuple[str, Reward]] = []
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number} of {path} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {number} of {path} is not a JSON object")
        identifier = str(row.get("example_id", f"line-{number}"))
        # Completion first: a row with neither a completion nor a gold record should be
        # reported for the thing it was supposed to carry, not for a validation failure on
        # the reference it was going to be scored against.
        completion = _completion_of(row, number)
        scored.append((identifier, verifier.score(completion, _gold_of(row, number, golds))))
    if not scored:
        raise ValueError(f"{path} holds no completions to score")
    return scored


def _cmd_verify_check(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Score a JSONL of completions and print the reward breakdown.

    The most useful command in the package for its size: it needs no GPU and no model, so the
    reward function that supplies every preference label in this project can be inspected on
    real completions in a second.
    """
    del provider
    golds: dict[str, AdviceRecord] = {}
    data = _optional_path(args.data)
    if data is not None:
        golds = {example.example_id: example.gold for example in Dataset.load(data).all_examples}
    verifier = lenient_verifier() if args.lenient else strict_verifier()
    scored = _score_file(Path(args.file), verifier, golds)
    breakdown = RewardBreakdown.over(reward for _, reward in scored)

    if args.json:
        print(
            json.dumps(
                {
                    "strict": not args.lenient,
                    "rows": [
                        {"example_id": name, **reward.model_dump(mode="json")}
                        for name, reward in scored
                    ],
                    "summary": breakdown.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"{'example':<28} {'parse':>5} {'schema':>6} {'F1':>6} {'exact':>5} {'reward':>7}  cause")
    for name, reward in scored:
        cause = reward.headline_violation.value if reward.headline_violation else "-"
        print(
            f"{name[:28]:<28} {_yes(reward.parsed):>5} {_yes(reward.schema_valid):>6} "
            f"{reward.field_f1:>6.3f} {_yes(reward.exact_match):>5} {reward.value:>7.4f}  {cause}"
        )
    print(
        f"\n{breakdown.count} completions, "
        f"parse {_rate(breakdown.parse_rate)}, schema {_rate(breakdown.schema_valid_rate)}, "
        f"field F1 {_rate(breakdown.mean_field_f1)}, exact {_rate(breakdown.exact_match_rate)}, "
        f"mean reward {_rate(breakdown.mean_value)}"
    )
    for kind, count in breakdown.violations.items():
        print(f"  {kind.value:<16} {count}")
    return 0


def _yes(value: bool) -> str:
    return "yes" if value else "no"


# --------------------------------------------------------------------------------------
# sft
# --------------------------------------------------------------------------------------


def _cmd_sft_train(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Fine-tune a fresh LoRA adapter on the gold records."""
    dataset = Dataset.load(Path(args.data))
    out = Path(args.out)
    loaded = _provider(args, provider).load(args.model, adapter=None, trainable=True)
    result = train_sft(
        loaded.model,
        loaded.tokenizer,
        dataset.train,
        dataset.val,
        _train_config(args, out),
        model_name=args.model,
        data_content_hash=dataset.content_hash(),
    )
    result.save(out)
    adapter = publish_adapter(result.adapter_path, out / "adapter", Stage.SFT)
    print(
        f"sft: {result.steps} steps, {result.supervised_tokens} supervised tokens, "
        f"final loss {result.final_train_loss:.4f}"
    )
    if result.best_val_loss is not None:
        print(f"best validation loss {result.best_val_loss:.4f} at step {result.best_step}")
    print(
        f"trainable {result.manifest.trainable_parameters:,} "
        f"({result.manifest.trainable_pct:.3f} %)"
    )
    print(f"adapter written to {adapter}")
    return 0


# --------------------------------------------------------------------------------------
# prefs
# --------------------------------------------------------------------------------------


def _cmd_prefs_mine(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Sample k completions per prompt and mine preference pairs from them."""
    dataset = Dataset.load(Path(args.data))
    examples = dataset.examples_for(args.split)
    if not examples:
        raise PipelineError(f"the {args.split!r} split is empty; nothing to sample from")

    adapter = _optional_path(args.adapter)
    loaded = _provider(args, provider).load(args.model, adapter=adapter)
    formatter = ChatFormatter(tokenizer=loaded.tokenizer)
    prompts = prompts_for_examples(examples, formatter=formatter, tokenizer=loaded.tokenizer)
    samples = sample_completions(
        loaded.model,
        loaded.tokenizer,
        prompts,
        k=args.k,
        config=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        model_name=_variant_label(args.model, adapter, None),
    )
    if args.samples_out:
        _write_text(
            Path(args.samples_out),
            "".join(f"{sample.model_dump_json()}\n" for sample in samples),
        )

    result = mine_pairs(
        samples,
        examples,
        strict_verifier(),
        min_margin=args.min_margin,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        seed=args.seed,
    )
    out = Path(args.out)
    _write_text(out, "".join(f"{pair.model_dump_json()}\n" for pair in result.pairs))
    stats = result.stats()
    if args.stats_out:
        _write_json(Path(args.stats_out), stats.as_dict())

    print(f"sampled {len(samples)} completions for {stats.prompts} prompts at k={args.k}")
    print(f"mined {stats.pairs} pairs to {out} (mean margin {_rate(stats.mean_margin)})")
    print(
        f"prompt yield {_rate(stats.prompt_yield)}, saturated {stats.saturated_prompts}, "
        f"flat {stats.flat_prompts}, dropped by cap {stats.pairs_dropped_by_cap}"
    )
    for name, yielded in stats.slices.items():
        print(f"  {name.value:<16} {yielded.pairs:>4} pairs from {yielded.prompts:>4} prompts")
    if not stats.pairs:
        print(
            "no pairs: raise --k or --temperature, or accept that the policy has saturated "
            "this data",
            file=sys.stderr,
        )
    return 0


# --------------------------------------------------------------------------------------
# dpo
# --------------------------------------------------------------------------------------


def _read_pairs(path: Path) -> list[PreferencePair]:
    text = path.read_text(encoding="utf-8")
    pairs = [PreferencePair.model_validate_json(line) for line in text.splitlines() if line.strip()]
    if not pairs:
        raise PipelineError(f"{path} holds no preference pairs; there is nothing to optimise")
    return pairs


def _cmd_dpo_train(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Run DPO on mined pairs, continuing from a supervised adapter."""
    pairs = _read_pairs(Path(args.pairs))
    adapter = _optional_path(args.adapter)
    out = Path(args.out)
    loaded = _provider(args, provider).load(args.model, adapter=adapter, trainable=True)
    result = train_dpo(
        loaded.model,
        loaded.tokenizer,
        pairs,
        _train_config(args, out),
        beta=args.beta,
        variant=args.variant,
        label_smoothing=args.label_smoothing,
        length_normalise=args.length_normalise,
        model_name=_variant_label(args.model, adapter, None),
    )
    result.save(out)
    written = publish_adapter(result.adapter_path, out / "adapter", Stage.DPO)
    print(
        f"dpo ({args.variant}, beta {args.beta}): {result.steps} steps over "
        f"{result.pairs_seen} pairs, final loss {result.final_loss:.4f}"
    )
    print(
        f"reward accuracy {result.initial_accuracy:.3f} -> {result.final_accuracy:.3f}, "
        f"margin slope {result.margin_slope:+.5f} per step"
    )
    print(f"adapter written to {written}")
    return 0


# --------------------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------------------


def _cmd_eval_run(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Decode one greedy completion per example and score the split."""
    dataset = Dataset.load(Path(args.data))
    examples = dataset.examples_for(args.split)
    if not examples:
        raise PipelineError(f"the {args.split!r} split is empty; there is nothing to evaluate")

    adapter = _optional_path(args.adapter)
    label = _variant_label(args.model, adapter, args.label)
    loaded = _provider(args, provider).load(args.model, adapter=adapter)
    samples = generate_samples(
        loaded.model,
        loaded.tokenizer,
        examples,
        model_name=label,
        formatter=ChatFormatter(tokenizer=loaded.tokenizer),
        config=GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0),
        batch_size=args.batch_size,
    )
    report = evaluate(samples, examples, model=label)
    out = Path(args.out)
    _write_text(out, report.model_dump_json(indent=2))
    block = report.overall
    print(f"{label} on {args.split} (n = {block.n}) -> {out}")
    print(
        f"  json {_rate(block.json_valid_rate)}  schema {_rate(block.schema_valid_rate)}  "
        f"field F1 {_rate(block.mean_field_f1)}  exact {_rate(block.exact_match_rate)}  "
        f"reward {_rate(block.mean_reward)}"
    )
    print(
        f"  parse gap {_rate(report.parse_gap.gap)} "
        f"({report.parse_gap.repaired} completions needed a repair)"
    )
    return 0


def _load_report(path: Path) -> EvalReport:
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def _cmd_eval_compare(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Apply the promotion gate to two evaluation reports.

    The gate's exit code is opt-in through `--gate`. Without it the command always succeeds,
    because the experiment script runs several comparisons in a row under `set -e` and a HOLD
    on one of them is a result to record rather than a reason to abandon the run.
    """
    del provider
    result = _comparison_from_args(args)
    if args.out:
        _write_text(Path(args.out), result.to_markdown())
    if args.json_out:
        _write_text(Path(args.json_out), result.model_dump_json(indent=2))
    gate(result)
    return result.exit_code if args.gate else 0


def _comparison_from_args(args: argparse.Namespace) -> ComparisonResult:
    """Read both reports and run the comparison with the arguments given."""
    return compare(
        _load_report(Path(args.baseline)),
        _load_report(Path(args.candidate)),
        margin=args.margin,
        floors=Floors(
            min_json_valid=args.json_floor,
            min_schema_valid=args.schema_floor,
            max_slice_regression=args.slice_tolerance,
            max_field_recall_drop=args.field_tolerance,
            min_field_support=args.field_support,
        ),
        alpha=args.alpha,
        n_resamples=args.resamples,
        seed=args.seed,
    )


def _cmd_eval_tax(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Measure what alignment cost outside the task it was trained on."""
    config = PipelineConfig(
        run_dir=Path(args.out).parent,
        model=args.model,
        seed=args.seed,
        eval_max_new_tokens=args.max_new_tokens,
        eval_batch_size=args.batch_size,
        dtype=args.dtype,
        device=args.device,
    )
    tax = measure_alignment_tax(
        config,
        provider=provider,
        adapter=_optional_path(args.adapter),
        baseline_adapter=_optional_path(args.baseline_adapter),
        margin=args.margin,
    )
    _write_json(Path(args.out), tax.model_dump(mode="json"))
    print(tax.to_markdown(), end="")
    return 0


# --------------------------------------------------------------------------------------
# crosscheck
# --------------------------------------------------------------------------------------


def _cmd_crosscheck(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Compare this package's DPO loss against the official TRL implementation.

    Exits non-zero on a `disagree`, and only on that. `explained` is a documented difference
    of convention rather than a defect, and `unavailable` means `trl` is not installed and
    nothing was compared -- which is reported as such rather than dressed up as agreement.
    """
    del provider
    batch = CrossCheckBatch.synthetic(size=args.pairs, seed=args.seed, tokens=args.tokens)
    results = crosscheck_all(
        batch, beta=args.beta, label_smoothing=args.label_smoothing, tolerance=args.tolerance
    )
    if args.json:
        print(json.dumps([result.as_dict() for result in results], indent=2, sort_keys=True))
    else:
        for result in results:
            print(result)
            if result.note:
                print(f"  note: {result.note}")
        available = [result for result in results if result.available]
        if available:
            worst = max(result.max_difference for result in available)
            print(f"\nmaximum absolute difference: {worst:.3e} (tolerance {args.tolerance:.1e})")
        else:
            print("\nnothing was compared: trl is not installed")
    return 1 if any(result.status == "disagree" for result in results) else 0


# --------------------------------------------------------------------------------------
# report and pipeline
# --------------------------------------------------------------------------------------


def _cmd_report(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Render a run directory as one Markdown document."""
    del provider
    text = build_report(Path(args.directory))
    if args.out:
        _write_text(Path(args.out), text)
    print(text, end="")
    return 0


def _pipeline_stages(names: Sequence[str] | None) -> tuple[Stage, ...] | None:
    if not names:
        return None
    return tuple(Stage(name) for name in names)


def _cmd_pipeline_run(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Run the whole experiment, skipping stages whose artefacts are already present."""
    config = PipelineConfig(
        run_dir=Path(args.out),
        model=args.model,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        sft_epochs=args.sft_epochs,
        dpo_epochs=args.dpo_epochs,
        batch_size=args.batch_size,
        dpo_batch_size=args.dpo_batch_size,
        dpo_gradient_accumulation_steps=args.dpo_grad_accum,
        dpo_learning_rate=args.dpo_lr,
        max_seq_length=args.max_seq_length,
        sample_k=args.k,
        temperature=args.temperature,
        beta=args.beta,
        variant=args.variant,
        margin=args.margin,
        json_floor=args.json_floor,
        dtype=args.dtype,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    return _report_pipeline(config, provider, args)


def _cmd_pipeline_smoke(args: argparse.Namespace, provider: ModelProvider | None) -> int:
    """Run every stage on the tiny model: seconds, on CPU, with nothing downloaded."""
    config = smoke_config(Path(args.out))
    return _report_pipeline(config, provider if provider is not None else SmokeProvider(), args)


def _report_pipeline(
    config: PipelineConfig, provider: ModelProvider | None, args: argparse.Namespace
) -> int:
    result = run_pipeline(
        config,
        provider=provider,
        stages=_pipeline_stages(args.only),
        force=args.force,
    )
    for stage_result in result.results:
        marker = "skipped" if stage_result.skipped else "ran"
        print(f"{stage_result.stage.value:<10} {marker}")
    print(f"\n{len(result.ran)} stages ran, {len(result.skipped)} skipped")
    print(f"artefacts under {config.layout.root}")
    return 0


# --------------------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Assemble the whole command tree.

    Built by a function rather than at import time so that `--help` text, defaults and
    choices can be asserted by a test without running anything.
    """
    parser = argparse.ArgumentParser(
        prog="sftdpo",
        description=(
            "Supervised fine-tuning then DPO of a small instruct model on a schema-constrained "
            "extraction task, with a deterministic verifier supplying the preference label."
        ),
    )
    parser.add_argument("--version", action="version", version=f"sftdpo {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    _build_data(commands)
    _build_verify(commands)
    _build_sft(commands)
    _build_prefs(commands)
    _build_dpo(commands)
    _build_eval(commands)
    _build_crosscheck(commands)
    _build_report(commands)
    _build_pipeline(commands)
    return parser


_SubParsers = Any


def _build_data(commands: _SubParsers) -> None:
    data = commands.add_parser("data", help="build and inspect the corpus")
    sub = data.add_subparsers(dest="data_command", required=True, metavar="SUBCOMMAND")

    build = sub.add_parser("build", help="generate a stratified corpus with disjoint splits")
    build.add_argument("--seed", type=int, default=1, help="root seed (default: 1)")
    build.add_argument("--n-train", type=int, default=96, help="training examples")
    build.add_argument("--n-val", type=int, default=24, help="validation examples")
    build.add_argument("--n-test", type=int, default=48, help="test examples")
    build.add_argument("--out", default="data", help="output directory (default: data)")
    build.set_defaults(handler=_cmd_data_build)

    stats = sub.add_parser("stats", help="describe a corpus already on disk")
    stats.add_argument("directory", help="a directory written by 'data build'")
    stats.add_argument("--json", action="store_true", help="emit the statistics as JSON")
    stats.set_defaults(handler=_cmd_data_stats)


def _build_verify(commands: _SubParsers) -> None:
    verify = commands.add_parser("verify", help="score completions with the reward function")
    sub = verify.add_subparsers(dest="verify_command", required=True, metavar="SUBCOMMAND")

    check = sub.add_parser(
        "check",
        help="score a JSONL of completions against gold and print the reward breakdown",
        description=(
            "Each line is a JSON object carrying a completion (any of text, completion, output "
            "or response) and either an inline 'gold' record or an 'example_id' that --data can "
            "resolve."
        ),
    )
    check.add_argument("--file", required=True, help="JSONL of completions to score")
    check.add_argument("--data", default=None, help="corpus directory, to resolve gold by id")
    check.add_argument(
        "--lenient",
        action="store_true",
        help="allow the parser to recover an object from prose and syntax slips",
    )
    check.add_argument("--json", action="store_true", help="emit the scores as JSON")
    check.set_defaults(handler=_cmd_verify_check)


def _build_sft(commands: _SubParsers) -> None:
    sft = commands.add_parser("sft", help="supervised fine-tuning")
    sub = sft.add_subparsers(dest="sft_command", required=True, metavar="SUBCOMMAND")

    train = sub.add_parser("train", help="train a LoRA adapter on the gold records")
    train.add_argument("--model", required=True, help="base checkpoint, hub id or path")
    train.add_argument("--data", required=True, help="corpus directory")
    train.add_argument("--out", required=True, help="run directory for logs and the adapter")
    _add_train_options(train)
    _add_lora_options(train)
    _add_runtime_options(train)
    train.set_defaults(handler=_cmd_sft_train)


def _build_prefs(commands: _SubParsers) -> None:
    prefs = commands.add_parser("prefs", help="preference data")
    sub = prefs.add_subparsers(dest="prefs_command", required=True, metavar="SUBCOMMAND")

    mine = sub.add_parser("mine", help="sample completions and mine (chosen, rejected) pairs")
    mine.add_argument("--model", required=True, help="base checkpoint")
    mine.add_argument("--adapter", default=None, help="adapter to sample from, or 'none'")
    mine.add_argument("--data", required=True, help="corpus directory")
    mine.add_argument("--split", default="train", choices=SPLITS, help="split to sample from")
    mine.add_argument("--k", type=int, default=6, help="completions per prompt (default: 6)")
    mine.add_argument("--out", required=True, help="JSONL to write the pairs to")
    mine.add_argument("--samples-out", default=None, help="JSONL to keep the raw completions in")
    mine.add_argument("--stats-out", default=None, help="JSON to write the mining statistics to")
    mine.add_argument("--temperature", type=float, default=0.9, help="sampling temperature")
    mine.add_argument("--top-p", type=float, default=0.95, help="nucleus mass")
    mine.add_argument("--max-new-tokens", type=int, default=320, help="token budget per sample")
    mine.add_argument("--batch-size", type=int, default=8, help="rows per generate call")
    mine.add_argument(
        "--min-margin", type=float, default=0.05, help="smallest reward gap worth a pair"
    )
    mine.add_argument(
        "--max-pairs-per-prompt", type=int, default=2, help="cap on pairs from one prompt"
    )
    mine.add_argument("--seed", type=int, default=0, help="sampling and tie-break seed")
    _add_runtime_options(mine)
    mine.set_defaults(handler=_cmd_prefs_mine)


def _build_dpo(commands: _SubParsers) -> None:
    dpo = commands.add_parser("dpo", help="direct preference optimisation")
    sub = dpo.add_subparsers(dest="dpo_command", required=True, metavar="SUBCOMMAND")

    train = sub.add_parser("train", help="optimise the adapter on mined preference pairs")
    train.add_argument("--model", required=True, help="base checkpoint")
    train.add_argument("--adapter", default=None, help="adapter to continue from, or 'none'")
    train.add_argument("--pairs", required=True, help="JSONL of preference pairs")
    train.add_argument("--out", required=True, help="run directory for logs and the adapter")
    train.add_argument("--beta", type=float, default=0.1, help="KL strength (default: 0.1)")
    train.add_argument("--variant", default="sigmoid", choices=VARIANTS, help="objective")
    train.add_argument(
        "--label-smoothing", type=float, default=0.0, help="assumed label-flip rate, cdpo only"
    )
    train.add_argument(
        "--length-normalise",
        action="store_true",
        help="score completions per token instead of per sequence",
    )
    # The pipeline's preference-stage defaults, not the supervised trainer's: see
    # `_add_train_options` for why the two must differ.
    _add_train_options(train, lr=1e-5, batch_size=1, grad_accum=4)
    _add_lora_options(train)
    _add_runtime_options(train)
    train.set_defaults(handler=_cmd_dpo_train)


def _build_eval(commands: _SubParsers) -> None:
    evaluation = commands.add_parser("eval", help="evaluation, gating and the alignment tax")
    sub = evaluation.add_subparsers(dest="eval_command", required=True, metavar="SUBCOMMAND")

    run = sub.add_parser("run", help="decode one greedy completion per example and score it")
    run.add_argument("--model", required=True, help="base checkpoint")
    run.add_argument("--adapter", default=None, help="adapter to evaluate, or 'none'")
    run.add_argument("--data", required=True, help="corpus directory")
    run.add_argument("--split", default="test", choices=SPLITS, help="split to evaluate")
    run.add_argument("--out", required=True, help="JSON file for the evaluation report")
    run.add_argument("--label", default=None, help="variant name recorded in the report")
    run.add_argument("--batch-size", type=int, default=8, help="rows per generate call")
    run.add_argument("--max-new-tokens", type=int, default=320, help="token budget per completion")
    _add_runtime_options(run)
    run.set_defaults(handler=_cmd_eval_run)

    compare_parser = sub.add_parser(
        "compare",
        help="apply the promotion gate to two evaluation reports",
        description=(
            "Paired bootstrap and exact McNemar over the same examples, with an absolute JSON "
            "validity floor and a per-slice non-regression rule. Pass --gate to make the "
            "decision the process exit code."
        ),
    )
    compare_parser.add_argument("baseline", help="the incumbent's evaluation report")
    compare_parser.add_argument("candidate", help="the challenger's evaluation report")
    compare_parser.add_argument(
        "--margin", type=float, default=0.0, help="non-inferiority margin (default: 0.0)"
    )
    compare_parser.add_argument(
        "--json-floor", type=float, default=0.0, help="minimum strict JSON validity"
    )
    compare_parser.add_argument(
        "--schema-floor", type=float, default=0.0, help="minimum schema validity"
    )
    compare_parser.add_argument(
        "--slice-tolerance", type=float, default=0.05, help="allowed per-slice regression"
    )
    compare_parser.add_argument(
        "--field-tolerance",
        type=float,
        default=0.10,
        help="allowed drop in any one field's recall",
    )
    compare_parser.add_argument(
        "--field-support",
        type=int,
        default=10,
        help="gold paths a field needs before its recall can block a promotion",
    )
    compare_parser.add_argument("--alpha", type=float, default=0.05, help="McNemar significance")
    compare_parser.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples")
    compare_parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    compare_parser.add_argument("--out", default=None, help="also write the Markdown here")
    compare_parser.add_argument("--json-out", default=None, help="also write the full result here")
    compare_parser.add_argument(
        "--gate", action="store_true", help="exit non-zero unless the decision is PROMOTE"
    )
    compare_parser.set_defaults(handler=_cmd_eval_compare)

    tax = sub.add_parser("tax", help="general instruction-following before and after alignment")
    tax.add_argument("--model", required=True, help="base checkpoint")
    tax.add_argument("--adapter", default=None, help="the aligned adapter under test")
    tax.add_argument(
        "--baseline-adapter", default=None, help="adapter to compare against, or 'none'"
    )
    tax.add_argument("--out", required=True, help="JSON file for the comparison")
    tax.add_argument(
        "--margin", type=float, default=0.05, help="allowed loss before it counts as a regression"
    )
    tax.add_argument("--batch-size", type=int, default=8, help="rows per generate call")
    tax.add_argument("--max-new-tokens", type=int, default=320, help="token budget per reply")
    tax.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    _add_runtime_options(tax)
    tax.set_defaults(handler=_cmd_eval_tax)


def _build_crosscheck(commands: _SubParsers) -> None:
    check = commands.add_parser(
        "crosscheck",
        help="compare this package's DPO loss against the official TRL implementation",
        description=(
            "Runs TRL's own arithmetic on inputs both implementations read identically and "
            "reports the largest difference found. Exits non-zero only on a disagreement that "
            "no documented convention explains."
        ),
    )
    check.add_argument(
        "--tolerance", type=float, default=1e-9, help="largest difference counted as agreement"
    )
    check.add_argument("--pairs", type=int, default=8, help="preference pairs to compare on")
    check.add_argument("--tokens", type=int, default=1, help="scored tokens per completion")
    check.add_argument("--beta", type=float, default=0.1, help="KL strength for both sides")
    check.add_argument(
        "--label-smoothing", type=float, default=0.1, help="smoothing used for the cdpo comparison"
    )
    check.add_argument("--seed", type=int, default=0, help="seed for the synthetic batch")
    check.add_argument("--json", action="store_true", help="emit the results as JSON")
    check.set_defaults(handler=_cmd_crosscheck)


def _build_report(commands: _SubParsers) -> None:
    report = commands.add_parser("report", help="render a run directory as one Markdown comparison")
    report.add_argument("directory", help="a run directory")
    report.add_argument("--out", default=None, help="also write the Markdown here")
    report.set_defaults(handler=_cmd_report)


def _build_pipeline(commands: _SubParsers) -> None:
    pipeline = commands.add_parser("pipeline", help="the whole experiment, end to end")
    sub = pipeline.add_subparsers(dest="pipeline_command", required=True, metavar="SUBCOMMAND")

    run = sub.add_parser(
        "run",
        help="run every stage, skipping those whose artefacts already exist",
        description=(
            "Stages write to the run directory and are skipped when their outputs are already "
            "there, so an interrupted run resumes rather than restarting."
        ),
    )
    run.add_argument("--out", required=True, help="run directory")
    run.add_argument("--model", default=DEFAULT_MODEL, help="base checkpoint")
    run.add_argument("--seed", type=int, default=1, help="root seed")
    run.add_argument("--n-train", type=int, default=96, help="training examples")
    run.add_argument("--n-val", type=int, default=24, help="validation examples")
    run.add_argument("--n-test", type=int, default=48, help="test examples")
    run.add_argument("--sft-epochs", type=int, default=3, help="supervised passes over the data")
    run.add_argument("--dpo-epochs", type=int, default=2, help="preference passes over the pairs")
    run.add_argument("--batch-size", type=int, default=4, help="examples per forward pass")
    # A DPO step puts four sequences through a model for every pair, so it does not fit at
    # the width an SFT step does; accumulation keeps the effective batch under control.
    run.add_argument(
        "--dpo-batch-size", type=int, default=1, help="preference pairs per forward pass"
    )
    run.add_argument(
        "--dpo-grad-accum",
        type=int,
        default=4,
        help="preference micro-batches per optimiser step",
    )
    # An order of magnitude below the supervised rate: DPO is a correction to a model that
    # already works, and at the supervised rate it walks the policy off the reference.
    run.add_argument("--dpo-lr", type=float, default=1e-5, help="preference-stage learning rate")
    run.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="token budget per example",
    )
    run.add_argument("--k", type=int, default=6, help="completions sampled per prompt")
    run.add_argument("--temperature", type=float, default=0.9, help="sampling temperature")
    run.add_argument("--beta", type=float, default=0.1, help="DPO KL strength")
    run.add_argument("--variant", default="sigmoid", choices=VARIANTS, help="DPO objective")
    run.add_argument("--margin", type=float, default=0.0, help="gate non-inferiority margin")
    run.add_argument("--json-floor", type=float, default=0.0, help="gate JSON validity floor")
    _add_lora_options(run)
    _add_runtime_options(run)
    _add_stage_selection(run)
    run.set_defaults(handler=_cmd_pipeline_run)

    smoke = sub.add_parser(
        "smoke",
        help="run every stage on the tiny CI model in seconds, with nothing downloaded",
    )
    smoke.add_argument("--out", required=True, help="run directory")
    _add_stage_selection(smoke)
    smoke.set_defaults(handler=_cmd_pipeline_smoke)


def _add_stage_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=[stage.value for stage in STAGE_ORDER],
        help="run only these stages, always in dependency order",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-run stages whose artefacts already exist"
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, provider: ModelProvider | None = None) -> int:
    """Parse `argv`, run the command, and return the process exit code.

    Returning the code rather than calling `sys.exit` keeps every command testable: a test
    asserts on an integer instead of catching `SystemExit`, and `__main__.py` is the one place
    that turns the number into a process status.

    Args:
        argv: Arguments after the program name; `sys.argv[1:]` by default.
        provider: Where models come from. Supplied by the tests so that the training and
            evaluation commands can be exercised without a checkpoint or a tokenizer; a real
            invocation leaves it None and gets `HubProvider`.

    Returns:
        0 on success, 1 on a failure the user can act on, and whatever argparse chose for a
        usage error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    handler = cast(Handler, args.handler)
    try:
        return handler(args, provider)
    except (PipelineError, OSError, ValueError, KeyError) as exc:
        # Broad on purpose: these are the failures a user caused -- a missing file, a corpus
        # that does not match its manifest, a split that is empty -- and a traceback would
        # bury the one line that says which. Anything else propagates, because an unexpected
        # exception in a training loop is a bug and should look like one.
        print(f"sftdpo: {exc}", file=sys.stderr)
        return 1
