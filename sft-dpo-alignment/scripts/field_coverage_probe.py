"""Count what happened to one field, record by record, under two adapters.

Written to answer a question the evaluation report could not: the DPO model's exact-match
rate fell while every other metric rose, and the report stores grades rather than
completions, so there was no way to see *which* field moved. This regenerates a split under
two adapters and counts a named field directly.

It is kept in the repository rather than thrown away because the answer it produced —
`flags` recall 0.610 to 0.000, the model having stopped emitting the field entirely — is the
finding behind `Floors.max_field_recall_drop`, and section 5 of `docs/RESULTS.md` cites its
output. A number a reader cannot regenerate is an assertion, not a result.

The permanent version of this check is `EvalReport.per_field`, which covers every field on
every evaluation at no extra cost. This script stays for the per-record breakdown that the
aggregate does not carry.

Usage:
    python scripts/field_coverage_probe.py --run runs/main --field flags \\
        --out docs/experiments/flags_hedge.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sftdpo.eval.generate import generate_samples
from sftdpo.modeling.chat import ChatFormatter
from sftdpo.modeling.loader import load_model
from sftdpo.schemas import Example, GenerationConfig
from sftdpo.task.dataset import Dataset
from sftdpo.verify.parse import extract_json


def _completions(
    model: str, adapter: str, examples: list[Example], *, max_new_tokens: int, batch_size: int
) -> dict[str, str]:
    """One greedy completion per example, keyed by example id."""
    loaded = load_model(model, adapter_path=adapter)
    samples = generate_samples(
        loaded.model,
        loaded.tokenizer,
        examples,
        model_name=adapter,
        formatter=ChatFormatter(tokenizer=loaded.tokenizer),
        config=GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0),
        batch_size=batch_size,
    )
    return {sample.example_id: sample.text for sample in samples}


def _members(record: dict[str, Any], field: str) -> tuple[list[str], bool]:
    """The field's members and whether the key was present at all.

    A key that is present but not a list counts as present with no members: the model tried
    and produced something unusable, which is a different failure from not trying.
    """
    if field not in record:
        return [], False
    value = record[field]
    return ([str(item) for item in value] if isinstance(value, list) else []), True


def probe(
    run_dir: Path, field: str, model: str, variants: tuple[str, ...], *, max_new_tokens: int
) -> dict[str, Any]:
    """Regenerate the test split under each variant's adapter and count `field`."""
    dataset = Dataset.load(run_dir / "data")
    examples = list(dataset.examples_for("test"))
    gold = {
        example.example_id: json.loads(example.gold.model_dump_json()).get(field) or []
        for example in examples
    }

    report: dict[str, Any] = {}
    for variant in variants:
        text = _completions(
            model,
            str(run_dir / variant / "adapter"),
            examples,
            max_new_tokens=max_new_tokens,
            batch_size=8,
        )
        counts: Counter[str] = Counter()
        expected = emitted = correct = 0
        for example in examples:
            want = gold[example.example_id]
            expected += len(want)
            parsed = extract_json(text[example.example_id], strict=True)
            if not parsed.ok or parsed.value is None:
                counts["unparseable"] += 1
                continue
            have, present = _members(parsed.value, field)
            if not present:
                counts["key_absent"] += 1
            emitted += len(have)
            correct += len(set(have) & set(want))
            if want and not have:
                counts["dropped_every_member"] += 1
            elif want and set(have) == set(want):
                counts["exact"] += 1
            elif want:
                counts["partial_or_wrong"] += 1
            elif have:
                counts["invented"] += 1
            else:
                counts["both_empty"] += 1
        report[variant] = {
            "counts": dict(sorted(counts.items())),
            "gold_members": expected,
            "emitted_members": emitted,
            "correct_members": correct,
            "recall": correct / expected if expected else 1.0,
            "precision": correct / emitted if emitted else 1.0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/main", help="run directory")
    parser.add_argument("--field", default="flags", help="top-level field to count")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="base model")
    parser.add_argument(
        "--variants", nargs="+", default=["sft", "dpo"], help="run subdirectories to compare"
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--out", default=None, help="write the counts here as JSON")
    args = parser.parse_args()

    report = probe(
        Path(args.run),
        args.field,
        args.model,
        tuple(args.variants),
        max_new_tokens=args.max_new_tokens,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out is not None:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
