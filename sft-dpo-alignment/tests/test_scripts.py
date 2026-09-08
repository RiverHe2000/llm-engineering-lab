"""The reproduction script must describe the run the results document reports.

`docs/RESULTS.md` opens by saying every number on it is produced by
`scripts/run_experiments.sh`. That is only true while the flags in the script equal the
configuration the run actually recorded, and nothing but this test holds the two together:
an earlier draft of the script carried different split sizes and left the preference-stage
learning rate at the supervised default, which is the configuration that destroyed the model.
A reproduction script that reproduces the failure rather than the result is worse than none.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parent.parent
SCRIPT: Final = ROOT / "scripts" / "run_experiments.sh"
RUN_CONFIG: Final = ROOT / "docs" / "experiments" / "run_config.json"

#: Script flag -> key in the committed run configuration.
FLAGS: Final[dict[str, str]] = {
    "--seed": "seed",
    "--n-train": "n_train",
    "--n-val": "n_val",
    "--n-test": "n_test",
    "--sft-epochs": "sft_epochs",
    "--k": "sample_k",
    "--temperature": "temperature",
    "--dpo-epochs": "dpo_epochs",
    "--dpo-lr": "dpo_learning_rate",
    "--dpo-batch-size": "dpo_batch_size",
    "--dpo-grad-accum": "dpo_gradient_accumulation_steps",
    "--beta": "beta",
    "--variant": "variant",
    "--model": "model",
}


def _pipeline_invocation(script: str) -> str:
    """The `sftdpo pipeline run` command, up to the blank line that ends it.

    Only that block is inspected, because the `--quick` overrides live on an earlier line and
    carry deliberately different sizes -- and the header comment mentions the command by name
    without invoking it, so the anchor is the invocation itself.
    """
    start = script.index("-m sftdpo pipeline run")
    end = script.find("\n\n", start)
    return script[start : end if end >= 0 else len(script)]


def _flag_value(block: str, flag: str) -> str:
    found = re.search(rf"{re.escape(flag)}\s+(\S+)", block)
    assert found is not None, f"{flag} is not passed to `sftdpo pipeline run`"
    return found.group(1)


@pytest.mark.parametrize(("flag", "key"), sorted(FLAGS.items()))
def test_the_reproduction_script_passes_the_documented_value(flag: str, key: str) -> None:
    block = _pipeline_invocation(SCRIPT.read_text(encoding="utf-8"))
    config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    passed = _flag_value(block, flag).strip('"')
    if passed.startswith("$"):
        # `--model "$SMALL"`: the default the variable falls back to is what must match.
        default = re.search(r"SMALL=\$\{SFTDPO_SMALL_MODEL:-([^}]+)\}", SCRIPT.read_text())
        assert default is not None
        passed = default.group(1)
    expected = config[key]
    if isinstance(expected, int | float):
        assert float(passed) == pytest.approx(float(expected)), (flag, key)
    else:
        assert passed == str(expected), (flag, key)


def test_the_preference_learning_rate_is_not_the_supervised_one() -> None:
    """The specific mistake this file exists to prevent, stated on its own."""
    block = _pipeline_invocation(SCRIPT.read_text(encoding="utf-8"))
    config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    assert float(_flag_value(block, "--dpo-lr")) == pytest.approx(config["dpo_learning_rate"])
    assert config["dpo_learning_rate"] < config["learning_rate"]


def test_the_script_does_not_force_offline_mode() -> None:
    """A fresh machine has to be allowed to download the checkpoints on its first run."""
    for name in ("run_experiments.sh", "run_comparisons.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "export HF_HUB_OFFLINE=1" not in text, name
