"""Model construction and the three fine-tuning strategies compared in this project.

* ``head``: freeze the encoder, train only the classification head (linear-probe style).
* ``lora``: freeze the encoder, inject LoRA into attention projections, train LoRA + head.
* ``full``: train everything (the classic, memory-hungry baseline).
"""

from __future__ import annotations

import re
from typing import Literal, cast

from torch import Tensor, nn

from loraeval.lora import (
    LoRAConfig,
    ParamCount,
    count_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
)

Strategy = Literal["head", "lora", "full"]
STRATEGIES: tuple[Strategy, ...] = ("head", "lora", "full")

# Classification-head parameter names for the common HF architectures:
# DistilBERT: pre_classifier.* / classifier.*; BERT/RoBERTa: classifier.*; Qwen/LLaMA: score.*
DEFAULT_HEAD_PATTERN = r"(^|\.)(pre_classifier|classifier|score)\."


def load_pretrained(model_name: str, num_labels: int) -> nn.Module:
    """Load an ``AutoModelForSequenceClassification`` with a fresh ``num_labels`` head."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return cast(nn.Module, model)


def apply_strategy(
    model: nn.Module,
    strategy: Strategy,
    lora: LoRAConfig | None = None,
    head_pattern: str = DEFAULT_HEAD_PATTERN,
) -> ParamCount:
    """Mutate ``model`` in place according to ``strategy`` and return the parameter budget."""
    head = re.compile(head_pattern)
    if strategy == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif strategy == "head":
        n_head = 0
        for name, p in model.named_parameters():
            is_head = head.search(name) is not None
            p.requires_grad_(is_head)
            n_head += int(is_head)
        if n_head == 0:
            raise ValueError(f"head_pattern={head_pattern!r} matched no parameters")
    elif strategy == "lora":
        if lora is None:
            raise ValueError("strategy='lora' requires a LoRAConfig")
        inject_lora(model, lora)
        mark_only_lora_as_trainable(model, extra_trainable=head_pattern)
    else:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    return count_parameters(model)


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Weights that were actually trained; for LoRA/head this is <1% of the model."""
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable}


def load_trainable_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Strict partial load: every key in ``state`` must exist in the model."""
    unknown = sorted(set(state) - set(model.state_dict()))
    if unknown:
        raise KeyError(f"state contains keys not in model: {unknown[:5]}")
    model.load_state_dict(state, strict=False)
