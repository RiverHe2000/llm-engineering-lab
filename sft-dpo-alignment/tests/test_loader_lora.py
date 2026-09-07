"""Tests for model loading, parameter accounting and the LoRA wiring.

Everything runs on the two-layer Qwen2 built from config, so the suite needs no downloads
and no GPU: the numbers asserted here (16,384 adapter parameters of 9,814,592, and the
attention/MLP split of that budget) are the real ones a `PeftModel` reports, which is what
makes them useful as a pin on the configuration.

`load_model` itself is exercised with the `transformers` and `peft` entry points replaced by
recorders. That covers the composition -- which arguments are forwarded, when the offline
flag is set, in what order the cheap validation happens -- without a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from sftdpo.modeling.loader import (
    DTYPE_ALIASES,
    TINY_MODEL_CONFIG,
    LoadedModel,
    ModelInfo,
    build_tiny_model,
    hub_offline,
    load_model,
    resolve_device,
    resolve_dtype,
)
from sftdpo.modeling.lora import (
    DEFAULT_TARGET_MODULES,
    QWEN2_ATTENTION_PROJECTIONS,
    QWEN2_MLP_PROJECTIONS,
    TrainableReport,
    attach_adapter,
    default_target_modules,
    lora_config,
    reference_context,
    trainable_parameter_report,
)

TINY_TOTAL_PARAMETERS = 9_798_208
ADAPTED_TOTAL_PARAMETERS = 9_814_592
ADAPTER_PARAMETERS = 16_384
ATTENTION_ONLY_PARAMETERS = 7_168

INPUT_IDS = torch.tensor([[11, 22, 33, 44]])


@pytest.fixture
def tiny() -> Any:
    """A fresh model per test: PEFT rewrites the module tree of the model it is given."""
    return build_tiny_model(seed=0)


@pytest.fixture
def adapted(tiny: Any) -> Any:
    return attach_adapter(tiny, lora_config())


def _excite_adapter(model: Any) -> None:
    """Give the adapter a non-zero effect.

    LoRA initialises `lora_B` to zero, so a freshly attached adapter is the identity and any
    test comparing adapted output against base output would pass vacuously.
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0.0, std=0.05)


def _logits(model: Any) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=INPUT_IDS).logits.clone()


class RecordingLoader:
    """Stands in for `AutoModelForCausalLM` / `AutoTokenizer`."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, name: str, **kwargs: Any) -> Any:
        self.calls.append((name, kwargs))
        return self.result


class RecordingPeft:
    """Stands in for `peft.PeftModel`."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[Any, str, dict[str, Any]]] = []

    def from_pretrained(self, model: Any, path: str, **kwargs: Any) -> Any:
        self.calls.append((model, path, kwargs))
        return self.result


# --------------------------------------------------------------------------------------
# loader.py: the tiny CI model
# --------------------------------------------------------------------------------------


def test_tiny_model_has_the_documented_parameter_count(tiny: Any) -> None:
    assert sum(p.numel() for p in tiny.parameters()) == TINY_TOTAL_PARAMETERS


def test_tiny_model_is_the_shape_the_config_promises(tiny: Any) -> None:
    assert tiny.config.num_hidden_layers == TINY_MODEL_CONFIG["num_hidden_layers"] == 2
    assert tiny.config.hidden_size == 64
    assert tiny.config.tie_word_embeddings is True


def test_tiny_model_is_reproducible_for_a_seed() -> None:
    left = build_tiny_model(seed=3).state_dict()["model.embed_tokens.weight"]
    right = build_tiny_model(seed=3).state_dict()["model.embed_tokens.weight"]
    assert torch.equal(left, right)


def test_tiny_model_depends_on_its_seed() -> None:
    left = build_tiny_model(seed=3).state_dict()["model.embed_tokens.weight"]
    right = build_tiny_model(seed=4).state_dict()["model.embed_tokens.weight"]
    assert not torch.equal(left, right)


def test_building_a_model_leaves_the_global_rng_alone() -> None:
    """Otherwise a test's random data would depend on whether another test built a model."""
    torch.manual_seed(1234)
    before = torch.rand(4)
    torch.manual_seed(1234)
    build_tiny_model(seed=99)
    after = torch.rand(4)
    assert torch.equal(before, after)


def test_tiny_model_accepts_config_overrides() -> None:
    single = build_tiny_model(seed=0, num_hidden_layers=1)
    assert single.config.num_hidden_layers == 1
    assert sum(p.numel() for p in single.parameters()) < TINY_TOTAL_PARAMETERS


def test_tiny_model_runs_a_forward_pass(tiny: Any) -> None:
    logits = _logits(tiny)
    assert logits.shape == (1, INPUT_IDS.shape[1], TINY_MODEL_CONFIG["vocab_size"])
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------------------
# loader.py: configuration resolution
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("float32", torch.float32),
        ("FP32", torch.float32),
        (" bf16 ", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("half", torch.float16),
    ],
)
def test_resolve_dtype_accepts_the_documented_aliases(alias: str, expected: Any) -> None:
    assert resolve_dtype(alias) is expected


def test_every_alias_maps_to_a_torch_dtype() -> None:
    assert all(isinstance(value, torch.dtype) for value in DTYPE_ALIASES.values())


def test_resolve_dtype_rejects_a_typo() -> None:
    with pytest.raises(ValueError, match="unknown dtype 'bflaot16'"):
        resolve_dtype("bflaot16")


@pytest.mark.parametrize("requested", [None, "auto"])
def test_resolve_device_prefers_cuda_when_it_exists(
    requested: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(requested) == "cuda"


@pytest.mark.parametrize("requested", [None, "auto"])
def test_resolve_device_falls_back_to_cpu(
    requested: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(requested) == "cpu"


def test_resolve_device_honours_an_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_honours_an_explicit_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"


def test_explicit_cuda_on_a_cpu_box_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CUDA device"):
        resolve_device("cuda")


def test_resolve_device_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="unknown device 'tpu'"):
        resolve_device("tpu")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("", False)],
)
def test_hub_offline_reads_the_environment(
    value: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hub_offline() is expected


def test_hub_offline_is_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert hub_offline() is False


# --------------------------------------------------------------------------------------
# loader.py: ModelInfo
# --------------------------------------------------------------------------------------


def test_model_info_counts_a_fully_trainable_model(tiny: Any) -> None:
    info = ModelInfo.from_model(tiny, name="tiny")
    assert info.total_parameters == TINY_TOTAL_PARAMETERS
    assert info.trainable_parameters == TINY_TOTAL_PARAMETERS
    assert info.trainable_pct == pytest.approx(100.0)
    assert info.dtype == "float32"
    assert info.device == "cpu"


def test_model_info_sees_frozen_weights(tiny: Any) -> None:
    for parameter in tiny.parameters():
        parameter.requires_grad_(False)
    info = ModelInfo.from_model(tiny, name="tiny")
    assert info.trainable_parameters == 0
    assert info.trainable_pct == 0.0


def test_model_info_records_the_adapter_share(adapted: Any) -> None:
    info = ModelInfo.from_model(adapted, name="tiny+lora")
    assert info.trainable_parameters == ADAPTER_PARAMETERS
    assert info.total_parameters == ADAPTED_TOTAL_PARAMETERS


def test_model_info_is_manifest_ready(tiny: Any) -> None:
    payload = json.loads(json.dumps(ModelInfo.from_model(tiny, name="tiny").as_dict()))
    assert payload["name"] == "tiny"
    assert payload["dtype"] == "float32"
    assert payload["trainable_pct"] == pytest.approx(100.0)


def test_model_info_refuses_a_model_with_no_parameters() -> None:
    with pytest.raises(ValueError, match="no parameters"):
        ModelInfo.from_model(torch.nn.Module(), name="empty")


# --------------------------------------------------------------------------------------
# loader.py: load_model
# --------------------------------------------------------------------------------------


def _patch_transformers(monkeypatch: pytest.MonkeyPatch, model: Any) -> tuple[Any, Any]:
    import transformers

    model_loader = RecordingLoader(model)
    tokenizer_loader = RecordingLoader(object())
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", model_loader)
    monkeypatch.setattr(transformers, "AutoTokenizer", tokenizer_loader)
    return model_loader, tokenizer_loader


def test_load_model_returns_model_tokenizer_and_accounting(
    tiny: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    model_loader, tokenizer_loader = _patch_transformers(monkeypatch, tiny)
    loaded = load_model("some/checkpoint", dtype="float32", device="cpu")
    assert isinstance(loaded, LoadedModel)
    assert loaded.model is tiny
    assert loaded.tokenizer is tokenizer_loader.result
    assert loaded.info.name == "some/checkpoint"
    assert loaded.info.total_parameters == TINY_TOTAL_PARAMETERS
    assert model_loader.calls[0][1]["dtype"] is torch.float32
    assert model_loader.calls[0][1]["local_files_only"] is False
    assert loaded.model.training is False


def test_load_model_passes_the_offline_flag_to_both_loaders(
    tiny: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model_loader, tokenizer_loader = _patch_transformers(monkeypatch, tiny)
    load_model("some/checkpoint", dtype="bf16", device="cpu")
    assert model_loader.calls[0][1]["local_files_only"] is True
    assert tokenizer_loader.calls[0][1]["local_files_only"] is True


def test_load_model_attaches_a_saved_adapter(
    tiny: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peft

    _patch_transformers(monkeypatch, tiny)
    peft_loader = RecordingPeft(tiny)
    monkeypatch.setattr(peft, "PeftModel", peft_loader)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    load_model("some/checkpoint", dtype="float32", device="cpu", adapter_path=adapter_dir)

    assert len(peft_loader.calls) == 1
    assert peft_loader.calls[0][1] == str(adapter_dir)
    assert peft_loader.calls[0][2] == {"is_trainable": False}


def test_load_model_rejects_a_missing_adapter_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter directory does not exist"):
        load_model("some/checkpoint", adapter_path=tmp_path / "nope")


def test_load_model_validates_before_it_loads_anything(
    tiny: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in the config should cost a millisecond, not a download."""
    model_loader, _ = _patch_transformers(monkeypatch, tiny)
    with pytest.raises(ValueError, match="unknown dtype"):
        load_model("some/checkpoint", dtype="float33", device="cpu")
    assert model_loader.calls == []


# --------------------------------------------------------------------------------------
# lora.py: configuration
# --------------------------------------------------------------------------------------


def test_default_targets_are_the_seven_qwen_projections() -> None:
    assert default_target_modules() == DEFAULT_TARGET_MODULES
    assert len(DEFAULT_TARGET_MODULES) == 7
    assert set(DEFAULT_TARGET_MODULES) == set(QWEN2_ATTENTION_PROJECTIONS) | set(
        QWEN2_MLP_PROJECTIONS
    )


def test_lora_config_defaults_are_the_ones_the_results_were_produced_with() -> None:
    cfg = lora_config()
    assert (cfg.r, cfg.lora_alpha, cfg.lora_dropout) == (8, 16, 0.0)
    assert cfg.task_type == "CAUSAL_LM"
    assert cfg.bias == "none"
    # peft normalises the list into a set, so order is not part of the contract.
    assert set(cfg.target_modules) == set(DEFAULT_TARGET_MODULES)


def test_lora_config_keeps_the_scale_at_two_by_default() -> None:
    cfg = lora_config(r=16, alpha=32)
    assert cfg.lora_alpha / cfg.r == 2.0


def test_lora_config_accepts_explicit_targets() -> None:
    cfg = lora_config(targets=["q_proj", "v_proj"])
    assert set(cfg.target_modules) == {"q_proj", "v_proj"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"r": 0}, "rank must be at least 1"),
        ({"r": -4}, "rank must be at least 1"),
        ({"alpha": 0}, "alpha must be at least 1"),
        ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
        ({"dropout": -0.1}, r"dropout must be in \[0, 1\)"),
        ({"targets": []}, "must not be empty"),
        ({"targets": ["q_proj", "q_proj"]}, "duplicate target modules"),
    ],
)
def test_lora_config_validates_its_arguments(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lora_config(**kwargs)


# --------------------------------------------------------------------------------------
# lora.py: attachment and accounting
# --------------------------------------------------------------------------------------


def test_adapter_size_is_pinned(adapted: Any) -> None:
    """The number every LoRA claim in the results rests on."""
    report = trainable_parameter_report(adapted)
    assert report.trainable == ADAPTER_PARAMETERS
    assert report.total == ADAPTED_TOTAL_PARAMETERS
    assert report.percentage == pytest.approx(0.167, abs=0.0005)


def test_only_adapter_tensors_are_trainable(adapted: Any) -> None:
    trainable = [name for name, p in adapted.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_adapter_reaches_every_targeted_projection(adapted: Any) -> None:
    adapted_names = {
        name.split(".")[-2] for name, _ in adapted.named_modules() if name.endswith("lora_A")
    }
    assert adapted_names == set(DEFAULT_TARGET_MODULES)


def test_mlp_projections_are_the_larger_half_of_the_budget(tiny: Any) -> None:
    """Targeting attention only would leave more than half the adapter budget unspent."""
    attention_only = attach_adapter(tiny, lora_config(targets=QWEN2_ATTENTION_PROJECTIONS))
    report = trainable_parameter_report(attention_only)
    assert report.trainable == ATTENTION_ONLY_PARAMETERS
    assert report.trainable < ADAPTER_PARAMETERS - report.trainable


def test_report_on_a_base_model_counts_everything(tiny: Any) -> None:
    report = trainable_parameter_report(tiny)
    assert report.trainable == report.total == TINY_TOTAL_PARAMETERS
    assert report.percentage == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("trainable", "total", "message"),
    [
        (0, 0, "must be positive"),
        (10, 5, "between 0 and total"),
        (-1, 5, "between 0 and total"),
    ],
)
def test_trainable_report_rejects_impossible_counts(
    trainable: int, total: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainableReport(trainable=trainable, total=total)


def test_trainable_report_renders_the_headline_number() -> None:
    report = TrainableReport(trainable=ADAPTER_PARAMETERS, total=ADAPTED_TOTAL_PARAMETERS)
    assert str(report) == "16,384 trainable of 9,814,592 (0.167 %)"
    assert report.as_dict()["trainable"] == ADAPTER_PARAMETERS


# --------------------------------------------------------------------------------------
# lora.py: the reference policy
# --------------------------------------------------------------------------------------


def test_a_fresh_adapter_is_the_identity(tiny: Any) -> None:
    """LoRA zero-initialises `lora_B`, so training starts exactly at the base model."""
    base = _logits(tiny)
    adapted = attach_adapter(tiny, lora_config())
    assert torch.equal(base, _logits(adapted))


def test_an_excited_adapter_changes_the_model(tiny: Any) -> None:
    base = _logits(tiny)
    adapted = attach_adapter(tiny, lora_config())
    _excite_adapter(adapted)
    assert not torch.equal(base, _logits(adapted))


def test_reference_context_reproduces_the_base_model_exactly(tiny: Any) -> None:
    """The result that removes the second copy of the weights from a DPO run."""
    base = _logits(tiny)
    adapted = attach_adapter(tiny, lora_config())
    _excite_adapter(adapted)
    with reference_context(adapted) as reference:
        assert torch.equal(base, _logits(reference))


def test_reference_context_puts_the_adapter_back(tiny: Any) -> None:
    adapted = attach_adapter(tiny, lora_config())
    _excite_adapter(adapted)
    policy = _logits(adapted)
    with reference_context(adapted) as reference:
        inside = _logits(reference)
    assert not torch.equal(policy, inside)
    assert torch.equal(policy, _logits(adapted))


def test_reference_context_yields_the_same_object(adapted: Any) -> None:
    with reference_context(adapted) as reference:
        assert reference is adapted


def test_reference_context_refuses_a_model_with_no_adapter(tiny: Any) -> None:
    with pytest.raises(TypeError, match="needs a PEFT model"), reference_context(tiny):
        pytest.fail("a model without an adapter has no reference policy to offer")
