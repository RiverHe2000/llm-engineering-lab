from __future__ import annotations

import pytest
import torch
from torch import nn

from loraeval.lora import (
    LoRAConfig,
    LoRALinear,
    count_parameters,
    inject_lora,
    load_lora_state_dict,
    lora_modules,
    lora_state_dict,
    mark_only_lora_as_trainable,
    merge_all,
    unmerge_all,
)


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [{"r": 0}, {"alpha": 0.0}, {"dropout": 1.0}, {"dropout": -0.1}, {"target_modules": "("}],
    )
    def test_invalid(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            LoRAConfig(**kwargs)  # type: ignore[arg-type]

    def test_scaling(self) -> None:
        assert LoRAConfig(r=8, alpha=16).scaling == 2.0


class TestLoRALinear:
    def test_equals_base_at_init(self) -> None:
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, LoRAConfig(r=4))
        x = torch.randn(3, 16)
        torch.testing.assert_close(lora(x), base(x))
        assert lora.in_features == 16 and lora.out_features == 8
        assert "r=4" in repr(lora)

    def test_base_frozen_and_lora_trainable(self) -> None:
        lora = LoRALinear(nn.Linear(16, 8), LoRAConfig(r=4))
        assert not lora.base.weight.requires_grad
        assert not lora.base.bias.requires_grad
        assert lora.lora_A.requires_grad and lora.lora_B.requires_grad
        assert lora.lora_A.shape == (4, 16) and lora.lora_B.shape == (8, 4)
        assert torch.equal(lora.lora_B, torch.zeros(8, 4))

    def test_merge_unmerge_round_trip(self) -> None:
        lora = LoRALinear(nn.Linear(16, 8), LoRAConfig(r=4, alpha=8))
        with torch.no_grad():
            lora.lora_B.normal_()
        original = lora.base.weight.clone()
        x = torch.randn(5, 16)
        unmerged_out = lora(x)
        assert lora.delta_weight().shape == (8, 16)

        lora.merge()
        assert "merged=True" in repr(lora)
        torch.testing.assert_close(lora(x), unmerged_out, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(lora.base.weight, original)
        lora.merge()  # idempotent

        lora.unmerge()
        assert "merged=False" in repr(lora)
        torch.testing.assert_close(lora.base.weight, original, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(lora(x), unmerged_out, atol=1e-6, rtol=1e-6)

    def test_gradients_flow_only_into_adapter(self) -> None:
        lora = LoRALinear(nn.Linear(16, 8), LoRAConfig(r=4))
        with torch.no_grad():
            lora.lora_B.normal_()
        lora(torch.randn(2, 16)).sum().backward()
        assert lora.base.weight.grad is None
        assert lora.lora_A.grad is not None and lora.lora_B.grad is not None
        assert torch.isfinite(lora.lora_A.grad).all()

    def test_dropout_only_in_training(self) -> None:
        lora = LoRALinear(nn.Linear(16, 8), LoRAConfig(r=4, dropout=0.5))
        with torch.no_grad():
            lora.lora_B.normal_()
        x = torch.randn(4, 16)
        lora.eval()
        torch.testing.assert_close(lora(x), lora(x))
        lora.train()
        assert not torch.allclose(lora(x), lora(x))


class TestInjection:
    def test_targets_are_wrapped_and_others_untouched(self, tiny_model: nn.Module) -> None:
        names = inject_lora(tiny_model, LoRAConfig(r=4, target_modules=r"\.(q_lin|v_lin)$"))
        assert len(names) == 2 * 2  # q and v in each of the 2 layers
        assert all(n.endswith(("q_lin", "v_lin")) for n in names)
        wrapped = lora_modules(tiny_model)
        assert set(wrapped) == set(names)
        layer0 = tiny_model.get_submodule("distilbert.transformer.layer.0.attention")
        assert isinstance(layer0.k_lin, nn.Linear)

    def test_model_output_unchanged_at_init(self, tiny_model: nn.Module) -> None:
        x = torch.randint(0, 40, (2, 7))
        mask = torch.ones_like(x)
        tiny_model.eval()
        with torch.no_grad():
            before = tiny_model(input_ids=x, attention_mask=mask).logits
            inject_lora(tiny_model, LoRAConfig(r=4))
            after = tiny_model(input_ids=x, attention_mask=mask).logits
        torch.testing.assert_close(before, after)

    def test_no_match_raises(self, tiny_model: nn.Module) -> None:
        with pytest.raises(ValueError, match=r"no nn\.Linear"):
            inject_lora(tiny_model, LoRAConfig(target_modules="does_not_exist"))

    def test_only_lora_and_head_trainable(self, tiny_model: nn.Module) -> None:
        inject_lora(tiny_model, LoRAConfig(r=4))
        mark_only_lora_as_trainable(tiny_model, extra_trainable=r"(pre_classifier|classifier)\.")
        for name, p in tiny_model.named_parameters():
            expected = ".lora_" in name or "classifier" in name
            assert p.requires_grad == expected, name
        counts = count_parameters(tiny_model)
        assert 0 < counts.trainable < counts.total
        assert "trainable" in str(counts)

    def test_merge_all_and_unmerge_all(self, tiny_model: nn.Module) -> None:
        inject_lora(tiny_model, LoRAConfig(r=4))
        for m in lora_modules(tiny_model).values():
            with torch.no_grad():
                m.lora_B.normal_()
        x = torch.randint(0, 40, (2, 7))
        tiny_model.eval()
        with torch.no_grad():
            ref = tiny_model(input_ids=x).logits
            assert merge_all(tiny_model) == 4
            torch.testing.assert_close(tiny_model(input_ids=x).logits, ref, atol=1e-5, rtol=1e-5)
            assert unmerge_all(tiny_model) == 4
            torch.testing.assert_close(tiny_model(input_ids=x).logits, ref, atol=1e-5, rtol=1e-5)

    def test_state_dict_round_trip(self, tiny_model: nn.Module) -> None:
        inject_lora(tiny_model, LoRAConfig(r=4))
        state = lora_state_dict(tiny_model, extra=r"classifier\.")
        assert all(".lora_" in k or "classifier" in k for k in state)
        assert any(".lora_A" in k for k in state) and any("pre_classifier" in k for k in state)
        n_adapter = sum(v.numel() for v in state.values())
        assert n_adapter < 0.5 * count_parameters(tiny_model).total

        for m in lora_modules(tiny_model).values():
            with torch.no_grad():
                m.lora_B.normal_()
        changed = lora_state_dict(tiny_model)
        load_lora_state_dict(tiny_model, state)
        restored = lora_state_dict(tiny_model)
        for k in restored:
            assert torch.equal(restored[k], state[k])
        assert any(not torch.equal(changed[k], restored[k]) for k in changed)

        with pytest.raises(KeyError):
            load_lora_state_dict(tiny_model, {"nope.lora_A": torch.zeros(1)})
