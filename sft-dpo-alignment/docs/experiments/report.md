# sft-dpo-alignment: run `main`

## Corpus

Seed 1, 640 examples, content hash `7fa21199f5eab0b5`.

| Split | n | Objectives present | Recommendations present | Flags present |
| --- | ---: | ---: | ---: | ---: |
| train | 400 | 0.9450 | 1.0000 | 0.9525 |
| val | 80 | 0.9125 | 1.0000 | 0.9625 |
| test | 160 | 0.9500 | 1.0000 | 0.9500 |

## Evaluation

| Variant | Model | n | JSON valid | Schema valid | Field F1 | Exact match | Mean reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | Qwen/Qwen2.5-0.5B-Instruct | 160 | 0.7438 | 0.2313 | 0.5265 | 0.0000 | 0.5109 |
| sft | Qwen/Qwen2.5-0.5B-Instruct+sft | 160 | 0.9750 | 0.8313 | 0.8914 | 0.4375 | 0.8961 |
| dpo | Qwen/Qwen2.5-0.5B-Instruct+dpo | 160 | 1.0000 | 0.9875 | 0.9133 | 0.0500 | 0.9455 |
| big | big-prompted | 160 | 0.7625 | 0.7625 | 0.7122 | 0.0500 | 0.7323 |
| mid | mid-prompted | 160 | 0.8125 | 0.3688 | 0.5544 | 0.0000 | 0.5689 |

## Schema validity per slice

| Slice | base | sft | dpo | big | mid |
| --- | ---: | ---: | ---: | ---: | ---: |
| absent_fields | 0.5185 | 0.9630 | 1.0000 | 1.0000 | 0.2593 |
| clean | 0.2593 | 0.9630 | 0.9630 | 1.0000 | 0.4074 |
| distractor | 0.2963 | 0.7407 | 1.0000 | 1.0000 | 0.7037 |
| long_context | 0.0000 | 0.7692 | 1.0000 | 0.4615 | 0.3077 |
| many_items | 0.0769 | 0.8077 | 0.9615 | 0.0769 | 0.2308 |
| mixed_formats | 0.2222 | 0.7407 | 1.0000 | 1.0000 | 0.2963 |

## Parse gap: strict JSON versus a repaired parse

| Variant | Strict | Lenient | Gap | Repaired |
| --- | ---: | ---: | ---: | ---: |
| base | 0.7438 | 0.8875 | 0.1437 | 23 |
| sft | 0.9750 | 0.9750 | 0.0000 | 0 |
| dpo | 1.0000 | 1.0000 | 0.0000 | 0 |
| big | 0.7625 | 0.8375 | 0.0750 | 12 |
| mid | 0.8125 | 0.9062 | 0.0938 | 15 |

The gap is the size of the repair step a deployment would need in front of the model. It cannot be negative, because a lenient parse accepts everything a strict one does.

## Training

### Supervised fine-tuning

| Quantity | Value |
| --- | ---: |
| steps | 300 |
| supervised tokens | 227,844 |
| final train loss | 0.0006 |
| loss reduction | 829.1090 |
| best val loss | 0.0008 |
| trainable parameters | 4,399,104 |
| trainable pct | 0.8826 |
| diverged | no |

### Direct preference optimisation

| Quantity | Value |
| --- | ---: |
| steps | 168 |
| pairs seen | 671 |
| beta | 0.1000 |
| variant | sigmoid |
| final loss | 0.1607 |
| initial reward accuracy | 0.5000 |
| final reward accuracy | 1.0000 |
| reward margin slope | 0.0089 |
| diverged | no |

## Preference mining

| Quantity | Value |
| --- | ---: |
| prompts | 400 |
| prompts with pairs | 377 |
| pairs | 671 |
| pairs written | 671 |
| mean margin | 0.3839 |
| prompt yield | 0.9425 |
| saturation rate | 0.0025 |
| flat prompts | 16 |
| single sample prompts | 0 |
| pairs dropped by cap | 0 |
| gold fallback used | no |

A high saturation rate with few pairs is a finished pipeline rather than a broken one: the policy has outgrown this data and the next move is harder prompts, not a lower margin.

## Promotion gates

| Comparison | Decision | n | Success difference | CI | McNemar p |
| --- | --- | ---: | ---: | --- | ---: |
| base_dpo | PROMOTE | 160 | +0.8438 | [+0.7812, +0.9000] | 0.0000 |
| base_sft | PROMOTE | 160 | +0.6813 | [+0.5938, +0.7625] | 0.0000 |
| sft_dpo | PROMOTE | 160 | +0.1625 | [+0.1062, +0.2188] | 0.0000 |

HOLD means the candidate is not worse but has not been shown to be better. It is a first-class outcome: collapsing it into either neighbour would either ship models on noise or call an underpowered run a failure.
