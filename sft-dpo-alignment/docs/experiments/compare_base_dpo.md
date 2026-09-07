# Promotion gate: Qwen/Qwen2.5-0.5B-Instruct+dpo vs Qwen/Qwen2.5-0.5B-Instruct

**Decision: PROMOTE** (exit code 0)

- the paired difference is +0.7562 with a 95% CI of [+0.6875, +0.8187], which excludes zero
- McNemar p = 7.52e-37 clears alpha = 0.0500
- no floor was breached, no slice regressed and no field lost recall

## Headline metrics (n = 160)

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| JSON validity (strict) | 0.7438 | 1.0000 | +0.2562 |
| Schema validity | 0.2313 | 0.9875 | +0.7563 |
| Mean field F1 | 0.5265 | 0.9133 | +0.3868 |
| Exact match | 0.0000 | 0.0500 | +0.0500 |
| Mean reward | 0.5109 | 0.9455 | +0.4346 |

## Paired comparison

| Quantity | Value |
| --- | --- |
| Success difference | +0.7562 |
| Difference CI (95%) | [+0.6875, +0.8187] |
| Field F1 difference | +0.3868 |
| Field F1 CI | [+0.3346, +0.4391] |
| Non-inferiority margin | 0.0200 |
| Non-inferior | yes |
| McNemar b / c | 0 / 121 |
| McNemar p (exact) | 7.52e-37 |
| Alpha | 0.0500 |
| Baseline success (Wilson) | 0.2313 [+0.1727, +0.3024] |
| Candidate success (Wilson) | 0.9875 [+0.9556, +0.9966] |

## Floors

| Floor | Required | Candidate | Cleared |
| --- | ---: | ---: | --- |
| JSON validity | 0.9500 | 1.0000 | yes |
| Schema validity | 0.0000 | 0.9875 | yes |

## Per slice (tolerance 0.0500)

| Slice | n | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 27 | 0.2593 | 0.9630 | +0.7037 | no |
| distractor | 27 | 0.2963 | 1.0000 | +0.7037 | no |
| mixed_formats | 27 | 0.2222 | 1.0000 | +0.7778 | no |
| absent_fields | 27 | 0.5185 | 1.0000 | +0.4815 | no |
| long_context | 26 | 0.0000 | 1.0000 | +1.0000 | no |
| many_items | 26 | 0.0769 | 0.9615 | +0.8846 | no |

## Per field recall (tolerance 0.1000)

| Field | Gold paths | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| client_name | 160 | 0.6250 | 1.0000 | +0.3750 | no |
| record_date | 160 | 0.6500 | 0.9875 | +0.3375 | no |
| risk_profile | 160 | 0.5813 | 1.0000 | +0.4187 | no |
| objectives | 411 | 0.2652 | 1.0000 | +0.7348 | no |
| recommendations | 1513 | 0.4679 | 0.8691 | +0.4012 | no |
| fees | 320 | 0.5000 | 1.0000 | +0.5000 | no |
| review_months | 160 | 0.5188 | 1.0000 | +0.4812 | no |
| flags | 200 | 0.0000 | 0.0000 | +0.0000 | no |
