# Promotion gate: Qwen/Qwen2.5-0.5B-Instruct+sft vs Qwen/Qwen2.5-0.5B-Instruct

**Decision: PROMOTE** (exit code 0)

- the paired difference is +0.6000 with a 95% CI of [+0.5186, +0.6813], which excludes zero
- McNemar p = 4.72e-25 clears alpha = 0.0500
- no floor was breached, no slice regressed and no field lost recall

## Headline metrics (n = 160)

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| JSON validity (strict) | 0.7438 | 0.9750 | +0.2312 |
| Schema validity | 0.2313 | 0.8313 | +0.6000 |
| Mean field F1 | 0.5265 | 0.8914 | +0.3648 |
| Exact match | 0.0000 | 0.4375 | +0.4375 |
| Mean reward | 0.5109 | 0.8961 | +0.3852 |

## Paired comparison

| Quantity | Value |
| --- | --- |
| Success difference | +0.6000 |
| Difference CI (95%) | [+0.5186, +0.6813] |
| Field F1 difference | +0.3648 |
| Field F1 CI | [+0.3076, +0.4239] |
| Non-inferiority margin | 0.0200 |
| Non-inferior | yes |
| McNemar b / c | 4 / 100 |
| McNemar p (exact) | 4.72e-25 |
| Alpha | 0.0500 |
| Baseline success (Wilson) | 0.2313 [+0.1727, +0.3024] |
| Candidate success (Wilson) | 0.8313 [+0.7656, +0.8814] |

## Floors

| Floor | Required | Candidate | Cleared |
| --- | ---: | ---: | --- |
| JSON validity | 0.9500 | 0.9750 | yes |
| Schema validity | 0.0000 | 0.8313 | yes |

## Per slice (tolerance 0.0500)

| Slice | n | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 27 | 0.2593 | 0.9630 | +0.7037 | no |
| distractor | 27 | 0.2963 | 0.7407 | +0.4444 | no |
| mixed_formats | 27 | 0.2222 | 0.7407 | +0.5185 | no |
| absent_fields | 27 | 0.5185 | 0.9630 | +0.4444 | no |
| long_context | 26 | 0.0000 | 0.7692 | +0.7692 | no |
| many_items | 26 | 0.0769 | 0.8077 | +0.7308 | no |

## Per field recall (tolerance 0.1000)

| Field | Gold paths | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| client_name | 160 | 0.6250 | 0.9750 | +0.3500 | no |
| record_date | 160 | 0.6500 | 0.9313 | +0.2812 | no |
| risk_profile | 160 | 0.5813 | 0.9688 | +0.3875 | no |
| objectives | 411 | 0.2652 | 0.9562 | +0.6910 | no |
| recommendations | 1513 | 0.4679 | 0.8387 | +0.3708 | no |
| fees | 320 | 0.5000 | 0.8938 | +0.3938 | no |
| review_months | 160 | 0.5188 | 0.9375 | +0.4187 | no |
| flags | 200 | 0.0000 | 0.6100 | +0.6100 | no |
