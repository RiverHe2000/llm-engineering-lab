# Promotion gate: Qwen/Qwen2.5-0.5B-Instruct+dpo vs mid-prompted

**Decision: PROMOTE** (exit code 0)

- the paired difference is +0.6188 with a 95% CI of [+0.5437, +0.6937], which excludes zero
- McNemar p = 8.05e-29 clears alpha = 0.0500
- no floor was breached, no slice regressed and no field lost recall

## Headline metrics (n = 160)

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| JSON validity (strict) | 0.8125 | 1.0000 | +0.1875 |
| Schema validity | 0.3688 | 0.9875 | +0.6188 |
| Mean field F1 | 0.5544 | 0.9133 | +0.3589 |
| Exact match | 0.0000 | 0.0500 | +0.0500 |
| Mean reward | 0.5689 | 0.9455 | +0.3766 |

## Paired comparison

| Quantity | Value |
| --- | --- |
| Success difference | +0.6188 |
| Difference CI (95%) | [+0.5437, +0.6937] |
| Field F1 difference | +0.3589 |
| Field F1 CI | [+0.2961, +0.4217] |
| Non-inferiority margin | 0.0200 |
| Non-inferior | yes |
| McNemar b / c | 1 / 100 |
| McNemar p (exact) | 8.05e-29 |
| Alpha | 0.0500 |
| Baseline success (Wilson) | 0.3688 [+0.2979, +0.4458] |
| Candidate success (Wilson) | 0.9875 [+0.9556, +0.9966] |

## Floors

| Floor | Required | Candidate | Cleared |
| --- | ---: | ---: | --- |
| JSON validity | 0.9500 | 1.0000 | yes |
| Schema validity | 0.0000 | 0.9875 | yes |

## Per slice (tolerance 0.0500)

| Slice | n | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 27 | 0.4074 | 0.9630 | +0.5556 | no |
| distractor | 27 | 0.7037 | 1.0000 | +0.2963 | no |
| mixed_formats | 27 | 0.2963 | 1.0000 | +0.7037 | no |
| absent_fields | 27 | 0.2593 | 1.0000 | +0.7407 | no |
| long_context | 26 | 0.3077 | 1.0000 | +0.6923 | no |
| many_items | 26 | 0.2308 | 0.9615 | +0.7308 | no |

## Per field recall (tolerance 0.1000)

| Field | Gold paths | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| client_name | 160 | 0.8125 | 1.0000 | +0.1875 | no |
| record_date | 160 | 0.7937 | 0.9875 | +0.1938 | no |
| risk_profile | 160 | 0.7625 | 1.0000 | +0.2375 | no |
| objectives | 411 | 0.1387 | 1.0000 | +0.8613 | no |
| recommendations | 1513 | 0.4362 | 0.8691 | +0.4329 | no |
| fees | 320 | 0.7063 | 1.0000 | +0.2937 | no |
| review_months | 160 | 0.7750 | 1.0000 | +0.2250 | no |
| flags | 200 | 0.0000 | 0.0000 | +0.0000 | no |
