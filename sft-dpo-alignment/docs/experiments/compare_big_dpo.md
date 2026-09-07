# Promotion gate: Qwen/Qwen2.5-0.5B-Instruct+dpo vs big-prompted

**Decision: PROMOTE** (exit code 0)

- the paired difference is +0.2250 with a 95% CI of [+0.1625, +0.2938], which excludes zero
- McNemar p = 2.84e-10 clears alpha = 0.0500
- no floor was breached, no slice regressed and no field lost recall

## Headline metrics (n = 160)

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| JSON validity (strict) | 0.7625 | 1.0000 | +0.2375 |
| Schema validity | 0.7625 | 0.9875 | +0.2250 |
| Mean field F1 | 0.7122 | 0.9133 | +0.2011 |
| Exact match | 0.0500 | 0.0500 | +0.0000 |
| Mean reward | 0.7323 | 0.9455 | +0.2132 |

## Paired comparison

| Quantity | Value |
| --- | --- |
| Success difference | +0.2250 |
| Difference CI (95%) | [+0.1625, +0.2938] |
| Field F1 difference | +0.2011 |
| Field F1 CI | [+0.1407, +0.2625] |
| Non-inferiority margin | 0.0200 |
| Non-inferior | yes |
| McNemar b / c | 1 / 37 |
| McNemar p (exact) | 2.84e-10 |
| Alpha | 0.0500 |
| Baseline success (Wilson) | 0.7625 [+0.6909, +0.8218] |
| Candidate success (Wilson) | 0.9875 [+0.9556, +0.9966] |

## Floors

| Floor | Required | Candidate | Cleared |
| --- | ---: | ---: | --- |
| JSON validity | 0.9500 | 1.0000 | yes |
| Schema validity | 0.0000 | 0.9875 | yes |

## Per slice (tolerance 0.0500)

| Slice | n | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 27 | 1.0000 | 0.9630 | -0.0370 | no |
| distractor | 27 | 1.0000 | 1.0000 | +0.0000 | no |
| mixed_formats | 27 | 1.0000 | 1.0000 | +0.0000 | no |
| absent_fields | 27 | 1.0000 | 1.0000 | +0.0000 | no |
| long_context | 26 | 0.4615 | 1.0000 | +0.5385 | no |
| many_items | 26 | 0.0769 | 0.9615 | +0.8846 | no |

## Per field recall (tolerance 0.1000)

| Field | Gold paths | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| client_name | 160 | 0.7625 | 1.0000 | +0.2375 | no |
| record_date | 160 | 0.7625 | 0.9875 | +0.2250 | no |
| risk_profile | 160 | 0.7625 | 1.0000 | +0.2375 | no |
| objectives | 411 | 0.7153 | 1.0000 | +0.2847 | no |
| recommendations | 1513 | 0.6147 | 0.8691 | +0.2545 | no |
| fees | 320 | 0.7438 | 1.0000 | +0.2562 | no |
| review_months | 160 | 0.7625 | 1.0000 | +0.2375 | no |
| flags | 200 | 0.0150 | 0.0000 | -0.0150 | no |
