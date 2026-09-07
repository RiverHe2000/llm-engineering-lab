# Promotion gate: Qwen/Qwen2.5-0.5B-Instruct+dpo vs Qwen/Qwen2.5-0.5B-Instruct+sft

**Decision: REJECT** (exit code 1)

- field 'flags' lost 0.6100 of its recall (0.6100 to 0.0000 over 200 gold paths), tolerance 0.1000

## Headline metrics (n = 160)

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| JSON validity (strict) | 0.9750 | 1.0000 | +0.0250 |
| Schema validity | 0.8313 | 0.9875 | +0.1562 |
| Mean field F1 | 0.8914 | 0.9133 | +0.0219 |
| Exact match | 0.4375 | 0.0500 | -0.3875 |
| Mean reward | 0.8961 | 0.9455 | +0.0494 |

## Paired comparison

| Quantity | Value |
| --- | --- |
| Success difference | +0.1562 |
| Difference CI (95%) | [+0.1000, +0.2125] |
| Field F1 difference | +0.0219 |
| Field F1 CI | [-0.0027, +0.0514] |
| Non-inferiority margin | 0.0200 |
| Non-inferior | yes |
| McNemar b / c | 1 / 26 |
| McNemar p (exact) | 4.17e-07 |
| Alpha | 0.0500 |
| Baseline success (Wilson) | 0.8313 [+0.7656, +0.8814] |
| Candidate success (Wilson) | 0.9875 [+0.9556, +0.9966] |

## Floors

| Floor | Required | Candidate | Cleared |
| --- | ---: | ---: | --- |
| JSON validity | 0.9500 | 1.0000 | yes |
| Schema validity | 0.0000 | 0.9875 | yes |

## Per slice (tolerance 0.0500)

| Slice | n | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 27 | 0.9630 | 0.9630 | +0.0000 | no |
| distractor | 27 | 0.7407 | 1.0000 | +0.2593 | no |
| mixed_formats | 27 | 0.7407 | 1.0000 | +0.2593 | no |
| absent_fields | 27 | 0.9630 | 1.0000 | +0.0370 | no |
| long_context | 26 | 0.7692 | 1.0000 | +0.2308 | no |
| many_items | 26 | 0.8077 | 0.9615 | +0.1538 | no |

## Per field recall (tolerance 0.1000)

| Field | Gold paths | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| client_name | 160 | 0.9750 | 1.0000 | +0.0250 | no |
| record_date | 160 | 0.9313 | 0.9875 | +0.0563 | no |
| risk_profile | 160 | 0.9688 | 1.0000 | +0.0312 | no |
| objectives | 411 | 0.9562 | 1.0000 | +0.0438 | no |
| recommendations | 1513 | 0.8387 | 0.8691 | +0.0304 | no |
| fees | 320 | 0.8938 | 1.0000 | +0.1062 | no |
| review_months | 160 | 0.9375 | 1.0000 | +0.0625 | no |
| flags | 200 | 0.6100 | 0.0000 | -0.6100 | yes |
