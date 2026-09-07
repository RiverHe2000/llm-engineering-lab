# Promotion gate: Qwen/Qwen2.5-0.5B-Instruct+dpo vs Qwen/Qwen2.5-0.5B-Instruct

**Decision: REJECT** (exit code 1)

- slice clean regressed by 0.2593, tolerance 0.0500
- slice distractor regressed by 0.1481, tolerance 0.0500
- slice mixed_formats regressed by 0.1111, tolerance 0.0500
- slice absent_fields regressed by 0.2593, tolerance 0.0500
- slice many_items regressed by 0.0769, tolerance 0.0500
- the difference CI lower bound -0.2000 does not clear the non-inferiority margin +0.0000

## Headline metrics (n = 160)

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| JSON validity (strict) | 0.7875 | 0.0000 | -0.7875 |
| Schema validity | 0.1437 | 0.0000 | -0.1437 |
| Mean field F1 | 0.5185 | 0.0000 | -0.5185 |
| Exact match | 0.0000 | 0.0000 | +0.0000 |
| Mean reward | 0.4973 | 0.0000 | -0.4973 |

## Paired comparison

| Quantity | Value |
| --- | --- |
| Success difference | -0.1437 |
| Difference CI (95%) | [-0.2000, -0.0938] |
| Field F1 difference | -0.5185 |
| Field F1 CI | [-0.5655, -0.4698] |
| Non-inferiority margin | 0.0000 |
| Non-inferior | no |
| McNemar b / c | 23 / 0 |
| McNemar p (exact) | 2.38e-07 |
| Alpha | 0.0500 |
| Baseline success (Wilson) | 0.1437 [+0.0977, +0.2065] |
| Candidate success (Wilson) | 0.0000 [+0.0000, +0.0234] |

## Floors

| Floor | Required | Candidate | Cleared |
| --- | ---: | ---: | --- |
| JSON validity | 0.0000 | 0.0000 | yes |
| Schema validity | 0.0000 | 0.0000 | yes |

## Per slice (tolerance 0.0500)

| Slice | n | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 27 | 0.2593 | 0.0000 | -0.2593 | yes |
| distractor | 27 | 0.1481 | 0.0000 | -0.1481 | yes |
| mixed_formats | 27 | 0.1111 | 0.0000 | -0.1111 | yes |
| absent_fields | 27 | 0.2593 | 0.0000 | -0.2593 | yes |
| long_context | 26 | 0.0000 | 0.0000 | +0.0000 | no |
| many_items | 26 | 0.0769 | 0.0000 | -0.0769 | yes |
