# Promotion gate: Qwen/Qwen2.5-0.5B-Instruct+dpo vs Qwen/Qwen2.5-0.5B-Instruct+sft

**Decision: REJECT** (exit code 1)

- slice clean regressed by 0.7778, tolerance 0.0500
- slice distractor regressed by 0.8148, tolerance 0.0500
- slice mixed_formats regressed by 0.6296, tolerance 0.0500
- slice absent_fields regressed by 0.8889, tolerance 0.0500
- slice long_context regressed by 0.8846, tolerance 0.0500
- slice many_items regressed by 0.9615, tolerance 0.0500
- the difference CI lower bound -0.8812 does not clear the non-inferiority margin +0.0000

## Headline metrics (n = 160)

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| JSON validity (strict) | 0.9812 | 0.0000 | -0.9812 |
| Schema validity | 0.8250 | 0.0000 | -0.8250 |
| Mean field F1 | 0.8473 | 0.0000 | -0.8473 |
| Exact match | 0.1500 | 0.0000 | -0.1500 |
| Mean reward | 0.8696 | 0.0000 | -0.8696 |

## Paired comparison

| Quantity | Value |
| --- | --- |
| Success difference | -0.8250 |
| Difference CI (95%) | [-0.8812, -0.7688] |
| Field F1 difference | -0.8473 |
| Field F1 CI | [-0.8756, -0.8176] |
| Non-inferiority margin | 0.0000 |
| Non-inferior | no |
| McNemar b / c | 132 / 0 |
| McNemar p (exact) | 3.67e-40 |
| Alpha | 0.0500 |
| Baseline success (Wilson) | 0.8250 [+0.7587, +0.8761] |
| Candidate success (Wilson) | 0.0000 [+0.0000, +0.0234] |

## Floors

| Floor | Required | Candidate | Cleared |
| --- | ---: | ---: | --- |
| JSON validity | 0.0000 | 0.0000 | yes |
| Schema validity | 0.0000 | 0.0000 | yes |

## Per slice (tolerance 0.0500)

| Slice | n | Baseline | Candidate | Delta | Regressed |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 27 | 0.7778 | 0.0000 | -0.7778 | yes |
| distractor | 27 | 0.8148 | 0.0000 | -0.8148 | yes |
| mixed_formats | 27 | 0.6296 | 0.0000 | -0.6296 | yes |
| absent_fields | 27 | 0.8889 | 0.0000 | -0.8889 | yes |
| long_context | 26 | 0.8846 | 0.0000 | -0.8846 | yes |
| many_items | 26 | 0.9615 | 0.0000 | -0.9615 | yes |
