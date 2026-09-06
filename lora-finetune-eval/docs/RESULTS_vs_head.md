Test set: n = 340 sentences. Intervals are 95% percentile bootstrap.

| Run | Strategy | LoRA | Trainable params | Accuracy % [95% CI] | Macro-F1 % [95% CI] | ECE | NLL | Best epoch | Train time |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| head_only | head | - | 592,899 (0.89%) | 85.0 [80.9, 88.8] | 79.0 [73.6, 83.9] | 0.039 | 0.350 | 6/6 | 4s |
| lora_r8 | lora | r=8, alpha=16 | 740,355 (1.10%) | 95.9 [93.8, 97.9] | 94.2 [91.0, 97.1] | 0.041 | 0.184 | 5/6 | 6s |
| full_ft | full | - | 66,955,779 (100.00%) | 94.4 [91.8, 96.8] | 92.2 [88.6, 95.6] | 0.033 | 0.207 | 3/5 | 10s |

### Paired comparison vs. `head_only` (McNemar's test)

| Run | Delta accuracy (pts) | baseline right / run wrong | baseline wrong / run right | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| lora_r8 | +10.9 | 1 | 38 | 0.0000 (chi2-corrected) | significant (p < 0.05) |
| full_ft | +9.4 | 4 | 36 | 0.0000 (chi2-corrected) | significant (p < 0.05) |

### Per-class F1 %

| Run | negative | neutral | positive |
|---|---:|---:|---:|
| head_only | 72.5 | 92.9 | 71.5 |
| lora_r8 | 92.3 | 98.1 | 92.3 |
| full_ft | 88.9 | 97.1 | 90.7 |
