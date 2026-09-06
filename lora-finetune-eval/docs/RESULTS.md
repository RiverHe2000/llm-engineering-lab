Test set: n = 340 sentences. Intervals are 95% percentile bootstrap.

| Run | Strategy | LoRA | Trainable params | Accuracy % [95% CI] | Macro-F1 % [95% CI] | ECE | NLL | Best epoch | Train time |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| head_only | head | - | 592,899 (0.89%) | 85.0 [80.9, 88.8] | 79.0 [73.6, 83.9] | 0.039 | 0.350 | 6/6 | 4s |
| lora_r4 | lora | r=4, alpha=8 | 666,627 (0.99%) | 95.6 [93.2, 97.6] | 93.9 [90.6, 96.9] | 0.033 | 0.184 | 5/6 | 6s |
| lora_r8 | lora | r=8, alpha=16 | 740,355 (1.10%) | 95.9 [93.8, 97.9] | 94.2 [91.0, 97.1] | 0.041 | 0.184 | 5/6 | 6s |
| lora_r16 | lora | r=16, alpha=32 | 887,811 (1.32%) | 96.2 [94.1, 97.9] | 94.5 [91.4, 97.3] | 0.045 | 0.188 | 5/6 | 6s |
| lora_r8_all_linear | lora | r=8, alpha=16 | 1,256,451 (1.86%) | 94.4 [91.8, 96.8] | 91.9 [88.1, 95.2] | 0.045 | 0.219 | 5/6 | 9s |
| full_ft | full | - | 66,955,779 (100.00%) | 94.4 [91.8, 96.8] | 92.2 [88.6, 95.6] | 0.033 | 0.207 | 3/5 | 10s |

### Paired comparison vs. `full_ft` (McNemar's test)

| Run | Delta accuracy (pts) | baseline right / run wrong | baseline wrong / run right | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| head_only | -9.4 | 36 | 4 | 0.0000 (chi2-corrected) | significant (p < 0.05) |
| lora_r4 | +1.2 | 2 | 6 | 0.2891 (exact) | not significant |
| lora_r8 | +1.5 | 1 | 6 | 0.1250 (exact) | not significant |
| lora_r16 | +1.8 | 0 | 6 | 0.0312 (exact) | significant (p < 0.05) |
| lora_r8_all_linear | +0.0 | 5 | 5 | 1.0000 (exact) | not significant |

### Per-class F1 %

| Run | negative | neutral | positive |
|---|---:|---:|---:|
| head_only | 72.5 | 92.9 | 71.5 |
| lora_r4 | 92.3 | 97.9 | 91.7 |
| lora_r8 | 92.3 | 98.1 | 92.3 |
| lora_r16 | 92.3 | 98.3 | 92.9 |
| lora_r8_all_linear | 88.2 | 97.6 | 89.9 |
| full_ft | 88.9 | 97.1 | 90.7 |
