# Alignment tax: Qwen/Qwen2.5-0.5B-Instruct+aligned vs Qwen/Qwen2.5-0.5B-Instruct

| Quantity | Value |
| --- | --- |
| Probes | 12 |
| Pass rate before | 0.5833 |
| Pass rate after | 0.4167 |
| Difference | -0.1667 |
| Difference CI (95%) | [-0.4167, +0.0000] |
| Probes lost / gained | 2 / 0 |
| McNemar p (exact) | 0.5000 |
| Within margin 0.0500 | no |

A drop here is a cost of the alignment method, not a fault in the harness: training a small model hard on one output format is expected to reduce general instruction-following. It is reported so the lift on the task is quoted with its price attached.
