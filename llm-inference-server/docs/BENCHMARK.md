## distilgpt2 (82M) on RTX 4070, bf16

model: distilbert/distilgpt2, device: cuda, dtype: bfloat16, torch: 2.11.0+cu128, python: 3.12.0, platform: Windows-11-10.0.26200-SP0, gpu: NVIDIA GeForce RTX 4070

| Batch | Prompt tok | New tok/req | Tokens/s | Batch p50 (ms) | Batch p95 (ms) | ms/token/request | Speed-up vs batch 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 64 | 362.6 | 175.2 | 184.2 | 2.74 | 1.00x |
| 2 | 9 | 64 | 650.3 | 195.0 | 206.1 | 3.05 | 1.79x |
| 4 | 9 | 64 | 1,243.6 | 206.9 | 216.8 | 3.23 | 3.43x |
| 8 | 9 | 64 | 2,448.0 | 198.3 | 230.8 | 3.10 | 6.75x |
| 16 | 9 | 64 | 3,857.0 | 268.6 | 303.7 | 4.20 | 10.64x |
| 32 | 9 | 64 | 8,922.0 | 230.2 | 241.7 | 3.60 | 24.61x |

## Qwen2.5-0.5B-Instruct (494M) on RTX 4070, bf16

model: Qwen/Qwen2.5-0.5B-Instruct, device: cuda, dtype: bfloat16, torch: 2.11.0+cu128, python: 3.12.0, platform: Windows-11-10.0.26200-SP0, gpu: NVIDIA GeForce RTX 4070

| Batch | Prompt tok | New tok/req | Tokens/s | Batch p50 (ms) | Batch p95 (ms) | ms/token/request | Speed-up vs batch 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 64 | 48.5 | 1,342.1 | 1,396.6 | 20.97 | 1.00x |
| 2 | 9 | 64 | 92.3 | 1,362.6 | 1,429.5 | 21.29 | 1.90x |
| 4 | 9 | 64 | 187.8 | 1,340.8 | 1,416.9 | 20.95 | 3.87x |
| 8 | 9 | 64 | 378.8 | 1,314.5 | 1,450.2 | 20.54 | 7.81x |
| 16 | 9 | 64 | 755.1 | 1,334.5 | 1,404.0 | 20.85 | 15.56x |
| 32 | 9 | 64 | 1,408.1 | 1,434.3 | 1,512.9 | 22.41 | 29.02x |

## distilgpt2 on CPU, fp32

model: distilbert/distilgpt2, device: cpu, dtype: float32, torch: 2.11.0+cu128, python: 3.12.0, platform: Windows-11-10.0.26200-SP0

| Batch | Prompt tok | New tok/req | Tokens/s | Batch p50 (ms) | Batch p95 (ms) | ms/token/request | Speed-up vs batch 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 32 | 114.2 | 280.3 | 280.8 | 8.76 | 1.00x |
| 4 | 9 | 32 | 346.8 | 369.1 | 375.0 | 11.53 | 3.04x |
| 8 | 9 | 32 | 582.1 | 439.8 | 446.7 | 13.74 | 5.10x |

## distilgpt2 on CPU, dynamic INT8

model: distilbert/distilgpt2, device: cpu, dtype: float32, torch: 2.11.0+cu128, python: 3.12.0, platform: Windows-11-10.0.26200-SP0, quantization: dynamic int8

| Batch | Prompt tok | New tok/req | Tokens/s | Batch p50 (ms) | Batch p95 (ms) | ms/token/request | Speed-up vs batch 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 32 | 145.2 | 220.4 | 222.8 | 6.89 | 1.00x |
| 4 | 9 | 32 | 441.5 | 289.9 | 292.7 | 9.06 | 3.04x |
| 8 | 9 | 32 | 793.0 | 322.8 | 328.1 | 10.09 | 5.46x |

