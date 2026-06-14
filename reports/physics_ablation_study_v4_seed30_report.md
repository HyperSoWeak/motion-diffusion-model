# Physics Ablation Report (g,f,s,r — root=0)

Date: 2026-06-14

## Summary

Collected from `eval/eval_physics.py` on generated `results.npy`.

## Evaluation Setup

```text
eval_root:     save/physics_ablate32_v4_seed30
sample_dirs:  {run_id}_samples/
prompts:       reports/physics_eval_prompts_32.txt
sampling:      32 prompts x 2 repetitions = 64 motions (if generate used defaults)
guidance_param: 2.5 (if generate used default)
```

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00815662 | 0.00239702 | 4.953e-06 | 0.00939499 | 0.00648395 |
| ablate_1000 | ground only | 0.01164 | 0.00997277 | 2.391e-05 | 0.01173 | 0.00599116 |
| ablate_0100 | foot only | 0.01027 | 0.00445668 | 8.485e-06 | 0.01135 | 0.00657200 |
| ablate_0010 | smooth only | 0.01254 | 0.00670573 | 1.713e-05 | 0.01223 | 0.00677318 |
| ablate_1100 | ground + foot | 0.01059 | 0.00639205 | 1.728e-05 | 0.00824825 | 0.00650097 |
| ablate_1010 | ground + smooth | 0.01287 | 0.00754025 | 2.094e-05 | 0.01178 | 0.00655515 |
| ablate_0110 | foot + smooth | 0.01055 | 0.00358665 | 7.112e-06 | 0.01567 | 0.00669573 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.01034 | 0.00314276 | 6.964e-06 | 0.01595 | 0.00634013 |

## Relative to baseline (`ablate_0000`)

Negative % means improvement (metric decreased).

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | +42.7% | +316.0% | +382.8% | +24.9% | -7.6% |
| ablate_0100 | +25.8% | +85.9% | +71.3% | +20.8% | +1.4% |
| ablate_0010 | +53.8% | +179.8% | +245.9% | +30.1% | +4.5% |
| ablate_1100 | +29.8% | +166.7% | +249.0% | -12.2% | +0.3% |
| ablate_1010 | +57.8% | +214.6% | +322.7% | +25.4% | +1.1% |
| ablate_0110 | +29.3% | +49.6% | +43.6% | +66.8% | +3.3% |
| ablate_1110 | +26.7% | +31.1% | +40.6% | +69.8% | -2.2% |

## Best per metric

```text
Foot skating: ablate_0000
Ground penetration rate: ablate_0000
Mean penetration: ablate_0000
Max penetration: ablate_1100
Mean acceleration: ablate_1000
```
