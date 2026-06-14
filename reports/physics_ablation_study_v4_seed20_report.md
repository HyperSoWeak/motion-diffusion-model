# Physics Ablation Report (g,f,s,r — root=0)

Date: 2026-06-14

## Summary

Collected from `eval/eval_physics.py` on generated `results.npy`.

## Evaluation Setup

```text
eval_root:     save/physics_ablate32_v4_seed20
sample_dirs:  {run_id}_samples/
prompts:       reports/physics_eval_prompts_32.txt
sampling:      32 prompts x 2 repetitions = 64 motions (if generate used defaults)
guidance_param: 2.5 (if generate used default)
```

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00921281 | 0.00232008 | 8.959e-06 | 0.03287 | 0.00700734 |
| ablate_1000 | ground only | 0.01002 | 0.00260417 | 7.819e-06 | 0.01621 | 0.00559584 |
| ablate_0100 | foot only | 0.00922933 | 0.00126065 | 3.158e-06 | 0.00742388 | 0.00627555 |
| ablate_0010 | smooth only | 0.00857268 | 0.00356297 | 7.142e-06 | 0.00745128 | 0.00512781 |
| ablate_1100 | ground + foot | 0.01057 | 0.00228456 | 8.903e-06 | 0.03707 | 0.00667736 |
| ablate_1010 | ground + smooth | 0.00950934 | 0.00100024 | 6.620e-06 | 0.05499 | 0.00616288 |
| ablate_0110 | foot + smooth | 0.01085 | 0.00216619 | 3.445e-06 | 0.00906835 | 0.00571021 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.01076 | 0.00420810 | 2.195e-05 | 0.01973 | 0.00642034 |

## Relative to baseline (`ablate_0000`)

Negative % means improvement (metric decreased).

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | +8.7% | +12.2% | -12.7% | -50.7% | -20.1% |
| ablate_0100 | +0.2% | -45.7% | -64.8% | -77.4% | -10.4% |
| ablate_0010 | -6.9% | +53.6% | -20.3% | -77.3% | -26.8% |
| ablate_1100 | +14.7% | -1.5% | -0.6% | +12.8% | -4.7% |
| ablate_1010 | +3.2% | -56.9% | -26.1% | +67.3% | -12.1% |
| ablate_0110 | +17.7% | -6.6% | -61.5% | -72.4% | -18.5% |
| ablate_1110 | +16.8% | +81.4% | +145.0% | -40.0% | -8.4% |

## Best per metric

```text
Foot skating: ablate_0010
Ground penetration rate: ablate_1010
Mean penetration: ablate_0100
Max penetration: ablate_0100
Mean acceleration: ablate_0010
```
