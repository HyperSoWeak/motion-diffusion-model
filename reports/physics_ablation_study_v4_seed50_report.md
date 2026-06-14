# Physics Ablation Report (g,f,s,r — root=0)

Date: 2026-06-14

## Summary

Collected from `eval/eval_physics.py` on generated `results.npy`.

## Evaluation Setup

```text
eval_root:     save/physics_ablate32_v4_seed50
sample_dirs:  {run_id}_samples/
prompts:       reports/physics_eval_prompts_32.txt
sampling:      32 prompts x 2 repetitions = 64 motions (if generate used defaults)
guidance_param: 2.5 (if generate used default)
```

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00956388 | 0.00240885 | 8.222e-06 | 0.03597 | 0.00719313 |
| ablate_1000 | ground only | 0.01122 | 0.00250947 | 3.925e-06 | 0.00714195 | 0.00557674 |
| ablate_0100 | foot only | 0.01309 | 0.00466383 | 1.352e-05 | 0.01119 | 0.00642911 |
| ablate_0010 | smooth only | 0.01125 | 0.00346828 | 7.597e-06 | 0.00819531 | 0.00651108 |
| ablate_1100 | ground + foot | 0.01245 | 0.00296520 | 4.039e-06 | 0.00848819 | 0.00602017 |
| ablate_1010 | ground + smooth | 0.01043 | 0.00682410 | 1.358e-05 | 0.00776204 | 0.00520747 |
| ablate_0110 | foot + smooth | 0.01218 | 0.00337358 | 8.785e-06 | 0.01006 | 0.00613138 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.01090 | 0.00497751 | 7.847e-06 | 0.00830917 | 0.00562933 |

## Relative to baseline (`ablate_0000`)

Negative % means improvement (metric decreased).

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | +17.4% | +4.2% | -52.3% | -80.1% | -22.5% |
| ablate_0100 | +36.9% | +93.6% | +64.5% | -68.9% | -10.6% |
| ablate_0010 | +17.6% | +44.0% | -7.6% | -77.2% | -9.5% |
| ablate_1100 | +30.1% | +23.1% | -50.9% | -76.4% | -16.3% |
| ablate_1010 | +9.1% | +183.3% | +65.1% | -78.4% | -27.6% |
| ablate_0110 | +27.4% | +40.0% | +6.9% | -72.0% | -14.8% |
| ablate_1110 | +13.9% | +106.6% | -4.6% | -76.9% | -21.7% |

## Best per metric

```text
Foot skating: ablate_0000
Ground penetration rate: ablate_0000
Mean penetration: ablate_1000
Max penetration: ablate_1000
Mean acceleration: ablate_1010
```
