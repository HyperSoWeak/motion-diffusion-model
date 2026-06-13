# Physics Ablation Report (g,f,s,r — root=0)

Date: 2026-06-01

## Summary

Collected from `eval/eval_physics.py` on generated `results.npy`.

## Evaluation Setup

```text
eval_root:     save/physics_ablate32
sample_dirs:  {run_id}_samples/
prompts:       reports/physics_eval_prompts_32.txt
sampling:      32 prompts x 2 repetitions = 64 motions (if generate used defaults)
guidance_param: 2.5 (if generate used default)
```

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00919910 | 0.00146386 | 2.676e-06 | 0.00884756 | 0.00676417 |
| ablate_1000 | ground only | 0.00892470 | 0.00200837 | 7.422e-06 | 0.01150 | 0.00721877 |
| ablate_0100 | foot only | 0.00860785 | 0.00168087 | 5.442e-06 | 0.00829466 | 0.00676458 |
| ablate_0010 | smooth only | 0.00914514 | 0.00052872 | 1.051e-06 | 0.01055 | 0.00752251 |
| ablate_1100 | ground + foot | 0.00985566 | 0.00024069 | 4.480e-07 | 0.00945542 | 0.00710455 |
| ablate_1010 | ground + smooth | 0.00781646 | 0.00060369 | 1.031e-06 | 0.01324 | 0.00738218 |
| ablate_0110 | foot + smooth | 0.00792051 | 0.00095486 | 2.065e-06 | 0.01369 | 0.00806162 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.00909758 | 0.00090357 | 4.553e-06 | 0.01562 | 0.00721118 |

## Relative to baseline (`ablate_0000`)

Negative % means improvement (metric decreased).

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | -3.0% | +37.2% | +177.3% | +29.9% | +6.7% |
| ablate_0100 | -6.4% | +14.8% | +103.3% | -6.2% | +0.0% |
| ablate_0010 | -0.6% | -63.9% | -60.7% | +19.2% | +11.2% |
| ablate_1100 | +7.1% | -83.6% | -83.3% | +6.9% | +5.0% |
| ablate_1010 | -15.0% | -58.8% | -61.5% | +49.7% | +9.1% |
| ablate_0110 | -13.9% | -34.8% | -22.8% | +54.7% | +19.2% |
| ablate_1110 | -1.1% | -38.3% | +70.1% | +76.6% | +6.6% |

## Best per metric

```text
Foot skating: ablate_1010
Ground penetration rate: ablate_1100
Mean penetration: ablate_1100
Max penetration: ablate_0100
Mean acceleration: ablate_0000
```