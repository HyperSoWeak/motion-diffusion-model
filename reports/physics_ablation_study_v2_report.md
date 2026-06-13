# Physics Ablation Report (g,f,s,r — root=0)

Date: 2026-06-13

## Summary

Collected from `eval/eval_physics.py` on generated `results.npy`.

## Evaluation Setup

```text
eval_root:     save/physics_ablate32_v2
sample_dirs:  {run_id}_samples/
prompts:       reports/physics_eval_prompts_32.txt
sampling:      32 prompts x 2 repetitions = 64 motions (if generate used defaults)
guidance_param: 2.5 (if generate used default)
```

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00973113 | 0.00159209 | 3.209e-06 | 0.00884756 | 0.00670044 |
| ablate_1000 | ground only | 0.00939495 | 0.00203598 | 6.071e-06 | 0.00937122 | 0.00733217 |
| ablate_0100 | foot only | 0.00862300 | 0.00146188 | 5.030e-06 | 0.00829466 | 0.00669989 |
| ablate_0010 | smooth only | 0.00989600 | 0.00048532 | 1.150e-06 | 0.01055 | 0.00744418 |
| ablate_1100 | ground + foot | 0.01013 | 0.00028409 | 5.034e-07 | 0.00945542 | 0.00691683 |
| ablate_1010 | ground + smooth | 0.00798043 | 0.00081084 | 1.185e-06 | 0.00564381 | 0.00735846 |
| ablate_0110 | foot + smooth | 0.00778078 | 0.00093513 | 1.996e-06 | 0.00549965 | 0.00785862 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.00934211 | 0.00119555 | 6.677e-06 | 0.01562 | 0.00717708 |

## Relative to baseline (`ablate_0000`)

Negative % means improvement (metric decreased).

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | -3.5% | +27.9% | +89.2% | +5.9% | +9.4% |
| ablate_0100 | -11.4% | -8.2% | +56.7% | -6.2% | -0.0% |
| ablate_0010 | +1.7% | -69.5% | -64.2% | +19.2% | +11.1% |
| ablate_1100 | +4.1% | -82.2% | -84.3% | +6.9% | +3.2% |
| ablate_1010 | -18.0% | -49.1% | -63.1% | -36.2% | +9.8% |
| ablate_0110 | -20.0% | -41.3% | -37.8% | -37.8% | +17.3% |
| ablate_1110 | -4.0% | -24.9% | +108.1% | +76.6% | +7.1% |

## Best per metric

```text
Foot skating: ablate_0110
Ground penetration rate: ablate_1100
Mean penetration: ablate_1100
Max penetration: ablate_0110
Mean acceleration: ablate_0100
```
