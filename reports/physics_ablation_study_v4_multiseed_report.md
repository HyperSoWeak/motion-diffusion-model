# Physics Ablation Multi-Seed Report

Date: 2026-06-14

## Setup

```text
seeds:        10, 20, 30
seed10 root:  save/physics_ablate32_v4_full_rerun
seed20 root:  save/physics_ablate32_v4_seed20
seed30 root:  save/physics_ablate32_v4_seed30
sampling:     32 prompts x 2 repetitions = 64 motions per seed
guidance:     2.5
```

## Mean +- Std

Lower is better for all metrics below.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00902325 +- 0.00078911 | 0.00210109 +- 0.00044759 | 5.704e-06 +- 2.953e-06 | 0.01702 +- 0.01373 | 0.00673536 +- 0.00026230 |
| ablate_1000 | ground only | 0.00980664 +- 0.00194449 | 0.00453164 +- 0.00477843 | 1.694e-05 +- 8.259e-06 | 0.04336 +- 0.05095 | 0.00616240 +- 0.00066883 |
| ablate_0100 | foot only | 0.00952871 +- 0.00064139 | 0.00218789 +- 0.00197571 | 4.244e-06 +- 3.816e-06 | 0.00859623 +- 0.00239276 | 0.00652610 +- 0.00023104 |
| ablate_0010 | smooth only | 0.00952352 +- 0.00267277 | 0.00404238 +- 0.00245895 | 9.699e-06 +- 6.541e-06 | 0.00948734 +- 0.00246376 | 0.00603401 +- 0.00083530 |
| ablate_1100 | ground + foot | 0.01005 +- 0.00091942 | 0.00297112 +- 0.00313456 | 8.928e-06 +- 8.342e-06 | 0.01715 +- 0.01729 | 0.00651061 +- 0.00016214 |
| ablate_1010 | ground + smooth | 0.01064 +- 0.00192918 | 0.00323153 +- 0.00373225 | 1.087e-05 +- 8.757e-06 | 0.02507 +- 0.02596 | 0.00664455 +- 0.00053204 |
| ablate_0110 | foot + smooth | 0.00997760 +- 0.00125663 | 0.00223524 +- 0.00131824 | 5.218e-06 +- 1.836e-06 | 0.01145 +- 0.00366710 | 0.00646754 +- 0.00067291 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.00994416 +- 0.00106997 | 0.00287050 +- 0.00149246 | 1.058e-05 +- 1.006e-05 | 0.01383 +- 0.00720141 | 0.00657337 +- 0.00033692 |

## Best by Mean

```text
Foot skating: ablate_0000 (0.00902325 +- 0.00078911)
Ground penetration rate: ablate_0000 (0.00210109 +- 0.00044759)
Mean penetration: ablate_0100 (4.244e-06 +- 3.816e-06)
Max penetration: ablate_0100 (0.00859623 +- 0.00239276)
Mean acceleration: ablate_0010 (0.00603401 +- 0.00083530)
```

## Mean Relative to Baseline

Negative % means improvement relative to baseline mean.

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | +8.7% | +115.7% | +197.0% | +154.7% | -8.5% |
| ablate_0100 | +5.6% | +4.1% | -25.6% | -49.5% | -3.1% |
| ablate_0010 | +5.5% | +92.4% | +70.1% | -44.3% | -10.4% |
| ablate_1100 | +11.4% | +41.4% | +56.5% | +0.7% | -3.3% |
| ablate_1010 | +18.0% | +53.8% | +90.5% | +47.3% | -1.3% |
| ablate_0110 | +10.6% | +6.4% | -8.5% | -32.7% | -4.0% |
| ablate_1110 | +10.2% | +36.6% | +85.5% | -18.8% | -2.4% |

## Interpretation

- Seed variance is large enough that single-seed ranking is not reliable.
- Across seeds 10/20/30, baseline has the best mean foot skating and ground penetration rate.
- Among fine-tuned ablations, `ablate_0100` has the best mean penetration and max penetration, while `ablate_0010` has the best mean acceleration.
- `ablate_1110` is not the best mean on any listed metric in this 3-seed aggregate.
- Relative to baseline mean, the fine-tuned losses mostly improve max penetration and mean acceleration, but do not robustly improve foot skating or ground penetration rate in this 3-seed run.
- The all-on setting is not a robust winner; this result supports reporting the losses as metric-dependent tradeoffs rather than a uniform improvement.
