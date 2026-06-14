# Physics Ablation 5-Seed Report

Date: 2026-06-14

## Setup

```text
seeds:        10, 20, 30, 40, 50
seed10 root:  save/physics_ablate32_v4_full_rerun
seed20 root:  save/physics_ablate32_v4_seed20
seed30 root:  save/physics_ablate32_v4_seed30
seed40 root:  save/physics_ablate32_v4_seed40
seed50 root:  save/physics_ablate32_v4_seed50
sampling:     32 prompts x 2 repetitions = 64 motions per seed
guidance:     2.5
device:       seed10 original run, seed20/30/40/50 rerun with run_ablation.sh params
```

## Mean +- Std

Lower is better for all metrics below.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00908686 +- 0.00055982 | 0.00245502 +- 0.00063343 | 7.988e-06 +- 3.924e-06 | 0.01937 +- 0.01233 | 0.00685254 +- 0.00024286 |
| ablate_1000 | ground only | 0.01002 +- 0.00137670 | 0.00354167 +- 0.00326886 | 1.159e-05 +- 8.386e-06 | 0.02892 +- 0.03676 | 0.00596427 +- 0.00049099 |
| ablate_0100 | foot only | 0.01011 +- 0.00157088 | 0.00237808 +- 0.00179343 | 5.486e-06 +- 4.835e-06 | 0.00919170 +- 0.00181909 | 0.00651756 +- 0.00015411 |
| ablate_0010 | smooth only | 0.01001 +- 0.00182093 | 0.00340909 +- 0.00185135 | 7.935e-06 +- 4.890e-06 | 0.01074 +- 0.00354593 | 0.00618959 +- 0.00056436 |
| ablate_1100 | ground + foot | 0.01011 +- 0.00153313 | 0.00267637 +- 0.00206741 | 6.839e-06 +- 5.867e-06 | 0.01549 +- 0.01148 | 0.00626034 +- 0.00033434 |
| ablate_1010 | ground + smooth | 0.01001 +- 0.00168143 | 0.00369555 +- 0.00287432 | 1.051e-05 +- 6.003e-06 | 0.01838 +- 0.01836 | 0.00607156 +- 0.00077827 |
| ablate_0110 | foot + smooth | 0.01050 +- 0.00116831 | 0.00241122 +- 0.00096781 | 6.519e-06 +- 1.981e-06 | 0.01275 +- 0.00406658 | 0.00631577 +- 0.00046521 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.00955202 +- 0.00147715 | 0.00306581 +- 0.00141291 | 8.594e-06 +- 6.957e-06 | 0.01293 +- 0.00512178 | 0.00610226 +- 0.00063261 |

## Best by Mean

```text
Foot skating: ablate_0000 (0.00908686 +- 0.00055982)
Ground penetration rate: ablate_0100 (0.00237808 +- 0.00179343)
Mean penetration: ablate_0100 (5.486e-06 +- 4.835e-06)
Max penetration: ablate_0100 (0.00919170 +- 0.00181909)
Mean acceleration: ablate_1000 (0.00596427 +- 0.00049099)
```

## Mean Relative to Baseline

Negative % means improvement relative to baseline mean.

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | +10.3% | +44.3% | +45.0% | +49.3% | -13.0% |
| ablate_0100 | +11.2% | -3.1% | -31.3% | -52.5% | -4.9% |
| ablate_0010 | +10.2% | +38.9% | -0.7% | -44.6% | -9.7% |
| ablate_1100 | +11.3% | +9.0% | -14.4% | -20.0% | -8.6% |
| ablate_1010 | +10.2% | +50.5% | +31.6% | -5.1% | -11.4% |
| ablate_0110 | +15.6% | -1.8% | -18.4% | -34.2% | -7.8% |
| ablate_1110 | +5.1% | +24.9% | +7.6% | -33.3% | -10.9% |

## Interpretation

- Seed variance remains large; single-seed ranking is not reliable.
- Best mean foot skating: `ablate_0000` (baseline (g,f,s,r off)).
- Best mean ground penetration rate: `ablate_0100` (foot only).
- Best mean mean penetration: `ablate_0100` (foot only).
- Best mean max penetration: `ablate_0100` (foot only).
- Best mean mean acceleration: `ablate_1000` (ground only).
- `ablate_1110` is not the best mean on any listed metric in this 5-seed aggregate.
- Relative to baseline, the fine-tuned losses show metric-dependent tradeoffs rather than a uniform improvement.
- The all-on setting should not be reported as a robust overall winner unless the report explicitly prioritizes a metric where it wins.

