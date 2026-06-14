# Physics Ablation Report (g,f,s,r — root=0)

Date: 2026-06-14

## Summary

Collected from `eval/eval_physics.py` on generated `results.npy`.

## Evaluation Setup

```text
eval_root:     save/physics_ablate32_v4_seed40
sample_dirs:  {run_id}_samples/
prompts:       reports/physics_eval_prompts_32.txt
sampling:      32 prompts x 2 repetitions = 64 motions (if generate used defaults)
guidance_param: 2.5 (if generate used default)
```

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00880068 | 0.00356297 | 1.461e-05 | 0.00981512 | 0.00686346 |
| ablate_1000 | ground only | 0.00944884 | 0.00160393 | 3.184e-06 | 0.00736848 | 0.00575743 |
| ablate_0100 | foot only | 0.00884759 | 0.00066288 | 1.177e-06 | 0.00898139 | 0.00658039 |
| ablate_0010 | smooth only | 0.01024 | 0.00145005 | 2.983e-06 | 0.01703 | 0.00633484 |
| ablate_1100 | ground + foot | 0.00796633 | 0.00150331 | 3.374e-06 | 0.01751 | 0.00574971 |
| ablate_1010 | ground + smooth | 0.00770540 | 0.00195904 | 6.385e-06 | 0.00892320 | 0.00521669 |
| ablate_0110 | foot + smooth | 0.01040 | 0.00197680 | 8.153e-06 | 0.01934 | 0.00604486 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.00703161 | 0.00174006 | 3.379e-06 | 0.01484 | 0.00516183 |

## Relative to baseline (`ablate_0000`)

Negative % means improvement (metric decreased).

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | +7.4% | -55.0% | -78.2% | -24.9% | -16.1% |
| ablate_0100 | +0.5% | -81.4% | -91.9% | -8.5% | -4.1% |
| ablate_0010 | +16.3% | -59.3% | -79.6% | +73.5% | -7.7% |
| ablate_1100 | -9.5% | -57.8% | -76.9% | +78.4% | -16.2% |
| ablate_1010 | -12.4% | -45.0% | -56.3% | -9.1% | -24.0% |
| ablate_0110 | +18.2% | -44.5% | -44.2% | +97.1% | -11.9% |
| ablate_1110 | -20.1% | -51.2% | -76.9% | +51.2% | -24.8% |

## Best per metric

```text
Foot skating: ablate_1110
Ground penetration rate: ablate_0100
Mean penetration: ablate_0100
Max penetration: ablate_1000
Mean acceleration: ablate_1110
```
