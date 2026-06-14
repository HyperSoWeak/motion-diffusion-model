# Physics Ablation Report (g,f,s,r — root=0)

Date: 2026-06-14

## Summary

Collected from `eval/eval_physics.py` on generated `results.npy`.

## Evaluation Setup

```text
eval_root:     save/physics_ablate32_v4_full_rerun
sample_dirs:  {run_id}_samples/
prompts:       reports/physics_eval_prompts_32.txt
sampling:      32 prompts x 2 repetitions = 64 motions
guidance_param: 2.5
seed:          10
device:        1
visualization: skipped
base_ckpt:     save/humanml_enc_512_50steps/model000750000.pt
fine_tune:     3068 steps to model000753836.pt
```

## Run Notes

- This rerun used the same ablation settings as `run_ablation.sh`, but with `RUN_TAG=v4_full_rerun` so existing outputs were not overwritten.
- GPU 0 hit CUDA OOM because another user's VLLM process occupied most memory, so the full rerun was executed on GPU 1.
- Dataset loaded by the scripts: HumanML3D / `t2m`, using `./dataset/humanml_opt.txt`.
- `ablate_1110` in this rerun matches the previous explicit-parameter 1110 retest values: `seed=10`, `num_repetitions=2`, `guidance_param=2.5`, `skip_visualization`.

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run | Label | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---|---:|---:|---:|---:|---:|
| ablate_0000 | baseline (g,f,s,r off) | 0.00970032 | 0.00158617 | 3.199e-06 | 0.00880124 | 0.00671480 |
| ablate_1000 | ground only | 0.00776613 | 0.00101799 | 1.908e-05 | 0.10213 | 0.00690020 |
| ablate_0100 | foot only | 0.00909174 | 0.00084635 | 1.089e-06 | 0.00701572 | 0.00673075 |
| ablate_0010 | smooth only | 0.00745618 | 0.00185843 | 4.824e-06 | 0.00878460 | 0.00620103 |
| ablate_1100 | ground + foot | 0.00898591 | 0.00023674 | 5.985e-07 | 0.00611912 | 0.00635351 |
| ablate_1010 | ground + smooth | 0.00955057 | 0.00115412 | 5.040e-06 | 0.00843611 | 0.00721563 |
| ablate_0110 | foot + smooth | 0.00853688 | 0.00095289 | 5.098e-06 | 0.00960381 | 0.00699668 |
| ablate_1110 | ground + foot + smooth (r=0) | 0.00873336 | 0.00126065 | 2.834e-06 | 0.00580553 | 0.00695965 |

## Relative to baseline (`ablate_0000`)

Negative % means improvement (metric decreased).

| Run | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
|---|---:|---:|---:|---:|---:|
| ablate_1000 | -19.9% | -35.8% | +496.6% | +1060.4% | +2.8% |
| ablate_0100 | -6.3% | -46.6% | -65.9% | -20.3% | +0.2% |
| ablate_0010 | -23.1% | +17.2% | +50.8% | -0.2% | -7.7% |
| ablate_1100 | -7.4% | -85.1% | -81.3% | -30.5% | -5.4% |
| ablate_1010 | -1.5% | -27.2% | +57.6% | -4.1% | +7.5% |
| ablate_0110 | -12.0% | -39.9% | +59.4% | +9.1% | +4.2% |
| ablate_1110 | -10.0% | -20.5% | -11.4% | -34.0% | +3.6% |

## Best per metric

```text
Foot skating: ablate_0010
Ground penetration rate: ablate_1100
Mean penetration: ablate_1100
Max penetration: ablate_1110
Mean acceleration: ablate_0010
```

## Interpretation

The physics losses do improve several target metrics relative to baseline, but the all-on setting is not uniformly best.

- `ablate_1110` improves over baseline on foot skating, ground penetration rate, mean penetration, and max penetration, but worsens mean acceleration slightly.
- `ablate_1110` is best only for max penetration in this rerun.
- `ablate_1100` is best for ground penetration rate and mean penetration, so adding `smooth` on top of `ground+foot` appears to hurt the average/rate penetration metrics here.
- `ablate_0010` is best for foot skating and mean acceleration, but it worsens ground penetration rate and mean penetration.
- The result supports a tradeoff interpretation: the added losses are useful, but the best lambda/combination depends on the metric; all-on is not a Pareto winner for every physics metric in this ablation.
