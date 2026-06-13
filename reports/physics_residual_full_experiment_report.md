# Physics Residual Fine-Tuning Experiment Report

Date: 2026-05-23

## Summary

This run completed a full-data smoke-to-grid experiment for lightweight physics residuals in MDM, followed by a longer 5-epoch fine-tuning round on the most promising lower regularization range.

Main result:

- One-epoch physics fine-tuning improves lightweight physics metrics on a small 8-prompt / 2-repetition sample set.
- `lambda_phys=0.1` gives the best small-grid physics metrics, but HumanML debug eval shows text-motion alignment degradation.
- A second round with 5-epoch fine-tuning on `lambda_phys={0.005,0.01,0.02}` found `lambda_phys=0.01` to be the best overall trade-off.
- Post-generation optimization still helps after fine-tuning.
- HumanML debug eval favors the 5-epoch lower-lambda models over the one-epoch `0.1` run.

## Data Setup

The local HumanML3D data was completed from the official MDM Google Drive bundle.

Final dataset counts:

```text
dataset/HumanML3D/new_joint_vecs: 29228 .npy files
dataset/HumanML3D/new_joints:     29228 .npy files
train.txt:     23384
val.txt:        1460
test.txt:       4384
train_val.txt: 24844
```

The dataset loader sanity check passed:

```text
motion shape: (2, 263, 1, 196)
```

## Code Change For Evaluation

I added `--skip_visualization` to `sample.generate`.

Reason: the default generation path renders mp4 files after writing `results.npy`, and ffmpeg was the bottleneck for metric-only grid evaluation. The default behavior is unchanged unless `--skip_visualization` is passed.

Changed files:

```text
utils/parser_util.py
sample/generate.py
```

## Fine-Tuning Setup

Base checkpoint:

```text
save/humanml_enc_512_50steps/model000750000.pt
```

Fine-tuning command template:

```bash
python -m train.train_mdm \
  --resume_checkpoint save/humanml_enc_512_50steps/model000750000.pt \
  --dataset humanml \
  --diffusion_steps 50 \
  --batch_size 32 \
  --lr 1e-5 \
  --num_steps 1 \
  --save_interval 1000 \
  --log_interval 50 \
  --lambda_phys <value> \
  --lambda_phys_ground 1.0 \
  --lambda_phys_foot 1.0 \
  --lambda_phys_smooth 0.05
```

In this training loop, `--num_steps 1` runs one full epoch because `num_epochs = num_steps // len(data) + 1`.

Grid:

| Run     | `lambda_phys` | Checkpoint                                               |
| ------- | ------------: | -------------------------------------------------------- |
| lam0001 |         0.001 | `save/physics_ft_lam0001_epoch1_clean/model000750768.pt` |
| lam001  |          0.01 | `save/physics_ft_lam001_epoch1_clean/model000750768.pt`  |
| lam005  |          0.05 | `save/physics_ft_lam005_epoch1_clean/model000750768.pt`  |
| lam01   |           0.1 | `save/physics_ft_lam01_epoch1_clean/model000750768.pt`   |

Training logs are in:

```text
reports/exp_logs/train_lam*_epoch1_clean.log
```

## Physics Evaluation Setup

Prompt set:

```text
assets/example_text_prompts.txt
```

Sampling:

```text
8 prompts x 2 repetitions = 16 generated motions per model
guidance_param = 2.5
```

Metric script:

```bash
python eval/eval_physics.py --results_path save/physics_eval_grid/<run>_samples/results.npy
```

## Physics Metrics

Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.

| Run      | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
| -------- | -----------: | ----------------------: | ---------------: | --------------: | ----------------: |
| baseline |     0.008437 |                0.006321 |       0.00012748 |        0.056766 |          0.005780 |
| lam0001  |     0.007368 |                0.004096 |       0.00001076 |        0.004889 |          0.003845 |
| lam001   |     0.007375 |                0.002202 |       0.00000181 |        0.006960 |          0.004100 |
| lam005   |     0.007353 |                0.005161 |       0.00001044 |        0.009445 |          0.004329 |
| lam01    |     0.006951 |                0.001847 |       0.00000295 |        0.011802 |          0.004341 |

Relative to baseline:

| Run     | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
| ------- | -----------: | ----------------------: | ---------------: | --------------: | ----------------: |
| lam0001 |       -12.7% |                  -35.2% |           -91.6% |          -91.4% |            -33.5% |
| lam001  |       -12.6% |                  -65.2% |           -98.6% |          -87.7% |            -29.1% |
| lam005  |       -12.9% |                  -18.4% |           -91.8% |          -83.4% |            -25.1% |
| lam01   |       -17.6% |                  -70.8% |           -97.7% |          -79.2% |            -24.9% |

Best per metric:

```text
Foot skating:            lam01
Ground penetration rate: lam01
Mean penetration:        lam001
Max penetration:         lam0001
Mean acceleration:       lam0001
```

## Post-Generation Optimization

Post-generation optimization was run on:

```text
baseline
lam01
```

Settings:

```text
steps = 200
lr = 1e-2
lambda_ground = 1.0
lambda_foot = 1.0
lambda_smooth = 0.05
lambda_anchor = 1.0
```

| Run      | Stage  | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
| -------- | ------ | -----------: | ----------------------: | ---------------: | --------------: | ----------------: |
| baseline | before |     0.008437 |                0.006321 |       0.00012748 |        0.056766 |          0.005780 |
| baseline | after  |     0.005812 |                0.006250 |       0.00006374 |        0.028390 |          0.005676 |
| lam01    | before |     0.006951 |                0.001847 |       0.00000295 |        0.011802 |          0.004341 |
| lam01    | after  |     0.005719 |                0.001823 |       0.00000148 |        0.005663 |          0.004177 |

Post-opt improves both baseline and fine-tuned outputs. The best physics result in this run is `lam01 + post-opt`.

## HumanML Debug Eval

I ran `eval.eval_humanml --eval_mode debug` for baseline and `lam01`.

This is not the full benchmark. It is useful as a quick quality regression check.

| Run      | Matching score ↓ | R-precision top1 ↑ | R-precision top2 ↑ | R-precision top3 ↑ |  FID ↓ | Diversity |
| -------- | ---------------: | -----------------: | -----------------: | -----------------: | -----: | --------: |
| baseline |           3.4827 |             0.4152 |             0.6037 |             0.7195 | 0.7927 |    9.0371 |
| lam01    |           3.6534 |             0.3875 |             0.5834 |             0.7012 | 0.5387 |    9.4245 |

Interpretation:

- `lam01` improves FID in debug eval.
- `lam01` worsens matching score and R-precision, so text-motion alignment likely degraded.
- This suggests `lambda_phys=0.1` is too aggressive if text alignment is a priority.

## Second Round: 5-Epoch Fine-Tuning

Reason for the second round:

- The first round showed that `lambda_phys=0.1` is strong on the small physics grid but too aggressive for text-motion alignment.
- I therefore moved to lower values and trained longer from the original base checkpoint, not from the one-epoch fine-tuned checkpoints.

Setup:

```text
base checkpoint: save/humanml_enc_512_50steps/model000750000.pt
batch_size:      32
lr:              1e-5
num_steps:       3068
effective run:   5 epochs
guidance_param:  2.5 for sampling/eval
```

Grid:

| Run        | `lambda_phys` | Checkpoint                                        |
| ---------- | ------------: | ------------------------------------------------- |
| lam005_5ep |         0.005 | `save/physics_ft_lam005_5epoch/model000753836.pt` |
| lam001_5ep |          0.01 | `save/physics_ft_lam001_5epoch/model000753836.pt` |
| lam002_5ep |          0.02 | `save/physics_ft_lam002_5epoch/model000753836.pt` |

Training logs:

```text
reports/exp_logs/train_lam005_5epoch.log
reports/exp_logs/train_lam001_5epoch.log
reports/exp_logs/train_lam002_5epoch.log
```

## 32-Prompt Physics Evaluation

I generated a larger deterministic prompt set from the first 32 HumanML3D test captions:

```text
reports/physics_eval_prompts_32.txt
```

Sampling:

```text
32 prompts x 2 repetitions = 64 generated motions per model
guidance_param = 2.5
```

Models evaluated:

```text
baseline
lam005_5ep
lam001_5ep
lam002_5ep
lam01_1ep
```

Lower is better except `foot_contact_rate`, which is diagnostic.

| Run        | Foot contact rate | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
| ---------- | ----------------: | -----------: | ----------------------: | ---------------: | --------------: | ----------------: |
| baseline   |          0.370569 |     0.009700 |                0.001586 |       0.00000320 |        0.008801 |          0.006715 |
| lam005_5ep |          0.406710 |     0.009122 |                0.001799 |       0.00000592 |        0.036462 |          0.006999 |
| lam001_5ep |          0.429950 |     0.008733 |                0.001261 |       0.00000283 |        0.005806 |          0.006960 |
| lam002_5ep |          0.422532 |     0.009838 |                0.001675 |       0.00000319 |        0.004710 |          0.006863 |
| lam01_1ep  |          0.396304 |     0.010021 |                0.002515 |       0.00003230 |        0.024837 |          0.005172 |

Relative to baseline:

| Run        | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
| ---------- | -----------: | ----------------------: | ---------------: | --------------: | ----------------: |
| lam005_5ep |        -6.0% |                  +13.4% |           +85.0% |         +314.3% |             +4.2% |
| lam001_5ep |       -10.0% |                  -20.5% |           -11.4% |          -34.0% |             +3.6% |
| lam002_5ep |        +1.4% |                   +5.6% |            -0.3% |          -46.5% |             +2.2% |
| lam01_1ep  |        +3.3% |                  +58.6% |          +909.9% |         +182.2% |            -23.0% |

Interpretation:

- `lam001_5ep` is the only second-round model that improves foot skating, ground penetration rate, mean penetration, and max penetration together.
- `lam002_5ep` gives the lowest max penetration before post-opt, but worsens foot skating and ground penetration rate.
- `lam005_5ep` and the one-epoch `lam01_1ep` do not generalize well on the larger prompt set.

## 32-Prompt Post-Generation Optimization

I ran post-generation physics optimization on:

```text
baseline
lam001_5ep
lam002_5ep
```

Settings:

```text
steps = 200
lr = 1e-2
lambda_ground = 1.0
lambda_foot = 1.0
lambda_smooth = 0.05
lambda_anchor = 1.0
```

| Run        | Stage  | Foot contact rate | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
| ---------- | ------ | ----------------: | -----------: | ----------------------: | ---------------: | --------------: | ----------------: |
| baseline   | before |          0.370569 |     0.009700 |                0.001586 |       0.00000320 |        0.008801 |          0.006715 |
| baseline   | after  |          0.369748 |     0.006350 |                0.001580 |       0.00000160 |        0.004333 |          0.006540 |
| lam001_5ep | before |          0.429950 |     0.008733 |                0.001261 |       0.00000283 |        0.005806 |          0.006960 |
| lam001_5ep | after  |          0.428965 |     0.006565 |                0.001255 |       0.00000142 |        0.002868 |          0.006742 |
| lam002_5ep | before |          0.422532 |     0.009838 |                0.001675 |       0.00000319 |        0.004710 |          0.006863 |
| lam002_5ep | after  |          0.421251 |     0.007500 |                0.001669 |       0.00000160 |        0.002353 |          0.006647 |

Post-opt effect relative to each model's own generated samples:

| Run        | Foot skating | Ground penetration rate | Mean penetration | Max penetration | Mean acceleration |
| ---------- | -----------: | ----------------------: | ---------------: | --------------: | ----------------: |
| baseline   |       -34.5% |                   -0.4% |           -50.0% |          -50.8% |             -2.6% |
| lam001_5ep |       -24.8% |                   -0.5% |           -50.0% |          -50.6% |             -3.1% |
| lam002_5ep |       -23.8% |                   -0.4% |           -50.0% |          -50.0% |             -3.1% |

Interpretation:

- Post-opt stacks cleanly with fine-tuning.
- `lam001_5ep + post-opt` keeps foot skating near baseline post-opt while reducing ground penetration rate, mean penetration, and max penetration.
- `lam002_5ep + post-opt` has the lowest max penetration, but worse foot skating and ground penetration rate than `lam001_5ep + post-opt`.

## HumanML Debug Eval: Second Round

I ran `eval.eval_humanml --eval_mode debug` for the two strongest second-round models and compare them with the earlier baseline and `lam01`.

This is still a debug eval, not the full HumanML benchmark.

| Run        | Matching score ↓ | R-precision top1 ↑ | R-precision top2 ↑ | R-precision top3 ↑ |  FID ↓ | Diversity |
| ---------- | ---------------: | -----------------: | -----------------: | -----------------: | -----: | --------: |
| baseline   |           3.4827 |             0.4152 |             0.6037 |             0.7195 | 0.7927 |    9.0371 |
| lam01_1ep  |           3.6534 |             0.3875 |             0.5834 |             0.7012 | 0.5387 |    9.4245 |
| lam001_5ep |           3.4174 |             0.4221 |             0.6168 |             0.7391 | 0.4829 |    9.4274 |
| lam002_5ep |           3.4195 |             0.4209 |             0.6256 |             0.7332 | 0.3836 |    9.3065 |

Interpretation:

- Both 5-epoch lower-lambda models recover the alignment loss seen in `lam01_1ep`.
- `lam001_5ep` has the best matching score and top1/top3 R-precision among the fine-tuned models.
- `lam002_5ep` has the best debug FID and top2 R-precision, but its physics metrics are less balanced.

## Final Conclusion

The completed research cycle supports this model choice:

```text
best overall checkpoint: save/physics_ft_lam001_5epoch/model000753836.pt
best overall setting:    lambda_phys=0.01, 5 epochs, lr=1e-5
```

Reason:

- It improves the larger 32-prompt physics grid on the main targeted metrics: foot skating, ground penetration rate, mean penetration, and max penetration.
- It avoids the HumanML debug alignment regression observed with `lambda_phys=0.1`.
- It remains compatible with post-generation optimization, which further reduces skating and penetration.

The cycle did not run the full HumanML benchmark. The quality gate used here is HumanML debug eval because it is fast enough to compare multiple candidates in the same experimental pass.

## Artifacts

Data:

```text
dataset/HumanML3D/
```

Fine-tuned checkpoints:

```text
save/physics_ft_lam0001_epoch1_clean/model000750768.pt
save/physics_ft_lam001_epoch1_clean/model000750768.pt
save/physics_ft_lam005_epoch1_clean/model000750768.pt
save/physics_ft_lam01_epoch1_clean/model000750768.pt
save/physics_ft_lam005_5epoch/model000753836.pt
save/physics_ft_lam001_5epoch/model000753836.pt
save/physics_ft_lam002_5epoch/model000753836.pt
```

Physics eval outputs:

```text
save/physics_eval_grid/*_samples/physics_metrics.json
save/physics_eval_grid/*_samples/results.npy
save/physics_eval_grid32/*_samples/physics_metrics.json
save/physics_eval_grid32/*_samples/results.npy
```

Post-opt outputs:

```text
save/physics_eval_grid/baseline_samples_physopt/
save/physics_eval_grid/lam01_samples_physopt/
save/physics_eval_grid32/baseline_samples_physopt/
save/physics_eval_grid32/lam001_5ep_samples_physopt/
save/physics_eval_grid32/lam002_5ep_samples_physopt/
```

Logs:

```text
reports/exp_logs/
```

Committed/reproducible source artifacts:

```text
sample/generate.py
utils/parser_util.py
.gitignore
reports/physics_eval_prompts_32.txt
reports/physics_residual_full_experiment_report.md
```
