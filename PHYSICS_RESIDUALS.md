# Physics Residuals for MDM

This branch adds lightweight, differentiable physics residuals for MDM without changing the model architecture.

## What Changed

Core files:

```text
utils/physics_losses.py
eval/eval_physics.py
sample/physics_optimize.py
diffusion/gaussian_diffusion.py
utils/model_util.py
utils/parser_util.py
reports/physics_residual_report.md
```

Main idea:

```text
L = L_MDM + lambda_phys * L_phys
```

`L_phys` is computed on MDM's predicted clean motion `x_0`, after recovering HumanML3D joints with `recover_from_ric`.

Implemented residuals:

```text
ground penetration
foot skating
acceleration smoothness
```

## Workflow

```text
baseline MDM generation
-> physics metrics on results.npy
-> post-generation XYZ optimization
-> optional fine-tuning with physics residuals on predicted x_0
```

## Evaluate Generated Motions

Run metrics on an MDM `results.npy` file produced by `sample/generate.py`:

```bash
python eval/eval_physics.py \
  --results_path save/.../results.npy
```

The script reports:

```text
ground_penetration_rate
mean_penetration_depth
max_penetration_depth
foot_contact_rate
foot_skating_score
mean_acceleration
```

## Post-Generation Optimization

This freezes MDM and optimizes the generated XYZ joint motion directly:

```bash
python sample/physics_optimize.py \
  --results_path save/.../results.npy \
  --steps 200 \
  --lr 1e-2 \
  --lambda_ground 1.0 \
  --lambda_foot 1.0 \
  --lambda_smooth 0.05 \
  --lambda_anchor 1.0 \
  --overwrite
```

The output directory defaults to the input directory with `_physopt` appended. It writes:

```text
results.npy
physics_optimization_metrics.json
```

`lambda_anchor` keeps the optimized XYZ motion close to the original generated motion. Increase it if the optimization changes the motion too much.

## Fine-Tuning

The training loss supports optional physics residuals on predicted clean motion `x_0`.

Example:

```bash
python -m train.train_mdm \
  --save_dir save/humanml_phys_ft \
  --resume_checkpoint save/humanml_enc_512_50steps/model000750000.pt \
  --dataset humanml \
  --lambda_phys 0.01 \
  --lambda_phys_ground 1.0 \
  --lambda_phys_foot 1.0 \
  --lambda_phys_smooth 0.05
```

Start with small `lambda_phys` values:

```text
0.001, 0.01, 0.05, 0.1
```

The fine-tuning hook currently supports HumanML3D `hml_vec` motions. It denormalizes the predicted `x_0`, recovers joints with `recover_from_ric`, and computes the physics residuals on those joints.

## Current Results

The most reliable completed result is post-generation optimization on one prompt:

```text
Prompt: the person walked forward and is picking up his toolbox
Checkpoint: save/humanml_enc_512_50steps/model000750000.pt
```

| Metric | Baseline | Post-Opt |
|---|---:|---:|
| Foot skating score | 0.01592 | 0.01341 |
| Mean acceleration | 0.00732 | 0.00717 |
| Mean penetration depth | 0.00000638 | 0.00000318 |
| Max penetration depth | 0.01476 | 0.00740 |
| Ground penetration rate | 0.000505 | 0.000505 |

Report:

```text
reports/physics_residual_report.md
```

## Known Limitation

The local `dataset/HumanML3D` directory currently contains only one motion `.npy` file. Because of that, the fine-tuning run in this branch is only a smoke test proving that the loss is wired into the training loop. Meaningful fine-tuning requires the full HumanML3D motion dataset.
