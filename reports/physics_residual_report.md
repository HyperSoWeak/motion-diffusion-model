# Physics Residuals for MDM - Short Report

## Goal

We tested a lightweight physics-guided extension for MDM. Instead of adding a full physics engine, we add differentiable residual losses on MDM's predicted clean motion `x_0`.

Implemented residuals:

- Ground penetration loss
- Foot skating loss
- Acceleration smoothness loss

## Implementation

New code:

- `utils/physics_losses.py`: shared physics losses and metrics
- `eval/eval_physics.py`: evaluates generated `results.npy`
- `sample/physics_optimize.py`: post-generation motion optimization
- `diffusion/gaussian_diffusion.py`: optional training loss hook via `--lambda_phys`

Fine-tuning is disabled by default. The original behavior is unchanged when `--lambda_phys 0`.

## Experiment Setup

Prompt:

```text
the person walked forward and is picking up his toolbox
```

Pretrained checkpoint:

```text
save/humanml_enc_512_50steps/model000750000.pt
```

Metrics were computed on generated XYZ joints:

- ground penetration rate
- mean / max penetration depth
- foot skating score
- mean acceleration

## Results

### Baseline vs Post-Generation Optimization

| Metric | Baseline | Post-Opt |
|---|---:|---:|
| Foot skating score | 0.01592 | 0.01341 |
| Mean acceleration | 0.00732 | 0.00717 |
| Mean penetration depth | 0.00000638 | 0.00000318 |
| Max penetration depth | 0.01476 | 0.00740 |
| Ground penetration rate | 0.000505 | 0.000505 |

Post-generation optimization reduced foot skating by about 15.8% and max penetration by about 49.8% on this sample.

### Physics Fine-Tuning Smoke Run

I also ran a short 7-step smoke fine-tune with:

```text
lambda_phys = 0.01
lambda_phys_ground = 1.0
lambda_phys_foot = 1.0
lambda_phys_smooth = 0.05
```

The training loop successfully logged:

```text
phys_ground
phys_foot
phys_smooth
phys_loss
```

This confirms that the residuals are connected to the predicted `x_0` training loss.

Important limitation: the local `dataset/HumanML3D` directory contains only one motion file, so this run is only a pipeline smoke test, not a meaningful full fine-tuning result.

## Conclusion

The most reliable current result is post-generation optimization: it is simple, stable, and already reduces physical artifacts on generated motion. The fine-tuning code path is implemented and verified, but meaningful evaluation requires the full HumanML3D motion dataset.

## Next Steps

1. Restore the full HumanML3D motion dataset.
2. Fine-tune from the pretrained MDM checkpoint for several thousand steps.
3. Evaluate baseline, post-optimized, and physics-finetuned MDM on the same prompt set.
4. Tune `lambda_phys` in `{0.001, 0.01, 0.05, 0.1}`.
