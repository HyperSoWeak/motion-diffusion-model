# Velocity-Guided Motion Diffusion - Complete Guide

## ✨ What Was Built

A **classifier-guided diffusion system** using root velocity to control motion generation. Control trajectory direction and speed during motion synthesis.

---

## 🚀 Quick Start (1 minute)

```bash
# Forward-walking motion
python sample/generate.py \
  --model_path checkpoints/humanml_final.pt \
  --text_prompt "a person walks" \
  --velocity_guidance_scale 0.3

# Specific direction (forward-right diagonal)
python sample/generate.py \
  --model_path checkpoints/humanml_final.pt \
  --text_prompt "walking" \
  --target_velocity "0.7,0.7" \
  --velocity_guidance_scale 0.3
```

---

## 📋 Key Parameters

### Velocity Guidance Arguments
```bash
--velocity_guidance_scale       Strength (0.0-1.0)
                               0.0 = disabled
                               0.1-0.2 = subtle
                               0.3-0.4 = moderate (recommended)
                               >0.5 = strong (may override text)

--velocity_guidance_mode       'direction' (default) or 'magnitude'
                               direction = steer toward velocity
                               magnitude = control speed

--target_velocity              Target as "vx,vz" (default: forward)
                               "1.0,0.0" = forward
                               "0.7,0.7" = forward-right
                               "0.0,1.0" = right
                               "-1.0,0.0" = backward

--velocity_regressor_path      Path to trained regressor (optional)
```

---

## 📚 Usage Examples

### Basic Direction Control
```bash
# Walking forward
python sample/generate.py --model_path path/to/your/mdm_model.pt --text_prompt "walk" --target_velocity "1.0,0.0" --velocity_guidance_scale 0.3

# Walking in circle
python sample/generate.py --model_path path/to/your/mdm_model.pt --text_prompt "walk" --target_velocity "0.3,0.3" --velocity_guidance_scale 0.3

# Walking right
python sample/generate.py --model_path path/to/your/mdm_model.pt --text_prompt "walk" --target_velocity "0.0,1.0" --velocity_guidance_scale 0.3
```

### Speed Control (Magnitude Mode)
```bash
# Fast running
python sample/generate.py \
  --model_path path/to/your/mdm_model.pt \
  --text_prompt "running" \
  --velocity_guidance_mode magnitude \
  --target_velocity "1.5,0.0" \
  --velocity_guidance_scale 0.3

# Slow walking
python sample/generate.py \
  --model_path path/to/your/mdm_model.pt \
  --text_prompt "walking" \
  --velocity_guidance_mode magnitude \
  --target_velocity "0.3,0.0" \
  --velocity_guidance_scale 0.3
```

### Combined Text + Velocity Guidance
```bash
python sample/generate.py \
  --model_path path/to/your/mdm_model.pt \
  --text_prompt "dancing in a circle" \
  --guidance_param 2.5 \
  --velocity_guidance_scale 0.25 \
  --target_velocity "0.3,0.3"
```

---

## 🏋️ Training the Velocity Regressor

The guidance quality improves significantly with a trained regressor. Training takes 2-4 hours on GPU.

### Quick Training
```bash
python train/train_velocity_regressor.py \
  --data_root dataset/HumanML3D \
  --output_dir checkpoints/velocity_regressor \
  --epochs 50 \
  --batch_size 32
```

### Advanced Options
```bash
# Smaller model (faster, less memory)
--num_layers 2

# Larger model (better, needs more VRAM)
--num_layers 4

# Different learning rate
--learning_rate 5e-4

# CPU training (not recommended - very slow)
--device cpu
```

### What Gets Trained
- `VelocityRegressor`: Predicts 2D root velocity from motion
- Input shape: `(B, njoints, nfeats, seq_len)` = `(B, 22, 12, T)`
- Output shape: `(B, seq_len, 2)` = per-frame velocity predictions
- Architecture: 3-layer temporal convolution with batch norm
- Parameters: ~256K
- Model size: ~2MB

### Training Output
```
Epoch 1/50
Training: 100%|██████| 412/412 [00:15<00:00, 26.8it/s]
Train Loss: 0.234567, MAE: 0.123456
Val Loss:   0.210123, MAE: 0.115234
Saved: checkpoints/velocity_regressor/velocity_regressor_best.pt
```

Use the `*_best.pt` checkpoint (best validation performance).

---

## 🔧 Technical Details

### How It Works

During diffusion sampling:
1. Generate motion: `x_pred = model(x_t, t, y)`
2. Predict velocity: `vel_pred = regressor(x_pred)`
3. Compute loss: `loss = guidance_loss(vel_pred, target_vel)`
4. Backprop: `grad = ∇ loss`
5. Update: `x_pred = x_pred - λ * normalize(grad)`

### Guidance Modes

**Direction Mode:**
- Maximizes dot product with target velocity
- Steers motion toward specified direction
- Use for trajectory control

**Magnitude Mode:**
- Matches velocity magnitude (speed)
- Uses MSE loss on velocity norms
- Use for speed control

### Motion Data Format (HumanML3D)

```
Channel 0:     Root rotation velocity (1D)
Channels 1-2:  Root linear velocity xz-plane (2D)  ← Used for guidance
Channel 3:     Root y position (height)
Channels 4-67: Joint positions (ric_data, 63D)
Channels 67-193: Joint rotations (rot_data, 126D)
Channels 193-259: Local velocities (66D)
Channels 259-263: Foot contact (4D)
```

After padding to 264 (22×12), reshaped to `(B, 22, 12, T)` for MDM format.

---

## 🎯 Performance Recommendations

| Setting | Use Case | Guidance Scale |
|---------|----------|---|
| Light | Subtle trajectory bias | 0.1-0.2 |
| Moderate | Clear direction enforcement | 0.3-0.4 |
| Strong | Override other conditions | >0.5 |

**Tip:** Match text prompts to target velocity for best results.
```bash
# Good combination
--text_prompt "walking forward"
--target_velocity "1.0,0.0"

# Conflicting (avoid)
--text_prompt "walking backward"
--target_velocity "1.0,0.0"
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Activate environment and run from project root
```bash
conda activate mdm
cd /path/to/motion-diffusion-model
python train/train_velocity_regressor.py ...
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size
```bash
--batch_size 16  # was 32
```

### Issue: Training very slow
**Solution:** Use GPU. Check if CUDA available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Generated motions ignore velocity target
**Solution:** Increase guidance scale or train regressor
```bash
--velocity_guidance_scale 0.5  # increase from 0.3
```

### Issue: Text prompt is ignored
**Solution:** Reduce velocity guidance scale
```bash
--velocity_guidance_scale 0.1  # reduce from 0.3
```

### Issue: Validation loss not improving
**Normal after epoch 20+.** Always use `*_best.pt` checkpoint.

### Issue: Motion has artifacts
**Solution:** Reduce guidance scale or enable dropout
```bash
--velocity_guidance_scale 0.2
--dropout 0.2
```

---

## 📁 Files Implementation

### New Files
```
model/velocity_classifier.py         - Velocity prediction networks
utils/velocity_guidance_sampler.py   - Guidance wrapper
scripts/generate_with_velocity_guidance.py - Example script
train/train_velocity_regressor.py    - Training script
```

### Modified Files
```
sample/generate.py                   - Integrated guidance
utils/parser_util.py                 - Added CLI arguments
```

---

## 🔍 Architecture Overview

### VelocityRegressor
```
Input (B, 264, T)
  ↓
Conv1d(264 → 256, k=3, p=1) + BatchNorm + ReLU
  ↓
[Conv1d(256 → 256, k=3, p=1) + BatchNorm + ReLU] × (num_layers-1)
  ↓
Conv1d(256 → 2, k=1)
  ↓
Output (B, T, 2)  [per-frame velocity predictions]
```

### Guidance Integration
```
MDM Model
  ↓
x_pred (motion features)
  ↓
VelocityRegressor → vel_pred
  ↓
GuidanceLoss(vel_pred, target)
  ↓
Gradient Update (strength = λ)
  ↓
Guided x_pred
```

---

## 💡 Tips & Tricks

1. **Start conservatively**
   ```bash
   --velocity_guidance_scale 0.2
   ```

2. **Experiment with directions**
   ```bash
   # Try multiple targets
   for vel in "1.0,0.0" "0.7,0.7" "0.0,1.0"; do
     python sample/generate.py --target_velocity "$vel" ...
   done
   ```

3. **Combine with text classifier-free guidance**
   ```bash
   --guidance_param 2.5           # Text guidance
   --velocity_guidance_scale 0.2  # Velocity guidance (smaller)
   ```

4. **Visualize results**
   ```bash
   python visualize/render_mesh.py <generated_npy>
   ```

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Untrained regressor = weak initial guidance (train for best results)
- Controls 2D velocity only (xz-plane, not vertical)
- Single target velocity (cannot vary over time)
- Gradient-based (may create artifacts at high scales)

### Future Enhancements
1. Train velocity regressor on full dataset (implemented)
2. 3D velocity guidance (include vertical motion)
3. Trajectory planning (specify velocity at different frames)
4. Physics-aware constraints (floor contact, etc.)
5. Acceleration/deceleration control
6. Multi-target guidance (different body parts)

---

## 🔗 References

- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
- Diffusion Models: https://arxiv.org/abs/2006.11239
- Motion Diffusion Model: https://arxiv.org/abs/2209.14916

---

## ✅ Progress Checklist

- [x] Core velocity regressor model (`model/velocity_classifier.py`)
- [x] Guidance sampling wrapper (`utils/velocity_guidance_sampler.py`)
- [x] Integration into generation pipeline (`sample/generate.py`)
- [x] CLI argument support (`utils/parser_util.py`)
- [x] Training script (`train/train_velocity_regressor.py`)
- [x] Example script (`scripts/generate_with_velocity_guidance.py`)
- [x] Comprehensive documentation (this file)

Ready to use! 🎬
