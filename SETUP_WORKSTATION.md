# 在工作站上跑 MDM

本教學針對此工作站環境（無 conda、Python 3.14 系統版本、RTX 4090）。

## 前置需求

- `uv` 已安裝（`~/.local/bin/uv`）
- NVIDIA GPU 可用（`nvidia-smi` 確認）

## Step 1：建立 Python 環境

```bash
cd /path/to/motion-diffusion-model

uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
```

## Step 2：安裝套件

```bash
# PyTorch（帶 CUDA 12.1）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# pip 和 setuptools（chumpy 的 build 需要）
uv pip install pip setuptools

# chumpy 需要 --no-build-isolation
uv pip install --no-build-isolation chumpy

# 其餘依賴
uv pip install \
  spacy smplx trimesh einops ftfy gdown "moviepy<2.0" \
  scikit-learn scipy matplotlib tqdm wandb clearml

# spacy 英文模型
python -m spacy download en_core_web_sm

# CLIP
uv pip install "git+https://github.com/openai/CLIP.git"
```

## Step 3：修相容性問題

系統套件版本較新，需要 patch 幾個地方：

### chumpy（numpy 新版移除了舊別名）

編輯 `.venv/lib/python3.10/site-packages/chumpy/__init__.py`，
把這行：
```python
from numpy import bool, int, float, complex, object, unicode, str, nan, inf
```
換成：
```python
from numpy import nan, inf
import numpy as _np
bool = _np.bool_
int = _np.int_
float = _np.float64
complex = _np.complex128
object = _np.object_
unicode = _np.str_
str = _np.str_
```

### moviepy（matplotlib 3.8+ 改名了 `tostring_rgb`）

編輯 `.venv/lib/python3.10/site-packages/moviepy/video/io/bindings.py`，
把 `mplfig_to_npimage` 最後幾行：
```python
buf = canvas.tostring_rgb()
image= np.frombuffer(buf, dtype=np.uint8)
return image.reshape(h,w,3)
```
換成：
```python
buf = canvas.buffer_rgba()
image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
return image[:, :, :3]
```

### plot_script.py（matplotlib 3.4+ 廢棄了 `p3.Axes3D(fig)`）

這個 patch 已經直接修在 repo 裡的 `data_loaders/humanml/utils/plot_script.py`，
`git pull` 後應該已經包含。如果沒有，把：
```python
ax = p3.Axes3D(fig)
```
換成：
```python
ax = fig.add_subplot(projection='3d')
```
並把 `ax.grid(b=False)` 改成 `ax.grid(False)`。

## Step 4：下載資料

### SMPL body model

```bash
PATH="$(pwd)/.venv/bin:$PATH" bash prepare/download_smpl_files.sh
```

### glove
```bash
PATH="$(pwd)/.venv/bin:$PATH" bash prepare/download_glove.sh
```
### t2m（text-to-motion benchmark）

```bash
PATH="$(pwd)/.venv/bin:$PATH" bash prepare/download_t2m_evaluators.sh
```

### HumanML3D

```bash
cd /tmp2/$USER   # 放在工作目錄外
git clone --depth=1 https://github.com/EricGuo5513/HumanML3D.git
unzip HumanML3D/HumanML3D/texts.zip -d HumanML3D/HumanML3D/
mkdir -p /path/to/motion-diffusion-model/dataset/HumanML3D
cp -r HumanML3D/HumanML3D/* /path/to/motion-diffusion-model/dataset/HumanML3D/
cd /path/to/motion-diffusion-model
```

其他資料: [Drive link](https://drive.google.com/drive/folders/1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu) 也要下載解壓縮放進 `dataset/HumanML3D/`。

## Step 5：下載預訓練模型

使用 50-step 版本（最快，效果接近）：

```bash
mkdir -p save
.venv/bin/gdown "https://drive.google.com/uc?id=1cfadR1eZ116TIdXK7qDX1RugAerEiJXr" \
  -O save/humanml_enc_512_50steps.zip
unzip save/humanml_enc_512_50steps.zip -d save/
```

其他可用模型見 [README.md](README.md) 的 pretrained models 段落。

## Step 6：跑 generation

```bash
source .venv/bin/activate

# 單一 prompt
python -m sample.generate \
  --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
  --text_prompt "a person waves their right hand." \
  --num_samples 1 \
  --num_repetitions 1 \
  --device 0

# 從檔案跑多個 prompts
python -m sample.generate \
  --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
  --input_text ./assets/example_text_prompts.txt \
  --device 0
```

輸出會存在 `save/humanml_enc_512_50steps/samples_.../`，包含：
- `results.npy`：raw motion data
- `samples_00_to_02.mp4`：stick figure 動畫（每個 mp4 最多 3 個 sample）

## 常用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num_samples` | 6 | 生成幾個不同動作 |
| `--num_repetitions` | 3 | 每個 prompt 重複幾次 |
| `--motion_length` | 6.0 | 動作長度（秒），最多 9.8 |
| `--seed` | 10 | 隨機 seed |
| `--device` | 0 | GPU id |
