#!/usr/bin/env bash
# setup_env.sh — one-shot environment setup for MDM on this workstation
# Usage: bash setup_env.sh
# Re-running is safe: each step is idempotent.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

UV="${HOME}/.local/bin/uv"
if [[ ! -x "$UV" ]]; then
    echo "[ERROR] uv not found at $UV. Install with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Redirect uv cache to /tmp2 to avoid home-directory quota issues
export UV_CACHE_DIR="/tmp2/${USER}/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# ── Step 1: Python 3.10 venv ─────────────────────────────────────────────────
echo "=== Step 1: creating Python 3.10 venv ==="
"$UV" python install 3.10
"$UV" venv --python 3.10 .venv
source .venv/bin/activate

# ── Step 2: PyTorch + core deps ──────────────────────────────────────────────
echo "=== Step 2: installing PyTorch (CUDA 12.1) ==="
"$UV" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 2b: pip + setuptools (needed by chumpy build) ==="
"$UV" pip install pip setuptools

echo "=== Step 2c: chumpy (no build isolation) ==="
"$UV" pip install --no-build-isolation chumpy

echo "=== Step 2d: remaining deps ==="
"$UV" pip install \
    spacy smplx trimesh einops ftfy gdown "moviepy<2.0" \
    scikit-learn scipy matplotlib tqdm wandb clearml \
    blobfile transformers huggingface-hub joblib

echo "=== Step 2e: spacy English model ==="
python -m spacy download en_core_web_sm

echo "=== Step 2f: CLIP ==="
"$UV" pip install "git+https://github.com/openai/CLIP.git"

# ── Step 3: compatibility patches ────────────────────────────────────────────
echo "=== Step 3: patching chumpy (numpy compat) ==="
CHUMPY_INIT="$REPO/.venv/lib/python3.10/site-packages/chumpy/__init__.py"
if [[ -f "$CHUMPY_INIT" ]] && grep -q "from numpy import bool, int" "$CHUMPY_INIT"; then
    python - <<'PYEOF'
import re, pathlib
p = pathlib.Path(".venv/lib/python3.10/site-packages/chumpy/__init__.py")
src = p.read_text()
old = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
new = (
    "from numpy import nan, inf\n"
    "import numpy as _np\n"
    "bool = _np.bool_\n"
    "int = _np.int_\n"
    "float = _np.float64\n"
    "complex = _np.complex128\n"
    "object = _np.object_\n"
    "unicode = _np.str_\n"
    "str = _np.str_"
)
if old in src:
    p.write_text(src.replace(old, new))
    print("  chumpy patched OK")
else:
    print("  chumpy already patched, skipping")
PYEOF
else
    echo "  chumpy patch not needed or file missing"
fi

echo "=== Step 3b: patching moviepy (matplotlib compat) ==="
MOVIEPY_BINDINGS="$REPO/.venv/lib/python3.10/site-packages/moviepy/video/io/bindings.py"
if [[ -f "$MOVIEPY_BINDINGS" ]] && grep -q "tostring_rgb" "$MOVIEPY_BINDINGS"; then
    python - <<'PYEOF'
import pathlib
p = pathlib.Path(".venv/lib/python3.10/site-packages/moviepy/video/io/bindings.py")
src = p.read_text()
old = (
    "    buf = canvas.tostring_rgb()\n"
    "    image= np.frombuffer(buf, dtype=np.uint8)\n"
    "    return image.reshape(h,w,3)"
)
new = (
    "    buf = canvas.buffer_rgba()\n"
    "    image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)\n"
    "    return image[:, :, :3]"
)
if old in src:
    p.write_text(src.replace(old, new))
    print("  moviepy patched OK")
else:
    print("  moviepy already patched or different version, skipping")
PYEOF
else
    echo "  moviepy patch not needed or file missing"
fi

echo "=== Step 3c: patching plot_script.py (matplotlib 3.4+ compat) ==="
python - <<'PYEOF'
import pathlib
p = pathlib.Path("data_loaders/humanml/utils/plot_script.py")
if not p.exists():
    print("  plot_script.py not found, skipping")
    exit()
src = p.read_text()
changed = False
if "p3.Axes3D(fig)" in src:
    src = src.replace("ax = p3.Axes3D(fig)", "ax = fig.add_subplot(projection='3d')")
    changed = True
if "ax.grid(b=False)" in src:
    src = src.replace("ax.grid(b=False)", "ax.grid(False)")
    changed = True
if changed:
    p.write_text(src)
    print("  plot_script.py patched OK")
else:
    print("  plot_script.py already OK, skipping")
PYEOF

# ── Step 4: download SMPL body models ────────────────────────────────────────
echo ""
echo "=== Step 4: SMPL body models ==="
echo "  Run this manually (requires registration at https://smpl.is.tue.mpg.de/):"
echo "    PATH=\"\$(pwd)/.venv/bin:\$PATH\" bash prepare/download_smpl_files.sh"
echo "  OR place smpl/ folder at body_models/smpl/"

# ── Step 5: HumanML3D dataset ─────────────────────────────────────────────────
echo ""
echo "=== Step 5: HumanML3D dataset ==="
if [[ -d "$REPO/dataset/HumanML3D/texts" ]]; then
    echo "  Already present."
else
    echo "  Not found. Run:"
    echo "    cd /tmp2/\$USER"
    echo "    git clone --depth=1 https://github.com/EricGuo5513/HumanML3D.git"
    echo "    unzip HumanML3D/HumanML3D/texts.zip -d HumanML3D/HumanML3D/"
    echo "    mkdir -p $REPO/dataset/HumanML3D"
    echo "    cp -r HumanML3D/HumanML3D/* $REPO/dataset/HumanML3D/"
fi

# ── Step 6: pretrained model ──────────────────────────────────────────────────
echo ""
echo "=== Step 6: pretrained model ==="
if ls "$REPO"/save/humanml_enc_512_50steps/model*.pt 2>/dev/null | grep -q .; then
    echo "  Already present."
else
    echo "  Downloading 50-step MDM (fastest, ~2 GB)..."
    mkdir -p "$REPO/save"
    python -m gdown "https://drive.google.com/uc?id=1cfadR1eZ116TIdXK7qDX1RugAerEiJXr" \
        -O "$REPO/save/humanml_enc_512_50steps.zip" || \
        echo "  [!] gdown failed — download manually from Google Drive and unzip to save/"
    if [[ -f "$REPO/save/humanml_enc_512_50steps.zip" ]]; then
        unzip -q "$REPO/save/humanml_enc_512_50steps.zip" -d "$REPO/save/"
        echo "  Unzipped OK."
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Setup complete (or partially complete — see above)"
echo " Activate env:  source .venv/bin/activate"
echo ""
echo " Quick sanity check:"
echo "   python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo ""
echo " Generate motion:"
echo "   python -m sample.generate \\"
echo "     --model_path ./save/humanml_enc_512_50steps/model000750000.pt \\"
echo "     --text_prompt \"a person waves their right hand.\" \\"
echo "     --num_samples 1 --num_repetitions 1 --device 0"
echo ""
echo " With physics guidance (our addition):"
echo "   python -m sample.generate \\"
echo "     --model_path ./save/humanml_enc_512_50steps/model000750000.pt \\"
echo "     --text_prompt \"a person walks forward.\" \\"
echo "     --guidance_param 2.5 --physics_guidance_scale 10. \\"
echo "     --num_samples 1 --num_repetitions 1 --device 0"
echo "════════════════════════════════════════════════════"
