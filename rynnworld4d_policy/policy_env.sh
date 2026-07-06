#!/bin/bash
# Installs RynnWorld-4D-Policy dependencies on top of the env already created by
# the main README (`conda create -n rynnworld4d ...` + `pip install -r requirements.txt`
# + `pip install -e .`).
#
# Run this AFTER `conda activate rynnworld4d`.
#
# Note: this script does NOT (re)install torch / torchvision — it assumes the
# cu121 build from the main README is already in place. flash-attn 2.8.3 is
# compiled against the currently installed torch, so make sure `python -c "import
# torch; print(torch.__version__)"` works before running.

set -e

# --- sanity checks ---
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ No conda environment is active. Run 'conda activate rynnworld4d' first."
    exit 1
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch is not importable in the current env. Follow the main README"
    echo "   env-setup first (conda create / pip install torch / pip install -r requirements.txt)."
    exit 1
fi

echo "=== Using Python: $(which python)"
echo "=== Torch version: $(python -c 'import torch; print(torch.__version__)')"

PIP="python -m pip"

echo "=== Upgrading pip ==="
$PIP install --upgrade pip

echo "=== Installing core ML packages (skipped if already satisfied) ==="
$PIP install \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    pytorch-lightning==2.6.1 \
    lightning==2.5.6 \
    lightning-utilities==0.15.3 \
    einops==0.8.2 \
    "einops-exts==0.0.4" \
    tokenizers==0.22.2 \
    safetensors==0.7.0 \
    huggingface_hub==0.36.2 \
    hf_transfer==0.1.9 \
    datasets==4.8.5

echo "=== Installing data/vision packages ==="
$PIP install \
    pandas==2.2.3 \
    opencv-python==4.10.0.84 \
    Pillow==12.2.0 \
    pillow_heif==1.3.0 \
    matplotlib==3.10.8 \
    scipy==1.13.1 \
    scikit-learn==1.7.0 \
    timm==1.0.25 \
    open_clip_torch==3.3.0 \
    kornia==0.8.2 \
    kornia_rs==0.1.11 \
    torchmetrics==1.9.0 \
    torchdata==0.11.0 \
    av==17.0.1 \
    moviepy==1.0.3

echo "=== Installing training utilities ==="
$PIP install \
    bitsandbytes==0.49.2 \
    xformers==0.0.29.post3 \
    tensorboard==2.18.0 \
    tensorboardX==2.6.4 \
    jsonargparse==4.46.0

echo "=== Installing flash-attn (built against the currently installed torch) ==="
$PIP install flash_attn==2.8.3 --no-build-isolation

echo "=== Installing 3D/geometry packages ==="
$PIP install \
    open3d==0.19.0 \
    trimesh==4.12.2 \
    pycolmap==4.0.4 \
    plyfile==1.1.3 \
    e3nn==0.6.0 \
    pyquaternion==0.9.9

echo "=== Installing remaining utility packages ==="
$PIP install \
    absl-py==2.4.0 \
    addict==2.4.0 \
    aiofiles==25.1.0 \
    aiohttp==3.13.3 \
    carvekit==4.1.2 \
    click==8.3.3 \
    dash==4.1.0 \
    docstring_parser==0.17.0 \
    evo==1.36.4 \
    fastapi==0.136.1 \
    Flask==3.1.3 \
    h5py==3.12.1 \
    hjson==3.1.0 \
    httpx==0.28.1 \
    ijson==3.5.0 \
    jsonlines==4.0.0 \
    kaleido==0.2.1 \
    loguru==0.7.3 \
    lz4==4.4.5 \
    modelscope==1.36.1 \
    natsort==8.4.0 \
    nbformat==5.10.4 \
    nest-asyncio==1.6.0 \
    numexpr==2.14.1 \
    onnxruntime==1.26.0 \
    opt-einsum-fx==0.1.4 \
    opt_einsum==3.4.0 \
    optree==0.14.0 \
    plotly==5.24.1 \
    proglog==0.1.12 \
    protobuf==6.33.5 \
    pyarrow==24.0.0 \
    pydantic==2.12.5 \
    pyDeprecate==0.7.0 \
    pypdf==6.10.2 \
    python-dotenv==1.2.2 \
    python-multipart==0.0.28 \
    rich==14.3.3 \
    rosbags==0.11.2 \
    seaborn==0.13.2 \
    sentry-sdk==2.54.0 \
    supervision==0.28.0 \
    tabulate==0.9.0 \
    tenacity==9.1.4 \
    torchelastic==0.2.2 \
    typeguard==4.5.1 \
    typer==0.25.1 \
    typeshed_client==2.9.0 \
    uvicorn==0.46.0

echo "=== Installing editable git packages (requires internet) ==="
$PIP install -e "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@2c21ea849ceec7b469a3e62ea0c0e270afc3281a#egg=depth_anything_3"
$PIP install -e "git+https://github.com/SandAI-org/MagiAttention.git@36edccedd108f47f1685d3c323c008699d4b96bb#egg=magi_attention"
$PIP install -e "git+https://github.com/UMass-Embodied-AGI/TesserAct.git@d2fe4b1d509e54a3cf95da40ec6e232b68361d07#egg=tesseract"

echo "=== Installing local ptlflow (optional) ==="
PTLFLOW_PATH="${PTLFLOW_PATH:-./third_party/ptlflow-main}"
if [ -d "$PTLFLOW_PATH" ]; then
    $PIP install -e "$PTLFLOW_PATH"
else
    echo "WARNING: ptlflow path not found at $PTLFLOW_PATH — skipping"
    echo "         (set PTLFLOW_PATH=/your/path or git clone it into third_party/ptlflow-main if you need optical-flow inference)"
fi

echo ""
echo "=== Done! Policy-specific dependencies installed into env: $CONDA_DEFAULT_ENV ==="
