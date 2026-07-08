#!/bin/bash
# RunPod bootstrap — set as the template's start command. Zero-terminal flow:
#   first boot on an empty volume  -> browser wizard on :8189 collects the keys,
#                                     provisions /workspace, then starts ComfyUI
#   later boots / Spot migrations  -> self-heals and goes straight to ComfyUI
#
# Designed for interruptible (Spot/Community) pods: the network volume persists
# but the container's installed pip packages are wiped on every migration, so
# this script RE-INSTALLS all Python deps into the launch interpreter on every
# boot. That is the price of "just launch it and it works" on Spot hardware.
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="$WORKSPACE/ComfyUI"
export COMFY_DIR="$COMFY"
export HF_HOME="$WORKSPACE/.hf_cache"
# Always use this pod's system python (resolve the absolute path now, before uv
# shadows it on PATH). The runpod/pytorch images ship torch in it.
export COMFY_SYSTEM_PYTHON="$(command -v python3)"
PYBIN="$COMFY_SYSTEM_PYTHON"
echo "[start.sh] Pinned ComfyUI python: $PYBIN"

mkdir -p "$WORKSPACE"

# 0. SELF-UPDATE: always run the latest committed provisioning logic. Without
#    this the pod can get stuck on stale code (e.g. a node missing from the
#    install list), and no amount of pushing fixes it. Force-reset discards the
#    workflow JSONs regenerated last boot; .env / models are untracked and kept.
if [ -d "$REPO_DIR/.git" ]; then
    echo "[start.sh] Updating provisioner to latest..."
    git -C "$REPO_DIR" fetch origin 2>/dev/null \
        && git -C "$REPO_DIR" reset --hard origin/main 2>/dev/null \
        || echo "[start.sh] (self-update skipped — offline or not a clone)"
fi
cd "$REPO_DIR"

# uv provides python for the provisioner CLI (fetched once, cached on volume)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 1. ComfyUI on the network volume (persists across pods)
if [ ! -f "$COMFY/main.py" ]; then
    echo "[start.sh] Installing ComfyUI onto the network volume..."
    uv run --python 3.12 provision.py --comfy-dir "$COMFY" --profile cloud install
fi

# route all model folders to the volume
cat > "$COMFY/extra_model_paths.yaml" <<EOF
runpod:
  base_path: $COMFY/models
  checkpoints: checkpoints
  diffusion_models: diffusion_models
  loras: loras
  vae: vae
  text_encoders: text_encoders
  clip_vision: clip_vision
  ipadapter: ipadapter
  controlnet: controlnet
  upscale_models: upscale_models
EOF

# 2. First boot with an empty volume and no saved keys -> browser wizard
if [ ! -f "$REPO_DIR/.env" ] || ! grep -q "CIVITAI_API_TOKEN=." "$REPO_DIR/.env"; then
    echo "[start.sh] No API keys yet — serving the setup wizard on port 8189."
    echo "[start.sh] Open the pod's Connect -> HTTP 8189 link to finish setup."
    uv run --python 3.12 provision.py wizard --host 0.0.0.0 --port 8189 --no-browser &
    WIZARD_PID=$!
    until uv run --python 3.12 provision.py --profile cloud --comfy-dir "$COMFY" dry-run \
          | grep -q '"to_download": \[\]'; do
        sleep 20
    done
    kill "$WIZARD_PID" 2>/dev/null || true
fi

# 3. Provision node folders (clones missing ones onto the volume) + models +
#    workflows. Idempotent; fast when the volume is already populated.
uv run --python 3.12 provision.py --comfy-dir "$COMFY" nodes || true
uv run --python 3.12 provision.py --profile cloud --comfy-dir "$COMFY" sync || true
uv run --python 3.12 provision.py --comfy-dir "$COMFY" workflows || true

# 4. RE-INSTALL PYTHON DEPS INTO THE LAUNCH INTERPRETER (survives migrations).
#    The node *folders* live on the volume, but their pip deps are wiped when a
#    Spot pod migrates — so reinstall them here, every boot, into $PYBIN.
echo "[start.sh] Ensuring ComfyUI + custom-node deps in $PYBIN (this can take a few minutes on a fresh container)..."
"$PYBIN" -m pip install -q -r "$COMFY/requirements.txt" || true
"$PYBIN" -m pip install -q --upgrade comfyui-frontend-package \
    comfyui-workflow-templates comfyui-embedded-docs || true
for req in "$COMFY"/custom_nodes/*/requirements.txt; do
    [ -f "$req" ] || continue
    echo "[start.sh]   deps: $(basename "$(dirname "$req")")"
    "$PYBIN" -m pip install -q -r "$req" || true
done
# FluxTrainer (sd-scripts) pins numpy<=1.26.4, which breaks ComfyUI's numpy-2.x
# deps (opencv/scipy/tifffile -> video + controlnet nodes). Restore it LAST.
"$PYBIN" -m pip install -q "numpy>=2.1,<2.8" || true
# FluxTrainer imports transformers' CLIPFeatureExtractor, removed in transformers
# 5.x -> whole pack fails to load. Alias it to the current name. Idempotent.
FT_LPW="$COMFY/custom_nodes/ComfyUI-FluxTrainer/library/sdxl_lpw_stable_diffusion.py"
[ -f "$FT_LPW" ] && sed -i \
    's/import CLIPFeatureExtractor,/import CLIPImageProcessor as CLIPFeatureExtractor,/' \
    "$FT_LPW" || true

# JupyterLab on :8888 for drag-and-drop dataset uploads (LoRA training)
if command -v jupyter >/dev/null 2>&1; then
    echo "[start.sh] Starting JupyterLab on :8888 (file uploads)"
    jupyter lab --allow-root --no-browser --ip 0.0.0.0 --port 8888 \
        --ServerApp.token='' --ServerApp.password='' \
        --ServerApp.root_dir="$WORKSPACE" >"$WORKSPACE/jupyter.log" 2>&1 &
fi

cd "$COMFY"
echo "[start.sh] Starting ComfyUI on :8188"
# --enable-cors-header: ComfyUI's host/origin check 403s behind RunPod's proxy
# (github.com/Comfy-Org/ComfyUI/issues/4865); this flag relaxes it.
exec "$PYBIN" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header
