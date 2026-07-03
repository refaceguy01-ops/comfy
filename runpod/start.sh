#!/bin/bash
# RunPod bootstrap — set as the template's start command. Zero-terminal flow:
#   first boot on an empty volume  -> browser wizard on :8189 collects the keys,
#                                     provisions /workspace, then starts ComfyUI
#   later boots                    -> straight to ComfyUI in under a minute
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="$WORKSPACE/ComfyUI"
export COMFY_DIR="$COMFY"
export HF_HOME="$WORKSPACE/.hf_cache"
# Never trust a venv left on the volume by a previous pod — always use this
# pod's system python (the runpod/pytorch images ship torch in it).
export COMFY_FORCE_SYSTEM_PYTHON=1

mkdir -p "$WORKSPACE"
cd "$REPO_DIR"

# uv provides python + deps (baked into the Docker image, or fetched once)
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
#    (expose port 8189 in the RunPod template; user opens it via Connect)
if [ ! -f "$REPO_DIR/.env" ] || ! grep -q "CIVITAI_API_TOKEN=." "$REPO_DIR/.env"; then
    echo "[start.sh] No API keys yet — serving the setup wizard on port 8189."
    echo "[start.sh] Open the pod's Connect -> HTTP 8189 link to finish setup."
    uv run --python 3.12 provision.py wizard --host 0.0.0.0 --port 8189 --no-browser &
    WIZARD_PID=$!
    # wait until the wizard has produced a complete model set
    until uv run --python 3.12 provision.py --profile cloud --comfy-dir "$COMFY" dry-run \
          | grep -q '"to_download": \[\]'; do
        sleep 20
    done
    kill "$WIZARD_PID" 2>/dev/null || true
else
    # 3. Idempotent sync: fills gaps, skips verified files (fast when complete);
    #    re-check node packs too (repairs partial first-boot installs)
    uv run --python 3.12 provision.py --comfy-dir "$COMFY" nodes || true
    uv run --python 3.12 provision.py --profile cloud --comfy-dir "$COMFY" sync || true
    uv run --python 3.12 provision.py --comfy-dir "$COMFY" workflows || true
fi

# 4. Launch ComfyUI for the pod's Connect button (port 8188).
#    Fresh/migrated pods have bare system pythons (and any venv on the volume
#    points at the previous pod's interpreter) — make sure ComfyUI's own deps
#    exist in whatever python we're about to launch with.
cd "$COMFY"
PYBIN="python3"
echo "[start.sh] Ensuring ComfyUI requirements in $PYBIN..."
"$PYBIN" -m pip install -r "$COMFY/requirements.txt" || true
# the web UI ships as separate packages — make sure they really landed
# (a quiet earlier failure here served 404s with a healthy API)
"$PYBIN" -m pip install --upgrade comfyui-frontend-package \
    comfyui-workflow-templates comfyui-embedded-docs || true
echo "[start.sh] Starting ComfyUI on :8188"
exec "$PYBIN" main.py --listen 0.0.0.0 --port 8188
