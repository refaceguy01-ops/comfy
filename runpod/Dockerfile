# Optional baked image for fast cold starts: ComfyUI deps + node packs, NO models
# (models live on the network volume, provisioned by start.sh on first boot).
#
#   docker build -t youruser/comfy-provisioner:latest -f runpod/Dockerfile .
#   docker push youruser/comfy-provisioner:latest
#
# RunPod template: image above, start command `bash /app/runpod/start.sh`,
# expose HTTP ports 8188 (ComfyUI) and 8189 (first-boot wizard),
# mount a network volume at /workspace (150 GB+ recommended).

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates ffmpeg libgl1 libglib2.0-0 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# uv: portable python + dependency manager used by provision.py
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY . /app
RUN chmod +x runpod/start.sh Setup.command

# warm the dependency cache so first boot doesn't pip-install the CLI's deps
RUN uv run --python 3.12 provision.py --help >/dev/null

EXPOSE 8188 8189
CMD ["bash", "/app/runpod/start.sh"]
