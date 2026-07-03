"""ComfyUI install/update, custom node management, GPU detection, workflow install.

Git-free by design: everything installs from GitHub codeload zip archives so a
beginner's machine needs neither git nor a system Python.
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

from . import config

UA = {"User-Agent": "comfy-provisioner/1.0"}

CUSTOM_NODES = [
    # (repo, needs pip install of its requirements.txt)
    ("ltdrdata/ComfyUI-Manager", True),
    ("Kosinkadink/ComfyUI-VideoHelperSuite", True),
    ("Fannovel16/ComfyUI-Frame-Interpolation", True),
    ("kijai/ComfyUI-KJNodes", True),
    ("kijai/ComfyUI-WanVideoWrapper", True),
    ("city96/ComfyUI-GGUF", True),
    ("cubiq/ComfyUI_IPAdapter_plus", False),
    ("Fannovel16/comfyui_controlnet_aux", True),
]

COMMON_WINDOWS_PATHS = [
    r"%USERPROFILE%\ComfyUI", r"%USERPROFILE%\Documents\ComfyUI",
    r"%USERPROFILE%\Downloads\ComfyUI_windows_portable\ComfyUI",
    r"C:\ComfyUI", r"C:\ComfyUI_windows_portable\ComfyUI",
    r"D:\ComfyUI", r"D:\ComfyUI_windows_portable\ComfyUI",
    r"%USERPROFILE%\AppData\Roaming\StabilityMatrix\Packages\ComfyUI",
]
COMMON_POSIX_PATHS = ["~/ComfyUI", "~/comfyui", "/workspace/ComfyUI", "/opt/ComfyUI"]


def is_comfy_dir(path: Path) -> bool:
    return (path / "main.py").exists() and (path / "comfy").is_dir()


def detect_comfy() -> Path | None:
    """Find an existing ComfyUI install: saved COMFY_DIR first, then common paths."""
    saved = config.comfy_dir()
    if saved and is_comfy_dir(saved):
        return saved
    candidates = COMMON_WINDOWS_PATHS if os.name == "nt" else COMMON_POSIX_PATHS
    for raw in candidates:
        p = Path(os.path.expandvars(os.path.expanduser(raw)))
        if is_comfy_dir(p):
            return p
    return None


def detect_gpu() -> dict:
    """GPU name + VRAM (GB) via nvidia-smi. Returns {} when no NVIDIA GPU found."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        line = out.stdout.strip().splitlines()[0]
        name, mem = line.rsplit(",", 1)
        return {"name": name.strip(), "vram_gb": round(int(mem.strip()) / 1024, 1)}
    except Exception:
        return {}


def pick_profile(vram_gb: float | None, cloud: bool = False) -> str:
    if cloud:
        return "cloud-80gb" if (vram_gb or 0) >= 70 else "cloud"
    if vram_gb is None or vram_gb >= 20:
        return "local-24gb"
    return "local-12gb"


def _github_zip(repo: str, dest_parent: Path, dest_name: str | None = None) -> Path:
    """Download a repo's default branch as a zip and extract (git-free clone)."""
    api = requests.get(f"https://api.github.com/repos/{repo}", headers=UA, timeout=30)
    branch = api.json().get("default_branch", "master") if api.ok else "master"
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    resp = requests.get(url, headers=UA, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        top = zf.namelist()[0].split("/")[0]
        zf.extractall(dest_parent)
    extracted = dest_parent / top
    final = dest_parent / (dest_name or repo.split("/")[1])
    if extracted != final:
        if final.exists():
            shutil.rmtree(final)
        extracted.replace(final)
    return final


def comfy_python(comfy_root: Path) -> list[str]:
    """The Python that runs this ComfyUI (portable builds bundle their own)."""
    embedded = comfy_root.parent / "python_embeded" / "python.exe"
    if embedded.exists():
        return [str(embedded)]
    venv = comfy_root / "venv"
    for cand in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if cand.exists():
            return [str(cand)]
    return [sys.executable]


def install_comfy(dest: Path, log=print) -> Path:
    """Fresh git-free ComfyUI install + venv with torch (CUDA on win/linux)."""
    log("Downloading ComfyUI…")
    root = _github_zip("comfyanonymous/ComfyUI", dest.parent, dest.name)
    log("Setting up ComfyUI's own Python environment (this takes a few minutes)…")
    venv_dir = root / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    py = comfy_python(root)
    subprocess.run([*py, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    if platform.system() in ("Windows", "Linux"):
        subprocess.run([*py, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                        "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    else:
        subprocess.run([*py, "-m", "pip", "install", "torch", "torchvision", "torchaudio"],
                       check=True)
    subprocess.run([*py, "-m", "pip", "install", "-r", str(root / "requirements.txt")],
                   check=True)
    return root


def update_comfy(comfy_root: Path, log=print) -> None:
    """Update an install we manage (zip-based). Git checkouts update via git pull."""
    if (comfy_root / ".git").exists():
        if shutil.which("git"):
            subprocess.run(["git", "-C", str(comfy_root), "pull", "--ff-only"], check=False)
        return
    log("Updating ComfyUI…")
    _github_zip("comfyanonymous/ComfyUI", comfy_root.parent, comfy_root.name + ".new")
    new = comfy_root.parent / (comfy_root.name + ".new")
    for item in new.iterdir():  # overlay, preserving models/user/custom_nodes
        if item.name in ("models", "user", "custom_nodes", "venv", "output", "input"):
            shutil.rmtree(item) if item.is_dir() else item.unlink()
            continue
        target = comfy_root / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        item.replace(target)
    shutil.rmtree(new, ignore_errors=True)
    py = comfy_python(comfy_root)
    subprocess.run([*py, "-m", "pip", "install", "-r",
                    str(comfy_root / "requirements.txt")], check=False)


def install_custom_nodes(comfy_root: Path, log=print) -> list[str]:
    """Install/refresh the custom-node set. Returns list of failures (names)."""
    nodes_dir = comfy_root / "custom_nodes"
    nodes_dir.mkdir(exist_ok=True)
    py = comfy_python(comfy_root)
    failures = []
    for repo, needs_pip in CUSTOM_NODES:
        name = repo.split("/")[1]
        try:
            if not (nodes_dir / name).exists():
                log(f"Installing node pack: {name}")
                _github_zip(repo, nodes_dir, name)
            req = nodes_dir / name / "requirements.txt"
            if needs_pip and req.exists():
                subprocess.run([*py, "-m", "pip", "install", "-r", str(req)],
                               check=True, capture_output=True)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    # FaceID workflows need insightface; wheels exist for common setups but the
    # install can fail without a C++ toolchain — best-effort, non-fatal.
    try:
        subprocess.run([*py, "-m", "pip", "install", "insightface", "onnxruntime"],
                       check=True, capture_output=True, timeout=1800)
    except Exception:
        failures.append("insightface (only the FaceID workflow needs it — "
                        "see TROUBLESHOOTING.md)")
    return failures


def install_sageattention(comfy_root: Path, log=print) -> bool:
    """Cloud-profile speedup (10–20% on Wan). Best-effort — needs a CUDA toolchain."""
    py = comfy_python(comfy_root)
    try:
        subprocess.run([*py, "-m", "pip", "install", "sageattention"],
                       check=True, capture_output=True, timeout=1800)
        return True
    except Exception:
        log("SageAttention install skipped (no compatible wheel) — ComfyUI still works.")
        return False


def install_workflows(comfy_root: Path, workflows_dir: Path, log=print) -> list[str]:
    """Copy generated workflow JSONs into ComfyUI's Workflows menu."""
    target = comfy_root / "user" / "default" / "workflows"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for wf in sorted(workflows_dir.glob("*.json")):
        shutil.copy2(wf, target / wf.name)
        copied.append(wf.name)
    log(f"Installed {len(copied)} workflows into ComfyUI's Workflows menu.")
    return copied


def launch(comfy_root: Path, listen: bool = False, extra_args: list[str] | None = None):
    """Start ComfyUI as a detached process; returns the Popen handle."""
    py = comfy_python(comfy_root)
    args = [*py, "-s", "main.py"]
    if listen:
        args += ["--listen", "0.0.0.0"]
    if extra_args:
        args += extra_args
    return subprocess.Popen(args, cwd=str(comfy_root))
