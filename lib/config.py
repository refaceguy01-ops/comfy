"""Environment / .env handling. The wizard writes .env; users never touch it."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
LOG_DIR = REPO_ROOT / "logs"

KNOWN_KEYS = ("CIVITAI_API_TOKEN", "HF_TOKEN", "COMFY_DIR")


def load_env() -> dict:
    """Read .env into a dict and export into os.environ (without clobbering)."""
    values = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    for key, val in values.items():
        os.environ.setdefault(key, val)
    return values


def save_env(**updates) -> None:
    """Persist key=value pairs into .env, preserving unrelated lines."""
    current = load_env()
    current.update({k: v.strip() for k, v in updates.items() if v is not None})
    lines = ["# Written by comfy-provisioner. Safe to delete; the wizard recreates it."]
    for key, val in current.items():
        lines.append(f"{key}={val}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key, val in current.items():
        os.environ[key] = val


def get(key: str, default: str | None = None) -> str | None:
    load_env()
    return os.environ.get(key, default)


def comfy_dir() -> Path | None:
    val = get("COMFY_DIR")
    return Path(val) if val else None
