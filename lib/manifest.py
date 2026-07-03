"""manifest.yaml schema + validation (pydantic) and profile filtering."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .config import REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "manifest.yaml"

PROFILES = ("local-12gb", "local-24gb", "cloud", "cloud-80gb")

# Only these destinations are allowed — catches typos before they scatter
# files around the ComfyUI tree.
ALLOWED_DESTS = {
    "models/diffusion_models", "models/checkpoints", "models/loras",
    "models/vae", "models/text_encoders", "models/clip_vision",
    "models/ipadapter", "models/controlnet", "models/upscale_models",
    "models/foley",
}


class ModelEntry(BaseModel):
    name: str
    desc: str = ""
    source: Literal["huggingface", "civitai", "url"]
    dest: str
    file: Optional[str] = None
    rename: Optional[str] = None
    sha256: Optional[str] = None
    size_gb: float = 0.0
    tags: list[str] = Field(default_factory=list)
    profiles: list[str] = Field(default_factory=list)
    gated: bool = False
    # huggingface
    repo: Optional[str] = None
    # civitai
    model_id: Optional[int] = None
    version_id: Optional[int] = None
    # url
    url: Optional[str] = None
    # recorded by the downloader from the Civitai API
    license: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.dest not in ALLOWED_DESTS:
            raise ValueError(f"{self.name}: unknown dest {self.dest!r}")
        if self.source == "huggingface" and not (self.repo and self.file):
            raise ValueError(f"{self.name}: huggingface entries need repo + file")
        if self.source == "civitai" and not (self.model_id and self.version_id):
            raise ValueError(f"{self.name}: civitai entries need model_id + version_id")
        if self.source == "url" and not self.url:
            raise ValueError(f"{self.name}: url entries need url")
        for p in self.profiles:
            if p not in PROFILES:
                raise ValueError(f"{self.name}: unknown profile {p!r}")
        return self

    @property
    def local_name(self) -> str:
        return self.rename or Path(self.file or self.url.split("?")[0]).name

    def target_path(self, comfy_root: Path) -> Path:
        return comfy_root / self.dest / self.local_name


class Manifest(BaseModel):
    schema_version: int = 1
    models: list[ModelEntry]
    user_loras: list[ModelEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self):
        names = [m.name for m in self.models + self.user_loras]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate entry names: {sorted(dupes)}")
        return self

    def entries_for(self, profile: str, include_optional: bool = True) -> list[ModelEntry]:
        out = []
        for entry in self.models + self.user_loras:
            if entry.profiles and profile not in entry.profiles:
                continue
            if not include_optional and "optional" in entry.tags:
                continue
            out.append(entry)
        return out


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(data)


def append_user_lora(entry: ModelEntry, path: Path = MANIFEST_PATH) -> None:
    """Append a user LoRA entry to manifest.yaml, preserving the file's comments."""
    manifest = load_manifest(path)  # validates uniqueness against existing names
    for existing in manifest.models + manifest.user_loras:
        if existing.name == entry.name:
            raise ValueError(f"entry {entry.name!r} already exists in the manifest")

    text = path.read_text(encoding="utf-8")
    dump = yaml.safe_dump(
        [entry.model_dump(exclude_none=True, exclude_defaults=True)],
        sort_keys=False, allow_unicode=True,
    )
    if "user_loras: []" in text:
        text = text.replace("user_loras: []", "user_loras:\n" + dump.rstrip("\n"))
    else:
        # subsequent additions: insert right after the user_loras: line
        # (sequence items at column 0 under a mapping key are valid YAML)
        lines = text.splitlines()
        idx = next(i for i, ln in enumerate(lines) if ln.startswith("user_loras:"))
        lines[idx + 1:idx + 1] = dump.rstrip("\n").splitlines()
        text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    load_manifest(path)  # re-validate after edit
