"""Civitai API client: version resolution, download URLs, safety + license checks."""
from __future__ import annotations

import re
from typing import Optional

import requests

from . import config

API = "https://civitai.com/api/v1"
UA = {"User-Agent": "comfy-provisioner/1.0"}

# Civitai tags that mark real-person / celebrity content — add-lora refuses these.
BLOCKED_TAGS = {"celebrity", "real person", "real_person", "actress", "actor",
                "idol", "instagram model", "famous person"}


class CivitaiError(RuntimeError):
    """Raised with a plain-English, user-facing message."""


def _token() -> Optional[str]:
    return config.get("CIVITAI_API_TOKEN")


def _get(path: str, **params) -> dict:
    token = _token()
    headers = dict(UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(f"{API}/{path}", params=params or None, headers=headers, timeout=60)
    if resp.status_code == 401:
        raise CivitaiError(
            "Civitai rejected your API key — it may have been copied with an extra "
            "space, or it may have been deleted. Re-enter it in the setup wizard.")
    if resp.status_code == 404:
        raise CivitaiError("Civitai says that model doesn't exist. Check the link.")
    resp.raise_for_status()
    return resp.json()


def check_token() -> bool:
    """True if the configured token is accepted by Civitai."""
    try:
        _get("models", limit=1)
        return True
    except CivitaiError:
        return False


def get_model(model_id: int) -> dict:
    return _get(f"models/{model_id}")


def get_version(version_id: int) -> dict:
    return _get(f"model-versions/{version_id}")


def download_url(version_id: int, file_meta: dict | None = None) -> str:
    """Direct download URL, with ?token= appended (many files require auth).

    Prefers the API-reported downloadUrl (note: /api/download/, not /api/v1/).
    """
    url = (file_meta or {}).get("downloadUrl") \
        or f"https://civitai.com/api/download/models/{version_id}"
    token = _token()
    if token:
        url += ("&" if "?" in url else "?") + f"token={token}"
    return url


def primary_file(version: dict) -> dict:
    files = version.get("files", [])
    for f in files:
        if f.get("primary"):
            return f
    if not files:
        raise CivitaiError("That Civitai version has no downloadable files.")
    return files[0]


def license_summary(model: dict) -> str:
    """Compact license string recorded into licenses.json for the commercial-use audit."""
    parts = [f"allowCommercialUse={model.get('allowCommercialUse')}"]
    for key in ("allowNoCredit", "allowDerivatives", "allowDifferentLicense"):
        parts.append(f"{key}={model.get(key)}")
    return "; ".join(parts)


URL_RE = re.compile(
    r"civitai\.com/models/(?P<model_id>\d+)(?:[^?\s]*)?"
    r"(?:\?.*modelVersionId=(?P<version_id>\d+))?", re.I)


def parse_model_url(url: str) -> tuple[int, Optional[int]]:
    """Accept a pasted model-page URL; return (model_id, version_id or None)."""
    m = URL_RE.search(url.strip())
    if not m:
        raise CivitaiError(
            "That doesn't look like a Civitai model link. It should look like "
            "https://civitai.com/models/12345/some-name — copy it from your browser's "
            "address bar on the model's page.")
    vid = m.group("version_id")
    return int(m.group("model_id")), int(vid) if vid else None


def refuse_if_unsafe(model: dict) -> None:
    """Hard policy gate: no real identifiable people, nothing involving minors."""
    name = model.get("name", "this model")
    if model.get("poi"):
        raise CivitaiError(
            f"'{name}' is flagged by Civitai as depicting a real, identifiable person. "
            "This tool doesn't download real-person models: using someone's likeness "
            "without consent is harmful and, in explicit contexts, illegal in most "
            "places. Generate a fictional face and use that instead.")
    if model.get("minor"):
        raise CivitaiError(f"'{name}' is flagged as involving minors. Refusing to download.")
    tags = {str(t).lower() for t in model.get("tags", [])}
    hit = tags & BLOCKED_TAGS
    if hit:
        raise CivitaiError(
            f"'{name}' is tagged {sorted(hit)} — it appears to depict a real person, "
            "so this tool won't download it. Fictional characters and styles are fine.")


def resolve_lora(url: str) -> dict:
    """Resolve a pasted Civitai URL into a manifest-ready LoRA entry dict.

    Raises CivitaiError (with a friendly message) for non-LoRAs and refused content.
    """
    model_id, version_id = parse_model_url(url)
    model = get_model(model_id)
    refuse_if_unsafe(model)

    if model.get("type") not in ("LORA", "LoCon", "DoRA"):
        raise CivitaiError(
            f"'{model.get('name')}' is a {model.get('type')}, not a LoRA. "
            "Only LoRA links can be added this way.")

    versions = model.get("modelVersions", [])
    if version_id:
        version = next((v for v in versions if v["id"] == version_id), None) \
            or get_version(version_id)
    else:
        if not versions:
            raise CivitaiError("That model page has no published versions yet.")
        version = versions[0]  # newest first

    f = primary_file(version)
    slug = re.sub(r"[^a-z0-9]+", "-", model["name"].lower()).strip("-")
    return {
        "name": f"lora-{slug}-{version['id']}",
        "desc": f"{model['name']} — {version.get('name', '')} (user-added LoRA)",
        "source": "civitai",
        "model_id": model_id,
        "version_id": version["id"],
        "file": f["name"],
        "dest": "models/loras",
        "sha256": (f.get("hashes") or {}).get("SHA256"),
        "size_gb": round(f.get("sizeKB", 0) / 1048576, 3),
        "tags": ["image", "optional"],
        "license": license_summary(model),
    }
