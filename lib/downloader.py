"""Download engine: resume, SHA-256 verification, idempotent skip, small parallel pool.

Safe to rerun on every pod boot — files already present and verified are skipped.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

from . import civitai, huggingface
from .config import REPO_ROOT
from .manifest import Manifest, ModelEntry

MAX_PARALLEL = 2          # Civitai rate-limits; keep this low
CHUNK = 1024 * 1024
LICENSES_FILE = REPO_ROOT / "licenses.json"

_license_lock = threading.Lock()


@dataclass
class SyncReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


def sha256_of(path: Path, progress: bool = False) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    bar = tqdm(total=size, unit="B", unit_scale=True, desc=f"verify {path.name}",
               leave=False) if progress else None
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            if bar:
                bar.update(len(chunk))
    if bar:
        bar.close()
    return h.hexdigest().upper()


def is_present_and_valid(entry: ModelEntry, comfy_root: Path, deep: bool = False) -> bool:
    target = entry.target_path(comfy_root)
    if not target.exists() or target.stat().st_size == 0:
        return False
    if deep and entry.sha256:
        return sha256_of(target, progress=True) == entry.sha256.upper()
    return True


def _http_download(url: str, target: Path, desc: str, headers: dict | None = None) -> None:
    """Streaming download with HTTP-Range resume into target + '.part'."""
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    pos = part.stat().st_size if part.exists() else 0
    req_headers = dict(headers or {})
    if pos:
        req_headers["Range"] = f"bytes={pos}-"

    with requests.get(url, stream=True, timeout=120, headers=req_headers,
                      allow_redirects=True) as resp:
        if pos and resp.status_code == 200:
            pos = 0  # server ignored Range; start over
        elif pos and resp.status_code != 206:
            resp.raise_for_status()
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + pos
        mode = "ab" if pos else "wb"
        with part.open(mode) as fh, tqdm(
                total=total or None, initial=pos, unit="B", unit_scale=True,
                desc=desc, leave=False) as bar:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))
    part.replace(target)


def _record_license(entry: ModelEntry, license_str: str) -> None:
    with _license_lock:
        data = {}
        if LICENSES_FILE.exists():
            data = json.loads(LICENSES_FILE.read_text(encoding="utf-8"))
        data[entry.name] = {
            "file": entry.local_name,
            "source": entry.source,
            "model_id": entry.model_id,
            "version_id": entry.version_id,
            "repo": entry.repo,
            "license": license_str,
        }
        LICENSES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def download_entry(entry: ModelEntry, comfy_root: Path) -> Path:
    target = entry.target_path(comfy_root)
    if entry.source == "huggingface":
        huggingface.download(entry.repo, entry.file, target)
        _record_license(entry, f"see huggingface.co/{entry.repo} (license on model card)")
    elif entry.source == "civitai":
        version = civitai.get_version(entry.version_id)
        # record license from the parent model page for the commercial-use audit
        try:
            model = civitai.get_model(entry.model_id)
            _record_license(entry, civitai.license_summary(model))
        except Exception:
            pass
        files = version.get("files", [])
        fmeta = next((f for f in files if f["name"] == entry.file), None) or \
            civitai.primary_file(version)
        url = civitai.download_url(entry.version_id, fmeta)
        _http_download(url, target, entry.local_name)
    else:  # plain url
        _http_download(entry.url, target, entry.local_name)

    if entry.sha256:
        actual = sha256_of(target)
        if actual != entry.sha256.upper():
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"{entry.name}: downloaded file failed its integrity check "
                f"(expected {entry.sha256[:12]}…, got {actual[:12]}…). "
                "The download may have been corrupted — run Setup again to retry.")
    return target


def sync(manifest: Manifest, profile: str, comfy_root: Path,
         include_optional: bool = True, status_cb=None) -> SyncReport:
    """Download everything the profile needs. Idempotent. status_cb(done, total, name)."""
    entries = [e for e in manifest.entries_for(profile, include_optional)
               if not is_present_and_valid(e, comfy_root)]
    report = SyncReport()
    already = [e.name for e in manifest.entries_for(profile, include_optional)
               if is_present_and_valid(e, comfy_root)]
    report.skipped.extend(already)

    total = len(entries)
    done = 0
    lock = threading.Lock()

    def work(entry: ModelEntry):
        nonlocal done
        try:
            download_entry(entry, comfy_root)
            with lock:
                done += 1
                report.downloaded.append(entry.name)
                if status_cb:
                    status_cb(done, total, entry.local_name)
        except Exception as exc:
            with lock:
                done += 1
                report.failed[entry.name] = str(exc)
                if status_cb:
                    status_cb(done, total, entry.local_name)

    if entries:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = [pool.submit(work, e) for e in entries]
            for f in as_completed(futures):
                f.result()
    return report


def dry_run(manifest: Manifest, profile: str, comfy_root: Path,
            include_optional: bool = True) -> dict:
    """What would be downloaded, and total size, without touching the network."""
    todo, have = [], []
    for entry in manifest.entries_for(profile, include_optional):
        (have if is_present_and_valid(entry, comfy_root) else todo).append(entry)
    free_gb = shutil.disk_usage(comfy_root if comfy_root.exists() else Path.cwd()).free / 1e9
    return {
        "profile": profile,
        "to_download": [(e.name, e.size_gb) for e in todo],
        "already_present": [e.name for e in have],
        "total_gb": round(sum(e.size_gb for e in todo), 1),
        "free_disk_gb": round(free_gb, 1),
    }


def verify(manifest: Manifest, profile: str, comfy_root: Path) -> dict:
    """Deep-verify hashes of present files; list missing ones."""
    missing, bad, good = [], [], []
    for entry in manifest.entries_for(profile):
        target = entry.target_path(comfy_root)
        if not target.exists():
            missing.append(entry.name)
        elif entry.sha256 and sha256_of(target, progress=True) != entry.sha256.upper():
            bad.append(entry.name)
        else:
            good.append(entry.name)
    return {"ok": good, "missing": missing, "corrupt": bad}
