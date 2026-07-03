"""Hugging Face download wrapper: symlink-free copies straight into the ComfyUI tree."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from . import config


class HFError(RuntimeError):
    """Raised with a plain-English, user-facing message."""


def download(repo: str, filename: str, target: Path) -> Path:
    """Download repo/filename to the exact `target` path (no symlinks, no cache tree).

    huggingface_hub handles resume + integrity itself; we download into the target
    directory then move the file to its final name.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    token = config.get("HF_TOKEN") or None
    try:
        got = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=target.parent / ".hf_tmp",
            token=token,
        )
    except GatedRepoError as e:
        raise HFError(
            f"'{repo}' needs a (free) Hugging Face account: log in at huggingface.co, "
            f"open huggingface.co/{repo}, click 'Agree and access repository', then "
            "paste an access token in the setup wizard.") from e
    except RepositoryNotFoundError as e:
        raise HFError(f"Hugging Face repo '{repo}' was not found — it may have moved.") from e

    got = Path(got)
    if got.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        got.replace(target)
    _cleanup_tmp(target.parent / ".hf_tmp")
    return target


def _cleanup_tmp(tmp: Path) -> None:
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


def check_gated_access(repo: str) -> bool:
    """True if the current HF_TOKEN can access `repo` (used by the wizard preflight)."""
    from huggingface_hub import auth_check
    try:
        auth_check(repo, token=config.get("HF_TOKEN") or None)
        return True
    except Exception:
        return False
