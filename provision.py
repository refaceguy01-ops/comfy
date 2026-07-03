# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyyaml>=6",
#   "pydantic>=2",
#   "requests>=2.31",
#   "tqdm>=4.66",
#   "huggingface_hub>=0.23",
# ]
# ///
"""comfy-provisioner CLI.

Beginners never see this — Setup.bat / Setup.command run `provision.py wizard`.
Power-user commands:

    provision.py install            install/update ComfyUI + custom nodes
    provision.py sync               download everything for the detected profile
    provision.py dry-run            show what would download + total size
    provision.py verify             deep-check hashes, list missing/corrupt files
    provision.py workflows          regenerate + install workflow JSONs
    provision.py add-lora <url>     add a Civitai LoRA to the manifest + download
    provision.py wizard             browser setup wizard (the default)

Options: --profile local-12gb|local-24gb|cloud|cloud-80gb   --comfy-dir PATH
         --required-only   --port N (wizard)   --no-browser (wizard)
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from lib import comfy, config, downloader
from lib.manifest import PROFILES, load_manifest


def _log_exception() -> Path:
    """Tracebacks go to logs/, never to a beginner's screen."""
    config.LOG_DIR.mkdir(exist_ok=True)
    log = config.LOG_DIR / f"error-{datetime.now():%Y%m%d-%H%M%S}.log"
    log.write_text(traceback.format_exc(), encoding="utf-8")
    return log


def _resolve_comfy(args) -> Path:
    if args.comfy_dir:
        root = Path(args.comfy_dir)
    else:
        root = comfy.detect_comfy()
    if not root:
        sys.exit("ComfyUI not found. Run the setup wizard (just run provision.py), "
                 "or pass --comfy-dir PATH.")
    if comfy.is_comfy_dir(root):  # don't persist scratch/test paths
        config.save_env(COMFY_DIR=str(root))
    return root


def _resolve_profile(args) -> str:
    if args.profile:
        return args.profile
    gpu = comfy.detect_gpu()
    profile = comfy.pick_profile(gpu.get("vram_gb"))
    print(f"GPU: {gpu.get('name', 'not detected')} "
          f"({gpu.get('vram_gb', '?')} GB) -> profile {profile}")
    return profile


def cmd_install(args):
    root = comfy.detect_comfy()
    if root:
        print(f"ComfyUI found at {root} — updating.")
        comfy.update_comfy(root)
    else:
        dest = Path(args.comfy_dir) if args.comfy_dir else Path.home() / "ComfyUI"
        root = comfy.install_comfy(dest)
    config.save_env(COMFY_DIR=str(root))
    failures = comfy.install_custom_nodes(root)
    if args.profile in ("cloud", "cloud-80gb"):
        comfy.install_sageattention(root)
    for f in failures:
        print(f"  ! node install failed: {f}")
    print(f"ComfyUI ready at {root}")


def cmd_nodes(args):
    """(Re)install custom node packs + their python deps into ComfyUI's python."""
    root = _resolve_comfy(args)
    failures = comfy.install_custom_nodes(root)
    for f in failures:
        print(f"  ! {f}")
    print("node packs done" + (f" ({len(failures)} failed)" if failures else ""))


def cmd_sync(args):
    root = _resolve_comfy(args)
    profile = _resolve_profile(args)
    manifest = load_manifest()
    plan = downloader.dry_run(manifest, profile, root, not args.required_only)
    print(f"{len(plan['to_download'])} files to download "
          f"({plan['total_gb']} GB, {plan['free_disk_gb']} GB free)")
    report = downloader.sync(manifest, profile, root, not args.required_only,
                             status_cb=lambda d, t, n: print(f"[{d}/{t}] {n}"))
    print(f"downloaded {len(report.downloaded)}, already present {len(report.skipped)}, "
          f"failed {len(report.failed)}")
    for name, err in report.failed.items():
        print(f"  ! {name}: {err}")
    if report.failed:
        sys.exit(1)


def cmd_dry_run(args):
    root = _resolve_comfy(args)
    profile = _resolve_profile(args)
    plan = downloader.dry_run(load_manifest(), profile, root, not args.required_only)
    print(json.dumps(plan, indent=2))


def cmd_verify(args):
    root = _resolve_comfy(args)
    profile = _resolve_profile(args)
    result = downloader.verify(load_manifest(), profile, root)
    print(json.dumps(result, indent=2))
    if result["missing"] or result["corrupt"]:
        sys.exit(1)


def cmd_workflows(args):
    from lib import workflows
    manifest = load_manifest()
    written = workflows.generate_all(manifest, profile=_resolve_profile(args))
    print(f"generated {len(written)} workflows in {workflows.OUTPUT_DIR}")
    root = comfy.detect_comfy()
    if root:
        comfy.install_workflows(root, workflows.OUTPUT_DIR)
    else:
        print("ComfyUI not detected — generated only; run the wizard to install them.")


def cmd_add_lora(args):
    from lib import civitai
    from lib.manifest import ModelEntry, append_user_lora
    try:
        entry_dict = civitai.resolve_lora(args.url)
    except civitai.CivitaiError as exc:
        sys.exit(str(exc))
    entry = ModelEntry.model_validate(entry_dict)
    append_user_lora(entry)
    print(f"Added '{entry.name}' to the manifest.")
    root = comfy.detect_comfy()
    if root:
        downloader.download_entry(entry, root)
        print(f"Downloaded to {entry.target_path(root)}")


def cmd_wizard(args):
    from lib import wizard
    wizard.run(port=args.port, open_browser=not args.no_browser,
               host=getattr(args, "host", "127.0.0.1"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="provision.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--comfy-dir")
    parser.add_argument("--required-only", action="store_true",
                        help="skip entries tagged optional")
    sub = parser.add_subparsers(dest="command")
    for name in ("install", "sync", "download", "dry-run", "verify", "workflows", "nodes"):
        sub.add_parser(name)
    p_lora = sub.add_parser("add-lora")
    p_lora.add_argument("url")
    p_wiz = sub.add_parser("wizard")
    p_wiz.add_argument("--port", type=int, default=8189)
    p_wiz.add_argument("--no-browser", action="store_true")
    p_wiz.add_argument("--host", default="127.0.0.1",
                       help="bind address (0.0.0.0 for the RunPod first-boot wizard)")

    args = parser.parse_args(argv)
    handlers = {
        "install": cmd_install, "sync": cmd_sync, "download": cmd_sync,
        "dry-run": cmd_dry_run, "verify": cmd_verify, "workflows": cmd_workflows,
        "nodes": cmd_nodes, "add-lora": cmd_add_lora, "wizard": cmd_wizard,
    }
    command = args.command or "wizard"
    if command == "wizard" and not hasattr(args, "port"):
        args.port, args.no_browser = 8189, False

    try:
        handlers[command](args)
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception:
        log = _log_exception()
        sys.exit(f"Something went wrong. A detailed log was saved to:\n  {log}\n"
                 "Common fixes: check your internet connection, free up disk space, "
                 "or re-run Setup and choose 'Check for missing files'.")


if __name__ == "__main__":
    main()
