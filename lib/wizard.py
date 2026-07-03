"""Browser setup wizard — the beginner-facing face of the provisioner.

A tiny stdlib HTTP server + single embedded page. No terminal knowledge needed:
Setup.bat / Setup.command / the RunPod first boot all land here.
"""
from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import civitai, comfy, config, downloader, workflows
from .manifest import ModelEntry, append_user_lora, load_manifest

STATE = {
    "phase": "idle",       # idle | installing | downloading | done | error
    "message": "",
    "done": 0, "total": 0, "current": "",
    "errors": [],
}
_lock = threading.Lock()


def _set(**kw):
    with _lock:
        STATE.update(kw)


def _log_error(exc: Exception) -> str:
    config.LOG_DIR.mkdir(exist_ok=True)
    log = config.LOG_DIR / f"wizard-error-{datetime.now():%Y%m%d-%H%M%S}.log"
    log.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    return str(log)


FRIENDLY = {
    "disk": "Not enough disk space. Free some space, then choose 'Check for missing files'.",
    "civitai": "Civitai rejected your key — it may have been copied with an extra space. "
               "Re-enter it on the key screen.",
}


def _provision_thread(opts: dict):
    """The whole install pipeline, reporting progress into STATE."""
    try:
        manifest = load_manifest()
        # 1. ComfyUI
        _set(phase="installing", message="Setting up ComfyUI…")
        root = config.comfy_dir()
        if opts.get("install_comfy"):
            base = Path(opts.get("install_dir", "~")).expanduser()
            root = comfy.install_comfy(base / "ComfyUI",
                                       log=lambda m: _set(message=m))
        if not root or not comfy.is_comfy_dir(Path(root)):
            raise RuntimeError("ComfyUI folder is not set or not valid.")
        root = Path(root)
        config.save_env(COMFY_DIR=str(root))

        _set(message="Installing helper node packs…")
        failures = comfy.install_custom_nodes(root, log=lambda m: _set(message=m))
        for f in failures:
            STATE["errors"].append(f"Node pack: {f}")

        # 2. Models
        profile = opts.get("profile") or comfy.pick_profile(
            comfy.detect_gpu().get("vram_gb"))
        plan = downloader.dry_run(manifest, profile, root,
                                  include_optional=opts.get("optional", True))
        _set(phase="downloading", total=len(plan["to_download"]), done=0,
             message=f"Downloading {len(plan['to_download'])} files "
                     f"({plan['total_gb']} GB)…")
        report = downloader.sync(
            manifest, profile, root, include_optional=opts.get("optional", True),
            status_cb=lambda d, t, n: _set(done=d, total=t, current=n,
                                           message=f"Downloading model {d} of {t}: {n}"))
        for name, err in report.failed.items():
            STATE["errors"].append(f"{name}: {err}")

        # 3. Workflows
        _set(message="Generating and installing workflows…")
        workflows.generate_all(manifest)
        comfy.install_workflows(root, workflows.OUTPUT_DIR, log=lambda m: None)

        if report.failed:
            _set(phase="error",
                 message="Some downloads failed — everything else is ready. "
                         "Run Setup again and choose 'Check for missing files' to retry.")
        else:
            _set(phase="done",
                 message="Done! Open ComfyUI — the workflows are already in your "
                         "Workflows menu.")
    except Exception as exc:
        log = _log_error(exc)
        msg = str(exc) if isinstance(exc, (civitai.CivitaiError, RuntimeError)) \
            else f"Something unexpected went wrong. Details were saved to {log}."
        _set(phase="error", message=msg)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet for beginners
        pass

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(self._state())
        elif self.path == "/api/progress":
            with _lock:
                self._json(dict(STATE))
        else:
            self._json({"error": "not found"}, 404)

    def _state(self):
        gpu = comfy.detect_gpu()
        detected = comfy.detect_comfy()
        profile = comfy.pick_profile(gpu.get("vram_gb"))
        manifest = load_manifest()
        root = detected or Path("~")
        plan = downloader.dry_run(manifest, profile, Path(root).expanduser())
        gated = any(e.gated for e in manifest.entries_for(profile))
        return {
            "comfy_dir": str(detected) if detected else None,
            "gpu": gpu, "profile": profile,
            "profile_explanation": _explain_profile(gpu, profile),
            "civitai_key_set": bool(config.get("CIVITAI_API_TOKEN")),
            "hf_key_set": bool(config.get("HF_TOKEN")),
            "hf_needed": gated,
            "download_gb": plan["total_gb"],
            "free_gb": plan["free_disk_gb"],
            "missing_count": len(plan["to_download"]),
            "total_count": len(manifest.entries_for(profile)),
        }

    def do_POST(self):
        try:
            body = self._read_body()
            if self.path == "/api/comfy-dir":
                path = Path(body["path"]).expanduser()
                if not comfy.is_comfy_dir(path):
                    return self._json({"ok": False, "message":
                        "That folder doesn't look like ComfyUI (no main.py inside). "
                        "Pick the folder that contains main.py and the models folder."})
                config.save_env(COMFY_DIR=str(path))
                return self._json({"ok": True})
            if self.path == "/api/tokens":
                updates = {}
                if body.get("civitai") is not None:
                    updates["CIVITAI_API_TOKEN"] = body["civitai"].strip()
                if body.get("hf") is not None:
                    updates["HF_TOKEN"] = body["hf"].strip()
                config.save_env(**updates)
                ok = civitai.check_token() if updates.get("CIVITAI_API_TOKEN") else True
                return self._json({"ok": ok, "message":
                    "" if ok else FRIENDLY["civitai"]})
            if self.path == "/api/start":
                _set(phase="installing", message="Starting…", errors=[], done=0, total=0)
                threading.Thread(target=_provision_thread, args=(body,),
                                 daemon=True).start()
                return self._json({"ok": True})
            if self.path == "/api/add-lora":
                try:
                    entry = ModelEntry.model_validate(civitai.resolve_lora(body["url"]))
                    append_user_lora(entry)
                    root = config.comfy_dir()
                    if root:
                        downloader.download_entry(entry, root)
                    return self._json({"ok": True, "message":
                        f"Added and downloaded: {entry.local_name}"})
                except civitai.CivitaiError as exc:
                    return self._json({"ok": False, "message": str(exc)})
            if self.path == "/api/workflows":
                manifest = load_manifest()
                workflows.generate_all(manifest)
                root = config.comfy_dir()
                if root:
                    comfy.install_workflows(root, workflows.OUTPUT_DIR, log=lambda m: None)
                return self._json({"ok": True, "message": "Workflows reinstalled."})
            if self.path == "/api/launch":
                root = config.comfy_dir()
                if not root:
                    return self._json({"ok": False, "message": "ComfyUI folder not set."})
                comfy.launch(root)
                return self._json({"ok": True,
                                   "message": "ComfyUI is starting — it opens at "
                                              "http://127.0.0.1:8188 in a minute."})
            self._json({"error": "not found"}, 404)
        except Exception as exc:
            log = _log_error(exc)
            self._json({"ok": False, "message":
                f"Something unexpected went wrong (details saved to {log})."})


def _explain_profile(gpu: dict, profile: str) -> str:
    name = gpu.get("name", "your graphics card")
    vram = gpu.get("vram_gb")
    if profile == "local-12gb":
        return (f"{name} has {vram} GB of video memory, so we'll use slightly "
                "compressed versions of the big video models. Same workflows, "
                "a touch less detail, no out-of-memory crashes.")
    if vram:
        return (f"{name} has {vram} GB of video memory — plenty. "
                "You get the full-quality versions of every model.")
    return ("No NVIDIA graphics card detected — picking the standard set. "
            "If this is a cloud pod, that's normal.")


def run(port: int = 8189, open_browser: bool = True, host: str = "127.0.0.1"):
    config.load_env()
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Setup wizard running at {url}  (leave this window open)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ───────────────────────────── the page ─────────────────────────────

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>ComfyUI Setup</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,sans-serif;background:#14161a;color:#e8e8e8;margin:0}
 .wrap{max-width:640px;margin:40px auto;padding:0 20px}
 h1{font-size:26px} h2{font-size:19px;margin-top:0}
 .card{background:#1e2128;border:1px solid #2e323b;border-radius:12px;padding:24px;margin:16px 0}
 button{background:#4f8cff;color:#fff;border:0;border-radius:8px;padding:12px 22px;
        font-size:15px;cursor:pointer;margin:6px 6px 0 0}
 button.alt{background:#2e323b}
 button:disabled{opacity:.5;cursor:default}
 input[type=text],input[type=password]{width:100%;box-sizing:border-box;background:#14161a;
   border:1px solid #3a3f4a;border-radius:8px;color:#e8e8e8;padding:11px;font-size:14px;margin:6px 0}
 .muted{color:#9aa0ab;font-size:14px;line-height:1.5}
 .ok{color:#6fd08c}.bad{color:#ff7b72}
 .bar{background:#2e323b;border-radius:8px;height:14px;overflow:hidden;margin:12px 0}
 .bar>div{background:#4f8cff;height:100%;width:0%;transition:width .5s}
 a{color:#7ab3ff} .hidden{display:none}
 .big{font-size:17px}
</style></head><body><div class="wrap">
<h1>🎬 ComfyUI model setup</h1>
<div id="app" class="card"><h2>One moment…</h2>
<p class="muted">Checking your computer…</p></div>
</div>
<script>
const $=s=>document.querySelector(s);
let S={};
async function api(p,body){const r=await fetch(p,body?{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  return r.json();}
async function refresh(){S=await api('/api/state');route();}
function esc(t){const d=document.createElement('div');d.textContent=t||'';return d.innerHTML;}

function route(){
  if(!S.comfy_dir) return showComfy();
  if(!S.civitai_key_set||(S.hf_needed&&!S.hf_key_set)) return showKeys();
  if(S.missing_count===0) return showMenu();
  showConfirm();
}

function showComfy(){$('#app').innerHTML=`
 <h2>Step 1 · Where is ComfyUI?</h2>
 <p class="muted">We couldn't find ComfyUI automatically. If you already have it,
 paste the folder path below (the folder that contains <b>main.py</b>).
 If you don't have it yet, we'll install it for you.</p>
 <input type="text" id="cpath" placeholder="e.g. C:\\ComfyUI  or  /home/me/ComfyUI">
 <div><button onclick="setComfy()">Use this folder</button>
 <button class="alt" onclick="installComfy()">Install ComfyUI for me</button></div>
 <p id="cmsg" class="bad"></p>`;}
async function setComfy(){const r=await api('/api/comfy-dir',{path:$('#cpath').value});
 if(r.ok){await refresh();}else{$('#cmsg').textContent=r.message;}}
async function installComfy(){S.install=true;showKeys();}

function showKeys(){$('#app').innerHTML=`
 <h2>Step 2 · Your download key${S.hf_needed?'s':''}</h2>
 <p class="muted">Models are downloaded from <b>Civitai</b>, which needs a free key
 (like a library card). Get it here:
 <a href="https://civitai.com/user/account" target="_blank">civitai.com/user/account</a>
 — scroll to <b>API Keys</b>, click <b>Add API key</b>, copy it, paste below.</p>
 <input type="password" id="ck" placeholder="Civitai API key"
   value="${S.civitai_key_set?'••••saved••••':''}">
 ${S.hf_needed?`<p class="muted">One model (the Flux VAE) also needs a free
 <b>Hugging Face</b> account: create a token at
 <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a>
 (type: Read), and click "Agree and access repository" at
 <a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell" target="_blank">this page</a>.</p>
 <input type="password" id="hk" placeholder="Hugging Face token (starts with hf_)"
   value="${S.hf_key_set?'••••saved••••':''}">`:''}
 <div><button onclick="saveKeys()">Save and continue</button></div>
 <p id="kmsg" class="bad"></p>`;}
async function saveKeys(){
 const ck=$('#ck').value, hk=$('#hk')?$('#hk').value:null, b={};
 if(ck&&!ck.includes('••••'))b.civitai=ck;
 if(hk&&!hk.includes('••••'))b.hf=hk;
 const r=await api('/api/tokens',b);
 if(r.ok){await refresh(); if(S.install||!S.comfy_dir)showConfirm();}
 else $('#kmsg').textContent=r.message;}

function showConfirm(){$('#app').innerHTML=`
 <h2>Step 3 · The big download</h2>
 <p class="big">${esc(S.profile_explanation)}</p>
 <p class="muted">Download size: <b>${S.download_gb} GB</b> ·
 Free disk space: <b>${S.free_gb} GB</b>
 ${S.free_gb<S.download_gb*1.1?'<span class="bad"><br>⚠ That may not be enough space — free some up first.</span>':''}
 </p>
 <p class="muted">This is a one-time download and it's safe to interrupt —
 running Setup again continues where it left off.</p>
 ${S.install?'<p class="muted">ComfyUI itself will be installed first.</p>':''}
 <button onclick="start()">Yes, download everything</button>
 <button class="alt" onclick="showMenu()">Not now</button>`;}
async function start(){
 await api('/api/start',{install_comfy:!!S.install,
   install_dir:'~',optional:true});
 showProgress();}

function showProgress(){$('#app').innerHTML=`
 <h2 id="ph">Working…</h2><div class="bar"><div id="fill"></div></div>
 <p id="pmsg" class="muted"></p><p id="perr" class="bad"></p>
 <div id="pdone" class="hidden">
  <button onclick="launch()">🚀 Launch ComfyUI now</button></div>`;
 poll();}
async function poll(){const p=await api('/api/progress');
 $('#pmsg').textContent=p.message;
 if(p.total)$('#fill').style.width=Math.round(100*p.done/p.total)+'%';
 $('#perr').innerHTML=(p.errors||[]).map(esc).join('<br>');
 if(p.phase==='done'){$('#ph').textContent='✅ All done!';
   $('#fill').style.width='100%';$('#pdone').classList.remove('hidden');}
 else if(p.phase==='error'){$('#ph').textContent='⚠ Almost';}
 else setTimeout(poll,1500);}

function showMenu(){$('#app').innerHTML=`
 <h2>Everything's ready ✨</h2>
 <p class="muted">${S.missing_count===0
   ?'All '+S.total_count+' model files are in place.'
   :S.missing_count+' file(s) are missing.'}</p>
 <button onclick="checkFiles()">1 · Check for missing files</button><br>
 <button onclick="showLora()">2 · Add a LoRA (paste a Civitai link)</button><br>
 <button onclick="reWf()">3 · Reinstall workflows</button><br>
 <button onclick="launch()">4 · Launch ComfyUI</button><br>
 <button class="alt" onclick="showHelp()">Help</button>
 <p id="mmsg" class="muted"></p>`;}
async function checkFiles(){await api('/api/start',{optional:true});showProgress();}
function showLora(){$('#app').innerHTML=`
 <h2>Add a LoRA</h2>
 <p class="muted">Paste the address of a LoRA's page on Civitai (from your browser's
 address bar). Note: LoRAs of real, identifiable people are not allowed — using
 someone's face without consent is harmful and illegal in most places.</p>
 <input type="text" id="lurl" placeholder="https://civitai.com/models/…">
 <button onclick="addLora()">Add it</button>
 <button class="alt" onclick="showMenu()">Back</button>
 <p id="lmsg" class="muted"></p>`;}
async function addLora(){$('#lmsg').textContent='Downloading…';
 const r=await api('/api/add-lora',{url:$('#lurl').value});
 $('#lmsg').className=r.ok?'ok':'bad';$('#lmsg').textContent=r.message;}
async function reWf(){const r=await api('/api/workflows');$('#mmsg').textContent=r.message;}
async function launch(){const r=await api('/api/launch');
 const el=$('#mmsg')||$('#pmsg');if(el)el.textContent=r.message;
 setTimeout(()=>window.open('http://127.0.0.1:8188'),8000);}
function showHelp(){$('#app').innerHTML=`
 <h2>Help</h2>
 <ul class="muted">
  <li><b>Download stopped halfway?</b> Just run Setup again — it continues.</li>
  <li><b>"Civitai rejected your key"?</b> Re-copy it carefully (no spaces) via
      option 2 on the key screen — or make a new key.</li>
  <li><b>Red boxes in ComfyUI?</b> Use "Reinstall workflows", and inside ComfyUI
      open Manager → "Install missing custom nodes".</li>
  <li><b>Out of memory during video?</b> Lower the resolution in the workflow
      (960x544), or enable both Lightning LoRA nodes.</li>
  <li>More: see TROUBLESHOOTING.md in this folder.</li>
 </ul><button class="alt" onclick="showMenu()">Back</button>`;}
refresh();
</script></body></html>
"""
