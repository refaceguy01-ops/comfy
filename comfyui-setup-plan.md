# ComfyUI Image/Video Generation Stack — Implementation Plan

Spec for Claude Code / Cowork. Goal: a single repo that provisions a complete ComfyUI
environment — locally (Windows/Linux, 24GB VRAM target) or on a RunPod-style cloud pod —
downloads all models/LoRAs from a declarative manifest, and generates ready-to-load
ComfyUI workflow JSON files.

**SCOPE: all workflows are image-driven.** Video = image-to-video (I2V) only. Images =
image-to-image and reference-image workflows only. No text-to-image, no text-to-video.
Do not download T2V model variants.

---

## 1. Repository layout

```
comfy-provisioner/
├── manifest.yaml            # single source of truth: every model, LoRA, VAE, encoder
├── provision.py             # CLI entry point: install / download / verify / workflows
├── lib/
│   ├── downloader.py        # HF + Civitai download engine (resume, hash, parallel)
│   ├── civitai.py           # Civitai API client (token auth, version resolution)
│   ├── huggingface.py       # hf_hub_download wrapper
│   ├── comfy.py             # ComfyUI install/update, custom node management
│   └── workflows.py         # workflow JSON generator (templated)
├── workflows/               # generated output
│   ├── wan22_i2v_remix.json
│   ├── wan22_t2v_base.json
│   ├── chroma_txt2img.json
│   └── sdxl_realistic.json
├── runpod/
│   ├── Dockerfile           # optional: baked image
│   └── start.sh             # pod bootstrap script (network volume aware)
└── .env.example             # CIVITAI_API_TOKEN, HF_TOKEN, COMFY_DIR
```

## 2. manifest.yaml — declarative model list

Each entry declares: `name`, `source` (`huggingface` | `civitai` | `url`), source-specific
ID (repo+filename, or Civitai model/version ID), `dest` (ComfyUI subfolder), `sha256`
(optional but preferred), `tags` (e.g. `video`, `image`, `required`, `optional`), and
`profile` (`local-24gb`, `local-12gb`, `cloud`). The downloader filters by profile so a
12GB machine pulls fp8/GGUF variants and a cloud pod pulls fp16.

### Video stack (Wan 2.2)

| Item | Source | Dest |
|---|---|---|
| Wan 2.2 Remix v3 — I2V high-noise + low-noise (primary creative model; native uncensored, no LoRA dependency) | Civitai | `models/diffusion_models/` |
| Wan 2.2 14B I2V fp8_scaled (stock baseline, high+low noise pair) | HF `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` | `models/diffusion_models/` |
| PalinGenesis I2V finetune (alternate, scaled FP8) | Civitai/HF | `models/diffusion_models/` |
| umt5_xxl_fp8_e4m3fn_scaled text encoder | HF (Comfy-Org repackage) | `models/text_encoders/` |
| wan_2.1_vae.safetensors (14B models) + wan2.2_vae (5B) | HF | `models/vae/` |
| Lightx2v / Lightning 4-step LoRAs (high + low noise variants) | HF/Civitai | `models/loras/` |
| RIFE frame-interpolation model (via ComfyUI-Frame-Interpolation) | node-managed | — |
| FlashVSR / 4x upscale model for video upscaling | Civitai/HF | `models/upscale_models/` |

### Image stack

| Item | Source | Dest |
|---|---|---|
| Chroma (Flux-arch, fully uncensored, best realism/prompt adherence; needs Flux VAE + T5/CLIP-L encoders) | HF | `models/diffusion_models/`, `models/vae/`, `models/text_encoders/` |
| RealVisXL V5 (top SDXL photoreal checkpoint) | Civitai | `models/checkpoints/` |
| Juggernaut XL (photoreal alternate, lighter) | Civitai | `models/checkpoints/` |
| LUSTIFY (SDXL, explicit-capable photoreal) | Civitai | `models/checkpoints/` |
| Pony Diffusion V6 XL (optional; largest LoRA ecosystem, stylized) | Civitai | `models/checkpoints/` |
| **IP-Adapter Plus (SDXL) + IP-Adapter FaceID models** — reference-image conditioning (subject/style transfer, character consistency) | HF `h94/IP-Adapter` | `models/ipadapter/` |
| **CLIP-Vision encoder (ViT-H)** — required by IP-Adapter | HF | `models/clip_vision/` |
| **ControlNet Union (SDXL)** — pose/depth/edge guidance from a reference image, one model covers all modes | HF/Civitai | `models/controlnet/` |
| SDXL anatomy/detail LoRAs — user-curated section, one line each | Civitai | `models/loras/` |
| 4x-UltraSharp or similar upscaler | Civitai | `models/upscale_models/` |

LoRA policy: the manifest ships with the structural LoRAs (Lightning/speed, detail,
anatomy-correction). Subject/style LoRAs are added by the user by pasting a Civitai
version ID — `provision.py add-lora <civitai-url>` resolves it, appends to the manifest,
and downloads. **Do not add LoRAs trained on real identifiable people.**

## 3. Downloader requirements (lib/downloader.py)

- Civitai: authenticate with `CIVITAI_API_TOKEN` (required for many hosted files); resolve
  model-page URLs → latest version → primary safetensors file via the public API
  (`/api/v1/models/{id}`, `/api/v1/model-versions/{id}`); follow the
  `?token=` download pattern.
- Hugging Face: use `huggingface_hub` with `HF_TOKEN` for gated repos; symlink-free copies
  into the ComfyUI tree (or `local_dir` mode).
- Resume partial downloads (HTTP Range), verify SHA-256 when present in manifest or API
  response, skip files already present+verified (idempotent — safe to rerun on every pod boot).
- Parallelism: 2–3 concurrent downloads max (Civitai rate limits), progress bars via `tqdm`.
- `--dry-run` prints total download size per profile before committing (full stack is
  ~80–120 GB; the 24GB-local profile with fp8 variants ~60 GB).

## 4. ComfyUI + custom nodes (lib/comfy.py)

- Install/update ComfyUI to latest (Wan 2.2 nodes require a recent version).
- Custom nodes via ComfyUI-Manager CLI or git clone + `pip install -r`:
  - `ComfyUI-Manager`
  - `ComfyUI-VideoHelperSuite` (video combine/save)
  - `ComfyUI-Frame-Interpolation` (RIFE)
  - `ComfyUI-KJNodes` and optionally Kijai's `ComfyUI-WanVideoWrapper` (cutting-edge Wan optimizations)
  - `ComfyUI-GGUF` (for low-VRAM quantized variants)
  - `ComfyUI_IPAdapter_plus` (reference-image conditioning for the image workflows)
  - `comfyui_controlnet_aux` (preprocessors: pose/depth/edge extraction from reference images)
  - SageAttention install step for the cloud profile (10–20% speedup on Wan renders).

## 5. Workflow JSON generation (lib/workflows.py)

Generate native-format ComfyUI workflow JSONs (nodes + links), not just API format, so they
open in the UI. Base them on the official ComfyUI Wan 2.2 I2V template, then swap model
names to manifest values. **Every workflow starts from a Load Image node** — the reference
image is the primary input; the text prompt only steers what happens to/around it.

1. **wan22_i2v_remix.json** — Load Image (start frame) → dual UNETLoader (Remix high/low
   noise) → umt5 CLIPLoader → Wan VAE → two-stage KSamplerAdvanced (high-noise steps 0–N,
   low-noise N–end) → optional Lightning LoRA loaders (bypassed by default toggle; when
   enabled: steps 4–8, CFG 1) → VAE Decode → RIFE interpolation → Video Combine.
   Defaults: 720p, 81 frames, 16→32 fps after interpolation.
2. **wan22_i2v_firstlast.json** — first-frame + last-frame variant (two Load Image nodes
   feeding WanFirstLastFrameToVideo) for controlled shot transitions, same two-stage
   sampler structure. Stock Wan 2.2 I2V pair by default, switchable to Remix.
3. **sdxl_img2img_reference.json** — the workhorse image workflow. Load Image (reference)
   → CheckpointLoader (RealVisXL default, switchable to Juggernaut/LUSTIFY) → IP-Adapter
   Plus (reference subject/style transfer, weight widget ~0.6–0.8) → optional ControlNet
   Union branch (pose/depth/edge preprocessor from the same or a second reference image,
   bypassed by default) → LoRA stack (default weight 0.7) → VAE Encode of the reference →
   KSampler with **denoise widget (0.3 = subtle variation … 0.75 = heavy reimagining,
   default 0.55)** → hires pass at denoise ≤0.5 → 4x upscale. A Note node explains the
   denoise dial in plain English.
4. **sdxl_faceid_character.json** — IP-Adapter FaceID variant for keeping a consistent
   (fictional) character across shots: face reference image → FaceID conditioning +
   full-body reference via standard IP-Adapter → same sampler chain. This is the film
   pre-viz / character-consistency workflow.
5. **chroma_img2img.json** — Chroma UNET + T5/CLIP-L + Flux VAE, Load Image → VAE Encode →
   sampler at adjustable denoise, ~26–30 steps, for maximum-realism reworking of a
   reference image. (Chroma's reference-conditioning ecosystem is thinner than SDXL's;
   this workflow is denoise-based img2img. Claude Code should check at build time whether
   a mature Chroma/Flux IP-Adapter or Redux-style reference node exists and add it if so.)

Each workflow embeds a Note node documenting settings, file expectations, and what the
reference image controls.

## 6. RunPod / cloud profile

- `runpod/start.sh`: mounts the network volume at `/workspace`, points `COMFY_DIR` and all
  model dirs there via `extra_model_paths.yaml`, runs `provision.py sync` (idempotent), then
  launches ComfyUI with `--listen`. Models persist on the volume across pods; only the
  container is ephemeral.
- Recommended pod: 24–48GB VRAM (4090/L40S/A6000-class) for fp8 14B Wan; 80GB (A100/H100)
  for fp16.
- Optional Dockerfile baking ComfyUI + nodes (not models) for fast cold starts.

## 7. User interface — BEGINNER-FIRST (critical requirement)

**The end user has NO programming experience.** They know how to use ComfyUI and rent a
RunPod pod, nothing more. They must never need to open a terminal, edit a config file,
or understand what Python is. Design everything around this.

### Local: one double-clickable launcher

- Ship `Setup.bat` (Windows) and `Setup.command` (Mac/Linux) at the repo root. Double-click
  is the entire install story. The script silently bootstraps its own portable Python
  (embedded distribution / uv) — never ask the user to install Python or Git.
- The launcher opens an **interactive wizard** (simple menu, or preferably a minimal local
  web page that auto-opens in the browser) that walks through:
  1. "Where is ComfyUI?" → auto-detect common install paths; offer "Install it for me."
  2. Detect GPU and VRAM automatically → silently pick the right variant set (fp8 vs fp16
     vs GGUF) and explain the choice in plain English. Never surface the word "profile."
  3. "Paste your Civitai API key" → link directly to the exact Civitai settings page,
     with a screenshot in the README showing where the key lives. Same for Hugging Face
     only if a gated model is in the manifest. Keys are saved automatically — the user
     never sees or edits a .env file.
  4. Show total download size vs. free disk space, ask one Yes/No, then download with a
     single overall progress bar ("Downloading model 4 of 17…").
  5. Finish screen: "Done! Open ComfyUI — the workflows are already in your Workflows
     menu." Plus a "Launch ComfyUI now" button.
- Re-running the launcher shows a simple menu: **[1] Check for missing files
  [2] Add a LoRA (paste a Civitai link)  [3] Reinstall workflows  [4] Help.**
  "Add a LoRA" accepts a pasted Civitai page URL and handles the rest (still rejecting
  real-person LoRAs, with a friendly explanation of why).
- Every error message in plain English with the fix: "Civitai rejected your key — it may
  have been copied with an extra space. Choose option 4 to re-enter it." Never show a
  Python traceback; log those to `logs/` instead.

### RunPod: zero-terminal template

- Publish/document a **RunPod template** whose start command runs `start.sh` automatically.
  User flow: create a network volume → deploy the template → wait → click the ComfyUI
  connect link. They never open the pod terminal.
- First boot detects an empty volume and serves the same browser wizard on an exposed
  HTTP port to collect the Civitai key and confirm the big download, then provisions the
  volume and starts ComfyUI. Later boots skip straight to ComfyUI in under a minute.
- README includes a click-by-click, screenshot-annotated RunPod walkthrough: which GPU to
  rent (with hourly-cost ballparks), what volume size to pick (recommend 150 GB+), and
  exactly which template fields to fill in.

### Documentation style

- README written for a non-programmer: numbered steps, screenshots, zero jargon. A
  "What is all this?" section explaining checkpoints, LoRAs, and VAEs in two sentences each.
- Troubleshooting page covering the five likeliest failures: out of disk space, bad API
  key, out of VRAM (→ the wizard's "switch to lighter models" option), interrupted
  download (→ just run Setup again), and red nodes in ComfyUI (→ "Reinstall workflows").

### Advanced CLI (kept underneath, for power users only)

```
provision.py install / sync / workflows / add-lora <url> / verify / dry-run
```
The wizard is a friendly shell over these commands; nothing requires them.

## 8. Build order for Claude Code

1. Scaffold repo, `.env` handling, manifest schema + validation (pydantic).
2. Civitai + HF clients with one real small-file integration test each.
3. Downloader engine (resume/hash/skip), then populate manifest with the tables above —
   resolve each entry's current version IDs/URLs at build time via the Civitai/HF APIs
   rather than hardcoding from memory.
4. ComfyUI installer + node manager.
5. Workflow generator — start by fetching the official Wan 2.2 template JSONs from
   docs.comfy.org as ground truth, then parametrize.
6. **The wizard + double-click launchers (Setup.bat / Setup.command)** — treat this as a
   first-class deliverable, not a wrapper afterthought. Test by pretending you cannot use
   a terminal: double-click must carry a fresh machine to "workflows loaded in ComfyUI."
7. RunPod template + start script with first-boot browser wizard; test full cold-boot on
   a pod with an empty network volume, terminal never opened.
8. README + troubleshooting written for a non-programmer (screenshots, numbered steps),
   with local (Windows + Linux) and RunPod walkthroughs and a plain-English GPU/VRAM
   decision table.
9. Final acceptance test: hand the repo to someone (or simulate someone) who only knows
   how to double-click and use ComfyUI. Every point where they'd get stuck is a bug.

## Notes & guardrails

- Everything here is Apache-2.0 or community-licensed for local use; check each Civitai
  model's license field before commercial film use — the downloader should record the
  license string from the API into a `licenses.json` audit file.
- No LoRAs of real identifiable people; no content involving minors — keep these as
  hard rules in the README and in `add-lora` (reject Civitai entries tagged as
  real-person/celebrity).
- Because every workflow is reference-image driven (and FaceID specifically preserves a
  face), the README must state plainly: reference images of real people may only be used
  with their consent / proper licensing — never in explicit contexts without it, which is
  illegal in most jurisdictions (NCII laws). For character-consistency work, generate a
  fictional face first and use that as the FaceID reference.
