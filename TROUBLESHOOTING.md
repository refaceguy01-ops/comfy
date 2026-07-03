# Troubleshooting

The five things that actually go wrong, and how to fix each one.
(Nothing here needs a terminal.)

## 1. "Not enough disk space" / download stops near the end

The full model set is 50–125 GB depending on your graphics card.

- Free up space on the drive that holds your **ComfyUI folder** (that's where
  models go — not this folder).
- Then run **Setup** again and choose **[1] Check for missing files**. It only
  downloads what's still missing.

## 2. "Civitai rejected your key"

- The key was probably copied with an extra space, or only partially.
- Go to [civitai.com/user/account](https://civitai.com/user/account) → **API
  Keys** → make a **new** key → copy it fresh.
- Run **Setup**, it will show the key screen again — paste and continue.
- Hugging Face version of the same problem: make sure your token *starts with
  `hf_`* and that you clicked **"Agree and access repository"** on the
  [FLUX.1-schnell page](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
  while logged in.

## 3. Out of VRAM (video generation crashes / computer freezes)

- In the video workflow, lower the resolution: change **1280 × 720** to
  **960 × 544** in the "Video size / length" node.
- Or enable the two purple **Lightning LoRA** nodes (right-click → Bypass to
  toggle) and set both samplers to 8 steps, cfg 1.0 — much lighter *and* faster.
- Still crashing? Run **Setup** → the wizard picks lighter model versions for
  12 GB cards automatically. If you upgraded/downgraded your GPU, just run it
  again.

## 4. Download was interrupted (closed laptop, lost Wi-Fi…)

Nothing is lost. Run **Setup** again → **[1] Check for missing files**.
Partial files resume where they stopped; finished files are skipped.

## 5. Red nodes when opening a workflow in ComfyUI

Red = ComfyUI is missing a node pack or a model file the workflow needs.

1. Run **Setup** → **[3] Reinstall workflows** (also re-installs node packs).
2. Inside ComfyUI: **Manager → Install missing custom nodes → Restart**.
3. Still red on the FaceID workflow only? That one needs a component called
   InsightFace that occasionally fails to install on Windows. Open ComfyUI
   **Manager → pip install** and enter `insightface onnxruntime`. If it errors
   about "Microsoft Visual C++", install "Visual Studio Build Tools" (free,
   from Microsoft) and try once more — or simply use the regular
   `sdxl_img2img_reference` workflow, which needs nothing extra.

## Something else?

Every crash writes a detailed report into the **`logs/`** folder next to
Setup.bat. If you ask for help (GitHub issue, Discord…), attach the newest file
from there — it contains no keys or personal data.
