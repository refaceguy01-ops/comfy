# ComfyUI Image & Video Studio — one-click setup

This folder sets up everything you need to make **images from reference images**
and **videos from still images** in ComfyUI: the models, the helper nodes, and
ready-made workflows that appear straight in ComfyUI's Workflows menu.

You do **not** need to know anything about Python, terminals, or config files.

---

## Install on your own PC (Windows)

1. Double-click **`Setup.bat`**.
2. A browser page opens and walks you through three steps:
   - **Where is ComfyUI?** It usually finds it by itself. If you don't have
     ComfyUI yet, click *"Install it for me."*
   - **Your download key.** Models come from [Civitai](https://civitai.com), which
     needs a free key (think: library card). The wizard links you to the exact
     page — [civitai.com/user/account](https://civitai.com/user/account), scroll
     to **API Keys → Add API key**, copy, paste. Done once, saved forever.
   - **The big download.** The wizard shows the total size (roughly 50–125 GB
     depending on your graphics card) and your free disk space. Click Yes and
     watch one progress bar.
3. When it says **Done!**, click **Launch ComfyUI** — the workflows are already
   in ComfyUI's **Workflows** menu.

On Mac/Linux, double-click **`Setup.command`** instead. Everything else is identical.

**Run Setup again anytime** to get a simple menu:
**[1] Check for missing files · [2] Add a LoRA · [3] Reinstall workflows · [4] Help.**
Interrupted downloads always continue where they left off.

## Install on RunPod (rented cloud GPU)

1. On [runpod.io](https://runpod.io): **Storage → New Network Volume** —
   pick **150 GB or more** (the models live here permanently; pods come and go).
2. **Deploy a pod** using this repo's Docker image as a template
   (see `runpod/Dockerfile` header for the template fields), attach your volume,
   and expose HTTP ports **8188** and **8189**.
3. **First boot only:** open the pod's **Connect → HTTP 8189** link — the same
   setup wizard appears in your browser. Paste your Civitai key, confirm the
   download, wait (the pod does everything itself).
4. Every boot after that: click **Connect → HTTP 8188** and you're in ComfyUI
   in under a minute.

### Which GPU should I rent?

| GPU | VRAM | Good for | Ballpark cost |
|---|---|---|---|
| RTX 4090 / L40S / A6000 | 24–48 GB | everything here, standard quality | $ |
| A100 / H100 | 80 GB | the fp16 "maximum quality" video tier | $$$ |

24–48 GB is the sweet spot; only rent 80 GB if you specifically want fp16 video.

## What's in the box

**Video (image → video, Wan 2.2):**
- `wan22_i2v_remix` — turn one still image into a ~5s, 720p, 32fps video.
  Uses the Wan 2.2 Remix creative model; optional 4-step "Lightning" turbo mode.
- `wan22_i2v_firstlast` — give a first **and** last frame; it animates between
  them. Perfect for planned shot transitions.

**Images (reference image → new image, SDXL & Chroma):**
- `sdxl_img2img_reference` — the workhorse. Your reference image goes in; the
  **denoise dial** decides how much changes (0.3 = touch-up … 0.75 = reimagine).
  IP-Adapter carries the subject/style over; optional ControlNet locks the pose.
- `sdxl_faceid_character` — keeps the **same fictional character's face** across
  many shots (film pre-viz). Face reference + body/style reference in, new shot out.
- `chroma_img2img` — maximum-realism reworking of a reference image.

Every workflow has a yellow **README note** inside explaining its dials in plain
English.

### What is all this? (30-second glossary)

- **Checkpoint** — the big "brain" file that actually paints images. Different
  checkpoints have different styles (RealVis = photoreal, Pony = stylized).
- **LoRA** — a small add-on file that teaches a checkpoint one specific thing
  (a style, an outfit, better hands). Stackable, adjustable strength.
- **VAE** — the translator between the model's internal math and actual pixels.
  Wrong VAE = washed-out colors. Ours are matched automatically.

## Ground rules (please read)

- **No real people without consent.** These workflows are reference-image driven,
  and the FaceID workflow deliberately preserves a face. Using a real person's
  likeness without their consent — especially in explicit content — is harmful
  and **illegal in most jurisdictions** (non-consensual intimate imagery laws).
  For consistent characters, generate a fictional face first and use *that* as
  your reference. The "Add a LoRA" feature automatically refuses LoRAs of real
  identifiable people.
- **Nothing involving minors.** Ever. Hard rule, also enforced in the tooling.
- **Commercial film use:** each downloaded model's license terms are recorded in
  `licenses.json` — check the entries (especially `allowCommercialUse`) before
  using outputs commercially.

## If something goes wrong

See **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — it covers the five usual
suspects (disk space, bad key, out of VRAM, interrupted downloads, red nodes)
with copy-paste-free fixes.

---

<details>
<summary><b>For power users: the CLI underneath</b></summary>

```
uv run provision.py install                 # install/update ComfyUI + node packs
uv run provision.py sync                    # download everything for your profile
uv run provision.py dry-run                 # what would download + total GB
uv run provision.py verify                  # deep hash check, list missing/corrupt
uv run provision.py workflows               # regenerate + install workflow JSONs
uv run provision.py add-lora <civitai-url>  # append a LoRA to manifest + download
uv run provision.py wizard                  # the browser wizard (default command)

# options: --profile local-12gb|local-24gb|cloud|cloud-80gb
#          --comfy-dir PATH   --required-only
```

`manifest.yaml` is the single source of truth for every model file — edit it to
add/remove models; hashes and version IDs were resolved from the live APIs.
The wizard is just a friendly shell over these commands.
</details>
