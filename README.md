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
   pick **300 GB** (the models live here permanently; pods come and go).
   Why 300: the full cloud model set is **~224 GB** (of which MiniMax H3 is
   ~63 GB), plus ComfyUI and its Python packages, plus room for your renders and
   H3's disk offloading. If you skip H3, 200 GB is enough.
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
| RTX 4090 / 5090 | 24–32 GB | Wan 2.2 video, all image work, LoRA training. **H3 only with heavy offloading — slow** | $ |
| L40S / A6000 | 48 GB | **the H3 minimum I'd recommend** — H3 fits without thrashing | $$ |
| A100 / H100 | 80 GB | comfortable 2K H3 work and the fp16 video tier | $$$ |

- **Not using MiniMax H3?** 24–48 GB is the sweet spot; nothing else here needs more.
- **Using H3?** Rent **48 GB or more**. H3's smallest footprint is **42.5 GB**
  (transformer + text encoder + both VAEs), so a 24 GB card has to shuffle
  weights between GPU and system RAM constantly. It runs — ComfyUI's dynamic
  offloading even makes it work on small cards — but it is *much* slower.
- **Honest gap:** nobody has published reliable H3 render times yet, and I
  haven't measured them. Expect it to be **several times slower than Wan 2.2**
  and budget your pod hours accordingly rather than trusting a number.

## What's in the box

**Video (image → video, Wan 2.2):**
- `wan22_i2v_remix` — turn one still image into a ~5s, 720p, 32fps video.
  Uses the Wan 2.2 Remix creative model; optional 4-step "Lightning" turbo mode.
- `wan22_i2v_firstlast` — give a first **and** last frame; it animates between
  them. Perfect for planned shot transitions.

**Video with sound (MiniMax H3 — optional, cloud/48 GB+):**
- `minimax_h3_r2v` — **the H3 workflow**: mix up to 9 reference images, 3
  reference videos and 3 reference audio clips, tag them in the prompt, and get
  2K video with synchronized sound. See "Video with sound" below.
- `minimax_h3_i2v` — one still image → video with sound.
- `minimax_h3_t2v` — prompt only, no image (secondary; the rest of this repo is
  deliberately reference-driven).

**Images (SDXL, Qwen & Chroma):**
- `qwen_edit_character` — **the character workflow**: give it a photo + an
  instruction ("she's now dancing in a nightclub") and it changes the scene while
  keeping the person identical. No denoise juggling — identity preservation is
  what this model is trained for. NSFW unlock LoRA included (bypassed by default).
- `sdxl_txt2img_lustify` — pure **text-to-image** on LUSTIFY: describe a scene,
  get a photo, no reference image needed. Has a slot for a **character LoRA** you
  trained (see "Train your own character" below) — drop it in and put its trigger
  word in the prompt.
- `sdxl_img2img_reference` — the workhorse. Your reference image goes in; the
  **denoise dial** decides how much changes (0.3 = touch-up … 0.75 = reimagine).
  IP-Adapter carries the subject/style over; optional ControlNet locks the pose.
- `sdxl_faceid_character` — keeps the **same fictional character's face** across
  many shots (film pre-viz). Face reference + body/style reference in, new shot out.
- `chroma_img2img` — maximum-realism reworking of a reference image.

**Training:**
- `sdxl_lora_trainer_lustify` — train your **own character LoRA** from a folder of
  images, right inside ComfyUI (see "Train your own character" below).

Every workflow has a yellow **README note** inside explaining its dials in plain
English.

### What is all this? (30-second glossary)

- **Checkpoint** — the big "brain" file that actually paints images. Different
  checkpoints have different styles (RealVis = photoreal, Pony = stylized).
- **LoRA** — a small add-on file that teaches a checkpoint one specific thing
  (a style, an outfit, better hands). Stackable, adjustable strength.
- **VAE** — the translator between the model's internal math and actual pixels.
  Wrong VAE = washed-out colors. Ours are matched automatically.
- **Trigger word** — a short made-up word (like `ohwxwoman`) that you assign to a
  character LoRA when training it. Typing that word in a prompt summons the
  character; leaving it out means they won't appear.

## Video with sound (MiniMax H3)

**What it is.** MiniMax H3 (Hailuo 3.0) generates video **and its own audio at
the same time** — dialogue, sound effects and music in a single pass, not dubbed
on afterwards. Output is up to **2K, 24fps, about 5–15 seconds** per clip.

**It's optional and big.** ~63 GB of models, and it needs a 48 GB GPU to be
pleasant. The wizard asks before downloading it, and you can add it later from
the Setup menu. On a 12 GB laptop it isn't offered at all — it genuinely
can't run there.

### Pointing it at your references

This is the part that's different from every other workflow here. You connect
reference images/videos/audio, then **refer to them in the prompt by tag**:

- `<Picture 1>`, `<Picture 2>` … up to **9** reference images
- `<Video 1>` … up to **3** reference videos
- `<Audio 1>` … up to **3** reference audio clips

> *"Use `<Picture 1>` as the character and `<Audio 1>` exactly as it is. She
> walks through neon rain at night, camera pushing in slowly."*

The numbers follow **the order you connected the inputs**, not the node names —
so if you rewire something, renumber the tags in your prompt to match.

### ⚠ Always listen to the audio before you use it

The sound is generated, not recorded, and it is **not reliable**. Speech can
come out garbled, mistimed, or in a voice you didn't ask for. Treat every clip's
audio as a draft: **listen to it before using it anywhere**, and if it misses,
change the seed or describe the sound more concretely in the prompt.

### H3 or Wan 2.2 Remix?

| | **Wan 2.2 Remix** | **MiniMax H3** |
|---|---|---|
| Resolution | 720p | up to 2K |
| Sound | none | yes, generated with the picture |
| References | 1 image (or first+last) | up to 9 images + 3 videos + 3 audio |
| Speed / size | fast, ~22 GB | much slower, ~63 GB, needs 48 GB VRAM |

**Use Wan** for iterating, silent b-roll, and anything on a smaller GPU. **Use
H3** for finished shots that need sound, or that need several references (a
character *and* a style *and* a voice) held together at once.

## Train your own character (make a custom LoRA)

You can teach the AI a **fictional** character from your own images, then bring them
back in any picture by typing a trigger word. It all happens inside ComfyUI — use
Setup → **[3] Train a character LoRA** for the guided version, or open the
`sdxl_lora_trainer_lustify` workflow directly.

1. **Build a dataset.** Collect **15–50 images of one character** — face *and*
   body, different angles, expressions, and outfits. Sharp, well-lit shots.
   **Variety and quality beat quantity**: 25 good, varied images beat 60 similar
   blurry ones. Put them in a folder inside ComfyUI's `input` folder, e.g.
   `input/character_dataset`.
2. **Pick a trigger word.** A short, unique, made-up word (`ohwxwoman`,
   `zqkhero`). Avoid real words/names so it doesn't collide with things the model
   already knows.
3. **Set three things in the workflow** and press Queue: the folder path, the
   trigger word (`class_tokens`), and the output name. No captioning required — the
   trigger word is used as the caption automatically. (Optional: drop a `.txt` file
   next to each image with a description for sharper results; Florence-2 is
   installed if you want to auto-generate them.)
4. **Wait.** Roughly **2–3 hours for ~30 images on a 4090/24GB**; much slower on a
   12GB laptop (set `blocks_to_swap` to ~20, or just train on a cloud GPU). Preview
   images appear every 500 steps, and a LoRA is saved into `models/loras` after
   each of 4 segments — so you get several checkpoints to compare.
5. **Test it.** Open `sdxl_txt2img_lustify`, pick your new LoRA in the LoRA node,
   enable it (right-click → Bypass to toggle) at **strength ~0.8**, and put your
   **trigger word** in the prompt. Try the different saved checkpoints and keep the
   one that looks best — later checkpoints aren't always better (they can
   "overcook").

**Fictional characters only.** Do not train on real, identifiable people. A
character LoRA reproduces a specific face on demand; doing that with a real person's
likeness without consent is harmful and, in explicit contexts, illegal in most
jurisdictions (NCII laws). Generate a fictional face first (text-to-image), collect
shots of *that*, and train on those.

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
