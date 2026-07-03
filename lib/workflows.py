"""Generate native-format ComfyUI workflow JSONs (nodes + links, UI-loadable).

Graphs follow the official ComfyUI Wan 2.2 I2V template structure, with model
filenames taken from the manifest. Every workflow starts from a Load Image node —
the reference image is the primary input (see the plan: image-driven only).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import REPO_ROOT
from .manifest import Manifest

OUTPUT_DIR = REPO_ROOT / "workflows"

# Node modes: 0 = active, 4 = bypassed (user re-enables in the UI)
ACTIVE, BYPASS = 0, 4

# Connectable inputs / outputs per node type: (name, TYPE) in definition order.
NODE_DEFS = {
    "LoadImage":              ([], [("IMAGE", "IMAGE"), ("MASK", "MASK")]),
    "UNETLoader":             ([], [("MODEL", "MODEL")]),
    "UnetLoaderGGUF":         ([], [("MODEL", "MODEL")]),
    "CLIPLoader":             ([], [("CLIP", "CLIP")]),
    "VAELoader":              ([], [("VAE", "VAE")]),
    "CheckpointLoaderSimple": ([], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")]),
    "CLIPTextEncode":         ([("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")]),
    "LoraLoaderModelOnly":    ([("model", "MODEL")], [("MODEL", "MODEL")]),
    "LoraLoader":             ([("model", "MODEL"), ("clip", "CLIP")],
                               [("MODEL", "MODEL"), ("CLIP", "CLIP")]),
    "ModelSamplingSD3":       ([("model", "MODEL")], [("MODEL", "MODEL")]),
    "WanImageToVideo":        ([("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                ("vae", "VAE"), ("start_image", "IMAGE")],
                               [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                ("latent", "LATENT")]),
    "WanFirstLastFrameToVideo": ([("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                  ("vae", "VAE"), ("start_image", "IMAGE"),
                                  ("end_image", "IMAGE")],
                                 [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                  ("latent", "LATENT")]),
    "KSamplerAdvanced":       ([("model", "MODEL"), ("positive", "CONDITIONING"),
                                ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                               [("LATENT", "LATENT")]),
    "KSampler":               ([("model", "MODEL"), ("positive", "CONDITIONING"),
                                ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                               [("LATENT", "LATENT")]),
    "VAEDecode":              ([("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")]),
    "VAEEncode":              ([("pixels", "IMAGE"), ("vae", "VAE")], [("LATENT", "LATENT")]),
    "LatentUpscaleBy":        ([("samples", "LATENT")], [("LATENT", "LATENT")]),
    "RIFE VFI":               ([("frames", "IMAGE")], [("IMAGE", "IMAGE")]),
    "VHS_VideoCombine":       ([("images", "IMAGE"), ("audio", "AUDIO")],
                               [("Filenames", "VHS_FILENAMES")]),
    "HunyuanModelLoader":     ([], [("HUNYUAN_MODEL", "HUNYUAN_MODEL")]),
    "HunyuanDependenciesLoader": ([], [("HUNYUAN_DEPS", "HUNYUAN_DEPS")]),
    "HunyuanFoleySampler":    ([("hunyuan_model", "HUNYUAN_MODEL"),
                                ("hunyuan_deps", "HUNYUAN_DEPS"), ("image", "IMAGE")],
                               [("audio_first", "AUDIO"), ("audio_batch", "AUDIO")]),
    "IPAdapterUnifiedLoader": ([("model", "MODEL"), ("ipadapter", "IPADAPTER")],
                               [("model", "MODEL"), ("ipadapter", "IPADAPTER")]),
    "IPAdapter":              ([("model", "MODEL"), ("ipadapter", "IPADAPTER"),
                                ("image", "IMAGE"), ("attn_mask", "MASK")],
                               [("MODEL", "MODEL")]),
    "IPAdapterUnifiedLoaderFaceID": ([("model", "MODEL"), ("ipadapter", "IPADAPTER")],
                                     [("MODEL", "MODEL"), ("ipadapter", "IPADAPTER")]),
    "IPAdapterFaceID":        ([("model", "MODEL"), ("ipadapter", "IPADAPTER"),
                                ("image", "IMAGE"), ("image_negative", "IMAGE"),
                                ("attn_mask", "MASK"), ("clip_vision", "CLIP_VISION")],
                               [("MODEL", "MODEL"), ("face_image", "IMAGE")]),
    "ControlNetLoader":       ([], [("CONTROL_NET", "CONTROL_NET")]),
    "SetUnionControlNetType": ([("control_net", "CONTROL_NET")],
                               [("CONTROL_NET", "CONTROL_NET")]),
    "ControlNetApplyAdvanced": ([("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                 ("control_net", "CONTROL_NET"), ("image", "IMAGE")],
                                [("positive", "CONDITIONING"), ("negative", "CONDITIONING")]),
    "AIO_Preprocessor":       ([("image", "IMAGE")], [("IMAGE", "IMAGE")]),
    "ImageScaleToTotalPixels": ([("image", "IMAGE")], [("IMAGE", "IMAGE")]),
    "UpscaleModelLoader":     ([], [("UPSCALE_MODEL", "UPSCALE_MODEL")]),
    "ImageUpscaleWithModel":  ([("upscale_model", "UPSCALE_MODEL"), ("image", "IMAGE")],
                               [("IMAGE", "IMAGE")]),
    "SaveImage":              ([("images", "IMAGE")], []),
    "Note":                   ([], []),
}


class Graph:
    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._next_node = 1
        self._next_link = 1

    def add(self, type_: str, pos: tuple[int, int], widgets=None, title=None,
            mode=ACTIVE, size=(315, 120)) -> dict:
        ins, outs = NODE_DEFS[type_]
        node = {
            "id": self._next_node,
            "type": type_,
            "pos": [pos[0], pos[1]],
            "size": [size[0], size[1]],
            "flags": {},
            "order": self._next_node - 1,
            "mode": mode,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in ins],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": type_},
        }
        if widgets is not None:
            node["widgets_values"] = widgets
        if title:
            node["title"] = title
        if type_ == "Note":
            node["properties"] = {}
            node["color"] = "#432"
            node["bgcolor"] = "#653"
        self.nodes.append(node)
        self._next_node += 1
        return node

    def link(self, src: dict, out_name: str, dst: dict, in_name: str):
        out_idx, out_def = next((i, o) for i, o in enumerate(src["outputs"])
                                if o["name"] == out_name)
        in_idx, in_def = next((i, o) for i, o in enumerate(dst["inputs"])
                              if o["name"] == in_name)
        link_id = self._next_link
        self._next_link += 1
        self.links.append([link_id, src["id"], out_idx, dst["id"], in_idx, in_def["type"]])
        out_def["links"].append(link_id)
        in_def["link"] = link_id

    def to_json(self) -> dict:
        return {
            "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {"ds": {"scale": 0.8, "offset": [0, 0]}},
            "version": 0.4,
        }


def _files(manifest: Manifest) -> dict:
    """entry-name -> local filename, for wiring manifest values into widgets."""
    return {e.name: e.local_name for e in manifest.models + manifest.user_loras}


NEG_VIDEO = ("blurry, low quality, distorted, deformed, static, watermark, text, "
             "jpeg artifacts, ugly, extra limbs")
NEG_IMAGE = ("cgi, 3d render, cartoon, anime, illustration, painting, airbrushed "
             "plastic skin, oversaturated, deformed, bad anatomy, extra fingers, "
             "watermark, text, blurry, lowres")


# ─────────────────────────── video workflows ───────────────────────────

def wan22_i2v(manifest: Manifest, remix: bool = True, gguf: bool = False) -> dict:
    f = _files(manifest)
    if gguf:  # 12GB profile ships the GGUF pair, which needs its own loader node
        high = f.get("wan22-remix-i2v-high-gguf-q6k", "wan22RemixI2VGGUFV30_highQ6K.gguf")
        low = f.get("wan22-remix-i2v-low-gguf-q6k", "wan22RemixI2VGGUFV30_lowQ6K.gguf")
    elif remix:
        high = f.get("wan22-remix-i2v-high-fp8", "wan22RemixT2VI2V_i2vHighV30.safetensors")
        low = f.get("wan22-remix-i2v-low-fp8", "wan22RemixT2VI2V_i2vLowV30.safetensors")
    else:
        high = f.get("wan22-i2v-high-fp8-stock", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
        low = f.get("wan22-i2v-low-fp8-stock", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
    loader = "UnetLoaderGGUF" if gguf else "UNETLoader"
    lw = (lambda n: [n]) if gguf else (lambda n: [n, "default"])

    g = Graph()
    note = g.add("Note", (-80, -320), size=(460, 260), title="README — Wan 2.2 I2V")
    note["widgets_values"] = [
        "WAN 2.2 IMAGE-TO-VIDEO" + (" (Remix v3)" if remix else " (stock)") + "\n\n"
        "1. Load your START IMAGE in the Load Image node — the video begins from it.\n"
        "2. Write what should HAPPEN in the positive prompt (motion, camera, mood).\n"
        "3. Defaults: 1280x720, 81 frames @16fps, interpolated to 32fps (~5s).\n\n"
        "SPEED: the two purple LoRA nodes are the Lightning 4-step LoRAs, OFF by "
        "default. Right-click each > Bypass to toggle ON, then set BOTH samplers: "
        "steps 8, cfg 1.0 (high: start 0 end 4 / low: start 4 end 8). Quality drops a "
        "little, render time drops ~5x.\n\n"
        "If you run out of VRAM: lower resolution to 960x544, or use the GGUF "
        "loader variant installed by the 12GB profile.\n\n"
        "AUDIO: the Foley nodes WATCH the finished frames and generate matching "
        "sound (48kHz) muxed into the video automatically. The audio prompt is "
        "optional — empty means purely scene-driven; type e.g. 'heavy rain on "
        "metal roof' to steer it. First audio run downloads two small helper "
        "models automatically. To render silent video, bypass the three Foley "
        "nodes."]

    image = g.add("LoadImage", (-80, 40), widgets=["start_frame.png", "image"],
                  title="Start image (the video begins here)", size=(340, 320))
    clip = g.add("CLIPLoader", (-80, 420),
                 widgets=[f.get("umt5-xxl-fp8-encoder", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                          "wan", "default"])
    pos = g.add("CLIPTextEncode", (320, 380), widgets=[
        "The subject begins to move naturally, cinematic camera slowly pushes in, "
        "soft natural lighting"], title="Positive prompt (what happens)", size=(400, 160))
    neg = g.add("CLIPTextEncode", (320, 580), widgets=[NEG_VIDEO],
                title="Negative prompt", size=(400, 120))
    vae = g.add("VAELoader", (-80, 760), widgets=[f.get("wan21-vae", "wan_2.1_vae.safetensors")])

    unet_hi = g.add(loader, (320, 40), widgets=lw(high),
                    title="High-noise model (stage 1)")
    unet_lo = g.add(loader, (320, 200), widgets=lw(low),
                    title="Low-noise model (stage 2)")
    lora_hi = g.add("LoraLoaderModelOnly", (700, 40), mode=BYPASS,
                    widgets=[f.get("wan22-lightx2v-i2v-lora-high",
                                   "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"), 1.0],
                    title="Lightning LoRA HIGH (bypassed - enable for 4-step)")
    lora_lo = g.add("LoraLoaderModelOnly", (700, 200), mode=BYPASS,
                    widgets=[f.get("wan22-lightx2v-i2v-lora-low",
                                   "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"), 1.0],
                    title="Lightning LoRA LOW (bypassed - enable for 4-step)")
    shift_hi = g.add("ModelSamplingSD3", (1080, 40), widgets=[8.0])
    shift_lo = g.add("ModelSamplingSD3", (1080, 200), widgets=[8.0])

    i2v = g.add("WanImageToVideo", (780, 420), widgets=[1280, 720, 81, 1],
                title="Video size / length", size=(320, 220))

    ks1 = g.add("KSamplerAdvanced", (1180, 420),
                widgets=["enable", 1234567890, "randomize", 20, 3.5, "euler", "simple",
                         0, 10, "enable"],
                title="Sampler stage 1 - high noise (steps 0-10)", size=(320, 340))
    ks2 = g.add("KSamplerAdvanced", (1560, 420),
                widgets=["disable", 0, "fixed", 20, 3.5, "euler", "simple",
                         10, 10000, "disable"],
                title="Sampler stage 2 - low noise (steps 10-20)", size=(320, 340))

    dec = g.add("VAEDecode", (1940, 420))
    rife = g.add("RIFE VFI", (1940, 560), widgets=["rife47.pth", 10, 2, True, True, 1.0],
                 title="RIFE 16->32 fps", size=(320, 200))

    foley_model = f.get("hunyuan-foley-fp8" if gguf else "hunyuan-foley-fp16",
                        "hunyuanvideo_foley.safetensors")
    fl = g.add("HunyuanModelLoader", (1940, 820),
               widgets=[foley_model, "bf16", "auto"], title="Foley model (audio)")
    fd = g.add("HunyuanDependenciesLoader", (1940, 960),
               widgets=[f.get("hunyuan-foley-vae", "vae_128d_48k_fp16.safetensors"),
                        f.get("hunyuan-foley-synchformer",
                              "synchformer_state_dict_fp16.safetensors")],
               title="Foley helpers")
    fs = g.add("HunyuanFoleySampler", (2320, 820), size=(340, 320),
               widgets=[16.0, 5.1, "", "noisy, harsh, distorted, music",
                        4.5, 50, "euler", 1],
               title="Audio generator (watches the video; prompt optional)")

    vid = g.add("VHS_VideoCombine", (2720, 420), size=(360, 340), title="Save video")
    vid["widgets_values"] = {
        "frame_rate": 32, "loop_count": 0,
        "filename_prefix": "wan22_i2v_remix" if remix else "wan22_i2v",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
        "save_metadata": True, "trim_to_audio": False, "pingpong": False,
        "save_output": True}

    g.link(clip, "CLIP", pos, "clip")
    g.link(clip, "CLIP", neg, "clip")
    g.link(pos, "CONDITIONING", i2v, "positive")
    g.link(neg, "CONDITIONING", i2v, "negative")
    g.link(vae, "VAE", i2v, "vae")
    g.link(image, "IMAGE", i2v, "start_image")
    g.link(unet_hi, "MODEL", lora_hi, "model")
    g.link(unet_lo, "MODEL", lora_lo, "model")
    g.link(lora_hi, "MODEL", shift_hi, "model")
    g.link(lora_lo, "MODEL", shift_lo, "model")
    g.link(shift_hi, "MODEL", ks1, "model")
    g.link(shift_lo, "MODEL", ks2, "model")
    g.link(i2v, "positive", ks1, "positive")
    g.link(i2v, "negative", ks1, "negative")
    g.link(i2v, "positive", ks2, "positive")
    g.link(i2v, "negative", ks2, "negative")
    g.link(i2v, "latent", ks1, "latent_image")
    g.link(ks1, "LATENT", ks2, "latent_image")
    g.link(ks2, "LATENT", dec, "samples")
    g.link(vae, "VAE", dec, "vae")
    g.link(dec, "IMAGE", rife, "frames")
    g.link(rife, "IMAGE", vid, "images")
    g.link(fl, "HUNYUAN_MODEL", fs, "hunyuan_model")
    g.link(fd, "HUNYUAN_DEPS", fs, "hunyuan_deps")
    g.link(dec, "IMAGE", fs, "image")   # pre-interpolation frames @16fps
    g.link(fs, "audio_first", vid, "audio")
    return g.to_json()


def wan22_i2v_firstlast(manifest: Manifest, gguf: bool = False) -> dict:
    f = _files(manifest)
    if gguf:  # 12GB profile has no stock pair on disk; use the Remix GGUF files
        high = f.get("wan22-remix-i2v-high-gguf-q6k", "wan22RemixI2VGGUFV30_highQ6K.gguf")
        low = f.get("wan22-remix-i2v-low-gguf-q6k", "wan22RemixI2VGGUFV30_lowQ6K.gguf")
    else:
        high = f.get("wan22-i2v-high-fp8-stock", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
        low = f.get("wan22-i2v-low-fp8-stock", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
    loader = "UnetLoaderGGUF" if gguf else "UNETLoader"
    lw = (lambda n: [n]) if gguf else (lambda n: [n, "default"])

    g = Graph()
    note = g.add("Note", (-80, -300), size=(460, 240), title="README — First/Last frame")
    note["widgets_values"] = [
        "WAN 2.2 FIRST-FRAME -> LAST-FRAME\n\n"
        "Load TWO images: where the shot starts and where it must end. The model "
        "animates the transition between them — great for controlled shot design.\n\n"
        "Uses the stock Wan 2.2 I2V pair by default (it follows endpoints more "
        "faithfully). To use Remix instead, pick the Remix files in the two UNET "
        "loaders.\n\nSame two-stage sampler as the main I2V workflow; the Lightning "
        "LoRA speed trick from that workflow applies here too."]

    img_a = g.add("LoadImage", (-80, 0), widgets=["first_frame.png", "image"],
                  title="FIRST frame", size=(300, 300))
    img_b = g.add("LoadImage", (-80, 340), widgets=["last_frame.png", "image"],
                  title="LAST frame", size=(300, 300))
    clip = g.add("CLIPLoader", (-80, 700),
                 widgets=[f.get("umt5-xxl-fp8-encoder", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                          "wan", "default"])
    pos = g.add("CLIPTextEncode", (300, 640), widgets=[
        "smooth continuous motion between the two frames, cinematic lighting"],
        title="Positive prompt (how to get from A to B)", size=(380, 150))
    neg = g.add("CLIPTextEncode", (300, 840), widgets=[NEG_VIDEO],
                title="Negative prompt", size=(380, 110))
    vae = g.add("VAELoader", (-80, 1000), widgets=[f.get("wan21-vae", "wan_2.1_vae.safetensors")])

    unet_hi = g.add(loader, (300, 0), widgets=lw(high),
                    title="High-noise model (stage 1)")
    unet_lo = g.add(loader, (300, 160), widgets=lw(low),
                    title="Low-noise model (stage 2)")
    shift_hi = g.add("ModelSamplingSD3", (700, 0), widgets=[8.0])
    shift_lo = g.add("ModelSamplingSD3", (700, 160), widgets=[8.0])

    flf = g.add("WanFirstLastFrameToVideo", (740, 640), widgets=[1280, 720, 81, 1],
                title="Video size / length", size=(320, 240))
    ks1 = g.add("KSamplerAdvanced", (1140, 400),
                widgets=["enable", 1234567890, "randomize", 20, 3.5, "euler", "simple",
                         0, 10, "enable"],
                title="Sampler stage 1 - high noise", size=(320, 340))
    ks2 = g.add("KSamplerAdvanced", (1520, 400),
                widgets=["disable", 0, "fixed", 20, 3.5, "euler", "simple",
                         10, 10000, "disable"],
                title="Sampler stage 2 - low noise", size=(320, 340))
    dec = g.add("VAEDecode", (1900, 400))
    rife = g.add("RIFE VFI", (1900, 540), widgets=["rife47.pth", 10, 2, True, True, 1.0],
                 title="RIFE 16->32 fps", size=(320, 200))

    foley_model = f.get("hunyuan-foley-fp8" if gguf else "hunyuan-foley-fp16",
                        "hunyuanvideo_foley.safetensors")
    fl = g.add("HunyuanModelLoader", (1900, 800),
               widgets=[foley_model, "bf16", "auto"], title="Foley model (audio)")
    fd = g.add("HunyuanDependenciesLoader", (1900, 940),
               widgets=[f.get("hunyuan-foley-vae", "vae_128d_48k_fp16.safetensors"),
                        f.get("hunyuan-foley-synchformer",
                              "synchformer_state_dict_fp16.safetensors")],
               title="Foley helpers")
    fs = g.add("HunyuanFoleySampler", (2280, 800), size=(340, 320),
               widgets=[16.0, 5.1, "", "noisy, harsh, distorted, music",
                        4.5, 50, "euler", 1],
               title="Audio generator (watches the video; prompt optional)")

    vid = g.add("VHS_VideoCombine", (2680, 400), size=(360, 340), title="Save video")
    vid["widgets_values"] = {
        "frame_rate": 32, "loop_count": 0, "filename_prefix": "wan22_firstlast",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
        "save_metadata": True, "trim_to_audio": False, "pingpong": False,
        "save_output": True}

    g.link(clip, "CLIP", pos, "clip")
    g.link(clip, "CLIP", neg, "clip")
    g.link(pos, "CONDITIONING", flf, "positive")
    g.link(neg, "CONDITIONING", flf, "negative")
    g.link(vae, "VAE", flf, "vae")
    g.link(img_a, "IMAGE", flf, "start_image")
    g.link(img_b, "IMAGE", flf, "end_image")
    g.link(unet_hi, "MODEL", shift_hi, "model")
    g.link(unet_lo, "MODEL", shift_lo, "model")
    g.link(shift_hi, "MODEL", ks1, "model")
    g.link(shift_lo, "MODEL", ks2, "model")
    g.link(flf, "positive", ks1, "positive")
    g.link(flf, "negative", ks1, "negative")
    g.link(flf, "positive", ks2, "positive")
    g.link(flf, "negative", ks2, "negative")
    g.link(flf, "latent", ks1, "latent_image")
    g.link(ks1, "LATENT", ks2, "latent_image")
    g.link(ks2, "LATENT", dec, "samples")
    g.link(vae, "VAE", dec, "vae")
    g.link(dec, "IMAGE", rife, "frames")
    g.link(rife, "IMAGE", vid, "images")
    g.link(fl, "HUNYUAN_MODEL", fs, "hunyuan_model")
    g.link(fd, "HUNYUAN_DEPS", fs, "hunyuan_deps")
    g.link(dec, "IMAGE", fs, "image")
    g.link(fs, "audio_first", vid, "audio")
    return g.to_json()


# ─────────────────────────── image workflows ───────────────────────────

def sdxl_img2img_reference(manifest: Manifest) -> dict:
    f = _files(manifest)
    g = Graph()
    note = g.add("Note", (-80, -380), size=(500, 320), title="README — the denoise dial")
    note["widgets_values"] = [
        "SDXL REFERENCE IMG2IMG — the workhorse image workflow.\n\n"
        "Load a reference image; the IP-Adapter carries its subject/style into the "
        "result, and the KSampler's DENOISE dial decides how far to stray from it:\n"
        "   0.30  subtle variation (same scene, cleaned up)\n"
        "   0.55  balanced re-imagining  (DEFAULT)\n"
        "   0.75  heavy re-imagining (keeps subject, new scene)\n\n"
        "CHECKPOINT: LUSTIFY by default — switch it in the loader if you add other "
        "checkpoints. LORA: the LoRA node is bypassed; pick your file and un-bypass "
        "to use (default weight 0.7).\n\n"
        "CONTROLNET (bypassed by default): un-bypass the 3 green nodes to force the "
        "pose/depth/edges of the reference. Pick the mode in AIO preprocessor + "
        "SetUnionControlNetType (openpose / depth / canny).\n\n"
        "IP-ADAPTER weight 0.5: raise toward 0.7 to cling to the reference, lower "
        "or bypass to let the prompt lead.\n\n"
        "LUSTIFY rules (from the model author): keep CFG between 2.5 and 4.5 — "
        "higher fries the image. Camera tags boost realism: 'shot on Canon EOS 5D', "
        "'shot on Kodak Funsaver', 'shot on Polaroid SX-70'. Keep prompts short and "
        "concrete. Faces in wide/distant shots warp on ANY SDXL model — get closer "
        "or inpaint the face after."]

    ref = g.add("LoadImage", (-80, 0), widgets=["reference.png", "image"],
                title="Reference image", size=(340, 320))
    ckpt = g.add("CheckpointLoaderSimple", (-80, 380),
                 widgets=[f.get("lustify-olt", "lustifySDXLNSFW_oltFIXEDTEXTURES.safetensors")],
                 title="Checkpoint (LUSTIFY default)")
    lora = g.add("LoraLoader", (300, 380), mode=BYPASS,
                 widgets=[f.get("ipadapter-faceid-plusv2-sdxl-lora",
                                "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"), 0.7, 0.7],
                 title="LoRA stack (bypassed - pick a file + enable)")
    ipl = g.add("IPAdapterUnifiedLoader", (680, 380), widgets=["PLUS (high strength)"],
                title="IP-Adapter loader")
    ipa = g.add("IPAdapter", (1060, 380), widgets=[0.5, 0.0, 1.0, "prompt is more important"],
                title="IP-Adapter (reference transfer, weight 0.4-0.6)")

    pos = g.add("CLIPTextEncode", (300, 620), widgets=[
        "candid amateur photo of the subject, shot on Canon EOS 5D, natural window "
        "light, detailed skin texture with pores, film grain, shallow depth of "
        "field, realistic color grading"],
        title="Positive prompt (what to change / keep)", size=(400, 150))
    neg = g.add("CLIPTextEncode", (300, 820), widgets=[NEG_IMAGE],
                title="Negative prompt", size=(400, 110))

    # ControlNet branch — bypassed by default
    prep = g.add("AIO_Preprocessor", (740, 620), mode=BYPASS,
                 widgets=["OpenposePreprocessor", 1024],
                 title="ControlNet preprocessor (bypassed)")
    cnl = g.add("ControlNetLoader", (740, 800), mode=BYPASS,
                widgets=[f.get("controlnet-union-sdxl", "controlnet-union-sdxl-promax.safetensors")],
                title="ControlNet Union (bypassed)")
    cnt = g.add("SetUnionControlNetType", (740, 920), mode=BYPASS, widgets=["openpose"],
                title="Union mode (bypassed)")
    cna = g.add("ControlNetApplyAdvanced", (1120, 700), mode=BYPASS,
                widgets=[0.7, 0.0, 0.8], title="Apply ControlNet (bypassed)")

    scale = g.add("ImageScaleToTotalPixels", (300, 60), widgets=["lanczos", 1.0],
                  title="Fit reference to SDXL size")
    enc = g.add("VAEEncode", (680, 60), title="Reference -> latent")

    ks = g.add("KSampler", (1500, 380),
               widgets=[1234567890, "randomize", 30, 3.5, "dpmpp_2m_sde", "karras", 0.55],
               title="Main sampler — DENOISE IS THE DIAL", size=(320, 280))
    up_lat = g.add("LatentUpscaleBy", (1500, 720), widgets=["bislerp", 1.5],
                   title="Hires 1.5x")
    ks2 = g.add("KSampler", (1860, 380),
                widgets=[1234567890, "fixed", 24, 3.5, "dpmpp_2m_sde", "karras", 0.4],
                title="Hires pass (denoise ~0.4)", size=(320, 280))
    dec = g.add("VAEDecode", (2220, 380))
    upm = g.add("UpscaleModelLoader", (2220, 520),
                widgets=[f.get("4x-ultrasharp", "4xUltrasharp_4xUltrasharpV10.pt")])
    up = g.add("ImageUpscaleWithModel", (2220, 640), title="4x upscale")
    save = g.add("SaveImage", (2580, 380), widgets=["sdxl_reference"], size=(360, 400))

    g.link(ckpt, "MODEL", lora, "model")
    g.link(ckpt, "CLIP", lora, "clip")
    g.link(lora, "MODEL", ipl, "model")
    g.link(ipl, "model", ipa, "model")
    g.link(ipl, "ipadapter", ipa, "ipadapter")
    g.link(ref, "IMAGE", ipa, "image")
    g.link(lora, "CLIP", pos, "clip")
    g.link(lora, "CLIP", neg, "clip")
    g.link(ref, "IMAGE", prep, "image")
    g.link(cnl, "CONTROL_NET", cnt, "control_net")
    g.link(pos, "CONDITIONING", cna, "positive")
    g.link(neg, "CONDITIONING", cna, "negative")
    g.link(cnt, "CONTROL_NET", cna, "control_net")
    g.link(prep, "IMAGE", cna, "image")
    g.link(ref, "IMAGE", scale, "image")
    g.link(scale, "IMAGE", enc, "pixels")
    g.link(ckpt, "VAE", enc, "vae")
    g.link(ipa, "MODEL", ks, "model")
    g.link(cna, "positive", ks, "positive")
    g.link(cna, "negative", ks, "negative")
    g.link(enc, "LATENT", ks, "latent_image")
    g.link(ks, "LATENT", up_lat, "samples")
    g.link(up_lat, "LATENT", ks2, "latent_image")
    g.link(ipa, "MODEL", ks2, "model")
    g.link(cna, "positive", ks2, "positive")
    g.link(cna, "negative", ks2, "negative")
    g.link(ks2, "LATENT", dec, "samples")
    g.link(ckpt, "VAE", dec, "vae")
    g.link(dec, "IMAGE", up, "image")
    g.link(upm, "UPSCALE_MODEL", up, "upscale_model")
    g.link(up, "IMAGE", save, "images")
    return g.to_json()


def sdxl_faceid_character(manifest: Manifest) -> dict:
    f = _files(manifest)
    g = Graph()
    note = g.add("Note", (-80, -400), size=(520, 340), title="README — character consistency")
    note["widgets_values"] = [
        "SDXL FACE-ID CHARACTER WORKFLOW (film pre-viz / consistent characters)\n\n"
        "Keeps the SAME fictional face across many shots:\n"
        "  1. FACE reference -> FaceID conditioning (locks facial identity)\n"
        "  2. BODY/STYLE reference -> standard IP-Adapter (wardrobe, build, style)\n"
        "  3. Prompt + denoise control the new shot around that character.\n\n"
        "IMPORTANT — real people: this workflow deliberately preserves a face. Only "
        "use photos of real people with their explicit consent and proper licensing; "
        "never in explicit contexts without it (illegal in most places). For film "
        "pre-viz: generate a fictional face with the img2img workflow first, then "
        "use THAT image as the FaceID reference here.\n\n"
        "First run downloads InsightFace face-analysis weights automatically.\n"
        "weight_faceidv2 1.0-1.5 = how hard the face is locked."]

    face = g.add("LoadImage", (-80, 0), widgets=["face_reference.png", "image"],
                 title="FACE reference (fictional character)", size=(300, 300))
    body = g.add("LoadImage", (-80, 340), widgets=["body_reference.png", "image"],
                 title="BODY / STYLE reference", size=(300, 300))
    ckpt = g.add("CheckpointLoaderSimple", (-80, 700),
                 widgets=[f.get("lustify-olt", "lustifySDXLNSFW_oltFIXEDTEXTURES.safetensors")],
                 title="Checkpoint")

    fidl = g.add("IPAdapterUnifiedLoaderFaceID", (300, 700),
                 widgets=["FACEID PLUS V2", 0.6, "CPU"], title="FaceID loader")
    fid = g.add("IPAdapterFaceID", (680, 700),
                widgets=[0.8, 1.0, "linear", "concat", 0.0, 1.0, "V only"],
                title="FaceID (locks the face)", size=(320, 260))
    ipl = g.add("IPAdapterUnifiedLoader", (1060, 700), widgets=["PLUS (high strength)"],
                title="IP-Adapter loader (body/style)")
    ipa = g.add("IPAdapter", (1440, 700), widgets=[0.5, 0.0, 1.0, "standard"],
                title="IP-Adapter (body/style transfer)")

    pos = g.add("CLIPTextEncode", (300, 1020), widgets=[
        "candid medium shot photo of the character on a rainy street at night, "
        "shot on Canon EOS 5D, cinematic lighting, natural skin texture, film "
        "grain, shallow depth of field"],
        title="Positive prompt (the new shot)", size=(420, 150))
    neg = g.add("CLIPTextEncode", (300, 1220), widgets=[NEG_IMAGE],
                title="Negative prompt", size=(420, 110))

    scale = g.add("ImageScaleToTotalPixels", (300, 380), widgets=["lanczos", 1.0],
                  title="Fit body ref to SDXL size")
    enc = g.add("VAEEncode", (680, 380), title="Body ref -> latent")

    ks = g.add("KSampler", (1820, 700),
               widgets=[1234567890, "randomize", 30, 3.5, "dpmpp_2m_sde", "karras", 0.7],
               title="Sampler (denoise 0.7 = new shot, same character)", size=(320, 280))
    dec = g.add("VAEDecode", (2180, 700))
    upm = g.add("UpscaleModelLoader", (2180, 840),
                widgets=[f.get("4x-ultrasharp", "4xUltrasharp_4xUltrasharpV10.pt")])
    up = g.add("ImageUpscaleWithModel", (2180, 960), title="4x upscale")
    save = g.add("SaveImage", (2540, 700), widgets=["faceid_character"], size=(360, 400))

    g.link(ckpt, "MODEL", fidl, "model")
    g.link(fidl, "MODEL", fid, "model")
    g.link(fidl, "ipadapter", fid, "ipadapter")
    g.link(face, "IMAGE", fid, "image")
    g.link(fid, "MODEL", ipl, "model")
    g.link(ipl, "model", ipa, "model")
    g.link(ipl, "ipadapter", ipa, "ipadapter")
    g.link(body, "IMAGE", ipa, "image")
    g.link(ckpt, "CLIP", pos, "clip")
    g.link(ckpt, "CLIP", neg, "clip")
    g.link(body, "IMAGE", scale, "image")
    g.link(scale, "IMAGE", enc, "pixels")
    g.link(ckpt, "VAE", enc, "vae")
    g.link(ipa, "MODEL", ks, "model")
    g.link(pos, "CONDITIONING", ks, "positive")
    g.link(neg, "CONDITIONING", ks, "negative")
    g.link(enc, "LATENT", ks, "latent_image")
    g.link(ks, "LATENT", dec, "samples")
    g.link(ckpt, "VAE", dec, "vae")
    g.link(dec, "IMAGE", up, "image")
    g.link(upm, "UPSCALE_MODEL", up, "upscale_model")
    g.link(up, "IMAGE", save, "images")
    return g.to_json()


def chroma_img2img(manifest: Manifest) -> dict:
    f = _files(manifest)
    g = Graph()
    note = g.add("Note", (-80, -320), size=(480, 260), title="README — Chroma img2img")
    note["widgets_values"] = [
        "CHROMA IMG2IMG — maximum-realism reworking of a reference image.\n\n"
        "Chroma (Flux architecture, fully uncensored) has the best raw realism and "
        "prompt adherence of the image stack, but its reference-conditioning "
        "ecosystem is thin: no mature Chroma IP-Adapter/Redux existed when this "
        "repo was built (checked 2026-07), so this is classic denoise-based "
        "img2img.\n\nDenoise: 0.3 subtle - 0.55 default - 0.75 heavy.\n"
        "Steps 26-30, cfg ~4.5. Needs ~18GB VRAM — 24GB profile and up.\n\n"
        "The Flux VAE (ae.safetensors) comes from a gated HF repo — the setup "
        "wizard handles the token."]

    ref = g.add("LoadImage", (-80, 0), widgets=["reference.png", "image"],
                title="Reference image", size=(340, 320))
    unet = g.add("UNETLoader", (320, 0),
                 widgets=[f.get("chroma1-hd", "Chroma1-HD.safetensors"), "default"],
                 title="Chroma model")
    clip = g.add("CLIPLoader", (320, 140),
                 widgets=[f.get("t5xxl-fp8-encoder", "t5xxl_fp8_e4m3fn_scaled.safetensors"),
                          "chroma", "default"], title="T5 text encoder")
    vae = g.add("VAELoader", (320, 280), widgets=[f.get("flux-vae", "ae.safetensors")],
                title="Flux VAE")
    pos = g.add("CLIPTextEncode", (700, 60), widgets=[
        "raw photograph, natural light, detailed skin texture, realistic color grading"],
        title="Positive prompt", size=(400, 150))
    neg = g.add("CLIPTextEncode", (700, 260), widgets=["low quality, blurry, watermark, text"],
                title="Negative prompt", size=(400, 110))
    scale = g.add("ImageScaleToTotalPixels", (700, 430), widgets=["lanczos", 1.05],
                  title="Fit to ~1MP")
    enc = g.add("VAEEncode", (1080, 430), title="Reference -> latent")
    ks = g.add("KSampler", (1160, 60),
               widgets=[1234567890, "randomize", 28, 4.5, "euler", "beta", 0.55],
               title="Sampler (denoise = how much changes)", size=(320, 280))
    dec = g.add("VAEDecode", (1540, 60))
    save = g.add("SaveImage", (1900, 60), widgets=["chroma_img2img"], size=(360, 400))

    g.link(clip, "CLIP", pos, "clip")
    g.link(clip, "CLIP", neg, "clip")
    g.link(ref, "IMAGE", scale, "image")
    g.link(scale, "IMAGE", enc, "pixels")
    g.link(vae, "VAE", enc, "vae")
    g.link(unet, "MODEL", ks, "model")
    g.link(pos, "CONDITIONING", ks, "positive")
    g.link(neg, "CONDITIONING", ks, "negative")
    g.link(enc, "LATENT", ks, "latent_image")
    g.link(ks, "LATENT", dec, "samples")
    g.link(vae, "VAE", dec, "vae")
    g.link(dec, "IMAGE", save, "images")
    return g.to_json()


ALL_WORKFLOWS = {
    "wan22_i2v_remix.json": lambda m, gguf=False: wan22_i2v(m, remix=True, gguf=gguf),
    "wan22_i2v_firstlast.json": lambda m, gguf=False: wan22_i2v_firstlast(m, gguf=gguf),
    "sdxl_img2img_reference.json": lambda m, gguf=False: sdxl_img2img_reference(m),
    "sdxl_faceid_character.json": lambda m, gguf=False: sdxl_faceid_character(m),
    "chroma_img2img.json": lambda m, gguf=False: chroma_img2img(m),
}


def generate_all(manifest: Manifest, out_dir: Path = OUTPUT_DIR,
                 profile: str | None = None) -> list[Path]:
    gguf = profile == "local-12gb"  # that profile downloads GGUF video models
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, fn in ALL_WORKFLOWS.items():
        data = fn(manifest, gguf=gguf)
        path = out_dir / filename
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        written.append(path)
    return written
