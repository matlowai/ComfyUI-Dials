#!/usr/bin/env python3
"""Build the example workflows (UI + API twins) for ComfyUI-Dials (MiniMax H3 text-to-video).

The column layout / overlap check follows the MOSS preset builder in the author's ComfyUI workflows.

  python3 workflows/build_workflows.py
Writes workflows/dials_studio_h3.json, workflows/dial_maker_h3.json, workflows/dials_pika_h3_flex.json, dials_pika_h3_flex_derope.json and *_api.json
(API twins, used for testing).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
PROMPT = "A busker plays guitar on a busy city sidewalk at sunset while people walk past and a dog watches."
# Pika H3 Flex (Pika-Labs/Pika-H3-Flex; Comfy repack in Comfy-Org/MiniMax-H3 loras/): 6-step lcm, no CFG.
PIKA_LORA = "pika_h3_flex_fl2va_v1_bf16.safetensors"
PIKA_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
PIKA_PROMPT = "integrated_multimodal_description: [Shot 1] 2D anime, cel-shaded, cinematic, a ruined stone throne hall lit by green torch flames, dust hanging in the shafts of light. A cute young magical girl spellblade with big bright eyes, long silver twin tails and a confident grin, wearing ornate fantasy plate armor with a small breastplate, pauldrons, gauntlets and thigh-high boots, sprints straight at a towering skeleton king slumped on a throne of bones, its rusted iron crown tilted, a huge notched cleaver resting in its bony fist. Glowing blue runes ignite along her longsword as she runs. She launches a fast three-strike combination without slowing: a low sweeping slash that throws sparks off the throne steps, a rising diagonal cut that shears through the king's raised forearm so the bones burst apart in a spray of splinters, and a spinning overhead strike that detonates a burst of blue arcane fire on impact as the king catches it on the flat of its cleaver with a hard metallic clang. The camera tracks with her in a fast lateral arc around the throne at chest height, matching her speed, then snaps to a tight two-shot as the blades lock, blue rune light and green flame reflections sliding across both weapons and her armor. The king's jaw drops open in a silent roar and its skull tilts down toward her; she keeps grinning. Bone fragments, sparks and blue embers scatter through the frame and settle on the flagstones. Every frame is crisp and sharp as if shot at a very high shutter speed, with fine detail held on the blade runes, the armor edges, her twin tails and the flying bone splinters throughout the movement.\n\noverall_soundscape: Rapid footfalls on flagstone under a full sprint; a rising crystalline hum as the runes ignite; three sharp steel impacts in fast succession, the first with a scrape of sparks off stone, the second with dry bone cracking and splinters skittering across the floor, the third a deep ringing clang wrapped in a whooshing burst of arcane fire that decays slowly; the low roar and flutter of torch flames throughout; a rumbling exhale of dust and grit from the throne as the king leans forward.\n\nnon_diegetic_music: N/A"
GAP, TITLE = 40, 34
COLS = {"left": -900, "mid": -380, "right": 140, "far": 660, "far2": 1180}
BLUE, GREEN, PURPLE, GRAY = ("#223", "#335"), ("#232", "#353"), ("#323", "#535"), ("#222", "#333")


def est_height(widgets, inputs, outputs, multiline=0):
    sockets = max(len(inputs), len(outputs))
    return TITLE + 24 * sockets + 28 * len(widgets) + 90 * multiline + 10


class G:
    def __init__(self):
        self.nodes, self.links, self.groups, self.api = [], [], [], {}
        self.nid = self.lid = 0
        self.cursor = {}

    def node(self, type_, col, width, widgets=(), inputs=(), outputs=(), api=None, title=None, color=None, multiline=0, height=None):
        """inputs: (name, type, widget_name_or_None); outputs: (name, type). Nodes auto-stack per column."""
        self.nid += 1
        h = height or est_height([w for w in widgets], [i for i in inputs if not i[2]], outputs, multiline)
        x, y = COLS[col], self.cursor.get(col, 0)
        self.cursor[col] = y + h + GAP
        n = {"id": self.nid, "type": type_, "pos": [x, y], "size": [width, h], "flags": {}, "order": self.nid, "mode": 0,
             "inputs": [dict({"name": nm, "type": t, "link": None}, **({"widget": {"name": w}} if w else {})) for nm, t, w in inputs],
             "outputs": [{"name": nm, "type": t, "links": []} for nm, t in outputs],
             "properties": {"Node name for S&R": type_}, "widgets_values": list(widgets)}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        if api is not None:
            self.api[str(self.nid)] = {"class_type": type_, "inputs": dict(api)}
        return self.nid

    def link(self, src, sslot, dst, dname):
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        di = next(i for i, x in enumerate(d["inputs"]) if x["name"] == dname)
        self.lid += 1
        self.links.append([self.lid, src, sslot, dst, di, s["outputs"][sslot]["type"]])
        s["outputs"][sslot]["links"].append(self.lid)
        d["inputs"][di]["link"] = self.lid
        if str(dst) in self.api:
            self.api[str(dst)]["inputs"][dname] = [str(src), sslot]

    def check_overlaps(self):
        R = [(n["id"], n["pos"][0], n["pos"][1] - TITLE, n["pos"][0] + n["size"][0], n["pos"][1] + n["size"][1]) for n in self.nodes]
        return [(a[0], b[0]) for i, a in enumerate(R) for b in R[i + 1:] if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]]

    def ui(self):
        return {"last_node_id": self.nid, "last_link_id": self.lid, "nodes": self.nodes, "links": self.links, "groups": self.groups,
                "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [1000, 200]}}, "version": 0.4}


def h3_render(g, model_src, note, prompt=PROMPT, length=124, seed=1001, prompt_lines=1, pika=False, prefix="video/dials",
              prompt_col="right", note_height=330, width=832, height=480, prompt_node=None):
    """Loaders + H3 T2V sampling chain on the right; model_src = (node, slot) of the dialed MODEL.
    pika=True swaps the 20-step res_multistep schedule for Pika H3 Flex's 6-step lcm schedule (no model input).
    prompt_node: a STRING node to read the prompt from (shared with a second pass) instead of the inline widget.
    Returns the node ids a second pass needs."""
    clip = g.node("CLIPLoader", "left", 320, [TE, "minimax", "default"], outputs=[("CLIP", "CLIP")],
                  api={"clip_name": TE, "type": "minimax", "device": "default"})
    vae = g.node("VAELoader", "left", 320, [VAE], outputs=[("VAE", "VAE")], api={"vae_name": VAE})
    linked = [("prompt", "STRING", "prompt")] if prompt_node else []
    cond = g.node("MiniMaxH3ImageToVideo", prompt_col, 420, ["" if prompt_node else prompt, width, height, length],
                  inputs=[("clip", "CLIP", None), ("vae", "VAE", None)] + linked, outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")],
                  api=dict({} if prompt_node else {"prompt": prompt}, width=width, height=height, length=length),
                  multiline=0 if prompt_node else prompt_lines, title="Pass 1 (MiniMax H3)" if prompt_node else "Prompt (MiniMax H3)")
    if prompt_node:
        g.link(prompt_node, 0, cond, "prompt")
    g.link(clip, 0, cond, "clip")
    g.link(vae, 0, cond, "vae")
    noise = g.node("RandomNoise", "right", 420, [seed, "fixed"], outputs=[("NOISE", "NOISE")], api={"noise_seed": seed})
    guider = g.node("BasicGuider", "right", 420, inputs=[("model", "MODEL", None), ("conditioning", "CONDITIONING", None)],
                    outputs=[("GUIDER", "GUIDER")], api={})
    if pika:
        sampler = g.node("KSamplerSelect", "right", 420, ["lcm"], outputs=[("SAMPLER", "SAMPLER")], api={"sampler_name": "lcm"},
                         title="Sampler: lcm (Pika)", color=BLUE)
        sched = g.node("ManualSigmas", "right", 420, [PIKA_SIGMAS], outputs=[("SIGMAS", "SIGMAS")], api={"sigmas": PIKA_SIGMAS},
                       title="Pika 6-step schedule (keep as is)", color=BLUE)
    else:
        sampler = g.node("KSamplerSelect", "right", 420, ["res_multistep"], outputs=[("SAMPLER", "SAMPLER")], api={"sampler_name": "res_multistep"})
        sched = g.node("BasicScheduler", "right", 420, ["simple", 20, 1.0], inputs=[("model", "MODEL", None)], outputs=[("SIGMAS", "SIGMAS")],
                       api={"scheduler": "simple", "steps": 20, "denoise": 1.0})
    out = g.node("SamplerCustomAdvanced", "right", 420,
                 inputs=[("noise", "NOISE", None), ("guider", "GUIDER", None), ("sampler", "SAMPLER", None), ("sigmas", "SIGMAS", None), ("latent_image", "LATENT", None)],
                 outputs=[("output", "LATENT"), ("denoised_output", "LATENT")], api={})
    dec = g.node("VAEDecode", "right", 420, inputs=[("samples", "LATENT", None), ("vae", "VAE", None)], outputs=[("IMAGE", "IMAGE")], api={})
    avae = g.node("VAELoader", "left", 320, [AUDIO_VAE], outputs=[("VAE", "VAE")], api={"vae_name": AUDIO_VAE}, title="Audio VAE")
    adec = g.node("VAEDecodeAudio", "right", 420, inputs=[("samples", "LATENT", None), ("vae", "VAE", None)], outputs=[("AUDIO", "AUDIO")], api={})
    vid = g.node("CreateVideo", "right", 420, [24, 8], inputs=[("images", "IMAGE", None), ("audio", "AUDIO", None)], outputs=[("VIDEO", "VIDEO")], api={"fps": 24.0})
    save = g.node("SaveVideo", "right", 420, [prefix, "auto", "auto"], inputs=[("video", "VIDEO", None)], outputs=[],
                  api={"filename_prefix": prefix, "format": "mp4", "codec": "h264"})
    g.link(model_src[0], model_src[1], guider, "model")
    if not pika:
        g.link(model_src[0], model_src[1], sched, "model")
    g.link(cond, 0, guider, "conditioning")
    for s, slot, d, nm in ((noise, 0, out, "noise"), (guider, 0, out, "guider"), (sampler, 0, out, "sampler"), (sched, 0, out, "sigmas"),
                           (cond, 1, out, "latent_image"), (out, 0, dec, "samples"), (vae, 0, dec, "vae"), (dec, 0, vid, "images"),
                           (out, 0, adec, "samples"), (avae, 0, adec, "vae"), (adec, 0, vid, "audio"), (vid, 0, save, "video")):
        g.link(s, slot, d, nm)
    g.node("MarkdownNote", "left", 320, [note], height=note_height)
    return {"clip": clip, "vae": vae, "avae": avae, "sampler": sampler, "latent": out, "images": dec, "audio": adec}


def studio():
    g = G()
    unet = g.node("UNETLoader", "left", 320, [UNET, "default"], outputs=[("MODEL", "MODEL")], api={"unet_name": UNET, "weight_dtype": "default"})
    m_dial = g.node("DialsLoadDial", "mid", 380, ["dials/h3_motion.safetensors"], outputs=[("dial", "DIAL"), ("report", "STRING")],
                    api={"dial_name": "dials/h3_motion.safetensors"}, title="Dial: Freeze ↔ Zoomies", color=PURPLE)
    m_apply = g.node("DialsApplyDial", "mid", 380, [1.0], inputs=[("model", "MODEL", None), ("dial", "DIAL", None)],
                     outputs=[("MODEL", "MODEL")], api={"strength": 1.0}, title="Motion  (−2 frozen … +2 zoomies)", color=PURPLE)
    e_dial = g.node("DialsLoadDial", "mid", 380, ["dials/h3_evil_sinister.safetensors"], outputs=[("dial", "DIAL"), ("report", "STRING")],
                    api={"dial_name": "dials/h3_evil_sinister.safetensors"}, title="Dial: Evil (pick a flavour)", color=GREEN)
    e_apply = g.node("DialsApplyDial", "mid", 380, [0.0], inputs=[("model", "MODEL", None), ("dial", "DIAL", None)],
                     outputs=[("MODEL", "MODEL")], api={"strength": 0.0}, title="Evil  (0 off … +2 strong)", color=GREEN)
    g.link(unet, 0, m_apply, "model")
    g.link(m_dial, 0, m_apply, "dial")
    g.link(m_apply, 0, e_apply, "model")
    g.link(e_dial, 0, e_apply, "dial")
    h3_render(g, (e_apply, 0), "**Dials Studio — MiniMax H3**\n\nDials are ~14 KB steering files made from prompt pairs (no training). "
              "The **Apply Dial** strength is a slider: negative moves toward the pair's A side (still), positive toward B (moving).\n\n"
              "- **−2**: near-frozen, cinemagraph-like.\n- **+2**: everything zooms.\n\n"
              "- **Evil**: pick a flavour (sinister breaks speech audio; unhinged really is unhinged). 0 = off.\n\n"
              "Dials stack with each other and with LoRAs: add more Load Dial → Apply Dial pairs in the chain. "
              "`h3_motion_lora.safetensors` is the same dial as a LoRA file (stock *Load LoRA (model only)*, use ~1.6× strength).")
    return g


def maker():
    g = G()
    unet = g.node("UNETLoader", "left", 320, [UNET, "default"], outputs=[("MODEL", "MODEL")], api={"unet_name": UNET, "weight_dtype": "default"})
    pairs = ("{subject} stands perfectly still, posing for a photograph. ||| {subject} runs quickly across the scene, moving energetically.\n"
             "{subject} sits motionless, frozen in place. ||| {subject} jumps up and spins around, full of movement.")
    subjects = "A young woman\nAn old man\nA golden dog\nA horse"
    make = g.node("DialsMakeDial", "mid", 420, [pairs, subjects, "my_dial", "post_refiner (hook)", True, True, True, 3],
                  inputs=[("clip", "CLIP", None), ("model", "MODEL", None)], outputs=[("dial", "DIAL"), ("report", "STRING")],
                  api={"pairs": pairs, "subjects": subjects, "name": "my_dial", "inject": "post_refiner (hook)", "save": True,
                       "debias": True, "calibrate": True, "nuisance_dims": 3},
                  title="Make Dial  (A ||| B per line)", color=PURPLE, multiline=3)
    rep = g.node("PreviewAny", "mid", 420, [], inputs=[("source", "*", None)], outputs=[("STRING", "STRING")], api={}, height=160)
    apply = g.node("DialsApplyDial", "mid", 420, [2.0], inputs=[("model", "MODEL", None), ("dial", "DIAL", None)],
                   outputs=[("MODEL", "MODEL")], api={"strength": 2.0}, title="Preview strength", color=PURPLE)
    g.link(make, 1, rep, "source")
    g.link(unet, 0, make, "model")
    g.link(unet, 0, apply, "model")
    g.link(make, 0, apply, "dial")
    clip = h3_render(g, (apply, 0), "**Dial Maker — MiniMax H3**\n\n1. Write pairs, one per line: `A ||| B`. Keep A and B identical except for the "
                     "quality you want to steer.\n2. Optional: `{subject}` expands over the subjects list, so subject details cancel out.\n"
                     "3. Run: the report shows how well the pairs agree (cos > 0.6 is a clean direction) and how much generic "
                     "wording shift was removed (debias). The dial is saved to "
                     "`models/loras/dials/<name>.safetensors`.\n4. Preview with the strength slider, then use the dial anywhere.")["clip"]
    g.link(clip, 0, make, "clip")
    return g


def add_groups(g, spec):
    """spec: (title, column, color). Each group frames every node in its column."""
    for title, col, color in spec:
        ns = [n for n in g.nodes if n["pos"][0] == COLS[col]]
        x0 = min(n["pos"][0] for n in ns) - 20
        y0 = min(n["pos"][1] for n in ns) - TITLE - 50
        x1 = max(n["pos"][0] + n["size"][0] for n in ns) + 20
        y1 = max(n["pos"][1] + n["size"][1] for n in ns) + 20
        g.groups.append({"title": title, "bounding": [x0, y0, x1 - x0, y1 - y0], "color": color, "font_size": 24})


PIKA_DIAL_NOTE = ("**Motion dial on Pika** (1280x704, spellblade prompt, seed 42, one take per strength; camera-removed "
                  "optical flow):\n- **+0.5**: +20 % motion, same scene.\n- **+1.0**: +23 %, still the same scene.\n"
                  "- **+1.5 / +2.0**: about +90 %; the scene restages and characters morph.\n- **+3**: loses the prompt.\n\n"
                  "**Evil** (off at 0): dread +0.5 to +1 darkens the scene and turns the characters menacing while keeping "
                  "the prompt. Other flavours: pick another `h3_evil_*` file.\n\n"
                  "**No background music:** write `non_diegetic_music: N/A` (Pika's prompt guide).\n\n"
                  "**Models:** the LoRA is `loras/pika_h3_flex_fl2va_v1_bf16.safetensors` in "
                  "[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (same weights as "
                  "[Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) `comfy/loras/`). "
                  "Base model, text encoder and VAEs as in the other Dials workflows.")
# De-rope pass 2 re-enters at base time 0.48 (just under the usual inject 0.5): sigma = 12*0.48 / (1 + 11*0.48).
# Default: then Pika's own knots (4 steps). Experimental 6+6: Pika's knot spacing squeezed into [0, 0.48] (6 steps).
PIKA_DEROPE_SIGMAS = "0.917197, 0.8, 0.571429, 0.333333, 0.0"
PIKA_DEROPE_6PLUS6_SIGMAS = "0.917197, 0.870968, 0.791209, 0.62069, 0.376963, 0.190225, 0.0"


def pika_front(g):
    """Base model -> Pika LoRA -> motion dial -> evil dial (off). Returns the dialed MODEL node."""
    unet = g.node("UNETLoader", "left", 320, [UNET, "default"], outputs=[("MODEL", "MODEL")], api={"unet_name": UNET, "weight_dtype": "default"})
    lora = g.node("LoraLoaderModelOnly", "left", 320, [PIKA_LORA, 1.0], inputs=[("model", "MODEL", None)], outputs=[("MODEL", "MODEL")],
                  api={"lora_name": PIKA_LORA, "strength_model": 1.0}, title="Pika H3 Flex LoRA (FL2VA)", color=BLUE)
    g.link(unet, 0, lora, "model")
    m_dial = g.node("DialsLoadDial", "mid", 380, ["dials/h3_motion.safetensors"], outputs=[("dial", "DIAL"), ("report", "STRING")],
                    api={"dial_name": "dials/h3_motion.safetensors"}, title="Dial: Freeze ↔ Zoomies", color=PURPLE)
    m_apply = g.node("DialsApplyDial", "mid", 380, [0.5], inputs=[("model", "MODEL", None), ("dial", "DIAL", None)],
                     outputs=[("MODEL", "MODEL")], api={"strength": 0.5}, title="Motion  (+0.5 a bit more … +1 a lot)", color=PURPLE)
    e_dial = g.node("DialsLoadDial", "mid", 380, ["dials/h3_evil_dread.safetensors"], outputs=[("dial", "DIAL"), ("report", "STRING")],
                    api={"dial_name": "dials/h3_evil_dread.safetensors"}, title="Dial: Evil (pick a flavour)", color=GREEN)
    e_apply = g.node("DialsApplyDial", "mid", 380, [0.0], inputs=[("model", "MODEL", None), ("dial", "DIAL", None)],
                     outputs=[("MODEL", "MODEL")], api={"strength": 0.0}, title="Evil  (0 off … +1 obvious)", color=GREEN)
    g.link(lora, 0, m_apply, "model")
    g.link(m_dial, 0, m_apply, "dial")
    g.link(m_apply, 0, e_apply, "model")
    g.link(e_dial, 0, e_apply, "dial")
    return e_apply


def pika():
    g = G()
    model = pika_front(g)
    h3_render(g, (model, 0), "**Pika H3 Flex + Dials — MiniMax H3, 6 steps**\n\n"
              "Pika H3 Flex is a 6-step, no-CFG LoRA for MiniMax H3. Keep the **lcm** sampler and the **6-step schedule** "
              "as they are; the motion dial sits between the LoRA and the sampler. Size: 1280x704 (about 0.9 MP).\n\n" + PIKA_DIAL_NOTE,
              prompt=PIKA_PROMPT, length=141, seed=42, prompt_lines=6, pika=True, prefix="video/dials_pika",
              prompt_col="mid", note_height=560, width=1280, height=704)
    add_groups(g, (("Models", "left", "#3f5159"), ("Dials + prompt", "mid", "#4f3f59"), ("Pika 6-step render", "right", "#3f4f3f")))
    return g


def pika_derope(sigmas=PIKA_DEROPE_SIGMAS, steps_title="De-rope steps: inject 0.48, Pika knots", experimental=False):
    """Pika + dial, then a de-rope pass (ComfyUI-MAINodes): find the jerky stretches, hold those frames, re-render the
    slowed clip from inject 0.48 and map it back to real time. Audio: pass 1's original track, kept intact."""
    g = G()
    model = pika_front(g)
    text = g.node("PrimitiveStringMultiline", "mid", 420, [PIKA_PROMPT], outputs=[("STRING", "STRING")], api={"value": PIKA_PROMPT},
                  title="Prompt (both passes)", multiline=6)
    p1 = h3_render(g, (model, 0), "**Pika H3 Flex + Dials + de-rope — MiniMax H3**\n\n"
                   "Pass 1 is the 6-step Pika render (saved as `dials_pika_plain`). The de-rope pass finds stretches that move "
                   "too fast for the frame rate, holds those frames, re-renders the slowed clip from inject 0.48 and maps it "
                   "back to real time (`dials_pika_derope`). **Needs [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes).**\n\n"
                   "**Audio:** the de-roped clip keeps pass 1's original soundtrack (H3 Audio Recover: *keep the original "
                   "performance*).\n\n**Time** (1280x704, 141 frames, RTX PRO 6000 Max-Q): Pika alone ~1.5 min; Pika + de-rope "
                   "~5.5 min, about the same as one 25-step render of the base model without de-rope.\n\n"
                   + ("**Experimental 6+6:** pass 2 takes 6 steps whose spacing copies Pika's own schedule, squeezed below "
                      "inject 0.48. These sigmas are off Pika's trained knots.\n\n" if experimental else "") + PIKA_DIAL_NOTE,
                   length=141, seed=42, pika=True, prefix="video/dials_pika_plain", prompt_col="mid", note_height=700,
                   width=1280, height=704, prompt_node=text)
    W = 420
    oracle = g.node("H3JerkOracle", "far", W, [141, 0.75, 4, True, "balanced (default)", 8, "value |d3| (default)", 0.0, 24, 0.0, 18, 6.7, "minimax-h3", 0],
                    inputs=[("samples", "LATENT", None)],
                    outputs=[("hold_map", "STRING"), ("segments", "STRING"), ("window_start", "INT"), ("window_len", "INT"), ("profile", "STRING"), ("report", "STRING")],
                    api={"length": 141, "q": 0.75, "d_max": 4, "ramp": True, "bridge": 8}, title="Find jerky motion (length = pass 1)", color=GREEN)
    remap = g.node("H3ClockRemap", "far", W, ["", "minimax-h3", 8, 4, 8, 1, 25.0, 0.0], inputs=[("hold_map", "STRING", "hold_map")],
                   outputs=[("hold_map", "STRING"), ("length", "INT"), ("report", "STRING"), ("profile_json", "STRING")],
                   api={"model_profile": "minimax-h3"})
    smear = g.node("H3TimeSmear", "far", W, [4, "", True, 24, 0.0, 18, 6.7], inputs=[("images", "IMAGE", None), ("hold_map", "STRING", "hold_map")],
                   outputs=[("images", "IMAGE"), ("hold_map_used", "STRING"), ("length", "INT"), ("report", "STRING")],
                   api={"dilation": 4}, title="Hold frames (slow the jerky stretches)", color=GREEN)
    asmear = g.node("H3AudioSmear", "far", W, ["", 24], inputs=[("audio", "AUDIO", None), ("hold_map", "STRING", "hold_map")],
                    outputs=[("AUDIO", "AUDIO")], api={"fps": 24})
    aenc = g.node("VAEEncodeAudio", "far", W, inputs=[("audio", "AUDIO", None), ("vae", "VAE", None)], outputs=[("LATENT", "LATENT")], api={})
    venc = g.node("VAEEncode", "far", W, inputs=[("pixels", "IMAGE", None), ("vae", "VAE", None)], outputs=[("LATENT", "LATENT")], api={})
    init = g.node("H3V2VInit", "far", W, [0, 0.0, 2, 0, False, False, "follow the original performance (0.5)", 0.5, 0, 0],
                  inputs=[("samples", "LATENT", None), ("oracle_samples", "LATENT", None), ("mask", "MASK", None), ("audio_latent", "LATENT", None)],
                  outputs=[("LATENT", "LATENT")], api={"audio_mode": "follow the original performance (0.5)", "audio_strength": 0.5})
    cond2 = g.node("MiniMaxH3ImageToVideo", "far2", W, ["", 1280, 704, 141],
                   inputs=[("clip", "CLIP", None), ("vae", "VAE", None), ("prompt", "STRING", "prompt"), ("length", "INT", "length")],
                   outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")], api={"width": 1280, "height": 704}, title="Pass 2 (slowed clip)")
    noise2 = g.node("RandomNoise", "far2", W, [43, "fixed"], outputs=[("NOISE", "NOISE")], api={"noise_seed": 43})
    guider2 = g.node("BasicGuider", "far2", W, inputs=[("model", "MODEL", None), ("conditioning", "CONDITIONING", None)], outputs=[("GUIDER", "GUIDER")], api={})
    tail = g.node("ManualSigmas", "far2", W, [sigmas], outputs=[("SIGMAS", "SIGMAS")], api={"sigmas": sigmas},
                  title=steps_title, color=BLUE)
    out2 = g.node("SamplerCustomAdvanced", "far2", W,
                  inputs=[("noise", "NOISE", None), ("guider", "GUIDER", None), ("sampler", "SAMPLER", None), ("sigmas", "SIGMAS", None), ("latent_image", "LATENT", None)],
                  outputs=[("output", "LATENT"), ("denoised_output", "LATENT")], api={})
    dec2 = g.node("VAEDecode", "far2", W, inputs=[("samples", "LATENT", None), ("vae", "VAE", None)], outputs=[("IMAGE", "IMAGE")], api={})
    adec2 = g.node("VAEDecodeAudio", "far2", W, inputs=[("samples", "LATENT", None), ("vae", "VAE", None)], outputs=[("AUDIO", "AUDIO")], api={})
    rec = g.node("H3ExactRecover", "far2", W, [""], inputs=[("images", "IMAGE", None), ("hold_map", "STRING", "hold_map")], outputs=[("IMAGE", "IMAGE")],
                 api={}, title="Back to real time", color=GREEN)
    arec = g.node("H3AudioRecover", "far2", W, ["", 24, 1.0, "keep the original performance (safe default)"],
                  inputs=[("audio", "AUDIO", None), ("hold_map", "STRING", "hold_map"), ("reference", "AUDIO", None)], outputs=[("AUDIO", "AUDIO")],
                  api={"fps": 24, "reference_mix": 1.0, "audio_source": "keep the original performance (safe default)"},
                  title="Audio: keep pass 1's original", color=GREEN)
    vid2 = g.node("CreateVideo", "far2", W, [24, 8], inputs=[("images", "IMAGE", None), ("audio", "AUDIO", None)], outputs=[("VIDEO", "VIDEO")], api={"fps": 24.0})
    save2 = g.node("SaveVideo", "far2", W, ["video/dials_pika_derope", "auto", "auto"], inputs=[("video", "VIDEO", None)], outputs=[],
                   api={"filename_prefix": "video/dials_pika_derope", "format": "mp4", "codec": "h264"})
    for s_, slot, d, nm in ((p1["latent"], 0, oracle, "samples"), (oracle, 0, remap, "hold_map"), (p1["images"], 0, smear, "images"),
                            (remap, 0, smear, "hold_map"), (p1["audio"], 0, asmear, "audio"), (smear, 1, asmear, "hold_map"),
                            (asmear, 0, aenc, "audio"), (p1["avae"], 0, aenc, "vae"), (smear, 0, venc, "pixels"), (p1["vae"], 0, venc, "vae"),
                            (venc, 0, init, "samples"), (aenc, 0, init, "audio_latent"), (p1["clip"], 0, cond2, "clip"), (p1["vae"], 0, cond2, "vae"),
                            (text, 0, cond2, "prompt"), (smear, 2, cond2, "length"), (model, 0, guider2, "model"), (cond2, 0, guider2, "conditioning"),
                            (noise2, 0, out2, "noise"), (guider2, 0, out2, "guider"), (p1["sampler"], 0, out2, "sampler"), (tail, 0, out2, "sigmas"),
                            (init, 0, out2, "latent_image"), (out2, 0, dec2, "samples"), (p1["vae"], 0, dec2, "vae"), (out2, 0, adec2, "samples"),
                            (p1["avae"], 0, adec2, "vae"), (dec2, 0, rec, "images"), (smear, 1, rec, "hold_map"), (adec2, 0, arec, "audio"),
                            (smear, 1, arec, "hold_map"), (p1["audio"], 0, arec, "reference"), (rec, 0, vid2, "images"), (arec, 0, vid2, "audio"),
                            (vid2, 0, save2, "video")):
        g.link(s_, slot, d, nm)
    add_groups(g, (("Models", "left", "#3f5159"), ("Dials + prompt", "mid", "#4f3f59"), ("Pass 1: Pika 6-step", "right", "#3f4f3f"),
                   ("De-rope 1: find and hold (MAINodes)", "far", "#59503f"), ("De-rope 2: re-render, back to real time", "far2", "#59503f")))
    return g


if __name__ == "__main__":
    for name, g in (("dials_studio_h3", studio()), ("dial_maker_h3", maker()), ("dials_pika_h3_flex", pika()),
                    ("dials_pika_h3_flex_derope", pika_derope()),
                    ("dials_pika_h3_flex_derope_6plus6_experimental",
                     pika_derope(PIKA_DEROPE_6PLUS6_SIGMAS, "EXPERIMENTAL 6+6: Pika-shaped steps below 0.48", experimental=True))):
        bad = g.check_overlaps()
        if bad:
            raise SystemExit(f"{name}: overlapping nodes {bad}")
        json.dump(g.ui(), open(os.path.join(HERE, f"{name}.json"), "w"), indent=1)
        json.dump(g.api, open(os.path.join(HERE, f"{name}_api.json"), "w"), indent=1)
        print(f"wrote {name}.json ({len(g.nodes)} nodes) and {name}_api.json")
