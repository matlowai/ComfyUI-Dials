#!/usr/bin/env python3
"""Build the example workflows (UI + API twins) for ComfyUI-Dials (MiniMax H3 text-to-video).

The column layout / overlap check follows the MOSS preset builder in the author's ComfyUI workflows.

  python3 workflows/build_workflows.py
Writes workflows/dials_studio_h3.json, workflows/dial_maker_h3.json and *_api.json (API twins, used for testing).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
PROMPT = "A busker plays guitar on a busy city sidewalk at sunset while people walk past and a dog watches."
GAP, TITLE = 40, 34
COLS = {"left": -900, "mid": -380, "right": 140}
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


def h3_render(g, model_src, note):
    """Loaders + H3 T2V sampling chain on the right; model_src = (node, slot) of the dialed MODEL."""
    clip = g.node("CLIPLoader", "left", 320, [TE, "minimax", "default"], outputs=[("CLIP", "CLIP")],
                  api={"clip_name": TE, "type": "minimax", "device": "default"})
    vae = g.node("VAELoader", "left", 320, [VAE], outputs=[("VAE", "VAE")], api={"vae_name": VAE})
    cond = g.node("MiniMaxH3ImageToVideo", "right", 420, [PROMPT, 832, 480, 124],
                  inputs=[("clip", "CLIP", None), ("vae", "VAE", None)], outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")],
                  api={"prompt": PROMPT, "width": 832, "height": 480, "length": 124}, multiline=1, title="Prompt (MiniMax H3)")
    g.link(clip, 0, cond, "clip")
    g.link(vae, 0, cond, "vae")
    noise = g.node("RandomNoise", "right", 420, [1001, "fixed"], outputs=[("NOISE", "NOISE")], api={"noise_seed": 1001})
    guider = g.node("BasicGuider", "right", 420, inputs=[("model", "MODEL", None), ("conditioning", "CONDITIONING", None)],
                    outputs=[("GUIDER", "GUIDER")], api={})
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
    save = g.node("SaveVideo", "right", 420, ["video/dials", "auto", "auto"], inputs=[("video", "VIDEO", None)], outputs=[],
                  api={"filename_prefix": "video/dials", "format": "mp4", "codec": "h264"})
    g.link(model_src[0], model_src[1], guider, "model")
    g.link(model_src[0], model_src[1], sched, "model")
    g.link(cond, 0, guider, "conditioning")
    for s, slot, d, nm in ((noise, 0, out, "noise"), (guider, 0, out, "guider"), (sampler, 0, out, "sampler"), (sched, 0, out, "sigmas"),
                           (cond, 1, out, "latent_image"), (out, 0, dec, "samples"), (vae, 0, dec, "vae"), (dec, 0, vid, "images"),
                           (out, 0, adec, "samples"), (avae, 0, adec, "vae"), (adec, 0, vid, "audio"), (vid, 0, save, "video")):
        g.link(s, slot, d, nm)
    g.node("MarkdownNote", "left", 320, [note], height=330)
    return clip


def studio():
    g = G()
    unet = g.node("UNETLoader", "left", 320, [UNET, "default"], outputs=[("MODEL", "MODEL")], api={"unet_name": UNET, "weight_dtype": "default"})
    m_dial = g.node("DialsLoadDial", "mid", 380, ["dials/h3_motion.safetensors"], outputs=[("dial", "DIAL"), ("report", "STRING")],
                    api={"dial_name": "dials/h3_motion.safetensors"}, title="Dial: Freeze ↔ Zoomies", color=PURPLE)
    m_apply = g.node("DialsApplyDial", "mid", 380, [1.0], inputs=[("model", "MODEL", None), ("dial", "DIAL", None)],
                     outputs=[("MODEL", "MODEL")], api={"strength": 1.0}, title="Motion  (−2 frozen … +2 zoomies)", color=PURPLE)
    g.link(unet, 0, m_apply, "model")
    g.link(m_dial, 0, m_apply, "dial")
    h3_render(g, (m_apply, 0), "**Dials Studio — MiniMax H3**\n\nDials are ~14 KB steering files made from prompt pairs (no training). "
              "The **Apply Dial** strength is a slider: negative moves toward the pair's A side (still), positive toward B (moving).\n\n"
              "- **−2**: near-frozen, cinemagraph-like.\n- **+2**: everything zooms.\n\n"
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
                     "`models/loras/dials/<name>.safetensors`.\n4. Preview with the strength slider, then use the dial anywhere.")
    g.link(clip, 0, make, "clip")
    return g


if __name__ == "__main__":
    for name, g in (("dials_studio_h3", studio()), ("dial_maker_h3", maker())):
        bad = g.check_overlaps()
        if bad:
            raise SystemExit(f"{name}: overlapping nodes {bad}")
        json.dump(g.ui(), open(os.path.join(HERE, f"{name}.json"), "w"), indent=1)
        json.dump(g.api, open(os.path.join(HERE, f"{name}_api.json"), "w"), indent=1)
        print(f"wrote {name}.json ({len(g.nodes)} nodes) and {name}_api.json")
