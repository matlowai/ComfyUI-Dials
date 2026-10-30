"""Dials: training-free steering directions for video models, built from prompt pairs.

A dial is one vector added to every prompt token. Two injection points (Make Dial `inject`):
  post_refiner  after H3's token_refiner, exactly where the DiT reads the prompt ("hook"): the most faithful. Applied by
                Apply Dial as a ModelPatcher object patch on token_refiner.forward; saved dials load with Load Dial.
  pre_refiner   into the bias of `condition_proj` (text encoder output -> DiT hidden; a linear layer with a bias that only
                ever sees text tokens). Saves as a standard `<layer>.diff_b` LoRA key, so the stock LoRA loader applies
                it too, with strength as the dial.

Make Dial: for each pair (A, B), d = mean_tokens(proj(enc(B))) - mean_tokens(proj(enc(A))); the dial is the mean of d
over pairs, so subjects and wordings that differ across pairs cancel. Two corrections, both on by default:
  debias     token means are dominated by the prompt template, so ANY extra wording moves them along a few shared
             "more content" directions. Built-in nuisance pairs (irrelevant wording changes) give those directions;
             their top components are projected out of the dial.
  calibrate  (pre_refiner only) the dial enters before the 2-block token_refiner, whose final RMSNorm shrinks it. The
             node measures the shift it actually produces after the refiner and rescales it so strength 1 = one average
             (debiased) pair difference in the space the DiT reads.
No rendering, no training.
"""
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, io

# diffusion model class -> (text projection layer the dial biases, text refiner after it)
TEXT_PROJECTIONS = {"MiniMaxH3Model": ("condition_proj", "token_refiner")}
NUISANCE_PAIRS = [
    "{subject}. ||| {subject}, seen on an ordinary afternoon in natural light.",
    "{subject} in a room. ||| {subject} in a room with a window, a wooden table and two chairs.",
    "{subject} wearing a blue shirt. ||| {subject} wearing a red shirt.",
    "{subject} near a tree. ||| {subject} near a lamp post on a quiet street.",
    "{subject} in the center of the frame. ||| {subject} on the right side of the frame, slightly out of focus.",
    "A photo of {subject}. ||| A detailed, carefully composed photograph of {subject} taken with a 50mm lens.",
]
NUISANCE_SUBJECTS = "A woman\nA man\nA dog\nAn old car"
DIAL_SUBDIR = "dials"
INJECT = ["post_refiner (hook)", "pre_refiner (LoRA file)"]
KEY_SUFFIX = {"pre": "diff_b", "post": "dials_shift"}  # file key: diffusion_model.<target>.<suffix>

Dial = io.Custom("DIAL")


def text_projection(model):
    dm = model.model.diffusion_model
    names = TEXT_PROJECTIONS.get(type(dm).__name__)
    if names is None:
        raise ValueError(f"Dials support {', '.join(TEXT_PROJECTIONS)} so far, got {type(dm).__name__}")
    return names[0], dm.get_submodule(names[0]), dm.get_submodule(names[1])


def unit(v):
    return v / v.norm().clamp_min(1e-12)


def remove_directions(v, basis):
    return v - basis @ (basis.T @ v) if basis is not None else v


class ShiftedForward:
    """Replacement for a text module's forward (H3 token_refiner) that adds a dial vector to its output. Installed as an
    object patch on `<module>.forward`, so the module tree and weight paths stay unchanged and dials stack."""

    def __init__(self, forward, shift):
        self.forward = forward
        self.shift = shift

    def __call__(self, x, *args, **kwargs):
        out = self.forward(x, *args, **kwargs)
        return out + self.shift.to(device=out.device, dtype=out.dtype)


def parse_pairs(pairs, subjects):
    subs = [s.strip() for s in subjects.splitlines() if s.strip() and not s.strip().startswith("#")] or [None]
    out = []
    for line in pairs.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|||" not in line:
            raise ValueError(f"each pair line needs 'A ||| B', got: {line[:80]}")
        a, b = (x.strip() for x in line.split("|||", 1))
        for s in subs:
            out.append((a.replace("{subject}", s), b.replace("{subject}", s)) if s else (a, b))
    if not out:
        raise ValueError("no prompt pairs given")
    return out


def dial_path(name):
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", name):
        raise ValueError("dial name: letters, digits, '_' and '-' only (max 80)")
    root = os.path.join(folder_paths.get_folder_paths("loras")[0], DIAL_SUBDIR)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{name}.safetensors")


def save_dial(dial, name):
    path = dial_path(name)
    meta = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in dial["meta"].items()}
    key = f"diffusion_model.{dial['target']}.{KEY_SUFFIX[dial['mode']]}"
    save_file({key: dial["vec"].to(torch.bfloat16).contiguous()}, path, metadata=meta)
    return path


def dial_report(dial):
    m = dial["meta"]
    return (f"dial '{m.get('name', '?')}' ({dial['mode']}_refiner, {dial['target']}): {m.get('n_pairs', '?')} pairs, "
            f"pair agreement cos {m.get('pair_agreement_cos', '?')}, |dial| {float(dial['vec'].norm()):.1f}")


class MakeDial(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DialsMakeDial",
            display_name="Make Dial",
            category="Dials",
            description="Turn prompt pairs (A ||| B per line) into a steering dial: negative strength moves toward A, positive toward B.",
            inputs=[
                io.Clip.Input("clip"),
                io.Model.Input("model"),
                io.String.Input("pairs", multiline=True, default="{subject} stands perfectly still, posing for a photograph. ||| {subject} runs quickly across the scene, moving energetically.",
                                tooltip="One pair per line: 'A ||| B'. Use {subject} to expand each line over the subjects list. Lines starting with # are ignored."),
                io.String.Input("subjects", multiline=True, default="A young woman\nAn old man\nA golden dog\nA horse",
                                tooltip="Optional: one subject per line, substituted for {subject}. Varied subjects cancel out, leaving only the A→B difference."),
                io.String.Input("name", default="my_dial", tooltip="File name under models/loras/dials/ when saving."),
                io.Combo.Input("inject", options=INJECT, default=INJECT[0],
                               tooltip="post_refiner (hook): where the DiT reads the prompt, most faithful; load with Load Dial. "
                                       "pre_refiner (LoRA file): a bias diff the stock LoRA loader can apply; slightly different feel."),
                io.Boolean.Input("save", default=True, tooltip="Also save to models/loras/dials/<name>.safetensors."),
                io.Boolean.Input("debias", default=True, optional=True, advanced=True,
                                 tooltip="Project out the generic 'more wording' directions (from built-in nuisance pairs) that every prompt-pair difference shares."),
                io.Boolean.Input("calibrate", default=True, optional=True, advanced=True,
                                 tooltip="Rescale so strength 1 equals one average pair difference after the text refiner (where the DiT reads it)."),
                io.Int.Input("nuisance_dims", default=3, min=1, max=12, optional=True, advanced=True,
                             tooltip="How many nuisance directions to remove."),
            ],
            outputs=[Dial.Output(display_name="dial"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, clip, model, pairs, subjects, name, inject, save, debias=True, calibrate=True, nuisance_dims=3) -> io.NodeOutput:
        pre_name, proj, refiner = text_projection(model)
        mode = "post" if inject.startswith("post") else "pre"
        target = pre_name if mode == "pre" else TEXT_PROJECTIONS[type(model.model.diffusion_model).__name__][1]
        prompts = parse_pairs(pairs, subjects)
        nuisance = parse_pairs("\n".join(NUISANCE_PAIRS), subjects if subjects.strip() else NUISANCE_SUBJECTS) if debias else []
        comfy.model_management.load_models_gpu([model])
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype_inference()
        cache = {}

        def encode(text):  # -> (token-mean before refiner, token-mean after refiner), float32 on device
            if text not in cache:
                ctx = clip.encode_from_tokens_scheduled(clip.tokenize(text))[0][0][0].to(device, dtype)
                pre = proj(ctx)
                cache[text] = (pre, pre.float().mean(0), refiner(pre).float().mean(0))
            return cache[text]

        def diffs(pp, which):
            return torch.stack([encode(b)[which] - encode(a)[which] for a, b in pp])

        def nuisance_basis(which):
            if not nuisance:
                return None
            return torch.linalg.svd(diffs(nuisance, which), full_matrices=False)[2][:nuisance_dims].T  # [hidden, k]

        d_pre, d_post = diffs(prompts, 1), diffs(prompts, 2)
        b_pre, b_post = nuisance_basis(1), nuisance_basis(2)
        d, basis = (d_pre, b_pre) if mode == "pre" else (d_post, b_post)
        raw = d.mean(0)
        vec = remove_directions(raw, basis)
        removed = 1 - float(vec.norm() ** 2 / raw.norm() ** 2)
        agree = float(torch.nn.functional.cosine_similarity(torch.stack([remove_directions(x, basis) for x in d]), vec.unsqueeze(0)).mean())
        note = ""
        if calibrate and mode == "pre":
            post_target = remove_directions(d_post.mean(0), b_post)
            probe = [t for pair in prompts[:8] for t in pair]
            shift = torch.stack([refiner(encode(t)[0] + vec.to(dtype)).float().mean(0) - encode(t)[2] for t in probe]).mean(0)
            along = float(shift @ unit(post_target))
            scale = float(post_target.norm()) / along if along > 0 else 1.0
            vec = vec * scale
            note = f", calibrated x{scale:.2f} (post-refiner alignment cos {float(torch.nn.functional.cosine_similarity(shift, post_target, dim=0)):.2f})"
        vec = vec.cpu()
        dial = {"vec": vec, "target": target, "mode": mode,
                "meta": {"name": name, "pairs": prompts, "n_pairs": str(len(prompts)), "pair_agreement_cos": f"{agree:.3f}",
                         "inject": inject, "debias_removed": f"{removed:.3f}" if debias else "off",
                         "calibrated": str(bool(calibrate and mode == "pre")), "format": "ComfyUI-Dials v1"}}
        report = dial_report(dial) + (f"; debias removed {removed:.0%} of the raw dial (generic wording shift)" if debias else "") + note
        if save:
            report += f"\nsaved {save_dial(dial, name)}"
        if agree < 0.4:
            report += "\nlow pair agreement: the pairs differ in more than one way; tighten the wording"
        return io.NodeOutput(dial, report)


class ApplyDial(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DialsApplyDial",
            display_name="Apply Dial",
            category="Dials",
            inputs=[
                io.Model.Input("model"),
                Dial.Input("dial"),
                io.Float.Input("strength", default=1.0, min=-20.0, max=20.0, step=0.05,
                               tooltip="Negative = toward A, positive = toward B. Stacks with LoRAs and other dials."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, dial, strength) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(model)
        m = model.clone()
        if dial["mode"] == "pre":
            m.add_patches({f"diffusion_model.{dial['target']}.bias": ("diff", (dial["vec"],))}, strength)
        else:
            name = f"diffusion_model.{dial['target']}.forward"
            m.add_object_patch(name, ShiftedForward(m.get_model_object(name), dial["vec"] * strength))
        return io.NodeOutput(m)


class LoadDial(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = [f for f in folder_paths.get_filename_list("loras") if f.replace("\\", "/").startswith(f"{DIAL_SUBDIR}/")]
        return io.Schema(
            node_id="DialsLoadDial",
            display_name="Load Dial",
            category="Dials",
            inputs=[io.Combo.Input("dial_name", options=files)],
            outputs=[Dial.Output(display_name="dial"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, dial_name) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("loras", dial_name)  # resolves inside the loras folders only
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
            if len(keys) != 1 or not keys[0].startswith("diffusion_model."):
                raise ValueError(f"{dial_name} is not a dial file (expected one diffusion_model.<layer>.<diff_b|dials_shift> key)")
            vec = f.get_tensor(keys[0]).float()
            meta = dict(f.metadata() or {})
        target, suffix = keys[0][len("diffusion_model."):].rsplit(".", 1)
        mode = {v: k for k, v in KEY_SUFFIX.items()}.get(suffix)
        if mode is None:
            raise ValueError(f"{dial_name}: unknown dial key suffix '{suffix}'")
        dial = {"vec": vec, "target": target, "mode": mode, "meta": meta}
        return io.NodeOutput(dial, dial_report(dial))


class CombineDials(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DialsCombineDials",
            display_name="Combine Dials",
            category="Dials",
            description="weight_a * a + weight_b * b. With orthogonalize on, b's direction is first removed from a (e.g. face expressiveness minus loudness).",
            inputs=[
                Dial.Input("a"),
                Dial.Input("b"),
                io.Float.Input("weight_a", default=1.0, min=-20.0, max=20.0, step=0.05),
                io.Float.Input("weight_b", default=0.0, min=-20.0, max=20.0, step=0.05),
                io.Boolean.Input("orthogonalize", default=False, tooltip="Remove b's direction from a before mixing."),
            ],
            outputs=[Dial.Output(display_name="dial"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, a, b, weight_a, weight_b, orthogonalize) -> io.NodeOutput:
        if (a["target"], a["mode"]) != (b["target"], b["mode"]):
            raise ValueError(f"dials use different injection points: {a['mode']} {a['target']} vs {b['mode']} {b['target']}")
        va, vb = a["vec"].float(), b["vec"].float()
        cos = float(torch.nn.functional.cosine_similarity(va, vb, dim=0))
        if orthogonalize:
            va = va - (va @ vb) / (vb @ vb) * vb
        vec = weight_a * va + weight_b * vb
        name = f"{a['meta'].get('name', 'a')}*{weight_a:g}{'-proj' if orthogonalize else ''}+{b['meta'].get('name', 'b')}*{weight_b:g}"
        dial = {"vec": vec, "target": a["target"], "mode": a["mode"], "meta": {"name": name, "parents": [a["meta"].get("name"), b["meta"].get("name")]}}
        return io.NodeOutput(dial, f"cos(a, b) = {cos:.3f}; {dial_report(dial)}")


class SaveDial(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DialsSaveDial",
            display_name="Save Dial",
            category="Dials",
            is_output_node=True,
            inputs=[Dial.Input("dial"), io.String.Input("name", default="my_dial")],
            outputs=[io.String.Output(display_name="path")],
        )

    @classmethod
    def execute(cls, dial, name) -> io.NodeOutput:
        dial = {**dial, "meta": {**dial["meta"], "name": name}}
        return io.NodeOutput(save_dial(dial, name))


class DialsExtension(ComfyExtension):
    async def get_node_list(self):
        return [MakeDial, ApplyDial, LoadDial, CombineDials, SaveDial]


async def comfy_entrypoint() -> DialsExtension:
    return DialsExtension()
