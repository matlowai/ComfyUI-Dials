# ComfyUI-Dials

**Training-free steering dials for MiniMax H3, made from prompt pairs.** Write two prompts that differ in one quality,
for example *"stands perfectly still"* vs *"runs across the scene"*. Dials turns the difference into a **~14 KB file**
that works like a strength slider:

- negative strength moves every generation toward the first prompt;
- positive strength moves it toward the second.

The first released dial is **Motion: Freeze ↔ Zoomies**:

| strength | what you get |
|---|---|
| **−2** | near-frozen, cinemagraph-like scenes (0.05× base motion) |
| 0 | the model as usual |
| **+2** | people sprint instead of walk, traffic rushes, the camera swings (about 2.5–3× base motion) |

Also included: a **villainy pack** of eight evil flavours (sinister, creepy, cold, unhinged, seductive, corrupted,
dread, impish). See [Released dials](#released-dials-dials).

![motion dial −2 / base / +2](assets/motion_dial_dog.gif)

[▶ Full demo video (three prompts)](assets/motion_dial_demo.mp4)

- **No training, no reference images, no rendering** needed to make a dial. Building one takes about a minute (one
  text-encoder pass per prompt).
- **~14 KB per dial.** A typical video LoRA is 80–160 MB.
- **Stacks** with LoRAs and with other dials.

## What a dial is, and how it differs from things you know

A dial is a **direction**: "more or less of something the model already understands", applied to every prompt you
write. Here's how that compares:

| | **Dial** | **Slider LoRA** (Concept Sliders style) | **LoRA** | **RefMod** (H3 reference mods) | **Textual-inversion embedding** (SD1.5 era) | **Prompt weighting** `(word:1.3)`, negative prompts |
|---|---|---|---|---|---|---|
| **what it is** | one vector added to the processed prompt | low-rank weight change trained to move along a concept | trained weight change | saved reference images/video/audio, fed as reference tokens | a trained fake word in the text encoder's input | reweighting your own prompt words |
| **how it's made** | **computed** from prompt pairs in ~1 minute; no training, no images | trained, typically on image or prompt pairs, for minutes to hours | trained on a dataset | encoded from your reference media | trained on example images, often hours | typed |
| **what it changes** | *how much* of a quality (motion, mood, …) in **any** prompt | *how much* of a quality | adds *new* things (faces, styles) | *what* to show: this character, this clip's motion | a *thing* you invoke by name | the words you wrote |
| **size** | ~14 KB | ~1–100 MB | 10s–100s of MB | ~1 MB+ | a few KB | — |
| **signed slider** | yes, −/+ in one file | yes | strength only | strength is a blur blend; Axis switches between two references | weight only | yes |
| **needs the concept in the prompt** | no; it steers whatever you prompt | no | trigger word | reference input | yes, its token | yes |
| **works on H3 today** | yes | needs training per concept | yes | yes | n/a (different encoder) | H3 is guidance-distilled (no CFG), so **negative prompts do nothing** |

The closest relatives:

- **Steering vectors / Activation Addition** (Turner et al., arXiv:2308.10248): add the activation difference of a
  prompt pair to a model. Dials do that in a video model's prompt space and package the result as a file.
- **Concept Sliders** (Gandikota et al., arXiv:2311.12092): the *trained* version of a slider.
- **SD-era tricks:** compared with an embedding, a dial is not a word you type. Compared with `ConditioningAverage`,
  it averages a difference over many pairs, so subject-specific detail cancels and only the concept is left.
- **Negative prompts:** a dial gives back the kind of steering negative prompts used to give. H3 has no CFG, so it
  ignores them.

What a dial **can't** do: teach the model something new (a person, a style it doesn't know). That's what LoRAs are
for. The combination works well: a LoRA for *what*, dials for *how*.

## How it works

H3's text encoder output passes through `condition_proj`, a linear layer that only sees prompt tokens, and then a
2-block `token_refiner` before the diffusion transformer reads it.

**Make Dial** encodes each prompt pair (A, B) and averages over tokens. Then it takes B − A and averages over all
pairs. Subjects and wording that vary across pairs cancel; the shared A→B difference remains.

Two places to put the result (`inject`):

- **post_refiner (hook, default):** the vector is added after the token refiner, exactly where the DiT reads the
  prompt. This is the cleanest and most faithful mode. **Apply Dial** installs it as a ModelPatcher object patch on
  `token_refiner.forward`. The module tree and weight paths stay untouched, so LoRAs and other dials stack normally.
  Saved hook dials load with **Load Dial**.
- **pre_refiner (LoRA file):** the vector goes into `condition_proj`'s bias as a standard `diff_b` LoRA key, so the
  **stock "Load LoRA (model only)" node** applies it too. It is gentler per unit and slightly different in feel,
  because the refiner reshapes it.

Two corrections, on by default:

- **Debias.** Before the refiner, token averages are dominated by the prompt template, so *any* extra wording moves
  them along a few shared "more content" directions. In pre_refiner mode that generic shift was 76–99 % of raw dials.
  Make Dial projects out the top directions of built-in nuisance pairs (irrelevant wording changes). After the
  refiner it removes only 2–5 %, because the refiner already normalizes most of it away.
- **Calibrate** (pre_refiner only). Make Dial runs the refiner and rescales so strength 1 means one average pair
  difference where the DiT reads the prompt.

## Nodes (category: Dials)

| node | what it does |
|---|---|
| **Make Dial** | CLIP + MODEL + prompt pairs (`A \|\|\| B` per line, optional `{subject}` template) → DIAL + report. Options: `inject` (hook / LoRA file), debias, calibrate. Saves to `models/loras/dials/<name>.safetensors`. |
| **Apply Dial** | MODEL + DIAL + strength → MODEL (a normal patch; stacks). |
| **Load Dial** | picks a saved dial from `models/loras/dials/` (hook or LoRA-file). |
| **Combine Dials** | `wa·a + wb·b`, optionally removing b's direction from a first. |
| **Save Dial** | saves any DIAL (e.g. a combination). |

## Quick start

1. Install:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/matlowai/ComfyUI-Dials
   ```
   No extra dependencies.
2. Copy `dials/*.safetensors` into `ComfyUI/models/loras/dials/`.
3. Load a workflow:
   - `workflows/dials_studio_h3.json`: the motion slider plus an evil slot on H3 text-to-video, with audio;
   - `workflows/dial_maker_h3.json`: make and preview your own dial;
   - `workflows/dials_pika_h3_flex*.json`: the motion dial on Pika H3 Flex (6 steps), with and without de-rope. See
     [Pika H3 Flex + Dials](#pika-h3-flex--dials).

   API-format twins (`*_api.json`) are included.

**Requirements:** MiniMax H3 in ComfyUI (Comfy-Org repack). Other DiTs with a text projection and refiner are a
table entry away (`TEXT_PROJECTIONS` in `nodes.py`).

**Writing good pairs:**
- Make A and B identical except for the one quality you want.
- Use `{subject}` with 4–8 varied subjects.
- Describe the effect you actually want. Pairs teach *literally*: "runs past the camera" teaches subjects to exit the
  frame. For whole-scene motion, describe the camera and environment moving too.

The report gives two numbers:
- **Pair agreement:** the mean cosine of each pair's difference to the dial. Above 0.6 is a clean direction.
- **Debias removed:** how much of the raw difference was generic wording shift.

## Released dials (`dials/`)

| file | mode | A (−) | B (+) | notes |
|---|---|---|---|---|
| `h3_motion.safetensors` | hook | still, posing, frozen | running, jumping, moving energetically | 24 pairs (8 subjects × 3), agreement 0.78. Useful range −2…+2 |
| `h3_motion_lora.safetensors` | LoRA file (stock LoRA loader or Load Dial) | same pairs | same | ~0.6× the hook's effect per unit, so use ~1.6× the strength |

### Villainy pack: eight flavours of evil (hook)

Every evil dial was made with **debias on**, so each one is a distinct direction. Motion is unrelated to all of them
(cosine ≈ 0), and they cluster sensibly: sinister ≈ cold, creepy ≈ dread, impish ≈ seductive. **+1 to +2** is the
sweet spot; **+4 is extreme** (sinister at +4 goes almost pitch black).

| file | A (−) | B (+) | what it does (from our tests) |
|---|---|---|---|
| `h3_evil_sinister` | warm, friendly presence | menacing presence, shadows across the face | dim rooms, looming foreground silhouettes. ⚠️ **Tends to break the audio:** speech gets crushed (Whisper word error 0.84 at +2 vs 0.00 at base on a two-line dialogue test). Great for atmosphere; for clear dialogue keep it low |
| `h3_evil_creepy` | natural smile and movement | unnervingly wide smile, subtly *wrong* movement | sterile, over-lit spaces, stiff staring people |
| `h3_evil_cold` | warm, expressive | flat, calculating, emotionless | the icy corporate-villain read |
| `h3_evil_unhinged` | calm, composed | maniacal, wild-eyed | ⚠️ **really is unhinged, beware:** screaming, lunging intimidation even at +2. It leans aggressive; we saw no gore, but use it knowingly |
| `h3_evil_seductive` | sincere smile | charming smile hiding dark intent | keeps speech clean (word error 0.00 at +2) |
| `h3_evil_corrupted` | clean, healthy surroundings | decay and rot creeping in | dark-fantasy decay |
| `h3_evil_dread` | ordinary peaceful moment (settings only) | foreboding, something about to happen | atmosphere without a character |
| `h3_evil_impish` | polite, innocent | mischievous, plotting a trick | the lighter, trickster end of evil |

These have been checked on a handful of scenes (an office, a library, a two-line dialogue), so treat them as
creative tools, not guarantees. Pair files for every dial are in `example_pairs/`. The library, with demo videos, is
on Hugging Face: **[MATLOWAI/H3-Dials](https://huggingface.co/MATLOWAI/H3-Dials)**.

## Pika H3 Flex + Dials

[Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) is a 6-step, no-CFG LoRA for MiniMax H3. Dials work on it
unchanged: the motion dial sits between the Pika LoRA and the sampler.

[▶ Motion dial on Pika, 0 to +3](assets/pika_dial_sweep.mp4)

| workflow | what it does |
|---|---|
| `dials_pika_h3_flex.json` | Pika 6-step + motion dial. 1280x704 (about 0.9 MP), 141 frames. |
| `dials_pika_h3_flex_derope.json` | the same, plus a de-rope pass: finds the stretches that move too fast for the frame rate, holds those frames, re-renders the slowed clip from inject 0.48 and maps it back to real time. Keeps pass 1's original audio. Saves both the plain and the de-roped clip. |
| `dials_pika_h3_flex_derope_6plus6_experimental.json` | **experimental:** the de-rope pass takes 6 steps whose spacing copies Pika's own schedule, squeezed below inject 0.48. These sigmas are off Pika's trained steps. |

**You need:** the Pika LoRA `pika_h3_flex_fl2va_v1_bf16.safetensors` (`loras/` in
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3); the same weights are in Pika-Labs/Pika-H3-Flex
`comfy/loras/`). The de-rope workflows also need [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes).
Keep the **lcm** sampler and the step schedules as they are.

**Motion dial on Pika** (1280x704, one prompt, seed 42, one take per strength; camera-removed optical flow):

| dial | motion vs dial 0 | what changes |
|---|---|---|
| +0.5 | +20 % | same scene, more action |
| +1.0 | +23 % | same scene, more action |
| +1.5 / +2.0 | about +90 % | the scene restages, characters morph |
| +3.0 | +137 % | loses the prompt |

On Pika, **+0.5 to +1.0** is the useful range.

**Time** (1280x704, 141 frames, one RTX PRO 6000 Max-Q):

| render | time |
|---|---|
| Pika 6 steps | about 1.5 min |
| Pika 6 steps + de-rope | about 5.5 min |
| MiniMax H3, 25 steps, no de-rope | about 5.5 min |

So a de-roped Pika clip costs about the same as one plain 25-step render of the base model.

**De-rope strength** (dial +1.0; how much the de-rope pass changed the clip, PSNR against the plain clip, higher = closer):

| de-rope pass | PSNR vs plain |
|---|---|
| Pika's last 3 steps (re-enters at about inject 0.25) | 22.5 dB |
| **inject 0.48, Pika's steps (the workflow default)** | 18.2 dB |
| experimental 6+6 below 0.48 | measurement in progress |

**No background music:** write `non_diegetic_music: N/A` (Pika's prompt guide).

## Measured

Seed-matched renders on six motion prompts, measuring camera-removed optical flow:

- **hook +2:** 2.7× base pixel motion; **−2:** 0.05× (near-frozen).
- **The node reproduces the reference implementation:** mean pixel difference 1.4/255 on a test clip, motion 4.266
  vs 4.262.

## Background

Dials came out of a study of why image-only LoRAs make H3 video stiller:
- An image-only "detail" LoKr costs about 20 % of subject motion.
- Post-hoc weight surgery couldn't separate the look from the stillness.
- Its change to the processed prompt points against this motion dial, consistent with captions full of "stands",
  "poses" and "sits".

A research write-up is in progress (MatlowAI).

## Thanks

Thanks to **DaveSnow** for the Natural_detailV2 dataset and his training recipe. His observation that training only
every second block reduces motion loss with image-only data started all of this. Thanks also to the H3 community
Discord, **Comfy-Org** (ComfyUI and the H3 repack), **MiniMax** (H3) and **ostris** (ai-toolkit). Powered by MiniMax
H3.

## License

- **AGPL-3.0** (`LICENSE`): everything in this repository **except the dial weight files**: the node code, workflows,
  pair files, scripts and docs.
- **MiniMax H3 Community License Agreement:** the dial weight files only, i.e. `dials/*.safetensors` and any dial you
  make with these nodes. They are Model Derivatives of MiniMax H3.
  - A copy of the Agreement: `dials/LICENSE-MiniMax-H3-Community.txt`.
  - The required notice: `dials/NOTICE`.
  - Use is limited to the Agreement's Applicable Territory and its Acceptable Use Policy.

Nothing else in this repository is covered by the MiniMax license.
