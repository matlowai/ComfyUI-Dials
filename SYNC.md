# Sync with ComfyUI-MAINodes

The Dials code in this pack is a copy of the Dials code in ComfyUI-MAINodes. This file is written by
`tools/sync_from_mainodes.py`; run that script to update the copy, and do not edit the copied files here.

- source: https://github.com/matlowai/ComfyUI-MAINodes.git
- commit: `382f640bcc2cf45d0d7c6c6416a3a5997c9154fa` (2026-11-16T18:33:08-05:00 Merge remote-tracking branch 'origin/main' into snap-research-2026-11-08)
- MAINodes version at that commit: 1.2.10
- synced: 2026-11-16
- command: `python tools/sync_from_mainodes.py <MAINodes checkout>`
- check without writing: `python tools/sync_from_mainodes.py <MAINodes checkout> --check`

## What was copied

| area | files | bytes |
|---|---|---|
| dials docs (README, ROADMAP, LESSONS, docs/) | 7 | 321768 |
| dials package (Python) | 10 | 126104 |
| dials research notes | 65 | 322191 |
| example workflows | 24 | 1143731 |
| blueprints | 2 | 153412 |
| tests | 17 | 153989 |
| rack web file | 1 | 135143 |
| node help pages | 2 | 12096 |
| generated | 1 | 8560 |
| **total** | 129 | 2376994 |

The full list is in `tools/sync_manifest.json`.

## Differences from MAINodes

Every copied file is byte-identical to MAINodes at the commit above, except:

1. `web/dial_rack.js`: two import lines (sha256 of this pack's copy: `07d389f2298ec020232019c220f3b9198e4c1886aa522474bd5b97db06790b42`).
   ```diff
   -import * as dialCatalog from "./snap/dials.js";
   +import * as dialCatalog from "./standalone/dial_catalog.js";
   -import * as modelCatalog from "./snap/models.js";
   +import * as modelCatalog from "./standalone/model_catalog.js";
   ```
   `web/standalone/dial_catalog.js` and `web/standalone/model_catalog.js` are this pack's own files: they give the
   rack the two functions it uses from each Snap module (`load`, `find`; `load`, `entry`) without MAINodes' Snap.
2. `web/standalone/model_graph.js` is generated from MAINodes' `snap_cards/models.json`: the family name patterns and, per
   known model file, its family and kind. The rack uses it to tell which model family is wired in.
3. Paths that differ (contents identical):
   - `subgraphs/Dials - H3 + De-rope.json` -> `blueprints/Dials - H3 + De-rope.json`
   - `subgraphs/Dials - Image.json` -> `blueprints/Dials - Image.json`

## Not copied

Weight files are never copied by the script. Skipped at this commit:

- `dials/research/word_dials/h3_jc_biolum.safetensors`
- `dials/research/word_dials/h3_jc_bland.safetensors`
- `dials/research/word_dials/h3_jc_honey.safetensors`
- `dials/research/word_dials/h3_jc_hush.safetensors`
- `dials/research/word_dials/h3_jc_origami.safetensors`
- `dials/research/word_dials/h3_jc_rust.safetensors`
- `dials/research/word_dials/h3_jc_stainedglass.safetensors`
- `dials/research/word_dials/h3_jc_static.safetensors`
- `dials/research/word_dials/h3_jc_still.safetensors`
- `dials/research/word_dials/h3_jc_thunder.safetensors`
- `dials/research/word_dials/h3_jc_underwater.safetensors`
- `dials/research/word_dials/h3_jc_vertigo.safetensors`

Dials-related files in MAINodes that are left there on purpose:

- `tests/test_hold_map_remap.py`: tests H3 Hold Map Remap in MAINodes' motion.py (Motion Lab), which is not in this pack.
- `render.py, web/render_preset.js, web/docs/MAIRenderPreset.md, web/docs/H3DeropeSigmas.md, tests/test_render_preset.py`: Render Preset and H3 De-rope Sigmas sit in the MAINodes/Dials menu but are render/de-rope nodes outside the dials package.
- `web/snap/dials.js, web/snap/models.js, web/snap/director/dialog.mjs, tests/snap/*dial*, .claude/skills/snap-dials/`: MAINodes' Snap (its dial picker, model graph and agent skill); this pack has small stand-ins in web/standalone/ instead.
- `snap_cards/models.json`: only family patterns and each file's family and kind are taken, into web/standalone/model_graph.js.
