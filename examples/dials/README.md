# Dial Rack example workflows

Each workflow comes in two files: the one to open in ComfyUI, and its API-format twin (`*_api.json`).

The model, encoder, VAE and LoRA file names in the loaders are the names on the machine the workflows were made on.
Choose your own files in the loader nodes after opening one.

## Run with this pack alone

These use the Dial Rack and ComfyUI's own nodes, nothing else. They sit in `_archive/`.

| workflow | what it shows |
|---|---|
| `_archive/one_dial.json` | the Dial Rack with one dial |
| `_archive/four_dials.json` | the Dial Rack with four dials |
| `_archive/pika_rack.json` | the Dial Rack on Pika H3 Flex (a LoRA loader before the rack) |

The workflows for the five original nodes (`Dials*`) are in [`../../workflows/`](../../workflows/). Of those,
`dials_studio_h3`, `dial_maker_h3` and `dials_pika_h3_flex` run with this pack alone; the two `*_derope*` ones need
ComfyUI-MAINodes.

## Need ComfyUI-MAINodes

These add the de-rope pass, whose nodes (`H3TimeSmear`, `H3V2VInit`, `H3ClockRemap`, `H3ExactRecover`, `H3JerkOracle`,
`H3AudioSmear`, `H3AudioRecover`) are in ComfyUI-MAINodes and not in this pack. With ComfyUI-MAINodes installed, it
also provides the Dial Rack nodes, and this pack loads only its five original nodes (see the top-level README).

| workflow | also needs from ComfyUI-MAINodes | other packs |
|---|---|---|
| `dials_rack_pika_derope.json` | `RelayVSRModelSettings`, `RelayVSRUpscaleVideo` | |
| `dials_prompt_set_pika_derope.json` | | |
| `dials_two_pictures_pika_derope.json` | | |
| `dials_picture_groups_pika_derope.json` | | |
| `dials_picture_pika_derope.json` | | ComfyUI-KJNodes (`LoadImagesFromFolderKJ`) |
| `dials_simple_pika_derope.json` | `MAIRenderPreset`, `H3DeropeSigmas`, `H3References`, `H3ReferenceToVideoRefs`, `RelayVSRModelSettings`, `RelayVSRUpscaleVideo` | |
| `dials_simple_pika_derope_kj.json` | the same as `dials_simple_pika_derope.json` | ComfyUI-KJNodes (`LoadImagesFromFolderKJ`), and a pack with `MiniMaxChunkFeedForward` and `MiniMaxLowVRAMAttention` |
| `_archive/dials_simple_pika_derope.json` | | |

The picture workflows load pictures from ComfyUI's `input/` folder (`dial_pair_A_lizard.png` and others); put your own
pictures there and choose them in the Load Image nodes.

## Blueprints

| blueprint | needs |
|---|---|
| `../../blueprints/Dials - Image.json` | this pack alone (ComfyUI's own nodes inside) |
| `../../blueprints/Dials - H3 + De-rope.json` | ComfyUI-MAINodes |
| `_archive/blueprints/Dials - Pika H3.json` | this pack alone (ComfyUI's own nodes inside) |
| `_archive/blueprints/Dials - Pika H3 + De-rope.json` | ComfyUI-MAINodes |

## How this list was checked

2026-11-16, ComfyUI 0.45.0: every node type in each file was compared with the node list of a ComfyUI started with
this pack as its only custom node pack, and with the list of one started with this pack and ComfyUI-MAINodes
(v1.2.10). The workflows were not run for this check.
