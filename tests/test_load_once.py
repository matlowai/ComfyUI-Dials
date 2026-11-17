#!/usr/bin/env python3
"""The load-once rule: who provides the Dial Rack nodes when ComfyUI-MAINodes is installed next to this pack.

  python tests/test_load_once.py      (uses the ComfyUI checkout two levels up, or $COMFYUI_ROOT)

CPU only, no model files, no running ComfyUI. Each case points ComfyUI's custom_nodes list at a temporary folder,
loads this pack's __init__.py the way ComfyUI does (a package under a fresh module name) with a stand-in PromptServer
that records routes, and looks at what the pack registered.
Checks:
  1. no other pack: the five original nodes and the six MAIDial* nodes, the /mainodes/dials routes, WEB_DIRECTORY = web;
  2. a MAINodes pack present (under another folder name): only the five original nodes, no routes, no WEB_DIRECTORY,
     dials/nodes.py not imported, one log line saying so;
  3. the same folder with a name ending in .disabled: as case 1;
  4. --disable-all-custom-nodes without MAINodes on the whitelist: as case 1; with it on the whitelist: as case 2;
  5. a second copy of this pack is not taken for MAINodes; a pack with an unrelated dials/ folder is not either;
  6. the web folder this pack serves has no file that needs MAINodes (every relative import resolves inside it).
Exit code 0 = pass.
"""
import importlib.util
import logging
import os
import re
import shutil
import sys
import tempfile
import types

from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = os.environ.get("COMFYUI_ROOT") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, COMFY)

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # before model_management picks a device
import folder_paths  # noqa: E402
import server  # noqa: E402

LEGACY = {"DialsMakeDial", "DialsApplyDial", "DialsLoadDial", "DialsCombineDials", "DialsSaveDial"}
RACK = {"MAIDialRack", "MAIDialHoldMap", "MAIDialMake", "MAIDialMakeFromPictures", "MAIDialMakeFromVideos", "MAIDialCombine"}
args = comfy.cli_args.args
count = 0


class Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def load(root):
    """Load the pack with `root` as ComfyUI's only custom_nodes folder -> (module, route paths, log lines, module name)."""
    global count
    count += 1
    name = f"dials_pack_case{count}"
    saved = folder_paths.folder_names_and_paths["custom_nodes"]
    folder_paths.folder_names_and_paths["custom_nodes"] = ([root], set())
    routes = web.RouteTableDef()
    server.PromptServer.instance = types.SimpleNamespace(routes=routes)
    handler, logger = Lines(), logging.getLogger()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        folder_paths.folder_names_and_paths["custom_nodes"] = saved
        logger.removeHandler(handler)
        logger.setLevel(level)
    return module, sorted({r.path for r in routes}), [x for x in handler.lines if "[ComfyUI-Dials]" in x], name


def provides_rack(root):
    module, routes, lines, name = load(root)
    assert set(module.NODE_CLASS_MAPPINGS) == LEGACY | RACK, sorted(module.NODE_CLASS_MAPPINGS)
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == LEGACY | RACK
    assert module.WEB_DIRECTORY == "./web" and os.path.isfile(os.path.join(PACK, module.WEB_DIRECTORY, "dial_rack.js"))
    assert routes and all(r.startswith("/mainodes/dials/") for r in routes), routes
    assert f"{name}.dials.nodes" in sys.modules and module.RACK_PROVIDER == "ComfyUI-Dials" and not lines, lines
    return routes


def defers(root):
    module, routes, lines, name = load(root)
    assert set(module.NODE_CLASS_MAPPINGS) == LEGACY, sorted(module.NODE_CLASS_MAPPINGS)
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == LEGACY
    assert not hasattr(module, "WEB_DIRECTORY"), module.WEB_DIRECTORY  # ComfyUI serves a pack's web folder only if this is set
    assert routes == [], routes
    assert not any(k.startswith(f"{name}.dials") for k in sys.modules), [k for k in sys.modules if k.startswith(name)]
    assert module.RACK_PROVIDER == "ComfyUI-MAINodes" and len(lines) == 1 and "ComfyUI-MAINodes is installed" in lines[0], lines
    return lines[0]


def fake_pack(root, name, init="NODE_CLASS_MAPPINGS = {}\n", rack='class DialRack:\n    node_id="MAIDialRack"\n'):
    os.makedirs(os.path.join(root, name, "dials"))
    open(os.path.join(root, name, "__init__.py"), "w").write(init)
    open(os.path.join(root, name, "dials", "nodes.py"), "w").write(rack)
    return os.path.join(root, name)


def check_web():
    """Every relative import of the served .js files is a file in web/ (or ComfyUI's own scripts/app.js, scripts/api.js)."""
    web_dir, seen = os.path.join(PACK, "web"), 0
    for folder, _, files in os.walk(web_dir):
        for f in files:
            if not f.endswith(".js"):
                continue
            path = os.path.join(folder, f)
            depth = os.path.relpath(folder, web_dir).count(os.sep) + (0 if folder == web_dir else 1)
            for spec in re.findall(r'^import [^"\']*["\']([^"\']+)["\'];', open(path).read(), re.M):
                seen += 1
                if spec in ("../" * (depth + 2) + "scripts/app.js", "../" * (depth + 2) + "scripts/api.js"):
                    continue
                target = os.path.normpath(os.path.join(folder, spec))
                assert spec.startswith("./") and target.startswith(web_dir + os.sep) and os.path.isfile(target), f"{path}: {spec}"
    assert seen >= 6, seen
    return seen


def main():
    assert not args.disable_all_custom_nodes
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        routes = provides_rack(empty)
        print(f"1. alone: {len(LEGACY)} original + {len(RACK)} rack nodes, {len(routes)} route paths, web served")

        present = os.path.join(tmp, "present")
        fake_pack(present, "some-other-folder-name")
        line = defers(present)
        print(f"2. next to MAINodes: {len(LEGACY)} original nodes only, no routes, no web folder\n   log: {line}")

        disabled = os.path.join(tmp, "disabled")
        fake_pack(disabled, "ComfyUI-MAINodes.disabled")
        provides_rack(disabled)
        print("3. MAINodes folder ending in .disabled: this pack provides the rack")

        args.disable_all_custom_nodes, args.whitelist_custom_nodes = True, ["ComfyUI-Dials"]
        try:
            provides_rack(present)
            args.whitelist_custom_nodes = ["ComfyUI-Dials", "some-other-folder-name"]
            defers(present)
        finally:
            args.disable_all_custom_nodes, args.whitelist_custom_nodes = False, []
        print("4. --disable-all-custom-nodes: provides the rack unless MAINodes is on the whitelist")

        twin = os.path.join(tmp, "twin")
        copy = fake_pack(twin, "ComfyUI-Dials-copy", init=open(os.path.join(PACK, "__init__.py")).read())
        shutil.copy(os.path.join(PACK, "dials", "nodes.py"), os.path.join(copy, "dials", "nodes.py"))
        fake_pack(twin, "unrelated", rack="NODE_CLASS_MAPPINGS = {}\n")
        os.makedirs(os.path.join(twin, "no_init", "dials"))
        shutil.copy(os.path.join(PACK, "dials", "nodes.py"), os.path.join(twin, "no_init", "dials", "nodes.py"))
        provides_rack(twin)
        print("5. a copy of this pack, an unrelated dials/ folder and a folder without __init__.py are not MAINodes")

    print(f"6. web folder: {check_web()} imports, all inside web/ or ComfyUI's scripts/")
    print("PASS")


if __name__ == "__main__":
    main()
