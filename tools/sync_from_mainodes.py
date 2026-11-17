#!/usr/bin/env python3
"""Copy the Dials files this pack shares with ComfyUI-MAINodes from a MAINodes checkout.

  python tools/sync_from_mainodes.py /path/to/ComfyUI-MAINodes            copy, then write SYNC.md
  python tools/sync_from_mainodes.py /path/to/ComfyUI-MAINodes --check    compare only; exit 1 on any difference
  ... --ref <commit or branch>                                            read another commit (default HEAD)

File contents are read from the commit itself (`git show <commit>:<path>`), never from the working tree, so untracked,
ignored and uncommitted material in the checkout cannot be copied and the commit named in SYNC.md is exactly what
was copied. The MAINodes checkout is only read.

What is shared (RULES below): the `dials/` Python package with its docs and research notes, the Dial Rack's web file and
node help pages, the dial tests, the example workflows and the two blueprints. Everything is copied byte for byte except:
  - web/dial_rack.js: two import lines point at this pack's stand-ins instead of MAINodes' Snap modules (REWRITES);
  - *.safetensors files are skipped (SKIP_SUFFIXES): this script adds no weight files;
  - web/standalone/model_graph.js is generated from MAINodes' snap_cards/models.json (file name, family and kind only).
A file that was copied by an earlier sync and is gone from MAINodes is reported, not deleted.
"""
import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.dirname(HERE)

# (area, source pattern, target path for a source path). Patterns are fnmatch on the full path; "dir/**" is a prefix.
RULES = [
    ("dials package (Python)", "dials/*.py", lambda p: p),
    ("dials docs (README, ROADMAP, LESSONS, docs/)", "dials/*.md", lambda p: p),
    ("dials docs (README, ROADMAP, LESSONS, docs/)", "dials/docs/**", lambda p: p),
    ("dials research notes", "dials/research/**", lambda p: p),
    ("tests", "tests/test_dial*.py", lambda p: p),
    ("rack web file", "web/dial_rack.js", lambda p: p),
    ("node help pages", "web/docs/MAIDial*.md", lambda p: p),
    ("example workflows", "examples/dials/**", lambda p: p),
    # not `subgraphs/`: ComfyUI lists every pack's subgraphs/*.json as blueprints, and MAINodes already ships these two
    ("blueprints", "subgraphs/Dials - *.json", lambda p: "blueprints/" + p.split("/", 1)[1]),
]
SKIP_SUFFIXES = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf")
# exact line -> replacement; each must occur exactly once in the file, or the sync stops
REWRITES = {
    "web/dial_rack.js": [
        ('import * as dialCatalog from "./snap/dials.js";', 'import * as dialCatalog from "./standalone/dial_catalog.js";'),
        ('import * as modelCatalog from "./snap/models.js";', 'import * as modelCatalog from "./standalone/model_catalog.js";'),
    ],
}
# every import the rack file may have after the rewrite; anything else is a new dependency someone has to look at
RACK_IMPORTS = {"../../scripts/app.js", "../../scripts/api.js", "./standalone/dial_catalog.js", "./standalone/model_catalog.js"}
MODEL_GRAPH_SOURCE = "snap_cards/models.json"
MODEL_GRAPH_TARGET = "web/standalone/model_graph.js"
# Dials-related files in MAINodes that are deliberately not shared, with the reason (written into SYNC.md)
NOT_SHARED = [
    ("tests/test_hold_map_remap.py", "tests H3 Hold Map Remap in MAINodes' motion.py (Motion Lab), which is not in this pack"),
    ("render.py, web/render_preset.js, web/docs/MAIRenderPreset.md, web/docs/H3DeropeSigmas.md, tests/test_render_preset.py",
     "Render Preset and H3 De-rope Sigmas sit in the MAINodes/Dials menu but are render/de-rope nodes outside the dials package"),
    ("web/snap/dials.js, web/snap/models.js, web/snap/director/dialog.mjs, tests/snap/*dial*, .claude/skills/snap-dials/",
     "MAINodes' Snap (its dial picker, model graph and agent skill); this pack has small stand-ins in web/standalone/ instead"),
    ("snap_cards/models.json", "only family patterns and each file's family and kind are taken, into web/standalone/model_graph.js"),
]
MANIFEST = "tools/sync_manifest.json"
NOTE = "SYNC.md"


def git(source, *args, text=True):
    return subprocess.run(["git", "-C", source, *args], check=True, capture_output=True, text=text).stdout


def matches(path, pattern):
    return path.startswith(pattern[:-2]) if pattern.endswith("/**") else fnmatch.fnmatchcase(path, pattern) and path.count("/") == pattern.count("/")


def plan(source, commit):
    """-> [(area, source path, target path)], [skipped source paths] for the files tracked in `commit`."""
    tracked = git(source, "ls-tree", "-r", "-z", "--name-only", commit).split("\0")
    out, skipped = [], []
    for path in sorted(p for p in tracked if p):
        for area, pattern, to in RULES:
            if matches(path, pattern):
                if path.endswith(SKIP_SUFFIXES):
                    skipped.append(path)
                else:
                    out.append((area, path, to(path)))
                break
    return out, skipped


def rewritten(path, data):
    if path not in REWRITES:
        return data
    text = data.decode("utf-8")
    lines = text.split("\n")
    for old, new in REWRITES[path]:
        if lines.count(old) != 1:
            sys.exit(f"{path}: expected exactly one line {old!r}, found {lines.count(old)}. The rack's imports changed; update REWRITES.")
        lines[lines.index(old)] = new
    imports = {line.split('"')[-2] for line in lines if line.startswith("import ") and '"' in line}
    if imports != RACK_IMPORTS:
        sys.exit(f"{path}: imports are {sorted(imports)}, expected {sorted(RACK_IMPORTS)}. A new import needs a stand-in or a copy.")
    return "\n".join(lines).encode("utf-8")


def model_graph(source, commit):
    """The stand-in's model table: which family a file name belongs to and what kind of file it is, nothing else."""
    graph = json.loads(git(source, "show", f"{commit}:{MODEL_GRAPH_SOURCE}"))
    families = [{"name": f["name"], "names": f["names"]} for f in graph["families"]]
    files = [{"file": e["file"], "family": e.get("family"), "kind": e.get("kind")} for e in graph["files"]]
    body = json.dumps({"families": families, "files": files}, indent=1, ensure_ascii=False)
    return (f"// Generated by tools/sync_from_mainodes.py from ComfyUI-MAINodes {MODEL_GRAPH_SOURCE}; do not edit by hand.\n"
            f"export const GRAPH = {body};\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="a ComfyUI-MAINodes git checkout (read only)")
    ap.add_argument("--ref", default="HEAD", help="commit or branch to read (default HEAD)")
    ap.add_argument("--check", action="store_true", help="compare only; write nothing")
    args = ap.parse_args()
    source = os.path.abspath(args.source)
    commit = git(source, "rev-parse", f"{args.ref}^{{commit}}").strip()
    described = git(source, "log", "-1", "--format=%cI %s", commit).strip()
    try:
        remote = git(source, "remote", "get-url", "origin").strip()
    except subprocess.CalledProcessError:
        remote = "(no origin remote)"
    try:
        version = next(line.split('"')[1] for line in git(source, "show", f"{commit}:pyproject.toml").splitlines() if line.startswith("version"))
    except (subprocess.CalledProcessError, StopIteration):
        version = "?"

    files, skipped = plan(source, commit)
    if not files:
        sys.exit(f"no Dials files found in {source} at {commit}; is this a ComfyUI-MAINodes checkout?")
    wanted = {target: (area, src, rewritten(src, git(source, "show", f"{commit}:{src}", text=False))) for area, src, target in files}
    wanted[MODEL_GRAPH_TARGET] = ("generated", MODEL_GRAPH_SOURCE, model_graph(source, commit))

    manifest_path = os.path.join(TARGET, MANIFEST)
    before = json.load(open(manifest_path))["files"] if os.path.isfile(manifest_path) else []
    stale = sorted(p for p in before if p not in wanted and os.path.isfile(os.path.join(TARGET, p)))

    differing = []
    for target, (_, _, data) in sorted(wanted.items()):
        path = os.path.join(TARGET, target)
        if not os.path.isfile(path) or open(path, "rb").read() != data:
            differing.append(target)
    if args.check:
        for target in differing:
            print("differs:", target)
        for target in stale:
            print("gone from MAINodes:", target)
        print(f"{len(wanted)} files checked against {commit[:12]}: {len(differing)} differ, {len(stale)} gone from MAINodes")
        sys.exit(1 if differing or stale else 0)

    for target in differing:
        path = os.path.join(TARGET, target)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(wanted[target][2])
    with open(manifest_path, "w") as f:
        json.dump({"source_commit": commit, "files": sorted(wanted)}, f, indent=1)
        f.write("\n")

    areas = {}
    for target, (area, _, data) in wanted.items():
        n, size = areas.get(area, (0, 0))
        areas[area] = (n + 1, size + len(data))
    changed = [(src, target) for target, (_, src, _) in sorted(wanted.items()) if src != target]
    rack = "web/dial_rack.js"
    note = [
        "# Sync with ComfyUI-MAINodes",
        "",
        "The Dials code in this pack is a copy of the Dials code in ComfyUI-MAINodes. This file is written by",
        "`tools/sync_from_mainodes.py`; run that script to update the copy, and do not edit the copied files here.",
        "",
        f"- source: {remote}",
        f"- commit: `{commit}` ({described})",
        f"- MAINodes version at that commit: {version}",
        f"- synced: {datetime.date.today().isoformat()}",
        f"- command: `python tools/sync_from_mainodes.py <MAINodes checkout>{'' if args.ref == 'HEAD' else ' --ref ' + args.ref}`",
        f"- check without writing: `python tools/sync_from_mainodes.py <MAINodes checkout> --check`",
        "",
        "## What was copied",
        "",
        "| area | files | bytes |",
        "|---|---|---|",
        *[f"| {area} | {n} | {size} |" for area, (n, size) in areas.items()],
        f"| **total** | {len(wanted)} | {sum(len(d) for _, _, d in wanted.values())} |",
        "",
        "The full list is in `tools/sync_manifest.json`.",
        "",
        "## Differences from MAINodes",
        "",
        "Every copied file is byte-identical to MAINodes at the commit above, except:",
        "",
        f"1. `{rack}`: two import lines (sha256 of this pack's copy: `{hashlib.sha256(wanted[rack][2]).hexdigest()}`).",
        "   ```diff",
        *[line for old, new in REWRITES[rack] for line in (f"   -{old}", f"   +{new}")],
        "   ```",
        "   `web/standalone/dial_catalog.js` and `web/standalone/model_catalog.js` are this pack's own files: they give the",
        "   rack the two functions it uses from each Snap module (`load`, `find`; `load`, `entry`) without MAINodes' Snap.",
        f"2. `{MODEL_GRAPH_TARGET}` is generated from MAINodes' `{MODEL_GRAPH_SOURCE}`: the family name patterns and, per",
        "   known model file, its family and kind. The rack uses it to tell which model family is wired in.",
        "3. Paths that differ (contents identical):",
        *[f"   - `{src}` -> `{target}`" for src, target in changed if target != MODEL_GRAPH_TARGET],
        "",
        "## Not copied",
        "",
        "Weight files are never copied by the script. Skipped at this commit:",
        "",
        *([f"- `{p}`" for p in skipped] or ["- (none)"]),
        "",
        "Dials-related files in MAINodes that are left there on purpose:",
        "",
        *[f"- `{what}`: {why}." for what, why in NOT_SHARED],
        "",
        *(["## Gone from MAINodes", "", "Copied by an earlier sync, no longer in MAINodes; still here until someone removes them:", "",
           *[f"- `{p}`" for p in stale], ""] if stale else []),
    ]
    with open(os.path.join(TARGET, NOTE), "w") as f:
        f.write("\n".join(note))
    print(f"copied from {commit[:12]}: {len(wanted)} files ({len(differing)} written, {len(wanted) - len(differing)} already identical), "
          f"{len(skipped)} weight files skipped, {len(stale)} gone from MAINodes")


if __name__ == "__main__":
    main()
