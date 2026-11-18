"""Which pack provides the Dial Rack nodes when ComfyUI-MAINodes is installed too.

ComfyUI-MAINodes carries the same Dials code as this pack (same node ids, same /mainodes/dials routes, same rack UI).
Two copies of the same node ids and routes in one ComfyUI would overwrite each other, so one rule decides:
if a MAINodes pack that ComfyUI will load is present, MAINodes provides them and this pack loads only its five original
nodes. The rule does not depend on load order (ComfyUI loads packs in directory-listing order).

MAINodes is recognised by what it contains, not by its folder name: a pack folder with an __init__.py and a
dials/nodes.py that declares the MAIDialRack node, which is not a copy of this pack (this pack's __init__.py carries
the name DIALS_STANDALONE_PACK). A folder ComfyUI skips (name ending in .disabled, left out by
--disable-all-custom-nodes / --whitelist-custom-nodes, or disabled by ComfyUI-Manager's policy) does not count.
"""
import os

MARK = "DIALS_STANDALONE_PACK"
RACK_NODE = 'node_id="MAIDialRack"'


def _contains(path, text):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return text in f.read()
    except OSError:
        return False


def has_mainodes_dials(pack_dir):
    """True for a pack folder holding MAINodes' dials package (and not a copy of this pack)."""
    init = os.path.join(pack_dir, "__init__.py")
    return os.path.isfile(init) and _contains(os.path.join(pack_dir, "dials", "nodes.py"), RACK_NODE) and not _contains(init, MARK)


def skipped_by_comfyui(root, name):
    """The checks ComfyUI's loader (nodes.init_external_custom_nodes) makes before importing a pack folder."""
    if name.endswith(".disabled"):
        return True
    try:
        from comfy.cli_args import args
    except Exception:
        return False
    if getattr(args, "disable_all_custom_nodes", False) and name not in (getattr(args, "whitelist_custom_nodes", None) or []):
        return True
    if getattr(args, "enable_manager", False):
        try:
            import comfyui_manager
            return bool(comfyui_manager.should_be_disabled(os.path.join(root, name)))
        except Exception:
            return False
    return False


def find_mainodes(own_dir, roots=None):
    """-> the folder of a MAINodes pack ComfyUI will load that has the dials package, or None."""
    if roots is None:
        import folder_paths
        roots = folder_paths.get_folder_paths("custom_nodes")
    own = os.path.realpath(own_dir)
    for root in roots:
        try:
            names = sorted(os.listdir(os.path.realpath(root)))
        except OSError:
            continue
        for name in names:
            path = os.path.join(root, name)
            if not os.path.isdir(path) or os.path.realpath(path) == own or skipped_by_comfyui(root, name):
                continue
            if has_mainodes_dials(path):
                return path
    return None
