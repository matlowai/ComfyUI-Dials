"""ComfyUI-Dials: training-free steering dials for MiniMax H3, MOSS-Image and Krea 2.

Two sets of nodes:
- the five original nodes (DialsMakeDial, DialsApplyDial, DialsLoadDial, DialsCombineDials, DialsSaveDial) in nodes.py,
  kept so workflows made with them still open;
- the Dial Rack set (MAIDialRack, MAIDialHoldMap, MAIDialMake, MAIDialMakeFromPictures, MAIDialMakeFromVideos,
  MAIDialCombine) in dials/, with the /mainodes/dials routes and the rack UI in web/. This code is a copy of the Dials
  code in ComfyUI-MAINodes (see SYNC.md), under the same node ids, so a workflow works with either pack.

Load-once rule: when ComfyUI-MAINodes is installed too, MAINodes provides the Dial Rack set and this pack loads only the
five original nodes: it does not import dials/nodes.py (which is what installs the routes) and does not set
WEB_DIRECTORY (so the rack UI is not served from here). See ownership.py.

ComfyUI's loader takes either NODE_CLASS_MAPPINGS or comfy_entrypoint, so the V3 node classes are put into the
mappings here, keyed by their schema's node_id.
"""
import logging
import os

from . import ownership
from .nodes import ApplyDial, CombineDials, LoadDial, MakeDial, SaveDial

DIALS_STANDALONE_PACK = True  # ownership.py looks for this name to tell a copy of this pack from ComfyUI-MAINodes

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _cls in (MakeDial, ApplyDial, LoadDial, CombineDials, SaveDial):
    _schema = _cls.GET_SCHEMA()
    NODE_CLASS_MAPPINGS[_schema.node_id] = _cls
    if _schema.display_name is not None:
        NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = _schema.display_name

MAINODES = ownership.find_mainodes(os.path.dirname(os.path.abspath(__file__)))
RACK_PROVIDER = "ComfyUI-MAINodes" if MAINODES else None
if MAINODES:
    logging.info("[ComfyUI-Dials] ComfyUI-MAINodes is installed (%s): it provides the Dial Rack nodes (MAIDial*), the "
                 "/mainodes/dials routes and the rack UI. This pack loads only its five original Dials* nodes.", MAINODES)
else:
    try:
        from .dials import nodes as _rack  # importing it installs the /mainodes/dials routes
    except Exception as _e:  # the original nodes must still load on a ComfyUI the rack code does not support
        logging.warning("[ComfyUI-Dials] Dial Rack nodes not loaded: %s: %s. The five original Dials* nodes are loaded.",
                        type(_e).__name__, _e)
    else:
        NODE_CLASS_MAPPINGS.update(_rack.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_rack.NODE_DISPLAY_NAME_MAPPINGS)
        WEB_DIRECTORY = "./web"  # set only here, so the rack UI is served only when this pack provides the rack
        RACK_PROVIDER = "ComfyUI-Dials"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
