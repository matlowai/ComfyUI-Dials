// Stand-in for ComfyUI-MAINodes' web/snap/dials.js, for the Dial Rack in this pack (web/dial_rack.js).
// The rack uses two functions of that module: load(force) and find(dial). They are written here without MAINodes' Snap:
// the catalog is every installed dial with its family, mode and info, from the pack's own route /mainodes/dials/catalog.
// Snap's "SFW only" filter (which hides dials a person's model graph tags) is a Snap feature and is not here.
import { api } from "../../../scripts/api.js";

let catalog = null, loading = null;
export function load(force) {
  if (force) { catalog = null; loading = null; }
  loading ??= api.fetchApi("/mainodes/dials/catalog").then((r) => (r.ok ? r.json() : { dials: [] })).then((c) => (catalog = c.dials ?? []), () => (catalog = []));
  return loading;
}
export const find = (dial) => (catalog ?? []).find((d) => d.dial === dial);
