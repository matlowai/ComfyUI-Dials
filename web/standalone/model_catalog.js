// Stand-in for ComfyUI-MAINodes' web/snap/models.js, for the Dial Rack in this pack (web/dial_rack.js).
// The rack uses two functions of that module: load() and entry(name). entry() answers, for a model file named in a
// loader widget, which family it belongs to (MiniMax H3, MOSS-Image, Krea 2) and what kind of file it is, so the rack
// can offer only the dials made for the wired model. Same rules as MAINodes, without its Snap:
//   - a file the model table knows (model_graph.js, generated from MAINodes' snap_cards/models.json) is what the table says;
//   - any other file: family from its name (the table's patterns), kind from the folder it is in (LoRA, encoder and VAE
//     folders are not diffusion models; a name with turbo/dmd/dmad/lightning/hyper is a turbo merge; the rest is "base").
// Not here: a person's own model graph (Snap's userdata snap/models.json), which in MAINodes overrides the table.
import { api } from "../../../scripts/api.js";
import { GRAPH } from "./model_graph.js";

const base = (f) => String(f ?? "").replace(/\\/g, "/").split("/").pop();
const families = GRAPH.families.map((f) => [new RegExp(f.names), f.name]);
const known = new Map(GRAPH.files.map((e) => [base(e.file), e]));

// name -> folder, from ComfyUI's own model folder lists (the first folder wins a name that is in two); read once per page
let folders = new Map(), loading = null;
export function load() {
  loading ??= api.fetchApi("/experiment/models").then((r) => r.json())
    .then((list) => Promise.all(list.map(async ({ name }) => [name, await (await api.fetchApi(`/experiment/models/${encodeURIComponent(name)}`)).json()])))
    .then((lists) => {
      folders = new Map();
      for (const [folder, files] of lists) for (const f of files) if (!folders.has(f.name)) folders.set(f.name, folder);
      return folders;
    }).catch(() => folders); // no folder lists: every unknown file is judged by its name alone
  return loading;
}

function infer(name) {
  const n = base(name).toLowerCase(), folder = folders.get(name) ?? null;
  const kind = { loras: "lora", text_encoders: "encoder", clip: "encoder", vae: "vae" }[folder] ?? (/turbo|dmd|dmad|lightning|hyper/.test(n) ? "turbo_merge" : "base");
  return { file: base(name), folder, family: families.find(([re]) => re.test(n))?.[1] ?? null, kind };
}
export function entry(name) {
  if (!name) return null;
  // a table entry whose file names its folder too is that file only, never another file with the same base name
  const hit = known.get(base(name)), e = hit && (!String(hit.file).includes("/") || String(name).replace(/\\/g, "/").endsWith(hit.file)) ? hit : null;
  return e ? { ...infer(name), ...e } : { ...infer(name), inferred: true };
}
