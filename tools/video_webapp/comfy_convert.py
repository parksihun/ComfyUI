"""UI workflow JSON (what ComfyUI saves) -> API prompt JSON (what POST /prompt takes).

Does what the frontend does at queue time, for the node kinds used by the workflows in this repo:
  - widget values become named inputs (names and order come from ComfyUI's /object_info)
  - subgraphs are flattened (inner node ids become "<instance id>:<inner id>")
  - Get/Set nodes (KJNodes) and Reroute are resolved to the real source
  - bypassed nodes are passed through by type, muted nodes are dropped
  - frontend-only nodes (notes, labels, group bypassers) are skipped
"""
import datetime
import random
import re

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
SUBGRAPH_IN, SUBGRAPH_OUT = -10, -20
MODE_MUTED, MODE_BYPASS = 2, 4
MAX_DEPTH = 200
FRONTEND_ONLY = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute", "GetNode", "SetNode",
                 "Label (rgthree)", "Fast Groups Bypasser (rgthree)", "Fast Groups Muter (rgthree)",
                 "Fast Bypasser (rgthree)", "Fast Muter (rgthree)", "Bookmark (rgthree)"}


def widget_names(info):
    """Widget names in the order the frontend stores widgets_values; None = a value that is not an input
    (the control_after_generate / upload button slot)."""
    names = []
    inputs = info.get("input", {})
    order = info.get("input_order") or {}
    for section in ("required", "optional"):
        specs = inputs.get(section) or {}
        for name in order.get(section) or list(specs):
            spec = specs.get(name)
            if not spec:
                continue
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            if not (isinstance(typ, list) or typ in WIDGET_TYPES):
                continue
            names.append(name)
            if opts.get("control_after_generate") or (typ == "INT" and name in ("seed", "noise_seed")):
                names.append(None)
            if opts.get("image_upload") or opts.get("audio_upload"):
                names.append(None)
    return names


class _Scope:
    """One graph level: the root workflow or one subgraph instance."""

    def __init__(self, nodes, links, prefix="", parent=None, instance=None, definition=None):
        self.nodes = {n["id"]: n for n in nodes}
        self.links = links
        self.prefix = prefix
        self.parent = parent          # scope holding the subgraph instance node
        self.instance = instance      # that instance node
        self.definition = definition
        self.setters = {}
        for n in nodes:
            if n.get("type") == "SetNode" and n.get("widgets_values"):
                self.setters[n["widgets_values"][0]] = n


def _norm_links(links):
    out = {}
    for l in links or []:
        if isinstance(l, dict):
            out[l["id"]] = (l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l.get("type"))
        else:
            out[l[0]] = (l[1], l[2], l[3], l[4], l[5] if len(l) > 5 else None)
    return out


class Converter:
    def __init__(self, workflow, object_info):
        self.wf = workflow
        self.info = object_info
        self.subgraphs = {s["id"]: s for s in (workflow.get("definitions") or {}).get("subgraphs", [])}
        self.root = _Scope(workflow["nodes"], _norm_links(workflow.get("links")))
        self.children = {}
        self.warnings = []

    # ---- scopes ------------------------------------------------------------------------------------
    def _child(self, scope, node):
        key = scope.prefix + str(node["id"])
        if key not in self.children:
            d = self.subgraphs[node["type"]]
            self.children[key] = _Scope(d["nodes"], _norm_links(d.get("links")), key + ":", scope, node, d)
        return self.children[key]

    # ---- link resolution ---------------------------------------------------------------------------
    def _input_source(self, scope, node, index, depth):
        inputs = node.get("inputs") or []
        if index >= len(inputs) or inputs[index].get("link") is None:
            return None
        link = scope.links.get(inputs[index]["link"])
        if link is None:
            return None
        return self._resolve(scope, link[0], link[1], depth + 1)

    def _resolve(self, scope, origin_id, slot, depth=0):
        """(node key, output slot) of the real node feeding this output, or None when nothing does."""
        if depth > MAX_DEPTH:
            raise ValueError("workflow link chain is too deep (a Get/Set or reroute loop?)")
        if origin_id == SUBGRAPH_IN:
            if scope.parent is None:
                return None
            return self._input_source(scope.parent, scope.instance, slot, depth)
        node = scope.nodes.get(origin_id)
        if node is None or node.get("mode") == MODE_MUTED:
            return None
        ntype = node.get("type")
        if node.get("mode") == MODE_BYPASS:
            return self._passthrough(scope, node, slot, depth)
        if ntype in self.subgraphs:
            child = self._child(scope, node)
            outs = child.definition.get("outputs") or []
            if slot >= len(outs):
                return None
            for lid in outs[slot].get("linkIds") or []:
                link = child.links.get(lid)
                if link is not None:
                    return self._resolve(child, link[0], link[1], depth + 1)
            return None
        if ntype == "GetNode":
            name = (node.get("widgets_values") or [None])[0]
            s = scope
            while s is not None:
                setter = s.setters.get(name)
                if setter is not None and setter.get("mode") not in (MODE_MUTED, MODE_BYPASS):
                    return self._input_source(s, setter, 0, depth)
                s = s.parent
            self.warnings.append(f"Get node '{name}' has no matching Set node")
            return None
        if ntype in ("SetNode", "Reroute"):
            return self._input_source(scope, node, 0, depth)
        if ntype not in self.info:
            return None
        return [scope.prefix + str(origin_id), slot]

    def _passthrough(self, scope, node, slot, depth):
        outputs = node.get("outputs") or []
        typ = outputs[slot].get("type") if slot < len(outputs) else None
        inputs = node.get("inputs") or []
        candidates = [slot] + list(range(len(inputs)))
        for i in candidates:
            if i < len(inputs) and inputs[i].get("link") is not None and inputs[i].get("type") == typ:
                return self._input_source(scope, node, i, depth)
        return None

    # ---- node -> api entry -------------------------------------------------------------------------
    def _widget_inputs(self, node):
        wv = node.get("widgets_values")
        if isinstance(wv, dict):
            return {k: v for k, v in wv.items() if k != "videopreview"}
        if not wv:
            return {}
        if node["type"] == "Power Lora Loader (rgthree)":
            loras = [v for v in wv if isinstance(v, dict) and "lora" in v]
            return {f"lora_{i + 1}": v for i, v in enumerate(loras)}
        out = {}
        for name, value in zip(widget_names(self.info[node["type"]]), wv):
            if name is not None:
                out[name] = value
        return out

    def _emit(self, scope, api):
        for nid, node in scope.nodes.items():
            if node.get("mode") in (MODE_MUTED, MODE_BYPASS):
                continue
            ntype = node.get("type")
            if ntype in self.subgraphs:
                self._emit(self._child(scope, node), api)
                continue
            if ntype not in self.info:
                if ntype not in FRONTEND_ONLY:
                    self.warnings.append(f"node {scope.prefix}{nid}: unknown type '{ntype}' skipped")
                continue
            inputs = self._widget_inputs(node)
            for i, inp in enumerate(node.get("inputs") or []):
                if inp.get("link") is None:
                    continue
                src = self._input_source(scope, node, i, 0)
                if src is not None:          # an unresolved link on a widget input keeps the widget value
                    inputs[inp["name"]] = src
            title = node.get("title") or self.info[ntype].get("display_name") or ntype
            if scope.instance is not None:      # "2nd_Section > KSampler": tells the sections' inner nodes apart
                title = f"{scope.instance.get('title') or scope.definition.get('name', '')} > {title}"
            api[scope.prefix + str(nid)] = {"inputs": inputs, "class_type": ntype, "_meta": {"title": title}}

    def convert(self):
        api = {}
        self._emit(self.root, api)
        return api


def workflow_to_api(workflow, object_info):
    """Returns (api_prompt, warnings)."""
    conv = Converter(workflow, object_info)
    return conv.convert(), conv.warnings


def is_api_format(data):
    return isinstance(data, dict) and "nodes" not in data and all(
        isinstance(v, dict) and "class_type" in v for v in data.values())


_ORDINAL = re.compile(r"^\s*(\d+)\s*(?:st|nd|rd|th)(?![a-z])", re.IGNORECASE)


def ordinal_of(title):
    """'3rd_CLIP Text Encode (Prompt)' -> 3, anything else -> None."""
    m = _ORDINAL.match(title or "")
    return int(m.group(1)) if m else None


def randomize_seeds(api):
    """New random value for every literal seed (what 'randomize' does in the UI). Samplers that add no noise
    keep theirs."""
    for node in api.values():
        inputs = node["inputs"]
        if inputs.get("add_noise") == "disable":
            continue
        for name in ("seed", "noise_seed"):
            if isinstance(inputs.get(name), int) and not isinstance(inputs[name], bool):
                inputs[name] = random.randint(1, 2 ** 48)


_DATE_TOKEN = re.compile(r"%date:([^%]+)%")
_DATE_PARTS = re.compile(r"yyyy|yy|MM|M|dd|d|hh|h|mm|m|ss|s")


def _format_date(fmt, now):
    values = {"yyyy": f"{now.year:04d}", "yy": f"{now.year % 100:02d}", "MM": f"{now.month:02d}", "M": str(now.month),
              "dd": f"{now.day:02d}", "d": str(now.day), "hh": f"{now.hour:02d}", "h": str(now.hour),
              "mm": f"{now.minute:02d}", "m": str(now.minute), "ss": f"{now.second:02d}", "s": str(now.second)}
    return _DATE_PARTS.sub(lambda m: values[m.group(0)], fmt)


def apply_text_replacements(api):
    """The frontend expands %date:yyyyMMdd% in filename prefixes before queueing; the server does not know it."""
    now = datetime.datetime.now()
    for node in api.values():
        value = node["inputs"].get("filename_prefix")
        if isinstance(value, str) and "%date:" in value:
            node["inputs"]["filename_prefix"] = _DATE_TOKEN.sub(lambda m: _format_date(m.group(1), now), value)
