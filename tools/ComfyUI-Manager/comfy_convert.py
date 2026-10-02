"""UI workflow JSON (what ComfyUI saves) -> API prompt JSON (what POST /prompt takes).

Does what the frontend does at queue time, for the node kinds used by the workflows in this repo:
  - widget values become named inputs (names and order come from ComfyUI's /object_info); a choice that brings
    its own inputs (SaveVideo's format, the resize node's resize_type) is followed by them as "<input>.<sub input>"
  - subgraphs are flattened (inner node ids become "<instance id>:<inner id>"); a value typed on the subgraph
    node itself (model, size, prompt...) replaces the value of the inner node it stands for
  - Get/Set nodes (KJNodes) and Reroute are resolved to the real source
  - bypassed nodes are passed through by type, muted nodes are dropped
  - frontend-only nodes (notes, labels, group bypassers) are skipped
"""
import datetime
import random
import re

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"      # a choice that brings its own inputs, named "<input>.<sub input>"
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
            if not (isinstance(typ, list) or typ in WIDGET_TYPES or opts.get("widgetType") in WIDGET_TYPES):
                continue
            names.append(name)
            if opts.get("control_after_generate") or (typ == "INT" and name in ("seed", "noise_seed")):
                names.append(None)
            if opts.get("image_upload") or opts.get("audio_upload"):
                names.append(None)
    return names


def widget_default(info, name):
    """Default the frontend gives a widget that has no saved value."""
    inputs = info.get("input", {})
    spec = (inputs.get("required") or {}).get(name) or (inputs.get("optional") or {}).get(name)
    typ = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in opts:
        return opts["default"]
    choices = typ if isinstance(typ, list) else opts.get("options")
    if choices:
        return choices[0]
    return {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(typ, "")


def _options(spec):
    return spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}


def _has_widget(spec):
    typ, opts = spec[0], _options(spec)
    if opts.get("forceInput") or opts.get("hidden"):
        return False
    # an input that takes several types ("FLOAT,INT", LTXVEmptyLatentAudio's frame_rate) names its widget in widgetType
    return isinstance(typ, list) or typ in WIDGET_TYPES or typ == DYNAMIC_COMBO or opts.get("widgetType") in WIDGET_TYPES


def _spec_default(spec):
    typ, opts = spec[0], _options(spec)
    if "default" in opts:
        return opts["default"]
    choices = typ if isinstance(typ, list) else opts.get("options")
    if choices:
        return choices[0]["key"] if isinstance(choices[0], dict) else choices[0]
    return {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(typ, "")


def named_widgets(info, values):
    """{input name: value} out of a node's widgets_values, in the order the frontend keeps them. Values that are
    missing at the end (the node got widgets since the workflow was saved) take their defaults, as in the frontend."""
    out, rest = {}, list(values)

    def take(name, spec):
        typ, opts = spec[0], _options(spec)
        out[name] = rest.pop(0) if rest else _spec_default(spec)
        if opts.get("control_after_generate") or (typ == "INT" and name.rsplit(".", 1)[-1] in ("seed", "noise_seed")):
            rest[:1] = []      # the 'randomize / fixed' choice next to a seed is not an input
        if opts.get("image_upload") or opts.get("audio_upload"):
            rest[:1] = []
        if typ == DYNAMIC_COMBO:
            chosen = next((o for o in opts.get("options") or [] if isinstance(o, dict) and o.get("key") == out[name]), None)
            for section in ("required", "optional"):
                for sub, subspec in (((chosen or {}).get("inputs") or {}).get(section) or {}).items():
                    if _has_widget(subspec):
                        take(f"{name}.{sub}", subspec)

    inputs, order = info.get("input", {}), info.get("input_order") or {}
    for section in ("required", "optional"):
        specs = inputs.get(section) or {}
        for name in order.get(section) or list(specs):
            if specs.get(name) and _has_widget(specs[name]):
                take(name, specs[name])
    return out


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
            _, index = self._instance_input(scope, slot)
            return None if index is None else self._input_source(scope.parent, scope.instance, index, depth)
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

    def _instance_input(self, scope, slot):
        """(input, its index) on the subgraph node for input `slot` of the subgraph definition. The node lists only
        the inputs that have a socket, so they are matched by name, not by position."""
        declared = scope.definition.get("inputs") or []
        inputs = (scope.instance or {}).get("inputs") or []
        if slot < len(declared):
            for i, x in enumerate(inputs):
                if x.get("name") == declared[slot].get("name"):
                    return x, i
            return None, None
        return (inputs[slot], slot) if slot < len(inputs) else (None, None)

    def _is_widget_input(self, scope, slot):
        """Input `slot` of the subgraph feeds a widget of an inner node (so the subgraph node shows it as a widget)."""
        declared = (scope.definition.get("inputs") or [])[slot]
        for lid in declared.get("linkIds") or []:
            link = scope.links.get(lid)
            target = scope.nodes.get(link[2]) if link is not None else None
            inputs = (target or {}).get("inputs") or []
            if link is not None and link[3] < len(inputs):
                return bool(inputs[link[3]].get("widget"))
        return declared.get("type") in WIDGET_TYPES

    def _promoted(self, scope, slot, depth=0):
        """(True, value) when input `slot` of the subgraph this scope is an instance of has nothing connected and the
        subgraph node carries the value itself: the inputs that feed inner widgets own the entries of the node's
        widgets_values, in the order the subgraph declares them."""
        declared = scope.definition.get("inputs") or [] if scope.definition else []
        if scope.instance is None or slot >= len(declared) or depth > MAX_DEPTH:
            return False, None
        own, _ = self._instance_input(scope, slot)
        if own is not None and own.get("link") is not None:      # connected outside; one level up it may again be such an input
            link = scope.parent.links.get(own["link"])
            if link is not None and link[0] == SUBGRAPH_IN:
                return self._promoted(scope.parent, link[1], depth + 1)
            return False, None
        values = scope.instance.get("widgets_values")
        if isinstance(values, dict):
            name = declared[slot].get("name")
            return (True, values[name]) if name in values else (False, None)
        widgets = [i for i in range(len(declared)) if self._is_widget_input(scope, i)]
        if not isinstance(values, list) or slot not in widgets or widgets.index(slot) >= len(values):
            return False, None
        return True, values[widgets.index(slot)]

    # ---- node -> api entry -------------------------------------------------------------------------
    def _widget_inputs(self, node):
        wv = node.get("widgets_values")
        if isinstance(wv, dict):
            return {k: v for k, v in wv.items() if k != "videopreview"}
        if not wv:
            return {}
        if node["type"] == "Power Lora Loader (rgthree)":
            loras = [v for v in wv if isinstance(v, dict) and "lora" in v]      # as the frontend sends them: no empty fields
            return {f"lora_{i + 1}": {k: x for k, x in v.items() if x is not None} for i, v in enumerate(loras)}
        return named_widgets(self.info[node["type"]], wv)

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
                    continue
                link = scope.links.get(inp["link"])
                if link is not None and link[0] == SUBGRAPH_IN:      # ...unless the subgraph node gives the value
                    found, value = self._promoted(scope, link[1])
                    if found:
                        inputs[inp["name"]] = value
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


_SEGMENT = re.compile(r"(?:구간|part|segment|seg|scene|shot)\s*[_#-]?\s*(\d+)", re.IGNORECASE)
_NEGATIVE = re.compile(r"negative|네거티브|부정", re.IGNORECASE)
VIDEO_OUT = ("VHS_VideoCombine", "SaveVideo", "SaveWEBM", "SaveAnimatedWEBP", "SaveAnimatedPNG")
# inputs that never lead to a prompt, so they are not followed when looking for the text encoders
_NOT_TEXT = {"clip", "vae", "model", "image", "images", "pixels", "latent", "latent_image", "samples", "mask", "clip_vision",
             "clip_vision_output", "start_image", "end_image", "control_net", "sampler", "sigmas", "noise", "audio"}


def _is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def text_encoders(api):
    """(positive, negative): the CLIPTextEncode nodes whose result ends up as a sampler's positive / negative prompt.
    One that feeds both (a negative made by zeroing the positive) counts as positive."""
    found = {"positive": set(), "negative": set()}

    def walk(link, role, seen):
        nid = str(link[0])
        node = api.get(nid)
        if node is None or nid in seen:
            return
        seen.add(nid)
        if node["class_type"] == "CLIPTextEncode":
            found[role].add(nid)
            return
        links = {k: v for k, v in node["inputs"].items() if _is_link(v)}
        if "positive" in links and "negative" in links:      # passes both on: output 0 is the positive one, 1 the negative
            walk(links["negative" if link[1] == 1 else "positive"], role, seen)
            return
        for name, value in links.items():
            if name not in _NOT_TEXT:
                walk(value, role, seen)

    for node in api.values():
        for name, role in (("positive", "positive"), ("negative", "negative"), ("conditioning", "positive")):
            if _is_link(node["inputs"].get(name)) and (name != "conditioning" or node["class_type"].endswith("Guider")):
                walk(node["inputs"][name], role, set())
    title = lambda nid: (api[nid].get("_meta") or {}).get("title") or ""
    for nid, node in api.items():      # what the tracing does not reach (custom samplers) is told by its title
        if node["class_type"] == "CLIPTextEncode" and nid not in found["positive"] | found["negative"]:
            if _NEGATIVE.search(title(nid)):
                found["negative"].add(nid)
            elif ordinal_of(title(nid)) or _SEGMENT.search(title(nid)):
                found["positive"].add(nid)
    positive = {nid for nid in found["positive"] if not (_NEGATIVE.search(title(nid)) and nid in found["negative"])}
    return positive, found["negative"] - positive


def find_slots(api):
    """Where a workflow takes what the manager gives it.
      load:    the LoadImage node of the start image (None when there is none)
      prompts: [[node ids] per segment, in order]; one group when the prompts are not numbered
      sizes:   empty-latent nodes whose width / height are plain numbers
      video:   it saves a video
      saves:   SaveImage nodes
    Segments are told by the title: '1st_...', '2nd_...' or '구간 1 ...', 'part 2 ...'."""
    title = lambda nid: (api[nid].get("_meta") or {}).get("title") or ""
    loads = [nid for nid, n in api.items() if n["class_type"] == "LoadImage"]
    first = [nid for nid in loads if re.search(r"1st|start|first|시작", title(nid), re.IGNORECASE)]
    positive, _ = text_encoders(api)
    numbered = {}
    for nid in positive:
        m = _SEGMENT.search(title(nid))
        number = ordinal_of(title(nid)) or (int(m.group(1)) if m else None)
        if number is not None:
            numbered.setdefault(number, []).append(nid)
    plain = sorted(positive - {nid for ids in numbered.values() for nid in ids})
    prompts = [sorted(numbered[k]) for k in sorted(numbered)] if len(numbered) > 1 else ([sorted(positive)] if positive else [])
    if len(numbered) > 1 and plain:      # an unnumbered one next to numbered ones: a common prompt, left as the workflow has it
        pass
    classes = {n["class_type"] for n in api.values()}
    return {
        "load": (first or loads or [None])[0], "loads": loads, "prompts": prompts,
        "sizes": [nid for nid, n in api.items() if re.match(r"Empty.*Latent", n["class_type"])
                  and all(isinstance(n["inputs"].get(k), int) for k in ("width", "height"))],
        "video": bool(classes & set(VIDEO_OUT)),
        "saves": [nid for nid, n in api.items() if n["class_type"] == "SaveImage"],
    }


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
