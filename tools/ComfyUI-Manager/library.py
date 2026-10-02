"""Library: what each generated image / video was made with.

ComfyUI embeds the queued graph in its outputs (PNG text chunks, WebP/JPEG EXIF, MP4/WebM/MKV container
tags). This module reads that back and turns it into: which workflow, which prompts, which models. It also
keeps an index of the output folder so the list can be sorted and filtered without re-reading every file.
"""
import hashlib
import io
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict

import av
from PIL import Image

import comfy_convert

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXT = (".mp4", ".webm", ".mkv", ".mov")
MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".sft", ".pt", ".pth")
SIMILAR_ENOUGH = 0.75
NO_METADATA = "메타데이터 없음"


# ---------------------------------------------------------------------------------------------- #
# reading the embedded metadata
# ---------------------------------------------------------------------------------------------- #
def _loads(value):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def read_metadata(path):
    """-> (prompt, workflow, media) : API graph, UI graph (either may be None) and width/height/duration."""
    ext = os.path.splitext(path)[1].lower()
    raw, media = {}, {"width": None, "height": None, "duration": None, "fps": None}
    if ext in IMAGE_EXT:
        with Image.open(path) as im:
            media.update(width=im.width, height=im.height)
            raw.update({k.lower(): v for k, v in im.info.items() if isinstance(v, str)})
            for value in im.getexif().values():          # WebP / JPEG: "prompt:{...}", "workflow:{...}"
                if isinstance(value, str) and ":" in value:
                    key, _, body = value.partition(":")
                    if key.lower() in ("prompt", "workflow"):
                        raw.setdefault(key.lower(), body)
    elif ext in VIDEO_EXT:
        with av.open(path) as container:
            raw.update({k.lower(): v for k, v in container.metadata.items()})
            if container.streams.video:
                stream = container.streams.video[0]
                media.update(width=stream.codec_context.width, height=stream.codec_context.height)
                if stream.average_rate:
                    media["fps"] = round(float(stream.average_rate), 2)
            if container.duration:
                media["duration"] = round(container.duration / av.time_base, 2)
        comment = _loads(raw.get("comment"))             # older VideoHelperSuite: everything in one comment tag
        if isinstance(comment, dict):
            for key in ("prompt", "workflow"):
                if key in comment:
                    raw.setdefault(key, comment[key])
    prompt, workflow = _loads(raw.get("prompt")), _loads(raw.get("workflow"))
    if not (isinstance(prompt, dict) and comfy_convert.is_api_format(prompt) and prompt):
        prompt = None
    if not (isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list)):
        workflow = None
    return prompt, workflow, media


# ---------------------------------------------------------------------------------------------- #
# which workflow
# ---------------------------------------------------------------------------------------------- #
def api_types(prompt):
    return Counter(n["class_type"] for n in prompt.values())


def workflow_types(workflow):
    """Node types that would actually be queued: bypassed / muted / frontend-only nodes left out,
    subgraph instances replaced by their inner nodes."""
    subgraphs = {s["id"]: s for s in (workflow.get("definitions") or {}).get("subgraphs", [])}
    types = Counter()

    def walk(nodes, depth=0):
        for n in nodes:
            if n.get("mode") in (comfy_convert.MODE_MUTED, comfy_convert.MODE_BYPASS):
                continue
            ntype = n.get("type")
            if ntype in subgraphs:
                if depth < 8:
                    walk(subgraphs[ntype].get("nodes", []), depth + 1)
            elif ntype not in comfy_convert.FRONTEND_ONLY:
                types[ntype] += 1

    walk(workflow.get("nodes", []))
    return types


def similarity(a, b):
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 0.0


class Catalogue:
    """The workflows we know by name: workflow/*.json (UI and API form) and this tool's own templates."""

    def __init__(self, workflow_dir, template_dir):
        self.dirs = [(workflow_dir, ""), (template_dir, "Manager: ")]
        self.stamp, self.entries = None, []

    def load(self):
        files = []      # (path, name shown): subfolders of workflow/ are included, shown as "sub/name"
        for base, prefix in self.dirs:
            for folder, dirs, names in os.walk(base):
                dirs.sort()
                for f in sorted(names):
                    if f.lower().endswith(".json"):
                        rel = os.path.relpath(os.path.join(folder, f), base).replace("\\", "/")
                        files.append((os.path.join(folder, f), prefix + re.sub(r"\.api$", "", os.path.splitext(rel)[0])))
        stamp = tuple((p, os.path.getmtime(p)) for p, _ in files)
        if stamp == self.stamp:
            return self.entries
        entries = []
        for path, name in files:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            if comfy_convert.is_api_format(data) and data:
                entries.append({"name": name, "id": None, "types": api_types(data), "created": os.path.getctime(path)})
            elif isinstance(data, dict) and isinstance(data.get("nodes"), list):
                entries.append({"name": name, "id": data.get("id"), "types": workflow_types(data), "created": os.path.getctime(path)})
        self.stamp, self.entries = stamp, entries
        return entries

    def match(self, types, workflow_id):
        """-> (name, how) with how in 'id' | 'similar' | None."""
        entries = self.load()
        scored = sorted(((similarity(types, e["types"]), e) for e in entries), key=lambda x: (-x[0], x[1]["created"], len(x[1]["name"])))      # between copies, the file that was there first
        same_id = [(s, e) for s, e in scored if workflow_id and e["id"] == workflow_id]
        if same_id:
            return same_id[0][1]["name"], "id"
        if scored and scored[0][0] >= SIMILAR_ENOUGH:
            return scored[0][1]["name"], "similar"
        return None, None


# ---------------------------------------------------------------------------------------------- #
# prompts, models, settings out of an API graph
# ---------------------------------------------------------------------------------------------- #
def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


PASS_THROUGH = {"ShowText|pysssss": "text", "easy showAnything": "anything", "PreviewAny": "source",
                "ShortsFreeVRAM": "value", "LayerUtility: PurgeVRAM": "anything", "VRAMCleanup": "input"}
JOINERS = {"StringConcatenate": ("string_a", "string_b", "delimiter"), "JoinStrings": ("string1", "string2", "delimiter")}
TEXT_HOLDER_KEYS = ("value", "text", "string", "prompt")
ENCODER_TEXT_KEYS = ("text", "prompt", "positive_prompt", "negative_prompt", "clip_l", "t5xxl", "text_g", "text_l")


class GraphReader:
    def __init__(self, prompt):
        self.p = prompt
        self.consumers = defaultdict(list)
        for nid, node in prompt.items():
            for name, value in node.get("inputs", {}).items():
                if is_link(value):
                    self.consumers[str(value[0])].append((nid, name))
        self.generator = None     # set by text(): the node that wrote the text while the graph ran

    def title(self, nid):
        node = self.p.get(str(nid), {})
        return (node.get("_meta") or {}).get("title") or node.get("class_type", "")

    def text(self, value, depth=0):
        """The string behind an input, or None when it only existed while the graph ran."""
        if isinstance(value, str):
            return value
        if not is_link(value) or depth > 12:
            return None
        nid = str(value[0])
        node = self.p.get(nid)
        if node is None:
            return None
        ctype, inputs = node["class_type"], node.get("inputs", {})
        if ctype in PASS_THROUGH:
            return self.text(inputs.get(PASS_THROUGH[ctype]), depth + 1)
        if ctype in JOINERS:
            a, b, delimiter = JOINERS[ctype]
            parts = [self.text(inputs.get(a, ""), depth + 1), self.text(inputs.get(b, ""), depth + 1)]
            if None in parts:
                return None
            joiner = inputs.get(delimiter) if isinstance(inputs.get(delimiter), str) else ""
            return joiner.join(part for part in parts if part)
        if ctype == "ShortsPromptsFanout":
            text = self._fanout_text(inputs, value[1])
            if text is None:
                self.generator = {"node": nid, "title": self.title(nid), "class": ctype, "instruction": "",
                                  "source": inputs.get("prompts_json") or inputs.get("prompts_file") or "prompts.json"}
            return text
        if not any(is_link(v) for v in inputs.values()):      # a plain text box
            for key in TEXT_HOLDER_KEYS:
                if isinstance(inputs.get(key), str):
                    return inputs[key]
        instruction = next((inputs[k] for k in ("prompt", "text", "custom_prompt") if isinstance(inputs.get(k), str) and inputs[k]), "")
        self.generator = {"node": nid, "title": self.title(nid), "class": ctype, "instruction": instruction}
        return None

    @staticmethod
    def _fanout_text(inputs, slot):
        """Shorts Prompts Fanout reads prompts.json when the graph runs; the file may still be there."""
        path = inputs.get("prompts_json") or ""
        if slot > 7 or not isinstance(path, str) or not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                segments = json.load(f).get("segments", [])
        except (OSError, ValueError):
            return None
        index = int(inputs.get("first_segment") or 1) + slot
        for i, sg in enumerate(segments):
            if int(sg.get("index", i + 1)) == index:
                return sg.get("positive_prompt", "")
        return ""

    def role(self, nid):
        """'positive' / 'negative' by following the conditioning to the first input with such a name."""
        seen, frontier = {nid}, [nid]
        for _ in range(8):
            nxt = []
            for cur in frontier:
                for target, name in self.consumers.get(cur, []):
                    low = name.lower()
                    if "negative" in low:
                        return "negative"
                    if "positive" in low:
                        return "positive"
                    if target not in seen:
                        seen.add(target)
                        nxt.append(target)
            frontier = nxt
        return "negative" if "neg" in self.title(nid).lower() else "positive"

    def prompts(self):
        found = []
        for nid, node in self.p.items():
            if "textencode" not in node["class_type"].lower():
                continue
            for key in ENCODER_TEXT_KEYS:
                if key not in node.get("inputs", {}):
                    continue
                self.generator = None
                text = self.text(node["inputs"][key])
                role = "negative" if "negative" in key else "positive" if "positive" in key else self.role(nid)
                if text is None:
                    found.append({"node": nid, "title": self.title(nid), "role": role, "text": None, "generated_by": self.generator})
                elif text.strip():
                    found.append({"node": nid, "title": self.title(nid), "role": role, "text": text.strip()})

        def order(item):
            n = comfy_convert.ordinal_of(item["title"])
            head = re.match(r"\d+", item["node"])
            return (n if n is not None else 10 ** 6, int(head.group()) if head else 10 ** 9, item["node"])

        found.sort(key=order)
        unique, seen = [], set()
        for item in found:
            key = (item["role"], item["text"])
            if item["text"] is None or key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def number(self, value, depth=0):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        if is_link(value) and depth < 6:
            inputs = self.p.get(str(value[0]), {}).get("inputs", {})
            for key in ("value", "seed", "int", "number"):
                if key in inputs:
                    return self.number(inputs[key], depth + 1)
        return None

    def models(self):
        models, loras = [], []
        for node in self.p.values():
            for key, value in node.get("inputs", {}).items():
                if isinstance(value, dict) and isinstance(value.get("lora"), str):      # Power Lora Loader rows
                    if value.get("on") and value["lora"] != "None":
                        loras.append({"name": value["lora"], "strength": value.get("strength")})
                elif isinstance(value, str) and value.lower().endswith(MODEL_EXT):
                    if "lora" in key.lower():
                        loras.append({"name": value, "strength": self.number(node["inputs"].get("strength_model"))})
                    else:
                        models.append({"kind": key.replace("_name", ""), "name": value})

        def dedupe(items):
            out, seen = [], set()
            for it in items:
                key = json.dumps(it, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    out.append(it)
            return out

        return dedupe(models), dedupe(loras)

    def settings(self):
        out = {"seeds": [], "samplers": [], "inputs": []}
        for node in self.p.values():
            inputs, ctype = node.get("inputs", {}), node["class_type"]
            if "sampler" in ctype.lower():
                if inputs.get("add_noise") != "disable":
                    seed = self.number(inputs.get("seed", inputs.get("noise_seed")))
                    if seed is not None and seed not in out["seeds"]:
                        out["seeds"].append(int(seed))
                steps, cfg = self.number(inputs.get("steps")), self.number(inputs.get("cfg"))
                if steps is not None:
                    line = f"{int(steps)} steps, cfg {cfg:g}" if cfg is not None else f"{int(steps)} steps"
                    for key in ("sampler_name", "scheduler"):
                        if isinstance(inputs.get(key), str):
                            line += ", " + inputs[key]
                    if line not in out["samplers"]:
                        out["samplers"].append(line)
            if ctype in ("LoadImage", "VHS_LoadVideo", "LoadVideo"):
                name = inputs.get("image") or inputs.get("video") or inputs.get("file")
                if isinstance(name, str) and name not in out["inputs"]:
                    out["inputs"].append(name)
        return out


def workflow_prompts(workflow):
    """Fallback when only the UI graph is embedded: text boxes of the prompt nodes, by title."""
    out = []
    for n in workflow.get("nodes", []):
        if "textencode" in str(n.get("type", "")).lower() and n.get("mode") not in (2, 4):
            values = n.get("widgets_values")
            text = values[0] if isinstance(values, list) and values and isinstance(values[0], str) else ""
            if text.strip():
                title = n.get("title") or n["type"]
                out.append({"node": str(n["id"]), "title": title, "text": text.strip(),
                            "role": "negative" if "neg" in title.lower() else "positive"})
    return out


def describe(path, catalogue):
    """Everything the UI shows for one file."""
    prompt, workflow, media = read_metadata(path)
    info = {"media": media, "has_prompt": prompt is not None, "has_workflow": workflow is not None,
            "workflow": NO_METADATA, "match": None, "workflow_id": None, "types": {},
            "prompts": [], "models": [], "loras": [], "settings": {}, "node_count": 0}
    if prompt is None and workflow is None:
        return info
    types = api_types(prompt) if prompt is not None else workflow_types(workflow)
    info["types"] = dict(types)
    info["node_count"] = sum(types.values())
    info["workflow_id"] = workflow.get("id") if workflow else None
    if prompt is not None:
        reader = GraphReader(prompt)
        info["prompts"] = reader.prompts()
        info["models"], info["loras"] = reader.models()
        info["settings"] = reader.settings()
    else:
        info["prompts"] = workflow_prompts(workflow)
    info["workflow"], info["match"] = name_workflow(types, info["workflow_id"], info["models"], catalogue)
    return info


def name_workflow(types, workflow_id, models, catalogue):
    if not types:
        return NO_METADATA, None
    name, how = catalogue.match(Counter(types), workflow_id)
    if name:
        return name, how
    main = next((m["name"] for m in models if m["kind"] in ("unet", "ckpt", "gguf", "model")), None)
    if main:
        return "미등록 (" + os.path.splitext(os.path.basename(main.replace("\\", "/")))[0] + ")", None
    return "미등록 워크플로우", None


# ---------------------------------------------------------------------------------------------- #
# index of the output folder
# ---------------------------------------------------------------------------------------------- #
class Library:
    def __init__(self, output_dir, data_dir, catalogue):
        self.roots = {"output": output_dir, "dropped": os.path.join(data_dir, "dropped")}
        self.data_dir = data_dir
        self.catalogue = catalogue
        self.index_path = os.path.join(data_dir, "library_index.json")
        self.lock = threading.Lock()
        self.items = {}
        os.makedirs(self.roots["dropped"], exist_ok=True)
        os.makedirs(os.path.join(data_dir, "thumbs"), exist_ok=True)
        try:
            with open(self.index_path, encoding="utf-8") as f:
                self.items = json.load(f)
        except (OSError, ValueError):
            pass

    def resolve(self, ref):
        """'output:sub/file.png' -> absolute path inside that root (nothing outside it)."""
        root, _, rel = (ref or "").partition(":")
        base = self.roots.get(root)
        if not base or not rel:
            raise ValueError("잘못된 파일 경로입니다")
        full = os.path.realpath(os.path.join(base, rel))
        if os.path.commonpath([full, os.path.realpath(base)]) != os.path.realpath(base) or not os.path.isfile(full):
            raise ValueError("파일을 찾을 수 없습니다: " + rel)
        return full

    def _entry(self, rel, full, stat):
        info = describe(full, self.catalogue)
        positives = [p for p in info["prompts"] if p["role"] == "positive"]
        first = next((p["text"] for p in positives if p["text"]), "")
        return {"mtime": stat.st_mtime, "size": stat.st_size,
                "kind": "video" if rel.lower().endswith(VIDEO_EXT) else "image",
                "width": info["media"]["width"], "height": info["media"]["height"], "duration": info["media"]["duration"],
                "has_meta": info["has_prompt"] or info["has_workflow"], "workflow_id": info["workflow_id"],
                "types": info["types"], "models": info["models"], "prompt_count": len(positives),
                "generated": any(p["text"] is None for p in positives), "snippet": first[:200],
                "search": " ".join(p["text"] for p in info["prompts"] if p["text"]).lower()[:4000]}

    def scan(self):
        """Bring the index in line with the output folder; only new or changed files are read."""
        base = self.roots["output"]
        seen, changed = set(), False
        for folder, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if not name.lower().endswith(IMAGE_EXT + VIDEO_EXT):
                    continue
                full = os.path.join(folder, name)
                rel = os.path.relpath(full, base).replace("\\", "/")
                seen.add(rel)
                try:
                    stat = os.stat(full)
                except OSError:
                    continue
                old = self.items.get(rel)
                if old and old["mtime"] == stat.st_mtime and old["size"] == stat.st_size:
                    continue
                try:
                    entry = self._entry(rel, full, stat)
                except Exception as e:      # an unreadable or half-written file must not stop the scan
                    entry = {"mtime": stat.st_mtime, "size": stat.st_size,
                             "kind": "video" if rel.lower().endswith(VIDEO_EXT) else "image",
                             "width": None, "height": None, "duration": None, "has_meta": False, "workflow_id": None,
                             "types": {}, "models": [], "prompt_count": 0, "generated": False, "snippet": "",
                             "search": "", "error": str(e)[:200]}
                with self.lock:
                    self.items[rel] = entry
                changed = True
        with self.lock:
            for rel in [r for r in self.items if r not in seen]:
                del self.items[rel]
                changed = True
            if changed:
                tmp = self.index_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.items, f, ensure_ascii=False)
                os.replace(tmp, self.index_path)

    def listing(self, sort="new", workflow="", query="", kind="", show_sidecars=False):
        with self.lock:
            items = dict(self.items)
        videos = {os.path.splitext(rel)[0].lower() for rel, e in items.items() if e["kind"] == "video"}
        rows = []
        for rel, e in items.items():
            stem = os.path.splitext(rel)[0].lower()
            # VideoHelperSuite writes <name>.png (first frame) next to <name>.mp4
            if e["kind"] == "image" and stem in videos and not show_sidecars:
                continue
            name, how = name_workflow(e["types"], e["workflow_id"], e["models"], self.catalogue)
            rows.append({"ref": "output:" + rel, "name": os.path.basename(rel), "folder": os.path.dirname(rel),
                         "kind": e["kind"], "mtime": e["mtime"], "size": e["size"], "width": e["width"], "height": e["height"],
                         "duration": e["duration"], "workflow": name, "match": how, "prompt_count": e["prompt_count"],
                         "generated": e["generated"], "snippet": e["snippet"], "_search": e["search"]})
        counts = Counter(r["workflow"] for r in rows)
        query = (query or "").strip().lower()
        rows = [r for r in rows if (not workflow or r["workflow"] == workflow) and (not kind or r["kind"] == kind)
                and (not query or query in r["_search"] or query in r["name"].lower())]
        if sort == "workflow":
            rows.sort(key=lambda r: (r["workflow"] == NO_METADATA, r["workflow"].lower(), -r["mtime"]))
        elif sort == "old":
            rows.sort(key=lambda r: r["mtime"])
        elif sort == "name":
            rows.sort(key=lambda r: r["name"].lower())
        else:
            rows.sort(key=lambda r: -r["mtime"])
        for r in rows:
            del r["_search"]
        return rows, sorted(({"name": n, "count": c} for n, c in counts.items()),
                            key=lambda w: (w["name"] == NO_METADATA, w["name"].lower()))

    def thumbnail(self, ref, size=360):
        full = self.resolve(ref)
        stat = os.stat(full)
        key = hashlib.sha1(f"{full}|{stat.st_mtime}|{stat.st_size}|{size}".encode()).hexdigest()
        out = os.path.join(self.data_dir, "thumbs", key + ".jpg")
        if not os.path.isfile(out):
            if full.lower().endswith(VIDEO_EXT):
                with av.open(full) as container:
                    image = next(container.decode(video=0)).to_image()
            else:
                image = Image.open(full)
            image = image.convert("RGB")
            image.thumbnail((size, size))
            image.save(out, "JPEG", quality=85)
        return out

    def frame(self, ref, which):
        """The first or the last frame of a video as a PNG of full size (kept in data/frames once made)."""
        full = self.resolve(ref)
        if not full.lower().endswith(VIDEO_EXT):
            raise ValueError("영상 파일이 아닙니다")
        stat = os.stat(full)
        key = hashlib.sha1(f"{full}|{stat.st_mtime}|{stat.st_size}|{which}".encode()).hexdigest()
        out = os.path.join(self.data_dir, "frames", key + ".png")
        if not os.path.isfile(out):
            image = None
            with av.open(full) as container:
                stream = container.streams.video[0]
                if which == "last":
                    if container.duration and container.duration > 3 * av.time_base:      # start decoding near the end
                        container.seek(container.duration - 3 * av.time_base)
                    for decoded in container.decode(stream):
                        image = decoded
                    if image is None:      # the seek landed behind the last key frame: go through the whole file
                        container.seek(0)
                        for decoded in container.decode(stream):
                            image = decoded
                else:
                    image = next(container.decode(stream))
                image = image.to_image()
            os.makedirs(os.path.dirname(out), exist_ok=True)
            image.save(out + ".tmp", "PNG")
            os.replace(out + ".tmp", out)
        return out

    def add_dropped(self, filename, data):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXT + VIDEO_EXT:
            raise ValueError("이미지(png, jpg, webp)나 영상(mp4, webm, mkv, mov) 파일만 볼 수 있습니다")
        name = hashlib.sha1(data).hexdigest()[:16] + ext
        with open(os.path.join(self.roots["dropped"], name), "wb") as f:
            f.write(data)
        return "dropped:" + name

    def _own(self, ref):
        """[the file, then what belongs to it]: VideoHelperSuite writes <name>.png (first frame) next to <name>.mp4,
        which the list hides, so it is renamed and deleted together with the video."""
        if not (ref or "").startswith("output:"):
            raise ValueError("보관함 목록에 있는 파일만 바꿀 수 있습니다")
        full = self.resolve(ref)
        side = os.path.splitext(full)[0] + ".png"
        return [full] + ([side] if full.lower().endswith(VIDEO_EXT) and os.path.isfile(side) else [])

    def rename(self, ref, name):
        """Give the file another name in the same folder (the extension stays). Returns the new ref."""
        files = self._own(ref)
        folder, old = os.path.split(files[0])
        stem, ext = os.path.splitext(old)
        name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", name or "").strip()
        if name.lower().endswith(ext.lower()):
            name = name[:-len(ext)]
        name = name.strip(" .")
        if not name:
            raise ValueError("새 이름을 입력하세요")
        moves = [(src, os.path.join(folder, name + os.path.splitext(src)[1])) for src in files]
        for src, dst in moves:
            if os.path.exists(dst) and os.path.normcase(src) != os.path.normcase(dst):
                raise ValueError("같은 이름의 파일이 이미 있습니다: " + os.path.basename(dst))
        for src, dst in moves:
            os.rename(src, dst)
        return "output:" + os.path.relpath(moves[0][1], os.path.realpath(self.roots["output"])).replace("\\", "/")

    def delete(self, ref):
        """Remove the file from the output folder for good. Returns the names removed."""
        files = self._own(ref)
        for path in files:
            os.remove(path)
        return [os.path.basename(path) for path in files]

    def embedded_workflow(self, ref):
        _, workflow, _ = read_metadata(self.resolve(ref))
        return workflow
