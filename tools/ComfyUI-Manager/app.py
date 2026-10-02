"""ComfyUI-Manager (this repo's tool, not the node-pack manager): make and keep track of images and videos.

Create: z-image turbo image -> Qwen scenario (N prompts) -> Wan 2.2 video. Nothing is generated here; every
step is queued on a running ComfyUI through its HTTP API (/prompt, /history, /view, /upload/image, /free, /ws).
Library: lists the output folder and shows, for any image or video, the workflow and prompts it was made with
(read from the metadata ComfyUI embeds in the file); works without ComfyUI running.
History: the jobs that were run. Jobs, the library's index and the user's tags / notes live in one SQLite file
(data/manager.db, see store.py), which is what connects a job to the files it made.

A small aiohttp server with a one-page UI; it only needs packages ComfyUI already ships (aiohttp, Pillow, PyAV).

    python_embeded\\python.exe tools\\ComfyUI-Manager\\app.py [--port 8288] [--comfy http://127.0.0.1:8188]
"""
import argparse
import asyncio
import base64
import copy
import ctypes
import io
import json
import logging
import ntpath
import os
import posixpath
import random
import re
import socket
import string
import sys
import time
import uuid
import webbrowser
from urllib.parse import quote, unquote, urlparse

import aiohttp
import av
from aiohttp import web
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)      # python_embeded does not put the script's folder on sys.path
import comfy_convert  # noqa: E402
import library  # noqa: E402
import store  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TEMPLATES = os.path.join(HERE, "templates")
DATA = os.path.join(HERE, "data")
SETTINGS = os.path.join(DATA, "settings.json")      # what was changed from the page
log = logging.getLogger("comfyui_manager")

DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8288,
    "comfy_url": "http://127.0.0.1:8188",
    "output_dir": "",                                     # empty = picked on the page, else found (see follow_comfy_output)
    "svi_workflow": "workflow/Wan2.2_I2V_SVI_Workflow_Kenpechi_v3.5.json",
    "i2v_api": "workflow/5_I2V_6seg_from_prompts.api.json",
}

DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted face, deformed hands, extra fingers, extra limbs, watermark, text, subtitles, "
    "logo, jpeg artifacts, flicker, static frame, morphing"
)

SCENARIO_BODY = (
    "The image is the FIRST FRAME of a video. Write a {n}-part scenario that continues from it. "
    "Each part is one continuous shot of about {s:g} seconds and starts exactly where the previous part ended. "
    "Keep the same person, outfit, location and lighting as in the image unless the direction says otherwise. "
    "Make the parts flow into each other as one story. No text, captions or logos in the scene.\n"
    "{direction}"
)
# for a vision node that returns the raw model text
SCENARIO_JSON = SCENARIO_BODY + (
    "Return ONLY a JSON object, no markdown, with exactly these keys:\n"
    "\"summary_ko\": Korean, 1-2 sentences describing the whole scenario.\n"
    "\"common_prompt\": English, 20-40 words: the look, outfit, location, lighting and visual style that stay "
    "the same in every part.\n"
    "\"segments\": an array of exactly {n} objects, in order. Each object has: "
    "\"positive_prompt\" (English, 40-70 words, natural sentences: what happens from the start to the end of this "
    "shot, body motion, expression, camera framing and movement) and \"scene_ko\" (Korean, one sentence)."
)
# the detailed version: spells out what every part has to contain, which is what makes the text rich
SCENARIO_RICH = (
    "The image is the FIRST FRAME of a video. First study it closely: who or what is in it, appearance, clothing, "
    "pose, expression, location, background objects, lighting, colours, camera distance and angle.\n"
    "Then write a {n}-part scenario that continues from this frame. Each part is one continuous shot of about "
    "{s:g} seconds and starts exactly in the pose and framing where the previous part ended. Keep the same subject, "
    "outfit, location and lighting unless the direction says otherwise. No text, captions or logos in the scene.\n"
    "{direction}"
    "Every part is ONE paragraph of 90-130 English words in natural present-tense sentences and covers, in this order:\n"
    "1) the starting pose and where the subject is in the frame;\n"
    "2) the action as two or three beats in time order (what moves first, what follows, how it ends), with hands, "
    "head, gaze, weight shift and pace;\n"
    "3) the facial expression and how it changes;\n"
    "4) secondary motion: hair, clothing, objects, background, light;\n"
    "5) the camera: shot size, angle and one clear movement (or static) with its speed;\n"
    "6) the final pose of the shot, which the next part continues from.\n"
    "Each part must describe a different action; do not reuse sentences between parts.\n"
    "Return ONLY a JSON object, no markdown, with exactly these keys:\n"
    "\"summary_ko\": Korean, 1-2 sentences describing the whole scenario.\n"
    "\"common_prompt\": English, 30-50 words: the look, outfit, location, lighting and visual style that stay "
    "the same in every part.\n"
    "\"segments\": an array of exactly {n} objects, in order. Each object has \"positive_prompt\" (the English "
    "paragraph described above) and \"scene_ko\" (Korean, one sentence)."
)
# for Qwen Chat, whose own protocol wraps the answer in {"message": ...}
SCENARIO_CHAT = (
    "This is not a workflow request: leave \"actions\" and \"choices\" empty and put the whole answer in "
    "\"message\". Look at the attached image.\n" + SCENARIO_BODY +
    "Write \"message\" as plain text lines in exactly this layout (labels in capitals, one item per line, "
    "no markdown):\n"
    "SUMMARY_KO: Korean, 1-2 sentences describing the whole scenario\n"
    "COMMON: English, 20-40 words: the look, outfit, location, lighting and visual style that stay the same\n"
    "PART 1: ENGLISH ONLY, 60-100 words, natural sentences: what happens from the start to the end of this shot, "
    "body motion, expression, camera framing and movement\n"
    "KO 1: Korean, one sentence describing part 1\n"
    "PART 2: ...\nKO 2: ...\n(continue the same way up to PART {n} and KO {n})"
)

VIDEO_EXT = (".mp4", ".webm", ".mkv", ".mov", ".gif", ".webp")
BROWSER_VIDEO_EXT = (".mp4", ".webm")


class AppError(Exception):
    """An error whose message is shown to the user as is."""


def read_settings():
    if os.path.isfile(SETTINGS):
        with open(SETTINGS, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------------------------- #
# scenario text -> structure
# ---------------------------------------------------------------------------------------------- #
def _clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _find_json(text):
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first:last + 1])
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _segments_from_json(g):
    raw = None
    for key in ("segments", "scenes", "parts", "shots", "prompts"):
        if isinstance(g.get(key), list):
            raw = g[key]
            break
    if raw is None:
        raw = [g[k] for i in range(1, 33) for k in (f"segment_{i}", f"part_{i}", f"scene_{i}", f"shot_{i}") if k in g]
    out = []
    for item in raw:
        if isinstance(item, str):
            item = {"positive_prompt": item}
        if not isinstance(item, dict):
            continue
        text = _clean(item.get("positive_prompt") or item.get("prompt") or item.get("description"))
        if text:
            out.append({"positive_prompt": text, "scene_ko": _clean(item.get("scene_ko") or item.get("ko"))})
    return out


_LABEL = re.compile(r"\b(SUMMARY_KO|COMMON|PART|KO)[ _]*(\d*)\s*[:：]")


def _parse_labeled(text):
    marks = list(_LABEL.finditer(text))
    out = {"summary_ko": "", "common_prompt": "", "parts": {}, "ko": {}}
    for i, m in enumerate(marks):
        body = _clean(text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]).strip("*# ")
        label, num = m.group(1), m.group(2)
        if label == "SUMMARY_KO":
            out["summary_ko"] = body
        elif label == "COMMON":
            out["common_prompt"] = body
        elif num and label == "PART":
            out["parts"].setdefault(int(num), body)
        elif num and label == "KO":
            out["ko"].setdefault(int(num), body)
    segs = [{"positive_prompt": out["parts"][k], "scene_ko": out["ko"].get(k, "")} for k in sorted(out["parts"]) if out["parts"][k]]
    return {"summary_ko": out["summary_ko"], "common_prompt": out["common_prompt"], "segments": segs}


def parse_scenario(text):
    """Model answer -> {summary_ko, common_prompt, segments:[{positive_prompt, scene_ko}]}; segments may be empty."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    data = _find_json(text)
    if data is not None:
        segs = _segments_from_json(data)
        if segs:
            return {"summary_ko": _clean(data.get("summary_ko")), "common_prompt": _clean(data.get("common_prompt")),
                    "segments": segs}
        if isinstance(data.get("message"), str):      # chat protocol echoed back as raw JSON
            text = data["message"]
    labeled = _parse_labeled(text)
    if labeled["segments"]:
        return labeled
    numbered = re.findall(r"(?m)^\s*(\d+)\s*[.)]\s+(.+)$", text)
    return {"summary_ko": labeled["summary_ko"], "common_prompt": labeled["common_prompt"],
            "segments": [{"positive_prompt": _clean(t), "scene_ko": ""} for _, t in numbered]}


def _fmt_time(sec):
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def build_scenario_doc(parsed, n, seconds, width, height, direction, image_ref):
    segs = list(parsed["segments"][:n])
    while len(segs) < n:
        segs.append({"positive_prompt": "", "scene_ko": ""})
    out = []
    for i, sg in enumerate(segs):
        start, end = round(i * seconds, 3), round((i + 1) * seconds, 3)
        out.append({"index": i + 1, "start": start, "end": end, "duration": round(seconds, 3),
                    "label": f"{_fmt_time(start)} - {_fmt_time(end)}",
                    "positive_prompt": sg["positive_prompt"], "motion": "", "camera": "", "scene_ko": sg.get("scene_ko", "")})
    return {
        "source": "comfyui_manager", "video_file": "", "image_file": "reference.png", "image_ref": image_ref,
        "direction": direction, "duration": round(n * seconds, 3), "width": width, "height": height,
        "segment_length": seconds, "fps": 16,
        "summary_ko": parsed.get("summary_ko", ""), "common_prompt": parsed.get("common_prompt", ""),
        "negative_prompt": DEFAULT_NEGATIVE, "segments": out,
    }


def _model_key(name):
    """A model name as it is compared with a folder name: 'Qwen3.8-27B-Uncensored (choz)' -> 'qwen3.827buncensored'."""
    return re.sub(r"[^a-z0-9.]", "", re.sub(r"\(.*?\)", "", name.lower()))


def sanitize_name(name):
    name = re.sub(r'[\\/:*?"<>|.\x00-\x1f]', "", name or "").strip()
    return re.sub(r"\s+", "_", name)[:40] or "scenario"


def upload_name(scenario_dir):
    """File name for the start image in ComfyUI's input folder. ASCII only: multipart file names get
    percent-encoded on the way, and the scenario folder may be Korean."""
    return "".join(c for c in scenario_dir if c.isascii() and (c.isalnum() or c == "_")).strip("_") + ".png"


# ---------------------------------------------------------------------------------------------- #
# ComfyUI's output folder as this PC reaches it
# ---------------------------------------------------------------------------------------------- #
def network_drives():
    """{'V:\\': ('192.168.1.220', 'ComfyUI')} for the network drives of this PC. Windows is only asked for the
    mapping; no share is opened, so a drive whose server is off costs nothing."""
    if os.name != "nt":
        return {}
    drives = {}
    for letter in string.ascii_uppercase:
        buf, size = ctypes.create_unicode_buffer(1024), ctypes.c_ulong(1024)
        if ctypes.windll.mpr.WNetGetConnectionW(letter + ":", buf, ctypes.byref(size)) == 0:
            host, _, share = buf.value.strip("\\").partition("\\")
            drives[letter + ":\\"] = (host, share)
    return drives


def output_under(folder, comfy_output):
    """`folder` was picked as the place where the server's results are. When it is a parent of the server's output
    folder (V:\\ for a server that saves to E:\\ComfyUI\\output), the output folder inside it is what is meant."""
    parts = [p for p in re.split(r"[\\/]", ntpath.splitdrive(comfy_output)[1]) if p]
    if parts and os.path.basename(os.path.normpath(folder)).lower() != parts[-1].lower():
        for i in range(len(parts)):
            path = os.path.join(folder, *parts[i:])
            if os.path.isdir(path):
                return os.path.normpath(path)
    return folder


def list_folders(path):
    """What the folder picker shows: the subfolders of `path` on this PC, or the drives when path is ''."""
    if not path:
        if os.name != "nt":
            return {"path": "", "parent": None, "dirs": [{"name": "/", "path": "/", "note": ""}]}
        shares, mask = network_drives(), ctypes.windll.kernel32.GetLogicalDrives()
        roots = [c + ":\\" for i, c in enumerate(string.ascii_uppercase) if mask >> i & 1]
        return {"path": "", "parent": None,
                "dirs": [{"name": r[:2], "path": r, "note": "\\\\" + "\\".join(shares[r]) if r in shares else ""} for r in roots]}
    path = os.path.abspath(path)
    names = []
    with os.scandir(path) as entries:
        for e in entries:
            try:
                if e.is_dir() and not e.name.startswith((".", "$")):
                    names.append(e.name)
            except OSError:
                pass
    parent = os.path.dirname(path)
    return {"path": path, "parent": "" if parent == path else parent,
            "dirs": [{"name": n, "path": os.path.join(path, n), "note": ""} for n in sorted(names, key=str.lower)]}


def path_module(folder):
    """ntpath or posixpath, whichever the machine that named `folder` uses."""
    return ntpath if "\\" in folder or folder[1:2] == ":" else posixpath


def _address(host):
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host.lower()


def is_this_machine(host):
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    try:
        return _address(host) in socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return False


def local_view(host, folder):
    """`folder` of the machine `host` as a path of this PC: the folder itself when host is this machine, otherwise
    through a network drive of that machine whose share is named after a folder on the way (E:\\ComfyUI shared as
    'ComfyUI' and mapped to V: makes E:\\ComfyUI\\output -> V:\\output) or after the drive (E, E$). None when there is none."""
    if is_this_machine(host):
        return folder if os.path.isdir(folder) else None
    drive, rest = ntpath.splitdrive(folder)
    parts = [drive.rstrip(":")] + [p for p in re.split(r"[\\/]", rest) if p]
    for root, (server, share) in network_drives().items():
        if _address(server) != _address(host):
            continue
        name = share.rsplit("\\", 1)[-1].rstrip("$").lower()
        for i, part in enumerate(parts):
            path = os.path.join(root, *parts[i + 1:])
            if part.lower() == name and os.path.isdir(path):
                return os.path.normpath(path)
    return None


# ---------------------------------------------------------------------------------------------- #
# ComfyUI client
# ---------------------------------------------------------------------------------------------- #
class Comfy:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.session = None
        self._object_info = None
        self.ws_ok = False

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))

    async def close(self):
        await self.session.close()

    async def get_json(self, path, timeout=30, **params):
        async with self.session.get(self.url + path, params=params or None, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()

    async def post_json(self, path, data, timeout=30):
        async with self.session.post(self.url + path, json=data, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            text = await r.text()
            try:
                body = json.loads(text) if text else {}
            except json.JSONDecodeError:
                body = {"error": text[:500]}
            return r.status, body

    async def object_info(self):
        if self._object_info is None:
            self._object_info = await self.get_json("/object_info", timeout=120)
        return self._object_info

    async def view_bytes(self, ref):
        params = {"filename": ref["filename"], "subfolder": ref.get("subfolder", ""), "type": ref.get("type", "output")}
        async with self.session.get(self.url + "/view", params=params, timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                raise AppError(f"ComfyUI에서 이미지를 읽지 못했습니다 ({r.status}): {ref['filename']}")
            return await r.read()

    async def upload_image(self, data, filename, subfolder="webapp"):
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename, content_type="image/png")
        form.add_field("subfolder", subfolder)
        form.add_field("type", "input")
        form.add_field("overwrite", "true")
        async with self.session.post(self.url + "/upload/image", data=form, timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                raise AppError(f"ComfyUI에 이미지를 올리지 못했습니다 ({r.status}): {(await r.text())[:300]}")
            body = await r.json()
        return {"filename": body["name"], "subfolder": body.get("subfolder", ""), "type": body.get("type", "input")}

    async def queue_prompt(self, prompt):
        status, body = await self.post_json("/prompt", {"prompt": prompt, "client_id": self.client_id}, timeout=120)
        if status == 200 and body.get("prompt_id"):
            return body["prompt_id"]
        lines = []
        err = body.get("error")
        if isinstance(err, dict):
            lines.append(f"{err.get('message', '')} {err.get('details', '')}".strip())
        elif err:
            lines.append(str(err))
        for nid, ne in (body.get("node_errors") or {}).items():
            title = (prompt.get(nid, {}).get("_meta") or {}).get("title") or ne.get("class_type", "")
            for e in ne.get("errors", []):
                lines.append(f"[{nid} {title}] {e.get('message', '')}: {e.get('details', '')}")
        raise AppError("ComfyUI가 작업을 거부했습니다 (" + str(status) + ")\n" + "\n".join(lines[:20]))


# ---------------------------------------------------------------------------------------------- #
# jobs
# ---------------------------------------------------------------------------------------------- #
class Job:
    def __init__(self, kind, label):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        self.state = "queued"        # queued -> running -> done | error | cancelled
        self.step = "대기 중"
        self.prompt_id = None
        self.titles = {}
        self.node = ""
        self.nodes_done = set()
        self.progress = None
        self.outputs = []
        self.result = None
        self.error = ""
        self.notes = []
        self.created = time.time()
        self.started = None
        self.finished = None
        self.task = None
        self.cancel_requested = False
        self.params = {}             # what the page asked for; kept in the history

    def to_dict(self):
        end = self.finished or time.time()
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "state": self.state, "step": self.step,
            "node": self.node, "nodes_done": len(self.nodes_done), "nodes_total": len(self.titles),
            "progress": self.progress, "outputs": self.outputs, "result": self.result, "error": self.error,
            "notes": self.notes, "elapsed": round(end - (self.started or self.created), 1),
        }


def job_keys(entry):
    """(scenario folder, start image) a job of the history is connected by: a scenario job made the folder from the
    image, a video job was made from the folder. The image is its path inside the result folder ('' when it is not there)."""
    params, result = entry.get("params") or {}, entry.get("result") or {}
    if entry["kind"] == "scenario":
        image = params.get("image") or {}
        rel = "/".join(q for q in re.split(r"[\\/]", image.get("subfolder") or "") + [image.get("filename") or ""] if q)
        return result.get("dir") or "", rel if image.get("type") == "output" else ""
    return (params.get("dir") or "" if entry["kind"] == "video" else ""), ""


def main_files(row):
    """The files a stored job is known by: the final video of a video job, everything an image job saved."""
    outputs, final = row["entry"].get("outputs") or [], (row["entry"].get("result") or {}).get("final") or {}
    return [rel for n, rel in sorted(row["files"].items()) if n < len(outputs) and (
        row["kind"] != "video" or (outputs[n]["filename"], outputs[n]["type"]) == (final.get("filename"), final.get("type")))]


def output_files(node_id, title, output):
    files = []
    for key in ("images", "gifs", "video", "videos"):
        for it in output.get(key) or []:
            if not isinstance(it, dict) or not it.get("filename"):
                continue
            ext = os.path.splitext(it["filename"])[1].lower()
            is_video = ext in VIDEO_EXT and (key != "images" or ext not in (".gif", ".webp") or bool(output.get("animated")))
            files.append({"node": node_id, "title": title, "filename": it["filename"], "subfolder": it.get("subfolder", ""),
                          "type": it.get("type", "output"), "kind": "video" if is_video else "image",
                          "playable": ext in BROWSER_VIDEO_EXT})
    return files


class App:
    def __init__(self, config):
        self.cfg = config
        self.comfy = Comfy(config["comfy_url"])
        saved = read_settings()
        self.recent = {key: [v for v in saved.get(key) or [] if isinstance(v, str)] for key in ("recent_urls", "recent_dirs")}
        # The result folder: where this PC sees what ComfyUI saves. Nothing is generated or copied here; the files are
        # read where the server put them. One folder per ComfyUI address, picked on the page.
        self.result_dirs = {k: v for k, v in (saved.get("result_dirs") or {}).items() if isinstance(v, str)}
        if "result_dirs" not in saved and saved.get("fetch_dir"):      # picked before this became the result folder
            self.result_dirs[config["comfy_url"]] = saved["fetch_dir"]
        picked = self.result_dirs.get(config["comfy_url"], "")
        self.output_dir = config["output_dir"] or picked or os.path.join(ROOT, "output")
        # config: config.json / --output-dir, chosen: picked on the page, auto: found, default: not known (<root>/output)
        self.output_source = "config" if config["output_dir"] else "chosen" if picked else "default"
        self.comfy_output = ""           # ComfyUI's output folder as its own machine names it (E:\ComfyUI\output)
        self._followed = None            # (address, that folder) last looked up, whether it was found, when
        self.jobs = {}
        self.by_prompt = {}
        self.lock = asyncio.Lock()       # one stage at a time: the stages hand VRAM over to each other
        self.last_stage = None
        self._svi_cache = None
        self._chat_models = (0, None)
        self._combos = {}
        self._llm = (0, None)
        self.ws = None
        self.address_changed = asyncio.Event()
        self.store = store.Store(os.path.join(DATA, "manager.db"))
        self.library = library.Library(self.output_dir, DATA, library.Catalogue(os.path.join(ROOT, "workflow"), TEMPLATES), self.store)
        self.import_history(os.path.join(DATA, "history.json"))

    # ---- helpers -----------------------------------------------------------------------------
    def path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)

    def scenario_dir(self, name):
        if not name or name != os.path.basename(name) or name in (".", ".."):
            raise AppError("잘못된 시나리오 폴더 이름입니다")
        return os.path.join(self.output_dir, name)

    def load_scenario(self, name):
        path = os.path.join(self.scenario_dir(name), "prompts.json")
        if not os.path.isfile(path):
            raise AppError(f"시나리오가 없습니다: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_scenario(self, name, doc):
        folder = self.scenario_dir(name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "prompts.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        with open(os.path.join(folder, "comfyui_prompts.txt"), "w", encoding="utf-8") as f:
            f.write("COMMON:\n" + doc.get("common_prompt", "") + "\n\nNEGATIVE:\n" + doc.get("negative_prompt", "") + "\n\n")
            for sg in doc.get("segments", []):
                f.write(f"### part_{sg['index']:02d} [{sg.get('label', '')}]\n{sg.get('positive_prompt', '')}\n{sg.get('scene_ko', '')}\n\n")

    def list_scenarios(self):
        found = []
        if os.path.isdir(self.output_dir):
            for name in os.listdir(self.output_dir):
                p = os.path.join(self.output_dir, name, "prompts.json")
                if name.endswith("_prompts") and os.path.isfile(p):
                    found.append((os.path.getmtime(p), name))
        return [{"dir": name, "mtime": int(m)} for m, name in sorted(found, reverse=True)[:100]]

    async def chat_models(self):
        """{'hf': [...], 'gguf': [...]} from the QwenVL-Mod chat endpoint, or None when it is not installed."""
        stamp, models = self._chat_models
        if time.time() - stamp > 30:
            try:
                models = await self.comfy.get_json("/qwenvl/chat/models", timeout=5)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                models = None
            self._chat_models = (time.time(), models)
        return models

    async def llm_on_disk(self):
        """What is in ComfyUI's models/LLM folder: (names of the .gguf files, folders that hold model weights), both
        lower case. None when ComfyUI cannot tell (not reachable, or the folder is not registered)."""
        stamp, found = self._llm
        if time.time() - stamp > 60:
            files, answered = [], False
            for folder in ("LLM", "llm"):
                try:
                    files += await self.comfy.get_json(f"/models/{folder}", timeout=20)
                    answered = True
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                    pass
            names = [f.replace("\\", "/").lower() for f in files if isinstance(f, str)]
            found = None if not answered else (
                {n.rsplit("/", 1)[-1] for n in names if n.endswith(".gguf")},
                {_model_key(n.split("/")[-2]) for n in names if "/" in n and n.endswith((".safetensors", ".bin"))})
            self._llm = (time.time(), found)
        return found

    @staticmethod
    def on_disk(found, names, gguf):
        """The models of a node's list that are really there; the whole list when that cannot be checked."""
        if found is None:
            return list(names)
        files, folders = found
        return [n for n in names if (n.lower() in files if gguf else _model_key(n.rsplit("/", 1)[-1]) in folders)]

    async def combo_values(self, class_type, name="model_name"):
        """Choices of a node's combo input ([] when the node is not installed)."""
        stamp, values = self._combos.get(class_type, (0, []))
        if time.time() - stamp > 60:
            info = (await self.comfy.get_json(f"/object_info/{class_type}", timeout=10)).get(class_type)
            values = info["input"]["required"][name][0] if info else []
            self._combos[class_type] = (time.time(), values)
        return values

    async def hand_over(self, stage, job):
        """Free VRAM held by the previous stage before a different one starts."""
        if self.last_stage == stage:
            return
        job.step = "이전 단계 모델을 VRAM에서 내리는 중"
        try:
            if self.last_stage in (None, "chat"):
                await self.comfy.post_json("/qwenvl/chat/unload", {"backend": "all"}, timeout=60)
            await self.comfy.post_json("/free", {"unload_models": True, "free_memory": True})
            await asyncio.sleep(2.0)      # the worker applies /free when it next wakes up; let it finish before queueing
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        self.last_stage = stage

    # ---- running a prompt on ComfyUI ---------------------------------------------------------
    async def run_prompt(self, job, prompt):
        job.titles = {nid: (n.get("_meta") or {}).get("title") or n["class_type"] for nid, n in prompt.items()}
        job.nodes_done, job.progress, job.node = set(), None, ""
        pid = await self.comfy.queue_prompt(prompt)
        job.prompt_id = pid
        self.by_prompt[pid] = job
        job.step = "ComfyUI 대기열"
        missing = 0
        try:
            while True:
                await asyncio.sleep(1.0)
                entry = (await self.comfy.get_json(f"/history/{pid}")).get(pid)
                if entry:
                    break
                queue = await self.comfy.get_json("/queue")
                ids = [item[1] for item in queue.get("queue_running", []) + queue.get("queue_pending", [])]
                if pid in [item[1] for item in queue.get("queue_running", [])]:
                    job.step = "ComfyUI 실행 중"
                missing = 0 if pid in ids else missing + 1
                if missing >= 2 and job.cancel_requested:      # removed from the queue before it started
                    raise asyncio.CancelledError()
                if missing >= 5:
                    raise AppError("ComfyUI 대기열에서 작업이 사라졌습니다 (ComfyUI가 재시작되었나요?)")
        finally:
            self.by_prompt.pop(pid, None)
        for nid, out in (entry.get("outputs") or {}).items():
            self.add_outputs(job, nid, out)
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            for name, data in status.get("messages") or []:
                if name == "execution_interrupted":
                    raise asyncio.CancelledError()
                if name == "execution_error":
                    title = job.titles.get(str(data.get("node_id")), data.get("node_type", ""))
                    raise AppError(f"[{data.get('node_id')} {title}] {data.get('exception_type', '')}: {data.get('exception_message', '')}")
            raise AppError("ComfyUI 실행 중 오류가 났습니다 (ComfyUI 콘솔을 확인하세요)")
        return entry.get("outputs") or {}

    def add_outputs(self, job, node_id, output):
        if not isinstance(output, dict):
            return
        seen = {(f["filename"], f["subfolder"], f["type"]) for f in job.outputs}
        for f in output_files(node_id, job.titles.get(node_id, node_id), output):
            if (f["filename"], f["subfolder"], f["type"]) not in seen:
                job.outputs.append(f)

    def on_ws(self, msg):
        data = msg.get("data") or {}
        job = self.by_prompt.get(data.get("prompt_id"))
        if job is None:
            return
        kind = msg.get("type")
        if kind == "execution_cached":
            job.nodes_done.update(data.get("nodes") or [])
        elif kind == "executing" and data.get("node") is not None:
            job.step = "ComfyUI 실행 중"
            job.node = job.titles.get(data["node"], str(data["node"]))
            job.nodes_done.add(data["node"])
            job.progress = None
        elif kind == "progress":
            job.progress = {"value": data.get("value", 0), "max": data.get("max", 0)}
        elif kind == "executed":
            self.add_outputs(job, str(data.get("node")), data.get("output") or {})

    async def ws_loop(self):
        while True:
            self.address_changed.clear()
            ws_url = "ws" + self.comfy.url[4:] + "/ws?clientId=" + self.comfy.client_id
            try:
                async with self.comfy.session.ws_connect(ws_url, heartbeat=30, max_msg_size=0) as ws:
                    self.ws = ws
                    self.comfy.ws_ok = True
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self.on_ws(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                pass
            self.ws = None
            self.comfy.ws_ok = False
            self.comfy._object_info = None      # ComfyUI may come back with different nodes or models
            try:                                # retry in 3 s, or at once when the address was changed
                await asyncio.wait_for(self.address_changed.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

    async def set_comfy(self, url):
        """Point the tool at another ComfyUI (local or on another machine) and remember it."""
        self.comfy.url = url
        self.comfy._object_info = None
        self._combos, self._chat_models, self._svi_cache, self.last_stage = {}, (0, None), None, None
        self._llm = (0, None)
        self.remember(comfy_url=url)
        self.comfy_output = ""
        self.apply_picked()
        self.address_changed.set()
        if self.ws is not None:
            await self.ws.close()

    def apply_picked(self):
        """The folder picked for the ComfyUI in use, before ComfyUI has been asked anything (the library works without it)."""
        self._followed = None
        if self.output_source != "config":
            picked = self.result_dirs.get(self.comfy.url, "")
            self.output_dir = self.library.roots["output"] = picked or os.path.join(ROOT, "output")
            self.output_source = "chosen" if picked else "default"

    def set_result_dir(self, path):
        """Pick, for the ComfyUI in use, the folder of this PC that shows what that server saves ('' = find it)."""
        if path:
            self.result_dirs[self.comfy.url] = path
            self.keep_recent("recent_dirs", path)
        else:
            self.result_dirs.pop(self.comfy.url, None)
        self.remember(result_dirs=self.result_dirs)
        self.apply_picked()

    async def follow_comfy_output(self, argv):
        """Settle which folder of this PC holds what ComfyUI saves, now that ComfyUI told how it was started. In order:
        config.json / --output-dir; the folder picked on the page for this server; else found without being told
        (ComfyUI on this machine: its --output-directory; on another machine: that folder through a network drive)."""
        folder = argv[argv.index("--output-directory") + 1] if "--output-directory" in argv[:-1] else ""
        folder = path_module(folder).normpath(folder) if folder else ""
        key = (self.comfy.url, folder)
        if self._followed and self._followed[0] == key and (self._followed[1] or time.time() - self._followed[2] < 30):
            return      # settled, or looked for a moment ago (a network drive may be connected later)
        if self.lock.locked():      # a running stage keeps the folder it started with
            return
        self._followed = (key, False, time.time())
        self.comfy_output = folder
        if self.output_source == "config":
            return
        picked = self.result_dirs.get(self.comfy.url, "")
        if picked:
            path = await asyncio.to_thread(output_under, picked, folder)
        else:
            path = await asyncio.to_thread(local_view, urlparse(self.comfy.url).hostname, folder) if folder else None
        if not path or self._followed[0] != key or self.lock.locked() or picked != self.result_dirs.get(self.comfy.url, ""):
            return
        if picked and path != picked:      # a parent of the output folder had been picked: keep what is really used
            self.result_dirs[self.comfy.url] = path
            self.remember(result_dirs=self.result_dirs)
            self.keep_recent("recent_dirs", path)
        self._followed = (key, True, time.time())
        self.output_dir = self.library.roots["output"] = path
        self.output_source = "chosen" if picked else "auto"

    def keep_recent(self, key, value):
        """Addresses and folders used before, newest first: the lists to pick from on the page."""
        self.recent[key] = ([value] + [v for v in self.recent[key] if v != value])[:15]
        self.remember(**{key: self.recent[key]})

    @staticmethod
    def remember(**changes):
        settings = read_settings()
        settings.update(changes)
        os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
        with open(SETTINGS, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False)

    async def find_outputs(self, job):
        """Look up, in the result folder, the files the job saved on the server (type 'output'). Nothing is copied.
        Returns '' or, for the user, why some of them cannot be seen from this PC."""
        base, missing = os.path.realpath(self.output_dir), []
        for f in job.outputs:
            if f["type"] != "output" or f.get("saved"):
                continue
            shown = os.path.normpath(os.path.join(self.output_dir, f["subfolder"], f["filename"]))
            dest = os.path.realpath(shown)
            if os.path.commonpath([dest, base]) != base:
                continue
            for _ in range(5):      # a file just written on another machine can take a moment to show on the network drive
                if os.path.isfile(dest) or self.output_source == "default":
                    break
                await asyncio.sleep(1.0)
            if os.path.isfile(dest):
                f["saved"] = shown
            else:
                missing.append(f["filename"])
        if not missing:
            return ""
        if self.output_source == "default":
            return "서버에는 저장되었지만 이 PC에서는 볼 수 없습니다. 설정에서 서버 결과 폴더를 지정하면 보관함에 나옵니다"
        return (f"서버 결과 폴더({self.output_dir})에서 보이지 않는 파일: {', '.join(missing[:3])}. "
                "설정의 서버 결과 폴더가 그 서버의 출력 폴더가 맞는지 확인하세요")

    def saved_text(self, f):
        """Where a saved file is: the path on ComfyUI's machine, with the path on this PC when that is a different one."""
        local = f.get("saved", "")
        if f["type"] != "output" or not self.comfy_output:
            return local
        there = path_module(self.comfy_output).join(self.comfy_output, *re.split(r"[\\/]", f["subfolder"]), f["filename"])
        if not local or os.path.normcase(there) == os.path.normcase(local):
            return there
        return f"{there} (이 PC에서는 {local})"

    def history_entry(self, job):
        """A job as the history shows it: what was asked, how it ended, and what it left behind."""
        entry = job.to_dict()
        for key in ("step", "node", "nodes_done", "nodes_total", "progress"):
            del entry[key]
        result = job.result or {}
        if job.kind == "scenario":      # the scenario itself is in its folder; the history only points at it
            doc = result.get("scenario") or {}
            result = {"dir": result.get("dir", ""), "warning": result.get("warning", ""),
                      "summary_ko": doc.get("summary_ko", ""), "segments": len(doc.get("segments") or [])}
        outputs = []
        for f in job.outputs:
            rel = "/".join(p for p in re.split(r"[\\/]", f["subfolder"]) + [f["filename"]] if p)
            here = f["type"] == "output" and os.path.isfile(os.path.join(self.output_dir, rel))
            outputs.append(dict(f, ref="output:" + rel if here else ""))      # ref: the file in the library
        entry.update(time=job.created, params=job.params, result=result, outputs=outputs, comfy_url=self.comfy.url)
        return entry

    def record(self, job):
        """Keep a finished job, connected to the files it made."""
        entry = self.history_entry(job)
        try:
            self.store.add_job(entry, store.root_key(self.output_dir), *job_keys(entry),
                               [f["ref"].partition(":")[2] or None for f in entry["outputs"]])
        except Exception:      # a job must not fail because its record could not be written
            log.exception("history not saved")

    def import_history(self, path):
        """The history an earlier version kept in history.json, taken over once."""
        if not os.path.isfile(path) or self.store.job_count():
            return
        try:
            with open(path, encoding="utf-8") as f:
                for entry in reversed(json.load(f)):
                    self.store.add_job(entry, store.root_key(self.output_dir), *job_keys(entry),
                                       [(f.get("ref") or "").partition(":")[2] or None for f in entry.get("outputs") or []])
            os.replace(path, path + ".imported")
        except (OSError, ValueError, KeyError):
            log.exception("history.json not imported")

    def stored_entry(self, row):
        """A job of the store as the page shows it: its files where they are now, and what it is connected to."""
        entry, here = row["entry"], row["root"] == store.root_key(self.output_dir)
        for n, f in enumerate(entry.get("outputs") or []):
            f["ref"] = "output:" + row["files"][n] if here and n in row["files"] else ""
        entry["links"] = self.links(row)
        return entry

    def links(self, row):
        """What a job is connected to: the scenario and start image it used, the scenarios and videos made from it.
        [{'label', 'ref' + 'name'} for a file | {'label', 'dir'} for a scenario folder]"""
        here = row["root"] == store.root_key(self.output_dir)
        out = []

        def file(label, rel):
            out.append({"label": label, "name": rel.rsplit("/", 1)[-1], "ref": "output:" + rel if here else ""})

        def videos(folder):
            for video in self.store.jobs_like("video", root=row["root"], scenario_dir=folder, state="done"):
                for rel in main_files(video):
                    file("만든 영상", rel)

        folder = row["scenario_dir"]
        if row["kind"] == "video" and folder:
            out.append({"label": "시나리오", "dir": folder})
            made = self.store.jobs_like("scenario", 1, root=row["root"], scenario_dir=folder)
            source = made[0]["source_rel"] if made else self.scenario_source(folder) if here else ""
            if source:
                file("시작 이미지", source)
        elif row["kind"] == "scenario":
            if row["source_rel"]:
                file("시작 이미지", row["source_rel"])
            if folder:
                out.append({"label": "시나리오", "dir": folder})
                videos(folder)
        elif row["kind"] == "image":
            for rel in main_files(row):
                for scenario in self.store.jobs_like("scenario", root=row["root"], source_rel=rel):
                    if scenario["scenario_dir"]:
                        out.append({"label": "이 이미지로 만든 시나리오", "dir": scenario["scenario_dir"]})
                        videos(scenario["scenario_dir"])
        return out

    def scenario_source(self, folder):
        """The start image a scenario folder names (for a scenario that was not made through a recorded job)."""
        try:
            image = self.load_scenario(folder).get("image_ref") or {}
        except (AppError, OSError, ValueError):
            return ""
        rel = "/".join(q for q in re.split(r"[\\/]", image.get("subfolder") or "") + [image.get("filename") or ""] if q)
        return rel if image.get("type") == "output" else ""

    def start_job(self, kind, label, coro_fn, *args):
        job = Job(kind, label)
        job.params = args[0] if args and isinstance(args[0], dict) else {}
        self.jobs[job.id] = job
        for old in sorted(self.jobs.values(), key=lambda j: j.created)[:-50]:
            self.jobs.pop(old.id, None)

        async def runner():
            try:
                async with self.lock:
                    job.state, job.started = "running", time.time()
                    job.result = await coro_fn(job, *args)
                job.state, job.step = "done", "완료"
            except asyncio.CancelledError:
                job.state, job.step = "cancelled", "중단됨"
            except AppError as e:
                job.state, job.step, job.error = "error", "오류", str(e)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                job.state, job.step = "error", "오류"
                job.error = f"ComfyUI({self.comfy.url})와 통신하지 못했습니다: {type(e).__name__} {e}"
            except Exception as e:  # shown in the UI instead of dying silently in a background task
                log.exception("job %s failed", job.id)
                job.state, job.step, job.error = "error", "오류", f"{type(e).__name__}: {e}"
            job.finished = time.time()
            self.record(job)

        job.task = asyncio.create_task(runner())
        return job

    # ---- stage 1: image ----------------------------------------------------------------------
    async def stage_image(self, job, p):
        with open(os.path.join(TEMPLATES, "image_zimage.api.json"), encoding="utf-8") as f:
            prompt = json.load(f)
        text = (p.get("prompt") or "").strip()
        if not text:
            raise AppError("이미지 프롬프트를 입력하세요")
        seed = int(p.get("seed") or 0) or random.randint(1, 2 ** 48)
        prompt["103"]["inputs"]["text"] = text
        prompt["105"]["inputs"].update(width=int(p.get("width") or 960) // 16 * 16, height=int(p.get("height") or 1424) // 16 * 16)
        prompt["107"]["inputs"].update(seed=seed, steps=int(p.get("steps") or 8))
        prompt["120"]["inputs"]["filename_prefix"] = "webapp/zimage_" + time.strftime("%Y%m%d")
        await self.hand_over("image", job)
        outputs = await self.run_prompt(job, prompt)
        images = (outputs.get("120") or {}).get("images") or []
        if not images:
            raise AppError("이미지가 만들어지지 않았습니다")
        unseen = await self.find_outputs(job)
        ref = {k: images[0].get(k, "") for k in ("filename", "subfolder", "type")}
        saved = next((" · ".join(filter(None, (self.saved_text(f), unseen))) for f in job.outputs
                      if f["type"] == "output" and f["filename"] == ref["filename"]), "")
        return {"image": ref, "saved": saved, "seed": seed, "width": prompt["105"]["inputs"]["width"], "height": prompt["105"]["inputs"]["height"]}

    # ---- stage 2: scenario -------------------------------------------------------------------
    async def stage_scenario(self, job, p):
        ref = p.get("image") or {}
        if not ref.get("filename"):
            raise AppError("먼저 이미지를 만들거나 올리세요")
        n = max(1, min(8, int(p.get("segments") or 6)))
        seconds = float(p.get("seconds") or 5)
        direction = (p.get("direction") or "").strip()
        analyzer = p.get("analyzer") or "gguf"
        template = SCENARIO_RICH if (p.get("detail") or "rich") == "rich" else SCENARIO_JSON

        job.step = "이미지 준비"
        image = Image.open(io.BytesIO(await self.comfy.view_bytes(ref))).convert("RGB")
        name = time.strftime("%Y%m%d_%H%M%S") + "_" + sanitize_name(p.get("name")) + "_prompts"
        folder = self.scenario_dir(name)
        os.makedirs(folder, exist_ok=True)
        png = io.BytesIO()
        image.save(png, "PNG")
        with open(os.path.join(folder, "reference.png"), "wb") as f:
            f.write(png.getvalue())

        clause = f"Direction from the user (follow it): {direction}\n" if direction else ""
        raw = ""
        if analyzer == "chat":
            raw = await self.analyze_chat(job, image, SCENARIO_CHAT.format(n=n, s=seconds, direction=clause), p)
        elif analyzer in ("gguf", "node"):
            uploaded = await self.comfy.upload_image(png.getvalue(), upload_name(name))
            run = self.analyze_gguf if analyzer == "gguf" else self.analyze_node
            raw = await run(job, uploaded, template.format(n=n, s=seconds, direction=clause), p)
        parsed = parse_scenario(raw)
        doc = build_scenario_doc(parsed, n, seconds, image.width, image.height, direction, ref)
        self.save_scenario(name, doc)
        if raw:
            with open(os.path.join(folder, "model_answer.txt"), "w", encoding="utf-8") as f:
                f.write(raw)
        got = len(parsed["segments"])
        warning = ""
        if analyzer != "manual" and got < n:
            warning = (f"모델이 구간 {n}개 중 {got}개만 형식에 맞게 썼습니다. 아래 원문을 참고해 빈 칸을 직접 채우거나 다시 생성하세요."
                       if got else "모델 답변에서 구간 프롬프트를 찾지 못했습니다. 아래 원문을 참고해 직접 채우거나 다시 생성하세요.")
        return {"dir": name, "scenario": doc, "raw": raw, "warning": warning}

    async def analyze_chat(self, job, image, instruction, p):
        models = await self.chat_models()
        if models is None:
            raise AppError("Qwen Chat을 쓸 수 없습니다. ComfyUI에 ComfyUI-QwenVL-Mod가 설치되어 있어야 합니다 "
                           "(분석 방법을 'QwenVL 노드'로 바꾸면 원본 QwenVL 노드로 분석합니다).")
        backend = p.get("chat_backend") or ("gguf" if models.get("gguf") else "hf")
        small = image.copy()
        small.thumbnail((1536, 1536))
        jpg = io.BytesIO()
        small.save(jpg, "JPEG", quality=92)
        await self.hand_over("chat", job)
        job.step = "Qwen Chat이 이미지를 분석하는 중 (모델 로딩 포함, 수 분 걸릴 수 있음)"
        body = {
            "backend": backend, "model": p.get("chat_model") or None,
            "messages": [{"role": "user", "content": instruction}],
            "graph": {"nodes": []}, "images": [base64.b64encode(jpg.getvalue()).decode("ascii")],
            "options": {"max_tokens": int(p.get("max_tokens") or 2048), "temperature": float(p.get("temperature") or 0.4)},
        }
        status, res = await self.comfy.post_json("/qwenvl/chat", body, timeout=1800)
        if status == 500:
            # QwenVL-Mod keeps the chat model loaded and, unpatched, fails every call after the first one
            # ("Fatal Decode Error at Pos 0"). A freshly loaded model does not have that problem.
            job.step = "Qwen Chat 오류, 모델을 다시 올려 재시도하는 중"
            await self.comfy.post_json("/qwenvl/chat/unload", {"backend": "all"}, timeout=60)
            status, res = await self.comfy.post_json("/qwenvl/chat", body, timeout=1800)
        if status != 200:
            raise AppError(f"Qwen Chat 오류 ({status}): {res.get('error', res)}")
        return res.get("message") or ""

    async def analyze_gguf(self, job, uploaded, instruction, p):
        """A QwenVL-Mod GGUF model asked directly (no preset, no chat protocol) through our own node."""
        models = [m for m in await self.combo_values("ShortsQwenGGUFVision") if not m.startswith("(")]
        models = self.on_disk(await self.llm_on_disk(), models, True) or models      # the default is one that is downloaded
        if not models:
            raise AppError("Qwen GGUF 직접 호출을 쓸 수 없습니다. ComfyUI에 ComfyUI-QwenVL-Mod와 최신 ComfyUI-ShortsRemake가 "
                           "있어야 합니다 (ComfyUI를 재시작했는지도 확인하세요).")
        with open(os.path.join(TEMPLATES, "scenario_gguf.api.json"), encoding="utf-8") as f:
            prompt = json.load(f)
        prompt["130"]["inputs"]["image"] = (uploaded["subfolder"] + "/" if uploaded["subfolder"] else "") + uploaded["filename"]
        q = prompt["111"]["inputs"]
        q.update(prompt=instruction, seed=random.randint(1, 2 ** 31),
                 model_name=p.get("gguf_model") if p.get("gguf_model") in models else models[0])
        await self.hand_over("gguf", job)
        outputs = await self.run_prompt(job, prompt)
        text = (outputs.get("114") or {}).get("text") or []
        return text[0] if text and isinstance(text[0], str) else ""

    async def analyze_node(self, job, uploaded, instruction, p):
        with open(os.path.join(TEMPLATES, "scenario_qwenvl.api.json"), encoding="utf-8") as f:
            prompt = json.load(f)
        prompt["130"]["inputs"]["image"] = (uploaded["subfolder"] + "/" if uploaded["subfolder"] else "") + uploaded["filename"]
        q = prompt["111"]["inputs"]
        q["custom_prompt"] = instruction
        q["seed"] = random.randint(1, 2 ** 31)
        if p.get("node_model"):
            q["model_name"] = p["node_model"]
        await self.hand_over("node", job)
        outputs = await self.run_prompt(job, prompt)
        text = (outputs.get("114") or {}).get("text") or []
        return text[0] if text and isinstance(text[0], str) else ""

    # ---- stage 3: video ----------------------------------------------------------------------
    async def svi_api(self):
        """The SVI workflow as an API prompt: templates/video_svi.api.json (exported from the UI) when it exists,
        otherwise the UI workflow converted with ComfyUI's node definitions."""
        override = os.path.join(TEMPLATES, "video_svi.api.json")
        path = override if os.path.isfile(override) else self.path(self.cfg["svi_workflow"])
        if not os.path.isfile(path):
            raise AppError(f"SVI 워크플로우 파일이 없습니다: {path}")
        key = (path, os.path.getmtime(path))
        if self._svi_cache is None or self._svi_cache[0] != key or self.comfy._object_info is None:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            warnings = []
            if not comfy_convert.is_api_format(data):
                data, warnings = comfy_convert.workflow_to_api(data, await self.comfy.object_info())
            self._svi_cache = (key, data, warnings)
        return copy.deepcopy(self._svi_cache[1]), list(self._svi_cache[2])

    @staticmethod
    def svi_slots(api):
        """(start image node id, [prompt node ids in section order], {param node id: node})."""
        loads = [nid for nid, n in api.items() if n["class_type"] == "LoadImage"]
        first = [nid for nid in loads if "1st" in api[nid]["_meta"]["title"].lower()]
        prompts = sorted(((comfy_convert.ordinal_of(n["_meta"]["title"]), nid) for nid, n in api.items()
                          if n["class_type"] == "CLIPTextEncode" and comfy_convert.ordinal_of(n["_meta"]["title"])))
        params = {nid: n for nid, n in api.items()
                  if n["class_type"] in ("INTConstant", "PrimitiveInt", "PrimitiveFloat", "FloatConstant")
                  and not isinstance(n["inputs"].get("value"), list)}
        return (first or loads or [None])[0], [nid for _, nid in prompts], params

    async def video_info(self, backend):
        if backend != "svi":
            return {"backend": backend, "slots": 6, "params": [], "warnings": []}
        api, warnings = await self.svi_api()
        load, prompts, params = self.svi_slots(api)
        if load is None or not prompts:
            warnings.append("워크플로우에서 시작 이미지(Load Image) 또는 '1st_…' 프롬프트 노드를 찾지 못했습니다")
        items = [{"id": nid, "title": n["_meta"]["title"], "value": n["inputs"].get("value"),
                  "float": n["class_type"] in ("PrimitiveFloat", "FloatConstant")} for nid, n in params.items()]
        items.sort(key=lambda p: (comfy_convert.ordinal_of(p["title"]) or 0, p["title"].lower()))
        return {"backend": backend, "slots": len(prompts), "warnings": warnings, "params": items}

    def build_svi(self, api, doc, texts, image_name, p, job):
        load, prompt_nodes, params = self.svi_slots(api)
        if load is None or not prompt_nodes:
            raise AppError("SVI 워크플로우에서 시작 이미지 노드나 '1st_…' 프롬프트 노드를 찾지 못했습니다")
        api[load]["inputs"]["image"] = image_name
        if len(texts) != len(prompt_nodes):
            job.notes.append(f"시나리오 구간 {len(texts)}개, 워크플로우 구간 {len(prompt_nodes)}개: "
                             + ("남는 구간은 마지막 프롬프트를 반복합니다" if len(texts) < len(prompt_nodes) else "뒤쪽 프롬프트는 쓰지 않습니다"))
        for i, nid in enumerate(prompt_nodes):
            api[nid]["inputs"]["text"] = texts[min(i, len(texts) - 1)]
        for nid, value in (p.get("params") or {}).items():
            if nid in params and value not in ("", None):
                api[nid]["inputs"]["value"] = float(value) if params[nid]["class_type"] in ("PrimitiveFloat", "FloatConstant") else int(float(value))
        if p.get("match_size", True):
            by_title = {n["_meta"]["title"].strip().lower(): n for n in params.values()}
            w, h = by_title.get("width"), by_title.get("height")
            if w and h and doc.get("width") and doc.get("height"):
                scale = max(w["inputs"]["value"], h["inputs"]["value"]) / max(doc["width"], doc["height"])
                w["inputs"]["value"] = max(16, round(doc["width"] * scale / 16) * 16)
                h["inputs"]["value"] = max(16, round(doc["height"] * scale / 16) * 16)
                job.notes.append(f"영상 크기 {w['inputs']['value']}x{h['inputs']['value']} (시작 이미지 비율)")
        return api

    @staticmethod
    def build_i2v(api, doc, texts, image_name):
        """Workflow 5 graph with the prompts.json reader replaced by the values themselves."""
        fan = next((nid for nid, n in api.items() if n["class_type"] == "ShortsPromptsFanout"), None)
        load = next((nid for nid, n in api.items() if n["class_type"] == "LoadImage"), None)
        if fan is None or load is None:
            raise AppError("I2V 템플릿에 ShortsPromptsFanout / LoadImage 노드가 없습니다")
        texts = texts[:6]
        slots = texts + [""] * (8 - len(texts))
        values = slots + [doc.get("common_prompt", ""), doc.get("negative_prompt") or DEFAULT_NEGATIVE,
                          len(texts), "\n".join(texts), len(texts)]
        del api[fan]
        joined = {}      # the workflow appends a fixed style text to every segment prompt; here the prompt goes in as is
        for nid, node in api.items():
            for name, v in node["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == fan:
                    node["inputs"][name] = values[v[1]]
                    if node["class_type"] == "StringConcatenate":
                        joined[nid] = values[v[1]]
        for nid in joined:
            del api[nid]
        for node in api.values():
            for name, v in node["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) in joined:
                    node["inputs"][name] = joined[str(v[0])]
        api[load]["inputs"]["image"] = image_name
        return api

    async def stage_video(self, job, p):
        name = p.get("dir") or ""
        doc = self.load_scenario(name)
        texts = [_clean(sg.get("positive_prompt")) for sg in doc.get("segments", [])]
        if not texts or not all(texts):
            raise AppError("비어 있는 구간 프롬프트가 있습니다. 모두 채우고 저장한 뒤 다시 시도하세요")
        if p.get("prepend_common") and doc.get("common_prompt"):
            texts = [doc["common_prompt"].rstrip(". ") + ". " + t for t in texts]
        ref_path = os.path.join(self.scenario_dir(name), "reference.png")
        if not os.path.isfile(ref_path):
            raise AppError(f"시작 이미지가 없습니다: {ref_path}")
        job.step = "시작 이미지를 ComfyUI로 전송"
        with open(ref_path, "rb") as f:
            uploaded = await self.comfy.upload_image(f.read(), upload_name(name))
        image_name = (uploaded["subfolder"] + "/" if uploaded["subfolder"] else "") + uploaded["filename"]

        backend = p.get("backend") or "svi"
        if backend == "svi":
            api, warnings = await self.svi_api()
            job.notes.extend(warnings)
            api = self.build_svi(api, doc, texts, image_name, p, job)
        else:
            path = self.path(self.cfg["i2v_api"])
            if not os.path.isfile(path):
                raise AppError(f"I2V 템플릿이 없습니다: {path}")
            with open(path, encoding="utf-8") as f:
                api = self.build_i2v(json.load(f), doc, texts, image_name)
        if p.get("random_seed", True):
            comfy_convert.randomize_seeds(api)
        comfy_convert.apply_text_replacements(api)
        await self.hand_over("video", job)
        await self.run_prompt(job, api)
        videos = [f for f in job.outputs if f["kind"] == "video"]
        if not videos:
            raise AppError("영상 파일이 만들어지지 않았습니다 (ComfyUI 콘솔을 확인하세요)")
        unseen = await self.find_outputs(job)
        final = [f for f in videos if f["type"] == "output"] or videos
        if self.saved_text(final[-1]):
            job.notes.append("저장: " + self.saved_text(final[-1]))
        if unseen:
            job.notes.append(unseen)
        return {"dir": name, "final": final[-1]}


# ---------------------------------------------------------------------------------------------- #
# HTTP handlers
# ---------------------------------------------------------------------------------------------- #
def json_error(message, status=400):
    return web.json_response({"error": message}, status=status)


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except AppError as e:
        return json_error(str(e))
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        app = request.app["app"]
        return json_error(f"ComfyUI({app.comfy.url})에 연결하지 못했습니다: {type(e).__name__} {e}", 502)


async def index(request):
    return web.FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-cache"})


async def api_status(request):
    app = request.app["app"]
    out = {"comfy_url": app.comfy.url, "comfy_ok": False, "ws_ok": app.comfy.ws_ok, "output_dir": app.output_dir,
           "catalogue": {"node": 0, "gguf": 0},
           "result_dir": app.result_dirs.get(app.comfy.url, ""), "output_source": app.output_source, "recent_urls": app.recent["recent_urls"], "recent_dirs": app.recent["recent_dirs"],
           "running": 0, "pending": 0, "vram": None, "chat": None, "node_models": [], "gguf_models": [],
           "svi_override": os.path.isfile(os.path.join(TEMPLATES, "video_svi.api.json"))}
    try:
        queue = await app.comfy.get_json("/queue", timeout=5)
        out.update(comfy_ok=True, running=len(queue.get("queue_running", [])), pending=len(queue.get("queue_pending", [])))
        stats = await app.comfy.get_json("/system_stats", timeout=5)
        await app.follow_comfy_output((stats.get("system") or {}).get("argv") or [])
        out.update(output_dir=app.output_dir, output_source=app.output_source, result_dir=app.result_dirs.get(app.comfy.url, ""))
        devices = stats.get("devices") or []
        if devices and devices[0].get("vram_total"):
            out["vram"] = {"name": devices[0].get("name", ""), "total": devices[0]["vram_total"], "free": devices[0].get("vram_free", 0)}
        # the nodes list their whole catalogue; only the models that are in the server's models/LLM folder are offered
        found = await app.llm_on_disk()
        chat = await app.chat_models()
        node = await app.combo_values("AILab_QwenVL_Advanced")
        gguf = [m for m in await app.combo_values("ShortsQwenGGUFVision") if not m.startswith("(")]
        out["catalogue"] = {"node": len(node), "gguf": len(gguf)}
        out["chat"] = chat and dict(chat, hf=app.on_disk(found, chat.get("hf") or [], False),
                                    gguf=app.on_disk(found, chat.get("gguf") or [], True))
        out["node_models"] = app.on_disk(found, node, False)
        out["gguf_models"] = app.on_disk(found, gguf, True)
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
        pass
    return web.json_response(out)


async def api_start(request):
    app = request.app["app"]
    kind = request.match_info["kind"]
    params = await request.json()
    stages = {"image": ("이미지 생성", app.stage_image), "scenario": ("시나리오 생성", app.stage_scenario),
              "video": ("영상 생성", app.stage_video)}
    if kind not in stages:
        return json_error("unknown stage", 404)
    label, fn = stages[kind]
    job = app.start_job(kind, label, fn, params)
    return web.json_response({"job": job.to_dict()})


async def api_job(request):
    job = request.app["app"].jobs.get(request.match_info["id"])
    if job is None:
        return json_error("job not found", 404)
    return web.json_response(job.to_dict())


async def api_history(request):
    """Jobs run with this program, newest first: the ones still running, then the finished ones on record."""
    app = request.app["app"]
    live = [app.history_entry(j) for j in sorted(app.jobs.values(), key=lambda j: -j.created) if j.finished is None]
    return web.json_response({"items": live + [app.stored_entry(row) for row in app.store.jobs(300)]})


async def api_cancel(request):
    app = request.app["app"]
    job = app.jobs.get(request.match_info["id"])
    if job is None:
        return json_error("job not found", 404)
    if job.state in ("queued", "running"):
        job.cancel_requested = True
        if job.prompt_id:
            await app.comfy.post_json("/queue", {"delete": [job.prompt_id]})
            await app.comfy.post_json("/interrupt", {"prompt_id": job.prompt_id})
        else:
            job.task.cancel()
    return web.json_response(job.to_dict())


async def api_comfy(request):
    """Change the ComfyUI address from the page."""
    app = request.app["app"]
    url = ((await request.json()).get("url") or "").strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    if not re.match(r"^https?://[^/\s]+$", url):
        raise AppError("주소는 http://호스트:포트 형식으로 입력하세요 (예: http://127.0.0.1:8188, http://192.168.0.10:8188)")
    if app.lock.locked():
        raise AppError("실행 중인 작업이 있어 지금은 주소를 바꿀 수 없습니다")
    await app.set_comfy(url)
    try:
        stats = await app.comfy.get_json("/system_stats", timeout=6)
        device = ((stats.get("devices") or [{}])[0]).get("name", "")
        app.keep_recent("recent_urls", url)
        return web.json_response({"url": url, "ok": True, "message": "연결했습니다" + (f" ({device})" if device else "")})
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
        return web.json_response({"url": url, "ok": False,
                                  "message": f"주소를 바꿨지만 연결되지 않습니다: {type(e).__name__}. ComfyUI가 켜져 있는지, "
                                             "다른 PC라면 --listen 0.0.0.0 으로 실행했는지 확인하세요"})


async def api_result_dir(request):
    """Pick the result folder for the ComfyUI in use from the page (empty: find it)."""
    app = request.app["app"]
    path = ((await request.json()).get("path") or "").strip().strip('"')
    if app.output_source == "config":
        raise AppError("서버 결과 폴더가 config.json의 output_dir (또는 --output-dir)로 고정되어 있습니다")
    if app.lock.locked():
        raise AppError("실행 중인 작업이 있어 지금은 폴더를 바꿀 수 없습니다")
    if not path:
        app.set_result_dir("")
        return web.json_response({"path": "", "message": "서버 결과 폴더를 자동으로 찾습니다"})
    if not os.path.isabs(path):
        raise AppError("전체 경로로 입력하세요 (예: V:\\output)")
    path = os.path.normpath(path)
    if not await asyncio.to_thread(os.path.isdir, path):
        raise AppError(f"폴더가 없습니다: {path}")
    used = await asyncio.to_thread(output_under, path, app.comfy_output)
    app.set_result_dir(used)
    return web.json_response({"path": used, "message": f"{path} 안의 서버 출력 폴더 {used} 을(를) 씁니다" if used != path
                              else f"서버 결과 폴더: {used}"})


async def api_folders(request):
    """The folder picker: subfolders of ?path= on the PC this program runs on (no path: its drives)."""
    path = request.query.get("path", "").strip().strip('"')
    if path and not os.path.isabs(path):
        raise AppError("전체 경로로 입력하세요 (예: D:\\videos)")
    try:
        return web.json_response(await asyncio.to_thread(list_folders, path))
    except OSError as e:
        raise AppError(f"폴더를 열 수 없습니다: {e.strerror or e}")


async def api_free(request):
    app = request.app["app"]
    await app.comfy.post_json("/qwenvl/chat/unload", {"backend": "all"}, timeout=60)
    await app.comfy.post_json("/free", {"unload_models": True, "free_memory": True})
    app.last_stage = "freed"
    return web.json_response({"ok": True})


async def api_upload(request):
    app = request.app["app"]
    field = await (await request.multipart()).next()
    if field is None or field.name != "image":
        return json_error("image 필드가 없습니다")
    data = await field.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except OSError:
        return json_error("이미지 파일이 아닙니다")
    png = io.BytesIO()
    image.save(png, "PNG")
    ref = await app.comfy.upload_image(png.getvalue(), time.strftime("%Y%m%d_%H%M%S") + "_upload.png")
    return web.json_response({"image": ref, "width": image.width, "height": image.height})


async def api_scenarios(request):
    return web.json_response({"scenarios": request.app["app"].list_scenarios()})


async def api_scenario_get(request):
    app = request.app["app"]
    name = request.query.get("dir", "")
    return web.json_response({"dir": name, "scenario": app.load_scenario(name)})


async def api_scenario_save(request):
    app = request.app["app"]
    body = await request.json()
    name = body.get("dir", "")
    doc = app.load_scenario(name)
    edit = body.get("scenario") or {}
    for key in ("summary_ko", "common_prompt", "negative_prompt"):
        if isinstance(edit.get(key), str):
            doc[key] = edit[key].strip()
    texts = edit.get("prompts")
    if isinstance(texts, list):
        seconds = float(doc.get("segment_length") or 5)
        old = doc.get("segments", [])
        doc["segments"] = []
        for i, text in enumerate(texts[:8]):
            sg = old[i] if i < len(old) else {"motion": "", "camera": "", "scene_ko": ""}
            start, end = round(i * seconds, 3), round((i + 1) * seconds, 3)
            sg.update(index=i + 1, start=start, end=end, duration=round(seconds, 3),
                      label=f"{_fmt_time(start)} - {_fmt_time(end)}", positive_prompt=_clean(text))
            doc["segments"].append(sg)
        doc["duration"] = round(len(doc["segments"]) * seconds, 3)
    app.save_scenario(name, doc)
    return web.json_response({"dir": name, "scenario": doc})


async def api_scenario_image(request):
    app = request.app["app"]
    path = os.path.join(app.scenario_dir(request.query.get("dir", "")), "reference.png")
    if not os.path.isfile(path):
        return json_error("not found", 404)
    return web.FileResponse(path, headers={"Cache-Control": "no-cache"})


async def api_video_info(request):
    return web.json_response(await request.app["app"].video_info(request.query.get("backend", "svi")))


# ---- library ---------------------------------------------------------------------------------------
def _library(request):
    return request.app["app"].library


async def api_library(request):
    lib, q = _library(request), request.query
    await asyncio.to_thread(lib.scan)
    rows, workflows, tags = lib.listing(q.get("sort", "new"), q.get("workflow", ""), q.get("q", ""), q.get("kind", ""),
                                        q.get("sidecars") == "1", q.get("fav") == "1", q.get("tag", ""))
    return web.json_response({"items": rows[:1000], "total": len(rows), "workflows": workflows, "tags": tags,
                              "folder": lib.roots["output"]})


async def api_library_item(request):
    lib, ref = _library(request), request.query.get("ref", "")
    try:
        full = lib.resolve(ref)
        info = await asyncio.to_thread(library.describe, full, lib.catalogue)
    except ValueError as e:
        raise AppError(str(e))
    stat = os.stat(full)
    info.pop("types")
    info.update(ref=ref, name=os.path.basename(full), path=full, size=stat.st_size, mtime=stat.st_mtime,
                kind="video" if full.lower().endswith(library.VIDEO_EXT) else "image", job=None, mark=None)
    if ref.startswith("output:"):      # what the store knows about it: the job that made it, the user's tags and note
        app, rel = request.app["app"], lib.rel_of(ref)
        row = app.store.job_of(lib.root, rel)
        if row:
            info["job"] = app.stored_entry(row)
            info["job"]["links"] = [x for x in info["job"]["links"] if x.get("ref", "").lower() != ("output:" + rel).lower()]
        info["mark"] = app.store.mark(lib.root, rel)
    return web.json_response(info)


async def api_library_mark(request):
    """Favourite, tags and note of a file: only what is given is changed."""
    lib, body = _library(request), await request.json()
    try:
        rel = lib.rel_of(body.get("ref", ""))
    except ValueError as e:
        raise AppError(str(e))
    tags = body.get("tags")
    if tags is not None:      # a list or 'a, b, c'; '#' in front is dropped, no doubles, the order typed is kept
        if isinstance(tags, str):
            tags = re.split(r"[,\n]", tags)
        tags = list(dict.fromkeys(t.strip().lstrip("#").strip()[:40] for t in tags if isinstance(t, str) and t.strip().lstrip("#").strip()))[:30]
    note = body.get("note")
    return web.json_response(request.app["app"].store.set_mark(
        lib.root, rel, body.get("favorite"), tags, None if note is None else str(note).strip()[:4000]))


async def api_library_file(request):
    try:
        full = _library(request).resolve(request.query.get("ref", ""))
    except ValueError as e:
        return json_error(str(e), 404)
    return web.FileResponse(full)


async def api_library_thumb(request):
    lib = _library(request)
    try:
        path = await asyncio.to_thread(lib.thumbnail, request.query.get("ref", ""))
    except (ValueError, OSError, StopIteration, av.FFmpegError) as e:
        return json_error(str(e), 404)
    return web.FileResponse(path, headers={"Cache-Control": "max-age=86400"})


async def api_library_frame(request):
    """The first (?which=first) or last (?which=last) frame of a video, as a PNG."""
    lib, which = _library(request), request.query.get("which", "first")
    try:
        path = await asyncio.to_thread(lib.frame, request.query.get("ref", ""), "last" if which == "last" else "first")
    except (ValueError, OSError, StopIteration, AttributeError, IndexError, av.FFmpegError) as e:
        return json_error(str(e) or "프레임을 뽑지 못했습니다", 404)
    return web.FileResponse(path, headers={"Cache-Control": "max-age=86400"})


async def api_library_workflow(request):
    """The workflow embedded in the file, as a download ComfyUI can open."""
    lib, ref = _library(request), request.query.get("ref", "")
    try:
        workflow = await asyncio.to_thread(lib.embedded_workflow, ref)
    except ValueError as e:
        return json_error(str(e), 404)
    if workflow is None:
        return json_error("이 파일에는 워크플로우가 들어 있지 않습니다", 404)
    name = os.path.splitext(os.path.basename(ref.partition(":")[2]))[0] + ".json"
    return web.Response(text=json.dumps(workflow, ensure_ascii=False), content_type="application/json",
                        headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(name)})


async def api_library_save_workflow(request):
    """Write the embedded workflow into workflow/ so ComfyUI lists it under a real name."""
    lib, body = _library(request), await request.json()
    try:
        workflow = await asyncio.to_thread(lib.embedded_workflow, body.get("ref", ""))
    except ValueError as e:
        raise AppError(str(e))
    if workflow is None:
        raise AppError("이 파일에는 워크플로우가 들어 있지 않습니다")
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", body.get("name") or "").strip().strip(".")
    if not name:
        raise AppError("저장할 이름을 입력하세요")
    folder = os.path.join(ROOT, "workflow")
    path = os.path.join(folder, name + ".json")
    if os.path.exists(path):
        raise AppError(f"workflow 폴더에 같은 이름이 이미 있습니다: {name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False)
    return web.json_response({"saved": path, "name": name})


async def api_library_rename(request):
    lib, body = _library(request), await request.json()
    try:
        ref = await asyncio.to_thread(lib.rename, body.get("ref", ""), body.get("name", ""))
    except ValueError as e:
        raise AppError(str(e))
    except OSError as e:
        raise AppError(f"이름을 바꾸지 못했습니다: {e.strerror or e}")
    return web.json_response({"ref": ref, "name": os.path.basename(ref.partition(":")[2])})


async def api_library_delete(request):
    lib, body = _library(request), await request.json()
    try:
        removed = await asyncio.to_thread(lib.delete, body.get("ref", ""))
    except ValueError as e:
        raise AppError(str(e))
    except OSError as e:
        raise AppError(f"삭제하지 못했습니다: {e.strerror or e}")
    return web.json_response({"removed": removed})


async def api_inspect(request):
    """A file dropped on the page: keep a copy under data/dropped and describe it like a library item."""
    lib = _library(request)
    field = await (await request.multipart()).next()
    if field is None or field.name != "file":
        return json_error("file 필드가 없습니다")
    data = bytes(await field.read())
    try:
        ref = lib.add_dropped(field.filename or "file", data)
    except ValueError as e:
        raise AppError(str(e))
    return web.json_response({"ref": ref, "name": unquote(field.filename or "")})


async def _proxy(request, path, params):
    app = request.app["app"]
    headers = {"Range": request.headers["Range"]} if "Range" in request.headers else {}
    async with app.comfy.session.get(app.comfy.url + path, params=params, headers=headers) as r:
        resp = web.StreamResponse(status=r.status)
        for h in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
            if h in r.headers:
                resp.headers[h] = r.headers[h]
        await resp.prepare(request)
        try:
            async for chunk in r.content.iter_chunked(1 << 16):
                await resp.write(chunk)
            await resp.write_eof()
        except (ConnectionResetError, aiohttp.ClientConnectionError):
            pass      # the browser stopped the download (seeking in a video does this)
        return resp


async def api_view(request):
    params = {k: request.query.get(k, "") for k in ("filename", "subfolder", "type")}
    path = "/vhs/viewvideo" if request.query.get("transcode") else "/view"
    return await _proxy(request, path, params)


async def on_startup(web_app):
    app = web_app["app"]
    await app.comfy.start()
    web_app["ws_task"] = asyncio.create_task(app.ws_loop())
    if app.cfg.get("open"):
        host = "127.0.0.1" if app.cfg["host"] in ("0.0.0.0", "::") else app.cfg["host"]
        asyncio.get_running_loop().call_later(1.0, webbrowser.open, f"http://{host}:{app.cfg['port']}")


async def on_cleanup(web_app):
    web_app["ws_task"].cancel()
    await web_app["app"].comfy.close()


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    path = os.path.join(HERE, "config.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg.update({k: v for k, v in read_settings().items() if k == "comfy_url"})
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=cfg["host"])
    ap.add_argument("--port", type=int, default=cfg["port"])
    ap.add_argument("--comfy", default=cfg["comfy_url"], help="ComfyUI address, e.g. http://127.0.0.1:8188")
    ap.add_argument("--output-dir", default=cfg["output_dir"],
                    help="folder of this PC that shows what ComfyUI saves (default: picked on the page, else found)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser once the server is up")
    args = ap.parse_args()
    cfg.update(host=args.host, port=args.port, comfy_url=args.comfy, output_dir=args.output_dir, open=args.open)
    return cfg


def make_app(cfg):
    web_app = web.Application(middlewares=[errors], client_max_size=4 * 1024 ** 3)      # dropped videos can be large
    web_app["app"] = App(cfg)
    web_app.add_routes([
        web.get("/", index),
        web.get("/api/status", api_status),
        web.post("/api/start/{kind}", api_start),
        web.get("/api/job/{id}", api_job),
        web.post("/api/job/{id}/cancel", api_cancel),
        web.post("/api/free", api_free),
        web.post("/api/comfy", api_comfy),
        web.post("/api/result_dir", api_result_dir),
        web.get("/api/folders", api_folders),
        web.post("/api/upload", api_upload),
        web.get("/api/scenarios", api_scenarios),
        web.get("/api/scenario", api_scenario_get),
        web.post("/api/scenario", api_scenario_save),
        web.get("/api/scenario/image", api_scenario_image),
        web.get("/api/video/info", api_video_info),
        web.get("/api/view", api_view),
        web.get("/api/library", api_library),
        web.get("/api/library/item", api_library_item),
        web.get("/api/library/file", api_library_file),
        web.get("/api/library/thumb", api_library_thumb),
        web.get("/api/library/frame", api_library_frame),
        web.get("/api/library/workflow", api_library_workflow),
        web.post("/api/library/save_workflow", api_library_save_workflow),
        web.post("/api/library/mark", api_library_mark),
        web.post("/api/library/rename", api_library_rename),
        web.post("/api/library/delete", api_library_delete),
        web.get("/api/history", api_history),
        web.post("/api/inspect", api_inspect),
    ])
    web_app.on_startup.append(on_startup)
    web_app.on_cleanup.append(on_cleanup)
    return web_app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    config = load_config()
    shown = "127.0.0.1" if config["host"] in ("0.0.0.0", "::") else config["host"]
    print(f"\n  ComfyUI-Manager:  http://{shown}:{config['port']}\n  ComfyUI:          {config['comfy_url']}\n")
    web.run_app(make_app(config), host=config["host"], port=config["port"], print=None)
