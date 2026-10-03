"""ComfyUI-Manager (this repo's tool, not the node-pack manager): make and keep track of images and videos.

Create: z-image turbo image -> Qwen scenario (N prompts) -> Wan 2.2 video. Nothing is generated here; every
step is queued on a running ComfyUI through its HTTP API (/prompt, /history, /view, /upload/image, /free, /ws).
Library: lists the output folder and shows, for any image or video, the workflow and prompts it was made with
(read from the metadata ComfyUI embeds in the file); works without ComfyUI running.
History: the works the user recorded after finishing a video; one opens into the create steps with the image and
prompts it was made with. Every job that ran is also kept (not shown as a list), which is what connects a file of the
library to the job that made it. All of it, with the library's index and the user's tags / notes, lives in one
SQLite file (data/manager.db, see store.py).

A small aiohttp server with a one-page UI; it only needs packages ComfyUI already ships (aiohttp, Pillow, PyAV).

    python_embeded\\python.exe tools\\ComfyUI-Manager\\app.py [--port 8288] [--comfy http://127.0.0.1:8188]
"""
import argparse
import asyncio
import base64
import copy
import ctypes
import hashlib
import io
import json
import logging
import logging.handlers
import ntpath
import os
import posixpath
import random
import re
import shutil
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
import models  # noqa: E402
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
    "image_workflow": "image_z_image_turbo.json",       # the image workflow picked at first
    "scenario_dir": "",                                   # empty = the folder 'scenario' next to the result folder
    "model_dir": "",                                      # empty = the folder 'model' next to the result folder
    "close_with_page": True,                              # stop when the last page in a browser has been closed (false: a server that keeps running)
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
    plain = "".join(c for c in scenario_dir if c.isascii() and (c.isalnum() or c == "_")).strip("_")
    # the hash keeps two folders apart whose names differ only in what was left out (Korean names)
    return (plain or "scenario") + "_" + hashlib.sha1(scenario_dir.encode("utf-8")).hexdigest()[:8] + ".png"


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
        self._workflows = {}             # name -> ((file, modified, node definitions), API prompt, warnings)
        self._wf_list = (0, {})          # when asked, {name: {'modified', 'created'}} of the server's workflow folder
        self._wf_files = {}              # name -> (modified, content)
        self._catalogue_task = None
        self._adopted = set()            # result folders whose old scenario folders were moved to the scenario folder
        self._chat_models = (0, None)
        self._combos = {}
        self._llm = (0, None)
        self._listed = (0, None)         # when asked, the relative names of every model ComfyUI lists
        self.ws = None
        self.address_changed = asyncio.Event()
        self.store = store.Store(os.path.join(DATA, "manager.db"))
        self.library = library.Library(self.output_dir, DATA, library.Catalogue(TEMPLATES), self.store)
        self.models = models.Models(lambda: self.model_root, self.store, self.library)
        self.copy_job = None             # the copy of the model menu that is running or was run last
        self.pages = {}                  # pages open in browsers: id -> when it last called in
        self.pages_gone_since = None     # when the last page went away (None while one is open or none was seen yet)
        self.use_catalogue()
        self.import_history(os.path.join(DATA, "history.json"))

    # ---- helpers -----------------------------------------------------------------------------
    def path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)

    @property
    def scenario_root(self):
        """Where the scenarios are kept: a folder of their own next to the result folder (V:\\output -> V:\\scenario), so
        they are not mixed with what ComfyUI saves. config.json's scenario_dir names another place."""
        if self.cfg.get("scenario_dir"):
            return self.cfg["scenario_dir"]
        output = os.path.normpath(self.output_dir)
        parent = os.path.dirname(output)
        return os.path.join(parent if parent and parent != output else output, "scenario")

    @property
    def model_root(self):
        """Where the models are: Easy-Install's top-level `model` next to the result folder (V:\\output -> V:\\model);
        ComfyUI's own `ComfyUI\\models` when there is no such folder. config.json's model_dir names another place."""
        if self.cfg.get("model_dir"):
            return self.cfg["model_dir"]
        output = os.path.normpath(self.output_dir)
        parent = os.path.dirname(output)
        parent = parent if parent and parent != output else output
        own = os.path.join(parent, "model")
        return own if os.path.isdir(own) else os.path.join(parent, "ComfyUI", "models")

    async def listed_models(self):
        """(the model folder names ComfyUI has, the relative names of the models it lists over all of them), both in
        lower case with '/', kept for a minute. None when ComfyUI cannot be asked."""
        stamp, found = self._listed
        if time.time() - stamp > 60:
            try:
                folders = [f for f in await self.comfy.get_json("/models", timeout=10) if isinstance(f, str)]
                lists = await asyncio.gather(*(self.comfy.get_json(f"/models/{f}", timeout=20) for f in folders), return_exceptions=True)
                found = ({f.lower() for f in folders} | {"unet", "clip"},      # the old names of diffusion_models and text_encoders
                         {n.replace("\\", "/").lower() for one in lists if isinstance(one, list) for n in one if isinstance(n, str)})
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError):
                found = None
            self._listed = (time.time(), found)
        return found

    def adopt_scenarios(self):
        """Scenario folders an earlier version made inside the result folder ('<name>_prompts' holding a prompts.json)
        are moved to the scenario folder, once per result folder; a name already there is left where it is."""
        key = os.path.normcase(os.path.normpath(self.output_dir))
        if key in self._adopted or not os.path.isdir(self.output_dir):
            return
        self._adopted.add(key)
        root = self.scenario_root
        if os.path.normcase(os.path.normpath(root)) == key:
            return
        for name in os.listdir(self.output_dir):
            source = os.path.join(self.output_dir, name)
            if name.endswith("_prompts") and os.path.isfile(os.path.join(source, "prompts.json")) and not os.path.exists(os.path.join(root, name)):
                try:      # only this program's own: the Shorts nodes keep their '<video>_prompts' folders in the output folder
                    with open(os.path.join(source, "prompts.json"), encoding="utf-8") as f:
                        if json.load(f).get("source") != "comfyui_manager":
                            continue
                except (OSError, ValueError, AttributeError):
                    continue
                try:
                    os.makedirs(root, exist_ok=True)
                    shutil.move(source, os.path.join(root, name))
                    log.info("scenario folder moved: %s -> %s", source, root)
                except OSError:
                    log.exception("scenario folder not moved: %s", source)

    def scenario_dir(self, name):
        if not name or name != os.path.basename(name) or name in (".", ".."):
            raise AppError("잘못된 시나리오 폴더 이름입니다")
        self.adopt_scenarios()
        return os.path.join(self.scenario_root, name)

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
        """Every scenario of the scenario folder, newest first, with what its list card shows."""
        self.adopt_scenarios()
        root, found, uses = self.scenario_root, [], self.store.scenario_uses()
        for name in os.listdir(root) if os.path.isdir(root) else []:
            path = os.path.join(root, name, "prompts.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            texts = [str(sg.get("positive_prompt") or "") for sg in doc.get("segments") or []]
            found.append({"dir": name, "name": re.sub(r"_prompts$", "", name), "mtime": int(os.path.getmtime(path)),
                          "segments": len(texts), "empty": sum(1 for t in texts if not t.strip()),
                          "summary": str(doc.get("summary_ko") or ""), "snippet": next((t for t in texts if t.strip()), "")[:200],
                          "source": str(doc.get("source_file") or ""), "works": uses.get(name, 0),
                          "search": (name + " " + str(doc.get("summary_ko") or "") + " " + " ".join(texts)).lower()[:6000]})
        return sorted(found, key=lambda s: -s["mtime"])

    def scenario_thumb(self, name, size=360):
        """A small copy of the scenario's start image for the list (kept in data/thumbs)."""
        path = os.path.join(self.scenario_dir(name), "reference.png")
        stat = os.stat(path)
        out = os.path.join(DATA, "thumbs", hashlib.sha1(f"{path}|{stat.st_mtime}|{stat.st_size}|{size}".encode()).hexdigest() + ".jpg")
        if not os.path.isfile(out):
            image = Image.open(path).convert("RGB")
            image.thumbnail((size, size))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            image.save(out, "JPEG", quality=85)
        return out

    def rename_scenario(self, name, new_name):
        """Another name for a scenario folder; the jobs and recorded works made from it follow."""
        wanted = sanitize_name(new_name) + "_prompts"
        if not (new_name or "").strip():
            raise AppError("새 이름을 입력하세요")
        source, target = self.scenario_dir(name), self.scenario_dir(wanted)
        if not os.path.isfile(os.path.join(source, "prompts.json")):
            raise AppError(f"시나리오가 없습니다: {name}")
        if wanted == name:
            return name
        if os.path.exists(target) and os.path.normcase(source) != os.path.normcase(target):
            raise AppError(f"같은 이름의 시나리오가 이미 있습니다: {wanted[:-len('_prompts')]}")
        os.rename(source, target)
        self.store.rename_scenario(name, wanted)
        return wanted

    def delete_scenario(self, name):
        """Remove a scenario folder for good (its prompts and its copy of the start image)."""
        folder = self.scenario_dir(name)
        if not os.path.isfile(os.path.join(folder, "prompts.json")):
            raise AppError(f"시나리오가 없습니다: {name}")
        shutil.rmtree(folder)

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
                    message = str(data.get("exception_message", ""))
                    if "CUDA error: invalid argument" in message:      # not this node: nothing big loads to the GPU any more
                        message = ("ComfyUI 서버의 GPU 메모리 상태가 깨져 큰 모델을 GPU에 올리지 못합니다 (CUDA error: invalid argument). "
                                   "이 상태에서는 다시 실행해도 같은 오류가 나니 ComfyUI를 재시작하세요 (설정의 'ComfyUI 재시작'). 영상 작업 뒤에 반복되면 "
                                   "서버의 ComfyUI 실행 옵션에 --disable-pinned-memory 가 있는지 확인하세요 (Start_ComfyUI_L40S.bat).")
                    raise AppError(f"[{data.get('node_id')} {title}] {data.get('exception_type', '')}: {message}")
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
        self._combos, self._chat_models, self._workflows, self.last_stage = {}, (0, None), {}, None
        self._llm, self._wf_list, self._wf_files, self._listed = (0, None), (0, {}), {}, (0, None)
        self.remember(comfy_url=url)
        self.use_catalogue()
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
        """Where a saved file is, as this PC reaches it (the result folder); only when it cannot be seen from here,
        the path on ComfyUI's own machine."""
        if f.get("saved") or f["type"] != "output" or not self.comfy_output:
            return f.get("saved", "")
        return path_module(self.comfy_output).join(self.comfy_output, *re.split(r"[\\/]", f["subfolder"]), f["filename"])

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

    def with_refs(self, row):
        """A job of the store with its files where they are now (renamed ones followed, deleted ones without ref)."""
        entry, here = row["entry"], row["root"] == store.root_key(self.output_dir)
        for n, f in enumerate(entry.get("outputs") or []):
            f["ref"] = "output:" + row["files"][n] if here and n in row["files"] else ""
        return entry

    def stored_entry(self, row):
        """A job of the store as the page shows it: its files where they are now, and what it is connected to."""
        entry = self.with_refs(row)
        entry["links"] = self.links(row)
        return entry

    # ---- works: a finished image -> scenario -> video the user recorded ---------------------------------
    def record_work(self, body):
        """Keep a finished video with everything it was made from, so it can be opened into the steps again."""
        job_id = str(body.get("video_job") or "")
        job = self.jobs.get(job_id)
        row = self.store.job(job_id)
        entry = row["entry"] if row else self.history_entry(job) if job else None
        if not entry or entry["kind"] != "video" or entry["state"] != "done":
            raise AppError("영상까지 끝난 작업만 기록할 수 있습니다")
        result, params = entry["result"], entry["params"]
        doc = result.get("scenario") or self.load_scenario(result["dir"])
        ref = doc.get("image_ref") or {}
        # what the page knows about the start image (prompt, workflow, seed), when it made it in this session
        made = (body.get("images") or {}).get(ref.get("filename") or "") or {}
        text = lambda v, n=4000: str(v if v is not None else "")[:n]
        work = {
            "id": uuid.uuid4().hex[:12], "time": time.time(),
            "title": text(doc.get("summary_ko") or made.get("prompt") or result["dir"], 200),
            "image": {"ref": ref, "workflow": text(made.get("workflow"), 300), "label": text(made.get("label"), 300),
                      "prompt": text(made.get("prompt")), "size": text(made.get("size"), 20), "steps": text(made.get("steps"), 10),
                      "seed": made.get("seed") or 0, "info": text(made.get("info"), 600)},
            "scenario": {"dir": result["dir"], "doc": doc,
                         "form": {k: text(v, 2000) for k, v in (body.get("scenario") or {}).items() if isinstance(k, str)}},
            "video": {"job": entry["id"], "workflow": params.get("workflow") or self.video_workflow(params),
                      "label": result.get("workflow") or "", "final": result["final"], "elapsed": entry.get("elapsed", 0),
                      "options": {k: bool(params.get(k, k != "prepend_common")) for k in ("random_seed", "match_size", "prepend_common")},
                      "params": params.get("params") or {}},
        }
        self.store.add_work(work, store.root_key(self.output_dir))
        return work

    def work_view(self, work):
        """A recorded work for the page: with its final video as the library has it now ('' when it is gone)."""
        row = self.store.job(work["video"].get("job"))
        final, ref = work["video"].get("final") or {}, ""
        if row:
            ref = next((f["ref"] for f in self.with_refs(row)["outputs"]
                        if (f["filename"], f["type"]) == (final.get("filename"), final.get("type"))), "")
            if not ref:      # renamed in the library: the job's files follow, the name kept in the record does not
                ref = next((f["ref"] for f in row["entry"]["outputs"] if f.get("ref") and f["kind"] == "video"), "")
        return dict(work, final_ref=ref)

    async def open_work(self, work):
        """Put the scenario folder back as it was when the work was recorded, so the steps show the prompts it was made
        with. Returns '' or what could not be restored."""
        name, doc = work["scenario"]["dir"], work["scenario"]["doc"]
        folder = self.scenario_dir(name)
        if not os.path.isfile(os.path.join(folder, "reference.png")):      # the folder was deleted: the start image again
            try:
                image = Image.open(io.BytesIO(await self.comfy.view_bytes(work["image"]["ref"]))).convert("RGB")
                os.makedirs(folder, exist_ok=True)
                image.save(os.path.join(folder, "reference.png"), "PNG")
            except (AppError, aiohttp.ClientError, asyncio.TimeoutError, OSError, KeyError, TypeError):
                return ("시나리오 폴더와 시작 이미지가 없어져서 되살리지 못했습니다. 프롬프트는 기록에 남아 있지만 "
                        "이 화면에서 다시 실행하려면 이미지를 새로 만들거나 올려야 합니다")
        self.save_scenario(name, doc)
        return ""


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
    # ---- workflows: which files the steps can run, and where each takes its inputs -------------------
    BUILTIN_IMAGE = "기본 (z-image turbo)"

    @staticmethod
    def workflow_label(name):
        """A workflow as it is shown: the server's file name without '.json' (so 'x.api' and 'x' stay apart)."""
        return re.sub(r"\.json$", "", name) if name else App.BUILTIN_IMAGE

    def default_video(self):
        """The configured SVI workflow, as a name inside the workflow folder."""
        return re.sub(r"^workflow[\\/]", "", self.cfg["svi_workflow"]).replace("\\", "/")

    # The workflows are the ones the connected ComfyUI keeps (its user/default/workflows folder, read through its
    # userdata API), not files of this PC: what is saved on the server is what the lists show.
    async def server_workflows(self):
        """{name: {'modified', 'created'}} of the server's workflow files ('sub/name.json'); asked every few seconds at most."""
        stamp, files = self._wf_list
        if time.time() - stamp > 5:
            raw = await self.comfy.get_json("/api/userdata", timeout=20, dir="workflows", recurse="true", split="false", full_info="true")
            files = {str(i["path"]).replace("\\", "/"): {"modified": i.get("modified") or 0, "created": i.get("created") or 0}
                     for i in raw if isinstance(i, dict) and str(i.get("path", "")).lower().endswith(".json")}
            self._wf_list = (time.time(), files)
        return files

    async def workflow_file(self, name):
        """The content of one of the server's workflow files (kept until the server says it changed)."""
        files = await self.server_workflows()
        if name not in files:
            raise AppError(f"서버에 그 워크플로우가 없습니다: {name}")
        cached = self._wf_files.get(name)
        if cached is None or cached[0] != files[name]["modified"]:
            url = self.comfy.url + "/api/userdata/" + quote("workflows/" + name, safe="")
            async with self.comfy.session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status != 200:
                    raise AppError(f"서버에서 워크플로우를 읽지 못했습니다 ({r.status}): {name}")
                try:
                    data = json.loads(await r.read())
                except ValueError:
                    raise AppError(f"워크플로우 파일이 아닙니다: {name}")
            cached = self._wf_files[name] = (files[name]["modified"], data)
        return cached[1]

    async def workflow_names(self):
        """The server's workflow files, every one, in name order."""
        return sorted(await self.server_workflows(), key=str.lower)

    async def workflow_api(self, name):
        """(API prompt, warnings) of a workflow: '' is the built-in image template, anything else one of the server's
        workflow files. A workflow saved from the UI is converted with that ComfyUI's node definitions."""
        override = os.path.join(TEMPLATES, "video_svi.api.json")      # an exact export of the SVI workflow, when given
        local = os.path.join(TEMPLATES, "image_zimage.api.json") if not name else override if (
            name == self.default_video() and os.path.isfile(override)) else None
        if local:
            with open(local, encoding="utf-8") as f:
                data, stamp = json.load(f), os.path.getmtime(local)
        else:
            data, stamp = await self.workflow_file(name), (await self.server_workflows())[name]["modified"]
        if not isinstance(data, dict):
            raise AppError(f"워크플로우 파일이 아닙니다: {name}")
        ui = not comfy_convert.is_api_format(data)
        key = (local, stamp, id(await self.comfy.object_info()) if ui else 0)
        cached = self._workflows.get(name)
        if cached is None or cached[0] != key:
            warnings = []
            if ui:
                data, warnings = comfy_convert.workflow_to_api(data, await self.comfy.object_info())
            cached = self._workflows[name] = (key, data, warnings)
        return copy.deepcopy(cached[1]), list(cached[2])

    @staticmethod
    def usable(api, kind):
        """'' when the step can run this workflow, else why not."""
        slots, classes = comfy_convert.find_slots(api), {n["class_type"] for n in api.values()}
        if not slots["prompts"]:
            return "프롬프트 노드(CLIP Text Encode)를 찾지 못했습니다"
        if kind == "image":
            return ("영상을 만드는 워크플로우입니다" if slots["video"] else "입력 이미지가 필요한 워크플로우입니다" if slots["loads"]
                    else "" if slots["saves"] else "이미지를 저장하는 노드(Save Image)가 없습니다")
        if not slots["video"]:
            return "영상을 저장하는 노드가 없습니다"
        if any("LoadVideo" in c for c in classes):
            return "원본 영상이 필요한 워크플로우입니다"
        return "" if slots["load"] else "시작 이미지 노드(Load Image)가 없습니다"

    async def workflows(self, kind):
        """The workflows the server keeps, all of them and under their names there, for the list of the image / video
        step. 'reason' says why the step cannot run one (it has nowhere to put the prompt or the start image)."""
        wanted = self.cfg["image_workflow"] if kind == "image" else self.default_video()
        try:
            names = await self.workflow_names()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError):
            return {"items": [], "default": wanted, "connected": False}
        items = []
        for name in names:
            try:
                api, warnings = await self.workflow_api(name)
                items.append({"name": name, "label": self.workflow_label(name),
                              "segments": 6 if self.has_fanout(api) else len(comfy_convert.find_slots(api)["prompts"]),
                              "missing": sorted({w.split("'")[1] for w in warnings if "unknown type" in w}),
                              "reason": self.usable(api, kind)})
            except (AppError, aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError, AttributeError, IndexError):
                items.append({"name": name, "label": self.workflow_label(name), "segments": 0, "missing": [],
                              "reason": "워크플로우로 읽지 못한 파일입니다"})
        return {"items": items, "default": wanted, "connected": True}

    # ---- home: what the connected ComfyUI can make right now -----------------------------------------
    BASE_FOLDERS = ("checkpoints", "diffusion_models", "unet_gguf")      # ComfyUI's lists of the models that make the picture

    @staticmethod
    def workflow_kind(api):
        """(group, step) of a workflow: group 'image' / 'video_sound' / 'video' / 'other' is what it makes, step is the
        step of 새 작업 that can run it ('image', 'video' or '')."""
        slots = comfy_convert.find_slots(api)
        step = "image" if not App.usable(api, "image") else "video" if not App.usable(api, "video") else ""
        if slots["video"]:
            sound = any(library.is_link(n.get("inputs", {}).get("audio")) for n in api.values() if "video" in n["class_type"].lower())
            return ("video_sound" if sound else "video"), step
        return ("image" if slots["saves"] else "other"), step

    async def home(self):
        """The server's workflows by what they make, each with the model files it loads and whether the server has
        them, and the server's models that no workflow loads."""
        try:
            names = await self.workflow_names()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError):
            return {"connected": False, "comfy": self.comfy.url, "workflows": [], "models": []}
        known = await self.listed_models()
        listed = known[1] if known else None

        def have(name):
            return None if listed is None else name.replace("\\", "/").lower() in listed

        items, loaded = [], {}
        for name in names:
            item = {"name": name, "label": self.workflow_label(name), "group": "other", "step": "", "start_image": False,
                    "models": [], "loras": [], "missing_nodes": [], "error": ""}
            try:
                api, warnings = await self.workflow_api(name)
                item["group"], item["step"] = self.workflow_kind(api)
                item["start_image"] = bool(comfy_convert.find_slots(api)["loads"])
                models, loras = library.GraphReader(api).models()
                item["models"] = [{"kind": m["kind"], "name": m["name"], "present": have(m["name"])} for m in models]
                item["loras"] = [{"name": l["name"], "strength": l.get("strength"), "present": have(l["name"])} for l in loras]
                item["missing_nodes"] = sorted({w.split("'")[1] for w in warnings if "unknown type" in w})
                for m in models + loras:
                    loaded.setdefault(m["name"].replace("\\", "/").lower(), []).append(item["label"])
            except (AppError, aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError, AttributeError, IndexError):
                item["error"] = "워크플로우로 읽지 못한 파일입니다"
            items.append(item)
        lists = await asyncio.gather(*(self.comfy.get_json(f"/models/{f}", timeout=20) for f in self.BASE_FOLDERS), return_exceptions=True)
        base, seen = [], set()
        for folder, one in zip(self.BASE_FOLDERS, lists):
            for model in one if isinstance(one, list) else []:
                key = str(model).replace("\\", "/").lower()
                if isinstance(model, str) and key not in seen:      # diffusion_models and unet_gguf look into the same folders
                    seen.add(key)
                    base.append({"folder": folder, "name": model, "workflows": loaded.get(key, [])})
        return {"connected": True, "comfy": self.comfy.url, "models_listed": listed is not None, "workflows": items, "models": base}

    # ---- the names the library gives: from the server's workflows, remembered in the store ----------
    def use_catalogue(self):
        """What the store remembers about the workflows of the ComfyUI in use (the library names files with it, also
        while that ComfyUI is off)."""
        self.library.catalogue.remote = [entry for _, entry in self.store.workflow_index(self.comfy.url).values() if entry]

    async def refresh_catalogue(self):
        """Read the server's new and changed workflow files and remember what identifies them."""
        files, known = await self.server_workflows(), self.store.workflow_index(self.comfy.url)
        changed = []
        for name, meta in files.items():
            if name in known and known[name][0] == meta["modified"]:
                continue
            try:
                entry = library.catalogue_entry(name, await self.workflow_file(name), meta["created"] / 1000)
            except (AppError, aiohttp.ClientError, asyncio.TimeoutError):
                continue
            changed.append((name, meta["modified"], entry))
        gone = [name for name in known if name not in files]
        if changed or gone:
            self.store.put_workflows(self.comfy.url, changed, gone)
            self.use_catalogue()

    def start_catalogue_refresh(self):
        """From the status poll: at most one refresh at a time, and never in the way of the answer."""
        if self._catalogue_task is None or self._catalogue_task.done():
            async def run():
                try:
                    await self.refresh_catalogue()
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError):
                    pass
            self._catalogue_task = asyncio.create_task(run())

    @staticmethod
    def has_fanout(api):
        return any(n["class_type"] == "ShortsPromptsFanout" for n in api.values())

    # ---- stage 1: image ----------------------------------------------------------------------
    async def stage_image(self, job, p):
        text = (p.get("prompt") or "").strip()
        if not text:
            raise AppError("이미지 프롬프트를 입력하세요")
        name = p.get("workflow") or ""
        prompt, warnings = await self.workflow_api(name)
        problem = self.usable(prompt, "image")
        if problem:
            raise AppError(f"이 워크플로우로는 이미지를 만들 수 없습니다 ({self.workflow_label(name)}): {problem}")
        slots = comfy_convert.find_slots(prompt)
        for nid in slots["prompts"][0]:
            prompt[nid]["inputs"]["text"] = text
        width, height = int(p.get("width") or 0) // 16 * 16, int(p.get("height") or 0) // 16 * 16
        if width and height:      # no size given: the size the workflow has
            for nid in slots["sizes"]:
                prompt[nid]["inputs"].update(width=width, height=height)
        seed = int(p.get("seed") or 0) or random.randint(1, 2 ** 48)
        for node in prompt.values():
            for key in ("seed", "noise_seed"):
                if isinstance(node["inputs"].get(key), int) and not isinstance(node["inputs"][key], bool):
                    node["inputs"][key] = seed
            if p.get("steps") and isinstance(node["inputs"].get("steps"), int) and "seed" in node["inputs"]:
                node["inputs"]["steps"] = int(p["steps"])      # only when asked: every workflow has its own number
        short = "zimage" if not name else re.sub(r"[^A-Za-z0-9]+", "_", self.workflow_label(name).rsplit("/", 1)[-1]).strip("_")[:40] or "image"
        for nid in slots["saves"]:
            prompt[nid]["inputs"]["filename_prefix"] = f"webapp/{short}_" + time.strftime("%Y%m%d")
        comfy_convert.apply_text_replacements(prompt)
        await self.hand_over("image", job)
        await self.run_prompt(job, prompt)
        images = [f for f in job.outputs if f["kind"] == "image"]
        if not images:
            raise AppError("이미지가 만들어지지 않았습니다")
        unseen = await self.find_outputs(job)
        made = ([f for f in images if f["type"] == "output"] or images)[-1]
        ref = {k: made[k] for k in ("filename", "subfolder", "type")}
        saved = " · ".join(filter(None, (self.saved_text(made) if made["type"] == "output" else "", unseen)))
        if not slots["sizes"] and width and height:
            saved = " · ".join(filter(None, (saved, "이 워크플로우는 크기를 워크플로우 값 그대로 씁니다")))
        size = prompt[slots["sizes"][0]]["inputs"] if slots["sizes"] else {"width": 0, "height": 0}
        return {"image": ref, "saved": saved, "seed": seed, "width": size["width"], "height": size["height"],
                "workflow": self.workflow_label(name), "missing": sorted({w.split("'")[1] for w in warnings if "unknown type" in w})}

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
    def video_workflow(self, p):
        """The workflow a video request names; requests of an earlier version said 'svi' / 'i2v'."""
        if p.get("workflow"):
            return p["workflow"]
        if p.get("backend") == "i2v":
            return re.sub(r"^workflow[\\/]", "", self.cfg["i2v_api"]).replace("\\", "/")
        return self.default_video()

    @staticmethod
    def numbers(api):
        """{node id: node} of the number constants a workflow exposes (segment length, size, steps...)."""
        return {nid: n for nid, n in api.items()
                if n["class_type"] in ("INTConstant", "PrimitiveInt", "PrimitiveFloat", "FloatConstant")
                and not isinstance(n["inputs"].get("value"), list)}

    async def video_info(self, name):
        api, warnings = await self.workflow_api(name)
        problem = self.usable(api, "video")
        if problem:
            warnings.append(problem)
        if self.has_fanout(api):
            return {"workflow": name, "slots": 6, "params": [], "warnings": warnings}
        items = [{"id": nid, "title": n["_meta"]["title"], "value": n["inputs"].get("value"),
                  "float": n["class_type"] in ("PrimitiveFloat", "FloatConstant")} for nid, n in self.numbers(api).items()]
        items.sort(key=lambda p: (comfy_convert.ordinal_of(p["title"]) or 0, p["title"].lower()))
        return {"workflow": name, "slots": len(comfy_convert.find_slots(api)["prompts"]), "warnings": warnings, "params": items}

    def build_svi(self, api, doc, texts, image_name, p, job):
        slots, params = comfy_convert.find_slots(api), self.numbers(api)
        load, prompt_nodes = slots["load"], slots["prompts"]
        api[load]["inputs"]["image"] = image_name
        if len(texts) != len(prompt_nodes):
            job.notes.append(f"시나리오 구간 {len(texts)}개, 워크플로우 구간 {len(prompt_nodes)}개: "
                             + ("남는 구간은 마지막 프롬프트를 반복합니다" if len(texts) < len(prompt_nodes) else "뒤쪽 프롬프트는 쓰지 않습니다"))
        for i, group in enumerate(prompt_nodes):
            for nid in group:
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

        workflow = self.video_workflow(p)
        api, warnings = await self.workflow_api(workflow)
        problem = self.usable(api, "video")
        if problem:
            raise AppError(f"이 워크플로우로는 영상을 만들 수 없습니다 ({self.workflow_label(workflow)}): {problem}")
        job.notes.append("워크플로우: " + self.workflow_label(workflow))
        job.notes.extend(warnings)
        if self.has_fanout(api):      # reads prompts.json through a node: give it the values themselves
            api = self.build_i2v(api, doc, texts, image_name)
        else:
            api = self.build_svi(api, doc, texts, image_name, p, job)
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
        # the scenario as it was run: a recorded work keeps it, whatever is done to the folder later
        return {"dir": name, "final": final[-1], "workflow": self.workflow_label(workflow), "scenario": doc}


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
        app.start_catalogue_refresh()
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


async def api_works(request):
    """The recorded works, newest first, without the scenarios they carry."""
    app = request.app["app"]
    items = []
    for work in app.store.works():
        view = app.work_view(work)
        view["scenario"] = {"dir": work["scenario"]["dir"], "segments": len(work["scenario"]["doc"].get("segments") or [])}
        items.append(view)
    return web.json_response({"items": items})


async def api_work_add(request):
    app = request.app["app"]
    return web.json_response({"work": app.work_view(app.record_work(await request.json()))})


async def api_work_open(request):
    app = request.app["app"]
    work = app.store.work(request.match_info["id"])
    if work is None:
        return json_error("기록을 찾을 수 없습니다", 404)
    if app.lock.locked():
        raise AppError("실행 중인 작업이 있어 지금은 기록을 열 수 없습니다 (시나리오 폴더를 기록할 때의 내용으로 되돌려야 합니다)")
    warning = await app.open_work(work)
    return web.json_response({"work": app.work_view(work), "warning": warning})


async def api_work_delete(request):
    """Forget a recorded work. Its files stay where they are."""
    request.app["app"].store.drop_work(request.match_info["id"])
    return web.json_response({"ok": True})


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


async def api_comfy_restart(request):
    """Restart the ComfyUI in use. ComfyUI itself cannot do that; the ComfyUI-Manager node pack on that server can
    (POST /manager/reboot), and starts it again with the options it was started with."""
    app = request.app["app"]
    if app.lock.locked():
        raise AppError("실행 중인 작업이 있습니다. 먼저 중단한 뒤 재시작하세요")
    try:
        async with app.comfy.session.post(app.comfy.url + "/manager/reboot", json={}, timeout=aiohttp.ClientTimeout(total=15)) as r:
            status, text = r.status, (await r.text())[:300]
    except (aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError, aiohttp.ClientOSError, asyncio.TimeoutError):
        status, text = 200, ""      # the process is replaced before it answers
    if status == 404:
        raise AppError("이 ComfyUI에는 재시작 기능이 없습니다 (서버에 ComfyUI-Manager 노드 팩이 있어야 합니다). 서버에서 직접 재시작하세요")
    if status == 403:
        raise AppError("서버의 ComfyUI-Manager가 재시작을 막고 있습니다 (security_level 설정). 서버에서 직접 재시작하세요")
    if status >= 400:
        raise AppError(f"재시작 요청이 거부되었습니다 ({status}): {text}")
    # what was known about that ComfyUI belongs to the process that is gone
    app.comfy._object_info = None
    app._combos, app._chat_models, app._llm, app._workflows, app.last_stage, app._followed = {}, (0, None), (0, None), {}, None, None
    return web.json_response({"message": "ComfyUI에 재시작을 요청했습니다. 다시 연결될 때까지 1~2분 걸립니다 (연결 표시가 파란색으로 돌아오면 끝난 것입니다)"})


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
    app = request.app["app"]
    return web.json_response({"scenarios": await asyncio.to_thread(app.list_scenarios), "folder": app.scenario_root})


async def api_scenario_rename(request):
    app, body = request.app["app"], await request.json()
    if app.lock.locked():
        raise AppError("실행 중인 작업이 있어 지금은 이름을 바꿀 수 없습니다")
    try:
        return web.json_response({"dir": app.rename_scenario(body.get("dir", ""), body.get("name", ""))})
    except OSError as e:
        raise AppError(f"이름을 바꾸지 못했습니다: {e.strerror or e}")


async def api_scenario_delete(request):
    """Delete one scenario ({'dir'}) or the chosen ones ({'dirs': [...]})."""
    app, body = request.app["app"], await request.json()
    if app.lock.locked():
        raise AppError("실행 중인 작업이 있어 지금은 지울 수 없습니다")
    removed, failed = [], []
    for name in body["dirs"] if isinstance(body.get("dirs"), list) else [body.get("dir", "")]:
        try:
            app.delete_scenario(str(name))
            removed.append(name)
        except AppError as e:
            failed.append({"dir": name, "error": str(e)})
        except OSError as e:
            failed.append({"dir": name, "error": e.strerror or str(e)})
    return web.json_response({"removed": removed, "failed": failed})


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
    # saved under another name: a new folder '<name>_prompts' with the same start image; the one it came from stays
    wanted = sanitize_name(body.get("name")) + "_prompts" if (body.get("name") or "").strip() else name
    if wanted != name:
        source, target = app.scenario_dir(name), app.scenario_dir(wanted)
        if os.path.exists(target):
            raise AppError(f"같은 이름의 시나리오가 이미 있습니다: {wanted[:-len('_prompts')]}. 다른 이름을 쓰세요")
        os.makedirs(target)
        for extra in ("reference.png", "model_answer.txt"):
            if os.path.isfile(os.path.join(source, extra)):
                shutil.copy2(os.path.join(source, extra), os.path.join(target, extra))
        name = wanted
    app.save_scenario(name, doc)
    return web.json_response({"dir": name, "scenario": doc})


async def api_scenario_image(request):
    app = request.app["app"]
    path = os.path.join(app.scenario_dir(request.query.get("dir", "")), "reference.png")
    if not os.path.isfile(path):
        return json_error("not found", 404)
    if request.query.get("thumb"):
        try:
            return web.FileResponse(await asyncio.to_thread(app.scenario_thumb, request.query["dir"]), headers={"Cache-Control": "max-age=3600"})
        except OSError:
            pass
    return web.FileResponse(path, headers={"Cache-Control": "no-cache"})


async def api_video_info(request):
    app = request.app["app"]
    return web.json_response(await app.video_info(request.query.get("workflow") or app.default_video()))


async def api_workflows(request):
    """The workflows a step can run (?kind=image | video)."""
    kind = request.query.get("kind", "video")
    return web.json_response(await request.app["app"].workflows("image" if kind == "image" else "video"))


async def api_home(request):
    """The first screen: the server's workflows by what they make, with their models, and the models nothing loads."""
    return web.json_response(await request.app["app"].home())


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
        if row and row.get("workflow"):      # what the job ran is known; the nodes only allow a guess
            info["workflow"], info["match"] = row["workflow"], "job"
    return web.json_response(info)


async def api_library_use(request):
    """Make a file of the library the start of a new work, from what ComfyUI recorded in it. A video gives a new
    scenario folder holding its segment prompts and its start image; an image becomes the start image, with its prompt."""
    app, lib, ref = request.app["app"], _library(request), (await request.json()).get("ref", "")
    try:
        full = lib.resolve(ref)
        prompt, _, media = await asyncio.to_thread(library.read_metadata, full)
        info = await asyncio.to_thread(library.describe, full, lib.catalogue)
    except ValueError as e:
        raise AppError(str(e))
    rel = lib.rel_of(ref) if ref.startswith("output:") else ""
    row = app.store.job_of(lib.root, rel) if rel else None
    workflow = (row or {}).get("workflow") or (info["workflow"] if info.get("match") else "")      # '' = not one the server keeps
    slots = comfy_convert.find_slots(prompt) if prompt else {"prompts": [], "load": None, "sizes": []}
    said = lambda ids: next((prompt[n]["inputs"]["text"] for n in ids if isinstance(prompt[n]["inputs"].get("text"), str)), "")
    texts = [said(group) for group in slots["prompts"]]      # '' where a node wrote the prompt while the workflow ran

    if not full.lower().endswith(library.VIDEO_EXT):
        if rel:      # ComfyUI reads it where it is
            image = {"filename": os.path.basename(rel), "subfolder": os.path.dirname(rel), "type": "output"}
        else:        # a file dropped on the page: ComfyUI gets a copy
            png = io.BytesIO()
            Image.open(full).convert("RGB").save(png, "PNG")
            image = await app.comfy.upload_image(png.getvalue(), time.strftime("%Y%m%d_%H%M%S") + "_upload.png")
        seeds = [n["inputs"][k] for n in (prompt or {}).values() for k in ("seed", "noise_seed") if isinstance(n["inputs"].get(k), int)]
        return web.json_response({"kind": "image", "image": image, "prompt": texts[0] if texts else "", "workflow": workflow,
                                  "seed": seeds[0] if seeds else 0, "width": media["width"], "height": media["height"]})

    if not texts:
        raise AppError("이 영상에는 프롬프트 정보가 들어 있지 않아 새 작업에 쓸 수 없습니다")
    # the start image: the file the workflow loaded, when the ComfyUI in use still has it; else the video's first frame
    image, image_ref, source = None, {}, "frame"
    loaded = prompt[slots["load"]]["inputs"].get("image") if slots["load"] else None
    if isinstance(loaded, str) and loaded:
        folder, _, filename = loaded.replace("\\", "/").rpartition("/")
        candidate = {"filename": filename, "subfolder": folder, "type": "input"}
        try:
            image, image_ref, source = Image.open(io.BytesIO(await app.comfy.view_bytes(candidate))).convert("RGB"), candidate, "input"
        except (AppError, aiohttp.ClientError, asyncio.TimeoutError, OSError):
            pass
    if image is None:
        try:
            image = Image.open(await asyncio.to_thread(lib.frame, ref, "first")).convert("RGB")
        except (ValueError, OSError, StopIteration, AttributeError, IndexError, av.FFmpegError) as e:
            raise AppError(f"시작 이미지를 구하지 못했습니다: {e}")
    name = time.strftime("%Y%m%d_%H%M%S") + "_" + sanitize_name(os.path.splitext(os.path.basename(full))[0]) + "_prompts"
    folder = app.scenario_dir(name)
    os.makedirs(folder, exist_ok=True)
    png = io.BytesIO()
    image.save(png, "PNG")
    with open(os.path.join(folder, "reference.png"), "wb") as f:
        f.write(png.getvalue())
    if not image_ref:      # so the scenario can be written again from this image
        try:
            image_ref = await app.comfy.upload_image(png.getvalue(), upload_name(name))
        except (AppError, aiohttp.ClientError, asyncio.TimeoutError):
            pass
    seconds = round(media["duration"] / len(texts), 1) if media.get("duration") else 5
    doc = build_scenario_doc({"summary_ko": "", "common_prompt": "", "segments": [{"positive_prompt": t, "scene_ko": ""} for t in texts]},
                             len(texts), seconds, image.width, image.height, "", image_ref)
    negative = said(sorted(comfy_convert.text_encoders(prompt)[1]))
    doc.update(source_file=os.path.basename(full), negative_prompt=negative or doc["negative_prompt"])
    app.save_scenario(name, doc)
    return web.json_response({"kind": "video", "dir": name, "workflow": workflow, "segments": len(texts),
                              "empty": sum(1 for t in texts if not t), "image_from": source})


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
    """Write the embedded workflow into the server's workflow folder so ComfyUI lists it under a real name."""
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
    app = request.app["app"]      # into the workflow folder of the connected ComfyUI, where its workflow list reads
    url = app.comfy.url + "/api/userdata/" + quote(f"workflows/{name}.json", safe="") + "?overwrite=false"
    async with app.comfy.session.post(url, data=json.dumps(workflow, ensure_ascii=False).encode("utf-8"),
                                      headers={"Content-Type": "application/json"}, timeout=aiohttp.ClientTimeout(total=60)) as r:
        if r.status == 409:
            raise AppError(f"서버의 워크플로우 폴더에 같은 이름이 이미 있습니다: {name}.json")
        if r.status != 200:
            raise AppError(f"서버에 저장하지 못했습니다 ({r.status}): {(await r.text())[:200]}")
    app._wf_list = (0, {})
    return web.json_response({"saved": name + ".json", "name": name})


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
    """Delete one file ({'ref'}) or the chosen ones ({'refs': [...]}); with several, the ones that cannot be
    deleted are reported and the others still go."""
    lib, body = _library(request), await request.json()
    if not isinstance(body.get("refs"), list):
        try:
            removed = await asyncio.to_thread(lib.delete, body.get("ref", ""))
        except ValueError as e:
            raise AppError(str(e))
        except OSError as e:
            raise AppError(f"삭제하지 못했습니다: {e.strerror or e}")
        return web.json_response({"removed": removed})
    removed, failed = [], []
    for ref in body["refs"][:2000]:
        try:
            removed += await asyncio.to_thread(lib.delete, str(ref))
        except ValueError as e:
            failed.append({"ref": ref, "error": str(e)})
        except OSError as e:
            failed.append({"ref": ref, "error": e.strerror or str(e)})
    return web.json_response({"removed": removed, "failed": failed})


# ---- models ----------------------------------------------------------------------------------------
async def api_models(request):
    app = request.app["app"]
    known = await app.listed_models()
    await asyncio.to_thread(app.library.scan)      # the 'used by' counts come from the library index
    try:
        folders, files, free = await asyncio.to_thread(app.models.listing, known)
    except ValueError as e:
        raise AppError(str(e))
    return web.json_response({"folder": app.model_root, "folders": folders, "files": files, "free": free,
                              "comfy_listed": known is not None})


async def api_models_rename(request):
    app, body = request.app["app"], await request.json()
    try:
        ref = await asyncio.to_thread(app.models.rename, body.get("ref", ""), body.get("name", ""))
    except ValueError as e:
        raise AppError(str(e))
    except OSError as e:
        raise AppError(f"이름을 바꾸지 못했습니다: {e.strerror or e}")
    app._listed = (0, None)      # ComfyUI lists the new name
    return web.json_response({"ref": ref, "name": os.path.basename(ref.partition(":")[2])})


async def api_models_delete(request):
    """Delete the chosen model files ({'refs': [...]}); the ones that cannot be deleted are reported, the others go."""
    app, body = request.app["app"], await request.json()
    removed, failed = [], []
    for ref in (body.get("refs") if isinstance(body.get("refs"), list) else [body.get("ref", "")])[:2000]:
        try:
            removed.append(await asyncio.to_thread(app.models.delete, str(ref)))
        except ValueError as e:
            failed.append({"ref": ref, "error": str(e)})
        except OSError as e:
            failed.append({"ref": ref, "error": e.strerror or str(e)})
    app._listed = (0, None)
    return web.json_response({"removed": removed, "failed": failed})


async def api_fs(request):
    """The other pane of the model menu: one folder of this PC (?path=, '' for the drives)."""
    path = request.query.get("path", "").strip().strip('"')
    if path and not os.path.isabs(path):
        raise AppError("전체 경로로 입력하세요 (예: D:\\models)")
    try:
        return web.json_response(await asyncio.to_thread(models.list_dir, path))
    except OSError as e:
        raise AppError(f"폴더를 열 수 없습니다: {e.strerror or e}")


def _model_path(app, item, folders):
    """A 'model:...' ref or an absolute path of this PC -> absolute path."""
    item = str(item or "").strip()
    if item.startswith("model:"):
        return app.models.resolve(item, folders)
    if not os.path.isabs(item) or not (os.path.isfile(item) or (folders and os.path.isdir(item))):
        raise ValueError("찾을 수 없습니다: " + item)
    return os.path.abspath(item)


async def api_models_copy(request):
    """Start copying files / folders ({'items': [...]}) into a folder ({'dest'}); one side must be the model folder.
    Out of the model folder the files keep the folders they sit in (loras/HIGH/a -> <dest>/loras/HIGH/a); into it
    they go straight into the folder given. One copy at a time; GET asks how far it is."""
    app = request.app["app"]
    if request.method == "GET":
        return web.json_response(app.copy_job.progress() if app.copy_job else {"state": "none"})
    if app.copy_job and app.copy_job.state in ("planning", "running"):
        raise AppError("복사가 진행 중입니다. 끝나거나 중단한 뒤에 다시 하세요")
    body = await request.json()
    try:
        items = [_model_path(app, it, True) for it in (body.get("items") or [])[:2000]]
        dest = _model_path(app, body.get("dest"), True)
    except ValueError as e:
        raise AppError(str(e))
    if not items:
        raise AppError("복사할 파일을 고르세요")
    if not os.path.isdir(dest):
        raise AppError("복사할 폴더가 아닙니다: " + dest)
    if not (app.models.inside(dest) or all(app.models.inside(it) for it in items)):
        raise AppError("모델 폴더에서 내보내거나 모델 폴더로 들여오는 복사만 됩니다")
    pairs = None
    if all(app.models.inside(it) for it in items) and not app.models.inside(dest):      # out of the model folder: as it is laid out there
        try:
            pairs = await asyncio.to_thread(models.mirrored, items, app.model_root, dest)
        except ValueError as e:
            raise AppError(str(e))
    app.copy_job = models.Copy(items, dest, pairs=pairs)
    asyncio.get_running_loop().run_in_executor(None, app.copy_job.run)
    return web.json_response({"started": len(items), "dest": dest})


async def api_models_incoming(request):
    """Downloaded models: the model files under ?path= (a folder of this PC) with the kind each one is and the model
    folder it belongs in. Without a path: the folder to start from (the user's Downloads)."""
    app = request.app["app"]
    path = request.query.get("path", "").strip().strip('"')
    if not path:
        return web.json_response({"suggest": os.path.join(os.path.expanduser("~"), "Downloads")})
    if not os.path.isabs(path) or not os.path.isdir(path):
        raise AppError("폴더를 찾을 수 없습니다. 전체 경로로 입력하세요 (예: D:\\Downloads): " + path)
    root = app.model_root
    if not os.path.isdir(root):
        raise AppError("모델 폴더가 없습니다: " + root)
    try:
        found = await asyncio.to_thread(models.incoming, path, root)
    except OSError as e:
        raise AppError(f"폴더를 읽지 못했습니다: {e.strerror or e}")
    found["model_root"] = root
    return web.json_response(found)


async def api_models_place(request):
    """Send downloaded model files into the model folder: {'items': [{'path', 'folder', 'sub'}], 'move': bool, 'base'}.
    `folder` is a folder of the model folder (it must be there); `sub` the folders below it the file goes into (made
    when missing: the folders the file sat in). After a move the folders left empty below `base` are removed.
    Runs as the copy job, so /api/models/copy tells how far it is."""
    app, body = request.app["app"], await request.json()
    if app.copy_job and app.copy_job.state in ("planning", "running"):
        raise AppError("복사가 진행 중입니다. 끝나거나 중단한 뒤에 다시 하세요")
    root = os.path.realpath(app.model_root)
    pairs = []
    for it in (body.get("items") or [])[:2000]:
        src, folder = os.path.abspath(str(it.get("path", ""))), str(it.get("folder", "")).strip().strip("/\\")
        dest = os.path.realpath(os.path.join(root, folder))
        if not os.path.isfile(src) or not src.lower().endswith(models.MODEL_EXT):
            raise AppError("모델 파일이 아니거나 없습니다: " + src)
        if not folder or not os.path.isdir(dest) or not models.within(dest, root) or dest == root:
            raise AppError("모델 폴더 안의 폴더가 아닙니다: " + folder)
        sub = [p for p in re.split(r"[\\/]+", str(it.get("sub") or "")) if p]
        if any(p in (".", "..") or re.search(r'[:*?"<>|\x00-\x1f]', p) for p in sub):
            raise AppError("하위 폴더 이름이 잘못되었습니다: " + str(it.get("sub")))
        pairs.append((src, os.path.join(dest, *sub, os.path.basename(src))))
    if not pairs:
        raise AppError("보낼 파일을 고르세요")
    base = str(body.get("base") or "")
    app.copy_job = models.Copy([], root, pairs=pairs, move=bool(body.get("move")), tidy=base if os.path.isdir(base) else None)
    asyncio.get_running_loop().run_in_executor(None, app.copy_job.run)
    app._listed = (0, None)
    return web.json_response({"started": len(pairs), "dest": root})


async def api_models_copy_cancel(request):
    app = request.app["app"]
    if app.copy_job:
        app.copy_job.stop = True
    return web.json_response({"ok": True})


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


async def api_page(request):
    """Each open page calls in every few seconds ({'id'}) and says goodbye when it closes ({'id', 'bye': true}),
    so the program can stop when no page is left (close_with_page)."""
    app, body = request.app["app"], await request.json()
    pid = str(body.get("id", ""))[:40]
    if pid:
        if body.get("bye"):
            app.pages.pop(pid, None)
        else:
            app.pages[pid] = time.time()
            app.pages_gone_since = None
    return web.json_response({"pages": len(app.pages), "close_with_page": bool(app.cfg.get("close_with_page"))})


async def page_watch(web_app):
    """With close_with_page: stop once every page has been closed for a while (a reload is not a close), or when no
    page came at all. Not in the middle of a stage or a copy; the ComfyUI job itself goes on without this program."""
    app = web_app["app"]
    started = app.pages_gone_since      # stays the value of pages_gone_since until a page has called in
    while True:
        await asyncio.sleep(3)
        now = time.time()
        for pid, seen in list(app.pages.items()):
            if now - seen > 20:
                del app.pages[pid]
        if app.pages:
            continue
        if app.pages_gone_since is None:
            app.pages_gone_since = now
        quiet = now - app.pages_gone_since
        never = app.pages_gone_since == started and quiet > 90
        if (quiet > 10 or never) and not app.lock.locked() and not (app.copy_job and app.copy_job.state in ("planning", "running")):
            log.info("no page is open any more: stopping" if not never else "no page came: stopping")
            logging.shutdown()
            os._exit(0)


async def on_startup(web_app):
    app = web_app["app"]
    await app.comfy.start()
    web_app["ws_task"] = asyncio.create_task(app.ws_loop())
    if app.cfg.get("open"):
        host = "127.0.0.1" if app.cfg["host"] in ("0.0.0.0", "::") else app.cfg["host"]
        asyncio.get_running_loop().call_later(1.0, webbrowser.open, f"http://{host}:{app.cfg['port']}")
    if app.cfg.get("close_with_page"):
        app.pages_gone_since = time.time()
        web_app["page_task"] = asyncio.create_task(page_watch(web_app))


async def on_cleanup(web_app):
    web_app["ws_task"].cancel()
    if "page_task" in web_app:
        web_app["page_task"].cancel()
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
    ap.add_argument("--close-with-page", action="store_true", help="stop when the last page has been closed in the browser")
    args = ap.parse_args()
    cfg.update(host=args.host, port=args.port, comfy_url=args.comfy, output_dir=args.output_dir, open=args.open,
               close_with_page=args.close_with_page or cfg["close_with_page"])
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
        web.post("/api/comfy/restart", api_comfy_restart),
        web.post("/api/result_dir", api_result_dir),
        web.get("/api/folders", api_folders),
        web.post("/api/upload", api_upload),
        web.get("/api/scenarios", api_scenarios),
        web.get("/api/scenario", api_scenario_get),
        web.post("/api/scenario", api_scenario_save),
        web.get("/api/scenario/image", api_scenario_image),
        web.post("/api/scenario/rename", api_scenario_rename),
        web.post("/api/scenario/delete", api_scenario_delete),
        web.get("/api/video/info", api_video_info),
        web.get("/api/workflows", api_workflows),
        web.get("/api/home", api_home),
        web.get("/api/view", api_view),
        web.get("/api/library", api_library),
        web.get("/api/library/item", api_library_item),
        web.get("/api/library/file", api_library_file),
        web.get("/api/library/thumb", api_library_thumb),
        web.get("/api/library/frame", api_library_frame),
        web.get("/api/library/workflow", api_library_workflow),
        web.post("/api/library/save_workflow", api_library_save_workflow),
        web.post("/api/library/use", api_library_use),
        web.post("/api/library/mark", api_library_mark),
        web.post("/api/library/rename", api_library_rename),
        web.post("/api/library/delete", api_library_delete),
        web.get("/api/models", api_models),
        web.post("/api/models/rename", api_models_rename),
        web.post("/api/models/delete", api_models_delete),
        web.get("/api/fs", api_fs),
        web.get("/api/models/copy", api_models_copy),
        web.post("/api/models/copy", api_models_copy),
        web.post("/api/models/copy/cancel", api_models_copy_cancel),
        web.get("/api/models/incoming", api_models_incoming),
        web.post("/api/models/place", api_models_place),
        web.get("/api/works", api_works),
        web.post("/api/works", api_work_add),
        web.post("/api/works/{id}/open", api_work_open),
        web.post("/api/works/{id}/delete", api_work_delete),
        web.post("/api/inspect", api_inspect),
        web.post("/api/page", api_page),
    ])
    web_app.on_startup.append(on_startup)
    web_app.on_cleanup.append(on_cleanup)
    return web_app


def alert(text):
    """Without a console (pythonw) the only way to tell is a message box."""
    if sys.stderr is None and os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, text, "ComfyUI-Manager", 0x10)
    else:
        print(text, file=sys.stderr)


if __name__ == "__main__":
    # the log goes to data\manager.log as well (the only place when run without a console)
    handlers = [logging.handlers.RotatingFileHandler(os.path.join(DATA, "manager.log"), maxBytes=1 << 20, backupCount=2, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    os.makedirs(DATA, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", handlers=handlers)
    config = load_config()
    shown = "127.0.0.1" if config["host"] in ("0.0.0.0", "::") else config["host"]
    print(f"\n  ComfyUI-Manager:  http://{shown}:{config['port']}\n  ComfyUI:          {config['comfy_url']}\n")
    log.info("ComfyUI-Manager on http://%s:%s (ComfyUI %s)%s", shown, config["port"], config["comfy_url"],
             " - stops when the page is closed" if config["close_with_page"] else " - keeps running")
    try:
        web.run_app(make_app(config), host=config["host"], port=config["port"], print=None)
    except OSError as e:      # the port is taken, most likely by another copy of this program
        log.error("cannot start: %s", e)
        alert(f"ComfyUI-Manager를 시작하지 못했습니다.\n\n{e}\n\n포트 {config['port']}를 다른 프로그램(또는 이미 켜 둔 ComfyUI-Manager)이 쓰고 있을 수 있습니다.")
        sys.exit(1)
