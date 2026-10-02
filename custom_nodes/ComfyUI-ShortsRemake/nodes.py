"""ComfyUI-ShortsRemake

Helper nodes for the "analyze a video in N-second segments -> prompts -> regenerate
clip by clip with a swapped person -> concatenate" pipeline.

Nodes
- Shorts Video Segments   : split a video file into N-second segments (frame batches, list output)
- Shorts Prompts Collector: gather VLM JSON answers into prompts.json / comfyui_prompts.txt
- Shorts Prompts Loader   : read prompts.json -> per-clip prompt / frame range lists for generation
- Shorts Clip Saver       : save one generated VIDEO as clip_NN.mp4 next to prompts.json
- Shorts Concat           : concatenate clip_NN.mp4 -> final.mp4 (ffmpeg)
- Shorts YouTube Download : YouTube URL (or local file) -> mp4 (H.264 preferred / re-encoded), optional start..end clip cut
- Shorts Reference Setup  : profile + background + props images -> Qwen-Image-Edit inputs + instruction + canvas size
- Shorts Reference Save   : composed reference -> <prompts dir>/reference.png (+ copy to ComfyUI/input)
- Shorts Reference Loader : prompts.json path -> reference IMAGE (reference.png / linked / fallback) + prompts_json passthrough
- Shorts Prompts Fanout   : prompts.json -> seg_1..seg_8 STRING outputs (for graphs with one text box per segment)
- Shorts Segments Collect : lazy seg_1..seg_8 IMAGE inputs; only the first `count` segments execute, frames concatenated
- Shorts Free VRAM        : pass-through that unloads every QwenVL model instance (+ ComfyUI models) between stages
- Shorts Size From Image  : width/height (multiples of 16) in the aspect ratio of an image, long side = max_side
- Shorts Qwen GGUF Vision : free-form question about an image to a ComfyUI-QwenVL-Mod GGUF model (no preset added)

List mechanics: outputs flagged in OUTPUT_IS_LIST make every downstream node run once
per segment, so a normal single-clip generation graph becomes a per-segment loop.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, jpeg artifacts, watermark, text, logo, subtitles, "
    "deformed, extra limbs, extra fingers, bad anatomy, distorted face, flicker, jitter, "
    "static image, frozen frame, duplicated subject, overexposed, oversaturated"
)

DEFAULT_TEMPLATE = "Character Description: {character}\nBackground description: {common} {segment}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # bundled with ComfyUI portable

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg") or "ffmpeg"


def _fmt_time(sec: float) -> str:
    m, s = divmod(float(sec), 60)
    return f"{int(m):02d}:{s:06.3f}"


def build_segments(duration: float, seg_len: float, min_tail: float = 1.5) -> list[dict]:
    if duration <= 0 or seg_len <= 0:
        return []
    n = max(1, math.ceil(duration / seg_len))
    segs = []
    for i in range(n):
        start = i * seg_len
        end = min((i + 1) * seg_len, duration)
        segs.append({"index": i + 1, "start": round(start, 3), "end": round(end, 3)})
    if len(segs) > 1 and (segs[-1]["end"] - segs[-1]["start"]) < min_tail:
        tail = segs.pop()
        segs[-1]["end"] = tail["end"]
    for s in segs:
        s["duration"] = round(s["end"] - s["start"], 3)
        s["label"] = f"{_fmt_time(s['start'])} - {_fmt_time(s['end'])}"
    return segs


def detect_scene_cuts(path: str, threshold: float = 0.5, sample_fps: float = 4.0, min_gap: float = 0.5) -> list[float]:
    """Return timestamps (sec) of hard cuts. Score = 1 - HSV histogram correlation between consecutive samples."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"[ShortsRemake] cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(0.5, sample_fps))))
    prev = None
    cuts: list[float] = []
    last_cut = -1e9
    fi = -1
    while True:
        # sequential decode (grab skips cheaply); seeking on 4K sources is far slower
        ok = cap.grab()
        if not ok:
            break
        fi += 1
        if fi % step:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        # use the container timestamp, not frame_index/fps (variable-rate sources drift otherwise)
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = (t_ms / 1000.0) if t_ms and t_ms > 0 else (fi / fps)
        small = cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev is not None:
            score = 1.0 - float(cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL))
            if score >= threshold and (t - last_cut) >= min_gap:
                cuts.append(round(t, 3))
                last_cut = t
        prev = hist
    cap.release()
    return cuts


def build_scene_segments(duration: float, cuts: list[float], max_len: float, min_len: float) -> list[dict]:
    """Segments bounded by scene cuts; pieces longer than max_len are split evenly, shorter than min_len merged."""
    bounds = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    raw: list[list[float]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 0.05:
            continue
        n = max(1, math.ceil((b - a) / max_len)) if max_len > 0 else 1
        step = (b - a) / n
        for i in range(n):
            raw.append([a + i * step, a + (i + 1) * step if i < n - 1 else b])
    # merge too-short pieces into the previous one (or next, for the first)
    merged: list[list[float]] = []
    for seg in raw:
        if merged and (seg[1] - seg[0]) < min_len:
            merged[-1][1] = seg[1]
        elif not merged and (seg[1] - seg[0]) < min_len and len(raw) > 1:
            merged.append(seg)  # keep; next piece will be merged into it if needed
        else:
            merged.append(seg)
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_len:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    segs = []
    for i, (a, b) in enumerate(merged):
        s = {"index": i + 1, "start": round(a, 3), "end": round(b, 3)}
        s["duration"] = round(b - a, 3)
        s["label"] = f"{_fmt_time(a)} - {_fmt_time(b)}"
        segs.append(s)
    return segs


def _video_info(path: str) -> tuple[float, float, int, int]:
    """(duration_sec, fps, width, height) via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"[ShortsRemake] cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = (n / fps) if fps > 0 else 0.0
    # prefer the container duration reported by ffmpeg (frame_count/fps is wrong for variable-rate files)
    try:
        cp = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", cp.stderr or "")
        if m:
            d2 = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            if d2 > 0:
                duration = d2
    except Exception:  # noqa: BLE001
        pass
    return duration, fps, w, h


def _video_codec(path: str) -> str:
    """fourcc of the video stream, lower-case ('avc1', 'av01', 'vp09', 'hev1', ...)."""
    import cv2

    cap = cv2.VideoCapture(path)
    v = int(cap.get(cv2.CAP_PROP_FOURCC)) if cap.isOpened() else 0
    cap.release()
    try:
        return v.to_bytes(4, "little").decode("ascii", "replace").strip("\x00 ").lower()
    except Exception:  # noqa: BLE001
        return ""


H264_FOURCC = ("avc1", "h264", "x264", "avc3")
SEEK_GAP = 4.0  # seconds: larger forward jumps seek, smaller ones decode forward (seeking is very slow on AV1/VP9)


def _read_frames_at(path: str, times: list[float], max_side: int) -> torch.Tensor:
    """Frames at the given timestamps (any order). One seek per big jump, otherwise sequential decode."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"[ShortsRemake] cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    half = 0.5 / fps
    n = len(times)
    out: list = [None] * n
    order = sorted(range(n), key=lambda i: float(times[i]))
    cur_idx = -1  # index of the last grabbed frame
    last_t = -1.0
    last_fr = None

    def frame_time() -> float:
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        return (t_ms / 1000.0) if t_ms and t_ms > 0 else (cur_idx / fps)

    def convert(fr):
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        if max_side > 0 and max(h, w) > max_side:
            sc = max_side / max(h, w)
            fr = cv2.resize(fr, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
        return fr

    for i in order:
        t = max(0.0, float(times[i]))
        if last_fr is not None and abs(t - last_t) <= half:
            out[i] = last_fr.copy()
            continue
        if last_fr is None or t < last_t or (t - last_t) > SEEK_GAP:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            last_t = -1.0
        got = None
        while True:
            if not cap.grab():
                break
            cur_idx += 1
            ft = frame_time()
            if ft + half >= t:
                ok, fr = cap.retrieve()
                if ok:
                    got = fr
                last_t = ft
                break
        if got is None:  # end of stream / decode error: repeat the previous frame
            if last_fr is None:
                cap.release()
                raise RuntimeError(f"[ShortsRemake] no frames decoded from {path}")
            out[i] = last_fr.copy()
            continue
        last_fr = convert(got)
        out[i] = last_fr
    cap.release()
    arr = np.stack(out).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _parse_json_block(text: str) -> dict:
    """Extract the first JSON object from a model answer (tolerates ``` fences / prose)."""
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = t.find("{", start + 1)
    return {}


def _merge_negative(default: str, extra) -> str:
    """Default negative list first, then model-suggested terms that are new (ASCII only, deduped)."""
    items = [x.strip() for x in (default or "").split(",") if x.strip()]
    seen = {x.lower() for x in items}
    if isinstance(extra, list):
        extra = ", ".join(str(x) for x in extra)
    for x in (extra or "").replace("\n", ",").split(","):
        x = x.strip().strip(".")
        if x and x.isascii() and x.lower() not in seen and len(x) < 60:
            items.append(x)
            seen.add(x.lower())
    return ", ".join(items)


def _parse_clip_spec(spec: str, count: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(1, count + 1))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return [i for i in sorted(out) if 1 <= i <= count]


def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


# --------------------------------------------------------------------------- #
# file pickers: dropdown choices for videos / prompts.json in output\ and input\
# --------------------------------------------------------------------------- #
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def _pick_bases() -> list[tuple[str, str]]:
    """(prefix, folder) pairs the dropdowns are built from: <root>/output, ComfyUI output (if different), input."""
    bases: list[tuple[str, str]] = []
    try:
        import folder_paths
        comfy_out = os.path.abspath(folder_paths.get_output_directory())
        inp = os.path.abspath(folder_paths.get_input_directory())
        root_out = os.path.join(os.path.dirname(os.path.abspath(folder_paths.base_path)), "output")
        if os.path.isdir(root_out):
            bases.append(("output", root_out))
        if os.path.isdir(comfy_out) and os.path.normcase(comfy_out) != os.path.normcase(os.path.abspath(root_out)):
            bases.append(("ComfyUI/output", comfy_out))
        if os.path.isdir(inp):
            bases.append(("input", inp))
    except Exception:  # noqa: BLE001
        pass
    return bases


def _list_video_choices() -> list[str]:
    items = []
    for prefix, base in _pick_bases():
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for n in names:
            p = os.path.join(base, n)
            if os.path.isfile(p) and n.lower().endswith(VIDEO_EXTS):
                items.append((os.path.getmtime(p), f"{prefix}/{n}"))
    items.sort(reverse=True)  # newest first
    return [x[1] for x in items]


def _list_prompts_choices() -> list[str]:
    items = []
    for prefix, base in _pick_bases():
        for p in glob.glob(os.path.join(base, "*", "prompts.json")) + glob.glob(os.path.join(base, "*", "*", "prompts.json")):
            rel = os.path.relpath(p, base).replace("\\", "/")
            items.append((os.path.getmtime(p), f"{prefix}/{rel}"))
    items.sort(reverse=True)
    return [x[1] for x in items]


def _resolve_pick(text, choice) -> str:
    """Typed path wins; otherwise map a dropdown choice ('output/x.mp4', 'input/y', or an uploaded bare name) to a path."""
    t = os.path.expanduser((text or "").strip().strip('"'))
    if t:
        return t
    c = (choice or "").strip()
    if not c:
        return ""
    if os.path.isabs(c):
        return c
    for prefix, base in _pick_bases():
        if c.startswith(prefix + "/"):
            return os.path.join(base, c[len(prefix) + 1:])
    try:
        import folder_paths
        return os.path.join(folder_paths.get_input_directory(), c)  # file uploaded through the Upload button
    except Exception:  # noqa: BLE001
        return c


# --------------------------------------------------------------------------- #
# 1. Video -> segments
# --------------------------------------------------------------------------- #
class ShortsVideoSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_file": (_list_video_choices(), {"video_upload": True,
                               "tooltip": "Videos in output/ (stage-0 downloads) and input/, newest first. Upload adds a file from this PC."}),
                "video_path": ("STRING", {"default": "", "placeholder": "empty = use video_file above; or type a full path"}),
                "split_mode": (["fixed", "scene"], {"default": "fixed",
                                                    "tooltip": "fixed: every segment_seconds. scene: cut at scene changes; scenes longer than segment_seconds are split, shorter than min_seconds merged."}),
                "segment_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5,
                                              "tooltip": "fixed: segment length. scene: maximum segment length."}),
                "min_seconds": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.5,
                                          "tooltip": "Pieces shorter than this are merged into the neighbour (fixed: only the last piece)."}),
                "scene_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                              "tooltip": "scene mode: 1 - histogram correlation needed to count as a cut. Lower = more cuts."}),
                "frames_per_segment": ("INT", {"default": 8, "min": 1, "max": 64}),
                "max_side": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 16,
                                     "tooltip": "Frames are downscaled so the longer side <= this (VLM input)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("segment_frames", "segment_index", "segment_label", "overview_frames", "count", "video_path", "segments_json")
    OUTPUT_IS_LIST = (True, True, True, False, False, False, False)
    FUNCTION = "split"
    CATEGORY = "ShortsRemake"
    DESCRIPTION = "Splits a video into N-second segments. List outputs make downstream nodes run once per segment."

    @classmethod
    def IS_CHANGED(cls, video_file, video_path, **kw):
        return _file_sig(_resolve_pick(video_path, video_file))

    @classmethod
    def VALIDATE_INPUTS(cls, video_file, video_path):
        return True  # the dropdown list is rebuilt on every refresh; a typed path or an older choice is fine

    def split(self, video_file, video_path, split_mode, segment_seconds, min_seconds, scene_threshold, frames_per_segment, max_side):
        video_path = _resolve_pick(video_path, video_file)
        if not video_path:
            raise ValueError("[ShortsRemake] pick a video in video_file or type a path in video_path")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"[ShortsRemake] video not found: {video_path}")
        duration, fps, w, h = _video_info(video_path)
        cuts: list[float] = []
        if split_mode == "scene":
            cuts = detect_scene_cuts(video_path, threshold=scene_threshold)
            segs = build_scene_segments(duration, cuts, segment_seconds, min_seconds)
            print(f"[ShortsRemake] scene cuts at {cuts}")
        else:
            segs = build_segments(duration, segment_seconds, min_seconds)
        if not segs:
            raise RuntimeError("[ShortsRemake] could not determine video duration")

        codec = _video_codec(video_path)
        if codec and codec not in H264_FOURCC:
            print(f"[ShortsRemake] note: '{codec}' video decodes slowly (and stage 3 reads it per segment). "
                  f"Run it through '0_YouTube_Download_Trim' with ensure_h264 first for a faster H.264 copy.")
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(len(segs))
        except Exception:  # noqa: BLE001
            pbar = None
        print(f"[ShortsRemake] extracting {frames_per_segment} frames x {len(segs)} segments from {os.path.basename(video_path)} ({codec or '?'})")

        frames_out, idx_out, label_out, overview = [], [], [], []
        for si, s in enumerate(segs):
            length = max(0.001, s["end"] - s["start"])
            k = max(1, frames_per_segment)
            if k == 1:
                times = [s["start"] + length / 2]
            else:
                pad = min(0.08, length * 0.02)
                times = [s["start"] + pad + (length - 2 * pad) * i / (k - 1) for i in range(k)]
            batch = _read_frames_at(video_path, times, max_side)
            frames_out.append(batch)
            idx_out.append(int(s["index"]))
            label_out.append(s["label"])
            overview.append(batch[len(batch) // 2:len(batch) // 2 + 1])
            s["sample_times"] = [round(t, 3) for t in times]
            if pbar is not None:
                pbar.update(1)
            if (si + 1) % 5 == 0 or si + 1 == len(segs):
                print(f"[ShortsRemake]   frames: {si + 1}/{len(segs)} segments")
        ov = torch.cat(overview, dim=0)
        meta = {"video_file": video_path, "duration": round(duration, 3), "source_fps": round(fps, 3),
                "width": w, "height": h, "segment_length": segment_seconds, "split_mode": split_mode,
                "scene_cuts": cuts, "segments": segs}
        print(f"[ShortsRemake] {os.path.basename(video_path)}: {duration:.2f}s -> {len(segs)} segments "
              f"({split_mode}, max {segment_seconds}s): {[s['label'] for s in segs]}")
        return (frames_out, idx_out, label_out, ov, len(segs), video_path, json.dumps(meta, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# 2. VLM answers -> prompts.json
# --------------------------------------------------------------------------- #
class ShortsPromptsCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_responses": ("STRING", {"forceInput": True}),
                "common_response": ("STRING", {"forceInput": True}),
                "segments_json": ("STRING", {"forceInput": True}),
                "out_dir": ("STRING", {"default": "", "placeholder": "empty = <video folder>/<video name>_prompts"}),
                "negative_prompt": ("STRING", {"default": DEFAULT_NEGATIVE, "multiline": True,
                                               "tooltip": "Used when the VLM does not return a negative_prompt."}),
                "target_fps": ("INT", {"default": 16, "min": 1, "max": 60, "tooltip": "Generator fps hint (Wan 16, LTX 24/25)."}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "prompts_json")
    OUTPUT_NODE = True
    FUNCTION = "collect"
    CATEGORY = "ShortsRemake"

    def collect(self, segment_responses, common_response, segments_json, out_dir, negative_prompt, target_fps):
        meta = json.loads(segments_json[0])
        segs = meta["segments"]
        common = _parse_json_block(common_response[0]) if common_response else {}
        neg_default = (negative_prompt[0] if negative_prompt else DEFAULT_NEGATIVE).strip()
        fps = int(target_fps[0]) if target_fps else 16

        video_file = meta["video_file"]
        od = (out_dir[0] if out_dir else "").strip()
        if not od:
            vp = Path(video_file)
            od = str(vp.parent / f"{vp.stem}_prompts")
        os.makedirs(od, exist_ok=True)

        out_segments = []
        for i, s in enumerate(segs):
            raw = segment_responses[i] if i < len(segment_responses) else ""
            g = _parse_json_block(raw)
            out_segments.append({
                "index": s["index"], "start": s["start"], "end": s["end"], "duration": s["duration"], "label": s["label"],
                "scene_ko": g.get("scene_ko") or g.get("scene_description_ko") or "",
                "positive_prompt": (g.get("positive_prompt") or "").strip(),
                "camera": g.get("camera") or g.get("camera_motion") or "",
                "motion": g.get("motion") or g.get("pose_prompt") or "",
                "raw_response": raw if not g else None,
            })

        result = {
            "video_file": video_file,
            "duration": meta.get("duration"),
            "source_fps": meta.get("source_fps"),
            "width": meta.get("width"), "height": meta.get("height"),
            "segment_length": meta.get("segment_length"),
            "fps": fps,
            "summary_ko": common.get("summary_ko", ""),
            "common_prompt": (common.get("common_prompt") or common.get("style_guide") or "").strip(),
            "person_in_video": common.get("person_in_video", ""),
            "negative_prompt": _merge_negative(neg_default, common.get("negative_prompt")),
            "segments": out_segments,
        }
        jp = os.path.join(od, "prompts.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(os.path.join(od, "comfyui_prompts.txt"), "w", encoding="utf-8") as f:
            f.write(f"# {video_file}\n# {len(out_segments)} clips x {meta.get('segment_length')}s @ {fps}fps\n\n")
            f.write("COMMON:\n" + result["common_prompt"] + "\n\nNEGATIVE:\n" + result["negative_prompt"] + "\n\n")
            for s in out_segments:
                f.write(f"### clip_{s['index']:02d} [{s['label']}]\nPOSITIVE:\n{s['positive_prompt']}\nMOTION:\n{s['motion']}\n\n")

        lines = [f"prompts.json -> {jp}", "", f"[COMMON] {result['common_prompt']}", f"[NEGATIVE] {result['negative_prompt']}", ""]
        for s in out_segments:
            lines.append(f"[clip_{s['index']:02d} {s['label']}] {s['positive_prompt']}")
            if s.get("raw_response"):
                lines.append("   ! JSON parse failed, raw answer kept in prompts.json")
        summary = "\n".join(lines)
        return {"ui": {"text": [summary]}, "result": (summary, jp)}


# --------------------------------------------------------------------------- #
# 3. prompts.json -> per-clip lists for generation
# --------------------------------------------------------------------------- #
class ShortsPromptsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_json": ("STRING", {"default": "", "placeholder": "C:/.../video_prompts/prompts.json"}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60, "tooltip": "Generator fps: Wan 16, LTX 24/25"}),
                "frame_step": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "frame_count = step*n+1  (Wan 4, LTX 8)"}),
                "max_side": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 16}),
                "divisible_by": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8}),
                "clips": ("STRING", {"default": "", "placeholder": "empty = all, or 1,3-4"}),
                "prompt_template": ("STRING", {"default": DEFAULT_TEMPLATE, "multiline": True,
                                               "tooltip": "{character} {common} {segment} {motion} {camera} placeholders"}),
            },
            "optional": {
                "character_description": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "STRING", "STRING", "STRING", "FLOAT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "pose_prompt", "segment_index", "skip_frames", "frame_count",
                    "negative", "common", "video_path", "fps", "width", "height", "count", "prompts_dir")
    OUTPUT_IS_LIST = (True, True, True, True, True, False, False, False, False, False, False, False, False)
    FUNCTION = "load"
    CATEGORY = "ShortsRemake"

    @classmethod
    def IS_CHANGED(cls, prompts_json, **kw):
        return _file_sig(prompts_json)

    def load(self, prompts_json, fps, frame_step, max_side, divisible_by, clips, prompt_template, character_description=""):
        prompts_json = os.path.expanduser(prompts_json.strip().strip('"'))
        if not os.path.isfile(prompts_json):
            raise FileNotFoundError(f"[ShortsRemake] prompts.json not found: {prompts_json}")
        with open(prompts_json, encoding="utf-8") as f:
            pr = json.load(f)
        video_path = pr.get("video_file", "")
        if not os.path.isfile(video_path):
            cand = os.path.join(os.path.dirname(os.path.dirname(prompts_json)), os.path.basename(video_path))
            if os.path.isfile(cand):
                video_path = cand
            else:
                raise FileNotFoundError(f"[ShortsRemake] source video not found: {video_path}")

        w = int(pr.get("width") or 0)
        h = int(pr.get("height") or 0)
        if not w or not h:
            _, _, w, h = _video_info(video_path)
        s = min(1.0, max_side / max(w, h))
        d = max(8, int(divisible_by))
        W = max(d, int(round(w * s / d)) * d)
        H = max(d, int(round(h * s / d)) * d)

        common = (pr.get("common_prompt") or "").strip()
        negative = (pr.get("negative_prompt") or DEFAULT_NEGATIVE).strip()
        character = (character_description or pr.get("character_description") or "").strip()
        segs = pr.get("segments", [])
        wanted = _parse_clip_spec(clips, len(segs))

        positive, pose, idx, skip, count = [], [], [], [], []
        last_index = max(int(s["index"]) for s in segs) if segs else 0
        for sg in segs:
            if int(sg["index"]) not in wanted:
                continue
            seg_text = (sg.get("positive_prompt") or "").strip()
            fields = {"character": character, "common": common, "segment": seg_text,
                      "motion": sg.get("motion", ""), "camera": sg.get("camera", "")}
            try:
                text = prompt_template.format(**fields)
            except (KeyError, IndexError, ValueError):
                text = f"{common} {seg_text}"
            text = re.sub(r"[ \t]+", " ", text).strip()
            if "{{SUBJECT}}" in text:
                print(f"[ShortsRemake] clip {sg['index']}: prompt still contains a SUBJECT placeholder")
            positive.append(text)
            pose.append((sg.get("motion") or seg_text or "a person moving naturally").strip())
            idx.append(int(sg["index"]))
            skip.append(int(round(float(sg["start"]) * fps)))
            n = int(round(float(sg["duration"]) * fps))
            if int(sg["index"]) == last_index:
                # last segment: never read past the end of the video
                n = ((n - 1) // frame_step) * frame_step + 1
            else:
                # 5s @16fps -> 81 (may borrow up to step-1 frames from the next segment)
                n = (n // frame_step) * frame_step + 1
            count.append(max(frame_step + 1, n))
        if not positive:
            raise RuntimeError("[ShortsRemake] no clips selected")
        print(f"[ShortsRemake] loaded {len(positive)} clips, {W}x{H} @ {fps}fps, frames per clip {count}")
        return (positive, pose, idx, skip, count, negative, common, video_path, float(fps), W, H, len(positive),
                os.path.dirname(prompts_json))


# --------------------------------------------------------------------------- #
# 4. save one clip
# --------------------------------------------------------------------------- #
class ShortsClipSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "segment_index": ("INT", {"default": 1, "min": 1, "max": 9999, "forceInput": True}),
                "out_dir": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "clip"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("clip_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "ShortsRemake"

    def save(self, video, segment_index, out_dir, prefix="clip"):
        from comfy_api.latest import VideoCodec, VideoContainer

        out_dir = out_dir.strip() or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}_{int(segment_index):02d}.mp4")
        video.save_to(path, format=VideoContainer.MP4, codec=VideoCodec.H264)
        print(f"[ShortsRemake] saved {path}")
        return {"ui": {"text": [path]}, "result": (path,)}


# --------------------------------------------------------------------------- #
# 5. concat clips
# --------------------------------------------------------------------------- #
class ShortsConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_path": ("STRING", {"forceInput": True}),
                "out_dir": ("STRING", {"default": "", "forceInput": True}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "filename": ("STRING", {"default": "final.mp4"}),
                "reencode": ("BOOLEAN", {"default": True, "tooltip": "Re-encode to H.264 (safe for mixed clips)."}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("final_path", "video")
    OUTPUT_NODE = True
    FUNCTION = "concat"
    CATEGORY = "ShortsRemake"

    def concat(self, clip_path, out_dir, fps, filename, reencode):
        from comfy_api.latest import VideoFromFile

        clips = sorted({p for p in clip_path if p and os.path.isfile(p)}, key=lambda p: os.path.basename(p))
        if not clips:
            raise RuntimeError("[ShortsRemake] no clip files to concatenate")
        od = (out_dir[0] if out_dir else "").strip() or os.path.dirname(clips[0])
        os.makedirs(od, exist_ok=True)
        fps_v = float(fps[0]) if fps else 16.0
        name = (filename[0] if filename else "final.mp4").strip() or "final.mp4"
        reenc = bool(reencode[0]) if reencode else True

        list_path = os.path.join(od, "concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for c in clips:
                safe = c.replace("\\", "/").replace("'", "'\\''")
                f.write("file '" + safe + "'\n")
        final = os.path.join(od, name)
        cmd = [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_path]
        if reenc:
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps_v), "-c:a", "aac"]
        else:
            cmd += ["-c", "copy"]
        cmd.append(final)
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if cp.returncode != 0 or not os.path.isfile(final):
            raise RuntimeError(f"[ShortsRemake] ffmpeg concat failed: {cp.stderr.strip()[:800]}")
        print(f"[ShortsRemake] {len(clips)} clips -> {final}")
        return {"ui": {"text": [final]}, "result": (final, VideoFromFile(final))}


# --------------------------------------------------------------------------- #
# 0. YouTube (or any yt-dlp URL) -> local mp4
# --------------------------------------------------------------------------- #
def _sanitize_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name or "").strip(" ._")
    return name[:80] or "video"


LEADING_BRACKET = re.compile(r"^\s*[\[(【（].*?[\])】）]\s*")
TITLE_KEEP = re.compile(r"[\w \-]")  # letters/digits of any script, space, hyphen


def _title_stem(title: str, fallback: str = "video", max_len: int = 40) -> str:
    """File-name stem from a video title: leading [tags] removed, then everything up to the first
    special character (& | # emoji ...), trimmed to max_len.
    'leggings fashion model dance & photo shoot 街拍' -> 'leggings fashion model dance'
    '[4K] 아이돌 댄스 챌린지 | 직캠 #shorts' -> '아이돌 댄스 챌린지'."""
    t = str(title or "").strip()
    for _ in range(3):
        t2 = LEADING_BRACKET.sub("", t)
        if t2 == t:
            break
        t = t2
    out = []
    for ch in t.lstrip(" -_."):
        if TITLE_KEEP.match(ch) and ch != "_":
            out.append(ch)
        else:
            break
    t = re.sub(r"\s+", " ", "".join(out)).strip(" -")
    if len(t) > max_len:
        t = t[:max_len].rstrip(" -")
    return t or _sanitize_name(fallback)


def _ytdlp_ffmpeg_location() -> str:
    """yt-dlp accepts a file path; it detects 'ffmpeg' in the basename (imageio's binary is ffmpeg-win-...exe)."""
    exe = _ffmpeg_exe()
    return exe if os.path.isfile(exe) else os.path.dirname(exe)


def parse_timecode(text) -> float | None:
    """'' -> None ; '80' / '80.5' -> seconds ; '1:20' / '1:20.5' / '1:02:03' -> seconds."""
    t = str(text or "").strip()
    if not t:
        return None
    parts = t.split(":")
    if len(parts) > 3 or not all(p.strip() for p in parts):
        raise ValueError(f"[ShortsRemake] bad time '{text}' (use seconds or m:ss or h:mm:ss)")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def _fmt_tag(sec: float) -> str:
    return f"{int(sec // 60)}m{sec % 60:04.1f}s".replace(".0s", "s")


def _video_rotation(path: str) -> int:
    """Display rotation stored in the container metadata (phone videos): 0 / 90 / 180 / 270."""
    try:
        cp = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
        txt = cp.stderr or ""
        m = re.search(r"rotation of (-?[\d.]+) degrees", txt) or re.search(r"\brotate\s*:\s*(-?\d+)", txt)
        if m:
            return int(round(float(m.group(1)))) % 360
    except Exception:  # noqa: BLE001
        pass
    return 0


ROTATE_FILTERS = {"90": "transpose=1", "180": "transpose=1,transpose=1", "270": "transpose=2"}


def trim_video(src: str, start, end, out_path: str, rotate: str = "auto") -> str:
    """Frame-accurate cut / re-encode to H.264+AAC. start/end in seconds, either may be None.
    ffmpeg applies the container rotation metadata automatically (the output is physically upright);
    rotate = 90 / 180 / 270 additionally rotates clockwise for videos that are sideways without metadata."""
    cmd = [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", src]
    if str(rotate) in ROTATE_FILTERS:
        cmd += ["-vf", ROTATE_FILTERS[str(rotate)]]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out_path]
    cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if cp.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(f"[ShortsRemake] ffmpeg trim failed: {cp.stderr.strip()[:800]}")
    return out_path


class ShortsYouTubeDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False,
                                   "placeholder": "https://www.youtube.com/watch?v=...  (or a local video path)"}),
                "start": ("STRING", {"default": "", "placeholder": "empty = from the beginning   e.g. 1:20 or 80",
                                     "tooltip": "Clip start. Seconds (80, 80.5) or m:ss / h:mm:ss. Leave both empty to keep the whole video."}),
                "end": ("STRING", {"default": "", "placeholder": "empty = to the end   e.g. 1:50 or 110",
                                   "tooltip": "Clip end. Seconds or m:ss / h:mm:ss."}),
                "out_dir": ("STRING", {"default": "", "placeholder": "empty = ComfyUI output folder (output/)"}),
                "max_height": ("INT", {"default": 1080, "min": 144, "max": 4320, "step": 1,
                                       "tooltip": "Highest video resolution to download (720/1080 is enough for analysis + pose driving)."}),
                "filename": ("STRING", {"default": "", "placeholder": "empty = <YYYYMMDD>_<video title>"}),
                "force_redownload": ("BOOLEAN", {"default": False}),
                "ensure_h264": ("BOOLEAN", {"default": True,
                                            "tooltip": "If the video is not H.264 (e.g. AV1/VP9 from YouTube) or carries rotation metadata (phone video), "
                                                       "re-encode it once to <name>_h264.mp4: fast to decode and physically upright."}),
                "rotate": (["auto", "90", "180", "270"], {"default": "auto",
                           "tooltip": "auto: apply the rotation stored in the file. 90/180/270: rotate clockwise in addition (for videos that are sideways without metadata)."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("video_path", "title", "duration", "full_video_path")
    FUNCTION = "download"
    CATEGORY = "ShortsRemake"
    DESCRIPTION = ("Downloads a YouTube (yt-dlp) URL as mp4, or takes a local video path, and optionally cuts the "
                   "start..end clip out of it (frame accurate, re-encoded). Returns the clip path (or the full video when no range).")

    @classmethod
    def IS_CHANGED(cls, url, start, end, out_dir, max_height, filename, force_redownload, ensure_h264=True, rotate="auto"):
        if force_redownload:
            return float("nan")
        return f"{url}|{start}|{end}|{out_dir}|{max_height}|{filename}|{ensure_h264}|{rotate}"

    def _fetch(self, url, out_dir, max_height, filename, force_redownload):
        """-> (full_video_path, title, duration)"""
        if os.path.isfile(os.path.expanduser(url)):
            p = os.path.abspath(os.path.expanduser(url))
            dur, _, _, _ = _video_info(p)
            return p, os.path.splitext(os.path.basename(p))[0], float(dur)

        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError("[ShortsRemake] yt-dlp is not installed. Run: python_embeded\\python.exe -m pip install yt-dlp") from None

        od = self._out_dir(out_dir)
        if (filename or "").strip():
            stem = _sanitize_name(filename)
        else:
            # <YYYYMMDD>_<video title>.mp4 ; a download of the same title from an earlier day is reused
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as probe:
                info0 = probe.extract_info(url, download=False) or {}
            if info0.get("entries"):
                info0 = info0["entries"][0]
            safe_title = _title_stem(info0.get("title") or "", info0.get("id") or "video")
            stem = time.strftime("%Y%m%d") + "_" + safe_title
            if not force_redownload:
                cands = [p for p in glob.glob(os.path.join(od, f"*_{glob.escape(safe_title)}.mp4")) if os.path.isfile(p)]
                if cands:
                    stem = os.path.splitext(os.path.basename(max(cands, key=os.path.getmtime)))[0]
                    print(f"[ShortsRemake] reusing earlier download {stem}.mp4")
        h = int(max_height)
        opts = {
            # prefer H.264 (avc1): decodes far faster than AV1/VP9 in OpenCV / VHS
            "format": (f"bestvideo[vcodec^=avc1][height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
                       f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
                       f"bestvideo[height<={h}]+bestaudio/best[height<={h}]/best"),
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(od, stem + ".%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "overwrites": bool(force_redownload),
            "ffmpeg_location": _ytdlp_ffmpeg_location(),
            "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
            "retries": 3,
        }
        print(f"[ShortsRemake] downloading {url} -> {od}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info.get("entries"):
                info = info["entries"][0]
            path = None
            for rd_ in info.get("requested_downloads") or []:
                if rd_.get("filepath") and os.path.isfile(rd_["filepath"]):
                    path = rd_["filepath"]
                    break
            if path is None:
                guess = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp4"
                path = guess if os.path.isfile(guess) else ydl.prepare_filename(info)
        if not os.path.isfile(path):
            raise RuntimeError(f"[ShortsRemake] download finished but file not found: {path}")
        path = os.path.abspath(path)
        title = info.get("title") or os.path.splitext(os.path.basename(path))[0]
        dur = float(info.get("duration") or 0.0)
        if dur <= 0:
            dur, _, _, _ = _video_info(path)
        print(f"[ShortsRemake] downloaded '{title}' ({dur:.1f}s) -> {path}")
        return path, title, dur

    @staticmethod
    def _out_dir(out_dir):
        od = (out_dir or "").strip().strip('"')
        if not od:
            try:
                import folder_paths
                comfy_out = os.path.abspath(folder_paths.get_output_directory())
                default_out = os.path.join(os.path.abspath(folder_paths.base_path), "output")
                # ComfyUI-Easy-Install layout: <root>\ComfyUI\ and <root>\output\ side by side
                root_out = os.path.join(os.path.dirname(os.path.abspath(folder_paths.base_path)), "output")
                if os.path.normcase(comfy_out) != os.path.normcase(default_out):
                    od = comfy_out                    # the launcher set --output-directory: go where ComfyUI saves
                else:
                    od = root_out if os.path.isdir(root_out) else comfy_out
            except Exception:  # noqa: BLE001
                od = os.path.join(os.getcwd(), "output")
        os.makedirs(od, exist_ok=True)
        return od

    def download(self, url, start, end, out_dir, max_height, filename, force_redownload, ensure_h264=True, rotate="auto"):
        url = (url or "").strip().strip('"')
        if not url:
            raise ValueError("[ShortsRemake] url is empty (YouTube URL or local video path)")
        t0, t1 = parse_timecode(start), parse_timecode(end)
        full, title, dur = self._fetch(url, out_dir, max_height, filename, force_redownload)
        rotate = str(rotate or "auto")
        if t0 is None and t1 is None:
            codec = _video_codec(full)
            meta_rot = _video_rotation(full)
            reasons = []
            if ensure_h264 and codec and codec not in H264_FOURCC:
                reasons.append(f"codec {codec}")
            if meta_rot:
                reasons.append(f"rotation metadata {meta_rot} deg")
            if rotate != "auto":
                reasons.append(f"forced rotate {rotate}")
            if reasons:
                stem = os.path.splitext(os.path.basename(full))[0]
                od = os.path.dirname(full) if os.path.isfile(url) and not (out_dir or "").strip() else self._out_dir(out_dir)
                h264 = os.path.join(od, f"{stem}_h264.mp4")
                if os.path.isfile(h264) and not force_redownload:
                    print(f"[ShortsRemake] H.264 copy already exists {h264}")
                else:
                    print(f"[ShortsRemake] re-encoding to upright H.264 ({', '.join(reasons)}): {h264}")
                    trim_video(full, None, None, h264, rotate)
                cdur, _, _, _ = _video_info(h264)
                return (h264, title, float(cdur), full)
            return (full, title, dur, full)

        if t0 is not None and t1 is not None and t1 <= t0:
            raise ValueError(f"[ShortsRemake] end ({end}) must be after start ({start})")
        if dur > 0 and t0 is not None and t0 >= dur:
            raise ValueError(f"[ShortsRemake] start ({start}) is beyond the video length ({dur:.1f}s)")
        if dur > 0 and t1 is not None and t1 > dur:
            t1 = dur
        tag = f"{_fmt_tag(t0 or 0.0)}-{_fmt_tag(t1) if t1 is not None else 'end'}"
        stem = os.path.splitext(os.path.basename(full))[0]
        od = os.path.dirname(full) if os.path.isfile(url) and not (out_dir or "").strip() else self._out_dir(out_dir)
        clip = os.path.join(od, f"{stem}_{tag}.mp4")
        if os.path.isfile(clip) and not force_redownload:
            print(f"[ShortsRemake] clip already exists {clip}")
        else:
            print(f"[ShortsRemake] trimming {t0 or 0:.2f}s -> {t1 if t1 is not None else dur:.2f}s -> {clip}")
            trim_video(full, t0, t1, clip, rotate)
        cdur, _, _, _ = _video_info(clip)
        return (clip, f"{title} [{tag}]", float(cdur), full)


# --------------------------------------------------------------------------- #
# 1b. reference frame composition helpers (profile + background + props -> one image)
# --------------------------------------------------------------------------- #
DEFAULT_REF_INSTRUCTION = (
    "Create one photorealistic image. Picture 1 shows the person to use: keep this person's face, hair, skin, body "
    "shape and clothing exactly as shown in Picture 1. {background} {props} {framing} "
    "One person only, sharp and well lit. No text, no captions, no watermark, no borders."
)

BG_CLAUSE_IMAGE = ("Place this person inside the scene shown in Picture {n}: keep that scene's layout, furniture, "
                   "lighting, colors and perspective unchanged.")
BG_CLAUSE_FRAME = ("Picture {n} is a frame from the original video. Put this person in exactly the position, pose, "
                   "scale and framing of the person in Picture {n}, completely replacing that person, and keep the "
                   "rest of Picture {n} (background, objects, lighting, camera angle) unchanged.")
BG_CLAUSE_TEXT = "Background and setting: {common}"
PROPS_CLAUSE = "The person holds or uses the objects shown in Picture {n}; keep those objects' look identical."
FRAMING_CLAUSE = "Camera framing: {camera}. The person's pose: {motion}."


def _load_image_file(path: str) -> torch.Tensor:
    from PIL import Image, ImageOps

    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _save_image_tensor(img: torch.Tensor, path: str) -> None:
    from PIL import Image

    if img.ndim == 4:
        img = img[0]
    arr = (img.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=4)


def _resolve_prompts_and_video(prompts_json: str) -> tuple[dict, str, str]:
    """prompts_json may be a prompts.json path OR a video path (then prompts are empty). -> (prompts, video_path, prompts_dir)"""
    p = os.path.expanduser((prompts_json or "").strip().strip('"'))
    if not p:
        raise ValueError("[ShortsRemake] prompts_json is empty (prompts.json path or video path)")
    if p.lower().endswith(".json"):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[ShortsRemake] prompts.json not found: {p}")
        with open(p, encoding="utf-8") as f:
            pr = json.load(f)
        video = pr.get("video_file", "")
        if not os.path.isfile(video):
            cand = os.path.join(os.path.dirname(os.path.dirname(p)), os.path.basename(video))
            if os.path.isfile(cand):
                video = cand
            else:
                raise FileNotFoundError(f"[ShortsRemake] source video not found: {video}")
        return pr, video, os.path.dirname(p)
    if os.path.isfile(p):
        vp = Path(p)
        return {}, p, str(vp.parent / f"{vp.stem}_prompts")
    raise FileNotFoundError(f"[ShortsRemake] not found: {p}")


class ShortsReferenceSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": ("IMAGE", {"tooltip": "Picture 1: the person to put into the video."}),
                "prompts_file": (_list_prompts_choices(), {"tooltip": "prompts.json files found under output/ and input/, newest first."}),
                "prompts_json": ("STRING", {"default": "", "placeholder": "empty = use prompts_file above; or type a path (prompts.json or a video)"}),
                "background_mode": (["video_first_frame", "prompt_only"], {"default": "video_first_frame",
                                     "tooltip": "When no background image is connected: use the original video's first frame (same scene, person replaced) or describe the background with the analysed prompt only."}),
                "max_side": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 16,
                                     "tooltip": "Composed reference size (video aspect ratio is kept)."}),
                "divisible_by": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8}),
                "instruction": ("STRING", {"default": DEFAULT_REF_INSTRUCTION, "multiline": True,
                                           "tooltip": "{background} {props} {framing} {common} placeholders are filled automatically."}),
                "extra_instruction": ("STRING", {"default": "", "multiline": True,
                                                 "placeholder": "optional, appended as-is (e.g. 'wearing a red jacket')"}),
            },
            "optional": {
                "background": ("IMAGE", {"tooltip": "Optional: a background/scene image."}),
                "props": ("IMAGE", {"tooltip": "Optional: an image of props/objects the person should hold or use."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "INT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image1", "image2", "image3", "prompt", "width", "height", "first_frame", "prompts_dir", "video_path")
    FUNCTION = "setup"
    CATEGORY = "ShortsRemake"
    DESCRIPTION = ("Arranges profile / background / props images for TextEncodeQwenImageEditPlus (image1..3), writes the "
                   "composition instruction and gives the video-shaped canvas size. Missing images are passed as None.")

    @classmethod
    def IS_CHANGED(cls, prompts_file, prompts_json, **kw):
        return _file_sig(_resolve_pick(prompts_json, prompts_file))

    @classmethod
    def VALIDATE_INPUTS(cls, prompts_file, prompts_json):
        return True

    def setup(self, profile, prompts_file, prompts_json, background_mode, max_side, divisible_by, instruction, extra_instruction,
              background=None, props=None):
        pr, video, pdir = _resolve_prompts_and_video(_resolve_pick(prompts_json, prompts_file))
        w = int(pr.get("width") or 0)
        h = int(pr.get("height") or 0)
        if not w or not h:
            _, _, w, h = _video_info(video)
        first = _read_frames_at(video, [0.05], max(max_side, 1024))
        s = min(1.0, max_side / max(w, h))
        d = max(8, int(divisible_by))
        W = max(d, int(round(w * s / d)) * d)
        H = max(d, int(round(h * s / d)) * d)

        seg0 = (pr.get("segments") or [{}])[0]
        common = (pr.get("common_prompt") or "").strip()
        camera = (seg0.get("camera") or "").strip() or "same framing as the original video"
        motion = (seg0.get("motion") or "").strip() or "natural, relaxed pose facing the camera"

        images = [profile]
        clauses = {"background": "", "props": "", "framing": FRAMING_CLAUSE.format(camera=camera, motion=motion),
                   "common": common}
        if background is not None:
            images.append(background)
            clauses["background"] = BG_CLAUSE_IMAGE.format(n=len(images))
        elif background_mode == "video_first_frame":
            images.append(first)
            clauses["background"] = BG_CLAUSE_FRAME.format(n=len(images))
        elif common:
            clauses["background"] = BG_CLAUSE_TEXT.format(common=common)
        if props is not None:
            images.append(props)
            clauses["props"] = PROPS_CLAUSE.format(n=len(images))
        while len(images) < 3:
            images.append(None)

        try:
            text = (instruction or DEFAULT_REF_INSTRUCTION).format(**clauses)
        except (KeyError, IndexError, ValueError):
            text = DEFAULT_REF_INSTRUCTION.format(**clauses)
        if (extra_instruction or "").strip():
            text += " " + extra_instruction.strip()
        text = re.sub(r"\s+", " ", text).strip()
        used = ["profile",
                "background" if background is not None else ("video frame" if images[1] is not None else "-"),
                "props" if props is not None else "-"]
        print(f"[ShortsRemake] reference canvas {W}x{H}, pictures: {used}")
        return (images[0], images[1], images[2], text, W, H, first, pdir, video)


class ShortsReferenceSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompts_dir": ("STRING", {"default": "", "forceInput": True}),
                "filename": ("STRING", {"default": "reference.png"}),
                "copy_to_input": ("BOOLEAN", {"default": True, "tooltip": "Also copy into ComfyUI/input so Load Image can pick it."}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("reference_path", "image")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "ShortsRemake"

    def save(self, image, prompts_dir, filename, copy_to_input):
        od = (prompts_dir or "").strip() or os.getcwd()
        name = (filename or "reference.png").strip() or "reference.png"
        if not name.lower().endswith(".png"):
            name += ".png"
        path = os.path.join(od, name)
        _save_image_tensor(image, path)
        msg = path
        if copy_to_input:
            try:
                import folder_paths
                dst = os.path.join(folder_paths.get_input_directory(), f"{os.path.basename(od.rstrip('/\\'))}_{name}")
                shutil.copyfile(path, dst)
                msg += f"\n(copied to input/{os.path.basename(dst)})"
            except Exception as e:  # noqa: BLE001
                print(f"[ShortsRemake] copy to input failed: {e}")
        print(f"[ShortsRemake] reference saved {path}")
        return {"ui": {"text": [msg]}, "result": (path, image[:1])}


class ShortsReferenceLoader:
    """Entry point of stage 2: takes the prompts.json path, hands it on to ShortsPromptsLoader and picks the
    reference image. (It must not depend on ShortsPromptsLoader: the character-description QwenVL sits between
    this node and the loader, which would otherwise form a cycle.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_file": (_list_prompts_choices(), {"tooltip": "prompts.json files found under output/ and input/, newest first."}),
                "prompts_json": ("STRING", {"default": "", "placeholder": "empty = use prompts_file above; or type a path"}),
                "filename": ("STRING", {"default": "reference.png"}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "Direct link from the composition step (highest priority)."}),
                "fallback": ("IMAGE", {"tooltip": "Used when no composed reference exists (e.g. the plain profile photo)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "prompts_json", "source")
    FUNCTION = "load"
    CATEGORY = "ShortsRemake"
    DESCRIPTION = "Picks the reference image: linked reference > <prompts.json folder>/reference.png > fallback image. Passes prompts_json through."

    @classmethod
    def IS_CHANGED(cls, prompts_file, prompts_json, filename, **kw):
        pj = _resolve_pick(prompts_json, prompts_file)
        return _file_sig(pj) + "|" + _file_sig(os.path.join(os.path.dirname(pj), (filename or "reference.png").strip()))

    @classmethod
    def VALIDATE_INPUTS(cls, prompts_file, prompts_json):
        return True

    def load(self, prompts_file, prompts_json, filename, reference=None, fallback=None):
        pj = _resolve_pick(prompts_json, prompts_file)
        if not pj or not os.path.isfile(pj):
            raise FileNotFoundError(f"[ShortsRemake] prompts.json not found: {pj}")
        if reference is not None:
            return (reference[:1], pj, "linked reference")
        path = os.path.join(os.path.dirname(pj), (filename or "reference.png").strip())
        if os.path.isfile(path):
            print(f"[ShortsRemake] reference image {path}")
            return (_load_image_file(path), pj, path)
        if fallback is not None:
            print("[ShortsRemake] no composed reference, using fallback image")
            return (fallback[:1], pj, "fallback image")
        raise FileNotFoundError(f"[ShortsRemake] no reference image: {path} (connect a fallback image)")


# --------------------------------------------------------------------------- #
# 6. prompts.json -> fixed string outputs (for hand-built multi-segment graphs, e.g. Wan 2.2 i2v 6-seg)
# --------------------------------------------------------------------------- #
FANOUT_SLOTS = 8


class ShortsPromptsFanout:
    """Reads prompts.json and exposes segment prompts as separate STRING outputs so they can be wired
    into a workflow that has one text box per segment (no list execution involved)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_file": (_list_prompts_choices(), {"tooltip": "prompts.json files found under output/ and input/, newest first."}),
                "prompts_json": ("STRING", {"default": "", "placeholder": "empty = use prompts_file above; or type a path"}),
                "first_segment": ("INT", {"default": 1, "min": 1, "max": 999,
                                          "tooltip": "prompts.json segment index that goes to seg_1 (e.g. 7 to render segments 7..14)."}),
                "template": ("STRING", {"default": "{segment}", "multiline": True,
                                        "tooltip": "Per-segment text. Placeholders: {segment} {motion} {camera} {common} {index}"}),
                "empty_text": ("STRING", {"default": "", "tooltip": "Used for seg_N outputs beyond the last segment."}),
            }
        }

    RETURN_TYPES = tuple(["STRING"] * FANOUT_SLOTS + ["STRING", "STRING", "INT", "STRING", "INT"])
    RETURN_NAMES = tuple([f"seg_{i + 1}" for i in range(FANOUT_SLOTS)] + ["common", "negative", "count", "summary", "n_slots"])
    FUNCTION = "fanout"
    CATEGORY = "ShortsRemake"

    @classmethod
    def IS_CHANGED(cls, prompts_file, prompts_json, **kw):
        return _file_sig(_resolve_pick(prompts_json, prompts_file))

    @classmethod
    def VALIDATE_INPUTS(cls, prompts_file, prompts_json):
        return True

    def fanout(self, prompts_file, prompts_json, first_segment, template, empty_text):
        pj = _resolve_pick(prompts_json, prompts_file)
        if not pj or not os.path.isfile(pj):
            raise FileNotFoundError(f"[ShortsRemake] prompts.json not found: {pj}")
        with open(pj, encoding="utf-8") as f:
            pr = json.load(f)
        segs = pr.get("segments", [])
        common = (pr.get("common_prompt") or "").strip()
        negative = (pr.get("negative_prompt") or DEFAULT_NEGATIVE).strip()
        by_index = {int(sg.get("index", i + 1)): sg for i, sg in enumerate(segs)}
        outs, lines = [], []
        for k in range(FANOUT_SLOTS):
            idx = int(first_segment) + k
            sg = by_index.get(idx)
            if sg is None:
                outs.append(empty_text or "")
                continue
            fields = {"segment": (sg.get("positive_prompt") or "").strip(), "motion": (sg.get("motion") or "").strip(),
                      "camera": (sg.get("camera") or "").strip(), "common": common, "index": idx}
            try:
                text = (template or "{segment}").format(**fields)
            except (KeyError, IndexError, ValueError):
                text = fields["segment"]
            text = re.sub(r"\s+", " ", text).strip()
            outs.append(text)
            lines.append(f"[seg_{k + 1} = clip {idx} {sg.get('label', '')}] {text}")
        summary = "\n".join(lines) if lines else "(no segments)"
        print(f"[ShortsRemake] fanout: {len(lines)} of {FANOUT_SLOTS} slots filled from {len(segs)} segments (first={first_segment})")
        return tuple(outs + [common, negative, len(segs), summary, len(lines)])


# --------------------------------------------------------------------------- #
# 7. lazy collector: run only the first `count` segment chains and concatenate their frames
# --------------------------------------------------------------------------- #
COLLECT_SLOTS = 8


class ShortsSegmentsCollect:
    """seg_1..seg_N are lazy: only the first `count` are requested, so the sampler chains of unused
    segments never execute (the graph simply ends after the last available segment)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 1, "min": 0, "max": COLLECT_SLOTS, "forceInput": True,
                                  "tooltip": "How many segments exist (connect Shorts Prompts Fanout -> n_slots)."}),
                "drop_duplicate_first_frame": ("BOOLEAN", {"default": True,
                                                           "tooltip": "Segments 2+ start with the previous segment's last frame; drop it."}),
            },
            "optional": {f"seg_{i + 1}": ("IMAGE", {"lazy": True}) for i in range(COLLECT_SLOTS)},
        }

    RETURN_TYPES = tuple(["IMAGE", "IMAGE", "INT"] + ["IMAGE"] * COLLECT_SLOTS)
    RETURN_NAMES = tuple(["frames", "last_frame", "count"] + [f"seg_{i + 1}" for i in range(COLLECT_SLOTS)])
    FUNCTION = "collect"
    CATEGORY = "ShortsRemake"
    DESCRIPTION = ("Lazy: only the first `count` segments run. frames = all of them joined; seg_N = that segment's own "
                   "frames (for per-segment previews/saves) - segments beyond count are execution-blocked, so their "
                   "save nodes are skipped instead of forcing the segment to render.")

    def check_lazy_status(self, count, drop_duplicate_first_frame=True, **kw):
        n = max(0, min(int(count), COLLECT_SLOTS))
        return [f"seg_{i}" for i in range(1, n + 1) if kw.get(f"seg_{i}") is None]

    def collect(self, count, drop_duplicate_first_frame=True, **kw):
        n = max(0, min(int(count), COLLECT_SLOTS))
        segs = [kw.get(f"seg_{i}") for i in range(1, n + 1)]
        segs = [t for t in segs if t is not None]
        if not segs:
            raise ValueError("[ShortsRemake] no segments to collect (count is 0 or seg_1 is not connected)")
        h, w = segs[0].shape[1], segs[0].shape[2]
        parts = []
        for i, t in enumerate(segs):
            if t.shape[1] != h or t.shape[2] != w:
                import comfy.utils
                t = comfy.utils.common_upscale(t.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)
            if i > 0 and drop_duplicate_first_frame and t.shape[0] > 1:
                t = t[1:]
            parts.append(t)
        merged = torch.cat(parts, dim=0)
        print(f"[ShortsRemake] collected {len(segs)} segment(s) -> {merged.shape[0]} frames")
        try:
            from comfy_execution.graph_utils import ExecutionBlocker
        except ImportError:  # older ComfyUI
            from comfy_execution.graph import ExecutionBlocker
        per_seg = [segs[i] if i < len(segs) else ExecutionBlocker(None) for i in range(COLLECT_SLOTS)]
        return tuple([merged, merged[-1:], len(segs)] + per_seg)


# --------------------------------------------------------------------------- #
# 8. free VRAM between stages (QwenVL holds its 16 GB model per node instance)
# --------------------------------------------------------------------------- #
def _free_qwenvl_models() -> int:
    """Call clear() on every live QwenVL node instance that still holds a model."""
    import gc

    n = 0
    for obj in gc.get_objects():
        try:
            cls_name = type(obj).__name__
            if "QwenVL" not in cls_name and not any("QwenVL" in b.__name__ for b in type(obj).__mro__[1:]):
                continue
            if getattr(obj, "model", None) is not None and callable(getattr(obj, "clear", None)):
                obj.clear()
                n += 1
        except Exception:  # noqa: BLE001
            continue
    return n


class ShortsFreeVRAM:
    """Pass-through node: whatever comes in goes out unchanged, but on the way every QwenVL model and
    (optionally) all ComfyUI-managed models are unloaded. Put it between stage 1 and the video stages."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"forceInput": True}),
                "unload_comfy_models": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Also unload ComfyUI-managed models (Wan, Qwen-Image, text encoders)."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "free"
    CATEGORY = "ShortsRemake"
    DESCRIPTION = "Unloads QwenVL (and optionally all ComfyUI) models, then passes its input through."

    @classmethod
    def IS_CHANGED(cls, value, unload_comfy_models):
        return float("nan")  # always run

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True  # accepts any input type

    def free(self, value, unload_comfy_models=True):
        import gc

        before = None
        try:
            if torch.cuda.is_available():
                before = torch.cuda.memory_allocated() / 1e9
        except Exception:  # noqa: BLE001
            pass
        n = _free_qwenvl_models()
        if unload_comfy_models:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache(True)
            except Exception as e:  # noqa: BLE001
                print(f"[ShortsRemake] unload_all_models failed: {e}")
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                after = torch.cuda.memory_allocated() / 1e9
                print(f"[ShortsRemake] free VRAM: {n} QwenVL instance(s) cleared, allocated {before:.1f} GB -> {after:.1f} GB")
            else:
                print(f"[ShortsRemake] free VRAM: {n} QwenVL instance(s) cleared")
        except Exception:  # noqa: BLE001
            pass
        return (value,)


# --------------------------------------------------------------------------- #
# 9. generation size from an image's aspect ratio
# --------------------------------------------------------------------------- #
class ShortsSizeFromImage:
    """width/height for the video generator in the same aspect ratio as the given image
    (long side = max_side, both multiples of divisible_by). 1080x1920 -> 720x1280, 1920x1080 -> 1280x720."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_side": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 16,
                                     "tooltip": "Length of the longer side. Wan 2.2 14B: 1280 (720p) or 832/960 for speed."}),
                "divisible_by": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("width", "height", "image", "info")
    FUNCTION = "size"
    CATEGORY = "ShortsRemake"

    def size(self, image, max_side, divisible_by):
        h, w = int(image.shape[1]), int(image.shape[2])
        sc = max_side / max(w, h)
        d = max(8, int(divisible_by))
        W = max(d, int(round(w * sc / d)) * d)
        H = max(d, int(round(h * sc / d)) * d)
        info = f"{w}x{h} -> {W}x{H} ({'portrait' if H > W else 'landscape' if W > H else 'square'})"
        print(f"[ShortsRemake] size from image: {info}")
        return (W, H, image, info)


# --------------------------------------------------------------------------- #
# 10. free-form question to a GGUF vision model of ComfyUI-QwenVL-Mod
# --------------------------------------------------------------------------- #
QWEN_GGUF_MISSING = "(ComfyUI-QwenVL-Mod not loaded)"
QWEN_GGUF_SYSTEM = "You are a helpful vision-language assistant. Answer directly with the final answer only. No <think> and no reasoning."
_QWEN_GGUF_LIST = (0.0, [])


def _qwen_gguf_module():
    """ComfyUI-QwenVL-Mod registers its files as top-level modules; its chat service finds them the same way."""
    import sys
    return sys.modules.get("AILab_QwenVL_GGUF")


def _qwen_gguf_models() -> list[str]:
    """Vision-capable catalog entries, the ones whose file is already on disk first (so the default works offline)."""
    global _QWEN_GGUF_LIST
    stamp, cached = _QWEN_GGUF_LIST
    if cached and time.time() - stamp < 30:
        return cached
    mod = _qwen_gguf_module()
    if mod is None:
        return [QWEN_GGUF_MISSING]
    models = (mod.GGUF_VL_CATALOG.get("models") or {})
    present, absent = [], []
    for name in sorted(k for k, e in models.items() if (e or {}).get("mmproj_filename")):
        r = mod._resolve_model_entry(name)
        filename = Path(r.model_filename)
        default = mod._resolve_base_dir(mod.GGUF_VL_CATALOG.get("base_dir") or "LLM/GGUF") / mod._safe_dirname(r.author or "") / r.repo_dirname / filename.name
        on_disk = filename.exists() if filename.is_absolute() else (
            default.exists() or mod.find_in_llm_paths(r.model_filename, r.author or "", r.repo_dirname or "") is not None)
        (present if on_disk else absent).append(name)
    _QWEN_GGUF_LIST = (time.time(), (present + absent) or [QWEN_GGUF_MISSING])
    return _QWEN_GGUF_LIST[1]


class ShortsQwenGGUFVision:
    """Sends system prompt + prompt (+ image) to a GGUF model from ComfyUI-QwenVL-Mod's catalog and returns the raw
    answer. Unlike that pack's own nodes no preset template is appended, and unlike its chat endpoint the model
    is not put into the workflow-assistant role, so a long instruction gets a long answer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_qwen_gguf_models(), {"tooltip": "GGUF vision models of ComfyUI-QwenVL-Mod. Files already downloaded are listed first."}),
                "system_prompt": ("STRING", {"default": QWEN_GGUF_SYSTEM, "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 3072, "min": 64, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 4294967295}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ask"
    CATEGORY = "ShortsRemake"

    def __init__(self):
        self.base = None

    def ask(self, model_name, system_prompt, prompt, max_tokens, temperature, seed, keep_model_loaded, image=None):
        mod = _qwen_gguf_module()
        if mod is None:
            raise RuntimeError("[ShortsRemake] ComfyUI-QwenVL-Mod is not installed (its GGUF loader is used by this node)")
        if self.base is None:
            self.base = mod.QwenVLGGUFBase()
        self.base._load_model(model_name=model_name, device="auto", ctx=None, n_batch=None, gpu_layers=None,
                              image_max_tokens=None, top_k=None, pool_size=None)
        images = []
        if image is not None and self.base.chat_handler is not None:
            b64 = mod._tensor_to_base64_png(image[0] if image.ndim == 4 else image)
            if b64:
                images.append(b64)
        user = ("/no_think\n" + prompt) if getattr(self.base, "is_qwen35", False) else prompt
        try:
            text = self.base._invoke(system_prompt=system_prompt, user_prompt=user, images_b64=images,
                                     max_tokens=max_tokens, temperature=temperature, top_p=0.9,
                                     repetition_penalty=1.05, seed=seed, model_name=model_name)
        finally:
            if not keep_model_loaded:
                self.base.clear()
        return (text,)


NODE_CLASS_MAPPINGS = {
    "ShortsVideoSegments": ShortsVideoSegments,
    "ShortsPromptsCollector": ShortsPromptsCollector,
    "ShortsPromptsLoader": ShortsPromptsLoader,
    "ShortsClipSaver": ShortsClipSaver,
    "ShortsConcat": ShortsConcat,
    "ShortsYouTubeDownload": ShortsYouTubeDownload,
    "ShortsReferenceSetup": ShortsReferenceSetup,
    "ShortsReferenceSave": ShortsReferenceSave,
    "ShortsReferenceLoader": ShortsReferenceLoader,
    "ShortsPromptsFanout": ShortsPromptsFanout,
    "ShortsSegmentsCollect": ShortsSegmentsCollect,
    "ShortsFreeVRAM": ShortsFreeVRAM,
    "ShortsSizeFromImage": ShortsSizeFromImage,
    "ShortsQwenGGUFVision": ShortsQwenGGUFVision,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShortsVideoSegments": "Shorts Video Segments",
    "ShortsPromptsCollector": "Shorts Prompts Collector",
    "ShortsPromptsLoader": "Shorts Prompts Loader",
    "ShortsClipSaver": "Shorts Clip Saver",
    "ShortsConcat": "Shorts Concat",
    "ShortsYouTubeDownload": "Shorts YouTube Download / Trim",
    "ShortsReferenceSetup": "Shorts Reference Setup (profile+background+props)",
    "ShortsReferenceSave": "Shorts Reference Save",
    "ShortsReferenceLoader": "Shorts Reference Loader",
    "ShortsPromptsFanout": "Shorts Prompts Fanout (seg_1..8)",
    "ShortsSegmentsCollect": "Shorts Segments Collect (lazy, stops after last segment)",
    "ShortsFreeVRAM": "Shorts Free VRAM (unload QwenVL + models)",
    "ShortsSizeFromImage": "Shorts Size From Image (width/height by aspect)",
    "ShortsQwenGGUFVision": "Shorts Qwen GGUF Vision (QwenVL-Mod model, free prompt)",
}
