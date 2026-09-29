"""ComfyUI-ShortsRemake

Helper nodes for the "analyze a video in N-second segments -> prompts -> regenerate
clip by clip with a swapped person -> concatenate" pipeline.

Nodes
- Shorts Video Segments   : split a video file into N-second segments (frame batches, list output)
- Shorts Prompts Collector: gather VLM JSON answers into prompts.json / comfyui_prompts.txt
- Shorts Prompts Loader   : read prompts.json -> per-clip prompt / frame range lists for generation
- Shorts Clip Saver       : save one generated VIDEO as clip_NN.mp4 next to prompts.json
- Shorts Concat           : concatenate clip_NN.mp4 -> final.mp4 (ffmpeg)

List mechanics: outputs flagged in OUTPUT_IS_LIST make every downstream node run once
per segment, so a normal single-clip generation graph becomes a per-segment loop.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
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


def _read_frames_at(path: str, times: list[float], max_side: int) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, fr = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1].copy())
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        if max_side > 0 and max(h, w) > max_side:
            s = max_side / max(h, w)
            fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"[ShortsRemake] no frames decoded from {path}")
    arr = np.stack(frames).astype(np.float32) / 255.0
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
# 1. Video -> segments
# --------------------------------------------------------------------------- #
class ShortsVideoSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "placeholder": "C:/path/to/video.mp4"}),
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
    def IS_CHANGED(cls, video_path, **kw):
        return _file_sig(video_path)

    def split(self, video_path, split_mode, segment_seconds, min_seconds, scene_threshold, frames_per_segment, max_side):
        video_path = os.path.expanduser(video_path.strip().strip('"'))
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

        frames_out, idx_out, label_out, overview = [], [], [], []
        for s in segs:
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


NODE_CLASS_MAPPINGS = {
    "ShortsVideoSegments": ShortsVideoSegments,
    "ShortsPromptsCollector": ShortsPromptsCollector,
    "ShortsPromptsLoader": ShortsPromptsLoader,
    "ShortsClipSaver": ShortsClipSaver,
    "ShortsConcat": ShortsConcat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShortsVideoSegments": "Shorts Video Segments",
    "ShortsPromptsCollector": "Shorts Prompts Collector",
    "ShortsPromptsLoader": "Shorts Prompts Loader",
    "ShortsClipSaver": "Shorts Clip Saver",
    "ShortsConcat": "Shorts Concat",
}
