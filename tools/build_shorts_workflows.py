"""Build the ShortsRemake workflows (UI/LiteGraph format + API format).

  0_YouTube_Download_Trim.json         YouTube URL -> mp4 (yt-dlp), prints the saved path
  1_Analyze_Prompts.json          video file -> 5s/scene segments -> QwenVL -> prompts.json
  2_Compose_Reference.json       profile + background + props images -> Qwen-Image-Edit-2511 -> reference.png
  3_Replace_Person_WanAnimate2.json
                                         reference.png (or profile) + prompts.json + source video
                                         -> Wan Animate 2 per segment -> clip_NN.mp4 -> final.mp4
  4_ALL_in_One.json                 stages 1 + 2 + 3 in one graph (video file + images in, final.mp4 out)

Stage 2 is flattened from the official video_wan_animate2.json template (loop nodes removed; per-segment
execution comes from ShortsPromptsLoader list outputs). Stage 2 mirrors image_qwen_image_edit_2511.json.

Run:  python_embeded\\python.exe tools\\build_shorts_workflows.py
(ComfyUI's node definitions are imported to write the *.api.json files; takes ~20 s.)
"""
import asyncio
import copy
import io
import json
import os
import sys

# repo root = ComfyUI-Easy-Install folder (this file lives in <root>/tools/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.join(ROOT, "ComfyUI")
TEMPLATES = os.path.join(ROOT, "python_embeded", "Lib", "site-packages", "comfyui_workflow_templates_json", "templates") + os.sep
OUT_DIRS = [
    os.path.join(ROOT, "workflow"),
    os.path.join(ROOT, "ComfyUI", "user", "default", "workflows"),  # usually a junction to workflow/
]
SAMPLE_VIDEO = os.path.join(ROOT, "ComfyUI", "input", "WomanTalking.mp4")
SAMPLE_PROMPTS = os.path.join(ROOT, "ComfyUI", "input", "WomanTalking_prompts", "prompts.json")

DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, jpeg artifacts, watermark, text, logo, subtitles, "
    "deformed, extra limbs, extra fingers, bad anatomy, distorted face, flicker, jitter, "
    "static image, frozen frame, duplicated subject, overexposed, oversaturated"
)

SEG_PROMPT = (
    "These frames are ONE consecutive segment of a video, in time order (first frame = segment start, last = segment end). "
    "You write prompts for an AI video generator (Wan) that will re-create this segment with a DIFFERENT person, "
    "so never describe the person's identity, face, hair, skin, age or clothing.\n"
    "Return ONLY a JSON object, no markdown, with exactly these keys:\n"
    "\"scene_ko\": Korean, 1-2 sentences, what is visible and what happens in this segment.\n"
    "\"positive_prompt\": English, 50-90 words, natural sentences: environment and background, the person's actions and body "
    "motion from first to last frame, camera framing and movement, lighting, colors. No text, logos or captions.\n"
    "\"camera\": English, one short line describing the camera framing and movement.\n"
    "\"motion\": English, one short line describing only the person's motion (for a pose prompt), e.g. "
    "\"a person talking to the camera, nodding and gesturing with one hand\".\n"
    "All values except scene_ko MUST be written in English."
)

COMMON_PROMPT = (
    "Each frame is taken from a different consecutive segment of the SAME video. "
    "Return ONLY a JSON object, no markdown, with exactly these keys:\n"
    "\"summary_ko\": Korean, 1-2 sentences summarizing the whole video.\n"
    "\"common_prompt\": English, 30-60 words describing what stays the same across the whole video: setting, environment, "
    "lighting, color palette, camera style, visual style. Describe looks only, no motion, and do NOT describe the person's "
    "identity, face, hair or clothing.\n"
    "\"person_in_video\": English, 1-2 sentences describing the main person's appearance (reference only).\n"
    "\"negative_prompt\": English, comma-separated list of visual artifacts to avoid for this kind of footage.\n"
    "All values except summary_ko MUST be written in English."
)

CHAR_PROMPT = (
    "Describe the main person in this image for an AI video generation prompt. English, one paragraph of 40-70 words: "
    "gender, apparent age range, hair (color, length, style), face features, skin tone, clothing and accessories, body type, "
    "and any object the person is holding. Looks only: no background description, no motion, no emotions, no name. "
    "Output the paragraph only."
)

PROMPT_TEMPLATE = "Character Description: {character}\nBackground description: {common} {segment}"

QWEN_MODEL = "Qwen3-VL-8B-Instruct"

# Qwen-Image-Edit-2511 (same files as the official image_qwen_image_edit_2511 template)
QIE_UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
QIE_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QIE_VAE = "qwen_image_vae.safetensors"
QIE_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# import the node module directly for its default instruction text
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "shorts_remake_nodes", os.path.join(ROOT, "custom_nodes", "ComfyUI-ShortsRemake", "nodes.py"))
shorts_nodes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shorts_nodes)

REF_INSTRUCTION = shorts_nodes.DEFAULT_REF_INSTRUCTION


def qwen_widgets(prompt, frame_count, max_tokens=512, model=QWEN_MODEL):
    # model_name, quantization, attention_mode, use_torch_compile, device, preset_prompt, custom_prompt,
    # max_tokens, temperature, top_p, num_beams, repetition_penalty, frame_count, video_frame_size,
    # keep_model_loaded, seed, control_after_generate
    return [model, "None (FP16)", "auto", False, "auto", "\U0001F4F9 Video Summary", prompt,
            max_tokens, 0.3, 0.9, 1, 1.2, frame_count, "auto", True, 1, "fixed"]


def node(nid, ntype, pos, size, inputs, outputs, widgets, title=None, extra=None):
    n = {"id": nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {}, "order": 0, "mode": 0,
         "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": ntype}}
    if widgets is not None:
        n["widgets_values"] = widgets
    if title:
        n["title"] = title
    if extra:
        n.update(extra)
    return n


def inp(name, typ, link, widget=False):
    d = {"name": name, "type": typ, "link": link}
    if widget:
        d["widget"] = {"name": name}
    return d


def outp(name, typ, links):
    return {"name": name, "type": typ, "links": links}


def note(nid, pos, size, text, title, color=("#432", "#653")):
    return node(nid, "MarkdownNote", pos, size, [], [], [text], title=title,
                extra={"color": color[0], "bgcolor": color[1]})


def shift(nodes, dx, dy):
    for n in nodes:
        n["pos"] = [n["pos"][0] + dx, n["pos"][1] + dy]
    return nodes


def assemble(wf_id, fragments, extra_links=(), ds=None):
    """fragments: list of (nodes, links{id: [id, o, os, t, ts, type]}). extra_links: cross-fragment links."""
    nodes, links = [], {}
    for ns, ls in fragments:
        nodes += copy.deepcopy(ns)
        links.update(copy.deepcopy(ls))
    by_id = {n["id"]: n for n in nodes}
    assert len(by_id) == len(nodes), "duplicate node ids"
    for l in extra_links:
        links[l[0]] = list(l)
        src, tgt = by_id[l[1]], by_id[l[3]]
        o = src["outputs"][l[2]]
        o["links"] = (o.get("links") or []) + [l[0]]
        tgt["inputs"][l[4]]["link"] = l[0]
    # sanity: every referenced link exists and points where the node says
    for n in nodes:
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                assert i["link"] in links, f"node {n['id']} input {i['name']} -> missing link {i['link']}"
                assert links[i["link"]][3] == n["id"], f"link {i['link']} target mismatch at node {n['id']}"
        for o in n.get("outputs", []):
            for lid in o.get("links") or []:
                assert lid in links and links[lid][1] == n["id"], f"link {lid} origin mismatch at node {n['id']}"
    for i, n in enumerate(nodes):
        n["order"] = i
    return {"id": wf_id, "revision": 0, "last_node_id": max(n["id"] for n in nodes), "last_link_id": max(links),
            "nodes": nodes, "links": list(links.values()), "groups": [], "config": {},
            "extra": {"ds": ds or {"scale": 0.5, "offset": [100, 100]}}, "version": 0.4}


# --------------------------------------------------------------------------- #
# Fragment A: YouTube download + analyze   (node ids 1-19, link ids 1-99)
# --------------------------------------------------------------------------- #
NOTE_ANALYZE = (
    "## 1단계: 영상 분석 → 프롬프트 추출\n\n"
    "**입력**: `Shorts Video Segments` 노드의 `video_path`에 영상 파일 전체 경로.\n"
    "유튜브 영상은 먼저 `0_YouTube_Download_Trim` 워크플로우로 받으면 경로가 `ComfyUI/input/shorts_downloads/<id>.mp4`로 표시됩니다.\n\n"
    "**동작**\n"
    "1. `Shorts Video Segments`가 영상을 구간으로 나누고 구간마다 `frames_per_segment`장을 뽑습니다.\n"
    "   - `split_mode` = fixed : `segment_seconds`(기본 5초) 고정 길이\n"
    "   - `split_mode` = scene : 장면 전환(컷)에서 자름. `segment_seconds`보다 긴 장면은 균등 분할, "
    "`min_seconds`보다 짧은 조각은 이웃과 합침. `scene_threshold`를 낮추면 컷을 더 민감하게 잡음\n"
    "2. 위쪽 QwenVL이 **구간마다 한 번씩** 실행되어 구간 프롬프트(JSON)를 씁니다.\n"
    "3. 아래쪽 QwenVL이 구간별 대표 프레임을 한꺼번에 보고 **공통 프롬프트 · 네거티브**를 씁니다.\n"
    "4. `Shorts Prompts Collector`가 결과를 모아 저장합니다.\n\n"
    "**출력** (`out_dir` 비우면 영상 옆 `<영상이름>_prompts/`)\n"
    "- `prompts.json` : common_prompt, negative_prompt, segments[].positive_prompt / motion / camera / scene_ko\n"
    "- `comfyui_prompts.txt` : 사람이 보는 정리본\n\n"
    "**다음 단계**: `2_Compose_Reference`(프로필+배경+소품 → reference.png) → "
    "`3_Replace_Person_WanAnimate2`(영상 생성). 두 워크플로우의 `prompts_json`에 이 파일 경로를 넣습니다.\n\n"
    "GPU 서버: Qwen3-VL-8B-Instruct(FP16, ~16GB). "
    "이 PC(CPU)에서 확인할 때는 QwenVL 노드를 `QwenVL Advanced (GGUF)`, device=cpu로 바꿔 쓰세요."
)


def frag_analyze(with_youtube=False):
    seg_inputs = [inp("video_path", "STRING", 7, True)] if with_youtube else []
    nodes = [
        note(10, [-1560, 80], [520, 560], NOTE_ANALYZE, "사용법 (1단계 분석)"),
        node(11, "ShortsYouTubeDownload", [-1000, 80], [420, 190], [],
             [outp("video_path", "STRING", [7]), outp("title", "STRING", None), outp("duration", "FLOAT", None)],
             ["https://www.youtube.com/watch?v=...", "", 1080, "", False],
             title="Shorts YouTube Download - 유튜브 링크 (또는 파일 경로)"),
        node(1, "ShortsVideoSegments", [-1000, 330], [420, 250], seg_inputs,
             [outp("segment_frames", "IMAGE", [1]), outp("segment_index", "INT", None), outp("segment_label", "STRING", None),
              outp("overview_frames", "IMAGE", [2]), outp("count", "INT", None), outp("video_path", "STRING", None),
              outp("segments_json", "STRING", [5])],
             ["", "fixed", 5.0, 1.5, 0.5, 8, 384],
             title="Shorts Video Segments - video_path에 영상 파일 경로, 5초 / 장면 단위로 나누기"),
        node(2, "AILab_QwenVL_Advanced", [-540, 80], [460, 620],
             [inp("image", "IMAGE", None), inp("video", "IMAGE", 1)],
             [outp("RESPONSE", "STRING", [3])], qwen_widgets(SEG_PROMPT, 8), title="QwenVL - 구간별 프롬프트 (구간 수만큼 실행)"),
        node(3, "AILab_QwenVL_Advanced", [-540, 760], [460, 620],
             [inp("image", "IMAGE", None), inp("video", "IMAGE", 2)],
             [outp("RESPONSE", "STRING", [4])], qwen_widgets(COMMON_PROMPT, 16), title="QwenVL - 공통 프롬프트 + 네거티브"),
        node(4, "ShortsPromptsCollector", [-40, 80], [460, 300],
             [inp("segment_responses", "STRING", 3), inp("common_response", "STRING", 4), inp("segments_json", "STRING", 5)],
             [outp("summary", "STRING", [6]), outp("prompts_json", "STRING", None)],
             ["", DEFAULT_NEGATIVE, 16]),
        node(5, "PreviewAny", [-40, 440], [640, 500], [inp("source", "*", 6)], [], [None, None, False]),
    ]
    links = {
        1: [1, 1, 0, 2, 1, "IMAGE"],
        2: [2, 1, 3, 3, 1, "IMAGE"],
        3: [3, 2, 0, 4, 0, "STRING"],
        4: [4, 3, 0, 4, 1, "STRING"],
        5: [5, 1, 6, 4, 2, "STRING"],
        6: [6, 4, 0, 5, 0, "STRING"],
    }
    if with_youtube:
        links[7] = [7, 11, 0, 1, 0, "STRING"]
    else:
        nodes = [n for n in nodes if n["id"] != 11]
    return nodes, links


# --------------------------------------------------------------------------- #
# Stage 0: YouTube download only   (node ids 11-13, link id 7)
# --------------------------------------------------------------------------- #
NOTE_DOWNLOAD = (
    "## 0단계: 유튜브 다운로드 + 구간 잘라내기\n\n"
    "`Shorts YouTube Download / Trim`의 `url`에 유튜브 링크(또는 이미 받아둔 파일 경로)를 넣고 Queue.\n\n"
    "**구간만 쓰려면** `start`, `end`에 시간을 넣습니다. `1:20` 처럼 분:초, 또는 `80` 처럼 초. 둘 다 비우면 전체 영상.\n"
    "- 한쪽만 넣어도 됩니다 (`start`만: 거기서 끝까지, `end`만: 처음부터 거기까지)\n"
    "- 잘라낸 파일은 `<원본이름>_1m20s-1m50s.mp4` 처럼 따로 저장되고 원본은 그대로 남습니다\n"
    "- 프레임 단위로 정확하게 자르기 위해 다시 인코딩합니다 (H.264, 화질 손실은 거의 없음)\n\n"
    "**다운로드 옵션**\n"
    "- `max_height`: 받을 최대 해상도 (1080 권장)\n"
    "- `out_dir` 비우면 `ComfyUI/input/shorts_downloads/` 에 `<영상id>.mp4` 로 저장\n"
    "- 같은 링크는 다시 받지 않고 받아둔 파일에서 구간만 다시 자릅니다. 다시 받으려면 `force_redownload` 켜기\n"
    "- 로그인/연령 제한 영상은 받을 수 없습니다\n"
    "- `ensure_h264` (기본 켜짐): AV1/VP9 영상이면 H.264로 한 번 변환해 `<이름>_h264.mp4`를 씁니다. "
    "AV1은 프레임 읽기가 수십 배 느려서 1단계가 몇 분씩 멈춘 것처럼 보입니다. 이미 받아둔 파일도 여기 `url`에 경로를 넣으면 변환됩니다\n\n"
    "결과 **파일 경로가 아래 미리보기에 표시**됩니다. 그 경로를 복사해\n"
    "`1_Analyze_Prompts` 또는 `4_ALL_in_One`의 `Shorts Video Segments` → `video_path`에 붙여 넣으세요."
)


def build_download():
    nodes = [
        note(13, [-1000, 80], [520, 600], NOTE_DOWNLOAD, "사용법 (0단계 다운로드 + 구간)"),
        node(11, "ShortsYouTubeDownload", [-440, 80], [460, 280], [],
             [outp("video_path", "STRING", [7]), outp("title", "STRING", None), outp("duration", "FLOAT", None),
              outp("full_video_path", "STRING", None)],
             ["https://www.youtube.com/watch?v=...", "", "", "", 1080, "", False, True],
             title="Shorts YouTube Download / Trim - 링크(또는 파일) + 구간"),
        node(12, "PreviewAny", [-440, 410], [460, 160], [inp("source", "*", 7)], [], [None, None, False],
             title="저장된 영상 경로 (복사해서 다음 단계에 입력)"),
    ]
    links = {7: [7, 11, 0, 12, 0, "STRING"]}
    return nodes, links


# --------------------------------------------------------------------------- #
# Fragment B: compose reference (Qwen-Image-Edit-2511)   (node ids 20-49, link ids 100-199)
# --------------------------------------------------------------------------- #
NOTE_COMPOSE = (
    "## 2단계: 프로필 + 배경 + 소품 → 참조 이미지 (Qwen-Image-Edit-2511)\n\n"
    "영상에 넣을 **사람 · 배경 · 소품**을 한 장의 참조 이미지(reference.png)로 합칩니다. "
    "3단계 Wan Animate 2는 이 이미지를 원본 영상의 동작대로 움직입니다.\n\n"
    "**입력**\n"
    "- `Load Image - 프로필` (필수): 바꿔 넣을 사람 사진 (상반신/전신)\n"
    "- `Load Image - 배경` (선택): 배경/장소 사진. **없으면 원본 영상의 첫 프레임을 배경으로 써서 사람만 바뀝니다.** "
    "쓰지 않을 때는 노드를 선택하고 Ctrl+B (bypass)\n"
    "- `Load Image - 소품` (선택): 들고 있을 물건/소품 사진. 쓰지 않을 때는 Ctrl+B\n"
    "- `Shorts Reference Setup`의 `prompts_json`: 1단계 결과 prompts.json 경로 (영상 파일 경로를 넣어도 됩니다)\n\n"
    "**동작**\n"
    "1. Setup 노드가 Picture 1=프로필, Picture 2=배경(또는 원본 첫 프레임), Picture 3=소품 순으로 정리하고 "
    "합성 지시문(`instruction`)을 채웁니다. 원하는 문구는 `extra_instruction`에 추가 (예: wearing a red jacket)\n"
    "2. Qwen-Image-Edit-2511이 원본 영상과 같은 비율(`max_side` 1024)로 한 장을 만듭니다.\n"
    "3. `Shorts Reference Save`가 `<prompts 폴더>/reference.png`로 저장하고 ComfyUI/input에도 복사합니다. "
    "3단계 `Shorts Reference Loader`가 이 파일을 자동으로 읽습니다.\n\n"
    "**모델** (ComfyUI 공식 템플릿 image_qwen_image_edit_2511과 동일)\n"
    "- diffusion_models/" + QIE_UNET + "\n"
    "- text_encoders/" + QIE_CLIP + "\n"
    "- vae/" + QIE_VAE + "\n"
    "- (선택) loras/" + QIE_LORA + " : 켜면 KSampler steps 4, cfg 1로 바꾸세요 (기본은 40 steps, cfg 3)\n\n"
    "**팁**: 결과가 마음에 안 들면 KSampler seed를 바꿔 다시 생성. 사람이 너무 작게 나오면 `extra_instruction`에 "
    "'medium shot, upper body fills the frame' 등을 추가. 결과가 원본 프레임과 너무 달라지면 프로필 사진을 배경 없는 정면 사진으로."
)


def frag_compose():
    x0, y0 = -1560, 0
    nodes = [
        note(39, [x0, y0], [560, 700], NOTE_COMPOSE, "사용법 (2단계 참조 이미지)", ("#223", "#335")),
        node(21, "LoadImage", [x0 + 600, y0], [320, 380], [],
             [outp("IMAGE", "IMAGE", [101]), outp("MASK", "MASK", None)], ["profile.png", "image"],
             title="Load Image - 프로필 (바꿔 넣을 인물, 필수)"),
        node(22, "LoadImage", [x0 + 600, y0 + 420], [320, 380], [],
             [outp("IMAGE", "IMAGE", [102]), outp("MASK", "MASK", None)], ["background.png", "image"],
             title="Load Image - 배경 (선택, 안 쓰면 Ctrl+B)", extra={"mode": 4}),
        node(23, "LoadImage", [x0 + 600, y0 + 840], [320, 380], [],
             [outp("IMAGE", "IMAGE", [103]), outp("MASK", "MASK", None)], ["props.png", "image"],
             title="Load Image - 소품 (선택, 안 쓰면 Ctrl+B)", extra={"mode": 4}),
        node(20, "ShortsReferenceSetup", [x0 + 980, y0], [520, 520],
             [inp("profile", "IMAGE", 101), inp("background", "IMAGE", 102), inp("props", "IMAGE", 103)],
             [outp("image1", "IMAGE", [110, 113]), outp("image2", "IMAGE", [111, 114]), outp("image3", "IMAGE", [112, 115]),
              outp("prompt", "STRING", [116]), outp("width", "INT", [117]), outp("height", "INT", [118]),
              outp("first_frame", "IMAGE", None), outp("prompts_dir", "STRING", [130]), outp("video_path", "STRING", None)],
             [SAMPLE_PROMPTS, "video_first_frame", 1024, 16, REF_INSTRUCTION, ""],
             title="Shorts Reference Setup - 프로필+배경+소품 정리"),
        # models
        node(24, "UNETLoader", [x0 + 980, y0 + 580], [400, 82], [], [outp("MODEL", "MODEL", [120])], [QIE_UNET, "default"]),
        node(25, "CLIPLoader", [x0 + 980, y0 + 700], [400, 106], [], [outp("CLIP", "CLIP", [121, 122])], [QIE_CLIP, "qwen_image", "default"]),
        node(26, "VAELoader", [x0 + 980, y0 + 850], [400, 58], [], [outp("VAE", "VAE", [123, 124, 125])], [QIE_VAE]),
        node(27, "ModelSamplingAuraFlow", [x0 + 1420, y0 + 580], [320, 58], [inp("model", "MODEL", 120)],
             [outp("MODEL", "MODEL", [126])], [3.1, "flow"]),
        node(28, "CFGNorm", [x0 + 1420, y0 + 680], [320, 82], [inp("model", "MODEL", 126)], [outp("MODEL", "MODEL", [127])], [1, False]),
        node(29, "LoraLoaderModelOnly", [x0 + 1420, y0 + 800], [320, 82], [inp("model", "MODEL", 127)],
             [outp("MODEL", "MODEL", [128])], [QIE_LORA, 1.0],
             title="Lightning 4-step LoRA (기본 bypass, 켜면 steps 4 / cfg 1)", extra={"mode": 4}),
        # conditioning
        node(30, "TextEncodeQwenImageEditPlus", [x0 + 1560, y0], [440, 220],
             [inp("clip", "CLIP", 121), inp("vae", "VAE", 123), inp("image1", "IMAGE", 110), inp("image2", "IMAGE", 111),
              inp("image3", "IMAGE", 112), inp("prompt", "STRING", 116, True)],
             [outp("CONDITIONING", "CONDITIONING", [140])], [""], title="Qwen Edit Encode - Positive (지시문은 Setup에서)"),
        node(31, "TextEncodeQwenImageEditPlus", [x0 + 1560, y0 + 280], [440, 220],
             [inp("clip", "CLIP", 122), inp("vae", "VAE", 124), inp("image1", "IMAGE", 113), inp("image2", "IMAGE", 114),
              inp("image3", "IMAGE", 115)],
             [outp("CONDITIONING", "CONDITIONING", [141])], [""], title="Qwen Edit Encode - Negative"),
        node(32, "FluxKontextMultiReferenceLatentMethod", [x0 + 2040, y0], [340, 58],
             [inp("conditioning", "CONDITIONING", 140)], [outp("CONDITIONING", "CONDITIONING", [142])], ["index_timestep_zero"]),
        node(33, "FluxKontextMultiReferenceLatentMethod", [x0 + 2040, y0 + 280], [340, 58],
             [inp("conditioning", "CONDITIONING", 141)], [outp("CONDITIONING", "CONDITIONING", [143])], ["index_timestep_zero"]),
        node(34, "EmptySD3LatentImage", [x0 + 2040, y0 + 400], [340, 106],
             [inp("width", "INT", 117, True), inp("height", "INT", 118, True)],
             [outp("LATENT", "LATENT", [144])], [1024, 1024, 1], title="Empty Latent - 원본 영상 비율"),
        node(35, "KSampler", [x0 + 2040, y0 + 560], [340, 262],
             [inp("model", "MODEL", 128), inp("positive", "CONDITIONING", 142), inp("negative", "CONDITIONING", 143),
              inp("latent_image", "LATENT", 144)],
             [outp("LATENT", "LATENT", [145])], [42, "fixed", 40, 3, "euler", "simple", 1], title="KSampler (steps 40 / cfg 3, LoRA 켜면 4 / 1)"),
        node(36, "VAEDecode", [x0 + 2420, y0 + 560], [210, 46], [inp("samples", "LATENT", 145), inp("vae", "VAE", 125)],
             [outp("IMAGE", "IMAGE", [146])], None),
        node(37, "ShortsReferenceSave", [x0 + 2420, y0 + 660], [340, 130],
             [inp("image", "IMAGE", 146), inp("prompts_dir", "STRING", 130)],
             [outp("reference_path", "STRING", None), outp("image", "IMAGE", [147])], ["reference.png", True],
             title="Shorts Reference Save - reference.png"),
        node(38, "PreviewImage", [x0 + 2420, y0], [400, 500], [inp("images", "IMAGE", 147)], [], None,
             title="합성된 참조 이미지"),
    ]
    L = {}

    def add(lid, o, os_, t, ts, typ):
        L[lid] = [lid, o, os_, t, ts, typ]

    add(101, 21, 0, 20, 0, "IMAGE"); add(102, 22, 0, 20, 1, "IMAGE"); add(103, 23, 0, 20, 2, "IMAGE")
    add(110, 20, 0, 30, 2, "IMAGE"); add(111, 20, 1, 30, 3, "IMAGE"); add(112, 20, 2, 30, 4, "IMAGE")
    add(113, 20, 0, 31, 2, "IMAGE"); add(114, 20, 1, 31, 3, "IMAGE"); add(115, 20, 2, 31, 4, "IMAGE")
    add(116, 20, 3, 30, 5, "STRING"); add(117, 20, 4, 34, 0, "INT"); add(118, 20, 5, 34, 1, "INT")
    add(120, 24, 0, 27, 0, "MODEL"); add(121, 25, 0, 30, 0, "CLIP"); add(122, 25, 0, 31, 0, "CLIP")
    add(123, 26, 0, 30, 1, "VAE"); add(124, 26, 0, 31, 1, "VAE"); add(125, 26, 0, 36, 1, "VAE")
    add(126, 27, 0, 28, 0, "MODEL"); add(127, 28, 0, 29, 0, "MODEL"); add(128, 29, 0, 35, 0, "MODEL")
    add(130, 20, 7, 37, 1, "STRING")
    add(140, 30, 0, 32, 0, "CONDITIONING"); add(141, 31, 0, 33, 0, "CONDITIONING")
    add(142, 32, 0, 35, 1, "CONDITIONING"); add(143, 33, 0, 35, 2, "CONDITIONING")
    add(144, 34, 0, 35, 3, "LATENT"); add(145, 35, 0, 36, 0, "LATENT"); add(146, 36, 0, 37, 0, "IMAGE")
    add(147, 37, 1, 38, 0, "IMAGE")
    return nodes, L


# --------------------------------------------------------------------------- #
# Fragment C: replace person (Wan Animate 2)   (my node ids 50-69, link ids 2000+, template ids 245..602)
# --------------------------------------------------------------------------- #
DROP = {288, 537, 595, 604, 605, 635, 636, 639, 642, 645, 646, 647, 648, 649, 651, 652, 653, 654, 661, 667, 669, 670, 671}

NOTE_REPLACE = (
    "## 3단계: 참조 이미지로 인물 교체 영상 생성 (Wan Animate 2)\n\n"
    "**입력**\n"
    "- `Shorts Reference Loader`의 `prompts_json`: 1단계에서 만든 prompts.json 경로 (여기 한 곳만). 원본 영상 경로는 그 안의 video_file을 자동으로 씁니다\n"
    "- 참조 이미지: 같은 노드가 prompts.json 옆의 **reference.png**(2단계 결과)를 자동으로 읽습니다. "
    "없으면 `Load Image - 프로필`의 사진을 그대로 씁니다 (사람만 참조, 배경은 프롬프트로)\n\n"
    "**동작** (구간 수만큼 자동 반복)\n"
    "1. Loader가 구간별 프롬프트 / 시작 프레임 / 프레임 수를 리스트로 내보냅니다.\n"
    "2. `Load Video (Path)`가 구간에 해당하는 원본 프레임(16fps)만 읽어 드라이빙 영상(동작)으로 씁니다.\n"
    "3. QwenVL이 참조 이미지 속 인물의 외형을 영어로 묘사 → `Character Description:` 으로 프롬프트 앞에 붙습니다.\n"
    "4. Wan Animate 2가 원본의 동작 + 참조 이미지(인물·배경·소품) + 구간 프롬프트로 클립을 생성합니다.\n"
    "5. `Shorts Clip Saver`가 clip_NN.mp4로 저장, `Shorts Concat`이 final.mp4로 합칩니다 (prompts.json 옆 폴더).\n\n"
    "**모델** (ComfyUI 공식 템플릿 video_wan_animate2와 동일, Comfy-Org/Wan-Animate-2)\n"
    "- diffusion_models/wan_animate_2_int8_convrot.safetensors\n"
    "- loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors\n"
    "- text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors\n"
    "- clip_vision/clip_vision_h.safetensors\n"
    "- vae/Wan2_1_VAE_bf16.safetensors\n\n"
    "**팁**\n"
    "- 해상도: Loader `max_side` 832 (원본 비율 유지, 16의 배수). L40S에서 832x480 5초 클립 약 1~2분.\n"
    "- 일부 구간만: Loader `clips`에 `1,3-4` 처럼 입력.\n"
    "- 배경이 참조 이미지와 달라지면 `WanAnimate2ToVideo`의 `reference_image_strength`를 1.2~1.5로 올려 보세요.\n"
    "- 프롬프트 형식은 Loader `prompt_template`에서 바꿀 수 있습니다."
)


def frag_replace():
    with io.open(TEMPLATES + "video_wan_animate2.json", encoding="utf-8") as f:
        tpl = json.load(f)
    sg = next(g for g in tpl["definitions"]["subgraphs"] if g["name"].startswith("Motion Transfer"))

    keep_nodes = [copy.deepcopy(n) for n in sg["nodes"] if n["id"] not in DROP and n["type"] not in ("MarkdownNote", "Note")]
    keep_ids = {n["id"] for n in keep_nodes}
    links = {}
    for l in sg["links"]:
        if l["origin_id"] in keep_ids and l["target_id"] in keep_ids:
            links[l["id"]] = [l["id"], l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l["type"]]

    xs = [n["pos"][0] for n in keep_nodes]
    ys = [n["pos"][1] for n in keep_nodes]
    x0, y0 = min(xs) - 1150, min(ys)

    my = [
        note(57, [x0 - 620, y0], [560, 700], NOTE_REPLACE, "사용법 (3단계 인물 교체)", ("#232", "#353")),
        node(52, "LoadImage", [x0, y0], [320, 400], [],
             [outp("IMAGE", "IMAGE", [2000]), outp("MASK", "MASK", None)], ["profile.png", "image"],
             title="Load Image - 프로필 (reference.png 없을 때 대신 사용)"),
        node(59, "ShortsReferenceLoader", [x0, y0 + 460], [360, 150],
             [inp("reference", "IMAGE", None), inp("fallback", "IMAGE", 2000)],
             [outp("image", "IMAGE", [2001, 2002]), outp("prompts_json", "STRING", [2023]), outp("source", "STRING", None)],
             [SAMPLE_PROMPTS, "reference.png"],
             title="Shorts Reference Loader - prompts.json 경로 입력 (3단계 시작점)"),
        node(58, "AILab_QwenVL_Advanced", [x0, y0 + 640], [460, 620],
             [inp("image", "IMAGE", 2002), inp("video", "IMAGE", None)],
             [outp("RESPONSE", "STRING", [2003])], qwen_widgets(CHAR_PROMPT, 1, 256),
             title="QwenVL - 참조 인물 외형 묘사 (Character Description)"),
        node(51, "ShortsPromptsLoader", [x0 + 520, y0], [480, 420],
             [inp("character_description", "STRING", 2003), inp("prompts_json", "STRING", 2023, True)],
             [outp("positive", "STRING", [2010]), outp("pose_prompt", "STRING", [2012]), outp("segment_index", "INT", [2020]),
              outp("skip_frames", "INT", [2005]), outp("frame_count", "INT", [2006, 2013]),
              outp("negative", "STRING", [2011]), outp("common", "STRING", None), outp("video_path", "STRING", [2004]),
              outp("fps", "FLOAT", [2016]), outp("width", "INT", [2014]), outp("height", "INT", [2015]),
              outp("count", "INT", None), outp("prompts_dir", "STRING", [2021, 2031])],
             ["", 16, 4, 832, 16, "", PROMPT_TEMPLATE],
             title="Shorts Prompts Loader - prompts.json → 구간 리스트"),
        node(53, "VHS_LoadVideoPath", [x0 + 520, y0 + 480], [420, 340],
             [inp("meta_batch", "VHS_BatchManager", None), inp("vae", "VAE", None),
              inp("video", "STRING", 2004, True), inp("frame_load_cap", "INT", 2006, True), inp("skip_first_frames", "INT", 2005, True)],
             [outp("IMAGE", "IMAGE", [2007]), outp("frame_count", "INT", None), outp("audio", "AUDIO", None), outp("video_info", "VHS_VIDEOINFO", None)],
             {"video": "", "force_rate": 16, "custom_width": 0, "custom_height": 0, "frame_load_cap": 0,
              "skip_first_frames": 0, "select_every_nth": 1, "format": "None"},
             title="Load Video (Path) - 원본 구간 프레임 (드라이빙)"),
    ]
    x1, y1 = max(xs) + 520, min(ys)
    my += [
        node(54, "ShortsClipSaver", [x1, y1], [340, 140],
             [inp("video", "VIDEO", 2030), inp("segment_index", "INT", 2020), inp("out_dir", "STRING", 2021)],
             [outp("clip_path", "STRING", [2032])], ["clip"]),
        node(55, "ShortsConcat", [x1, y1 + 200], [340, 180],
             [inp("clip_path", "STRING", 2032), inp("out_dir", "STRING", 2031)],
             [outp("final_path", "STRING", None), outp("video", "VIDEO", [2033])], [16.0, "final.mp4", True]),
        node(56, "SaveVideo", [x1, y1 + 440], [340, 320], [inp("video", "VIDEO", 2033)], [],
             ["video/shorts_final", "auto", "auto", "auto"], title="Save Video - final.mp4 미리보기/복사본"),
    ]

    new_links = [
        [2000, 52, 0, 59, 2, "IMAGE"],            # profile -> reference loader fallback
        [2001, 59, 0, 590, 0, "IMAGE"],           # reference -> reference resize
        [2002, 59, 0, 58, 0, "IMAGE"],            # reference -> QwenVL
        [2003, 58, 0, 51, 0, "STRING"],           # character description -> loader
        [2004, 51, 7, 53, 2, "STRING"],           # video_path -> VHS video
        [2005, 51, 3, 53, 4, "INT"],              # skip_frames -> VHS skip_first_frames
        [2006, 51, 4, 53, 3, "INT"],              # frame_count -> VHS frame_load_cap
        [2007, 53, 0, 600, 0, "IMAGE"],           # driving frames -> resize (pose video)
        [2010, 51, 0, 582, 1, "STRING"],          # positive -> CLIPTextEncode positive
        [2011, 51, 5, 581, 1, "STRING"],          # negative -> CLIPTextEncode negative
        [2012, 51, 1, 585, 1, "STRING"],          # pose prompt -> CLIPTextEncode pose
        [2013, 51, 4, 587, 11, "INT"],            # frame_count -> WanAnimate2ToVideo length
        [2014, 51, 9, 600, 1, "INT"],             # width -> resize width
        [2015, 51, 10, 600, 2, "INT"],            # height -> resize height
        [2016, 51, 8, 245, 2, "FLOAT"],           # fps -> CreateVideo fps
        [2017, 601, 0, 245, 0, "IMAGE"],          # VAEDecode -> CreateVideo images
        [2020, 51, 2, 54, 1, "INT"],              # segment_index -> saver
        [2021, 51, 12, 54, 2, "STRING"],          # prompts_dir -> saver out_dir
        [2023, 59, 1, 51, 1, "STRING"],           # reference loader prompts_json -> prompts loader
        [2030, 245, 0, 54, 0, "VIDEO"],           # CreateVideo -> saver
        [2031, 51, 12, 55, 1, "STRING"],          # prompts_dir -> concat out_dir
        [2032, 54, 0, 55, 0, "STRING"],           # clip_path -> concat
        [2033, 55, 1, 56, 0, "VIDEO"],            # final video -> SaveVideo
    ]
    for l in new_links:
        links[l[0]] = l

    rewire = {  # (node, input name) -> new link id (None = disconnect, keep widget)
        (590, "input"): 2001, (600, "input"): 2007, (600, "resize_type.width"): 2014, (600, "resize_type.height"): 2015,
        (582, "text"): 2010, (581, "text"): 2011, (585, "text"): 2012,
        (587, "length"): 2013, (587, "continue_motion"): None, (587, "video_frame_offset"): None,
        (587, "pose_strength"): None, (587, "pose_start_percent"): None, (587, "pose_end_percent"): None,
        (587, "reference_image_strength"): None,
        (588, "switch"): None, (594, "device"): None, (594, "dtype"): None,
        (578, "unet_name"): None, (579, "lora_name"): None, (580, "clip_name"): None, (583, "clip_name"): None,
        (584, "vae_name"): None,
        (245, "images"): 2017, (245, "audio"): None, (245, "fps"): 2016,
    }
    valid = set(links.keys())
    for n in keep_nodes:
        for i in n.get("inputs", []):
            key = (n["id"], i["name"])
            if key in rewire:
                i["link"] = rewire[key]
            elif i.get("link") is not None and i["link"] not in valid:
                i["link"] = None
        for o in n.get("outputs", []):
            if o.get("links"):
                o["links"] = [x for x in o["links"] if x in valid and links[x][1] == n["id"]]
                if not o["links"]:
                    o["links"] = None
        if n["id"] == 601:
            n["outputs"][0]["links"] = [2017]
        if n["id"] == 245:
            n["outputs"][0]["links"] = [2030]
        if n["id"] == 587:
            for o in n["outputs"]:
                if o["name"] == "video_frame_offset":
                    o["links"] = None
        if n["id"] == 588:
            n["widgets_values"] = [False]
        if n["id"] == 594:
            n["widgets_values"] = ["gpu", "int8"]
        if n["id"] == 597:
            n["widgets_values"] = [True, 42, "fixed", 1]  # fixed seed for reproducible clips

    my_ids = {n["id"] for n in my}
    links = {k: v for k, v in links.items()
             if (v[1] in keep_ids or v[1] in my_ids) and (v[3] in keep_ids or v[3] in my_ids)}
    nodes = my + keep_nodes
    # normalise: top-left of the fragment -> (0, 0)
    mx = min(n["pos"][0] for n in nodes)
    myy = min(n["pos"][1] for n in nodes)
    shift(nodes, -mx, -myy)
    return nodes, links


# --------------------------------------------------------------------------- #
# API-format export (what the frontend's "Export (API)" produces)
# --------------------------------------------------------------------------- #
LINKLESS_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# nodes whose widgets cannot be derived from INPUT_TYPES (dynamic V3 inputs / custom packages not imported)
WIDGET_NAMES = {
    "AILab_QwenVL_Advanced": ["model_name", "quantization", "attention_mode", "use_torch_compile", "device", "preset_prompt",
                              "custom_prompt", "max_tokens", "temperature", "top_p", "num_beams", "repetition_penalty",
                              "frame_count", "video_frame_size", "keep_model_loaded", "seed", None],
    "ResizeImageMaskNode": ["resize_type", "resize_type.width", "resize_type.height", "resize_type.crop", "scale_method"],
    "SaveVideo": ["filename_prefix", "format", "format.codec", "codec"],
    "ComfySwitchNode": ["switch"],
    "PreviewAny": [None, None, None],
}
SKIP_TYPES = {"MarkdownNote", "Note", "PrimitiveNode", "Reroute"}

_NODE_DEFS = None


def node_defs():
    global _NODE_DEFS
    if _NODE_DEFS is None:
        sys.path.insert(0, COMFY)
        cwd = os.getcwd()
        os.chdir(COMFY)
        import nodes as comfy_nodes  # noqa: E402
        asyncio.run(comfy_nodes.init_extra_nodes(init_custom_nodes=False))
        os.chdir(cwd)
        _NODE_DEFS = dict(comfy_nodes.NODE_CLASS_MAPPINGS)
        _NODE_DEFS.update(shorts_nodes.NODE_CLASS_MAPPINGS)
    return _NODE_DEFS


def widget_names_for(ntype):
    """Ordered widget names as the frontend lays them out; None = a value to skip (control_after_generate / upload)."""
    if ntype in WIDGET_NAMES:
        return WIDGET_NAMES[ntype]
    cls = node_defs().get(ntype)
    if cls is None:
        raise KeyError(f"unknown node type {ntype}")
    it = cls.INPUT_TYPES()
    names = []
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            is_combo = isinstance(typ, (list, tuple)) or typ == "COMBO"
            if opts.get("forceInput"):
                continue
            if not (is_combo or typ in LINKLESS_WIDGET_TYPES):
                continue
            names.append(name)
            if name in ("seed", "noise_seed") or opts.get("control_after_generate"):
                names.append(None)
            if opts.get("image_upload") or opts.get("video_upload") or opts.get("audio_upload"):
                names.append(None)
    return names


def export_api(wf):
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}
    bypassed = {nid for nid, n in nodes.items() if n.get("mode") == 4}
    muted = {nid for nid, n in nodes.items() if n.get("mode") == 2}

    def resolve(origin, slot, typ):
        """Follow through bypassed nodes: output of a bypassed node -> its same-typed input's source."""
        if origin in muted:
            return None
        if origin not in bypassed:
            return [str(origin), slot]
        n = nodes[origin]
        for i in n.get("inputs", []):
            if i.get("link") is not None and i.get("type") == typ:
                l = links[i["link"]]
                return resolve(l[1], l[2], typ)
        return None

    api = {}
    for nid, n in nodes.items():
        if n["type"] in SKIP_TYPES or nid in bypassed or nid in muted:
            continue
        inputs = {}
        wv = n.get("widgets_values")
        if isinstance(wv, dict):
            inputs.update(wv)
        elif wv:
            names = widget_names_for(n["type"])
            vals = list(wv)
            if len(vals) != len(names):
                raise ValueError(f"node {nid} {n['type']}: {len(vals)} widget values but {len(names)} names {names}")
            for name, v in zip(names, vals):
                if name is not None:
                    inputs[name] = v
        for i in n.get("inputs", []):
            if i.get("link") is None:
                continue
            l = links[i["link"]]
            src = resolve(l[1], l[2], l[5])
            if src is not None:
                inputs[i["name"]] = src
        entry = {"inputs": inputs, "class_type": n["type"]}
        if n.get("title"):
            entry["_meta"] = {"title": n["title"]}
        api[str(nid)] = entry
    return api


def write_all(wf, filename):
    seen = set()
    for d in OUT_DIRS:
        rp = os.path.realpath(d)
        if rp in seen or not os.path.isdir(d):
            continue
        seen.add(rp)
        p = os.path.join(d, filename)
        with io.open(p, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=2)
        print("wrote", p)
        api = export_api(wf)
        pa = os.path.join(d, filename.replace(".json", ".api.json"))
        with io.open(pa, "w", encoding="utf-8") as f:
            json.dump(api, f, ensure_ascii=False, indent=1)
        print("wrote", pa, f"({len(api)} nodes)")


# --------------------------------------------------------------------------- #
NOTE_ALL = (
    "# Shorts Remake 올인원: 영상 파일 + 프로필/배경/소품 사진 → 인물 교체 영상\n\n"
    "유튜브 영상은 먼저 `0_YouTube_Download_Trim` 워크플로우로 받고, 표시된 파일 경로를 여기 `video_path`에 넣습니다.\n"
    "위에서 아래로 세 구역이 한 번의 Queue로 이어서 실행됩니다.\n\n"
    "1. **분석** (왼쪽 위): `Shorts Video Segments`의 `video_path`에 영상 경로 → 5초/장면 구간마다 QwenVL 프롬프트 → prompts.json\n"
    "2. **참조 이미지 합성** (왼쪽 아래): `Load Image` 프로필(필수) · 배경(선택) · 소품(선택) → Qwen-Image-Edit-2511 → reference.png. "
    "prompts.json 경로는 1구역에서 자동으로 연결됩니다 (`Shorts Reference Setup`의 prompts_json 입력).\n"
    "3. **영상 생성** (오른쪽): Wan Animate 2가 원본 동작 그대로 참조 이미지를 움직여 구간별 clip_NN.mp4 → final.mp4\n\n"
    "**입력 3곳만 바꾸면 됩니다**: 영상 파일 경로, 프로필 사진, (선택) 배경/소품 사진. 배경 노드를 bypass(Ctrl+B)하면 원본 영상의 배경이 그대로 유지되고 사람만 바뀝니다.\n\n"
    "**출력**: 영상 옆 `<영상이름>_prompts/` 폴더 안에 prompts.json, reference.png, clip_NN.mp4, final.mp4. "
    "`Save Video`로 ComfyUI/output/video/ 에도 복사.\n\n"
    "**VRAM**: Qwen3-VL-8B(16GB) → Qwen-Image-Edit fp8(~22GB) → Wan Animate 2 int8(~20GB) 순으로 ComfyUI가 모델을 교체해 로드합니다. L40S 48GB 기준.\n\n"
    "**단계별로 확인하며 하려면** 1 → 2 → 3 워크플로우를 따로 실행하세요 (prompts.json / reference.png를 중간에 편집 가능)."
)


def build_all():
    # stage 0
    d_nodes, d_links = build_download()
    write_all(assemble("shorts-0-youtube-download", [(d_nodes, d_links)], ds={"scale": 0.8, "offset": [1050, 60]}),
              "0_YouTube_Download_Trim.json")

    # stage 1
    a_nodes, a_links = frag_analyze()
    write_all(assemble("shorts-1-analyze", [(a_nodes, a_links)], ds={"scale": 0.6, "offset": [1650, 60]}),
              "1_Analyze_Prompts.json")

    # stage 2
    c_nodes, c_links = frag_compose()
    write_all(assemble("shorts-1b-compose-reference", [(c_nodes, c_links)], ds={"scale": 0.45, "offset": [1650, 100]}),
              "2_Compose_Reference.json")

    # stage 2
    r_nodes, r_links = frag_replace()
    write_all(assemble("shorts-2-replace-person", [(r_nodes, r_links)], ds={"scale": 0.45, "offset": [100, 100]}),
              "3_Replace_Person_WanAnimate2.json")

    # all-in-one: stage 1 at top-left, stage 1b below it, stage 2 to the right
    a_nodes, a_links = frag_analyze()
    c_nodes, c_links = frag_compose()
    r_nodes, r_links = frag_replace()
    a_nodes = [n for n in a_nodes if n["id"] != 10]  # per-stage notes replaced by one overview note
    c_nodes = [n for n in c_nodes if n["id"] != 39]
    r_nodes = [n for n in r_nodes if n["id"] != 57]
    # in the all-in-one graph the compose stage reads prompts.json from the collector, not from a fixed path
    for n in c_nodes:
        if n["id"] == 20:
            n["inputs"].append(inp("prompts_json", "STRING", 300, True))
            n["widgets_values"][0] = ""
    for n in r_nodes:
        if n["id"] == 59:
            n["inputs"].append(inp("prompts_json", "STRING", 301, True))
            n["widgets_values"][0] = ""
        if n["id"] == 52:
            n["title"] = "Load Image - 프로필 (합성 실패 시 대체용, 평소엔 미사용)"
    shift(c_nodes, 0, 1500)
    shift(r_nodes, 900, 0)
    overview = note(70, [-2200, 80], [600, 760], NOTE_ALL, "올인원 사용법", ("#322", "#533"))
    cross = [
        [300, 4, 1, 20, 3, "STRING"],   # collector prompts_json -> reference setup
        [301, 4, 1, 59, 2, "STRING"],   # collector prompts_json -> reference loader (stage-2 entry point)
        [302, 37, 1, 59, 0, "IMAGE"],   # saved reference image -> reference loader
    ]
    wf = assemble("shorts-all-remake", [([overview], {}), (a_nodes, a_links), (c_nodes, c_links), (r_nodes, r_links)],
                  extra_links=cross, ds={"scale": 0.3, "offset": [2300, 60]})
    write_all(wf, "4_ALL_in_One.json")


if __name__ == "__main__":
    build_all()
