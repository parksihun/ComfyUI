"""Build the two ShortsRemake workflows (UI/LiteGraph format).

1. Shorts_1_Analyze_Prompts.json      video -> 5s segments -> QwenVL -> prompts.json
2. Shorts_2_Replace_Person_WanAnimate2.json
   profile image + prompts.json + source video -> Wan Animate 2 per segment -> clip_NN.mp4 -> final.mp4
   (flattened from the official video_wan_animate2.json template, loop nodes removed,
    per-segment execution comes from ShortsPromptsLoader list outputs)
"""
import copy
import io
import json
import os

# repo root = ComfyUI-Easy-Install folder (this file lives in <root>/tools/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "python_embeded", "Lib", "site-packages", "comfyui_workflow_templates_json", "templates") + os.sep
OUT_DIRS = [
    os.path.join(ROOT, "workflow"),
    os.path.join(ROOT, "ComfyUI", "user", "default", "workflows"),
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
    "Describe this person's appearance for an AI video generation prompt. English, one paragraph of 40-70 words: "
    "gender, apparent age range, hair (color, length, style), face features, skin tone, clothing and accessories, body type. "
    "Looks only: no background, no motion, no emotions, no name. Output the paragraph only."
)

PROMPT_TEMPLATE = "Character Description: {character}\nBackground description: {common} {segment}"

QWEN_MODEL = "Qwen3-VL-8B-Instruct"


def qwen_widgets(prompt, frame_count, max_tokens=512, model=QWEN_MODEL):
    # model_name, quantization, attention_mode, use_torch_compile, device, preset_prompt, custom_prompt,
    # max_tokens, temperature, top_p, num_beams, repetition_penalty, frame_count, video_frame_size,
    # keep_model_loaded, seed, control_after_generate
    return [model, "None (FP16)", "auto", False, "auto", "\U0001F4F9 Video Summary", prompt,
            max_tokens, 0.3, 0.9, 1, 1.2, frame_count, "auto", True, 1, "fixed"]


def node(nid, ntype, pos, size, inputs, outputs, widgets, title=None, extra=None):
    n = {"id": nid, "type": ntype, "pos": pos, "size": size, "flags": {}, "order": 0, "mode": 0,
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


def write_all(wf, filename):
    for d in OUT_DIRS:
        p = os.path.join(d, filename)
        with io.open(p, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=2)
        print("wrote", p)


# --------------------------------------------------------------------------- #
# Workflow 1: analyze
# --------------------------------------------------------------------------- #
def build_analyze():
    note = (
        "## 1단계: 영상 분석 → 프롬프트 추출\n\n"
        "**입력**: `Shorts Video Segments` 노드의 `video_path`에 받아둔 영상 전체 경로.\n\n"
        "**동작**\n"
        "1. 영상을 구간으로 나누고 구간마다 `frames_per_segment`장을 뽑습니다.\n"
        "   - `split_mode` = fixed : `segment_seconds`(기본 5초) 고정 길이\n"
        "   - `split_mode` = scene : 장면 전환(컷) 지점에서 자름. `segment_seconds`보다 긴 장면은 균등 분할, "
        "`min_seconds`보다 짧은 조각은 이웃과 합침. `scene_threshold`를 낮추면 컷을 더 민감하게 잡음\n"
        "2. 위쪽 QwenVL이 **구간마다 한 번씩** 실행되어 구간 프롬프트(JSON)를 씁니다.\n"
        "3. 아래쪽 QwenVL이 구간별 대표 프레임을 한꺼번에 보고 **공통 프롬프트 · 네거티브**를 씁니다.\n"
        "4. `Shorts Prompts Collector`가 결과를 모아 저장합니다.\n\n"
        "**출력** (`out_dir` 비우면 영상 옆 `<영상이름>_prompts/`)\n"
        "- `prompts.json` : common_prompt, negative_prompt, segments[].positive_prompt / motion / camera / scene_ko\n"
        "- `comfyui_prompts.txt` : 사람이 보는 정리본\n\n"
        "**다음 단계**: `Shorts_2_Replace_Person_WanAnimate2` 워크플로우의 `prompts_json`에 이 파일 경로를 넣습니다.\n\n"
        "GPU 서버: Qwen3-VL-8B-Instruct(FP16, ~16GB). 더 정밀하게: Qwen3-VL-32B-Instruct-FP8.\n"
        "이 PC(CPU)에서 확인할 때는 QwenVL 노드를 `QwenVL Advanced (GGUF)`, device=cpu로 바꿔 쓰세요."
    )
    nodes = [
        node(10, "MarkdownNote", [-1560, 80], [520, 520], [], [], [note], title="사용법 (1단계 분석)",
             extra={"color": "#432", "bgcolor": "#653"}),
        node(1, "ShortsVideoSegments", [-1000, 80], [420, 220], [],
             [outp("segment_frames", "IMAGE", [1]), outp("segment_index", "INT", None), outp("segment_label", "STRING", None),
              outp("overview_frames", "IMAGE", [2]), outp("count", "INT", None), outp("video_path", "STRING", None),
              outp("segments_json", "STRING", [5])],
             [SAMPLE_VIDEO, "fixed", 5.0, 1.5, 0.5, 8, 384]),
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
    links = [
        [1, 1, 0, 2, 1, "IMAGE"],
        [2, 1, 3, 3, 1, "IMAGE"],
        [3, 2, 0, 4, 0, "STRING"],
        [4, 3, 0, 4, 1, "STRING"],
        [5, 1, 6, 4, 2, "STRING"],
        [6, 4, 0, 5, 0, "STRING"],
    ]
    for i, n in enumerate(nodes):
        n["order"] = i
    wf = {"id": "shorts-1-analyze", "revision": 0, "last_node_id": 10, "last_link_id": 6, "nodes": nodes, "links": links,
          "groups": [], "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [1650, 60]}}, "version": 0.4}
    write_all(wf, "Shorts_1_Analyze_Prompts.json")


# --------------------------------------------------------------------------- #
# Workflow 2: replace person (Wan Animate 2), flattened from the official template
# --------------------------------------------------------------------------- #
DROP = {288, 537, 595, 604, 605, 635, 636, 639, 642, 645, 646, 647, 648, 649, 651, 652, 653, 654, 661, 667, 669, 670, 671}


def build_replace():
    with io.open(TEMPLATES + "video_wan_animate2.json", encoding="utf-8") as f:
        tpl = json.load(f)
    sg = next(g for g in tpl["definitions"]["subgraphs"] if g["name"].startswith("Motion Transfer"))

    keep_nodes = [copy.deepcopy(n) for n in sg["nodes"] if n["id"] not in DROP and n["type"] not in ("MarkdownNote", "Note")]
    keep_ids = {n["id"] for n in keep_nodes}
    # internal links only (both ends kept)
    links = {}
    for l in sg["links"]:
        if l["origin_id"] in keep_ids and l["target_id"] in keep_ids:
            links[l["id"]] = [l["id"], l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l["type"]]

    # ---- my nodes ----
    xs = [n["pos"][0] for n in keep_nodes]
    ys = [n["pos"][1] for n in keep_nodes]
    x0, y0 = min(xs) - 1150, min(ys)

    note = (
        "## 2단계: 프로필 이미지로 인물 교체 (Wan Animate 2)\n\n"
        "**입력**\n"
        "- `Load Image`: 바꿔 넣을 사람의 프로필 사진 (상반신/전신, 배경 무관)\n"
        "- `Shorts Prompts Loader`의 `prompts_json`: 1단계에서 만든 prompts.json 경로\n"
        "- 원본 영상 경로는 prompts.json 안의 video_file을 자동으로 씁니다\n\n"
        "**동작** (구간 수만큼 자동 반복)\n"
        "1. Loader가 구간별 프롬프트 / 시작 프레임 / 프레임 수를 리스트로 내보냅니다.\n"
        "2. `Load Video (Path)`가 구간에 해당하는 원본 프레임(16fps)만 읽어 드라이빙 영상으로 씁니다.\n"
        "3. QwenVL이 프로필 사진의 외형을 영어로 묘사 → `Character Description:` 으로 프롬프트 앞에 붙습니다.\n"
        "4. Wan Animate 2가 원본의 동작 + 프로필 인물 + 구간 프롬프트(배경)로 클립을 생성합니다.\n"
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
        "- 배경을 원본과 비슷하게: 1단계 common_prompt/segment prompt가 배경을 담당합니다. 직접 수정하려면 prompts.json 편집.\n"
        "- 프롬프트 형식은 Loader `prompt_template`에서 바꿀 수 있습니다."
    )
    my = [
        node(7, "MarkdownNote", [x0 - 620, y0], [560, 640], [], [], [note], title="사용법 (2단계 인물 교체)",
             extra={"color": "#223", "bgcolor": "#335"}),
        node(2, "LoadImage", [x0, y0], [320, 400], [],
             [outp("IMAGE", "IMAGE", [2001, 2002]), outp("MASK", "MASK", None)], ["profile.png", "image"],
             title="Load Image - 프로필 사진 (바꿔 넣을 인물)"),
        node(8, "AILab_QwenVL_Advanced", [x0, y0 + 440], [460, 620],
             [inp("image", "IMAGE", 2002), inp("video", "IMAGE", None)],
             [outp("RESPONSE", "STRING", [2003])], qwen_widgets(CHAR_PROMPT, 1, 256),
             title="QwenVL - 프로필 외형 묘사 (Character Description)"),
        node(1, "ShortsPromptsLoader", [x0 + 520, y0], [480, 420],
             [inp("character_description", "STRING", 2003)],
             [outp("positive", "STRING", [2010]), outp("pose_prompt", "STRING", [2012]), outp("segment_index", "INT", [2020]),
              outp("skip_frames", "INT", [2005]), outp("frame_count", "INT", [2006, 2013]),
              outp("negative", "STRING", [2011]), outp("common", "STRING", None), outp("video_path", "STRING", [2004]),
              outp("fps", "FLOAT", [2016]), outp("width", "INT", [2014]), outp("height", "INT", [2015]),
              outp("count", "INT", None), outp("prompts_dir", "STRING", [2021, 2031])],
             [SAMPLE_PROMPTS, 16, 4, 832, 16, "", PROMPT_TEMPLATE]),
        node(3, "VHS_LoadVideoPath", [x0 + 520, y0 + 480], [420, 340],
             [inp("meta_batch", "VHS_BatchManager", None), inp("vae", "VAE", None),
              inp("video", "STRING", 2004, True), inp("frame_load_cap", "INT", 2006, True), inp("skip_first_frames", "INT", 2005, True)],
             [outp("IMAGE", "IMAGE", [2007]), outp("frame_count", "INT", None), outp("audio", "AUDIO", None), outp("video_info", "VHS_VIDEOINFO", None)],
             {"video": "", "force_rate": 16, "custom_width": 0, "custom_height": 0, "frame_load_cap": 0,
              "skip_first_frames": 0, "select_every_nth": 1, "format": "None"},
             title="Load Video (Path) - 원본 구간 프레임 (드라이빙)"),
    ]
    # right side: saver / concat / save
    x1, y1 = max(xs) + 520, min(ys)
    my += [
        node(4, "ShortsClipSaver", [x1, y1], [340, 140],
             [inp("video", "VIDEO", 2030), inp("segment_index", "INT", 2020), inp("out_dir", "STRING", 2021)],
             [outp("clip_path", "STRING", [2032])], ["clip"]),
        node(5, "ShortsConcat", [x1, y1 + 200], [340, 180],
             [inp("clip_path", "STRING", 2032), inp("out_dir", "STRING", 2031)],
             [outp("final_path", "STRING", None), outp("video", "VIDEO", [2033])], [16.0, "final.mp4", True]),
        node(6, "SaveVideo", [x1, y1 + 440], [340, 320], [inp("video", "VIDEO", 2033)], [],
             ["video/shorts_final", "auto", "auto", "auto"], title="Save Video - final.mp4 미리보기/복사본"),
    ]

    # ---- new links (id, origin, oslot, target, tslot, type) ----
    new_links = [
        [2001, 2, 0, 590, 0, "IMAGE"],            # profile -> reference resize
        [2002, 2, 0, 8, 0, "IMAGE"],              # profile -> QwenVL
        [2003, 8, 0, 1, 0, "STRING"],             # character description -> loader
        [2004, 1, 7, 3, 2, "STRING"],             # video_path -> VHS video
        [2005, 1, 3, 3, 4, "INT"],                # skip_frames -> VHS skip_first_frames
        [2006, 1, 4, 3, 3, "INT"],                # frame_count -> VHS frame_load_cap
        [2007, 3, 0, 600, 0, "IMAGE"],            # driving frames -> resize (pose video)
        [2010, 1, 0, 582, 1, "STRING"],           # positive -> CLIPTextEncode positive
        [2011, 1, 5, 581, 1, "STRING"],           # negative -> CLIPTextEncode negative
        [2012, 1, 1, 585, 1, "STRING"],           # pose prompt -> CLIPTextEncode pose
        [2013, 1, 4, 587, 11, "INT"],             # frame_count -> WanAnimate2ToVideo length
        [2014, 1, 9, 600, 1, "INT"],              # width -> resize width
        [2015, 1, 10, 600, 2, "INT"],             # height -> resize height
        [2016, 1, 8, 245, 2, "FLOAT"],            # fps -> CreateVideo fps
        [2017, 601, 0, 245, 0, "IMAGE"],          # VAEDecode -> CreateVideo images
        [2020, 1, 2, 4, 1, "INT"],                # segment_index -> saver
        [2021, 1, 12, 4, 2, "STRING"],            # prompts_dir -> saver out_dir
        [2030, 245, 0, 4, 0, "VIDEO"],            # CreateVideo -> saver
        [2031, 1, 12, 5, 1, "STRING"],            # prompts_dir -> concat out_dir
        [2032, 4, 0, 5, 0, "STRING"],             # clip_path -> concat
        [2033, 5, 1, 6, 0, "VIDEO"],              # final video -> SaveVideo
    ]
    for l in new_links:
        links[l[0]] = l

    # rewire template nodes: replace inputs' link ids
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
    # 601 VAEDecode -> only CreateVideo now
    for n in keep_nodes:
        if n["id"] == 601:
            n["outputs"][0]["links"] = [2017]
        if n["id"] == 245:
            n["outputs"][0]["links"] = [2030]
        if n["id"] == 587:
            # remove dangling video_frame_offset output links
            for o in n["outputs"]:
                if o["name"] == "video_frame_offset":
                    o["links"] = None
        if n["id"] == 588:
            n["widgets_values"] = [False]
        if n["id"] == 594:
            n["widgets_values"] = ["gpu", "int8"]
        if n["id"] == 597:
            # SamplerCustom: add_noise, seed, control, cfg -> fixed seed for reproducible clips
            n["widgets_values"] = [True, 42, "fixed", 1]

    # drop links that reference dropped nodes
    links = {k: v for k, v in links.items() if (v[1] in keep_ids or v[1] <= 10) and (v[3] in keep_ids or v[3] <= 10)}

    nodes = my + keep_nodes
    for i, n in enumerate(nodes):
        n["order"] = i
    wf = {"id": "shorts-2-replace-person", "revision": 0, "last_node_id": max(n["id"] for n in nodes),
          "last_link_id": max(links.keys()), "nodes": nodes, "links": list(links.values()),
          "groups": [], "config": {}, "extra": {"ds": {"scale": 0.45, "offset": [-(x0 - 700), -(y0 - 60)]}}, "version": 0.4}
    write_all(wf, "Shorts_2_Replace_Person_WanAnimate2.json")


def patch_api_prompts():
    """Keep the API-format copies (made by the frontend's graphToPrompt) in sync with the prompt texts above."""
    for fn, prompts in (("Shorts_1_Analyze_Prompts.api.json", {"2": SEG_PROMPT, "3": COMMON_PROMPT}),
                        ("Shorts_2_Replace_Person_WanAnimate2.api.json", {"8": CHAR_PROMPT})):
        p = os.path.join(OUT_DIRS[0], fn)
        if not os.path.isfile(p):
            continue
        with io.open(p, encoding="utf-8") as f:
            api = json.load(f)
        for nid, text in prompts.items():
            if nid in api:
                api[nid]["inputs"]["custom_prompt"] = text
        with io.open(p, "w", encoding="utf-8") as f:
            json.dump(api, f, ensure_ascii=False, indent=1)
        print("patched", p)


if __name__ == "__main__":
    build_analyze()
    build_replace()
    patch_api_prompts()
