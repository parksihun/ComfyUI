# Shorts Remake 파이프라인 (영상 분석 → 프롬프트 → 인물 교체 생성)

받아둔 영상을 구간별로 분석해 프롬프트를 뽑고, 프로필 이미지의 인물로 바꾼 영상을 구간별로 생성해 합칩니다.

```
원본 영상 ──▶ [1단계] Shorts_1_Analyze_Prompts ──▶ prompts.json (공통 / 네거티브 / 구간별)
                                                        │
프로필 사진 ──▶ [2단계] Shorts_2_Replace_Person_WanAnimate2 ──▶ clip_01.mp4 … clip_NN.mp4 ──▶ final.mp4
```

## 필요한 것

| 항목 | 내용 |
|---|---|
| 커스텀 노드 | `custom_nodes/ComfyUI-ShortsRemake` (이 저장소에 포함), ComfyUI-QwenVL, VideoHelperSuite |
| 1단계 모델 | Qwen3-VL-8B-Instruct (QwenVL 노드가 첫 실행 시 `models/LLM/`에 자동 다운로드, 약 16GB VRAM) |
| 2단계 모델 | ComfyUI 공식 `video_wan_animate2` 템플릿과 동일 (Comfy-Org/Wan-Animate-2): `diffusion_models/wan_animate_2_int8_convrot.safetensors`, `loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `clip_vision/clip_vision_h.safetensors`, `vae/Wan2_1_VAE_bf16.safetensors` |
| GPU | L40S 48GB 기준 설계. 이 테스트 PC(GPU 없음)에서는 1단계만 GGUF+CPU로 실행 가능 |

## 1단계: Shorts_1_Analyze_Prompts.json

1. `Shorts Video Segments` 노드의 `video_path`에 영상 전체 경로 입력.
2. `split_mode`
   - `fixed`: `segment_seconds`(기본 5초)마다 자름. 마지막 조각이 `min_seconds`보다 짧으면 앞 구간에 합침.
   - `scene`: 장면 전환(컷)에서 자름. `segment_seconds`보다 긴 장면은 균등 분할, `min_seconds`보다 짧은 조각은 이웃과 합침. `scene_threshold`(기본 0.5)를 낮추면 컷을 더 민감하게 잡음.
3. Queue. QwenVL이 구간 수만큼 자동 반복 실행되고, 마지막에 구간별 대표 프레임을 모아 공통 프롬프트와 네거티브를 씁니다.
4. 결과: `<영상 폴더>/<영상이름>_prompts/prompts.json`, `comfyui_prompts.txt` (`out_dir`로 변경 가능)

prompts.json 구조:

```json
{
  "video_file": "...mp4", "duration": 13.2, "segment_length": 5.0, "fps": 16,
  "common_prompt": "…배경/조명/색감/카메라 스타일 (영어)…",
  "negative_prompt": "…",
  "person_in_video": "원본 인물 외형 (참고용)",
  "segments": [
    {"index": 1, "start": 0.0, "end": 5.0, "label": "00:00.000 - 00:05.000",
     "positive_prompt": "…구간 배경·동작·카메라 (영어)…", "motion": "…인물 동작 한 줄…",
     "camera": "…", "scene_ko": "…한국어 장면 설명…"}
  ]
}
```

프롬프트를 손보려면 이 JSON을 직접 편집하면 됩니다. 2단계는 이 파일만 읽습니다.

## 2단계: Shorts_2_Replace_Person_WanAnimate2.json

1. `Load Image`에 바꿔 넣을 사람의 프로필 사진 업로드 (상반신/전신, 배경 무관).
2. `Shorts Prompts Loader`의 `prompts_json`에 1단계 결과 경로 입력. 원본 영상 경로는 JSON 안의 값을 자동으로 씁니다.
3. Queue. 구간 수만큼 자동 반복:
   - `Load Video (Path)`가 구간의 원본 프레임(16fps)만 읽어 **드라이빙 영상**으로 사용
   - QwenVL이 프로필 사진 외형을 묘사 → `Character Description:` 으로 프롬프트 앞에 결합
   - Wan Animate 2: 원본의 동작 + 프로필 인물 + 구간 프롬프트(배경) → 클립 생성
   - `Shorts Clip Saver` → `clip_NN.mp4`, `Shorts Concat` → `final.mp4` (prompts.json 옆 폴더), `Save Video`로 ComfyUI output에도 복사
4. 옵션
   - `clips`: `1,3-4` 처럼 일부 구간만 생성
   - `max_side`: 출력 해상도 (원본 비율 유지, 16의 배수). 기본 832
   - `prompt_template`: 최종 프롬프트 형식. 기본 `Character Description: {character}\nBackground description: {common} {segment}`
   - 컨텍스트 윈도우 스위치(노드 588)는 꺼져 있음. 5초 클립(81프레임)은 필요 없음

## API 방식 실행 (서버 자동화)

`*.api.json` 파일은 `POST http://<server>:8188/prompt` 에 그대로 넣을 수 있는 형식입니다.
노드 `"1"`의 `video_path`(1단계) / `prompts_json`(2단계), 노드 `"2"`의 `image`(프로필 파일명)만 바꿔 보내면 됩니다.

```bash
curl -X POST http://127.0.0.1:8188/prompt -H "Content-Type: application/json" -d @Shorts_1_Analyze_Prompts.api.json
```

(실제로는 `{"prompt": <api.json 내용>}` 로 감싸야 합니다.)

## 이 PC(CPU)에서 확인한 것

- 커스텀 노드 5개 로딩, 두 워크플로우 모두 누락 노드 없이 UI에서 열림
- 1단계를 GGUF Qwen3VL-4B + CPU로 실제 실행해 prompts.json 생성 확인
- 2단계는 Wan 모델이 없어 구조 검증과 리스트 반복(로더 → 구간별 프레임 로드 → 저장 → 합치기)만 확인
- 장면 전환 감지는 ffmpeg의 `scene` 필터와 같은 지점을 잡는 것을 확인
