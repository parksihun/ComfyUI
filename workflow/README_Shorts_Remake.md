# Shorts Remake 파이프라인 (유튜브 → 분석 → 참조 이미지 → 인물 교체 영상)

유튜브 영상을 받아 5초(또는 장면) 단위로 분석해 프롬프트를 뽑고, 내가 넣은 **프로필·배경·소품** 사진으로
참조 이미지를 만든 뒤, 원본 영상과 같은 동작·구도의 새 영상을 구간별로 생성해 합칩니다.

```
유튜브 링크 ──▶ [0] 0_YouTube_Download_Trim ──▶ <id>.mp4 (경로 표시)
                                                        │
영상 파일 ────▶ [1] 1_Analyze_Prompts ──▶ prompts.json (공통 / 네거티브 / 구간별)
                                                        │
프로필 사진 ──┐                                          ▼
배경 사진 ────┼─▶ [2] 2_Compose_Reference ──▶ reference.png (사람+배경+소품 한 장, 원본 비율)
소품 사진 ────┘                                          │
                                                        ▼
                 [3] 3_Replace_Person_WanAnimate2 ──▶ clip_01.mp4 … clip_NN.mp4 ──▶ final.mp4

4_ALL_in_One = [1] + [2] + [3] 을 한 그래프로 (Queue 한 번). 다운로드([0])는 따로 실행
```

| 파일 | 용도 |
|---|---|
| `0_YouTube_Download_Trim.json` | 유튜브 링크(또는 파일) → mp4. `start`/`end`로 구간만 잘라내기. 결과 경로를 화면에 표시 |
| `4_ALL_in_One.json` | **올인원 (1→2→3)**. 영상 파일 경로 + 사진만 넣고 Queue 한 번. 중간 결과를 손볼 필요가 없을 때 |
| `1_Analyze_Prompts.json` | 분석만. prompts.json을 확인·수정하고 싶을 때 |
| `2_Compose_Reference.json` | 참조 이미지만. seed를 바꿔 가며 마음에 드는 합성을 고를 때 |
| `3_Replace_Person_WanAnimate2.json` | 영상 생성만. 일부 구간(`clips`)만 다시 만들 때 |
| `5_I2V_6seg_from_prompts.json` | **빠른 대안**: 1번 프롬프트 + 참조 이미지 → Wan 2.2 I2V 6구간(30초). 원본 동작을 옮기지 않고 프롬프트대로 움직임. 클립당 시간이 훨씬 짧고 VRAM 여유 있음 |
| `*.api.json` | 같은 그래프의 API 형식 (서버 자동화용, 아래 참고) |

## 필요한 것

| 항목 | 내용 |
|---|---|
| 커스텀 노드 | `custom_nodes/ComfyUI-ShortsRemake` (이 저장소, `setup.bat`이 링크 + yt-dlp 설치), ComfyUI-QwenVL, VideoHelperSuite |
| 1단계 모델 | Qwen3-VL-8B-Instruct (QwenVL 노드가 첫 실행 시 `models/LLM/`에 자동 다운로드, 약 16GB VRAM) |
| 2단계 모델 | ComfyUI 공식 `image_qwen_image_edit_2511` 템플릿과 동일: `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors`, `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `vae/qwen_image_vae.safetensors`, (선택) `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` |
| 3단계 모델 | ComfyUI 공식 `video_wan_animate2` 템플릿과 동일 (Comfy-Org/Wan-Animate-2): `diffusion_models/wan_animate_2_int8_convrot.safetensors`, `loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `clip_vision/clip_vision_h.safetensors`, `vae/Wan2_1_VAE_bf16.safetensors` |
| GPU | L40S 48GB 기준 설계. 이 테스트 PC(GPU 없음)에서는 1단계만 GGUF+CPU로 실행 가능 |
| 유튜브 | yt-dlp (python_embeded에 설치). 영상+오디오 병합은 내장 imageio-ffmpeg 사용. 로그인/연령 제한 영상은 받을 수 없음 |

## 오프라인 서버 준비

실행 중에 네트워크를 쓰는 곳은 QwenVL 노드의 모델 자동 다운로드뿐이므로, 아래를 미리 해 두면 완전히 오프라인으로 돕니다.

1. 인터넷 되는 PC에서 `Download_Models_Shorts.bat` 실행. Wan Animate 2, Qwen-Image-Edit 2511, Qwen3-VL-8B-Instruct를 모두 받습니다.
   - safetensors 파일 → `model\<폴더>\`
   - Qwen3-VL-8B-Instruct → `ComfyUI\models\LLM\Qwen\Qwen3-VL-8B-Instruct\` (config.json + safetensors 여러 개 + 토크나이저 파일)
2. `model\`과 `ComfyUI\models\LLM\`을 서버의 같은 위치로 복사. 유튜브 영상(mp4)도 미리 받아 같이 복사.
3. 서버에서는 평소대로 `Start_ComfyUI_L40S.bat`로 실행. 이 파일에 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`이 들어 있어 huggingface_hub/transformers가 인터넷을 시도하지 않습니다 (QwenVL 노드는 모델 폴더가 있으면 바로 로컬에서 읽습니다).
4. 워크플로우는 `1_Analyze_Prompts`(또는 `4_ALL_in_One`)부터 시작하고 `video_path`에 복사해 둔 mp4 경로를 넣습니다. `0_YouTube_Download_Trim`는 오프라인에서 쓰지 않습니다.

## 0단계: 0_YouTube_Download_Trim.json

`Shorts YouTube Download / Trim`의 `url`에 링크(또는 받아둔 파일 경로)를 넣고 Queue. `max_height`(기본 1080)까지 받아
ComfyUI 출력 폴더(`output/<id>.mp4`)에 저장하고, 결과 경로를 `PreviewAny`에 표시합니다.

**구간만 쓰려면** `start`, `end`에 `1:20` / `1:50` (또는 초 단위 `80` / `110`)을 넣습니다. 한쪽만 넣어도 되고, 둘 다 비우면 전체 영상입니다.
잘라낸 클립은 `<원본이름>_1m20s-1m50s.mp4`로 따로 저장되며 원본은 남습니다. 프레임 단위로 정확히 자르기 위해 H.264로 다시 인코딩합니다.
같은 링크는 다시 받지 않고 받아둔 원본에서 구간만 다시 자릅니다 (`force_redownload`로 강제).

**영상 코덱 주의**: 유튜브는 AV1/VP9로 내려올 때가 있는데, AV1은 OpenCV 프레임 읽기가 수십 배 느려 1단계 `Shorts Video Segments`가 94초 영상에 5분 이상 걸립니다. 0단계는 H.264(avc1)를 우선 받고, 아니면 `ensure_h264`(기본 켜짐)로 한 번 H.264 변환해 `<이름>_h264.mp4`를 내보냅니다. 다른 경로로 받아둔 파일도 `url`에 경로를 넣고 0단계를 돌리면 같은 변환을 거칩니다. 1단계와 3단계에는 이 변환된 파일을 넣으세요.

**옆으로 누운 영상**: 폰으로 찍은 영상은 화면은 똑바른데 파일 안에 회전 정보만 있는 경우가 많습니다. 0단계는 회전 정보가 있으면 변환하면서 실제로 똑바로 세웁니다. 회전 정보 없이 누워 있는 영상은 `rotate`를 90/180/270(시계 방향)으로 지정하면 됩니다.
표시된 경로를 복사해 1단계 또는 올인원의 `video_path`에 넣습니다.

## 올인원: 4_ALL_in_One.json

바꿀 곳은 3곳입니다.

1. `Shorts Video Segments`의 `video_path` : 영상 파일 경로 (0단계에서 받은 경로)
2. `Load Image - 프로필` : 바꿔 넣을 사람 사진 (상반신/전신)
3. (선택) `Load Image - 배경`, `Load Image - 소품` : 쓰지 않을 노드는 선택 후 **Ctrl+B**(bypass).
   배경을 bypass하면 **원본 영상의 첫 프레임이 배경**이 되어 사람만 바뀝니다.

Queue 하면 구간 분석(QwenVL, 구간 수만큼) → 참조 이미지 합성(Qwen-Image-Edit) → 구간별 Wan Animate 2 → 합치기가
한 번에 진행됩니다. 결과는 영상 옆 `<영상이름>_prompts/` 폴더에 `prompts.json`, `reference.png`,
`clip_NN.mp4`, `final.mp4` 로 남고, `Save Video`가 `ComfyUI/output/video/`에도 복사합니다.

## 1단계: 1_Analyze_Prompts.json

1. `Shorts Video Segments`의 `video_path`에 영상 파일 전체 경로 (0단계 결과 또는 받아둔 파일).
2. `split_mode`
   - `fixed`: `segment_seconds`(기본 5초)마다 자름. 마지막 조각이 `min_seconds`보다 짧으면 앞 구간에 합침.
   - `scene`: 장면 전환(컷)에서 자름. `segment_seconds`보다 긴 장면은 균등 분할, `min_seconds`보다 짧은 조각은 이웃과 합침.
     `scene_threshold`(기본 0.5)를 낮추면 컷을 더 민감하게 잡음.
3. Queue. QwenVL이 구간 수만큼 자동 반복 실행되고, 마지막에 구간별 대표 프레임을 모아 공통 프롬프트와 네거티브를 씁니다.
4. 결과: `<영상 폴더>/<영상이름>_prompts/prompts.json`, `comfyui_prompts.txt` (`out_dir`로 변경 가능)

prompts.json 구조:

```json
{
  "video_file": "...mp4", "duration": 13.2, "segment_length": 5.0, "fps": 16, "width": 1080, "height": 1920,
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

프롬프트를 손보려면 이 JSON을 직접 편집하면 됩니다. 2/3단계는 이 파일만 읽습니다.

## 2단계: 2_Compose_Reference.json

1. `Load Image - 프로필`(필수), `배경`(선택), `소품`(선택) 업로드. 안 쓰는 노드는 Ctrl+B.
2. `Shorts Reference Setup`의 `prompts_json`에 1단계 결과 경로. (prompts.json 없이 영상 경로를 넣어도 동작)
3. Queue. Setup 노드가 Picture 1=프로필, Picture 2=배경(없으면 원본 첫 프레임), Picture 3=소품 순으로 정리하고 합성 지시문을 만듭니다.
   Qwen-Image-Edit-2511이 원본 영상 비율(`max_side` 1024)로 한 장을 생성 → `reference.png`가 prompts.json 옆에 저장되고 `ComfyUI/input`에도 복사됩니다.
4. 옵션
   - `extra_instruction`: 추가 지시 (예: `wearing a red jacket`, `medium shot, upper body fills the frame`)
   - `background_mode`: 배경 이미지가 없을 때 `video_first_frame`(원본 프레임 사용) / `prompt_only`(공통 프롬프트 텍스트로만)
   - KSampler `seed`를 바꿔 다시 생성. Lightning LoRA를 켜면(Ctrl+B 해제) steps 4, cfg 1로.

## 3단계: 3_Replace_Person_WanAnimate2.json

1. `Shorts Reference Loader`의 `prompts_json`에 1단계 결과 경로 입력 (3단계의 유일한 입력 칸). 원본 영상 경로는 JSON 안의 값을 자동으로 씁니다.
2. 같은 노드가 prompts.json 옆의 `reference.png`를 자동으로 읽어 참조 이미지로 씁니다.
   없으면 `Load Image - 프로필`의 사진을 그대로 참조로 씁니다 (배경은 프롬프트로만).
3. Queue. 구간 수만큼 자동 반복:
   - `Load Video (Path)`가 구간의 원본 프레임(16fps)만 읽어 **드라이빙 영상**으로 사용
   - QwenVL이 참조 이미지 속 인물 외형을 묘사 → `Character Description:` 으로 프롬프트 앞에 결합
   - Wan Animate 2: 원본의 동작 + 참조 이미지(인물·배경·소품) + 구간 프롬프트 → 클립 생성
   - `Shorts Clip Saver` → `clip_NN.mp4`, `Shorts Concat` → `final.mp4` (prompts.json 옆 폴더), `Save Video`로 ComfyUI output에도 복사
4. 옵션
   - `clips`: `1,3-4` 처럼 일부 구간만 생성
   - `max_side`: 출력 해상도 (원본 비율 유지, 16의 배수). 기본 832
   - `prompt_template`: 최종 프롬프트 형식. 기본 `Character Description: {character}\nBackground description: {common} {segment}`
   - 배경이 참조 이미지와 달라지면 `WanAnimate2ToVideo`의 `reference_image_strength`를 1.2~1.5로
   - 컨텍스트 윈도우 스위치(노드 588)는 꺼져 있음. 5초 클립(81프레임)은 필요 없음

## 5번: 5_I2V_6seg_from_prompts.json (Wan 2.2 I2V 6구간)

`video_wan22_14b_i2v_6seg_30s`와 같은 그래프에 `Shorts Prompts Fanout` 노드를 붙인 것입니다. 구간별 프롬프트 6개를 손으로 적는 대신 prompts.json에서 읽어 채웁니다.

1. `Shorts Prompts Fanout`의 `prompts_json`에 1번 결과 경로. 구간이 6개보다 많으면 `first_segment`를 7, 13으로 바꿔 여러 번 돌립니다.
2. `시작 이미지`에 합성한 참조 이미지(전신, 세로 영상이면 9:16). width/height는 세로면 720/1280.
3. 파란 `공통 스타일` 노드는 모든 구간 뒤에 붙는 문장입니다. Fanout의 `common` 출력을 여기 연결하면 1번이 뽑은 배경 설명이 대신 들어갑니다.
4. Queue. 구간 1~6이 앞 구간의 마지막 프레임에서 이어져 생성되고 `output/video/wan14b_30s/full_*.mp4`로 합쳐 저장됩니다.
   마지막 프레임은 `next_start_*.png`로 따로 저장됩니다.
5. **구간이 6개보다 적으면** (30초 미만 영상) `Shorts Segments Collect`가 있는 구간까지만 실행하고 끝냅니다. 없는 구간의 샘플러는 돌지 않습니다.
   **6개보다 많으면** `first_segment`를 7, 13으로 바꿔 다시 돌리고, 그때 `시작 이미지`에 이전 회차의 `next_start_*.png`를 넣으면 이어집니다.

3번과의 차이: 3번은 원본 영상의 동작을 그대로 옮기고(Wan Animate 2), 5번은 프롬프트 설명대로 움직입니다(Wan 2.2 I2V + lightx2v 4-step). 춤처럼 동작 재현이 중요하면 3번, 속도와 안정성이 중요하면 5번입니다.

## API 방식 실행 (서버 자동화)

`*.api.json` 파일은 `POST http://<server>:8188/prompt` 에 `{"prompt": <api.json 내용>}` 로 감싸 보내면 됩니다.

| 파일 | 바꿀 값 |
|---|---|
| 0_ | 노드 `"11"`의 `url` |
| 1_ | 노드 `"1"`의 `video_path` |
| 4_ALL | 노드 `"1"`의 `video_path`; 노드 `"21"`(프로필) / `"22"`(배경) / `"23"`(소품)의 `image` 파일명. 배경·소품을 안 쓰면 `"22"`, `"23"` 항목을 지우고 노드 `"20"`의 `background`/`props` 입력을 제거 |
| 2_ | 노드 `"20"`의 `prompts_json`, 노드 `"21"`~`"23"`의 `image` |
| 3_ | 노드 `"59"`의 `prompts_json`, 노드 `"52"`의 `image` |

```bash
curl -X POST http://127.0.0.1:8188/prompt -H "Content-Type: application/json" -d @4_ALL_in_One.api.json
```

## 워크플로우 파일 다시 만들기

노드나 프롬프트 문구를 바꿨으면 `tools/build_shorts_workflows.py`를 수정하고 실행하면 UI용 JSON과 API JSON이 함께 다시 생성됩니다.

```bat
python_embeded\python.exe tools\build_shorts_workflows.py
```

## 이 PC(CPU)에서 확인한 것

- 커스텀 노드 9개 로딩, 다섯 워크플로우 모두 누락 노드 없이 UI에서 열림 (ComfyUI --cpu 서버로 API 검증)
- `Shorts YouTube Download`: 19초짜리 공개 영상 실제 다운로드 (360p, 영상+오디오 병합) 확인
- `Shorts Reference Setup / Save / Loader`: 합성 영상으로 Picture 배치·지시문·캔버스 크기·저장·우선순위 확인
- 1단계를 GGUF Qwen3VL-4B + CPU로 실제 실행해 prompts.json 생성 확인 (이전 버전)
- 2/3단계는 Qwen-Image-Edit / Wan 모델이 없어 구조 검증만
