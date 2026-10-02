# ComfyUI-Manager: 이미지·영상 생성 관리

ComfyUI와 따로 도는 작은 웹 서버입니다. 화면 위쪽에 탭이 둘 있습니다.
(노드 팩을 설치해 주는 ComfyUI-Manager와는 이름만 같은 다른 프로그램입니다.)

- **생성**: 이미지 → 시나리오 → 영상을 ComfyUI API로 순서대로 실행합니다
- **보관함**: `output` 폴더의 이미지와 영상을, 만든 워크플로우와 프롬프트와 함께 보여 줍니다. ComfyUI가 꺼져 있어도 됩니다

## 보관함

- **파일 확인**: 이미지나 영상을 끌어다 놓으면 그것을 만든 워크플로우 이름과 프롬프트(구간이 여러 개면 전부), 네거티브, 모델·LoRA, seed, 샘플러 설정이 나옵니다
- **목록**: `output` 폴더 전체를 읽어 썸네일로 보여 줍니다. 정렬은 최신순 / 오래된순 / **워크플로우별** / 이름순, 워크플로우 이름을 누르면 그것만 걸러 보고, 프롬프트 내용으로 검색할 수 있습니다
- **워크플로우 JSON 받기 / workflow 폴더에 저장**: 파일에 들어 있는 워크플로우를 꺼냅니다. `workflow 폴더에 저장`을 하면 ComfyUI 워크플로우 목록에 그 이름으로 나타납니다 (영상을 ComfyUI에 끌어다 놓으면 "Unsaved Workflow"로 열리는 것을 대신하는 방법)

정보는 ComfyUI가 파일 안에 넣어 둔 메타데이터에서 읽습니다 (PNG, WebP/JPEG, MP4/WebM/MKV).

| 표시 | 뜻 |
|---|---|
| 워크플로우 이름 | `workflow\` 폴더의 파일과 ID가 같거나 노드 구성이 같으면 그 파일 이름 |
| `미등록 (모델명)` | `workflow\` 폴더에 같은 워크플로우가 없음. 저장하면 다음부터 그 이름으로 묶입니다 |
| `메타데이터 없음` | ComfyUI 정보가 없는 파일 (내려받은 영상, 메타데이터 저장을 끈 경우) |
| "실행 중에 … 노드가 쓴 프롬프트" | Qwen 같은 노드가 실행 중에 만든 글은 파일에 남지 않습니다. 대신 그 노드에 준 지시문을 보여 줍니다 |

Video Combine이 영상 옆에 같이 저장하는 미리보기 PNG는 기본으로 숨깁니다. 목록 정보와 썸네일은 `tools\ComfyUI-Manager\data\`에 저장되어 두 번째부터는 빨리 열립니다 (지워도 다시 만들어집니다).

## 생성

직접 생성하지는 않고, 단계마다 ComfyUI API에 작업을 넣고 결과를 받아 옵니다.

1. **이미지**: z-image turbo로 영상의 첫 프레임을 만듭니다 (가지고 있는 이미지를 올려도 됩니다)
2. **시나리오**: Qwen이 이미지를 보고 이어지는 영상을 구간 프롬프트 6개로 씁니다 (화면에서 고칠 수 있습니다)
3. **영상**: 시작 이미지와 구간 프롬프트를 Wan 2.2 SVI 워크플로우에 넣어 영상을 만듭니다

## 실행

1. ComfyUI를 먼저 켭니다 (`Start_ComfyUI_L40S.bat`)
2. `tools\ComfyUI-Manager\Start_ComfyUI_Manager.bat` 실행 (보관함만 쓸 때는 ComfyUI 없이 이것만 실행해도 됩니다)
3. 브라우저에서 `http://127.0.0.1:8288` (다른 PC에서는 `http://<서버 IP>:8288`)

추가로 설치할 파이썬 패키지는 없습니다 (ComfyUI에 들어 있는 aiohttp, Pillow, PyAV만 씁니다).

## 생성 탭 화면

- **1. 이미지**: 프롬프트, 크기, seed(0이면 랜덤)를 정하고 `이미지 생성`. 여러 장 만들면 아래 썸네일에서 쓸 것을 고릅니다
- **2. 시나리오**: 연출 방향(선택), 구간 수, 분석 방법과 모델을 고르고 `시나리오 생성`. 오른쪽에 구간별 프롬프트가 나오고 고친 뒤 `시나리오 저장`
- **3. 영상**: 워크플로우를 고르고 `영상 생성`. 구간이 끝날 때마다 미리보기가 추가되고 마지막에 최종 영상이 맨 위에 표시됩니다
- 위쪽 `1→3 한 번에 실행`은 현재 입력값으로 세 단계를 이어서 돌립니다. `중단`은 실행 중인 ComfyUI 작업을 멈춥니다

### 분석 방법

| 방법 | 쓰는 것 | 비고 |
|---|---|---|
| Qwen GGUF 직접 호출 (기본) | `Shorts Qwen GGUF Vision` 노드 (`templates/scenario_gguf.api.json`) | QwenVL-Mod의 GGUF 모델을 프리셋·채팅 규약 없이 부릅니다. 받아 둔 모델이 목록 맨 위에 옵니다. Mod와 ShortsRemake 노드가 필요합니다 |
| Qwen Chat | ComfyUI-QwenVL-Mod의 채팅 엔드포인트 (`/qwenvl/chat`) | 같은 모델이지만 채팅은 워크플로우 조수 역할이라 글이 짧게 나옵니다 |
| QwenVL 노드 | 원본 ComfyUI-QwenVL 노드 (`templates/scenario_qwenvl.api.json`) | 기본 Qwen3-VL-8B-Instruct |
| 직접 작성 | 모델 없음 | 빈 칸 6개를 만들어 직접 씁니다 |

`프롬프트 분량`이 `자세히`면 구간마다 시작 자세, 동작 순서, 표정, 옷·머리 움직임, 카메라, 끝 자세를 쓰게 하는 지시문(구간당 90~130단어)을,
`보통`이면 짧은 지시문(40~70단어)을 씁니다. Qwen Chat에는 적용되지 않습니다.
같은 이미지와 모델(Qwen3.5-9B GGUF Q8)로 재 본 구간당 단어 수: Qwen Chat 약 20(영어 대신 한국어로 나와서 이후 지시문에 영어 지정을 추가함, 추가 후는 재지 않음), 직접 호출 + 보통 약 30, 직접 호출 + 자세히 약 85.

모델이 형식을 지키지 않아 구간을 못 읽으면 경고와 함께 답변 원문을 보여 줍니다. 원문을 보고 빈 칸을 채우거나 다시 생성하세요.

### 영상 워크플로우

| 방식 | 쓰는 파일 | 비고 |
|---|---|---|
| Wan 2.2 SVI | `workflow/Wan2.2_I2V_SVI_Workflow_Kenpechi_v3.5.json` | 시작 이미지(`Load Image_1st`)와 `1st_`~`6th_` 프롬프트 노드에 값을 넣습니다. LoRA, 샘플러 등 나머지는 워크플로우 파일 그대로입니다 |
| Wan 2.2 I2V 6구간 | `workflow/5_I2V_6seg_from_prompts.api.json` | 구간 수만큼만 실행합니다. 해상도는 시작 이미지 비율로 자동 |

SVI 워크플로우는 구간이 6개로 고정입니다. 시나리오 구간이 더 적으면 마지막 프롬프트를 반복합니다.
`워크플로우 설정`을 펼치면 워크플로우의 숫자 설정(구간별 초, Width/Height, 스텝 등)을 이번 실행에만 바꿀 수 있습니다.
`시작 이미지 비율에 맞춤`을 켜면 Width/Height를 긴 변은 유지하고 시작 이미지 비율로 맞춥니다.

SVI 워크플로우는 UI 저장 형식(Get/Set 노드, 서브그래프 포함)이라 실행할 때 `comfy_convert.py`가 API 형식으로 바꿉니다.
ComfyUI에서 워크플로우를 고친 뒤 저장하면 다음 실행부터 반영됩니다. 변환 대신 ComfyUI의 `Export (API)`로 저장한 파일을 쓰려면
`templates/video_svi.api.json`으로 넣어 두세요. 이 파일이 있으면 그것을 씁니다.

## VRAM

단계가 바뀔 때 이전 단계 모델을 내립니다 (`/free`, Qwen Chat은 `/qwenvl/chat/unload`). 같은 단계를 반복할 때는 모델을 그대로 둡니다.
ComfyUI UI에서 다른 작업을 같이 돌리면 이 순서가 깨지니 영상 단계 중에는 피하세요. `VRAM 비우기` 버튼으로 직접 내릴 수도 있습니다.

## 저장 위치

- 생성 이미지: `output/webapp/zimage_<날짜>_NNNNN_.png`
- 시나리오: `output/<날짜_시간>_<이름>_prompts/` 안에 `prompts.json`, `reference.png`(시작 이미지), `model_answer.txt`(모델 답변 원문)
  - `prompts.json`은 Shorts 워크플로우(5번 등)의 Fanout / Reference Loader 노드가 읽는 형식과 같습니다
- 영상: 워크플로우의 저장 노드가 정한 곳 (SVI는 `output/Video/wanvideo_<날짜>_…mp4`)
- 시작 이미지는 영상 단계 전에 ComfyUI `input/webapp/`으로 복사됩니다

## 설정 (`config.json`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `host` / `port` | `0.0.0.0` / `8288` | 웹 프로그램 주소. 이 PC에서만 쓰려면 host를 `127.0.0.1`로 |
| `comfy_url` | `http://127.0.0.1:8188` | ComfyUI 주소 |
| `output_dir` | (빈 값) | 시나리오 폴더 위치. 비우면 `ComfyUI-Easy-Install/output` |
| `svi_workflow` | `workflow/Wan2.2_…v3.5.json` | SVI 워크플로우 파일 |
| `i2v_api` | `workflow/5_I2V_6seg_from_prompts.api.json` | I2V 6구간 API 템플릿 |

실행 옵션 `--host --port --comfy --output-dir`가 config.json보다 우선합니다.
이미지 모델, 스텝, 샘플러는 `templates/image_zimage.api.json`에서, QwenVL 노드 기본값은 `templates/scenario_qwenvl.api.json`에서 바꿉니다.

## 오프라인 서버에 옮길 것

`tools\ComfyUI-Manager\` 폴더와 `workflow\`의 두 파일(SVI 워크플로우, `5_I2V_6seg_from_prompts.api.json`).
모델과 커스텀 노드는 해당 워크플로우를 ComfyUI에서 돌릴 때와 같습니다.

## 파일

| 파일 | 내용 |
|---|---|
| `app.py` | 웹 서버, 단계 실행, ComfyUI 통신 |
| `library.py` | 파일의 메타데이터 읽기 (워크플로우·프롬프트·모델), output 폴더 목록 |
| `comfy_convert.py` | UI 워크플로우 JSON → API JSON 변환 |
| `static/index.html` | 화면 |
| `templates/*.api.json` | 이미지 생성 / QwenVL 노드 분석용 API 템플릿 |
