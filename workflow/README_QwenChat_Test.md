# Qwen Chat 테스트: 이미지 분석 → 영상 프롬프트 (6_QwenChat_Test)

Qwen Chat(ComfyUI-QwenVL-Mod의 사이드바 채팅)이 **시작 이미지를 보고 영상 구간 프롬프트 6개를 써서 노드에 넣는지** 확인하는 워크플로우입니다.

그래프는 Wan 2.2 SVI 워크플로우에서 프롬프트를 넣는 부분만 떼어 온 것입니다 (노드 이름이 같습니다).

| 노드 | 역할 |
|---|---|
| `Load Image_1st` | 시작 이미지. Qwen Chat이 자동으로 봅니다 |
| `1st_CLIP Text Encode (Prompt)` ~ `6th_…` | 구간 1~6 프롬프트. Qwen Chat이 여기에 씁니다 |
| `CLIP Text Encode (Prompt)_Negative` | 네거티브 프롬프트 |

출력 노드가 없어서 Queue로 실행되는 것은 없습니다. 채팅만으로 테스트합니다.

## 준비

1. ComfyUI에서 `6_QwenChat_Test` 워크플로우를 엽니다
2. 왼쪽 사이드바에서 말풍선 아이콘 **Qwen Chat**을 엽니다
3. 위쪽 Model에서 모델을 고릅니다. **받아 둔 파일과 같은 이름**을 골라야 합니다 (예: `…Q8_0.gguf`만 받았으면 Q8_0)
4. 설정(⚙)에서 **Max tokens를 2048**로 올립니다 (기본 1024는 프롬프트 6개를 쓰다 끊길 수 있습니다)
5. `Load Image_1st` 노드에 시작 이미지를 올립니다

## 받아 둔 모델이 쓰이는지 확인하는 법

첫 문장을 보낸 뒤 **ComfyUI 콘솔 창**을 봅니다.

| 콘솔에 나오는 줄 | 뜻 |
|---|---|
| `[QwenVL] Using model from alternate LLM path: …\model\LLM\GGUF\…` | `model\LLM`에 받아 둔 파일을 찾았음 |
| `[QwenVL] Using mmproj from alternate LLM path: …mmproj-BF16.gguf` | 이미지용 mmproj도 찾았음 |
| `[QwenVL] Loading GGUF: <파일명> (device=cuda, gpu_layers=-1, ctx=…)` | 그 파일을 GPU에 올리는 중. `device=cpu`면 GPU를 못 쓰는 것 |
| 다운로드 진행 표시, `hf_hub_download failed` | 파일을 못 찾아 새로 받으려는 것. 고른 모델 이름과 받아 둔 파일 이름이 다릅니다 |

그 밖에 답이 온 뒤 VRAM 사용량이 모델 크기만큼(Q8_0 9B는 약 11GB) 늘어 있으면 올라간 것입니다. `ComfyUI\models\LLM\GGUF` 아래에 새 파일이 생기지 않았는지도 확인하면 됩니다.

## Qwen Chat이 보는 것과 할 수 있는 것

- 열려 있는 워크플로우의 모든 노드와 위젯 값 (노드 200개까지, 메모 노드의 글 포함)
- `Load Image` 노드에 들어 있는 이미지 (최대 3장, 자동 첨부). 채팅창의 첨부 버튼으로 올린 이미지가 있으면 그것만 봅니다
- 할 수 있는 것: 위젯 값 바꾸기, 노드 bypass/켜기, Queue 실행. 답 아래 Applied 목록이 실제로 바뀐 것입니다
- 대화 기록이 쌓이면 이전 지시가 섞이니, 테스트를 바꿀 때는 채팅창의 Clear로 지우세요

## 순서대로 보내 볼 문장

**1. 이미지 분석** (이미지를 실제로 보는지)
```
시작 이미지를 한국어로 자세히 설명해줘. 인물, 옷, 장소, 조명, 카메라 구도. 노드는 건드리지 마.
```

**2. 영상 프롬프트 1개** (답으로만 받기)
```
이 이미지를 첫 프레임으로 하는 5초 영상의 프롬프트를 영어로 써줘. 인물의 움직임과 카메라 움직임을 넣고, 노드는 건드리지 마.
```

**3. 구간 프롬프트 6개를 노드에 쓰기** (핵심 테스트, 자세한 요청)
```
이 이미지를 첫 프레임으로 이어지는 영상을 6구간(구간당 약 5초)으로 나눠줘. 앞 구간이 끝난 자세에서 다음 구간이 시작해야 해. 구간마다 영어로 90~130단어의 한 문단을 쓰고, 문단에는 순서대로 (1) 시작 자세와 화면 속 위치, (2) 동작을 시간 순서로 2~3단계(손, 고개, 시선, 체중 이동, 속도), (3) 표정과 그 변화, (4) 머리카락·옷·배경·빛의 움직임, (5) 카메라의 샷 크기·앵글·움직임과 속도, (6) 끝 자세를 넣어줘. 구간마다 다른 동작이어야 하고 같은 문장을 반복하지 마. 완성한 문단을 1st~6th CLIP Text Encode (Prompt) 노드의 text에 순서대로 넣어줘. 답변 message에는 프롬프트를 다시 적지 말고 한국어로 한 줄 요약만 써. 네거티브 노드는 그대로 두고, 실행(queue)은 하지 마.
```

**3-1. 같은 것을 짧게 요청** (분량 비교용)
```
이 이미지를 첫 프레임으로 이어지는 영상을 6구간으로 나눠줘. 각 구간은 약 5초이고 앞 구간이 끝난 자세에서 이어져야 해. 구간마다 영어 프롬프트(40~70단어, 인물 동작과 카메라 움직임)를 써서 1st~6th CLIP Text Encode (Prompt) 노드의 text에 순서대로 넣어줘. 네거티브 노드는 그대로 두고, 실행(queue)은 하지 마.
```

**4. 연출 방향 주기**
```
창밖을 보다가 돌아서 카메라 쪽으로 걸어오며 미소 짓는 내용으로 6구간을 다시 써서 같은 노드에 넣어줘. 실행은 하지 마.
```

**5. 한 구간만 고치기**
```
3rd 구간만 더 천천히 움직이고 카메라가 가까이 다가가게 고쳐줘. 다른 구간은 그대로 둬.
```

**확인할 것**
- 3번 답 아래 Applied 목록에 1st~6th 여섯 개가 모두 있는지 (Rejected가 있으면 노드나 위젯 이름을 잘못 짚은 것)
- 프롬프트에 이미지 속 인물, 옷, 장소가 반영되어 있는지
- 구간이 서로 이어지는지, 같은 문장의 반복이 아닌지
- 5번에서 3rd만 바뀌는지

채워진 프롬프트는 SVI 워크플로우의 같은 이름 노드에 그대로 붙여 넣을 수 있습니다.

## 글의 분량은 요청 문장이 정합니다

모델은 요청받은 만큼만 씁니다. 한 줄로 부탁하면 짧게, 구간마다 무엇을 넣을지 적어 주면 길게 나옵니다. 이 워크플로우와 같은 이미지, 같은 모델(Qwen3.5-9B GGUF Q8)로 잰 결과입니다.

| 보낸 문장 | 구간당 영어 단어 수 |
|---|---|
| 3-1번 (40~70단어로 요청) | 41~46 |
| 3번 (90~130단어 + 넣을 내용 6가지) | 115~130 |

두 경우 모두 1st~6th 노드 여섯 개에 영어로 정확히 들어갔습니다. `video_dasiwa-wan22` 워크플로우의 Qwen3_VQA 노드에 긴 지시문이 들어 있는 것도 같은 이유입니다: 그 글은 모델에게 주는 작업 지시서이고, 지시가 자세할수록 결과도 자세해집니다.

그 밖에 분량에 영향을 주는 것:
- **Max tokens**: 답이 끊기면 올립니다. 3번은 2048이면 충분했습니다 (답변 message에 프롬프트를 다시 적지 말라고 한 이유)
- **temperature**: 기본 0.2는 표현이 단조롭습니다. 0.6~0.8로 올리면 어휘가 다양해집니다
- **문장에 넣는 요구**: 더 풍부하게 하려면 단어 수를 올리거나 조명, 질감, 분위기, 배경 움직임 같은 항목을 더 적습니다

채팅 없이 Queue로 같은 일을 하려면 `7_Qwen_Image_to_VideoPrompts` 워크플로우를 쓰면 됩니다 (같은 GGUF 모델을 직접 부르고 지시문이 노드에 들어 있습니다).

## 6. 웹 프로그램(tools/video_webapp)의 시나리오 지시문

웹 프로그램에서 분석 방법을 `Qwen Chat`으로 고르면 아래 지시문과 이미지를 Qwen Chat에 보내고, 답을 구간 6개로 읽습니다 (기본값인 `Qwen GGUF 직접 호출`은 채팅을 거치지 않습니다). 여기서 같은 모델로 미리 보내 보면 그 모델이 형식을 지키는지 알 수 있습니다.

Clear로 대화를 지우고, 아래 지시문을 그대로 붙여 보냅니다.

```
This is not a workflow request: leave "actions" and "choices" empty and put the whole answer in "message". Look at the attached image.
The image is the FIRST FRAME of a video. Write a 6-part scenario that continues from it. Each part is one continuous shot of about 5 seconds and starts exactly where the previous part ended. Keep the same person, outfit, location and lighting as in the image unless the direction says otherwise. Make the parts flow into each other as one story. No text, captions or logos in the scene.
Write "message" as plain text lines in exactly this layout (labels in capitals, one item per line, no markdown):
SUMMARY_KO: Korean, 1-2 sentences describing the whole scenario
COMMON: English, 20-40 words: the look, outfit, location, lighting and visual style that stay the same
PART 1: ENGLISH ONLY, 60-100 words, natural sentences: what happens from the start to the end of this shot, body motion, expression, camera framing and movement
KO 1: Korean, one sentence describing part 1
PART 2: ...
KO 2: ...
(continue the same way up to PART 6 and KO 6)
```

**기대하는 답**: `SUMMARY_KO:`, `COMMON:`, `PART 1:` … `PART 6:`, `KO 1:` … `KO 6:` 로 시작하는 줄들. 이렇게 나오면 웹 프로그램에서 그대로 6구간으로 읽힙니다.

**다르게 나오면**: 형식 없이 줄글이거나 PART가 6개보다 적으면 웹 프로그램에서 경고가 뜹니다. 모델을 바꿔 보거나, 웹 프로그램의 분석 방법을 `QwenVL 노드`로 바꾸세요.

웹 프로그램은 워크플로우 없이(빈 그래프) 보내고 여기서는 이 워크플로우가 같이 가므로 결과가 완전히 같지는 않습니다. 연출 방향을 넣어 보려면 `Write "message" as plain text` 줄 앞에 `Direction from the user (follow it): …` 한 줄을 추가하세요.
