# Qwen Chat 테스트 (6_QwenChat_Test)

Qwen Chat(ComfyUI-QwenVL-Mod의 사이드바 채팅)이 **열려 있는 워크플로우를 읽고 고치고 실행**하는지 확인하는 작은 워크플로우입니다. 그래프는 z-image turbo로 이미지 한 장을 만드는 것이 전부입니다.

## 준비

1. ComfyUI에서 `6_QwenChat_Test` 워크플로우를 엽니다
2. 왼쪽 사이드바에서 말풍선 아이콘 **Qwen Chat**을 엽니다
3. 위쪽 Model에서 모델을 고릅니다 (GGUF는 mmproj 파일이 같은 폴더에 있어야 이미지를 봅니다)
4. `분석할 이미지` 노드에 테스트할 이미지를 올립니다

## Qwen Chat이 보는 것과 할 수 있는 것

- 열려 있는 워크플로우의 모든 노드와 위젯 값 (노드 200개까지). 메모 노드의 글도 같이 가므로 이 워크플로우의 메모는 짧게 두었습니다
- `Load Image` 노드에 들어 있는 이미지 (최대 3장, 자동으로 첨부). 채팅창의 첨부 버튼으로 올린 이미지가 있으면 그것만 봅니다
- 할 수 있는 것: 위젯 값 바꾸기, 노드 bypass/켜기, Queue 실행. 노드를 추가하거나 선을 연결하지는 못합니다
- Queue를 실행하기 직전에 채팅 모델은 VRAM에서 자동으로 내려갑니다. 다음 대화 때 다시 올라오므로 그때 첫 답이 느립니다
- 대화 기록이 쌓이면 이전 지시가 섞이니, 테스트를 바꿀 때는 채팅창의 Clear로 지우세요

## 순서대로 보내 볼 문장

**1. 대화** (모델이 뜨는지)
```
안녕, 이 워크플로우는 뭘 하는 거야?
```

**2. 이미지 분석** (`분석할 이미지`를 실제로 보는지)
```
이 이미지를 한국어로 자세히 설명해줘. 인물, 옷, 장소, 조명, 카메라 구도.
```

**3. 값 바꾸기** (적용되면 답 아래에 Applied 목록이 붙고 노드 값이 바뀝니다)
```
KSampler의 steps를 6으로 바꿔줘
```
```
이미지 크기를 720x1280으로 바꿔줘
```

**4. bypass**
```
이미지 확대 노드를 bypass 해줘
```
```
이미지 확대 노드를 다시 켜줘
```

**5. 프롬프트 작성 + 실행** (`이미지 프롬프트`에 영어 프롬프트를 쓰고 Queue까지)
```
해변을 걷는 강아지 사진을 생성해줘
```

**6. 이미지를 보고 프롬프트 작성** (이미지 → 글 → 위젯)
```
분석할 이미지와 같은 분위기의 이미지를 만들 프롬프트를 이미지 프롬프트 노드에 써줘. 실행은 하지 마.
```

**확인할 것**
- 답이 한국어로 오는지, 없는 노드나 위젯을 지어내지 않는지 (Rejected 목록)
- 5번에서 Queue가 실제로 들어가고 `결과 이미지`가 나오는지
- 답이 중간에 끊기면 설정(⚙)의 Max tokens를 올립니다

## 7. 웹 프로그램(tools/video_webapp)의 시나리오 지시문

웹 프로그램은 2단계에서 아래 지시문과 이미지를 Qwen Chat에 보냅니다. 여기서 같은 모델로 미리 보내 보면 그 모델이 형식을 지키는지 알 수 있습니다.

1. 설정(⚙)에서 **Max tokens를 2048**로 올립니다 (기본 1024는 6구간을 쓰기에 모자랍니다)
2. Clear로 대화를 지우고, 아래 지시문을 그대로 붙여 보냅니다

```
This is not a workflow request: leave "actions" and "choices" empty and put the whole answer in "message". Look at the attached image.
The image is the FIRST FRAME of a video. Write a 6-part scenario that continues from it. Each part is one continuous shot of about 5 seconds and starts exactly where the previous part ended. Keep the same person, outfit, location and lighting as in the image unless the direction says otherwise. Make the parts flow into each other as one story. No text, captions or logos in the scene.
Write "message" as plain text lines in exactly this layout (labels in capitals, one item per line, no markdown):
SUMMARY_KO: Korean, 1-2 sentences describing the whole scenario
COMMON: English, 20-40 words: the look, outfit, location, lighting and visual style that stay the same
PART 1: English, 40-70 words, natural sentences: what happens from the start to the end of this shot, body motion, expression, camera framing and movement
KO 1: Korean, one sentence describing part 1
PART 2: ...
KO 2: ...
(continue the same way up to PART 6 and KO 6)
```

**기대하는 답**: `SUMMARY_KO:`, `COMMON:`, `PART 1:` … `PART 6:`, `KO 1:` … `KO 6:` 로 시작하는 줄들. 이렇게 나오면 웹 프로그램에서 그대로 6구간으로 읽힙니다.

**다르게 나오면**: 형식 없이 줄글이거나 PART가 6개보다 적으면 웹 프로그램에서 경고가 뜹니다. 모델을 바꿔 보거나, 웹 프로그램의 분석 방법을 `QwenVL 노드`로 바꾸세요.

웹 프로그램은 워크플로우 없이(빈 그래프) 보내고 여기서는 이 워크플로우가 같이 가므로 결과가 완전히 같지는 않습니다. 연출 방향을 넣어 보려면 `Write "message" as plain text` 줄 앞에 `Direction from the user (follow it): …` 한 줄을 추가하세요.
