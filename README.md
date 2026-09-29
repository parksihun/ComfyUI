# ComfyUI-Easy-Install 작업 저장소

ComfyUI-Easy-Install 폴더 안에서 **우리가 만든 것만** 추적하는 저장소입니다.
`python_embeded`, `ComfyUI` 본체, 모델, 다른 커스텀 노드는 `.gitignore`로 제외되어 있어
어떤 PC의 ComfyUI-Easy-Install 폴더에 넣어도 충돌하지 않습니다.

```
.
├── workflow/                      워크플로우 JSON (+ API 형식 .api.json) 와 설명서
│   ├── Shorts_1_Analyze_Prompts.json          영상 → 구간 분할 → QwenVL → prompts.json
│   ├── Shorts_2_Replace_Person_WanAnimate2.json 프로필 사진으로 인물 교체 → clip_NN.mp4 → final.mp4
│   ├── YouTube_Video_Analysis_GPU.json / _CPU.json  영상 내용 요약 (단일 실행)
│   └── README_Shorts_Remake.md
├── custom_nodes/ComfyUI-ShortsRemake/   위 워크플로우가 쓰는 커스텀 노드 5개
├── tools/build_shorts_workflows.py      Shorts_* 워크플로우 JSON 생성 스크립트
├── setup.bat                            새 PC에서 노드 링크 + 워크플로우 복사
└── Start ComfyUI CPU.bat                GPU 없는 PC용 실행 스크립트 (--cpu)
```

## 새 PC에 받기

이미 ComfyUI-Easy-Install이 설치된 폴더에서:

```bat
cd C:\path\to\ComfyUI-Easy-Install
git init
git remote add origin https://github.com/parksihun/ComfyUI.git
git fetch origin
git checkout -f -b main origin/main
setup.bat
```

`setup.bat`은 `custom_nodes\ComfyUI-ShortsRemake`를 `ComfyUI\custom_nodes\`에 정션(폴더 링크)으로 연결하고,
`workflow\*.json`을 ComfyUI 사이드바용 폴더로 복사합니다. 이후 ComfyUI를 재시작하면 됩니다.

## 이 PC에서 수정 후 올리기

```bat
git add -A
git commit -m "설명"
git push
```

`ComfyUI\custom_nodes\ComfyUI-ShortsRemake`는 정션이므로 그 안에서 고쳐도 `custom_nodes\ComfyUI-ShortsRemake`가 바뀌고 그대로 커밋됩니다.
워크플로우를 ComfyUI UI에서 저장하면 `ComfyUI\user\default\workflows\`에 저장되므로, 올리려면 `workflow\`로 복사해야 합니다.

## 필요한 것

- ComfyUI-QwenVL, ComfyUI-VideoHelperSuite, ComfyUI-KJNodes (Easy-Install 기본 포함)
- 모델: `workflow/README_Shorts_Remake.md` 참고 (Qwen3-VL-8B, Wan Animate 2 5개 파일)
- GPU 서버(L40S) 기준. GPU 없는 PC는 `Start ComfyUI CPU.bat` + 1단계 워크플로우를 GGUF 노드로 바꿔 확인만 가능
