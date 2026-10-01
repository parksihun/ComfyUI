# ComfyUI-Easy-Install 작업 저장소

ComfyUI-Easy-Install 폴더 안에서 **우리가 만든 것만** 추적하는 저장소입니다.
`python_embeded`, `ComfyUI` 본체, 모델, 다른 커스텀 노드는 `.gitignore`로 제외되어 있어
어떤 PC의 ComfyUI-Easy-Install 폴더에 넣어도 충돌하지 않습니다.

```
.
├── workflow/                      워크플로우 JSON (+ API 형식 .api.json) 와 설명서
│   ├── 0_YouTube_Download_Trim.json           유튜브 링크(또는 파일) → mp4, start/end 구간 잘라내기
│   ├── 1_Analyze_Prompts.json                 영상 → 구간 분할 → QwenVL → prompts.json
│   ├── 2_Compose_Reference.json               프로필+배경+소품 → Qwen-Image-Edit → reference.png
│   ├── 3_Replace_Person_WanAnimate2.json      참조 이미지로 인물 교체 → clip_NN.mp4 → final.mp4
│   ├── 4_ALL_in_One.json                      올인원 (1→2→3): 영상 파일 + 사진 → 인물 교체 영상
│   ├── 5_I2V_6seg_from_prompts.json           빠른 대안: 프롬프트 + 참조 이미지 → Wan 2.2 I2V 6구간 30초
│   ├── 6_QwenChat_Test.json                   Qwen Chat(QwenVL-Mod 사이드바) 테스트용 작은 z-image 그래프, 보낼 문장은 README_QwenChat_Test.md
│   ├── video_wan22_14b_i2v_6seg_30s.json      5번의 원본 (수동 프롬프트 6구간)
│   ├── image_qwen_image_2_1_image_edit.json   Qwen-Image 2.1 이미지 편집 (옷 갈아입히기 등, ComfyUI 0.37 이상)
│   ├── Wan2.2_I2V_SVI_Workflow_Kenpechi_v3.5.json  Wan 2.2 I2V + SVI 장편 연결 (Kenpechi, 추가 커스텀 노드 필요)
│   ├── YouTube_Video_Analysis_GPU.json / _CPU.json  영상 내용 요약 (단일 실행)
│   └── README_Shorts_Remake.md
│   └── wan22_14b_flf_chain.json               Wan 2.2 첫/끝 프레임 구간 연결 (WanChain 노드 사용)
├── custom_nodes/ComfyUI-ShortsRemake/   위 워크플로우가 쓰는 커스텀 노드 10개 (yt-dlp 필요, Install_Python_Packages.bat으로 설치)
├── custom_nodes/ComfyUI-WanChain/       Wan 2.2 FLF 구간 연결 보조 노드 2개 (Load Image Optional, Collect Segments)
├── custom_nodes/ComfyUI-Wan22FMLF/      (외부) Wan Advanced I2V - SVI 워크플로우용
├── custom_nodes/ComfyUI-Frame-Interpolation/ (외부) RIFE VFI + ckpts/rife/rife49.pth
├── custom_nodes/ComfyUI-Custom-Scripts/ (외부) pysssss Math Expression 등
├── tools/build_shorts_workflows.py      0_~4_ 워크플로우 JSON 생성 스크립트
├── tools/video_webapp/                  영상 생성 웹 프로그램: z-image 이미지 → Qwen 시나리오 6개 → Wan 2.2 SVI 영상
│                                        (ComfyUI API 연동, Start_Video_WebApp.bat으로 실행, 설명은 폴더 안 README.md)
├── Setup_Folders.bat                    폴더 생성 + workflow/custom_nodes 정션 (새 PC에서 한 번, extra_model_paths.yaml과 함께)
├── Install_Python_Packages.bat          워크플로우용 추가 파이썬 패키지 (yt-dlp 등, 인터넷 필요)
├── Start_ComfyUI_L40S.bat               GPU 서버(L40S) 실행 스크립트 (SageAttention, 외부 접속, 오프라인 설정 포함)
└── Start ComfyUI CPU.bat                GPU 없는 PC용 실행 스크립트 (--cpu)

실행 스크립트(L40S / CPU)는 `--input-directory`, `--output-directory`, `--temp-directory` 옵션으로
루트의 `input\`, `output\`, `temp\` 폴더를 쓰고, `--extra-model-paths-config`로 루트의 `extra_model_paths.yaml`
(모델 폴더 `model\`)을 읽도록 되어 있습니다. ComfyUI는 기본적으로 `ComfyUI\extra_model_paths.yaml`만 찾으므로 이 옵션이 없으면 `model\`이 보이지 않습니다.
```

## 새 PC에 받기

이미 ComfyUI-Easy-Install이 설치된 폴더에서:

```bat
cd C:\path\to\ComfyUI-Easy-Install
git init
git remote add origin https://github.com/parksihun/ComfyUI.git
git fetch origin
git checkout -f -b main origin/main
Setup_Folders.bat
```

`Setup_Folders.bat`은 `model\`, `input\`, `output\`, `temp\` 폴더를 만들고, `workflow\`와 `custom_nodes\` 아래 폴더들(ShortsRemake, WanChain, 외부 노드 3개)을
`ComfyUI\` 안에 정션(폴더 링크)으로 연결합니다. 유튜브 다운로드 노드를 쓰려면 `Install_Python_Packages.bat`으로 yt-dlp를 추가 설치합니다(인터넷 필요). 이후 ComfyUI를 재시작하면 됩니다.

## 이 PC에서 수정 후 올리기

```bat
git add -A
git commit -m "설명"
git push
```

`ComfyUI\custom_nodes\ComfyUI-ShortsRemake`, `ComfyUI-WanChain`은 정션이므로 그 안에서 고쳐도 `custom_nodes\` 쪽 원본이 바뀌고 그대로 커밋됩니다.
워크플로우를 ComfyUI UI에서 저장하면 `ComfyUI\user\default\workflows\`에 저장되므로, 올리려면 `workflow\`로 복사해야 합니다.

## 필요한 것

- ComfyUI-QwenVL, ComfyUI-VideoHelperSuite (Easy-Install 기본 포함)
- 모델: `workflow/README_Shorts_Remake.md` 참고 (Qwen3-VL-8B, Qwen-Image-Edit-2511 3개 파일, Wan Animate 2 5개 파일)
- GPU 서버(L40S) 기준. GPU 없는 PC는 `Start ComfyUI CPU.bat` + 1단계 워크플로우를 GGUF 노드로 바꿔 확인만 가능
