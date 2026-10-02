@Echo off&&cd /D %~dp0
Title ComfyUI-Easy-Install [L40S]

set "path=%windir%\System32;%windir%\System32\WindowsPowerShell\v1.0;%PATH%"

REM ============================================================
REM  L40S (48GB / TCC mode) settings
REM ------------------------------------------------------------
REM  --disable-cuda-malloc : TCC 모드에서 cudaMallocAsync 미지원 (필수)
REM  --use-sage-attention  : SageAttention 2.2.0 활성
REM  (--highvram 은 아래 주석 참고, 제거됨)
REM  --bf16-vae            : Ada 네이티브 bf16, VAE 검은 화면 방지
REM  --listen 0.0.0.0      : 외부 PC 브라우저 접속 허용
REM  --disable-auto-launch : 헤드리스 - 브라우저 자동 실행 안 함
REM ============================================================

REM  --highvram 제거 (2026-09-29): Wan 2.2 14B high/low 두 모델 + Qwen3-VL 을 번갈아 쓰는 Shorts 파이프라인에서는
REM  모델을 전부 VRAM 에 상주시키면 46GB 를 넘겨 "Not enough GPU memory" 가 남. 기본(normal) 모드는 안 쓰는 모델을 RAM 으로 내림.
set "COMFY_ARGS=--windows-standalone-build --use-sage-attention --disable-cuda-malloc --bf16-vae --preview-method auto --listen 0.0.0.0 --port 8188 --disable-auto-launch"

REM  input / output / temp live on E:\ComfyUI on this server (kept there on purpose; do not point them back at the install folder)
REM  --temp-directory : ComfyUI appends \temp itself, so E:\ComfyUI is passed and the files go to E:\ComfyUI\temp
REM  --extra-model-paths-config : 루트의 extra_model_paths.yaml (model\ 폴더) 을 읽음. ComfyUI 는 기본적으로 ComfyUI\extra_model_paths.yaml 만 찾음
if not exist "E:\ComfyUI\input" mkdir "E:\ComfyUI\input"
if not exist "E:\ComfyUI\output" mkdir "E:\ComfyUI\output"
set "COMFY_ARGS=%COMFY_ARGS% --input-directory "E:\ComfyUI\input" --output-directory "E:\ComfyUI\output" --temp-directory "E:\ComfyUI" --extra-model-paths-config "%~dp0extra_model_paths.yaml""

REM ============================================================
REM  오프라인 서버: huggingface_hub / transformers 가 인터넷을 시도하지 않게 함
REM  (QwenVL 노드가 models\LLM\Qwen\Qwen3-VL-8B-Instruct 를 바로 로컬에서 읽음.
REM   켜져 있으면 연결 시간 초과를 기다리느라 첫 로딩이 몇 분씩 멈춤)
REM  인터넷이 되는 환경에서 모델을 새로 받아야 할 때는 이 네 줄을 REM 처리
REM ============================================================
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set DO_NOT_TRACK=1

REM --- 선택 옵션 (필요할 때 위 줄 끝에 붙이세요) ---------------
REM  --fast fp16_accumulation   : 속도 향상, 극히 드물게 품질 저하
REM  --reserve-vram 2.0         : 다른 프로세스와 GPU 공유 시
REM  --front-end-version Comfy-Org/ComfyUI_frontend@latest
REM ------------------------------------------------------------

for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "([regex]::Match((Get-Content '%~f0' -Raw), '--port\s+(\d+)')).Groups[1].Value"') do set PORT=%%A
if "%PORT%"=="" set PORT=8188

for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { 1 } else { 0 }"') do set INUSE=%%A
if "%INUSE%"=="1" (
    echo Hey [92m%USERNAME%[0m ComfyUI is already running on port [92m%PORT%[0m.
    echo [93mPress any key to exit...[0m&&pause>nul&&exit
)

REM ============================================================
REM  Windows 방화벽 인바운드 규칙 (외부 PC 접속용)
REM  - 규칙이 없으면 자동 추가, 관리자 권한이 아니면 UAC 승인 창이 뜸
REM  - 최초 1회만 필요, 이후에는 건너뜀
REM ============================================================
netsh advfirewall firewall show rule name="ComfyUI-%PORT%" >nul 2>&1
if not errorlevel 1 goto :fw_ok

echo  [93mAdding Windows Firewall rule for port %PORT%, admin approval required...[0m
net session >nul 2>&1
if not errorlevel 1 (
    netsh advfirewall firewall add rule name="ComfyUI-%PORT%" dir=in action=allow protocol=TCP localport=%PORT% profile=any >nul
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process netsh -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList 'advfirewall firewall add rule name=ComfyUI-%PORT% dir=in action=allow protocol=TCP localport=%PORT% profile=any'"
)

netsh advfirewall firewall show rule name="ComfyUI-%PORT%" >nul 2>&1
if errorlevel 1 (
    echo  [91mFirewall rule was NOT added - external access may be blocked.[0m
    echo  [91mRun this file once with "Run as administrator".[0m
) else (
    echo  [92mFirewall rule ComfyUI-%PORT% added.[0m
)
:fw_ok

echo.
echo  [92mComfyUI[0m starting on port [92m%PORT%[0m  ^(NVIDIA L40S^)
echo  This VM   : [96mhttp://localhost:%PORT%[0m
for /f %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-NetIPAddress -AddressFamily IPv4).IPAddress.Where({$_ -notlike '127.*' -and $_ -notlike '169.254.*'})"') do echo  Other PC  : [96mhttp://%%I:%PORT%[0m
echo.

.\python_embeded\python.exe -I -W ignore::FutureWarning ComfyUI\main.py %COMFY_ARGS%
pause
