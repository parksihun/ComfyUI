@echo off
cd /D "%~dp0..\.."
title Video WebApp (image - scenario - SVI video)

REM ================================================================
REM  Web program that drives ComfyUI through its API:
REM    1. z-image turbo image  2. Qwen scenario (6 prompts)
REM    3. Wan 2.2 SVI video
REM
REM  Start ComfyUI first (Start_ComfyUI_L40S.bat), then run this.
REM  Page:    http://127.0.0.1:8288   (other PCs: http://<server ip>:8288)
REM  Options: --port 8288  --comfy http://127.0.0.1:8188  --host 127.0.0.1
REM           (defaults are in tools\video_webapp\config.json)
REM  No extra python packages are needed.
REM ================================================================

if not exist "python_embeded\python.exe" (
    echo  python_embeded\python.exe not found. This file must stay in
    echo  ComfyUI-Easy-Install\tools\video_webapp\
    pause
    exit /b 1
)

.\python_embeded\python.exe -I tools\video_webapp\app.py --open %*

echo.
pause
