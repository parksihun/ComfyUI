@echo off
cd /D "%~dp0..\.."
title ComfyUI-Manager (closes with the page in the browser)

REM ================================================================
REM  Web program for making and keeping track of images / videos.
REM    Create : z-image turbo image - Qwen scenario (6 prompts) -
REM             Wan 2.2 video, queued on ComfyUI through its API
REM    Library: every file in output\ with the workflow and the
REM             prompts it was made with
REM    Models : the model folder, copy to / from other folders
REM
REM  Create needs ComfyUI running (Start_ComfyUI_L40S.bat);
REM  the library works without it.
REM  Page:    http://127.0.0.1:8288   (other PCs: http://<server ip>:8288)
REM  Options: --port 8288  --comfy http://127.0.0.1:8188  --host 127.0.0.1
REM           (defaults are in tools\ComfyUI-Manager\config.json)
REM  No extra python packages are needed.
REM
REM  Opens the page in the browser and shows the log here. When the
REM  page has been closed in the browser the program stops by itself
REM  and this window closes with it ("close_with_page" in config.json;
REM  set it to false for a server that other PCs use). The log is also
REM  kept in tools\ComfyUI-Manager\data\manager.log.
REM ================================================================

if not exist "python_embeded\python.exe" (
    echo  python_embeded\python.exe not found. This file must stay in
    echo  ComfyUI-Easy-Install\tools\ComfyUI-Manager\
    pause
    exit /b 1
)

.\python_embeded\python.exe -I tools\ComfyUI-Manager\app.py --open %*

REM only a start that failed (port taken, error) keeps the window open to read
if errorlevel 1 (
    echo.
    pause
)
