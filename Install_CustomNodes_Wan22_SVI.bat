@echo off
setlocal
cd /D %~dp0
title ComfyUI - Custom nodes for Wan2.2_I2V_SVI_Workflow_Kenpechi_v3.5

REM ================================================================
REM  Custom nodes the Kenpechi Wan 2.2 I2V SVI workflow needs that are
REM  not part of ComfyUI-Easy-Install. Needs internet + git.
REM  Offline server: run this on an online PC, then copy the three
REM  folders under ComfyUI\custom_nodes\ to the server.
REM
REM    ComfyUI-Wan22FMLF           Wan Advanced I2V (Ultimate), SVI long video
REM    ComfyUI-Frame-Interpolation RIFE VFI (+ ckpts\rife\rife49.pth)
REM    ComfyUI-Custom-Scripts      Math Expression (pysssss)
REM  Already in Easy-Install: rgthree, KJNodes (ImageBatchExtendWithOverlap),
REM  Easy-Use, WanVideoWrapper (NAG), GGUF, VideoHelperSuite.
REM ================================================================

set "CN=%~dp0ComfyUI\custom_nodes"
set "PY=%~dp0python_embeded\python.exe"

where git >nul 2>&1
if errorlevel 1 (
    echo  [91mgit not found.[0m Install Git for Windows first.
    pause
    exit /b 1
)

call :CLONE ComfyUI-Wan22FMLF           https://github.com/wallen0322/ComfyUI-Wan22FMLF.git
call :CLONE ComfyUI-Frame-Interpolation https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
call :CLONE ComfyUI-Custom-Scripts      https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git

echo.
echo  python packages for ComfyUI-Frame-Interpolation (no cupy)
"%PY%" -m pip install -q -r "%CN%\ComfyUI-Frame-Interpolation\requirements-no-cupy.txt"

echo.
echo  rife49.pth for RIFE VFI
set "RIFE=%CN%\ComfyUI-Frame-Interpolation\ckpts\rife"
if not exist "%RIFE%\" mkdir "%RIFE%"
if exist "%RIFE%\rife49.pth" (
    echo    skip   rife49.pth
) else (
    curl -L --fail --retry 5 --retry-delay 5 -o "%RIFE%\rife49.pth" "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth"
)

echo.
echo  done. Restart ComfyUI. Then run Download_Models_Wan22_SVI.bat for the models.
pause
exit /b 0

:CLONE
if exist "%CN%\%~1\" (
    echo    skip   %~1 ^(already installed^)
) else (
    echo    clone  %~1
    git clone --depth 1 %~2 "%CN%\%~1"
)
goto :eof
