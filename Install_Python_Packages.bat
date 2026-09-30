@echo off
cd /d %~dp0
Title ComfyUI-Easy-Install - python packages for the workflows

REM ================================================================
REM  Extra python packages the custom nodes in custom_nodes\ need.
REM  Needs internet. ComfyUI's own requirements (einops, kornia, scipy,
REM  opencv ...) are already part of ComfyUI-Easy-Install.
REM
REM    ComfyUI-ShortsRemake        yt-dlp   (0_YouTube_Download_Trim only)
REM    ComfyUI-Frame-Interpolation requirements-no-cupy.txt (all already present, kept for safety)
REM
REM  Offline server: run this on an online PC, then copy
REM  python_embeded\Lib\site-packages\yt_dlp to the server (only needed
REM  if the YouTube download node is used there).
REM ================================================================

if not exist "python_embeded\python.exe" (
    echo  [91mpython_embeded\python.exe not found.[0m Run this from the ComfyUI-Easy-Install folder.
    pause & exit /b 1
)

echo  [1/2] ComfyUI-ShortsRemake (yt-dlp)
python_embeded\python.exe -m pip install -r "custom_nodes\ComfyUI-ShortsRemakeequirements.txt"

echo  [2/2] ComfyUI-Frame-Interpolation (RIFE VFI)
python_embeded\python.exe -m pip install -r "custom_nodes\ComfyUI-Frame-Interpolationequirements-no-cupy.txt"

echo.
echo  done. Restart ComfyUI if it was running.
pause
