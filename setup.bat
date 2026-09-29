@echo off
cd /d %~dp0
Title ComfyUI-Easy-Install - repo setup

echo [1/3] custom_nodes\* -> ComfyUI\custom_nodes (junctions)
if not exist "ComfyUI\custom_nodes" (
    echo   ComfyUI\custom_nodes not found. Run this from the ComfyUI-Easy-Install folder.
    pause & exit /b 1
)
for /d %%D in ("custom_nodes\*") do (
    if exist "ComfyUI\custom_nodes\%%~nxD" (
        echo   %%~nxD already present, skipping
    ) else (
        mklink /J "ComfyUI\custom_nodes\%%~nxD" "%%D"
    )
)

echo [2/3] python packages for ComfyUI-ShortsRemake (yt-dlp)
python_embeded\python.exe -m pip install -q -r "custom_nodes\ComfyUI-ShortsRemake\requirements.txt"

echo [3/3] workflow\*.json -> ComfyUI\user\default\workflows (copy, shows up in the ComfyUI sidebar)
if not exist "ComfyUI\user\default\workflows" mkdir "ComfyUI\user\default\workflows"
for %%F in (workflow\*.json) do (
    echo %%~nxF | findstr /i "\.api\.json" >nul || copy /Y "%%F" "ComfyUI\user\default\workflows\" >nul
)

echo.
echo done. Restart ComfyUI to load the nodes.
pause
