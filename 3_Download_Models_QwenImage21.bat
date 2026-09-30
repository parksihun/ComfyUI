@echo off
setlocal
cd /D %~dp0
title ComfyUI - Model Download (Qwen-Image 2.1 image edit)

REM ================================================================
REM  Models for workflow\image_qwen_image_2_1_image_edit.json
REM  (ComfyUI official template "Image Edit (Qwen Image 2.1)", int8 variants)
REM
REM  - Files that already exist in model\ are skipped.
REM  - Interrupted downloads resume from *.part on the next run.
REM  - Optional: set HF_TOKEN=hf_xxx before running if Hugging Face
REM    rate-limits you.
REM
REM  Disk needed: about 17.3 GB
REM    diffusion_models\qwen_image_2.1_int8_convrot.safetensors   7.3 GB
REM    text_encoders\qwen3vl_8b_int8_convrot.safetensors          9.4 GB
REM    vae\qwen_image_2.1_vae_bf16.safetensors                     0.7 GB
REM  + two sample images (input\) used by the template, a few hundred KB
REM
REM  NOTE: the workflow needs the QwenImage21Cache / TextEncodeQwenImage21
REM  nodes, which exist from ComfyUI 0.37. A 0.36 server shows them as
REM  missing nodes until ComfyUI is updated.
REM ================================================================

set "MODEL=%~dp0model"
set "INPUT=%~dp0input"
set "HF=https://huggingface.co"
set /a OK=0
set /a SKIP=0
set /a FAIL=0

set "AUTH="
if defined HF_TOKEN set AUTH=-H "Authorization: Bearer %HF_TOKEN%"

echo.
echo  ============================================================
echo   Qwen-Image 2.1 - Model Download
echo   Target: %MODEL%
echo  ============================================================
echo.

where curl >nul 2>&1
if errorlevel 1 (
    echo  [91mcurl not found.[0m Windows 10 1803+ includes it in System32.
    pause
    exit /b 1
)

if not exist "%MODEL%\" (
    echo  [91mmodel\ folder not found.[0m Run 1_Setup.bat first.
    pause
    exit /b 1
)

echo  [1/2] Qwen-Image 2.1 (Comfy-Org/Qwen-Image-2.1, int8)
echo.
call :DL diffusion_models qwen_image_2.1_int8_convrot.safetensors ^
    "%HF%/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
call :DL text_encoders qwen3vl_8b_int8_convrot.safetensors ^
    "%HF%/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
call :DL vae qwen_image_2.1_vae_bf16.safetensors ^
    "%HF%/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"

echo.
echo  [2/2] sample input images for the template (input\)
echo.
if not exist "%INPUT%\" mkdir "%INPUT%"
call :DLTO "%INPUT%" portrait_model_denim.png ^
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/portrait_model_denim.png"
call :DLTO "%INPUT%" clothing_light_blue_denim_shirt.png ^
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/clothing_light_blue_denim_shirt.png"

echo.
echo  ============================================================
echo   Downloaded: %OK%    Skipped: %SKIP%    Failed: %FAIL%
if %FAIL% GTR 0 (
    echo.
    echo   [93mRe-run this file to resume failed downloads.[0m
)
echo.
echo   Restart ComfyUI so the new files show up in the dropdowns.
echo  ============================================================
echo.
pause
exit /b 0

REM ---------------------------------------------------------------
REM  :DL  <model subfolder>  <filename>  <url>
REM ---------------------------------------------------------------
:DL
call :DLTO "%MODEL%\%~1" "%~2" "%~3"
goto :eof

REM ---------------------------------------------------------------
REM  :DLTO  <folder>  <filename>  <url>
REM ---------------------------------------------------------------
:DLTO
set "DEST=%~1\%~2"
if exist "%DEST%" (
    echo    skip   %~2
    set /a SKIP+=1
    goto :eof
)
if not exist "%~1\" mkdir "%~1"
echo    get    %~2
curl -L --fail --retry 5 --retry-delay 5 -C - %AUTH% -o "%DEST%.part" "%~3"
if errorlevel 1 (
    echo    [91mFAILED[0m %~2
    set /a FAIL+=1
    goto :eof
)
move /y "%DEST%.part" "%DEST%" >nul
set /a OK+=1
echo.
goto :eof
