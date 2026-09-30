@echo off
setlocal
cd /D %~dp0
title ComfyUI - Model Download (Shorts Remake: Wan Animate 2 + Qwen-Image-Edit 2511 + Qwen3-VL)

REM ================================================================
REM  Models for the Shorts Remake workflows
REM    workflow\1_Analyze_Prompts / 2_Compose_Reference /
REM    3_Replace_Person_WanAnimate2 / 4_ALL_in_One
REM
REM  - Files that already exist in model\ are skipped.
REM  - Interrupted downloads resume from *.part on the next run.
REM  - Optional: set HF_TOKEN=hf_xxx before running if Hugging Face
REM    rate-limits you.
REM
REM  Everything the Shorts workflows need is fetched here so the
REM  server can run fully OFFLINE afterwards (nothing is downloaded
REM  at run time). Disk needed (only the missing files are fetched):
REM    Wan Animate 2       ~19 GB  (diffusion 17 GB, lora 0.7 GB, clip_vision 1.3 GB, vae 0.25 GB)
REM    Qwen-Image-Edit 2511 ~31 GB (usually already present from Download_Models.bat)
REM    Qwen3-VL-8B-Instruct ~17 GB (stage 1 vision-language model, goes to ComfyUI\models\LLM)
REM ================================================================

set "MODEL=%~dp0model"
set "HF=https://huggingface.co"
set /a OK=0
set /a SKIP=0
set /a FAIL=0

set "AUTH="
if defined HF_TOKEN set AUTH=-H "Authorization: Bearer %HF_TOKEN%"

echo.
echo  ============================================================
echo   Shorts Remake - Model Download
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

REM ---------------------------------------------------------------
echo  [1/3] Wan Animate 2  ^(stage 2, Comfy-Org/Wan-Animate-2^)
echo.
call :DL diffusion_models wan_animate_2_int8_convrot.safetensors ^
    "%HF%/Comfy-Org/Wan-Animate-2/resolve/main/diffusion_models/wan_animate_2_int8_convrot.safetensors"
call :DL loras lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors ^
    "%HF%/Comfy-Org/Wan-Animate-2/resolve/main/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
call :DL text_encoders umt5_xxl_fp8_e4m3fn_scaled.safetensors ^
    "%HF%/Comfy-Org/Wan-Animate-2/resolve/main/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
call :DL clip_vision clip_vision_h.safetensors ^
    "%HF%/Comfy-Org/Wan-Animate-2/resolve/main/clip_vision/clip_vision_h.safetensors"
call :DL vae Wan2_1_VAE_bf16.safetensors ^
    "%HF%/Comfy-Org/Wan-Animate-2/resolve/main/vae/Wan2_1_VAE_bf16.safetensors"

REM ---------------------------------------------------------------
echo.
echo  [2/3] Qwen-Image-Edit 2511  ^(stage 1b, reference image^)
echo.
call :DL diffusion_models qwen_image_edit_2511_fp8mixed.safetensors ^
    "%HF%/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"
call :DL text_encoders qwen_2.5_vl_7b_fp8_scaled.safetensors ^
    "%HF%/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
call :DL vae qwen_image_vae.safetensors ^
    "%HF%/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"
call :DL loras Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors ^
    "%HF%/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

REM ---------------------------------------------------------------
echo.
echo  [3/3] Qwen3-VL-8B-Instruct  ^(stage 1, QwenVL node; ComfyUI\models\LLM\Qwen\Qwen3-VL-8B-Instruct^)
echo.
set "LLM_DIR=%~dp0ComfyUI\models\LLM\Qwen\Qwen3-VL-8B-Instruct"
REM snapshot_download checks every file: already complete files are skipped, partial ones resume.
echo    get    LLM\Qwen\Qwen3-VL-8B-Instruct  ^(huggingface_hub snapshot: verifies / resumes, ~17 GB^)
set HF_HUB_OFFLINE=
set TRANSFORMERS_OFFLINE=
"%~dp0python_embeded\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-VL-8B-Instruct', local_dir=r'%LLM_DIR%', ignore_patterns=['*.md', '.git*'], max_workers=4)"
if errorlevel 1 (
    echo    [91mFAILED[0m Qwen3-VL-8B-Instruct
    set /a FAIL+=1
) else (
    echo    ok     LLM\Qwen\Qwen3-VL-8B-Instruct complete
    set /a OK+=1
)

:SUMMARY
echo.
echo  ============================================================
echo   Downloaded: %OK%    Skipped: %SKIP%    Failed: %FAIL%
if %FAIL% GTR 0 (
    echo.
    echo   [93mRe-run this file to resume failed downloads.[0m
)
echo.
echo   Restart ComfyUI so the new files show up in the dropdowns.
echo   Offline server: copy model\ and ComfyUI\models\LLM\ over, then
echo   start with 5_Start_ComfyUI_L40S.bat as usual.
echo  ============================================================
echo.
pause
exit /b 0

REM ---------------------------------------------------------------
REM  :DL  <subfolder>  <filename>  <url>
REM ---------------------------------------------------------------
:DL
set "DEST=%MODEL%\%~1\%~2"
if exist "%DEST%" (
    echo    skip   %~1\%~2
    set /a SKIP+=1
    goto :eof
)
if not exist "%MODEL%\%~1\" mkdir "%MODEL%\%~1"
echo    get    %~1\%~2
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
