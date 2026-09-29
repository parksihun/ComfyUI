@echo off
setlocal
cd /D %~dp0
title ComfyUI - Model Download (Wan 2.2 I2V SVI - Kenpechi v3.5)

REM ================================================================
REM  Models for workflow\Wan2.2_I2V_SVI_Workflow_Kenpechi_v3.5.json
REM
REM  - Files that already exist are skipped; interrupted downloads
REM    resume from *.part on the next run.
REM  - Optional: set HF_TOKEN=hf_xxx if Hugging Face rate-limits you.
REM
REM  Disk needed (only missing files are fetched):
REM    base Wan 2.2 I2V 14B fp8 set        ~36.8 GB  (usually already present)
REM    SVI v2 PRO LoRA HIGH + LOW           2.5 GB   loras\HIGH, loras\LOW
REM    lightx2v 4-step LoRA HIGH + LOW      1.4 GB   loras\HIGH, loras\LOW
REM    RealESRGAN_x2plus (upscale, optional group)  0.07 GB
REM    rife49.pth (RIFE VFI frame interpolation)    0.02 GB
REM    GGUF Q8 models (optional, asked)     30.8 GB
REM
REM  NOT downloaded here: the Power Lora Loader entries (DR34ML4Y, NSFW-22,
REM  BreastRub, Body-Cumshot, Undressing, ...) are Civitai LoRAs that need a
REM  login. Get them yourself into loras\HIGH / loras\LOW, or switch them off
REM  in the Power Lora Loader nodes.
REM
REM  Custom nodes this workflow needs (install with ComfyUI-Manager):
REM    ComfyUI-Wan22FMLF (Wan Advanced I2V), ComfyUI-Frame-Interpolation (RIFE VFI),
REM    ComfyUI-Custom-Scripts (pysssss Math Expression), plus rgthree, KJNodes,
REM    Easy-Use, WanVideoWrapper, GGUF, VideoHelperSuite (already in Easy-Install).
REM ================================================================

set "MODEL=%~dp0model"
set "RIFE_DIR=%~dp0ComfyUI\custom_nodes\ComfyUI-Frame-Interpolation\ckpts\rife"
set "HF=https://huggingface.co"
set /a OK=0
set /a SKIP=0
set /a FAIL=0

set "AUTH="
if defined HF_TOKEN set AUTH=-H "Authorization: Bearer %HF_TOKEN%"

echo.
echo  ============================================================
echo   Wan 2.2 I2V SVI (Kenpechi v3.5) - Model Download
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
    echo  [91mmodel\ folder not found.[0m Run Setup_Folders.bat first.
    pause
    exit /b 1
)

choice /c YN /n /m "  Also download the GGUF Q8 models (2 x 15.4 GB, only for the GGUF group)? [Y/N] "
set "GGUF=%errorlevel%"
echo.

echo  [1/5] Wan 2.2 I2V 14B base set (Comfy-Org repackaged, fp8)
echo.
call :DL diffusion_models wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors ^
    "%HF%/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
call :DL diffusion_models wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors ^
    "%HF%/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
call :DL text_encoders umt5_xxl_fp8_e4m3fn_scaled.safetensors ^
    "%HF%/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
call :DL vae wan_2.1_vae.safetensors ^
    "%HF%/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
call :DL clip_vision clip_vision_h.safetensors ^
    "%HF%/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

echo.
echo  [2/5] SVI v2 PRO LoRAs (Stable Video Infinity, Kijai/WanVideo_comfy)
echo.
call :DL loras\HIGH SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors ^
    "%HF%/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
call :DL loras\LOW SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors ^
    "%HF%/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"

echo.
echo  [3/5] lightx2v 4-step LoRAs (HIGH v1030 / LOW 1022)
echo.
call :DL loras\HIGH Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors ^
    "%HF%/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors"
call :DL loras\LOW Wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors ^
    "%HF%/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"

echo.
echo  [4/5] Upscale + frame interpolation models
echo.
call :DL upscale_models RealESRGAN_x2plus.pth ^
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
if exist "%~dp0ComfyUI\custom_nodes\ComfyUI-Frame-Interpolation\" (
    call :DLTO "%RIFE_DIR%" rife49.pth "%HF%/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth"
) else (
    echo    [93mskip[0m rife49.pth - ComfyUI-Frame-Interpolation is not installed yet ^(install it with Manager, then re-run^)
)

echo.
if not "%GGUF%"=="1" goto :SUMMARY
echo  [5/5] GGUF Q8 models (QuantStack) - optional GGUF group
echo.
call :DL diffusion_models Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf ^
    "%HF%/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf"
call :DL diffusion_models Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf ^
    "%HF%/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"

:SUMMARY
echo.
echo  ============================================================
echo   Downloaded: %OK%    Skipped: %SKIP%    Failed: %FAIL%
if %FAIL% GTR 0 (
    echo.
    echo   [93mRe-run this file to resume failed downloads.[0m
)
echo.
echo   Power Lora Loader entries (Civitai, manual): see the REM block at the top.
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
