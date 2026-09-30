@echo off
cd /d %~dp0
Title ComfyUI-Easy-Install - setup

echo.
echo  ============================================================
echo   ComfyUI-Easy-Install setup
echo   %~dp0
echo  ============================================================
echo.
if not exist "ComfyUI\" (
    echo  [91mComfyUI\ folder not found.[0m Run this from the ComfyUI-Easy-Install folder.
    pause & exit /b 1
)

echo  [1/4] folders: model\ (+ subfolders), input\, output\, workflow\, temp\
for %%D in (model model\checkpoints model\diffusion_models model\unet model\vae model\loras model\loras\HIGH model\loras\LOW ^
            model\clip model\clip_vision model\text_encoders model\controlnet model\embeddings model\upscale_models ^
            model\style_models model\gligen model\hypernetworks model\photomaker model\configs model\latent_upscale_models ^
            input output workflow temp) do (
    if not exist "%%D" mkdir "%%D"
)
if not exist "extra_model_paths.yaml" (
    echo        [93mextra_model_paths.yaml is missing[0m - the launchers pass it with --extra-model-paths-config,
    echo        without it ComfyUI does not see the model\ folder.
)

echo  [2/4] workflow\ -^> ComfyUI\user\default\workflows (junction, sidebar shows the files directly)
set "WF_OLD=%~dp0ComfyUI\user\default\workflows"
set "WF_NEW=%~dp0workflow"
dir /a:l "%~dp0ComfyUI\user\default" 2>nul | find /i "workflows" >nul
if not errorlevel 1 (
    echo        already linked
) else (
    if exist "%WF_OLD%\" (
        echo        moving existing workflows into workflow\ ...
        robocopy "%WF_OLD%" "%WF_NEW%" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS >nul
        if exist "%WF_OLD%\" rd /s /q "%WF_OLD%"
    )
    if not exist "%~dp0ComfyUI\user\default" mkdir "%~dp0ComfyUI\user\default"
    mklink /J "%WF_OLD%" "%WF_NEW%" >nul && echo        linked || echo        [91mlink failed[0m - run: mklink /J "%WF_OLD%" "%WF_NEW%"
)

echo  [3/4] custom_nodes\* -^> ComfyUI\custom_nodes (junctions)
if not exist "ComfyUI\custom_nodes" mkdir "ComfyUI\custom_nodes"
for /d %%D in ("custom_nodes\*") do (
    if exist "ComfyUI\custom_nodes\%%~nxD" (
        echo        %%~nxD already present
    ) else (
        mklink /J "ComfyUI\custom_nodes\%%~nxD" "%%D" >nul && echo        linked %%~nxD
    )
)

echo  [4/4] done
echo.
echo  ============================================================
echo   ComfyUI will use (via the launcher options):
echo     model      %~dp0model\
echo     input      %~dp0input\
echo     output     %~dp0output\
echo     workflow   %~dp0workflow\
echo     temp       %~dp0temp\
echo.
echo   Next: Install_Python_Packages.bat (yt-dlp for the YouTube node; online only),
echo        put the models in model\ (see workflow\README_Shorts_Remake.md), then start ComfyUI
echo   (Start_ComfyUI_L40S.bat on the server, Start ComfyUI CPU.bat here).
echo   Restart ComfyUI if it was running so the nodes load.
echo  ============================================================
echo.
pause
