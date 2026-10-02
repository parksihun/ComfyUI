@Echo off&&cd /D %~dp0
Title ComfyUI-Easy-Install [K620]

set "path=%windir%\System32;%windir%\System32\WindowsPowerShell\v1.0;%PATH%"

set PORT=8188
for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "([regex]::Match((Get-Content '%~f0' -Raw), '--port\s+(\d+)')).Groups[1].Value"') do set PORT=%%A
for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { 1 } else { 0 }"') do set INUSE=%%A
if "%INUSE%"=="1" (
    echo Hey [92m%USERNAME%[0m! ComfyUI is already running on port [92m%PORT%[0m.
    echo [93mPress any key to exit...[0m&&pause>nul&&exit
)

REM  Quadro K620 PC: the card is too old for this torch build, so ComfyUI runs with --cpu.
REM  input / output live on V: on this PC (kept there on purpose; do not point them back at the install folder)
if not exist "V:\input" mkdir "V:\input"
if not exist "V:\output" mkdir "V:\output"
.\python_embeded\python.exe -I -W ignore::FutureWarning ComfyUI\main.py --windows-standalone-build --cpu --input-directory "V:\input" --output-directory "V:\output" --temp-directory "%~dp0." --extra-model-paths-config "%~dp0extra_model_paths.yaml"

echo.
echo [92m:: Press any key to exit ::[0m&Pause>nul

