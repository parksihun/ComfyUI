@Echo off&&cd /D %~dp0
Title ComfyUI-Easy-Install [RTX 4060]

set CUDA_PATH=
set CUDA_HOME=
set CUDA_BIN_PATH=
set CUDNN_PATH=
set CUDNN_HOME=
set PYTHONPATH=
set PYTHONHOME=
set VIRTUAL_ENV=
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
for /f "tokens=1* delims==" %%a in ('set CUDA_PATH_V 2^>nul') do set "%%a="

set "path=%windir%\System32;%windir%\System32\WindowsPowerShell\v1.0;%PATH%"

set PORT=8188
for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "([regex]::Match((Get-Content '%~f0' -Raw), '--port\s+(\d+)')).Groups[1].Value"') do set PORT=%%A
for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { 1 } else { 0 }"') do set INUSE=%%A
if "%INUSE%"=="1" (
    echo Hey [92m%USERNAME%[0m ComfyUI is already running on port [92m%PORT%[0m.
    echo [93mPress any key to exit...[0m&&pause>nul&&exit
)

REM  Home PC (RTX 4060 8GB). Same as "Start ComfyUI SageAttention.bat" plus the root folders of this repo:
REM  input\ output\ temp\ and model\ (extra_model_paths.yaml). ComfyUI appends \temp to --temp-directory, so the root (.) is passed.
REM  Local access only (no --listen); VRAM mode is left to ComfyUI's automatic choice.
.\python_embeded\python.exe -I -W ignore::FutureWarning ComfyUI\main.py --windows-standalone-build --use-sage-attention --input-directory "%~dp0input" --output-directory "%~dp0output" --temp-directory "%~dp0." --extra-model-paths-config "%~dp0extra_model_paths.yaml"
pause
