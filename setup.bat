@echo off

set PYTHON_VER=3.10.11

:: Check if Python version meets the recommended version
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)

IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

:: Activate the virtual environment
call .\venv\Scripts\activate.bat

echo Installing Torch and torchvision...
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

echo Cloning and installing bitsandbytes-windows...
git clone https://github.com/Keith-Hon/bitsandbytes-windows.git
cd bitsandbytes-windows
pip3 install -e .
cd ..

echo Downloading pretrained models...
curl -L -o ./checkpoint.pth https://huggingface.co/ckpt/minigpt4/resolve/main/minigpt4.pth
curl -L -o ./blip2_pretrained_flant5xxl.pth https://huggingface.co/ckpt/minigpt4/resolve/main/blip2_pretrained_flant5xxl.pth

echo Installing cmake, lit, salesforce-lavis, accelerate, and transformers...
pip install cmake
pip install lit
pip install -q salesforce-lavis
pip install -q accelerate
pip install -q git+https://github.com/huggingface/transformers.git -U

echo Setup complete within virtual environment!