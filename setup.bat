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
curl -L -o ./checkpoint.pth https://huggingface.co/ckpt/minigpt4/resolve/main/prerained_minigpt4.pth
curl -L -o ./blip2_pretrained_flant5xxl.pth https://huggingface.co/ckpt/minigpt4/resolve/main/blip2_pretrained_flant5xxl.pth
curl -L -o ./models.zip https://huggingface.co/pipyp/minigpt4py/resolve/main/models.zip
python extract.py

echo Installing cmake, lit, salesforce-lavis, accelerate, and transformers...
pip install cmake
pip install lit
pip install -q salesforce-lavis
pip install -q accelerate
pip install -q git+https://github.com/huggingface/transformers.git -U

:: Adding the extra required libraries...
echo Installing argparse, csv, os, random, glob, time, numpy, Pillow, cv2, tqdm, tensorflow, huggingface-hub, pathlib, copy, and keras...
pip install argparse
pip install csv
pip install os
pip install random
pip install glob
pip install time
pip install numpy
pip install Pillow
pip install opencv-python-headless
pip install tqdm
pip install tensorflow
pip install huggingface-hub
pip install pathlib
pip install copy
pip install keras

echo Setup complete within virtual environment!