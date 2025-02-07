@echo off

echo This script was successfully tested
echo on Windows 11 with Python 3.10.6
pause
python -c "import os; os.system('wget "https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2"')
python -c "import bz2, shutil; shutil.copyfileobj(bz2.BZ2File('openh264-1.8.0-win64.dll.bz2'), open('openh264-1.8.0-win64.dll', 'wb'))"

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install xformers
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl
pip install open3d

git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
pip install -r requirements.txt
mkdir checkpoints
cd checkpoints
python -c "import os; os.system('wget "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"')

cd..
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
cd metric_depth
mkdir checkpoints
cd checkpoints
python -c "import os; os.system('wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth"')

cd ..
cd ..
cd ..
cd ..

xcopy "src\metric_dpt_func.py" "Video-Depth-Anything\Depth-Anything-V2\metric_depth" /K
xcopy "src\video_metric_convert.py" "Video-Depth-Anything" /K
