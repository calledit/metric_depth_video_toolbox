echo windows installer for the metric_depth_video_toolbox

set "CONDA=%UserProfile%\miniconda3\condabin\conda.bat"
CALL "%CONDA%"  create -n mdvt python=3.11 -y
CALL "%UserProfile%\miniconda3\Scripts\activate.bat" mdvt
pip install numpy open3d opencv-python

CALL "%CONDA%" install -c conda-forge ffmpeg

echo install ML models and dependencies
echo install torch with cuda support


rem cuda version here is 12.4 but it would not matter
CALL "%CONDA%" install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
@echo on
echo install dependecy requirements compatible with cuda 12.4 and torch 2.5.1
pip install xformers==0.0.29.post1 --index-url https://download.pytorch.org/whl/cu124
pip install tqdm einops imageio easydict matplotlib triton-windows==3.2.0.post19
rem Video-Depth-Anything
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
git checkout 3628f50d55e81183c7cc7025f2c22190fa37ef28
mkdir checkpoints
cd checkpoints
curl -O -L https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
cd ..

rem install Depth-Anything-V2
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2

cd metric_depth
mkdir checkpoints
cd checkpoints
curl -O -L https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
cd ..

cd ..
cd ..
cd ..

rem copy over function file to be able to import vda metric
xcopy "other\metric_dpt_func.py" "Video-Depth-Anything\Depth-Anything-V2\metric_depth\" /K

echo install of depth estimator done




echo installing infill modell StereoCrafter

rem this askes for credentials which are not techinically required do it manually instead of using --recursive
git clone --recursive https://github.com/TencentARC/StereoCrafter
cd StereoCrafter

pip install transformers diffusers accelerate scipy

# in StereoCrafter project root directory
mkdir weights
cd ./weights

git clone https://huggingface.co/TencentARC/StereoCrafter

cd .. 
cd ..
echo infill model installed

echo install masking dependencies
pip install rembg[gpu] filetype watchdog aiohttp asyncer gradio
CALL "%CONDA%"  install -c conda-forge cudnn=9.*