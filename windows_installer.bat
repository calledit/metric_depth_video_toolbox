echo windows installer for the metric_depth_video_toolbox

set "CONDA=%UserProfile%\miniconda3\condabin\conda.bat"

CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

CALL "%CONDA%"  create -n mdvt python=3.11 -y
CALL "%UserProfile%\miniconda3\Scripts\activate.bat" mdvt


pip install numpy open3d opencv-python glfw PyOpenGL

rem to use the movie_2_3D.py you need scenedetect and PySide6 for GUI
pip install scenedetect[opencv-headless] PySide6



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

rem install Depth-Anything-V2 inside Video-Depth-Anything
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd ..

python download_weights.py vda

rem copy over function file to be able to import vda metric
xcopy "other\metric_dpt_func.py" "Video-Depth-Anything\Depth-Anything-V2\metric_depth\" /K

echo install da3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3

pip install moviepy==1.0.3 addict plyfile pycolmap trimesh evo

echo install of depth estimator done




echo installing infill modell StereoCrafter

git clone --recursive https://github.com/TencentARC/StereoCrafter
pip install transformers diffusers accelerate scipy
python download_weights.py stereocrafter

echo infill model installed

echo installing infill modell StereoProPainter
git clone https://github.com/calledit/StereoProPainter
echo infill model installed

echo installing infill modell m2svid
git clone https://github.com/google-research/m2svid
echo infill model installed

echo installing infill modell inspatio-world
git clone https://github.com/inspatio/inspatio-world
pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2%2Bcu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl

echo infill model installed

pip install OmegaConf kornia open-clip-torch pytorch_lightning pytorch-msssim gdown
python download_weights.py m2svid

echo install masking dependencies
pip install rembg[gpu] filetype watchdog aiohttp asyncer gradio
CALL "%CONDA%"  install -c conda-forge cudnn=9.*

echo install DepthCrafter model
git clone https://github.com/Tencent/DepthCrafter
pip install mediapy decord

echo install Unikd3D model
git clone https://github.com/lpiccinelli-eth/UniK3D

pip install timm wandb


echo install ffmpeg for muxing final result
rem install ffmpeg using winget instead of conda as the conda install breaks PySide6
winget install ffmpeg