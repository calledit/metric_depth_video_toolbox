#!/bin/bash



DOWNLOAD='curl'
DOWNLOAD_ARG='-LO'
PIP='pip3.11'
if [ "$(uname)" == "Linux" ]; then
    DOWNLOAD='wget'
	DOWNLOAD_ARG=''
    PIP='pip'
fi


$PIP install open3d numpy opencv-python

#Stuff required for Depth-Anything and other third party tools
$PIP install einops easydict imageio xformers==0.0.29 tqdm

#install support for madpose for accurate camera tracking
if [[ " $@ " =~ " -madpose " ]]; then
	if [ "$(uname)" == "Linux" ]; then
		sudo apt-get install -y libeigen3-dev libceres-dev libopencv-dev
	else
		brew install cmake ninja opencv
	fi
	
	git clone https://github.com/MarkYu98/madpose

	cd madpose/ext/
	
	git clone --recursive https://github.com/pybind/pybind11
	git clone --recursive https://github.com/tsattler/RansacLib
	
	cd ..
	
	$PIP install .
	
	cd ..
	
	exit
	
fi


#install support for stereocrafter for ML based infill
if [[ " $@ " =~ " -stereocrafter " ]]; then
	git clone --recursive https://github.com/TencentARC/StereoCrafter

	cd StereoCrafter
	
	pip install transformers diffusers accelerate

	# in StereoCrafter project root directory
	mkdir weights
	cd ./weights
	
	sudo apt-get install -y git-lfs
	git clone https://huggingface.co/TencentARC/StereoCrafter
	
	#git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1 #Not needed, is automaticly downloaded 
	
	
	
	exit
	
fi

if [[ " $@ " =~ " -depthpro " ]]; then
	git clone https://github.com/apple/ml-depth-pro

	cd ml-depth-pro
	
	pip install -e .
	
	source get_pretrained_models.sh
	
	cd ..
	
	ln -s ml-depth-pro/checkpoints checkpoints
	
	exit
	
fi


#install support for depthcrafter for defusion based depth
if [[ " $@ " =~ " -depthcrafter " ]]; then
	git clone https://github.com/Tencent/DepthCrafter

	cd DepthCrafter
	
	#apt install ffmpeg
	
	pip install diffusers transformers mediapy decord accelerate
	
	
	exit
	
fi


if [[ " $@ " =~ " -geometrycrafter " ]]; then
	
	git clone https://github.com/calledit/GeometryCrafter
	#git clone https://github.com/TencentARC/GeometryCrafter

	cd GeometryCrafter
	
	git checkout reduce-cuda-memory-use
	
	
	$PIP install decord diffusers transformers accelerate kornia
	
	$PIP install scipy 
	
	
	exit
	
fi



#install support for Moge
if [[ " $@ " =~ " -moge " ]]; then
	git clone https://github.com/microsoft/MoGe

	cd MoGe
	git checkout dd158c05461f2353287a182afb2adf0fda46436f
	$PIP install -r requirements.txt

	cd ..
	
	exit
fi


#install support for Mega-sam
if [[ " $@ " =~ " -megasam " ]]; then
	git clone --recursive https://github.com/mega-sam/mega-sam

	cd mega-sam
	
	$PIP install torch_scatter
	
	#install mega-sam specific version of Droid-slam
	cd base
	python setup.py install
	cd ..
	
	
	cd ..
	
	exit
fi


#install support for unidepth
if [[ " $@ " =~ " -unik3d " ]]; then
	git clone https://github.com/lpiccinelli-eth/UniK3D

	cd UniK3D
	
	$PIP install timm wandb
	
	#You get warning if you dont install this but it is not needed
 	#cd unidepth/ops/knn;bash compile.sh;cd ../../../
	
	cd ..
	
	exit
fi


#install support for unidepth
if [[ " $@ " =~ " -unidepth " ]]; then
	git clone https://github.com/lpiccinelli-eth/UniDepth

	cd UniDepth
	
	$PIP install timm wandb
	$DOWNLOAD $DOWNLOAD_ARG https://raw.githubusercontent.com/AbdBarho/xformers-wheels/refs/heads/main/xformers/components/attention/nystrom.py
	sed -i 's/from xformers\.components\.attention import NystromAttention/from nystrom import NystromAttention/g' unidepth/layers/nystrom_attention.py

 	cd unidepth/ops/knn;bash compile.sh;cd ../../../
	
	cd ..
	
	exit
fi

MODEL_URL='https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth'
if [[ " $@ " =~ " -small " ]]; then
	MODEL_URL='https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth'
fi

#install Video-Depth-Anything metric
if [[ " $@ " =~ " -videometricany " ]]; then
	git clone https://github.com/DepthAnything/Video-Depth-Anything
	cd Video-Depth-Anything
	git checkout 3628f50d55e81183c7cc7025f2c22190fa37ef28
	#pip install -r requirements.txt
	cd metric_depth
	mkdir checkpoints
	cd checkpoints
	$DOWNLOAD $DOWNLOAD_ARG https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_${MODEL_SIZE}.pth
	cd ..
	cd ..
	cd ..
	
	exit
fi

if [[ " $@ " =~ " -promptda " ]]; then
	git clone https://github.com/DepthAnything/PromptDA
	$PIP install huggingface_hub

	exit
fi

#install Video-Depth-Anything
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
git checkout 3628f50d55e81183c7cc7025f2c22190fa37ef28

mkdir checkpoints
cd checkpoints
$DOWNLOAD $DOWNLOAD_ARG $MODEL_URL
cd ..


#install Depth-Anything-V2
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2

cd metric_depth
mkdir checkpoints
cd checkpoints
$DOWNLOAD $DOWNLOAD_ARG https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
cd ..

cd ..
cd ..
cd ..

cp -a other/metric_dpt_func.py Video-Depth-Anything/Depth-Anything-V2/metric_depth/.
