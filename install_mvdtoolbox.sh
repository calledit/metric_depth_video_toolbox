#!/bin/bash

pip install open3d numpy opencv-python

#Stuff required for Depth-Anything and other third party tools
pip install einops easydict imageio xformers==0.0.29 tqdm


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
	
	if [ "$(uname)" == "Linux" ]; then
		pip install .
	else
		pip3.11 install .
	fi
	
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
	
	
	pip install decord diffusers transformers accelerate kornia
	
	pip install scipy 
	
	
	exit
	
fi



#install support for Moge
if [[ " $@ " =~ " -moge " ]]; then
	git clone https://github.com/microsoft/MoGe

	cd MoGe
	git checkout dd158c05461f2353287a182afb2adf0fda46436f
	pip install -r requirements.txt

	cd ..
	
	exit
fi


#install support for Mega-sam
if [[ " $@ " =~ " -megasam " ]]; then
	git clone --recursive https://github.com/mega-sam/mega-sam

	cd mega-sam
	
	pip install torch_scatter
	
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
	
	pip install timm wandb
	
	#You get warning if you dont install this but it is not needed
 	#cd unidepth/ops/knn;bash compile.sh;cd ../../../
	
	cd ..
	
	exit
fi


#install support for unidepth
if [[ " $@ " =~ " -unidepth " ]]; then
	git clone https://github.com/lpiccinelli-eth/UniDepth

	cd UniDepth
	
	pip install timm wandb
	wget https://raw.githubusercontent.com/AbdBarho/xformers-wheels/refs/heads/main/xformers/components/attention/nystrom.py
	sed -i 's/from xformers\.components\.attention import NystromAttention/from nystrom import NystromAttention/g' unidepth/layers/nystrom_attention.py

 	cd unidepth/ops/knn;bash compile.sh;cd ../../../
	
	cd ..
	
	exit
fi

#install Video-Depth-Anything
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
git checkout 3628f50d55e81183c7cc7025f2c22190fa37ef28
#pip install -r requirements.txt
cd metric_depth
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth
cd ..
cd ..
cd ..