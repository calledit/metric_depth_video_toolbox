#!/bin/bash


#install Video-Depth-Anything
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
pip install -r requirements.txt

mkdir checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
cd ..


#install Depth-Anything-V2
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt

cd metric_depth
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
cd ..

cd ..
cd ..
cd ..

cp -a src/metric_dpt_func.py Video-Depth-Anything/Depth-Anything-V2/metric_depth/.
cp -a src/video_metric_convert.py Video-Depth-Anything/.

#to install with support for unidepth (requires cuda 11.8 and torch 2.2.0)
if [[ " $@ " =~ " -unidepth " ]]; then
	git clone https://github.com/lpiccinelli-eth/UniDepth

	cd UniDepth

	#export VENV_DIR=<YOUR-VENVS-DIR>
	#export NAME=Unidepth

	#python -m venv $VENV_DIR/$NAME
	#source $VENV_DIR/$NAME/bin/activate

	# Install UniDepth and dependencies
	pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

	# Install Pillow-SIMD (Optional)
	pip uninstall pillow
	CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
	
	cd ..
	
	cp -a src/unidepth_video.py UniDepth/.
fi
