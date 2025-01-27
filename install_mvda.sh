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

cp -a metric_dpt_func.py Video-Depth-Anything/Depth-Anything-V2/metric_depth/.
cp -a video_metric_convert.py Video-Depth-Anything/.

