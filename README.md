# Metric video depth anything
Generate metric 3d video with ML.

By taking the stability in the videos from [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) and combining it with the  metric version of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) it is posible to generate stable metric depth videos.



## Usage 
There are various python scripts with diffrent functions. 

```bash

# Create a metric depth video from a normal video (Note that the video_metric_convert.py script is copied to the Video-Depth-Anything folder on installation.)
cd Video-Depth-Anything
python video_metric_convert.py --input_video some_video.mkv

# Convert the output video to grayscale
python rgb_depth_to_greyscale.py --video some_video_depth.mkv

# View the depth video in 3d. Requires open3d (pip install open3d)
python 3d_view_depthfile.py --video some_video_depth.mkv --color_video some_video.mkv --xfov 48


```

## Output
The result is a metric depth video file called something like _outputs/{filename}_depth.mkv_.

The depth file is a normal video file with RGB values where the _red_ and _green_ channels represent the
upper 8 bits (duplicated to reduce compression artefacts) and the _blue_ channel represent
the lower 8 bits. The values are scaled to the argument --max_depth, default is 6 meters.


## Install
```bash
git clone https://github.com/calledit/metric_video_depth_anything
cd metric_video_depth_anything
./install_mvda.sh
```

### Requirements
Has been tested on machines that support Cuda 12.4 on vast.ai "template PyTorch (cuDNN Devel)"

### Limitations
Depth-Anything-V2 does not take any FOV input and it does not give any FOV outputs. I recommend PerspectiveFields https://huggingface.co/spaces/jinlinyi/PerspectiveFields to estimate the original FOV if you want to project the output in to 3D. But since Depth-Anything-V2 does not take FOV as input the results may look distorted, to thin, to wide or to narrow as the model may have estimated a different FOV internally and used that for its depth estimations. This is especially problematic in dolly zoom shots, where the FOV is very hard to get right.

Longer shots with loots of camera movement may be problematic, metric_video_depth_anything uses metric conversion constants based on rolling averages to to try to mitigate this but it can still be an issue.
