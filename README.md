# Metric video depth anything
Project that uses **Video-Depth-Anything** and metric **Depth-Anything-V2** to generate metric depth videos

### Requirements
Whatever is needed for **Video-Depth-Anything**  **Depth-Anything-V2**
It has been tested on machines that support Cuda 12.4 on vast.ai "template PyTorch (cuDNN Devel)"



## Install
```bash
git clone https://github.com/calledit/metric_video_depth_anything
cd metric_video_depth_anything
./install_mvda.sh
```

## Usage 

```bash

# Create a metric depth video from a normal video

# Note that the script is copied to the Video-Depth-Anything folder on installation.
cd Video-Depth-Anything
python video_metric_convert.py --input_video some_video.mp4

# Convert the output video to grayscale
python rgb_depth_to_greyscale.py --video some_video_depth.mp4


```

## Output
The result is a metric depth video file called something like **outputs/{filename}_depth.mp4**.

The depth file is a normal video file with RGB values where the **red** and **green** channels represent the
upper 8 bits (duplicated to reduce compression artefacts) and the **blue** channel represent
the lower 8 bits. The values are scaled to **MODEL_maxOUTPUT_depth** default is 6 meters.

### Limitations
Depth-Anything-V2 does not take any FOV input and it does not give any FOV outputs. I recommend PerspectiveFields https://huggingface.co/spaces/jinlinyi/PerspectiveFields to estimate the original FOV if you want to project the output in to 3D. But since Depth-Anything-V2 does not take FOV as input the results may look distorted, to thin, to wide or to narrow as the model may have estimated a different FOV internally and used that for its depth estimations. This is especially problematic in dolly zoom shots, where the FOV is very hard to get right.

Longer shots with loots of camera movement may be problematic, metric_video_depth_anything uses metric conversion constants based on rolling averages to to try to mitigate this but it can still be an issue.
