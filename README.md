# Metric depth video toolbox (MDVToolbox)

Tools for Generating and working with metric 3D depth videos.

![gif_banner](https://github.com/user-attachments/assets/4d737bb3-6fb6-4135-b01e-b35528371d22)

_Banner created with 3d_view_depthfile.py_


## Video showcase
https://youtu.be/nEiUloZ591Q

Stereo video clip samples can be found here:
https://github.com/calledit/metric_depth_video_toolbox/releases/tag/Showcase

Demo video of movie 3d conversion using [movie_2_3D.py](movie_2_3D.py)
https://www.youtube.com/watch?v=PLFjoNgkZDY

## This Repo consists of:
1. Tools for generating metric 3D depth videos based on:
   a) the Depth-Anything series of machine learning models.
   b) The [MoGe](https://github.com/microsoft/MoGe) machine learning model.
   c) The [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)  machine learning model.
   d) DepthPro, UniK3D, depthcrafter, MVSAnywhere
2. A tool for viewing, rendering and visualising metric 3D videos from novel camera perspectives.
3. A tool for 3D stereo rendering. Converting normal video in to 3D video.
4. A tool for adding parallax infill to generated stero video based on [StereoCrafter](https://github.com/TencentARC/StereoCrafter).
5. Tools for using metric 3D videos for camera tracking(camera pose estimation) and (full scene 3D reconstruction).
6. Tools for automaticly creating masks and doing ML infill over logos or subtitles in videos.


## Usage 

See [HOWTO.md](https://github.com/calledit/metric_depth_video_toolbox/blob/main/HOWTO.md) for a simple beginner guide.

See [HOWTO_movie2_3d.md](https://github.com/calledit/metric_depth_video_toolbox/blob/main/HOWTO_movie2_3d.md) for a guide on how to convert a full movie into 3D.

See [USAGE.md](https://github.com/calledit/metric_depth_video_toolbox/blob/main/USAGE.md) Info on each tool and the arguments it can take.

### Camera tracking
There are multiple tracking tools that support camera tracking:

- [sam_track_video.py](sam_track_video.py) a tool based on [Mega-sam](https://github.com/mega-sam/mega-sam) it uses a combination of machine learning and depth maps to track the camera in 3d space.
- [track_points_in_video.py](track_points_in_video.py) uses cotracker3 to track 2D points over full length videos. These tracked points can be used for camera tracking and 3d reconstruction.
- [align_3d_points.py](align_3d_points.py) is a tool that can extract 3d camera movment from depth video and tracked 2d points. It offers three difrrent algorithms. 
    1. Madpose PnPSolver [madpose library](https://github.com/MarkYu98/madpose). Better than traditonal PnPSolve, but suffers to long term drift as it is a fram 2 frame solution.
    2. SVD based rotational solver asuming the camera is stationary and only tracking rotation. If the camera is trully still this is the best option.
    3. Iterative camera movmenet untill best fitt. Offers better tracking than madpose and is very fast.
- [optical_flow.py](optical_flow.py) a tool to generate a optical flow video from a RGB video.

### Data export
[convert_metric_depth_video_to_other_format.py](convert_metric_depth_video_to_other_format.py) is a tool that can take data and export the data in to standard formats that can be used in external tools like blender.

**Supported export data:**
- video frames to .ply point clouds.
- video frames to .obj model files
- Camera tracking data in to blender .blend and alembic camera tracking files .abc
- Rescaled depth video, rescaled based on triangulation and camera tracking data.
- Depth video in to 16 and 8 bit greyscale format.

### Stereo rendering
- [stereo_rerender.py](stereo_rerender.py) can render color video together with depth video in to side by side 3d stereo video with paralax infill masks.
- [stereo_crafter_infill.py](stereo_crafter_infill.py) fills in the missing paralax areas in side by side 3d stereo video using stable video diffusion.
- [movie_2_3D.py](movie_2_3D.py) is a automated 3D converter, that splits up a movie in to scenes and automaticly converts each scene in to 3d then stitches everything together again.

### Depth estimation
There are varoius tools for doing depth estimation in the toolbox.
- [video_metric_convert.py](video_metric_convert.py) Converts video in to metric depth video using the Depth-Anything series of machine learning models.
- [videoanythingmetric_video.py](videoanythingmetric_video.py) Converts video in to metric depth using Video-Depth-Anything-Metric
- [moge_video.py](moge_video.py) Converts a video in to metric depth video using MoGe.
- [unidepth_video.py](unidepth_video.py) Converts a video in to metric depth video using UniDepth.
- [unik3d_video.py](unik3d_video.py) Converts a video in to metric depth video using UniK3D.
- [depthpro_video.py](depthpro_video.py) Converts a video in to metric depth video using DepthPro.
- [depthcrafter_video.py](depthcrafter_video.py) Converts video in to metric depth video using depthcrafter and a metric depth reference video.
- [geometrycrafter_video.py](geometrycrafter_video.py) Stablilizes a metric video generated with a single frame depthmodel.
- [upscale_depth_promptda.py](upscale_depth_promptda.py) Upscales metric depth video given a full resolution color video using PromptDA.
- [video_mvsa.py](video_mvsa.py) Converts a video together with camera poses in to metric depth video using MVSAnywhere.


### Viewing 3d video
[3d_view_depthfile.py](3d_view_depthfile.py) is a 3d viewer that can be used to either view 3d video in a open3d window. Or to render 3d video from novel perspectives in to new video. See README banner for example render.


### Masking tools
- [generate_video_mask.py](generate_video_mask.py) Creates a mask video that masks of all humans in a video.
- [apply_inpainting.sh](apply_inpainting.sh) Removes logos or overlays from video using ML inpanting.

## RGB encoded metric 3D depth video format
The rgb encoded video depth format is a normal video file with RGB values(that has to be saved as lossless video). Where the _red_ and _green_ channels represent the upper 8 bits of the depth (duplicated to make visualization easy), the _blue_ channel represent
the lower 8 bits. Only 16bits of the 24 bit rgb data is used to keep down filesizes. The values are scaled to the argument --max_depth, default is 100 meters.

**With the default --max_depth of 100 meters each visible "ridge" represents a depth of 390mm (=100/256) and the depth resolution is about 1.5 mm.**


As depth estimation models improve (especially for distant things) this 16bit depth format will need to be replaced with something better. At that point the use of the full 24bits might be the easiest solution(using 24bits and keeping the resolution at 1mm; the max depth would be 16km which is the distancde to the horizon if standing 20m up from the ground), but one could also encode deepth logaritmicly making things in the distance less accurate.


## Install

### Windows
1. install git https://git-scm.com/downloads/win
2. Install miniconda https://docs.conda.io/en/latest/
3. Open the Anacoda prompt(miniconda) from the start menu
4. run
```batch
git clone https://github.com/calledit/metric_depth_video_toolbox
cd metric_depth_video_toolbox
windows_installer.bat
```
5. use `conda activate mdvt` to activate conda and use the tools. Some ML models are not supported on windows but the most esential ones are supported like Sterecrafter and depth anything.

### Ubuntu/Debian and OSX

```bash


git clone https://github.com/calledit/metric_depth_video_toolbox
cd metric_depth_video_toolbox

# on linux
sudo apt-get install -y libgl1
./install_mdvtoolbox.sh

#Optional (only required for some tools)
./install_mdvtoolbox.sh -megasam
./install_mdvtoolbox.sh -geometrycrafter
./install_mdvtoolbox.sh -unik3d
./install_mdvtoolbox.sh -depthpro
./install_mdvtoolbox.sh -stereocrafter
./install_mdvtoolbox.sh -madpose
./install_mdvtoolbox.sh -unidepth
./install_mdvtoolbox.sh -moge
./install_mdvtoolbox.sh -promptda

# if using headless linux you need to start a virtual x11 server
apt-get install xvfb
Xvfb :2 &
export DISPLAY=:2

# OSX (OSX only supports post processing of depth videos not generation of them. As the ML models need CUDA)
# (open3d requires python3.11 on OSX (as of 2025)))
pip3.11 install open3d numpy opencv-python

```

### Requirements
The tools that reuire ML models have been tested on machines with nvida 3090 cards that support Cuda 12.4 and Torch 2.5.1 on [vast.ai](https://cloud.vast.ai/?ref_id=148636) using "template PyTorch (cuDNN Devel)"

### Next steps
- Currently waiting for new stable and accurate depth models.

### Contributing
Is appreciated. Even for simple things like spelling.
