# Metric video depth anything

Tools for Generating and working with monocular-depth-estimation ML and metric 3D videos.

![3D GIF](https://github.com/user-attachments/assets/7b78d85c-40c0-45c8-91c4-41bf779f6f50)

## Video showcase
https://youtu.be/nEiUloZ591Q

Stero video clip samples can be found here:
https://github.com/calledit/metric_video_depth_anything/releases/tag/Showcase

## This Repo consists of:
1. A tool for generating metric 3D depth videos based on the Depth-Anything series of machine learning models.
By taking the stability in the videos from [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) and combining it with the  metric version of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) it is possible to generate stable metric depth videos.
2. A tool for generating metric 3D depth videos based on [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
3. Tools for working with and visualising the metric 3D videos.
4. Tools for doing things like 3D stereo rendering or viewing a video from above or othervise altering the camera perspective.
5. (WIP) Tools for using the generated metric 3D videos for camera tracking(camera pose estimation) and (full scene 3D recunstruction).


## Usage 

#### video_metric_convert.py
_Uses ML to create stable metric depth video from any normal video file_
```bash
# Create a metric depth video from a normal video (Note that the video_metric_convert.py script is copied to the Video-Depth-Anything folder on installation.)

# Video-Depth-Anything memory usage scales with aspect ratio. If you are using a 3090 with 24Gb memory and video with 16:9 aspect you need to lower the --input_size to 440 or crop the video down. Aspect ratio of 4:3 works well.
cd Video-Depth-Anything
python video_metric_convert.py --color_video some_video.mkv

```

#### unidepth_video.py (rquires installation with  ./install_mvda.sh -unidepth )
_Uses ML to create FOV locked metric depth video from any normal video file._ UniDepth is not made for video so the videos it produces are very jittery. However UniDepth has the capability of using FOV as given by the user. Which means it's output tend to be more accurate as a whole. That said UniDepth has been trained wiht less data so it strugles with certain types of scenes. 
```bash
# Create a metric depth video from a normal video (Note that the unidepth_video.py script is copied to the UniDepth folder on installation.)

cd UniDepth
python unidepth_video.py --color_video some_video.mkv -xfov 45

```

#### stero_rerender.py
_Uses a generated depth video together with the source color video to render a new stereo 3D video. To use stero_rerender.py you need to know the camera FOV. If you dont you can estimate it using [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields)_
```bash
# Renders a stereo 3D video that can be used on 3d tv's and vr headsets.
python stero_rerender.py --depth_video some_video_depth.mkv --color_video some_video.mkv --xfov 48

```

#### 3d_view_depthfile.py
_Opens a depth video in a 3d viewer, for viewing. To use 3d_view_depthfile.py you need to know the camera FOV. If you dont you can estimate it using [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields)_
```bash
# View the depth video in 3D. Requires open3d (pip install open3d)
python 3d_view_depthfile.py --depth_video some_video_depth.mkv --color_video some_video.mkv --xfov 48

```

#### rgb_depth_to_greyscale.py
_Converts a RGB encoded depth video to a simple greyscale video, a format which certain software likes to work with._
```bash
# Convert the output video to grayscale
python rgb_depth_to_greyscale.py --depth_video some_video_depth.mkv
```

#### create_video_mask.sh
_Create a vido mask for the videos main subjects uses rembg and ffmpeg. Install with pip install rembg_
```bash
#Create a vido mask
./create_video_mask.sh some_video.mkv
```

#### track_points_in_video.py
_Tracks points in the video. Used for 3D alignment and camera tracking. Generates a file called some_video_tracking.json what contains tracking points for the entire video._
```bash
#track points
python track_points_in_video.py --color_video some_video.mkv
```

#### align_3d_points.py (WIP)
_Uses tracked points in the video and projectes them on to the depth video for 3D alignment and camera tracking._
```bash
#align 3d points
python align_3d_points.py --track_file some_video_tracking.json --color_video some_video.mkv --depth_video some_video_depth.mkv --xfov 45 
```

## Output
The result is a metric depth video file called something like outputs/{filename}_depth.mkv.

The depth file is a normal video file with RGB values where the _red_ and _green_ channels represent the
upper 8 bits (duplicated to reduce compression artefacts) and the _blue_ channel represent
the lower 8 bits. The values are scaled to the argument --max_depth, default is 20 meters.


## Install
```bash


git clone https://github.com/calledit/metric_video_depth_anything
cd metric_video_depth_anything

# on linux
sudo apt-get install -y libgl1
./install_mvda.sh
pip install open3d numpy

# Follow this if you want to save videos with the avc1 codec which is required if you want to watch the exported videos on a Quest device (i think)
echo https://swiftlane.com/blog/generating-mp4s-using-opencv-python-with-the-avc1-codec/

# if using headless linux
apt-get install xvfb
# then run before using the tools (ie. start a virtual x11 server)
Xvfb :2 &
export DISPLAY=:2

# on OSX (OSX only supports post processing of depth videos not generation of them. As the ML models need CUDA)

# First setup any required venv (open3d requires python3.11 on OSX (as of 2025)))
pip3.11 install open3d numpy opencv-python

```

### Requirements
Has been tested on machines that support Cuda 12.4 on [vast.ai](https://cloud.vast.ai/?ref_id=148636) "template PyTorch (cuDNN Devel)"

### Limitations
Depth-Anything-V2 does not take any FOV input and it does not give any FOV outputs. I recommend [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields) to estimate the original FOV if you want to project the output in to 3D. But since Depth-Anything-V2 does not take FOV as input the results may look distorted, to thin, to wide or to narrow as the model may have estimated a different FOV internally and used that for its depth estimations. This is especially problematic in dolly zoom shots, where the FOV is very hard to get right.

Longer shots with loots of camera movement may be problematic, video_metric_convert.py uses metric conversion constants based on rolling averages to to try to mitigate this but it can still be an issue.
