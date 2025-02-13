# Depth video toolbox

Tools for Generating and working with monocular-depth-estimation ML and metric 3D videos.

![gif_banner](https://github.com/user-attachments/assets/4d737bb3-6fb6-4135-b01e-b35528371d22)

## Video showcase
https://youtu.be/nEiUloZ591Q

Stereo video clip samples can be found here:
https://github.com/calledit/depth_video_toolbox/releases/tag/Showcase

## This Repo consists of:
1. A tool for generating metric 3D depth videos based on the Depth-Anything series of machine learning models.
By taking the stability in the videos from [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) and combining it with the  metric version of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) it is possible to generate stable metric depth videos.
2. A tool for generating metric 3D depth videos based on [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
3. Tools for working with and visualising the metric 3D videos.
4. Tools for doing things like 3D stereo rendering or viewing a video from above or othervise altering the camera perspective.
5. Tools for using the generated metric 3D videos for camera tracking(camera pose estimation) and (full scene 3D reconstruction).


## Usage 


#### pipline.py
Takes a color image and a mask. Splits it in to clips and does all the required steps required to create stabilized stereoscopic 3d videos from the clips.

```
usage: pipline.py [-h] --color_video COLOR_VIDEO --mask_video MASK_VIDEO --xfov XFOV --mask_depth MASK_DEPTH --clip_name CLIP_NAME --clip_starttime CLIP_STARTTIME --clip_len CLIP_LEN

Take a clip from a color video and make it in to a stereo 3D video

options:
  -h, --help            show this help message and exit
  --color_video COLOR_VIDEO
                        video file to use as color input
  --mask_video MASK_VIDEO
                        Mask video file
  --xfov XFOV           camera field of view in x direction
  --mask_depth MASK_DEPTH
                        The depth in meters that is considerd background. (used for infill)
  --clip_name CLIP_NAME
                        A name to give the clip
  --clip_starttime CLIP_STARTTIME
                        Clip start time given as mm:ss
  --clip_len CLIP_LEN   Clip length time given as mm:ss
  				
example:
create_video_mask.sh some_video.mkv #generate mask
python pipline.py --color_video some_video.mkv --mask_video mask.mp4 --xfov 55 --mask_depth 2.0 --clip_name clip_nr_1 --clip_starttime 02:11 --clip_len 01:25
```

#### video_metric_convert.py
_Uses ML to create stable metric depth video from any normal video file_
```bash
# Create a metric depth video from a normal video (Note that the video_metric_convert.py script is copied to the Video-Depth-Anything folder on installation.)

# Video-Depth-Anything memory usage scales with aspect ratio. If you are using a 3090 with 24Gb memory and video with 16:9 aspect you need to lower the --input_size to 440 or crop the video down. Aspect ratio of 4:3 works well.
cd Video-Depth-Anything
python video_metric_convert.py --color_video some_video.mkv

```

#### unidepth_video.py (rquires installation with  ./install_mvda.sh -unidepth )
_Uses ML to create FOV locked metric depth video from any normal video file._ UniDepth is not made for video so the videos it produces are very jittery. However UniDepth has the capability of using FOV as given by the user. Which means it's output tend to be more accurate as a whole. That said UniDepth has been trained with less data so it struggles with certain types of scenes. **UniDepth requires Cuda 11.8 and Torch 2.2.0.**
```bash
# Create a metric depth video from a normal video (Note that the unidepth_video.py script is copied to the UniDepth folder on installation.)

cd UniDepth
python unidepth_video.py --color_video some_video.mkv -xfov 45

```

#### stereo_rerender.py
Uses a generated depth video together with the source color video to render a new stereo 3D video. To use stereo_rerender.py you need to know the camera FOV. If you dont you can estimate it using [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields)
```bash
usage: stereo_rerender.py [-h] --depth_video DEPTH_VIDEO [--color_video COLOR_VIDEO] [--xfov XFOV] [--yfov YFOV] [--max_depth MAX_DEPTH] [--transformation_file TRANSFORMATION_FILE]
                         [--transformation_lock_frame TRANSFORMATION_LOCK_FRAME] [--pupillary_distance PUPILLARY_DISTANCE] [--max_frames MAX_FRAMES] [--touchly0] [--touchly1]
                         [--touchly_max_depth TOUCHLY_MAX_DEPTH] [--compressed] [--infill_mask] [--remove_edges] [--mask_depth MASK_DEPTH] [--save_background] [--load_background LOAD_BACKGROUND]

Take a rgb encoded depth video and a color video, and render them it as a stereoscopic 3D video.that can be used on 3d tvs and vr headsets.

options:
  -h, --help            show this help message and exit
  --depth_video DEPTH_VIDEO
                        video file to use as input
  --color_video COLOR_VIDEO
                        video file to use as color input
  --xfov XFOV           fov in deg in the x-direction, calculated from aspectratio and yfov in not given
  --yfov YFOV           fov in deg in the y-direction, calculated from aspectratio and xfov in not given
  --max_depth MAX_DEPTH
                        the max depth that the input video uses
  --transformation_file TRANSFORMATION_FILE
                        file with scene transformations from the aligner
  --transformation_lock_frame TRANSFORMATION_LOCK_FRAME
                        the frame that the transfomrmation will use as a base
  --pupillary_distance PUPILLARY_DISTANCE
                        pupillary distance in mm
  --max_frames MAX_FRAMES
                        quit after max_frames nr of frames
  --touchly0            Render as touchly0 format. ie. stereo video with 3d
  --touchly1            Render as touchly1 format. ie. mono video with 3d
  --touchly_max_depth TOUCHLY_MAX_DEPTH
                        the max depth that touchly is cliped to
  --compressed          Render the video in a compressed format. Reduces file size but also quality.
  --infill_mask         Save infill mask video.
  --remove_edges        Tries to remove edges that was not visible in image(it is a bit slow)
  --mask_depth MASK_DEPTH
                        Saves a compound backfround version of the mesh that can be used as infill. Set to background distance in meter. (only works for non moving cameras)
  --save_background     Save the compound background as a file. To be ussed as infill.
  --load_background LOAD_BACKGROUND
                        Load the compound background as a file. To be used as infill.

example:
python stereo_rerender.py --depth_video some_video_depth.mkv --color_video some_video.mkv --xfov 48

```

#### 3d_view_depthfile.py
Opens a depth video in a 3d viewer, for viewing. To use 3d_view_depthfile.py you need to know the camera FOV. If you dont you can estimate it using [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields)
```bash
usage: 3d_view_depthfile.py [-h] --depth_video DEPTH_VIDEO [--color_video COLOR_VIDEO] [--xfov XFOV] [--yfov YFOV] [--max_depth MAX_DEPTH] [--render] [--remove_edges] [--mask_depth MASK_DEPTH]
                            [--compressed] [--draw_frame DRAW_FRAME] [--max_frames MAX_FRAMES] [--transformation_file TRANSFORMATION_FILE] [--transformation_lock_frame TRANSFORMATION_LOCK_FRAME]
                            [--x X] [--y Y] [--z Z] [--tx TX] [--ty TY] [--tz TZ]

Take a rgb encoded depth video and a color video, and view it/render as 3D

options:
  -h, --help            show this help message and exit
  --depth_video DEPTH_VIDEO
                        video file to use as input
  --color_video COLOR_VIDEO
                        video file to use as color input
  --xfov XFOV           fov in deg in the x-direction, calculated from aspectratio and yfov in not given
  --yfov YFOV           fov in deg in the y-direction, calculated from aspectratio and xfov in not given
  --max_depth MAX_DEPTH
                        the max depth that the video uses
  --render              Render to video insted of GUI
  --remove_edges        Tries to remove edges that was not visible in image(it is a bit slow)
  --mask_depth MASK_DEPTH
                        Only keeps parts further away than specifid depth
  --compressed          Render the video in a compressed format. Reduces file size but also quality.
  --draw_frame DRAW_FRAME
                        open gui with specific frame
  --max_frames MAX_FRAMES
                        quit after max_frames nr of frames
  --transformation_file TRANSFORMATION_FILE
                        file with scene transformations from the aligner
  --transformation_lock_frame TRANSFORMATION_LOCK_FRAME
                        the frame that the transfomrmation will use as a base
  --x X                 set position of cammera x cordicate in meters
  --y Y                 set position of cammera y cordicate in meters
  --z Z                 set position of cammera z cordicate in meters
  --tx TX               set poistion of camera target x cordinate in meters
  --ty TY               set poistion of camera target y cordinate in meters
  --tz TZ               set poistion of camera target z cordinate in meters
python 3d_view_depthfile.py --depth_video some_video_depth.mkv --color_video some_video.mkv --xfov 48

```

#### rgb_depth_to_greyscale.py
Converts a RGB encoded depth video to a simple greyscale video, a format which certain software likes to work with. 8 bit greyscale looses allot of details due to low depth resolution. The 16bit format has more details but does not compress well and is not well supported.
```bash
usage: rgb_depth_to_greyscale.py [-h] --depth_video DEPTH_VIDEO [--bit16] [--max_depth MAX_DEPTH]

Generate a depth video in greyscale from a rgb encoded depth video

options:
  -h, --help            show this help message and exit
  --depth_video DEPTH_VIDEO
                        video file to use as input
  --bit16               Store output as 16bit file
  --max_depth MAX_DEPTH
                        the max depth that the video uses

example:
python rgb_depth_to_greyscale.py --depth_video some_video_depth.mkv
```

#### create_video_mask.sh
Uses ML to create a video mask for the main subjects in the video based on rembg. The masks can be used to filter out moving objects when running alignment.
```bash
#Create a vido mask
./create_video_mask.sh some_video.mkv
```


#### apply_inpainting.sh
Uses ML to paint over logos, text overlays or other objects from a video, can be useful to do before running the depth ML models as they tend to produce less accurate results when the video has logos or text overlays.
```bash
example:
Create a overlay_mask.png that is white where the overlay is.
./create_video_mask.sh some_video.mkv
```

#### track_points_in_video.py
Tracks points in the video. Uses the ML model cotracker3 to track points in the video. Outputs a _tracking.json_ file that contains tracking points for the entire video.
```bash
usage: track_points_in_video.py [-h] --color_video COLOR_VIDEO

Generate a json tracking file from a video

options:
  -h, --help            show this help message and exit
  --color_video COLOR_VIDEO
                        video file to use as input

example:
python track_points_in_video.py --color_video some_video.mkv
```

#### align_3d_points.py
Uses tracked points in the video and projectes them on to the depth video for 3D alignment and camera tracking. Outputs a _transformations.json_ file describing the camera movment and rotation.
```bash
usage: align_3d_points.py [-h] --track_file TRACK_FILE [--mask_video MASK_VIDEO] [--strict_mask] [--xfov XFOV] [--yfov YFOV] --depth_video DEPTH_VIDEO [--max_depth MAX_DEPTH]
                          [--color_video COLOR_VIDEO] [--assume_stationary_camera]

Align 3D video based on depth video and a point tracking file

options:
  -h, --help            show this help message and exit
  --track_file TRACK_FILE
                        file with 2d point tracking data
  --mask_video MASK_VIDEO
                        black and white mask video for thigns that should not be tracked
  --strict_mask         Remove any points that has ever been masked out even in frames where they are not masked
  --xfov XFOV           fov in deg in the x-direction, calculated from aspectratio and yfov in not given
  --yfov YFOV           fov in deg in the y-direction, calculated from aspectratio and xfov in not given
  --depth_video DEPTH_VIDEO
                        depth video
  --max_depth MAX_DEPTH
                        the max depth that the video uses
  --color_video COLOR_VIDEO
                        video file to use as color input only used when debuging
  --assume_stationary_camera
                        Makes the algorithm assume the camera a stationary_camera, leads to better tracking.

example:
python align_3d_points.py --track_file some_video_tracking.json --color_video some_video.mkv --depth_video some_video_depth.mkv --xfov 45 
```

## 3D depth video format
The video depth format is a normal video file with RGB values where the _red_ and _green_ channels represent the
upper 8 bits of the depth (duplicated to reduce compression artefacts) and the _blue_ channel represent
the lower 8 bits. The values are scaled to the argument --max_depth, default is 20 meters.


## Install
```bash


git clone https://github.com/calledit/depth_video_toolbox
cd depth_video_toolbox

# on linux
sudo apt-get install -y libgl1
./install_mvda.sh
pip install open3d numpy opencv-python

#if you want to use paralax ML infill
./install_mvda.sh -stereocrafter

#if you want to use 3d camera tracking and 3d reconstruction
./install_mvda.sh -madpose


# If you want to export directly to the avc1 codec using the --compress argument
echo https://swiftlane.com/blog/generating-mp4s-using-opencv-python-with-the-avc1-codec/

# if using headless linux
apt-get install xvfb
# then run before using the tools (ie. start a virtual x11 server)
Xvfb :2 &
export DISPLAY=:2

# on OSX (OSX only supports post processing of depth videos not generation of them. As the ML models need CUDA)

# First setup any required venv (open3d requires python3.11 on OSX (as of 2025)))
pip3.11 install open3d numpy opencv-python

#On Windows (Not tested or "officially" supported, but anecdotally working)
See https://github.com/calledit/depth_video_toolbox/issues/1#issuecomment-2632040738

```

### Requirements
Has been tested on machines that support Cuda 12.4 on [vast.ai](https://cloud.vast.ai/?ref_id=148636) "template PyTorch (cuDNN Devel)"

### Limitations
Depth-Anything-V2 does not take any FOV input and it does not give any FOV outputs. I recommend [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields) to estimate the original FOV if you want to project the output in to 3D. But since Depth-Anything-V2 does not take FOV as input the results may look distorted, to thin, to wide or to narrow as the model may have estimated a different FOV internally and used that for its depth estimations. This is especially problematic in dolly zoom shots, where the FOV is very hard to get right.

Longer shots with loots of camera movement may be problematic, video_metric_convert.py uses metric conversion constants based on rolling averages to to try to mitigate this but it can still be an issue.


### Camera tracking
align_3d_points.py is a tool to extract camera movment from the video. It is based on the [madpose library](https://github.com/MarkYu98/madpose), madpose offers excellent camera pose estimation and given enogh acurate non moving tracking markers, and somewhat acurate stable metric depth it can give very accurate pose estimations even for scenes where traditional pnpSolve or SVD solutions strugle (like scenes where there is little camera movment or where the depth maps are not perfeecly accurate). 

