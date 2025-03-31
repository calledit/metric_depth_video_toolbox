# Metric depth video toolbox - Usage man

The following usage manual is not allways keept up to date for exact info on supported arguments use --help

#### video_metric_convert.py
_Uses ML to create stable metric depth video from any normal video file_
By taking the stability in the videos from [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) and combining it with the  metric version of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) it is possible to generate stable metric depth videos. That is what this tool does.
```bash
usage: video_metric_convert.py [-h] --color_video COLOR_VIDEO [--input_size INPUT_SIZE] [--max_res MAX_RES] [--max_frames MAX_FRAMES] [--target_fps TARGET_FPS] [--max_depth MAX_DEPTH]
                               [--no_rolling_average]

Video Depth Anything

options:
  -h, --help            show this help message and exit
  --color_video COLOR_VIDEO
  --input_size INPUT_SIZE
  --max_res MAX_RES
  --max_frames MAX_FRAMES
                        maximum length of the input video, -1 means no limit
  --target_fps TARGET_FPS
                        target fps of the input video, -1 means the original fps
  --max_depth MAX_DEPTH
                        the max depth that the video uses
  --no_rolling_average  Bases the conversion from affine to metric on the first 60 frames. Good for videos where the camera does not move.


# Notes:
# Video-Depth-Anything memory usage scales with aspect ratio. If you are using a 3090 with 24Gb memory and video with 16:9 aspect you need to lower the --input_size to 440 or crop the video down. Aspect ratio of 4:3 works well.

example:
python video_metric_convert.py --color_video some_video.mkv

```

#### unidepth_video.py (rquires installation with  ./install_mvdtoolbox.sh -unidepth )
_Uses ML to create FOV locked metric depth video from any normal video file._ UniDepth is not made for video so the videos it produces are jittery. However UniDepth has the capability of using FOV as given by the user. Which means it's output tend to be more accurate as a whole. That said UniDepth has been trained with less data than many other models so it struggles with certain types of scenes.

```bash
# Create a metric depth video from a normal video

cd UniDepth
python unidepth_video.py --color_video some_video.mkv -xfov 45

```

#### moge_video.py (rquires installation with  ./install_mvdtoolbox.sh -moge )
_Uses ML to create FOV "locked" metric depth video from any normal video file._ Moge is not made for video so the videos it produces are jittery. 

```bash
# Create a metric depth video from a normal video

python moge_video.py --color_video some_video.mkv -xfov 45

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
Opens a depth video in a 3d viewer, for viewing. Can also render depth videos from novel perspectives ussing --render. To use 3d_view_depthfile.py you need to know the camera FOV. If you dont you can estimate it using [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields) or simply experiment with --xfov until things look right.
```bash
usage: 3d_view_depthfile.py [-h] --depth_video DEPTH_VIDEO [--color_video COLOR_VIDEO] [--xfov XFOV] [--yfov YFOV] [--max_depth MAX_DEPTH] [--render] [--remove_edges]
                            [--show_camera] [--compressed] [--draw_frame DRAW_FRAME] [--max_frames MAX_FRAMES] [--transformation_file TRANSFORMATION_FILE]
                            [--transformation_lock_frame TRANSFORMATION_LOCK_FRAME] [--x X] [--y Y] [--z Z] [--tx TX] [--ty TY] [--tz TZ]

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
  --remove_edges        Tries to remove edges that was not visible in image
  --show_camera         Shows lines representing the camera frustrum
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

example:
python 3d_view_depthfile.py --depth_video some_video_depth.mkv --color_video some_video.mkv --xfov 48

```

#### convert_metric_depth_video_to_other_format.py
Converts a RGB encoded depth video to other formats. Either 3d formats like .ply (point cloud files) or .obj (3d mesh) or to a simple greyscale video. The 8 bit greyscale format loses lots of details due to low depth resolution of only 8 bits. The 16bit format has more details but does not compress well and is not well supported.
The tool can also use 2D tracking points in combination with camera transformations to do SLAM triangulation and output a "perfect" .ply that is not based on the estimated depth i a similar way to how colmap works, this can be usefull as reference or as "ground truth". Good tranformation data is required for this to work. Use the mega-sam tool to get accurate tranformations.

Can be used export the camera transformation and triangulated points to .abc alembic format and .blend belnder for usage in other software using --save_alembic.

```
usage: convert_metric_depth_video_to_other_format.py [-h] --depth_video DEPTH_VIDEO [--bit16] [--bit8] [--max_depth MAX_DEPTH] [--save_ply SAVE_PLY] [--save_obj SAVE_OBJ] [--color_video COLOR_VIDEO]
                                                     [--xfov XFOV] [--yfov YFOV] [--min_frames MIN_FRAMES] [--max_frames MAX_FRAMES] [--transformation_file TRANSFORMATION_FILE]
                                                     [--transformation_lock_frame TRANSFORMATION_LOCK_FRAME] [--remove_edges] [--track_file TRACK_FILE] [--strict_mask] [--mask_video MASK_VIDEO]
                                                     [--show_scene_point_clouds] [--save_alembic] [--save_rescaled_depth]

Convert depth video other formats like .obj or .ply or greyscale video

options:
  -h, --help            show this help message and exit
  --depth_video DEPTH_VIDEO
                        video file to use as input
  --bit16               Convert depth video to a 16bit mono grayscale video file
  --bit8                Convert depth video to a rgb grayscale video file
  --max_depth MAX_DEPTH
                        the max depth that the video uses
  --save_ply SAVE_PLY   folder to save .ply pointcloud files in
  --save_obj SAVE_OBJ   folder to save .obj mesh files in
  --color_video COLOR_VIDEO
                        video file to use as color input
  --xfov XFOV           fov in deg in the x-direction, calculated from aspectratio and yfov in not given
  --yfov YFOV           fov in deg in the y-direction, calculated from aspectratio and xfov in not given
  --min_frames MIN_FRAMES
                        start convertion after nr of frames
  --max_frames MAX_FRAMES
                        quit after max_frames nr of frames
  --transformation_file TRANSFORMATION_FILE
                        file with scene transformations from the aligner
  --transformation_lock_frame TRANSFORMATION_LOCK_FRAME
                        the frame that the transformation will use as a base
  --remove_edges        Tries to remove edges that was not visible in image
  --track_file TRACK_FILE
                        file with 2d point tracking data
  --strict_mask         Remove any points that has ever been masked out even in frames where they are not masked
  --mask_video MASK_VIDEO
                        black and white mask video for thigns that should not be tracked
  --show_scene_point_clouds
                        Opens window and shows the resulting pointclouds
  --save_alembic        Save data to a alembic file
  --save_rescaled_depth
                        Saves a video with rescaled depth
  
python convert_metric_depth_video_to_other_format.py --depth_video some_video_depth.mkv --color_video some_video.mp4 --xfov 55 --save_ply ply_output_folder

# Export the entire scene as a .ply files based on points in the tracking file and the transformations in the transformations file
# this will also output a rescaled depth video that has been corrected to be more like the triangulated depth
python convert_metric_depth_video_to_other_format.py --color_video dancing_crop.mp4 --depth_video dancing_crop.mp4_depth.mkv --transformation_file dancing_crop.mp4_depth.mkv_transformations.json --mask_video dancing_crop_mask.mp4 --track_file dancing_crop.mp4_tracking_120.json --save_rescaled_depth --yfov 31.2
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
Create a overlay_mask.png that is white where the overlay is located.
./apply_inpainting.sh some_video.mkv
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

#### sam_track_video.py (rquires installation with  ./install_mvdtoolbox.sh -megasam )
Use [Mega-sam](https://github.com/mega-sam/mega-sam) to track the camera. Outputs a transfomations.json file. Mega-sam merges traditonal SLAM methods with data from estimated ML depth videos to track the camera.

```bash
usage: sam_track_video.py [-h] --color_video COLOR_VIDEO --depth_video DEPTH_VIDEO [--mask_video MASK_VIDEO] [--max_frames MAX_FRAMES] [--max_depth MAX_DEPTH] [--xfov XFOV] [--yfov YFOV]

Mega-sam camera tracker

options:
  -h, --help            show this help message and exit
  --color_video COLOR_VIDEO
  --depth_video DEPTH_VIDEO
                        depth video
  --mask_video MASK_VIDEO
                        black and white mask video for thigns that should not be tracked
  --max_frames MAX_FRAMES
                        maximum length of the input video, -1 means no limit
  --max_depth MAX_DEPTH
                        the max depth that the video uses
  --xfov XFOV           fov in deg in the x-direction, calculated from aspectratio and yfov in not given
  --yfov YFOV           fov in deg in the y-direction, calculated from aspectratio and xfov in not given
  
example:
python sam_track_video.py --yfov 50 --color_video ~/somevideo.mp4 --depth_video ~/somevideo.mp4_depth.mkv
```

#### align_3d_points.py
Uses tracked points in the video to do camera tracking. Outputs a _transformations.json_ file describing the camera movment and rotation.
```bash
usage: align_3d_points.py [-h] --track_file TRACK_FILE [--mask_video MASK_VIDEO] [--strict_mask] [--xfov XFOV] [--yfov YFOV] --depth_video DEPTH_VIDEO [--max_frames MAX_FRAMES]
                          [--max_depth MAX_DEPTH] [--color_video COLOR_VIDEO] [--assume_stationary_camera] [--use_madpose]

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
  --max_frames MAX_FRAMES
                        quit after max_frames nr of frames
  --max_depth MAX_DEPTH
                        the max depth that the video uses
  --color_video COLOR_VIDEO
                        video file to use as color input only used when debuging
  --assume_stationary_camera
                        Makes the algorithm assume the camera a stationary_camera, leads to better tracking.
  --use_madpose         Uses madpose for camera pose estimation.

example:
python align_3d_points.py --track_file some_video_tracking.json --color_video some_video.mkv --depth_video some_video_depth.mkv --xfov 45 
```


