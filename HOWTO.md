# Metric depth video toolbox - Usage examples

This guide contains a walkthrogh of how to use the tools in the metric depth video toolbox.

## Start
Select a video to work with. This should be a clip, preferably less than 6-7 minutes long (due to GPU memmory usage), and there should not be any cuts in the video. The video should preferably have the same zoom level over the hole clip. Due to GPU memmory constraints in Video-Depth-Anything the aspect ratio is best keept under 16:9.

I will use [in_office_720p.mp4](https://github.com/calledit/metric_depth_video_toolbox/releases/download/ExampleFiles/in_office_720p.mp4) with two individuals walking in a hallway obtained from pexels.com

## Step 0
On your **LINUX** machine Install metric depth video toolbox:

This will install required packages and download the needed ML models.
```
git clone https://github.com/calledit/metric_depth_video_toolbox
cd metric_depth_video_toolbox

./install_mvdtoolbox.sh
```

## Step 1
Generate a metric depth video from the source video

```
python video_metric_convert.py --color_video ~/in_office_720p.mp4

the result is a metric 3d video file called ~/in_office_720p.mp4_depth.mkv
```

### Step 1.5
View result:
```
python3.11 3d_view_depthfile.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 30
````

## Step 2
Generate a mask video from the source video

```
./create_video_mask.sh -install

./create_video_mask.sh ~/in_office_720p.mp4

the result is a black and white mask video ~/in_office_720p.mp4_mask.mkv
```


## Step 3
Generate tracking points from the source video, more iterations = more points. (But to many points may cause later triangulation to crash due to memmory usage.)

```
python track_points_in_video.py --color_video ~/in_office_720p.mp4 --nr_iterations 2

the result is a tracking file called ~/in_office_720p.mp4_tracking.json

```

## Step 4
Generate camera transformations from the depth and the source video. I just looked at the video and made a guess of 35 deg(which was wildly wrong) but megasam managed to solve the FOV anyway. If there is lots of movment you can add the mask here aswell (my experience is that using the mask here often makes the tracking a tiny bit worse as mega-sam is quite good at ignoring movment by itself)

```
./install_mvdtoolbox.sh -megasam #takes a long time to install
python sam_track_video.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 35

The result is a refined FOV:
Estimated intrinsics: fovx: 16.024212493035197 fovy 9.05395233745844

And a tranformtions file  ~/in_office_720p.mp4_depth.mkv_transformations.json
and a debug depth video file called in_office_720p.mp4_depth.mkv_megasam.mkv
```


## Step 5
Triangulate points to get acurate depth readings and realigin the metric depth video to fit the more accurate depth readings.

```
python3.11 convert_metric_depth_video_to_other_format.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 9 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --track_file ~/in_office_720p.mp4_tracking.json --mask_video ~/in_office_720p.mp4_mask.mkv --show_scene_point_clouds --use_triangulated_points --save_rescaled_depth --show_both_point_clouds --global_align

The result is a rescaled depth video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv
And two .ply files with point cloud data for the scene. One ply file with tirangualted points and one with averages of the depth map called in_office_720p.mp4_depth.mkv_avgmonodepth.ply, in_office_720p.mp4_depth.mkv_triangulated.ply.
You can run the script again with the new _rescled.mkv file to get a rescaled version of the avgmonodepth.ply file.

```


## Step  6
View the final result with camera movment canceled out, edges removed, a background .ply file inserted and visulisation for the camera view frustrum.

```
python3.11 3d_view_depthfile.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv --yfov 9 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --background_ply ~/in_office_720p.mp4_depth.mkv_avgmonodepth.ply --remove_edges --show_camera

```
