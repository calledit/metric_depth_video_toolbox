# Metric depth video toolbox - Usage examples

This guide contains a walkthrogh of how to use the tools in the metric depth video toolbox.

## Start
Select a video to work with. This should be a clip, preferably less than 6-7 minutes long (due to GPU memmory usage), and there should not be any cuts in the video. The video should preferably have the same zoom level over the hole clip. Due to GPU memmory constraints in Video-Depth-Anything the aspect ratio is best keept under 16:9.

I will use [in_office_720p.mp4](https://github.com/calledit/metric_depth_video_toolbox/releases/download/ExampleFiles/in_office_720p.mp4) with two individuals walking in a hallway obtained from pexels.com
<img width="930" alt="video_in" src="https://github.com/user-attachments/assets/3ff877b1-fa68-4a0c-8089-6ee6d4fccecc" />
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
<img width="936" alt="depth" src="https://github.com/user-attachments/assets/1b3325ab-8447-4ea5-b536-2dd4a226975b" />

### Step 1.5
View result in 3D:
```
python3.11 3d_view_depthfile.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 40
````

## Step 2
Generate a mask video from the source video

```
./create_video_mask.sh -install

./create_video_mask.sh ~/in_office_720p.mp4

the result is a black and white mask video ~/in_office_720p.mp4_mask.mkv
```
<img width="929" alt="mask" src="https://github.com/user-attachments/assets/d282bc87-2026-4f9e-9f85-21bfb69dace0" />

## Step 3
Generate tracking points from the source video, more iterations = more points steps_bewtwen_track_init is the numer of frames betwen initation of new tracking points.

```
python track_points_in_video.py --color_video ~/in_office_720p.mp4 --nr_iterations 4 --steps_bewtwen_track_init 30

the result is a tracking file called ~/in_office_720p.mp4_tracking.json
```
Visualised here as tiny dots in the images:
<img width="926" alt="tracking_on_video" src="https://github.com/user-attachments/assets/d3f66daf-25ab-4f05-8067-ffbe6822a595" />

## Step 4
Generate camera transformations from the depth and the source video. We make a guess of 30-50 deg and chose 40 deg. Later analysis showed that the real FOV is something like 42 deg. See [RECOVER_FOV.md](RECOVER_FOV.md) for more info on recovering the FOV of a video. If the video has paralax you can run sam_track_video.py with --optimize_intrinsic and it will give you a accurate FOV.

```
./install_mvdtoolbox.sh -megasam #takes a long time to install
python sam_track_video.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 40


The result is a transformations file  ~/in_office_720p.mp4_depth.mkv_transformations.json
and two debug videos file called _megasam.mkv
```


## Step 5
Triangulate points to get acurate depth readings and realigin the metric depth video to fit the more accurate depth readings.

```
python3.11 convert_metric_depth_video_to_other_format.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 40 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --track_file ~/in_office_720p.mp4_tracking.json --mask_video ~/in_office_720p.mp4_mask.mkv --show_scene_point_clouds --use_triangulated_points --tringulation_min_observations 20 --save_rescaled_depth --show_both_point_clouds --global_align

The result is a rescaled depth video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv
And two .ply files with point cloud data for the scene. One ply file with tirangualted points and one with averages of the depth map called in_office_720p.mp4_depth.mkv_avgmonodepth.ply, in_office_720p.mp4_depth.mkv_triangulated.ply.

You can run the script again with the new _rescled.mkv file to get a rescaled version of the _avgmonodepth.ply file.
python3.11 convert_metric_depth_video_to_other_format.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv --yfov 40 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --track_file ~/in_office_720p.mp4_tracking.json --mask_video ~/in_office_720p.mp4_mask.mkv --show_scene_point_clouds

```


## Step  6
View the result where the two subjects are walking throgh a point cloud.
Camera movment has been canceled out, edges removed, a background .ply file inserted and we have added visulisation for the camera view-frustrum.
Finally we use the mask video to mask out the bakground so we only see the point cloud.

```
python3.11 3d_view_depthfile.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv --yfov 40 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --background_ply ~/in_office_720p.mp4_depth.mkv_avgmonodepth.ply --remove_edges --show_camera
--x -0.1 --y 0 --z -3 --mask_video ~/in_office_720p.mp4_mask.mkv --invert_mask --background_ply ~/in_office_720p.mp4_depth.mkv_rescaled.mkv_avgmonodepth.ply

```
<img width="339" alt="in_the_clouds" src="https://github.com/user-attachments/assets/bf0e8edd-c234-4563-ac8e-e434ce76bf13" />
