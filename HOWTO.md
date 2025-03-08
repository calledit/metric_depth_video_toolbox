# Metric depth video toolbox - Usage examples

This guide contains a walkthrogh of how to use the tools in the metric depth video toolbox.

## Part one: generating rescaled metric depth video, camera tracking data and points clouds
Select a video to work with. This should be a clip, preferably less than 6-7 minutes long (due to GPU memmory usage), and there should not be any cuts in the video. The video should preferably have the same zoom level over the hole clip. Due to GPU memmory constraints in Video-Depth-Anything the aspect ratio is best keept under 16:9.
If you want to convert an entire movie split it up and do it scene by scene. There are tools that can cut down a movie to its scenes automatically.

I will use [in_office_720p.mp4](https://github.com/calledit/metric_depth_video_toolbox/releases/download/ExampleFiles/in_office_720p.mp4) with two individuals walking in a hallway obtained from pexels.com
<img width="930" alt="video_in" src="https://github.com/user-attachments/assets/3ff877b1-fa68-4a0c-8089-6ee6d4fccecc" />
### Step 0
On your **LINUX** machine Install metric depth video toolbox:

This will install required packages and download the needed ML models.
```
git clone https://github.com/calledit/metric_depth_video_toolbox
cd metric_depth_video_toolbox

./install_mvdtoolbox.sh
```

### Step 1
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

### Step 2
Generate a mask video from the source video

```
./create_video_mask.sh -install

./create_video_mask.sh ~/in_office_720p.mp4

the result is a black and white mask video ~/in_office_720p.mp4_mask.mkv
```
<img width="929" alt="mask" src="https://github.com/user-attachments/assets/d282bc87-2026-4f9e-9f85-21bfb69dace0" />

### Step 3
Generate tracking points from the source video, more iterations = more points steps_bewtwen_track_init is the numer of frames betwen initation of new tracking points.

```
python track_points_in_video.py --color_video ~/in_office_720p.mp4 --nr_iterations 4 --steps_bewtwen_track_init 30

the result is a tracking file called ~/in_office_720p.mp4_tracking.json
```
Visualised here as tiny dots in the images:
<img width="926" alt="tracking_on_video" src="https://github.com/user-attachments/assets/d3f66daf-25ab-4f05-8067-ffbe6822a595" />

### Step 4
Generate camera transformations from the depth and the source video. We make a guess of 30-50 deg and chose 40 deg. Later analysis showed that the real FOV is something like 42 deg. See [RECOVER_FOV.md](RECOVER_FOV.md) for more info on recovering the FOV of a video. If the video has paralax you can run sam_track_video.py with --optimize_intrinsic and it will give you a accurate FOV.

```
./install_mvdtoolbox.sh -megasam #takes a long time to install
python sam_track_video.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 40


The result is a transformations file  ~/in_office_720p.mp4_depth.mkv_transformations.json
and two debug videos file called _megasam.mkv
```


### Step 5
Triangulate points to get acurate depth readings and realigin the metric depth video to fit the more accurate depth readings.

```
python3.11 convert_metric_depth_video_to_other_format.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv --yfov 40 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --track_file ~/in_office_720p.mp4_tracking.json --mask_video ~/in_office_720p.mp4_mask.mkv --show_scene_point_clouds --use_triangulated_points --tringulation_min_observations 20 --save_rescaled_depth --show_both_point_clouds --global_align

The result is a rescaled depth video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv
And two .ply files with point cloud data for the scene. One ply file with tirangualted points and one with averages of the depth map called in_office_720p.mp4_depth.mkv_avgmonodepth.ply, in_office_720p.mp4_depth.mkv_triangulated.ply.

You can run the script again with the new _rescled.mkv file to get a rescaled version of the _avgmonodepth.ply file.
python3.11 convert_metric_depth_video_to_other_format.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv --yfov 40 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --track_file ~/in_office_720p.mp4_tracking.json --mask_video ~/in_office_720p.mp4_mask.mkv --show_scene_point_clouds

```


### Step  6
View the result where the two subjects are walking throgh a point cloud.
Camera movment has been canceled out, edges removed, a background .ply file inserted and we have added visulisation for the camera view-frustrum.
Finally we use the mask video to mask out the bakground so we only see the point cloud.

```
python3.11 3d_view_depthfile.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv --yfov 40 --transformation_file ~/in_office_720p.mp4_depth.mkv_transformations.json --background_ply ~/in_office_720p.mp4_depth.mkv_avgmonodepth.ply --remove_edges --show_camera
--x -0.1 --y 0 --z -3 --mask_video ~/in_office_720p.mp4_mask.mkv --invert_mask --background_ply ~/in_office_720p.mp4_depth.mkv_rescaled.mkv_avgmonodepth.ply

```
<img width="339" alt="in_the_clouds" src="https://github.com/user-attachments/assets/bf0e8edd-c234-4563-ac8e-e434ce76bf13" />

## Part two: generating side by side stereo video
Now that we have our rescaled depth video we can create stereo video (technically you dont need to rescale the depth video to create stero video but the end result will end up slightly better if you do rescale it first)

### Step 7 

This renders one frame for the right eye then one for the left you can alter the pupillary distance with --pupillary_distance if you want, but the default of 63 mm is more or less industry standard and is good enogh for most people.
We tell stereo_rerender.py to remove all edges as we will use infill to fill them in later, and we add a argument to add create a infill_mask file. If you dont want to add infill, just skip the last two arguments.

```
python3.11 stereo_rerender.py --color_video ~/in_office_720p.mp4 --depth_video ~/in_office_720p.mp4_depth.mkv_rescaled.mkv --yfov 40 --infill_mask --remove_edges
```
Raw side by side stero (black where there is paralax):
<img width="950" alt="sbs" src="https://github.com/user-attachments/assets/672b9703-3215-400e-bb72-54a8f119366a" />

Side stero infill mask. Green, red and blue where there is paralax. Blue represents the edge of a infill area that is furthest from the camera and red is the edge closest to the camera:
<img width="950" alt="sbs" src="https://github.com/user-attachments/assets/19784521-487d-4c13-9f04-a054a14e287c" />

### Step 8
Here we will use ML to add paralax infill using the tool stereo_crafter_infill.py
Stereocrafter is based on stable defusion so is very slow, be patient. On a 3090 around 0.2 fps have been observed. That is a minute of video recorded at 30 fps will take 2.5 hours to process. Think of it as a proccess you run over night.

```
./install_mvdtoolbox.sh -stereocrafter #downloads and installs stereocrafter in the right folder

python3.11 stereo_crafter_infill.py --sbs_color_video ~/dancing_crop.mp4_depth.mkv_rescaled.mkv_stereo.mkv --sbs_mask_video ~/dancing_crop.mp4_depth.mkv_rescaled.mkv_stereo.mkv_infillmask.mkv
```
The result should be a video file named:
~/dancing_crop.mp4_depth.mkv_rescaled.mkv_stereo.mkv_infilled.mkv


### Final step compress and add back audio
Here we use ffmpeg to extract the original audio and add it back in the video as well as compressing the large uncompressed video file in to a video format/size that a modern VR headset or other stereo capable device can handle.
```
#Extract audio as a wave file
ffmpeg -i ~/in_office_720p.mp4 ~/in_office_720p.wav


#Compress video for viewing on otehr devices and add back audio
ffmpeg -i ~/dancing_crop.mp4_depth.mkv_rescaled.mkv_stereo.mkv_infilled.mkv -i ~/in_office_720p.wav -c:v libx265 -crf 18 -tag:v hvc1 -pix_fmt yuv420p -c:a aac -map 0:v:0 -map 1:a:0 ~/dancing_crop_final_stero.mp4
```




