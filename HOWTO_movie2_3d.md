# Metric depth video toolbox - movie_2_3D.py

This guide contains a walkthrogh of how to use the Metric depth video toolbox tool movie_2_3D.py to automatically
convert a full movie into side by side 3d stereo video.

## Part zero: install
On your linux machine with a graphics card that has 24Gb of memory or more.
*Things may work on cards that have less memory but that has not been tested. The metric depth video toolbox is built and tested with nvidia 3090s rented on vast.ai*

Make sure you have alot of free harddrive space we will be working with very large high
resolutuion lossless video files. A 1080p video may require as much as 10GB per minute of video.
Most of these files can be thrown away once they have been used and you are done, but while you are working you will need allot of diskspace.


Installing:
```
git clone https://github.com/calledit/metric_depth_video_toolbox
cd metric_depth_video_toolbox

# if using headless linux you need to start a virtual x11 server
apt-get install xvfb
Xvfb :2 &
export DISPLAY=:2

apt install ffmpeg
./install_mvdtoolbox.sh
./install_mvdtoolbox.sh -stereocrafter
./create_video_mask.sh -install

pip install --upgrade scenedetect[opencv-headless]

```


## Part one: generating scene information


### step 1
Select a video to work with, in this example we will work with the 1997 movie Starship Troopers.

```
ls -lah Starship.Troopers.mp4
mkdir scenes
cd scenes
#Create a index of all scenes
scenedetect -i ../Starship.Troopers.mp4 list-scenes save-html

#make images smaler
echo '<style>img {width: 200px;}</style>' >> Starship.Troopers-Scenes.html

```

We can now open the Starship.Troopers-Scenes.html and see if the scene detection worked.

It should look something like this:
<img alt="scenes" src="https://github.com/user-attachments/assets/b7452d28-4745-42e2-bf94-1dee992a9711" />

### step 2 
*This step is optional*

At this point we might want to look at the Scenes.html file and decide what scenes are credits or other visuals that are not pure camera shots.
The reason we are interested in this is that the metric depth video toolbox supports multiple depth models (As of writing this depthcrafter, geometrycrafter and video-depth-anything is supported). Video-depth-anything is faster and better at most normal camera shots, whereas depthcrafter is better at handling things like credits or other visuals that are not filmed with a camera.

To select which scenes you want to use depthcrafter for you will need to open the Starship.Troopers-Scenes.csv file and ad a
new column named "Engine". For the scenes that you want to use depthcrafter for you simply write "depthcrafter", "geometrycrafter", or "vda" in the cell.

When you are done save the CSV file.


## Part two: converting the movie

This is mostly a waiting game you can run a few test scenes by using the argument --end_scene 4 or you can start directly. Progress is saved continuously (But it you abort the process the last scene that was left unfinished will need to be removed and be redone)

```
python3.11 movie_2_3D.py --scene_file ~/Starship.Troopers-Scenes.csv --color_video ~/Starship.Troopers.mp4 
```

When this is all done you will have video scene files for all scenes and two final result files:
- a Starship.Troopers.mp4_stereo3d.mp4 that can be sent to most consumer electronics.
- a Starship.Troopers.mp4_stereo3d.mkv which is a lossless video file.

The result of the first 3 minutes of the example can be seen here:
https://www.youtube.com/watch?v=NzI8Js6aYiI


## Making it faster
Rendering can be made faster by running it in parralel, to do that 3 passes must be done
```
echo Pass one, only do depth estimation and convergence calculations
python3.11 movie_2_3D.py --scene_file ~/Starship.Troopers-Scenes.csv --color_video ~/Starship.Troopers.mp4 --no_render --end_scene 20

echo Pass two, parallel run 11 rendering proccesses in parallel
python3.11 movie_2_3D.py --scene_file ~/Starship.Troopers-Scenes.csv --color_video ~/Starship.Troopers.mp4 --parallel 11 --end_scene 20

echo Pass three, do paralax infill
python3.11 movie_2_3D.py --scene_file ~/Starship.Troopers-Scenes.csv --color_video ~/Starship.Troopers.mp4 --end_scene 20
```
