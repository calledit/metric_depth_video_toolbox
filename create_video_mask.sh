#!/bin/bash

if [[ " $@ " =~ " -install " ]]; then
	
	apt-get -y install ffmpeg
	pip install rembg[gpu] filetype watchdog aiohttp asyncer gradio

	#rembg[gpu] needs cudnn
	wget https://developer.download.nvidia.com/compute/cudnn/9.7.0/local_installers/cudnn-local-repo-ubuntu2404-9.7.0_1.0-1_amd64.deb
	dpkg -i cudnn-local-repo-ubuntu2404-9.7.0_1.0-1_amd64.deb
	cp /var/cudnn-local-repo-ubuntu2404-9.7.0/cudnn-*-keyring.gpg /usr/share/keyrings/
	apt-get update
	apt-get -y install cudnn
	echo requirments are installed you can now run without -install
	exit
fi

COLOR_VIDEO=$1

MASK_VIDEO="${COLOR_VIDEO}_mask.mp4"

# Extract width:
VIDEO_WIDTH=$(ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width \
  -of csv=p=0 \
  "$COLOR_VIDEO")

# Extract height:
VIDEO_HEIGHT=$(ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=height \
  -of csv=p=0 \
  "$COLOR_VIDEO")

# Extract raw frame rate as a fraction (e.g. "30000/1001"):
FPS_FRACTION=$(ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=avg_frame_rate \
  -of csv=p=0 \
  "$COLOR_VIDEO")
	  
OUTPUTVIDEO="${COLOR_VIDEO}_mask.mkv"

ffmpeg -i ${COLOR_VIDEO} -an -f rawvideo -pix_fmt rgb24 pipe:1 | rembg b -om ${VIDEO_WIDTH} ${VIDEO_HEIGHT} | ffmpeg -y -f image2pipe -framerate ${FPS_FRACTION} -vcodec png -i pipe:0 -c:v libx265 -crf 23 -pix_fmt yuv420p "${OUTPUTVIDEO}"