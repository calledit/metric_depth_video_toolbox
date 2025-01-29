#!/bin/bash

#pip3.11 install rembg

COLOR_VIDEO=$1

VIDEO_WIDTH=1440
VIDEO_HEIGHT=1080
VIDEO_FPS=25

ffmpeg -i COLOR_VIDEO -ss 1 -an -f rawvideo -pix_fmt rgb24 pipe:1 | rembg b ${VIDEO_WIDTH} ${VIDEO_HEIGHT} | ffmpeg -y -f rawvideo -pix_fmt rgb24 -s ${VIDEO_WIDTH}x${VIDEO_HEIGHT} -r ${VIDEO_FPS} -i pipe:0 -preset fast mask.mp4