#!/bin/bash

#pip3.11 install rembg

COLOR_VIDEO=$1

VIDEO_WIDTH=1440
VIDEO_HEIGHT=1080
VIDEO_FPS=25

ffmpeg -i ${COLOR_VIDEO} -an -f rawvideo -pix_fmt rgb24 pipe:1 | rembg b ${VIDEO_WIDTH} ${VIDEO_HEIGHT} | ffmpeg -f image2pipe -framerate ${VIDEO_FPS} -vcodec png -i pipe:0 mask.mp4