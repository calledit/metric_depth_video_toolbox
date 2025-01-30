#!/bin/bash

#pip3.11 install rembg

COLOR_VIDEO=$1

VIDEO_WIDTH=1440
VIDEO_HEIGHT=1080
VIDEO_FPS=25

ffmpeg -i ${COLOR_VIDEO} -ss 1 -an -f rawvideo -pix_fmt rgb24 pipe:1 | rembg b -om ${VIDEO_WIDTH} ${VIDEO_HEIGHT} | ffmpeg -y -f image2pipe -framerate ${VIDEO_FPS} -vcodec png -i pipe:0 -c:v libx265 -crf 23 -pix_fmt yuv420p mask.mp4