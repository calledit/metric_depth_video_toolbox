#!/bin/bash

VIDEO=$1

FOV=40
echo testing xfov: ${FOV}
mv "${VIDEO}" "${VIDEO}_fov_${FOV}.mp4"
python unidepth_video.py --color_video "${VIDEO}_fov_${FOV}.mp4" --xfov ${FOV} --max_len 100
mv "${VIDEO}_fov_${FOV}.mp4" "${VIDEO}"

FOV=45
echo testing xfov: ${FOV}
mv "${VIDEO}" "${VIDEO}_fov_${FOV}.mp4"
python unidepth_video.py --color_video "${VIDEO}_fov_${FOV}.mp4" --xfov ${FOV} --max_len 100
mv "${VIDEO}_fov_${FOV}.mp4" "${VIDEO}"

FOV=50
mv "${VIDEO}" "${VIDEO}_fov_${FOV}.mp4"
python unidepth_video.py --color_video "${VIDEO}_fov_${FOV}.mp4" --xfov ${FOV} --max_len 100
mv "${VIDEO}_fov_${FOV}.mp4" "${VIDEO}"


FOV=55
echo testing xfov: ${FOV}
mv "${VIDEO}" "${VIDEO}_fov_${FOV}.mp4"
python unidepth_video.py --color_video "${VIDEO}_fov_${FOV}.mp4" --xfov ${FOV} --max_len 100
mv "${VIDEO}_fov_${FOV}.mp4" "${VIDEO}"

FOV=60
echo testing xfov: ${FOV}
mv "${VIDEO}" "${VIDEO}_fov_${FOV}.mp4"
python unidepth_video.py --color_video "${VIDEO}_fov_${FOV}.mp4" --xfov ${FOV} --max_len 100
mv "${VIDEO}_fov_${FOV}.mp4" "${VIDEO}"

FOV=65
echo testing xfov: ${FOV}
mv "${VIDEO}" "${VIDEO}_fov_${FOV}.mp4"
python unidepth_video.py --color_video "${VIDEO}_fov_${FOV}.mp4" --xfov ${FOV} --max_len 100
mv "${VIDEO}_fov_${FOV}.mp4" "${VIDEO}"

FOV=75
echo testing xfov: ${FOV}
mv "${VIDEO}" "${VIDEO}_fov_${FOV}.mp4"
python unidepth_video.py --color_video "${VIDEO}_fov_${FOV}.mp4" --xfov ${FOV} --max_len 100
mv "${VIDEO}_fov_${FOV}.mp4" "${VIDEO}"