#!/bin/bash

#apt-get install parallel ffmpeg
#pip install iopaint

INPUT_video=$1

echo Creating image files from video
mkdir data/
mkdir data/imgs_org/
mkdir data/imgs/


#By adding -vf "scale=720:540" you can scale here
ffmpeg -i $INPUT_video data/imgs_org/%07d.png

# Extract raw frame rate as a fraction (e.g. "30000/1001"):
FPS_FRACTION=$(ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=avg_frame_rate \
  -of csv=p=0 \
  "$INPUT_video")

if [ ! -f "overlay_mask.png" ]; then
    echo "File overlay_mask.png does not exist."
fi

echo moving image files to separate folders for paralization 

#How many CPUs to use (you may run out of cuda memmory if to high)
NR_CPUs=8

# Define source folder
SOURCE="data/imgs_org"

# Get total number of files
TOTAL_FILES=$(find "$SOURCE" -maxdepth 1 -type f | wc -l)

# Number of folders
NUM_FOLDERS=100

# Calculate files per folder (some folders might have one extra file)
FILES_PER_FOLDER=$((TOTAL_FILES / NUM_FOLDERS))
EXTRA_FILES=$((TOTAL_FILES % NUM_FOLDERS))

# Create folders
mkdir -p split_folders
cd split_folders
mkdir -p folder_{1..101}

# Distribute files
i=1
folder_num=1
for file in "../$SOURCE"/*; do
  mv "$file" "folder_$folder_num/"

  # Switch to the next folder when enough files are moved
  if (( i % FILES_PER_FOLDER == 0 )); then
    if (( folder_num <= EXTRA_FILES )); then
      # Distribute the extra files evenly among the first few folders
      ((i++))
    fi
    ((folder_num++))
  fi

  ((i++))
done
cd ..


#Run iopaint once to predownload the model
iopaint run --model=lama --device cuda --image /dev/zero --mask /dev/zero --output /dev/null

echo doing inpainting
#migan is faster but to slow
find split_folders -mindepth 1 -maxdepth 1 -type d | parallel -j $NR_CPUs 'iopaint run --model=lama --device cuda --image {} --mask overlay_mask.png --output data/imgs/'

echo recombining inpainted images to a video

ffmpeg -framerate $FPS_FRACTION -i data/imgs/%07d.png -i "$INPUT_video" -map 0:v:0 -map 1:a? -c:v ffv1 -pix_fmt bgra -level 3 -g 1 -slices 16 -slicecrc 1 -c:a copy data/preprocessed_video.mkv
