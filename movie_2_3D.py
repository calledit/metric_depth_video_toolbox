import subprocess
import argparse
import csv
import cv2
import os

def write_frames_to_file(input_video, nr_frames_to_copy, scene_video_file, frame_rate, frame_width, frame_height):
    
    out = cv2.VideoWriter(scene_video_file, cv2.VideoWriter_fourcc(*"FFV1"), frame_rate, (frame_width, frame_height))
    nr_frames = 1
    while input_video.isOpened():
        ret, raw_frame = input_video.read()
        if not ret:
            break
        
        out.write(raw_frame)
        if nr_frames >= nr_frames_to_copy:
            break
        nr_frames += 1
    
    out.release()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Takes a movie and converts it in to stereo 3D')
    
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=True)
    parser.add_argument('--xfov', type=float, default=42.0, help='Default camera field of view in x direction', required=False)

    parser.add_argument('--scene_file', type=str, help='csv from PySceneDetect describing the scenes', required=True)
    parser.add_argument('--csv_delimiter', type=str, default=',', help='Delimiter used in csv', required=False)
    
    parser.add_argument('--output_dir', type=str, default='output', help='folder where output will be placed', required=False)

    parser.add_argument('--end_scene', type=int, default=-1, help='Stop after a certain scene nr', required=False)
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    raw_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

    scenes = []
    with open(args.scene_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the first row it contains garbage timestamp data
        dict_reader = csv.DictReader(csvfile, delimiter=args.csv_delimiter)
        for row in dict_reader:
            scenes.append(row)

    video_files_to_concat = []
    
    for scene in scenes:
        print("Handle scene:", scene)
        depth_engine = 'vda'
        if 'Engine' in scene and scene['Engine'] != '':
            depth_engine = scene['Engine']
        
        
        scene_xfov = str(args.xfov)
        if 'Xfov' in scene and scene['Xfov'] != '':
            scene_xfov = scene['Xfov']
        
        #generate scene video file
        scene_video_file = os.path.join(args.output_dir, 'scene_'+str(scene['Scene Number'])+'.mkv')
        if False:#dont write new files when experimenting
            write_frames_to_file(raw_video, int(scene['Length (frames)']), scene_video_file, frame_rate, frame_width, frame_height)
        
        #Generate scene depth file
        if depth_engine == 'depthcrafter':
            #to use depth crafter we first need a metric reference. We use moge as it is the most robust metric depth model avalibe right now
            subprocess.run("python moge_video.py --output_dir moge_output --color_video "+scene_video_file, shell=True)
            single_frame_depth_video_file = scene_video_file + "_depth.mkv"
            
            subprocess.run("python depthcrafter_video.py --color_video "+scene_video_file+" --depth_file "+single_frame_depth_video_file, shell=True)
            scene_depth_video_file = scene_video_file + "_depthcrafter_depth.mkv"
        else:
            subprocess.run("python video_metric_convert.py --color_video "+scene_video_file, shell=True)
            scene_depth_video_file = scene_video_file + "_depth.mkv"
        
        #Generate stereo 3d video and infill mask
        subprocess.run("python stereo_rerender.py --color_video "+scene_video_file+" --xfov "+scene_xfov+" --depth_video "+scene_depth_video_file+" --infill_mask", shell=True)
        scene_sbs = scene_depth_video_file + "_stereo.mkv"
        scene_sbs_infill = scene_depth_video_file + "_infillmask.mkv"
        
        #Do infill
        subprocess.run("python stereo_crafter_infill.py --sbs_color_video "+scene_sbs+" --sbs_mask_video "+scene_sbs_infill, shell=True)
        scene_infilled = scene_sbs+"_infilled.mkv"
        
        video_files_to_concat.append(scene_infilled)
        
        if args.end_scene == int(scene['Scene Number']):
            break
            
    
    # Write the ffmpeg concat file
    ffmpeg_concat_file = 'ffmpeg_concat.txt'
    with open(ffmpeg_concat_file, 'w') as f:
        for video in video_files_to_concat:
            if not os.path.exists(video):
                print("ERROR scene file is missing")
            f.write(f"file '{video}'\n")
    
    video_name = os.path.basename(args.color_video)
    result_video_file = args.output_dir+os.sep+video_name+"_stereo3d.mkv"
    
    #Use ffmpeg to join all scenes and add back original audio
    subprocess.run("ffmpeg -f concat -safe 0 -i "+ffmpeg_concat_file+" -i "+args.color_video+" -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -shortest "+result_video_file, shell=True)
    
    