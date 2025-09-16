import subprocess
import argparse
import csv
import cv2
import time
import os
import json
import numpy as np
from pathlib import Path

def write_frames_to_file(input_video, nr_frames_to_copy, scene_video_file, frame_rate, frame_width, frame_height):

    out = None
    if not os.path.exists(scene_video_file):
        out = cv2.VideoWriter(scene_video_file, cv2.VideoWriter_fourcc(*"FFV1"), frame_rate, (frame_width, frame_height))
    nr_frames = 1
    while input_video.isOpened():
        ret, raw_frame = input_video.read()
        if not ret:
            break

        if out is not None:
            out.write(raw_frame)
        if nr_frames >= nr_frames_to_copy:
            break
        nr_frames += 1

    if out is not None:
        out.release()

def wait_for_first(processes):
    """
    Wait for any process in the given list to finish.

    Parameters:
        processes (list): A list of subprocess.Popen objects.

    Returns: list new_processes
    """
    if not processes:
        return []

    while True:
        for i, p in enumerate(processes):
            if p.poll() is not None:  # Process has finished
                finished = p
                new_processes = processes[:i] + processes[i+1:]
                return new_processes
        time.sleep(0.1)  # Sleep briefly to avoid busy waiting

def is_valid_video(file_path):
    """
    Returns True if the file exists and its size is at least 2Kb (2048 bytes),
    otherwise returns False.
    """
    return os.path.exists(file_path) and os.path.getsize(file_path) >= 2048

if __name__ == '__main__':

    python = "python"

    parser = argparse.ArgumentParser(description='Takes a movie and converts it in to stereo 3D')

    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=True)

    parser.add_argument('--scene_file', type=str, default=None, help='csv from PySceneDetect describing the scenes', required=False)
    parser.add_argument('--csv_delimiter', type=str, default=',', help='Delimiter used in csv', required=False)

    parser.add_argument('--output_dir', type=str, default='output', help='folder where output will be placed', required=False)

    parser.add_argument('--end_scene', type=int, default=-1, help='Stop after a certain scene nr', required=False)

    parser.add_argument('--no_render', action='store_true', help='Skip rendering and subseqvent steps.', required=False)
    parser.add_argument('--parallel', type=int, default=1, help='Run some steps in parallel, for faster processing.')
    parser.add_argument('--no_infill', action='store_true', help='Dont do infill.', required=False)



    args = parser.parse_args()

    skip_last_step = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    #if the user did not specify a scene file we create one
    if args.scene_file is None:
        p = Path(args.color_video)
        args.scene_file = p.with_name(p.stem + "-Scenes.csv")
        if not os.path.exists(args.scene_file):
            subprocess.run("scenedetect -i "+args.color_video+" list-scenes", shell=True)

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

    parallels = []
    
    print("Step one: create video files for all scenes")
    scene_video_files = []
    for scene in scenes:
        print("Handle scene:", scene)


        #generate scene video file
        scene['scene_video_file'] = os.path.join(args.output_dir, 'scene_'+str(scene['Scene Number'])+'.mkv')
        
        if not os.path.exists(scene['scene_video_file']):
            write_frames_to_file(raw_video, int(scene['Length (frames)']), scene['scene_video_file'], frame_rate, frame_width, frame_height)
        scene_video_files.append(scene)
        if args.end_scene == int(scene['Scene Number']):
            break
    
        
    print("Step two: estimate depth for all scenes")
    
    batch_file = args.color_video+'_batching.txt'
    if os.path.exists(batch_file):
        os.remove(batch_file)
        
    for scene in scene_video_files:
        scene['depth_video_file'] = scene['scene_video_file'] + "_depth.mkv"
        scene['xfovs_file'] = scene['depth_video_file'] + "_xfovs.json"
        scene_org_xfovs_file = scene['depth_video_file'] + "_org_xfovs.json"
        single_frame_depth_video_file = scene['scene_video_file'] + "_single_frame_depth.mkv"
        
        
        depth_engine = 'vda'
        if 'Engine' in scene and scene['Engine'] != '':
            depth_engine = scene['Engine']
        

        scene['xfov'] = None
        #Get Field of view
        #If we use engine that is not VDA or depthcrafter we need to esitmate accurate FOV and a proper metric reference depth video so we use unik3d for that
        if depth_engine != 'vda' and depth_engine != 'depthcrafter':
            if not os.path.exists(single_frame_depth_video_file):
                if not os.path.exists(scene_org_xfovs_file):
                    subprocess.run(python+" unik3d_video.py --color_video "+scene['scene_video_file'], shell=True)
                    subprocess.run("mv "+scene['xfovs_file']+" "+scene_org_xfovs_file, shell=True)

                with open(scene_org_xfovs_file) as json_file_handle:
                    xfovs = json.load(json_file_handle)
                    scene['xfov'] = np.mean(xfovs)
                subprocess.run(python+" unik3d_video.py --xfov "+str(scene['xfov'])+" --color_video "+scene['scene_video_file'], shell=True)
                subprocess.run("mv "+scene['depth_video_file']+" "+single_frame_depth_video_file, shell=True)

            assert is_valid_video(single_frame_depth_video_file), "Could not generate metric reference video file for depthcrafter"
        else:
            scene['xfov'] = 42.0
        
        #VDA is batchable so gets its own flow
        if depth_engine == 'vda':
            if not os.path.exists(scene['depth_video_file']):
                with open(batch_file, "a", encoding="utf-8") as f:
                    f.write(scene['scene_video_file']+"\n")
        else:
            if depth_engine == 'depthcrafter':
                if not os.path.exists(scene['depth_video_file']):
                    subprocess.run(python+" depthcrafter_video.py --color_video "+scene['scene_video_file']+" --depth_video "+single_frame_depth_video_file, shell=True)
            else:#geometrycrafter
                if not os.path.exists(scene['depth_video_file']):
                    #+" --depth_video "+single_frame_depth_video_file+" --xfov_file "+scene['xfovs_file']
                    subprocess.run(python+" geometrycrafter_video.py --color_video "+scene['scene_video_file']+" --depth_video "+single_frame_depth_video_file+" --xfov_file "+scene['xfovs_file'], shell=True)
                    #subprocess.run(python+" geometrycrafter_video.py --color_video "+scene['scene_video_file'], shell=True)

            assert is_valid_video(scene['depth_video_file']), "Could not generate: "+scene['depth_video_file']
    
    if os.path.exists(batch_file):
        subprocess.run(python+" video_metric_convert.py --color_video "+batch_file, shell=True)
        os.remove(batch_file)
    
    
    print("Step three: generate masks for focus point")
    for scene in scene_video_files:
        scene['mask_video_file'] = scene['scene_video_file'] + "_mask.mkv"
        if not os.path.exists(scene['mask_video_file']):
            with open(batch_file, "a", encoding="utf-8") as f:
                f.write(scene['scene_video_file']+"\n")
        
    if os.path.exists(batch_file):
        subprocess.run(python+" generate_video_mask.py --color_video "+batch_file, shell=True)
        os.remove(batch_file)
    
    
    print("Step four: find convergence depth for focus point")
    for scene in scene_video_files:
        assert is_valid_video(scene['mask_video_file']), "Could find valid mask video: "+scene['mask_video_file']
        scene['convergence_file'] = scene['depth_video_file'] + "_convergence_depths.json"
        if not os.path.exists(scene['convergence_file']):
            subprocess.run(python+" find_convergence_depth.py --depth_video "+scene['depth_video_file']+" --mask_video "+scene['mask_video_file'], shell=True)
    
    if args.no_render:
        exit(0)
    
    
    
    
    print("Step five: render SBS frames")
    for scene in scene_video_files:
        infill = True
        if 'Infill' in scene and scene['Infill'] == 'No':
            infill = False
        
        #Generate stereo 3d video and infill mask
        scene['sbs'] = scene['depth_video_file'] + "_stereo.mkv"
        scene['sbs_infill'] = scene['sbs'] + "_infillmask.mkv"
        if not os.path.exists(scene['sbs']):
            if scene['xfov'] is not None:
                xfov_str = "--xfov "+str(scene['xfov'])
            else:
                xfov_str = "--xfov_file "+scene['xfovs_file']

            infm = ''
            if infill:
                infm = '--infill_mask'

            parallels.append(subprocess.Popen(python+" stereo_rerender.py --green_and_black_infill_mask --color_video "+scene['scene_video_file']+" --convergence_file "+scene['convergence_file']+" "+xfov_str+" --depth_video "+scene['depth_video_file']+" "+infm, shell=True))

        if len(parallels) >= args.parallel:
            parallels = wait_for_first(parallels)
    for proc in parallels:
        proc.wait()
    
    print("Step six: do SBS infill")
    for scene in scene_video_files:

        assert is_valid_video(scene['sbs']), "Could not find proper stereo video file to so infill on"+scene['sbs']

        if args.no_infill or not infill:
            video_files_to_concat.append(scene['sbs'])
        else:

            #Do infill
            scene['infilled'] = scene['sbs']+"_infilled.mkv"
            if not os.path.exists(scene['infilled']) and not skip_last_step:
               subprocess.run(python+" stereo_crafter_infill.py --sbs_color_video "+scene['sbs']+" --sbs_mask_video "+scene['sbs_infill'], shell=True)

            assert is_valid_video(scene['infilled']), "Could not generate infilled stereo video file"

            video_files_to_concat.append(scene['infilled'])



    if skip_last_step or args.no_render:
        exit(0)

    # Write the ffmpeg concat file
    ffmpeg_concat_file = 'ffmpeg_concat.txt'
    with open(ffmpeg_concat_file, 'w') as f:
        for video in video_files_to_concat:
            if not os.path.exists(video):
                print("ERROR scene file is missing")
            f.write(f"file '{video}'\n")

    video_name = os.path.basename(args.color_video)
    mp4_result_video_file = args.output_dir+os.sep+video_name+"_SBS.mp4"

    #Use ffmpeg to join all scenes and add back original audio
    ret = subprocess.run(
        "ffmpeg -y -f concat -safe 0 -i "
        + ffmpeg_concat_file
        + " -i "
        + args.color_video
        + " -map 0:v:0 -map 1:a:0 "
        + "-c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p "
        + "-c:a copy  -shortest "
        + mp4_result_video_file,
        shell=True
    )
    
    #if ffmpeg fails we asumme it is cause the original audiocodec was not aac so we rencode the audio
    if ret.returncode != 0:
        subprocess.run(
        "ffmpeg -y -f concat -safe 0 -i "
        + ffmpeg_concat_file
        + " -i "
        + args.color_video
        + " -map 0:v:0 -map 1:a:0 "
        + "-c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p "
        + "-c:a aac -shortest "
        + mp4_result_video_file,
        shell=True
    )