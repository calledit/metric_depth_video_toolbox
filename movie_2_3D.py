import subprocess
import argparse
import csv
import cv2
import time
import os
import json
import numpy as np
from pathlib import Path
from math import floor
from typing import List, Dict, Tuple, Optional

import depth_frames_helper


# -------------------------
# Utility helpers (as-is)
# -------------------------

def write_frames_to_file(input_video, nr_frames_to_copy, scene_video_file, frame_rate, frame_width, frame_height):
    out = None
    if scene_video_file is not None:
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


def validate_video_lengths(scene_video_files):
    incorrect_files = []

    for scene in scene_video_files:
        video_path = scene['infilled']
        expected_frames = int(scene['Length (frames)'])

        # Check if file exists
        if not os.path.isfile(video_path):
            print(f"❌ File does not exist: {video_path}")
            incorrect_files.append((video_path, "File not found"))
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open file: {video_path}")
            incorrect_files.append((video_path, "Could not open"))
            continue

        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if actual_frames != expected_frames:
            print(f"⚠️ Mismatch in {video_path}: expected {expected_frames}, got {actual_frames}")
            incorrect_files.append((video_path, f"Expected {expected_frames}, got {actual_frames}"))

    if incorrect_files:
        print(f"Some files had issues delete them and run again: {incorrect_files}")
        return False

    return True


def _seconds_to_timecode(seconds: float) -> str:
    ms = round(seconds * 1000)
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def split_scenes(scenes, max_scene_frames: int = 1500):
    """
    Split scenes longer than max_scene_frames while preserving any extra keys
    present in the input dictionaries. Renumbers 'Scene Number' consecutively.

    Returns a new list of dicts.
    """
    out = []

    for scene in scenes:
        # Parse needed numeric fields (input may be strings)
        sf = int(scene['Start Frame'])
        ef = int(scene['End Frame'])
        ss = float(scene['Start Time (seconds)'])
        es = float(scene['End Time (seconds)'])
        length_frames = ef - sf + 1

        # Seconds per frame (constant-rate assumption)
        spf = (es - ss) / (ef - sf) if ef != sf else 0.0

        def make_chunk_dict(chunk_sf, chunk_ef):
            chunk_ss = ss + (chunk_sf - sf) * spf
            chunk_es = ss + (chunk_ef - sf) * spf
            chunk_len = chunk_ef - chunk_sf + 1
            # Start with a copy to preserve unknown/extra keys
            d = scene.copy()
            # Overwrite computed fields
            d['Scene Number'] = None  # will be filled later
            d['Start Frame'] = str(chunk_sf)
            d['Start Time (seconds)'] = f"{chunk_ss:.3f}"
            d['Start Timecode'] = _seconds_to_timecode(chunk_ss)
            d['End Frame'] = str(chunk_ef)
            d['End Time (seconds)'] = f"{chunk_es:.3f}"
            d['End Timecode'] = _seconds_to_timecode(chunk_es)
            d['Length (frames)'] = str(chunk_len)
            d['Length (seconds)'] = f"{max(0.0, chunk_es - chunk_ss):.3f}"
            d['Length (timecode)'] = _seconds_to_timecode(max(0.0, chunk_es - chunk_ss))
            return d

        if length_frames <= 0:
            # Keep as-is (but still preserve/normalize fields)
            out.append(make_chunk_dict(sf, ef))
            continue

        if length_frames <= max_scene_frames:
            out.append(make_chunk_dict(sf, ef))
            continue

        # Split into chunks
        remaining = length_frames
        chunk_start = sf
        while remaining > 0:
            chunk_len = min(remaining, max_scene_frames)
            chunk_end = chunk_start + chunk_len - 1
            out.append(make_chunk_dict(chunk_start, chunk_end))
            remaining -= chunk_len
            chunk_start = chunk_end + 1

    # Renumber consecutively
    for i, d in enumerate(out, start=1):
        d['Scene Number'] = str(i)

    return out


# -------------------------
# New step-wise functions
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Takes a movie and converts it in to stereo 3D')
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=True)
    parser.add_argument('--scene_file', type=str, default=None, help='csv from PySceneDetect describing the scenes', required=False)
    parser.add_argument('--csv_delimiter', type=str, default=',', help='Delimiter used in csv', required=False)
    parser.add_argument('--output_dir', type=str, default='output', help='folder where output will be placed', required=False)
    parser.add_argument('--end_scene', type=int, default=-1, help='Stop after a certain scene nr', required=False)
    parser.add_argument('--no_render', action='store_true', help='Skip rendering and subseqvent steps.', required=False)
    parser.add_argument('--parallel', type=int, default=1, help='Run some steps in parallel, for faster processing.')
    parser.add_argument('--max_scene_frames', type=int, default=1500, help='Max length of scene in nr of frames, longer scenes will be processed in chunks.')
    parser.add_argument('--no_infill', action='store_true', help='Dont do infill.', required=False)
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_scene_file(args: argparse.Namespace) -> None:
    """
    If the user did not provide a scene file, create one with PySceneDetect and move it
    into the output directory.
    """
    if args.scene_file is not None:
        return

    video_name = os.path.splitext(os.path.basename(args.color_video))[0]
    scene_file = video_name + "-Scenes.csv"
    args.scene_file = os.path.join(args.output_dir, scene_file)
    if not os.path.exists(args.scene_file):
        subprocess.run(f"scenedetect -i {args.color_video} list-scenes", shell=True)
        subprocess.run(f"mv {scene_file} {args.scene_file}", shell=True)


def open_input_video(color_video_path: str):
    cap = cv2.VideoCapture(color_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    return cap, frame_width, frame_height, frame_rate


def load_and_split_scenes(scene_csv_path: str, csv_delimiter: str, max_scene_frames: int) -> List[Dict]:
    scenes = []
    with open(scene_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the first row; it contains garbage timestamp data
        dict_reader = csv.DictReader(csvfile, delimiter=csv_delimiter)
        for row in dict_reader:
            scenes.append(row)
    return split_scenes(scenes, max_scene_frames=max_scene_frames)


def plan_scene_files(scenes: List[Dict], output_dir: str, end_scene: int) -> List[Dict]:
    """
    Prepare per-scene output paths and flags. Returns the list (possibly truncated by end_scene).
    """
    scene_video_files = []
    any_left = False

    for scene in scenes:
        scene_num = str(scene['Scene Number'])
        base = os.path.join(output_dir, f'scene_{scene_num}.mkv')

        scene['scene_video_file'] = base
        scene['depth_video_file'] = base + "_depth.mkv"
        scene['mask_video_file'] = base + "_mask.mkv"
        scene['xfovs_file'] = scene['depth_video_file'] + "_xfovs.json"
        scene['sbs'] = scene['depth_video_file'] + "_stereo.mkv"
        scene['sbs_infill'] = scene['sbs'] + "_infillmask.mkv"
        scene['infilled'] = scene['sbs'] + "_infilled.mkv"

        # Per-scene infill preference
        scene['infill'] = not ('Infill' in scene and scene['Infill'] == 'No')

        # 'finished' if either stereo or infilled exists
        scene['finished'] = os.path.exists(scene['sbs']) or os.path.exists(scene['infilled'])

        scene_video_files.append(scene)

        if not scene['finished'] and not os.path.exists(scene['scene_video_file']):
            any_left = True

        if end_scene == int(scene['Scene Number']):
            break

    return scene_video_files


def step1_create_scene_videos(raw_video, scene_video_files: List[Dict], frame_rate, frame_width, frame_height) -> None:
    """
    Create per-scene raw clips (FFV1) by copying frames from the full input video.
    """
    print("Step one: create video files for all scenes")

    any_left = any((not s['finished'] and not os.path.exists(s['scene_video_file'])) for s in scene_video_files)
    if not any_left:
        return

    for scene in scene_video_files:
        tmp_file = None
        print("scene:", str(scene['Scene Number']))
        if not scene['finished'] and not os.path.exists(scene['scene_video_file']):
            tmp_file = str(scene['scene_video_file']) + "_tmp.mkv"
            print("create:", str(scene['scene_video_file']))
        write_frames_to_file(raw_video, int(scene['Length (frames)']), tmp_file, frame_rate, frame_width, frame_height)
        if tmp_file is not None:
            depth_frames_helper.verify_and_move(tmp_file, int(scene['Length (frames)']), scene['scene_video_file'])


def step2_estimate_depth(args: argparse.Namespace, scene_video_files: List[Dict]) -> None:
    """
    Estimate depth for all scenes, batching VDA where possible.
    """
    print("Step two: estimate depth for all scenes")

    python = "python"
    batch_file = args.color_video + '_batching.txt'
    # Clean batch placeholder
    if os.path.exists(batch_file):
        os.remove(batch_file)

    for scene in scene_video_files:
        scene_org_xfovs_file = scene['depth_video_file'] + "_org_xfovs.json"
        single_frame_depth_video_file = scene['scene_video_file'] + "_single_frame_depth.mkv"

        depth_engine = 'vda'
        if 'Engine' in scene and scene['Engine'] != '':
            depth_engine = scene['Engine']

        scene['xfov'] = None

        # If engine not vda/depthcrafter, estimate xfov + metric reference via unik3d
        if depth_engine not in ('vda', 'depthcrafter'):
            if not os.path.exists(single_frame_depth_video_file):
                if not os.path.exists(scene_org_xfovs_file):
                    if not scene['finished']:
                        subprocess.run(f"{python} unik3d_video.py --color_video {scene['scene_video_file']}", shell=True)
                        subprocess.run(f"mv {scene['xfovs_file']} {scene_org_xfovs_file}", shell=True)

                with open(scene_org_xfovs_file) as json_file_handle:
                    xfovs = json.load(json_file_handle)
                    scene['xfov'] = float(np.mean(xfovs))
                if not scene['finished']:
                    subprocess.run(f"{python} unik3d_video.py --xfov {scene['xfov']} --color_video {scene['scene_video_file']}", shell=True)
                    subprocess.run(f"mv {scene['depth_video_file']} {single_frame_depth_video_file}", shell=True)

            assert is_valid_video(single_frame_depth_video_file), "Could not generate metric reference video file for depthcrafter"
        else:
            scene['xfov'] = 42.0

        # Generate depth per engine
        if depth_engine == 'vda':
            if not os.path.exists(scene['depth_video_file']):
                if not scene['finished']:
                    with open(batch_file, "a", encoding="utf-8") as f:
                        f.write(scene['scene_video_file'] + "\n")
        else:
            if depth_engine == 'depthcrafter':
                if not os.path.exists(scene['depth_video_file']):
                    if not scene['finished']:
                        subprocess.run(f"{python} depthcrafter_video.py --color_video {scene['scene_video_file']} --depth_video {single_frame_depth_video_file}", shell=True)
            else:  # geometrycrafter
                if not os.path.exists(scene['depth_video_file']):
                    if not scene['finished']:
                        subprocess.run(f"{python} geometrycrafter_video.py --color_video {scene['scene_video_file']} --depth_video {single_frame_depth_video_file} --xfov_file {scene['xfovs_file']}", shell=True)

            assert is_valid_video(scene['depth_video_file']), "Could not generate: " + scene['depth_video_file']

    # Run batch VDA, if any
    if os.path.exists(batch_file):
        subprocess.run(f"{python} video_metric_convert.py --color_video {batch_file}", shell=True)
        os.remove(batch_file)


def step3_generate_masks(args: argparse.Namespace, scene_video_files: List[Dict]) -> None:
    """
    Generate segmentation masks for focus points, batched.
    """
    print("Step three: generate masks for focus point")
    python = "python"
    batch_file = args.color_video + '_batching.txt'
    if os.path.exists(batch_file):
        os.remove(batch_file)

    for scene in scene_video_files:
        if not os.path.exists(scene['mask_video_file']):
            if not scene['finished']:
                with open(batch_file, "a", encoding="utf-8") as f:
                    f.write(scene['scene_video_file'] + "\n")

    if os.path.exists(batch_file):
        subprocess.run(f"{python} generate_video_mask.py --color_video {batch_file}", shell=True)
        os.remove(batch_file)


def step4_find_convergence(scene_video_files: List[Dict]) -> None:
    """
    Compute convergence depths for each scene.
    """
    print("Step four: find convergence depth for focus point")
    python = "python"
    for scene in scene_video_files:
        if not scene['finished']:
            assert is_valid_video(scene['mask_video_file']), "Could find valid mask video: " + scene['mask_video_file']
            scene['convergence_file'] = scene['depth_video_file'] + "_convergence_depths.json"
            if not os.path.exists(scene['convergence_file']):
                subprocess.run(f"{python} find_convergence_depth.py --depth_video {scene['depth_video_file']} --mask_video {scene['mask_video_file']}", shell=True)


def step5_render_sbs(args: argparse.Namespace, scene_video_files: List[Dict]) -> None:
    """
    Render stereo (SBS) frames; parallelize up to args.parallel.
    """
    print("Step five: render SBS frames")
    python = "python"
    parallels = []

    for scene in scene_video_files:
        if not os.path.exists(scene['sbs']):
            # Prefer explicit xfov, else pass xfov_file
            if scene.get('xfov') is not None:
                xfov_str = f"--xfov {scene['xfov']}"
            else:
                xfov_str = f"--xfov_file {scene['xfovs_file']}"

            infm = '--infill_mask' if scene.get('infill', True) else ''

            if not scene['finished']:
                cmd = f"{python} stereo_rerender.py --color_video {scene['scene_video_file']} --convergence_file {scene['convergence_file']} {xfov_str} --depth_video {scene['depth_video_file']} {infm}"
                parallels.append(subprocess.Popen(cmd, shell=True))

            if len(parallels) >= args.parallel:
                parallels = wait_for_first(parallels)

    for proc in parallels:
        proc.wait()


def step6_infill_and_collect(args: argparse.Namespace, scene_video_files: List[Dict]) -> List[str]:
    """
    (Optional) infill SBS, collect final per-scene video paths to join.
    """
    print("Step six: do SBS infill")

    python = "python"
    batch_file = args.color_video + '_batching.txt'
    batch_file2 = args.color_video + '_batching2.txt'
    if os.path.exists(batch_file):
        os.remove(batch_file)
    if os.path.exists(batch_file2):
        os.remove(batch_file2)

    video_files_to_concat = []

    for scene in scene_video_files:
        if not scene['finished']:
            assert is_valid_video(scene['sbs']), "Could not find proper stereo video file to do infill on " + scene['sbs']

        # If infill disabled globally or per-scene, skip infill step
        do_infill = (not args.no_infill) and scene.get('infill', True)
        if not do_infill:
            video_files_to_concat.append(scene['sbs'])
            continue

        # Queue infill work
        if not os.path.exists(scene['infilled']):
            with open(batch_file, "a", encoding="utf-8") as f:
                f.write(scene['sbs'] + "\n")
            with open(batch_file2, "a", encoding="utf-8") as f:
                f.write(scene['sbs_infill'] + "\n")

        video_files_to_concat.append(scene['infilled'])

    if os.path.exists(batch_file):
        subprocess.run(f"{python} stereo_crafter_infill.py --sbs_color_video {batch_file} --sbs_mask_video {batch_file2}", shell=True)
        os.remove(batch_file)
        os.remove(batch_file2)

    return video_files_to_concat


def step7_concat_and_mux(args: argparse.Namespace, video_files_to_concat: List[str]) -> None:
    """
    Validate and then join all per-scene outputs with original audio using ffmpeg.
    """
    print("Step seven: encode all scenes as a new video file")

    # Validate lengths of 'infilled' targets (as original code does)
    # Note: validate_video_lengths expects scene dicts; original code validated
    # scene_video_files. To preserve behavior, we do it in the caller before this function
    # if needed. Here we just proceed to concat.

    ffmpeg_concat_file = 'ffmpeg_concat.txt'
    with open(ffmpeg_concat_file, 'w') as f:
        for video in video_files_to_concat:
            if not os.path.exists(video):
                print("ERROR scene file is missing:", video)
            f.write(f"file '{video}'\n")

    video_name = os.path.basename(args.color_video)
    mp4_result_video_file = os.path.join(args.output_dir, video_name + "_SBS.mp4")

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

    # If ffmpeg fails, re-encode audio to AAC
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


# -------------------------
# Orchestrator / main
# -------------------------

def main():
    args = parse_args()
    skip_last_step = False

    ensure_output_dir(args.output_dir)
    ensure_scene_file(args)

    raw_video, frame_width, frame_height, frame_rate = open_input_video(args.color_video)

    scenes = load_and_split_scenes(args.scene_file, args.csv_delimiter, args.max_scene_frames)
    scene_video_files = plan_scene_files(scenes, args.output_dir, args.end_scene)

    # Step 1: create scene clips
    step1_create_scene_videos(raw_video, scene_video_files, frame_rate, frame_width, frame_height)

    # Step 2: depth estimation
    step2_estimate_depth(args, scene_video_files)

    # Step 3: masks
    step3_generate_masks(args, scene_video_files)

    # Step 4: convergence
    step4_find_convergence(scene_video_files)

    if args.no_render:
        return

    # Step 5: render SBS
    step5_render_sbs(args, scene_video_files)

    # Step 6: infill & collect outputs
    video_files_to_concat = step6_infill_and_collect(args, scene_video_files)

    # Validate before final concat (preserves original assert)
    assert validate_video_lengths(scene_video_files), "Something was wrong with one of the video files"

    if skip_last_step or args.no_render:
        return

    # Step 7: concat & mux with audio
    step7_concat_and_mux(args, video_files_to_concat)


if __name__ == '__main__':
    main()

