import argparse
import numpy as np
import os
import torch
import cv2

import depth_frames_helper

import sys
sys.path.append("Video-Depth-Anything")
sys.path.append("Video-Depth-Anything"+os.sep+"Depth-Anything-V2"+os.sep+"metric_depth")
import metric_dpt_func

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

def compute_scale_and_shift_full(prediction, target, mask = None):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    if mask is None:
        mask = np.ones_like(target)==1
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


# -----------------------
# Helpers for batch mode
# -----------------------
def _is_txt(path):
    return isinstance(path, str) and path.lower().endswith(".txt")


def _read_list_file(path):
    """
    Returns a list of stripped lines, ignoring blanks and lines starting with '#'.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _normalize_optional_depth_entry(entry):
    """
    Treat '', '-', 'none', 'None' as None.
    """
    if entry is None:
        return None
    s = entry.strip()
    if s == "" or s == "-" or s.lower() == "none":
        return None
    return s


def run_on_pair(video_depth_anything, DEVICE, args, color_video_path, depth_video_path):
    print(f"\n=== Processing ===")
    print(f"color_video: {color_video_path}")
    print(f"depth_video: {depth_video_path}")

    # establish sizes & target_fps for sizing
    size_frame, target_fps = read_video_frames(color_video_path, 1, args.target_fps, 99999999)
    height = size_frame.shape[1]
    width = size_frame.shape[2]
    rat = min(height, width) / max(height, width)
    siz = args.input_size/rat

    print("read video frames")
    frames, target_fps = read_video_frames(color_video_path, args.max_frames, args.target_fps, siz)

    ref_frames = None
    if depth_video_path is not None:
        ref_frames, _ = read_video_frames(depth_video_path, 32, args.target_fps, siz)

    print("infer depths")
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    nr_frames = len(frames)

    # Figure out metric conversion factor and rescale depth
    targets = []
    sources = []

    # We only do the first 32 as that is enough and video_depth_anything.infer_video_depth tries to align everything to frame 0 anyway
    print("Use 32 first frames to calculate metric conversion constants")
    for i in range(0, min(32, nr_frames)):
        norm_inv = depths[i]

        # get the metric depthmap
        if ref_frames is not None:
            rgb = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2RGB)
            # Decode video depth
            metric_depth = depth_frames_helper.decode_rgb_depth_frame(rgb, args.max_depth, True)
        else:
            metric_depth = metric_dpt_func.get_metric_depth(frames[i])

        inv_metric_depth = 1/metric_depth

        targets.append(inv_metric_depth)
        sources.append(norm_inv)

    frames = None

    scale, shift = compute_scale_and_shift_full(np.concatenate(sources), np.concatenate(targets))

    targets, sources = None, None

    print("scale:", scale, "shift:", shift)

    for i in range(0, nr_frames):
        print("---- frame ", i, " ---")
        norm_inv = depths[i]

        # Convert from inverse rel depth to inverse metric depth
        inverse_reconstructed_metric_depth = (norm_inv * scale) + shift

        reconstructed_metric_depth = 1/inverse_reconstructed_metric_depth
        if reconstructed_metric_depth.min() < 0.0:
            print("WARNING: depth model gave minus depth values, depth behind the camera. Ignoring those depth values.")
        #under zero depth means that the model failed we set that to max depth as it is more likley than behind the camera
        reconstructed_metric_depth[reconstructed_metric_depth < 0.0] = float(args.max_depth)

        depths[i] = reconstructed_metric_depth

    depth_frames_helper.save_depth_video(depths, color_video_path+'_depth.mkv', fps, args.max_depth, width, height)
    print(f"saved: {color_video_path+'_depth.mkv'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--depth_video', type=str, required=False, help='reference metric depth video or a .txt list (optional)')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--model', type=str, default='vitl', help='vitl or with vits, downlaod vits with install script.')
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    #DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    print("loading model")
    video_depth_anything = VideoDepthAnything(**model_configs[args.model])
    video_depth_anything.load_state_dict(torch.load('Video-Depth-Anything/checkpoints/video_depth_anything_' + args.model + '.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # -----------------------
    # Single vs Batch logic
    # -----------------------
    if _is_txt(args.color_video):
        # Batch mode; depth_video is optional here
        color_list = _read_list_file(args.color_video)

        if args.depth_video is not None:
            if not _is_txt(args.depth_video):
                raise ValueError("If --color_video is a .txt file, then --depth_video must also be a .txt file (or omitted).")
            depth_list_raw = _read_list_file(args.depth_video)
            if len(color_list) != len(depth_list_raw):
                raise ValueError(
                    f"List length mismatch: {args.color_video} has {len(color_list)} entries, "
                    f"{args.depth_video} has {len(depth_list_raw)} entries."
                )
            depth_list = [_normalize_optional_depth_entry(x) for x in depth_list_raw]
        else:
            # No depth list provided, fall back to metric DPT for all entries
            depth_list = [None] * len(color_list)

        for idx, (c_path, d_path) in enumerate(zip(color_list, depth_list), start=1):
            print(f"\n##### [{idx}/{len(color_list)}] #####")
            run_on_pair(video_depth_anything, DEVICE, args, c_path, d_path)

    else:
        # Single run (original behavior). depth_video remains optional.
        run_on_pair(video_depth_anything, DEVICE, args, args.color_video, _normalize_optional_depth_entry(args.depth_video))
