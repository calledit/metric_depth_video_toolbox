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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--depth_video', type=str, required=False, help='reference metric depth video')
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

    size_frame, target_fps = read_video_frames(args.color_video, 1, args.target_fps, 99999999)
    height = size_frame.shape[1]
    width = size_frame.shape[2]
    rat = min(height, width) / max(height, width)
    siz = args.input_size/rat

    print("read video frames")
    frames, target_fps = read_video_frames(args.color_video, args.max_frames, args.target_fps, siz)

    ref_frames = None
    if args.depth_video is not None:
        ref_frames, _ = read_video_frames(args.depth_video, 32, args.target_fps, siz)

    print("infer depths")
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    nr_frames = len(frames)

    max_depth = depths.max()

    #Figure out metric conversion factor and rescale depth

    targets = []
    sources = []

    #We only do the first 32 as that is enogh and video_depth_anything.infer_video_depth tries to align everything to frame 0 anyway
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

        #Convert from inverse rel depth to inverse metric depth
        inverse_reconstructed_metric_depth = (norm_inv * scale) + shift

        reconstructed_metric_depth = 1/inverse_reconstructed_metric_depth

        depths[i] = reconstructed_metric_depth



    depth_frames_helper.save_depth_video(depths, args.color_video+'_depth.mkv', fps, args.max_depth, width, height)
