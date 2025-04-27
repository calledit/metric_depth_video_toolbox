import argparse
import numpy as np
import os
import torch
import cv2

import depth_frames_helper

import sys
sys.path.append("Video-Depth-Anything"+os.sep+"metric_depth")

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything metric')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs['vitl'])
    video_depth_anything.load_state_dict(torch.load("Video-Depth-Anything"+os.sep+"metric_depth"+os.sep+"checkpoints"+os.sep+"metric_video_depth_anything_vitl.pth", map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    size_frame, target_fps = read_video_frames(args.color_video, 1, args.target_fps, 99999999)
    height = size_frame.shape[1]
    width = size_frame.shape[2]
    rat = min(height, width) / max(height, width)
    siz = args.input_size/rat

    frames, target_fps = read_video_frames(args.color_video, args.max_frames, args.target_fps, siz)

    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    
    depth_frames_helper.save_depth_video(depths, args.color_video+'_depth.mkv', fps, args.max_depth, width, height)