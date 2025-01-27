# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import numpy as np
import os
import torch
import cv2

import sys
sys.path.append("Depth-Anything-V2/metric_depth")
import metric_dpt_func

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video


def save_24bit(frames, output_video_path, fps):
    """
    Saves depth maps encoded in the R, G and B channels of a video (to increse accuracy as when compared to gray scale)
    """
    height = frames.shape[1]
    width = frames.shape[2]

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print("max metric depth: ", frames.max())

    MODEL_maxOUTPUT_depth = 6 ####XXXXX hack imortant pick a value  slitght above max metric depth to save the depth in th video file nicly

    for i in range(frames.shape[0]):
        depth = frames[i]
        scaled_depth = (((255**4)/MODEL_maxOUTPUT_depth)*depth.astype(np.float64)).astype(np.uint32)

        # View the depth as raw bytes: shape (H, W, 4)
        depth_bytes = scaled_depth.view(np.uint8).reshape(height, width, 4)


        R = (depth_bytes[:, :, 3]) # Most significant bits in R and G channel (duplicated to reduce compression artifacts)
        G = (depth_bytes[:, :, 3])
        B = (depth_bytes[:, :, 2]) # Least significant bit in blue channel
        bgr24bit = np.dstack((B, G, R))
        out.write(bgr24bit)

    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--shift', type=float, default=0.0, help='Use this shift value instead of the rolling average')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)

    height = frames.shape[1]
    width = frames.shape[2]
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE)

    max_depth = depths.max()

    #Figure out metric conversion factor and rescale depth

    #often_control_metric_depth = 4
    metric_scalings = []
    metric_shifts = []
    last_metric_depth, last_inv_depth = None, None
    for i in range(0, len(frames)):

        print("---- frame ", i, " ---")
        #We techinally only calculate the metric conversion every so often as it genreally is slow to change
        #if i % often_control_metric_depth == 0 or len(metric_scalings) < 32/often_control_metric_depth:

		# get the metric depthmap
        metric_depth = metric_dpt_func.get_metric_depth(frames[i])

        metric_min = metric_depth.min()
        # We align both depth maps to their respective midpoints and use those aligned depthmaps to calculate the metric scaling
        # I am unsure if the actual midpoint is the best point to scale around

        inv_depth = max_depth - depths[i] #the video depth is inverted so we uninvert it
        zero_mask = inv_depth == 0.0
        inv_depth[zero_mask] = 0.01 # Fix devide by zero

        rel2met_scale = np.mean((metric_depth-metric_min)/inv_depth)
        metric_scalings.append(rel2met_scale)

        inv_depth[zero_mask] = 0.0 # Fix devide by zero

        # Calculate the rolling avergae scaling
        rolling_avg_scale = np.mean(metric_scalings)


        print("rolling_avg_scale:", rolling_avg_scale)

        # We rescale the video depth to match the rolling avg metric depth
        rescaled_depth = inv_depth * rolling_avg_scale

		#We only use pixels closer than 2 meter to calculate shift
        close_pixels = metric_depth < 2.0
        metric_vs_rel_shift = np.mean(metric_depth[close_pixels] - rescaled_depth[close_pixels])
        metric_shifts.append(metric_vs_rel_shift)

        rolling_avg_shift = np.mean(metric_shifts)
        
        use_shift = rolling_avg_shift
        if args.use_shift is 0.0:
            use_shift = rolling_avg_shift
            print("rolling_avg_shift:", rolling_avg_shift)

        if len(metric_scalings) > 100:
            metric_scalings.pop(0)
            metric_shifts.pop(0)

        # Then we shift the depth so that the camera pos matches metric
        depths[i] = rescaled_depth + use_shift


    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    output_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depth.mp4')
    save_24bit(depths, output_video_path, fps)