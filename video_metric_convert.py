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
    height = frames.shape[1]
    width = frames.shape[2]

    #writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '0'])

    #out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print("max metric depth: ", frames.max())

    MODEL_maxOUTPUT_depth = 6 ####XXXXX hack imortant pick a value  slitght above max metric depth to save the depth in th video file nicly

    #ycrcb_ofset = 16
    #Y_range = 235 - ycrcb_ofset
    #C_range = 240 - ycrcb_ofset

    #Y_convert = 255/Y_range
    #C_convert = 255/C_range

    for i in range(frames.shape[0]):
        depth = frames[i]
        #print(depth)
        #exit(0)
        scaled_depth = (((255**4)/MODEL_maxOUTPUT_depth)*depth.astype(np.float64)).astype(np.uint32)

        # View the depth as raw bytes: shape (H, W, 4)
        depth_bytes = scaled_depth.view(np.uint8).reshape(height, width, 4)

        # We only have 3 channels to store in so we discard the least significant bits
        #Y = (depth_bytes[:, :, 3]/Y_convert) + ycrcb_ofset #Chnages indicate late differances
        #Y = (depth_bytes[:, :, 3]) #Chnages indicate late differances
        #Cb = (depth_bytes[:, :, 2]/C_convert) + ycrcb_ofset #Chnages Cb medium difs
        #Cr = (depth_bytes[:, :, 2]/C_convert) + ycrcb_ofset #and in Cr small chnages

        #scale = 64
        #3Cr = np.ones((height, width), dtype=np.uint8)*(128-scale)+(depth_bytes[:, :, 2]/(256/scale))
        #Cb = np.ones((height, width), dtype=np.uint8)*(128-scale)+(depth_bytes[:, :, 2]/(256/scale))
        #Cb = np.ones((height, width), dtype=np.uint8)*(128) # +(depth_bytes[:, :, 2]/(256/scale))

        #ycrcb24bit = np.dstack((Y, Cr, Cb)).astype(np.uint8)

        #print("ycrcb in: ",ycrcb24bit[0][0])
        #rgb24bit  = np.rint(cv2.cvtColor(ycrcb24bit.astype(np.float32), cv2.COLOR_YCrCb2RGB)).astype(np.uint8)
        #rgb24bit  = cv2.cvtColor(ycrcb24bit, cv2.COLOR_YCrCb2RGB)
        #print("rgb: ",rgb24bit[0][0])

        #print("decode")
        #ycrcb = np.rint(cv2.cvtColor(rgb24bit, cv2.COLOR_RGB2YCrCb)).astype(np.uint8)
        #ycrcb = cv2.cvtColor(rgb24bit, cv2.COLOR_RGB2YCrCb)
        #print("ycrcb out: ",ycrcb[0][0])
        #print(ycrcb)
        #print(rgb24bit)


        R = (depth_bytes[:, :, 3]) #Chnages indicate late differances
        G = (depth_bytes[:, :, 3])
        #G = np.zeros((height, width), dtype=np.uint8)
        B = (depth_bytes[:, :, 2])
        bgr24bit = np.dstack((B, G, R))
        #rgb24bit = np.dstack((R, G, B))
        #print("rgb: ",rgb24bit[0][0])
        #print("bgr: ",bgr24bit[0][0])
        #exit(0)
        out.write(bgr24bit)
        #writer.append_data(rgb24bit)

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
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE)

    max_depth = depths.max()

    #Figure out metric conversion factor and rescale depth
    
    #often_control_metric_depth = 4
    metric_scalings = []
    for i in range(0, len(frames)):
        
        #We only calculate the metric conversion every so often as it genreally is slow to change
        
        ## Nah we need to know the metric_shift so this need to be calculated every frame
        #if i % often_control_metric_depth == 0 or len(metric_scalings) < 32/often_control_metric_depth:
        
        metric_depth = metric_dpt_func.get_metric_depth(frames[i])
        
        # We align both depth maps to their respective midpoints and use those aligned depthmaps to calculate the metric scaling
        # I am unsure if the actual midpoint is the best point to scale around
        metric_mid_point_shift = metric_depth.mean()
        vid_depth_mid_point_shift = depths[i].mean()

        inv_depth = (max_depth-vid_depth_mid_point_shift) - depths[i] #the video depth is inverted so we uninvert it
        zero_mask = inv_depth == 0.0
        inv_depth[zero_mask] = 0.01 # Fix devide by zero

        metric_scalings.append(np.mean((metric_depth-metric_mid_point_shift)/inv_depth))
        
        inv_depth[zero_mask] = 0.0 # Fix devide by zero
        
        
    
        if len(metric_scalings) > 32:
            metric_scalings.pop(0)
        
        rolling_avg_scale = np.mean(metric_scalings)
        
        # We rescale the video depth to match the rolling avg metric depth
        rescaled_depth = inv_depth * rolling_avg_scale
        
        # Then we shift the depth so that the camera pos matches metric (XXX: Feels like this could be done in a better way)
        depths[i] = metric_mid_point_shift+rescaled_depth


    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    output_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depth.mp4')
    save_24bit(depths, output_video_path, fps)