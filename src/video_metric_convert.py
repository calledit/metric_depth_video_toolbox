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


# Also copyright Me, for the parts i wrote.

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


def save_24bit(frames, output_video_path, fps, max_depth_arg):
    """
    Saves depth maps encoded in the R, G and B channels of a video (to increse accuracy as when compared to gray scale)
    """
    height = frames.shape[1]
    width = frames.shape[2]

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))

    max_depth = frames.max()
    print("max metric depth: ", max_depth)

    MODEL_maxOUTPUT_depth = max_depth_arg ### pick a value slitght above max metric depth to save the depth in th video file nicly
    # if you pick a high value you will lose resolution
    
    # incase you did not pick a absolute value we max out (this mean each video will have depth relative to max_depth)
    # (if you want to use the video as a depth souce a absolute value is prefrable)
    if MODEL_maxOUTPUT_depth < max_depth:
        print("warning: output depth is deeper than max_depth. The depth will be clipped")

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
    parser.add_argument('--color_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1440)
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--max_depth', default=6, type=int, help='the max depth that the video uses', required=False)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs['vitl'])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_vitl.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames, target_fps = read_video_frames(args.color_video, args.max_len, args.target_fps, args.max_res)

    height = frames.shape[1]
    width = frames.shape[2]
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE)

    max_depth = depths.max()

    #Figure out metric conversion factor and rescale depth

    #often_control_metric_depth = 4

    std_std_constants = []
    inv_metric_means = []
    norm_inv_means = []

    last_metric_depth, last_inv_depth = None, None
    for i in range(0, len(frames)):

        print("---- frame ", i, " ---")
        #We techinally only calculate the metric conversion every so often as it genreally is slow to change
        #if i % often_control_metric_depth == 0 or len(std_std_constants) < 32/often_control_metric_depth:

		# get the metric depthmap
        metric_depth = metric_dpt_func.get_metric_depth(frames[i])

        metric_min = metric_depth.min()
        metric_max = metric_depth.max()


        inverse_metric_max = 1/metric_min
        inverse_metric_min = 1/metric_max

        #The inverse_metric_min comes from the unstable depthmap if you use the real one there is brutal jittering so we overide it and just use 0
        inverse_metric_min = 0

        inv_metric_depth = 1/metric_depth

        inv_metric_std = inv_metric_depth.std()
        inv_metric_mean = inv_metric_depth.mean() - inverse_metric_min

        norm_inv = depths[i]
        norm_inv_std = norm_inv.std()
        norm_inv_mean = norm_inv.mean()

        std_std_constant = inv_metric_std / norm_inv_std


        #Debug stuff
        print("inv_metric_std: ", inv_metric_std)
        print("norm_inv_std: ", norm_inv_std)
        print("std_std_constant: ", std_std_constant)
        print("norm_inv_mean: ", norm_inv_mean)
        print("inv_metric_mean: ", inv_metric_mean)
        print("std_std_constant: ", std_std_constant)

        std_std_constants.append(std_std_constant)
        inv_metric_means.append(inv_metric_mean)
        norm_inv_means.append(norm_inv_mean)


        #When there is less than 10 frames in the rolling average we use the first frame for reference instead the rolling average need to stabilize
        if len(std_std_constants) < 10:
            std_std_constant = std_std_constants[0]
            inv_metric_mean = inv_metric_means[0]
            norm_inv_mean = norm_inv_means[0]
        else:
            std_std_constant = np.mean(std_std_constants)
            inv_metric_mean = np.mean(inv_metric_means)
            norm_inv_mean = np.mean(norm_inv_means)
            
            
        # Looking the constants instead of using rolling averages can give better results but it may allso have issues with moving cameras (i think)
        #std_std_constant = 0.00031527123
        #inv_metric_mean = 0.395
        #norm_inv_mean = 680

        #Convert from metric model std to rel model std
        inverse_depth_m_min = ((norm_inv - norm_inv_mean) * std_std_constant) + inv_metric_mean

        #the above can also be done using min and max instead of standard deviations but it is less robust (i think)
        #inverse_depth_m_min = norm_inverse_depth * (inverse_metric_max-inverse_metric_min)

        inverse_reconstructed_metric_depth = inverse_depth_m_min + inverse_metric_min

        metric_depth2 = 1/inverse_reconstructed_metric_depth

        
        # clear the rolling average when it has over 100 frames
        if len(std_std_constants) > 100:
            std_std_constants.pop(0)
            inv_metric_means.pop(0)
            norm_inv_means.pop(0)

        depths[i] = metric_depth2


    video_name = os.path.basename(args.color_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    output_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depth.mkv')
    save_24bit(depths, output_video_path, fps, args.max_depth)