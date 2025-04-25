import argparse
import numpy as np
import os
import torch
import cv2

import sys
sys.path.append("Video-Depth-Anything"+os.sep+"metric_depth")

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

def save_24bit(frames, output_video_path, fps, max_depth_arg, rescale_width, rescale_height):
    """
    Saves depth maps encoded in the R, G and B channels of a video (to increse accuracy as when compared to gray scale)
    """



    MODEL_maxOUTPUT_depth = max_depth_arg ### pick a value slitght above max metric depth to save the depth in th video file nicly
    # if you pick a high value you will lose resolution

    if isinstance(frames, np.ndarray):
        height = frames.shape[1]
        width = frames.shape[2]
        max_depth = frames.max()
        print("max metric depth: ", max_depth)
        # incase you did not pick a absolute value we max out (this mean each video will have depth relative to max_depth)
        # (if you want to use the video as a depth souce a absolute value is prefrable)
        if MODEL_maxOUTPUT_depth < max_depth:
            print("warning: output depth is deeper than max_depth. The depth will be clipped")
        nr_frames = frames.shape[0]
    else:
        nr_frames = len(frames)
        height = frames[0].shape[0]
        width = frames[0].shape[1]

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (rescale_width, rescale_height))

    for i in range(nr_frames):
        depth = cv2.resize(frames[i], (rescale_width, rescale_height), interpolation=cv2.INTER_LINEAR)
        scaled_depth = (((255**4)/MODEL_maxOUTPUT_depth)*depth.astype(np.float64)).astype(np.uint32)

        # View the depth as raw bytes: shape (H, W, 4)
        depth_bytes = scaled_depth.view(np.uint8).reshape(rescale_height, rescale_width, 4)


        R = (depth_bytes[:, :, 3]) # Most significant bits in R and G channel (duplicated to reduce compression artifacts)
        G = (depth_bytes[:, :, 3])
        B = (depth_bytes[:, :, 2]) # Least significant bit in blue channel
        bgr24bit = np.dstack((B, G, R))
        out.write(bgr24bit)

    out.release()

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

    nr_frames = len(frames)

    max_depth = depths.max()

    output_video_path = args.color_video+'_depth.mkv'
    save_24bit(depths, output_video_path, fps, args.max_depth, width, height)