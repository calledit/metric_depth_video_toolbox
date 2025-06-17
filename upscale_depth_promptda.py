import argparse
import numpy as np
import os
import torch
import cv2
import math

import depth_frames_helper

import sys
sys.path.append("PromptDA")

from promptda.promptda import PromptDA


def closest_higher_multiple(x, base=14):
    return math.ceil(x / base) * base

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PromptDA depth video upscaling')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--depth_video', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_video = cv2.VideoCapture(args.depth_video)
    color_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(color_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(color_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = color_video.get(cv2.CAP_PROP_FPS)


    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to(DEVICE).eval()
    rescale_height, rescale_width = 192,256
    color_rescale_height, color_rescale_width = closest_higher_multiple(frame_height), closest_higher_multiple(frame_width)

    depths = []

    frame_n = 0
    while color_video.isOpened():

        frame_n += 1
        print("--- frame ",frame_n," ----")

        if args.max_frames < frame_n and args.max_frames != -1:
            break

        ret, color_frame = color_video.read()
        if not ret:
            break
        rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

        ret, depth_frame = depth_video.read()
        if not ret:
            break
        depth_rgb = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
        metric_depth = depth_frames_helper.decode_rgb_depth_frame(depth_rgb, args.max_depth, True)

        low_res_metric_depth = cv2.resize(metric_depth, (rescale_width, rescale_height), interpolation=cv2.INTER_LINEAR)

        if frame_width != color_rescale_width and rescale_height != color_rescale_height:
            rgb_multiple_of_14 = cv2.resize(rgb, (color_rescale_width, color_rescale_height), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_multiple_of_14 = rgb

        rgb_tensor = (torch.from_numpy(rgb_multiple_of_14).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(DEVICE)  # (1, 3, H, W)
        depth_tensor = torch.from_numpy(low_res_metric_depth).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # (1, 1, H, W)


        upscaled_depth = model.predict(rgb_tensor, depth_tensor)

        upscaled_depth = upscaled_depth.squeeze().detach().cpu().numpy()

        if frame_width != color_rescale_width and rescale_height != color_rescale_height:
            upscaled_depth = cv2.resize(upscaled_depth, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)


        depths.append(upscaled_depth)


    depth_frames_helper.save_depth_video(depths, args.depth_video+'_upscaled.mkv', frame_rate, args.max_depth, frame_width, frame_height)
