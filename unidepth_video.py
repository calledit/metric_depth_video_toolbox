import argparse
import numpy as np
import os
import torch
import cv2

import depth_frames_helper
import depth_map_tools

import sys
print("please ignore warnings about depreciation warnings for xformers components")
print("the loading takes a while the first time just wait")
sys.path.append("UniDepth")
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid
from unidepth.utils.camera import Pinhole


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDVT Unidepth video converter')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)

    args = parser.parse_args()

    if args.xfov is None and args.yfov is None:
        print("Either --xfov or --yfov is required.")
        exit(0)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")

    MODEL_maxOUTPUT_depth = args.max_depth

    raw_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)
    cam_matrix_torch = torch.from_numpy(cam_matrix)

    model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14").to(DEVICE)
    model.interpolation_mode = "bilinear"

    depths = []

    frame_n = 0
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        frame_n += 1
        print("--- frame ",frame_n," ----")

        if args.max_len < frame_n and args.max_len != -1:
            break

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        predictions = model.infer(rgb_torch, cam_matrix_torch)
        depths.append(predictions["depth"].squeeze().cpu().numpy())
        pred_intrinsic = predictions["intrinsics"].squeeze().cpu().numpy()
        fovx, fovy = depth_map_tools.fov_from_camera_matrix(pred_intrinsic)
        print("fovx:", fovx, "fovy:", fovy)

    video_name = os.path.basename(args.color_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    depth_frames_helper.save_depth_video(depths, args.color_video+'_depth.mkv', fps, args.max_depth, frame_width, frame_height)
