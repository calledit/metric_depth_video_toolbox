import gc
import os
import numpy as np
import torch

from diffusers.training_utils import set_seed
import argparse
import cv2

import sys
sys.path.append("DepthCrafter")
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import vis_sequence_depth, save_video, read_video_frames

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stereo crafter metric')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--depth_video', type=str, required=True, help='reference metric depth video used to obtain conversion constants, can be created with any image depth model')
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--max_res', type=int, default=1024)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_maxOUTPUT_depth = args.max_depth

    print("load depthcrafter net")
    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
        "tencent/DepthCrafter",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    # load weights of other components from the provided checkpoint
    pipe = DepthCrafterPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    cpu_offload = 'model'

    # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
    if cpu_offload is not None:
        if cpu_offload == "sequential":
            # This will slow, but save more memory
            pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
    else:
        pipe.to("cuda")

    # enable attention slicing and xformers memory efficient attention
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(e)
        print("Xformers is not enabled")
    pipe.enable_attention_slicing()

    set_seed(42)

    print("load color frames")
    frames, _ = read_video_frames(args.color_video, args.max_frames, -1, args.max_res)

    craft_width, craft_height = frames.shape[2], frames.shape[1]

    print("load reference depth frames")
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

    depth_ref_frames = []

    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

        # Decode video depth
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = rgb[..., 0].astype(np.uint32)
        depth_unit[..., 2] = rgb[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        resized_depth_map = cv2.resize(depth, (craft_width, craft_height), interpolation=cv2.INTER_LINEAR)
        depth_ref_frames.append(resized_depth_map)

    if raw_video is not None:
        raw_video.release()


    print("Run the depthcrafter net")
    # inference the depth map using the DepthCrafter pipeline
    with torch.inference_mode():
        res = pipe(
            frames,
            height=craft_height,
            width=craft_width,
            output_type="np",
            guidance_scale=1.0,
            num_inference_steps=5,
            window_size=110,
            overlap=25,
            track_time=True,
        ).frames[0]
        # convert the three-channel output to a single channel depth map
        depths = res.sum(-1) / res.shape[-1]

        
    
    # Recover scale and shift
    targets = []
    sources = []
    for i in range(0, min(len(depths), len(depth_ref_frames))):

        ref_depth = depth_ref_frames[i]

        #only use as refernce if valid, when all depth values are zero that means this frame is not suposed to be used
        if np.all(ref_depth == 0):
            continue

        inv_metric_depth = 1/ref_depth

        targets.append(inv_metric_depth)
        sources.append(depths[i])

    scale, shift = compute_scale_and_shift_full(np.concatenate(sources), np.concatenate(targets))
    print("scale:", scale, "shift:", shift)

    out_depths = []
    for i in range(0, len(depths)):

        print("---- frame ", i, " ---")

        norm_inv = depths[i]

        #Convert from inverse rel depth to inverse metric depth
        inverse_reconstructed_metric_depth = (norm_inv * scale) + shift

        reconstructed_metric_depth = 1/inverse_reconstructed_metric_depth

        out_depths.append(cv2.resize(reconstructed_metric_depth, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR))

    depths = np.array(out_depths)
    output_video_path = args.color_video + "_depthcrafter_depth.mkv"

    save_24bit(depths, output_video_path, frame_rate, args.max_depth)