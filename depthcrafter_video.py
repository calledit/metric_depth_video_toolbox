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

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))

    for i in range(nr_frames):
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
    parser = argparse.ArgumentParser(description='Depthcrafter metric prompt')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--depth_video', type=str, required=True, help='reference metric depth video used to obtain conversion constants, can be created with any image depth model')
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--max_res', type=int, default=768)
    parser.add_argument('--use_depth_prompting', default=False, action='store_true', help='Prompts depthcrafter with depth from the reference', required=False)


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

    latents = None
    if args.use_depth_prompting:
        print("create initial prompt depth latents")
        with torch.no_grad():

            needs_upcasting = (
                pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
            )
            if needs_upcasting:
                pipe.vae.to(dtype=torch.float32)

            from diffusers.utils.torch_utils import randn_tensor
            #TODO: FIX BUG if there is more than 110 frames the latents will bu used on subseqvent batches to
            nr_frames = min(len(depth_ref_frames), 110)
            if nr_frames > 110:
                raise ValueError("Due to implementation details videos longer than 110 frames cant be prompted")
            shape = (1, nr_frames, 4, 72, 96)
            latents = randn_tensor(shape)
            latents = latents * pipe.scheduler.init_noise_sigma


            inv_depth_ref_frames = []
            for i in range(0, min(args.max_frames, len(depth_ref_frames))):

                print("frame: ", i)

                ref_depth = depth_ref_frames[i]

                # only use as refernce if valid, when all depth values are zero that means this frame is not suposed to be used
                if np.all(ref_depth == 0):
                    continue

                inv_depth_ref = 1/ref_depth
                inv_depth_refx3 = np.stack((inv_depth_ref,)*3, axis=-1)

                #convert np depth to torch depth
                frame = torch.from_numpy(inv_depth_refx3.transpose(2, 0, 1))
                frame = frame * 2.0 - 1.0
                frame = frame.unsqueeze(0)
                frame = frame.to(DEVICE)

                frame_latents = pipe.vae.encode(frame).latent_dist.mode()

                #This probably wont work since latent has other shape
                latents[:, i, :, :, :] = frame_latents

            if needs_upcasting:
                pipe.vae.to(dtype=torch.float16)


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
            latents=latents,
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

        #nothing can be 0 depth from the camera that must be some type of error set it max distance
        ref_depth[ref_depth == 0.0] = args.max_depth

        inv_metric_depth = 1/ref_depth

        targets.append(inv_metric_depth)
        sources.append(depths[i])

    depth_ref_frames = None
    src = np.concatenate(sources)
    sources = None
    trg = np.concatenate(targets)
    targets = None

    scale, shift = compute_scale_and_shift_full(src, trg)
    print("scale:", scale, "shift:", shift)
    trg, src = None, None

    out_depths = []
    for i in range(0, len(depths)):


        norm_inv = depths[i]

        #Convert from inverse rel depth to inverse metric depth
        inverse_reconstructed_metric_depth = (norm_inv * scale) + shift
        inverse_reconstructed_metric_depth[ inverse_reconstructed_metric_depth == 0.0] = 1e-4

        reconstructed_metric_depth = np.clip(1/inverse_reconstructed_metric_depth, 0 , args.max_depth)

        reconstructed_metric_depth = np.nan_to_num(reconstructed_metric_depth, nan=args.max_depth)

        out_depths.append(np.clip(cv2.resize(reconstructed_metric_depth, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR), 0 , args.max_depth))

    depths = None
    print("Save depth file")
    output_video_path = args.color_video + "_depth.mkv"
    save_24bit(out_depths, output_video_path, frame_rate, args.max_depth)