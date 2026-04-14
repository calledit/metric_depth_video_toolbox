import argparse
import numpy as np
import os
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from StereoCrafter.pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
from scipy.ndimage import binary_dilation

import depth_frames_helper
from infill_common import transfer_lhm_video_refmask, mark_lower_side

# -----------------------
# Config / Globals
# -----------------------
num_inference_steps = None  # More steps look better but slowe set by arg
black = np.array([0, 0, 0], dtype=np.uint8)
blue = np.array([0, 0, 255], dtype=np.uint8)
pipeline = None

# Allow only ONE generate_infilled_frames on GPU at any time.
_GPU_GATE = Semaphore(1)

# -----------------------
# Helpers for batch mode
# -----------------------
def _is_txt(path: str) -> bool:
    return isinstance(path, str) and path.lower().endswith(".txt")

def _read_list_file(path: str):
    """
    Returns a list of stripped lines, ignoring blanks and lines starting with '#'.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items

def generate_infilled_frames(input_frames, input_masks, fps: float):
    """GPU-heavy section — guarded by a global semaphore so only one thread runs at once."""
    global num_inference_steps, pipeline
    # Pre-format on CPU
    input_frames = torch.tensor(input_frames).permute(0, 3, 1, 2).float() / 255.0
    frames_mask  = torch.tensor(input_masks).permute(0, 1, 2).float() / 255.0

    with _GPU_GATE:
        # Everything under this lock may use lots of VRAM.
        video_latents = pipeline(
            frames=input_frames,
            frames_mask=frames_mask,
            height=input_frames.shape[2],
            width=input_frames.shape[3],
            num_frames=len(input_frames),
            output_type="latent",
            min_guidance_scale=1.01,
            max_guidance_scale=1.01,
            decode_chunk_size=8,
            fps=fps,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=num_inference_steps,
        ).frames[0]

        video_latents = video_latents.unsqueeze(0)
        if video_latents == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        # decode_latents uses VAE (GPU); keep it inside the gate.
        video_frames = pipeline.decode_latents(
            video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2
        )
        del video_latents
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="np")[0]
        
        # Proactively free/cycle memory between clips
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return (video_frames * 255).astype(np.uint8)

def deal_with_frame_chunk(keep_first_three, chunk, out, keep_last_three, frame_width, frame_height, fps):
    pic_width = int(frame_width // 2)

    # Reasonable working size for the diffusion model decode
    new_width = 1024
    new_height = 768

    right_input, left_input = [], []
    right_mask_input, left_mask_input = [], []

    for img_and_mask in chunk:
        # Right mask
        org_img_mask = img_and_mask[1][:frame_height, pic_width:]
        img_mask_true_paralax = ~np.all(org_img_mask == black, axis=-1)
        img_mask_resized = np.array(
            cv2.resize(img_mask_true_paralax.astype(np.uint8) * 255, (new_width, new_height)) > 0
        ).astype(np.uint8) * 255
        right_mask_input.append(img_mask_resized)

        # Right image
        org_img = img_and_mask[0][:frame_height, pic_width:]
        img_resized = cv2.resize(org_img, (new_width, new_height))
        right_input.append(img_resized)

        # Left mask (fliplr)
        org_img_mask = np.fliplr(img_and_mask[1][:frame_height, :pic_width])
        img_mask_true_paralax = ~np.all(org_img_mask == black, axis=-1)
        img_mask_resized = np.array(
            cv2.resize(img_mask_true_paralax.astype(np.uint8) * 255, (new_width, new_height)) > 0
        ).astype(np.uint8) * 255
        left_mask_input.append(img_mask_resized)

        # Left image (fliplr)
        org_img = np.fliplr(img_and_mask[0][:frame_height, :pic_width])
        img_resized = cv2.resize(org_img, (new_width, new_height))
        left_input.append(img_resized)

    right_mask_input = np.array(right_mask_input)
    left_mask_input  = np.array(left_mask_input)
    right_input      = np.array(right_input)
    left_input       = np.array(left_input)

    print("generating left side images")
    if np.all(left_mask_input == 0):
        left_frames = left_input
    else:
        left_frames = generate_infilled_frames(left_input, left_mask_input, fps)
        left_frames = transfer_lhm_video_refmask(left_frames, left_input, left_mask_input)

    print("generating right side images")
    if np.all(right_mask_input == 0):
        right_frames = right_input
    else:
        right_frames = generate_infilled_frames(right_input, right_mask_input, fps)
        right_frames = transfer_lhm_video_refmask(right_frames, right_input, right_mask_input)

    sttart = 0 if keep_first_three else 3
    eend = len(left_frames) if keep_last_three else len(left_frames) - 3

    proccessed_frames = []
    for j in range(sttart, eend):
        left_img  = cv2.resize(np.fliplr(left_frames[j]), (pic_width, frame_height))
        right_img = cv2.resize(right_frames[j], (pic_width, frame_height))

        right_org_img = chunk[j][0][:frame_height, pic_width:].copy()
        left_org_img  = chunk[j][0][:frame_height, :pic_width].copy()
        right_mask    = chunk[j][1][:frame_height, pic_width:]
        left_mask     = chunk[j][1][:frame_height, :pic_width]

        # invert mask: original black = source (keep), white = infill region
        right_black_mask = np.all(right_mask == black, axis=-1)
        left_black_mask  = np.all(left_mask == black, axis=-1)

        left_org_img[~left_black_mask]  = left_img[~left_black_mask]
        right_org_img[~right_black_mask] = right_img[~right_black_mask]

        basic_out_image = cv2.hconcat([left_org_img, right_org_img])
        basic_out_image_uint8 = np.clip(basic_out_image, 0, 255).astype(np.uint8)
        proccessed_frames.append(basic_out_image_uint8)

        # Edge blending to avoid halos
        right_mask_blue = mark_lower_side(right_mask)
        right_backedge_mask = np.all(right_mask_blue == blue, axis=-1)
        left_mask_blue = mark_lower_side(left_mask)
        left_backedge_mask = np.all(left_mask_blue == blue, axis=-1)

        right_backedge_mask = binary_dilation(right_backedge_mask, iterations=6)
        left_backedge_mask  = binary_dilation(left_backedge_mask,  iterations=6)

        right_alpha = cv2.GaussianBlur(right_backedge_mask.astype(np.float32), (15, 15), 0)[..., np.newaxis]
        left_alpha  = cv2.GaussianBlur(left_backedge_mask.astype(np.float32),  (15, 15), 0)[..., np.newaxis]

        left_img  = left_alpha * left_img + (1 - left_alpha) * left_org_img
        right_img = right_alpha * right_img + (1 - right_alpha) * right_org_img

        out_image = cv2.hconcat([left_img, right_img])
        out_image_uint8 = np.clip(out_image, 0, 255).astype(np.uint8)
        out.write(cv2.cvtColor(out_image_uint8, cv2.COLOR_RGB2BGR))

    return proccessed_frames

def process_pair(sbs_color_video_path: str, sbs_mask_video_path: str, args):
    if not os.path.isfile(sbs_color_video_path):
        raise Exception(f"input sbs_color_video does not exist: {sbs_color_video_path}")

    if not os.path.isfile(sbs_mask_video_path):
        raise Exception(f"input sbs_mask_video does not exist: {sbs_mask_video_path}")
    
    print(f"Processing: {sbs_color_video_path}")
    raw_video = cv2.VideoCapture(sbs_color_video_path)
    frame_width  = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = raw_video.get(cv2.CAP_PROP_FPS)
    out_size     = (frame_width, frame_height)

    mask_video = cv2.VideoCapture(sbs_mask_video_path)
    m_frame_width  = int(mask_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    m_frame_height = int(mask_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    assert frame_width == m_frame_width and frame_height == m_frame_height, "mask ans color video not same resolution"


    output_tmp_video_file = sbs_color_video_path + "_tmp_infilled.mkv"
    output_video_file = sbs_color_video_path + "_infilled.mkv"
    codec = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(output_tmp_video_file, codec, fps, out_size)

    frame_buffer = []
    first_chunk = True
    last_chunk = False
    frame_n = 0
    frames_chunk = 25

    try:
        while raw_video.isOpened():
            print(f"Frame: {frame_n} {frame_n / max(fps, 1e-6)}s")
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            frame_n += 1

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

            ret_mask, mask_frame = mask_video.read()
            if not ret_mask:
                # If mask video ended early, assume blank mask remainder
                mask_frame = np.zeros_like(raw_frame)
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)

            frame_buffer.append([rgb, mask_frame])

            if len(frame_buffer) >= frames_chunk:
                proccessed_frames = deal_with_frame_chunk(
                    first_chunk, frame_buffer, out, last_chunk,
                    frame_width, frame_height, fps
                )

                if first_chunk:
                    first_chunk = False
                frame_buffer = [
                    (proccessed_frames[-6], frame_buffer[-6][1]),
                    (proccessed_frames[-5], frame_buffer[-5][1]),
                    (proccessed_frames[-4], frame_buffer[-4][1]),
                    frame_buffer[-3],
                    frame_buffer[-2],
                    frame_buffer[-1],
                ]  # keep overlap

            if args.max_frames != -1 and frame_n >= args.max_frames:
                break

        last_chunk = True
        deal_with_frame_chunk(
            first_chunk, frame_buffer, out, last_chunk,
            frame_width, frame_height, fps
        )
    finally:
        raw_video.release()
        mask_video.release()
        out.release()

    depth_frames_helper.verify_and_move(output_tmp_video_file, frame_n, output_video_file)

    print(f"Done. Wrote: {output_video_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Crafter infill script')
    parser.add_argument('--sbs_color_video', type=str, required=True, help='side by side stereo video renderd with point clouds in the masked area')
    parser.add_argument('--sbs_mask_video', type=str, required=True, help='side by side stereo video mask')
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--num_inference_steps', default=5, type=int, help='Numer of defussion steps. More look better but is slower', required=False)
    args = parser.parse_args()

    num_inference_steps = args.num_inference_steps

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -----------------------
    # Load pipeline once (shared)
    # -----------------------
    img2vid_path = 'weights/stable-video-diffusion-img2vid-xt-1-1'
    unet_path = 'StereoCrafter/weights/StereoCrafter'

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        img2vid_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        img2vid_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        img2vid_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    ).to("cuda" if DEVICE == 'cuda' else "cpu")

    # -----------------------
    # Single vs Batch logic
    # -----------------------
    if _is_txt(args.sbs_color_video):
        if not _is_txt(args.sbs_mask_video):
            raise ValueError("If --sbs_color_video is a .txt file, then --sbs_mask_video must also be a .txt file.")

        color_list = _read_list_file(args.sbs_color_video)
        mask_list  = _read_list_file(args.sbs_mask_video)

        if len(color_list) != len(mask_list):
            raise ValueError(
                f"List length mismatch: {args.sbs_color_video} has {len(color_list)} entries, "
                f"{args.sbs_mask_video} has {len(mask_list)} entries."
            )

        print(f"Batch mode: {len(color_list)} pairs")
        # Run up to 2 clips in parallel. GPU sections are serialized by _GPU_GATE.
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(process_pair, c_path, m_path, args)
                       for (c_path, m_path) in zip(color_list, mask_list)]
            # Consume as they finish to keep the pool busy
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    # Surface errors but keep other jobs running
                    print(f"[ERROR] A clip failed: {e}")

    else:
        # Single-file mode (original behavior)
        process_pair(args.sbs_color_video, args.sbs_mask_video, args)
