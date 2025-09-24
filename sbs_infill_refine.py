import argparse
import numpy as np
import os
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

import diffuerase
import depth_frames_helper
from scipy.ndimage import binary_dilation


# -----------------------
# Config / Globals
# -----------------------
num_inference_steps = None  # Set by argument
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

def refine_frames(input_frames, input_masks):
    """GPU-heavy section — guarded by a global semaphore so only one thread runs at once."""
    global num_inference_steps, pipeline
    # Pre-format on CPU

    out_frames = None
    with _GPU_GATE:
        out_frames = diffuerase.run_infill_on_frames(input_frames, input_masks, propainer_frames = input_frames)
        # Everything under this lock may use lots of VRAM.
                # Proactively free/cycle memory between clips
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return out_frames

def mark_lower_side(normals_img, max_steps=30):
    H, W = normals_img.shape[:2]
    orig = normals_img
    valid = ~np.all(orig == 0, axis=-1)
    ys, xs = np.nonzero(valid)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    dirs = ((orig[ys, xs, :2].astype(np.float32) / 255) * 2 - 1)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    good = (norms[:, 0] > 1e-6)
    pts = pts[good]
    dirs = dirs[good] / norms[good]

    N = pts.shape[0]
    alive = np.ones(N, dtype=bool)
    res_pts = -np.ones((N, 2), dtype=int)

    for t in range(1, max_steps):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        p = pts[idx] + dirs[idx] * t
        xi = np.rint(p[:, 0]).astype(int)
        yi = np.rint(p[:, 1]).astype(int)

        inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi_in = xi[inb]; yi_in = yi[inb]
        orig_vals = orig[yi_in, xi_in]
        bg_hit = np.all(orig_vals == 0, axis=1)

        hit_idx = idx[inb][bg_hit]
        if hit_idx.size > 0:
            p0 = pts[hit_idx] + dirs[hit_idx] * (t - 1)
            xb = np.rint(p0[:, 0]).astype(int)
            yb = np.rint(p0[:, 1]).astype(int)
            res_pts[hit_idx, 0] = xb
            res_pts[hit_idx, 1] = yb

        idx_oob = idx[~inb]
        alive[idx_oob] = False
        alive[hit_idx] = False

    output = np.zeros_like(orig)
    xb = res_pts[:, 0]; yb = res_pts[:, 1]
    valid_hits = (xb >= 0) & (yb >= 0)
    output[yb[valid_hits], xb[valid_hits]] = (0, 0, 255)
    return output

def deal_with_frame_chunk(keep_first_three, chunk, out, keep_last_three, frame_width, frame_height, fps):
    pic_width = int(frame_width // 2)
    
    # Reasonable working size for the diffusion model decode
    new_width = 1024
    new_height = 768

    right_inputl, left_inputl = [], []
    right_mask_inputl, left_mask_inputl = [], []

    for img_and_mask in chunk:
        # Right mask
        org_img_mask = img_and_mask[1][:frame_height, pic_width:]
        img_mask_true_paralax = ~np.all(org_img_mask == black, axis=-1)
        img_mask_resized = np.array(
            cv2.resize(img_mask_true_paralax.astype(np.uint8) * 255, (new_width, new_height)) > 0
        ).astype(np.uint8) * 255
        right_mask_inputl.append(img_mask_resized)

        # Right image
        org_img = img_and_mask[0][:frame_height, pic_width:]
        img_resized = cv2.resize(org_img, (new_width, new_height))
        right_inputl.append(img_resized)

        # Left mask
        org_img_mask = img_and_mask[1][:frame_height, :pic_width]
        img_mask_true_paralax = ~np.all(org_img_mask == black, axis=-1)
        img_mask_resized = np.array(
            cv2.resize(img_mask_true_paralax.astype(np.uint8) * 255, (new_width, new_height)) > 0
        ).astype(np.uint8) * 255
        left_mask_inputl.append(img_mask_resized)

        # Left image
        org_img = img_and_mask[0][:frame_height, :pic_width]
        img_resized = cv2.resize(org_img, (new_width, new_height))
        left_inputl.append(img_resized)

    right_mask_input = np.array(right_mask_inputl)
    left_mask_input  = np.array(left_mask_inputl)
    right_input      = np.array(right_inputl)
    left_input       = np.array(left_inputl)

    print("generating left side images")
    if np.all(left_mask_input == 0):
        left_frames = left_input
    else:
        left_frames = refine_frames(left_inputl, left_mask_inputl)


    print("generating right side images")
    if np.all(right_mask_input == 0):
        right_frames = right_input
    else:
        right_frames = refine_frames(right_inputl, right_mask_inputl)

    sttart = 0 if keep_first_three else 3
    eend = len(left_frames) if keep_last_three else len(left_frames) - 3

    proccessed_frames = []
    for j in range(sttart, eend):
        left_img  = cv2.resize(left_frames[j], (pic_width, frame_height))
        right_img = cv2.resize(right_frames[j], (pic_width, frame_height))

        combine = False
        if combine:

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

    output_tmp_video_file = sbs_color_video_path + "_tmp_refined.mkv"
    output_video_file = sbs_color_video_path + "_refined.mkv"
    codec = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(output_tmp_video_file, codec, fps, out_size)

    frame_buffer = []
    first_chunk = True
    last_chunk = False
    frame_n = 0
    frames_chunk = 180

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
    parser.add_argument('--num_inference_steps', default=3, type=int, help='Numer of defussion steps. More look better but is slower', required=False)
    args = parser.parse_args()


    num_inference_steps = args.num_inference_steps

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
