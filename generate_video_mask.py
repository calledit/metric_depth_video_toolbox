import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from PIL import Image
from rembg import remove, new_session

import depth_frames_helper


def rembg_mask_from_rgb(rgb_np: np.ndarray, session) -> np.ndarray:
    """
    rgb_np: HxWx3 uint8 (RGB)
    returns: HxW uint8 mask (0..255)
    """
    # rembg works well with PIL Image input
    mask_img = remove(Image.fromarray(rgb_np, mode="RGB"), only_mask=True, session=session)
    return np.asarray(mask_img, dtype=np.uint8)


def process_batch(frames_rgb, session, use_threads=False, max_workers=4):
    """
    frames_rgb: list[np.ndarray HxWx3 uint8]
    returns: list[np.ndarray HxW uint8]
    """
    if not use_threads:
        # GPU path: do them in sequence to avoid kernel contention
        return [rembg_mask_from_rgb(rgb, session) for rgb in frames_rgb]

    # CPU path: parallelize calls into rembg (it drops into C/ORT and releases GIL)
    masks = [None] * len(frames_rgb)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(rembg_mask_from_rgb, frames_rgb[i], session): i for i in range(len(frames_rgb))}
        for fut in as_completed(futures):
            i = futures[fut]
            masks[i] = fut.result()
    return masks


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
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def process_single_video(color_video_path: str, args, DEVICE: str, use_threads: bool, session) -> None:
    output_tmp_file = color_video_path + '_tmp_mask.mkv'
    output_file = color_video_path + '_mask.mkv'

    if not os.path.isfile(color_video_path):
        raise Exception(f"input color_video does not exist: {color_video_path}")

    cap = cv2.VideoCapture(color_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {color_video_path}")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    masks_out = []
    frame_n = 0
    batch_rgb = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                # flush last partial batch
                if batch_rgb:
                    masks_out.extend(process_batch(batch_rgb, session, use_threads=use_threads))
                break

            frame_n += 1
            print(f"--- frame {frame_n} ----")
            if args.max_frames != -1 and frame_n > args.max_frames:
                # flush what we have and stop
                if batch_rgb:
                    masks_out.extend(process_batch(batch_rgb, session, use_threads=use_threads))
                    batch_rgb.clear()
                break

            # Convert BGR -> RGB (rembg expects RGB)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            batch_rgb.append(rgb)

            # If batch is full, process it
            if len(batch_rgb) >= max(1, args.batch_size):
                masks_out.extend(process_batch(batch_rgb, session, use_threads=use_threads))
                batch_rgb.clear()

    finally:
        cap.release()

    # Stack masks and save via your helper
    if masks_out:
        masks_np = np.asarray(masks_out, dtype=np.uint8)
    else:
        masks_np = np.zeros((0, frame_height, frame_width), dtype=np.uint8)

    depth_frames_helper.save_grayscale_video(
        masks_np,
        output_tmp_file,
        frame_rate,
        255.0,
        frame_width,
        frame_height
    )

    depth_frames_helper.verify_and_move(output_tmp_file, len(masks_np), output_file)

    print(f"Done. Wrote: {output_file}  (frames: {len(masks_np)}, batch_size={args.batch_size}, device={DEVICE})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDVT mask video generator')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum number of frames; -1 = no limit')
    parser.add_argument('--batch_size', type=int, default=8, help='micro-batch size for rembg')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_threads = (DEVICE == 'cpu')  # threading helps CPU path; on GPU it can hurt

    # Prepare rembg session (uses onnxruntime-gpu if present)
    session = new_session()

    # -----------------------
    # Single vs Batch logic
    # -----------------------
    if _is_txt(args.color_video):
        video_list = _read_list_file(args.color_video)
        print(f"Batch mode: {len(video_list)} entries from {args.color_video}")
        for idx, vid_path in enumerate(video_list, start=1):
            print(f"\n##### [{idx}/{len(video_list)}] {vid_path} #####")
            process_single_video(vid_path, args, DEVICE, use_threads, session)
    else:
        # Single file (original behavior)
        process_single_video(args.color_video, args, DEVICE, use_threads, session)
