#!/usr/bin/env python3
import argparse
import gc
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

# ---------- utils ----------

def build_output_path(in_path: str) -> str:
    # Keep original suffix; append suffix for flow video
    p = Path(in_path)
    return str(p.with_name(p.name + "_opticalflow.mkv"))

def cv2_to_pil(frame_bgr):
    # OpenCV -> PIL RGB
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

def pil_to_cv2_bgr(pil_rgb):
    return cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)

def load_pair_pil(pil1, pil2, weights):
    # Torchvision optical-flow weights expect (img1, img2)
    t = weights.transforms()
    im1, im2 = t(pil1, pil2)     # [3,H,W] each, float32 normalized
    return im1, im2

def pad8_batch(t1, t2):
    # RAFT requires H,W divisible by 8
    _, _, H, W = t1.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return t1, t2, (H, W, 0, 0)
    t1p = F.pad(t1, (0, pad_w, 0, pad_h))  # (l,r,t,b)
    t2p = F.pad(t2, (0, pad_w, 0, pad_h))
    return t1p, t2p, (H, W, pad_h, pad_w)

def crop_flow(flow, meta):
    H, W, pad_h, pad_w = meta
    return flow[:, :, :H, :W] if (pad_h or pad_w) else flow

def free_cuda():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def planned_total_out_frames(total_in_frames, last_mode):
    if total_in_frames <= 0:
        return 0
    if last_mode == "none":
        return max(0, total_in_frames - 1)
    # prev/self -> one flow per input frame
    return total_in_frames

# ---------- core ----------

def process_pairs_batch(model, device, weights, pair_pils, amp, channels_last):
    """
    pair_pils: list of tuples (pil_t, pil_t1)
    Returns list of BGR flow frames (uint8 HxWx3)
    """
    # Build batch tensors
    im1_list, im2_list = [], []
    for pil1, pil2 in pair_pils:
        im1, im2 = load_pair_pil(pil1, pil2, weights)  # [3,H,W]
        im1_list.append(im1.unsqueeze(0))
        im2_list.append(im2.unsqueeze(0))
    t1 = torch.cat(im1_list, dim=0)
    t2 = torch.cat(im2_list, dim=0)

    # Move to device
    if channels_last and device.type == "cuda":
        t1 = t1.to(device, memory_format=torch.channels_last, non_blocking=True)
        t2 = t2.to(device, memory_format=torch.channels_last, non_blocking=True)
    else:
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)

    # /8 pad
    t1p, t2p, meta = pad8_batch(t1, t2)

    # Forward
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda" and amp))
    with ctx:
        out = model(t1p, t2p)

    # Torchvision RAFT outputs vary
    if isinstance(out, (list, tuple)):
        flow = out[-1]                      # [B,2,H',W']
    elif isinstance(out, dict) and "flow" in out:
        flow = out["flow"]
    else:
        raise TypeError(f"Unexpected RAFT output type: {type(out)}")

    flow = crop_flow(flow, meta)            # [B,2,H,W]
    flow_imgs = flow_to_image(flow.detach().cpu())  # [B,3,H,W] uint8 RGB

    # Convert to BGR for OpenCV writer
    bgr_list = []
    for k in range(flow_imgs.shape[0]):
        rgb = flow_imgs[k].permute(1, 2, 0).numpy()  # HWC
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr_list.append(bgr)

    # cleanup
    del t1, t2, t1p, t2p, flow, flow_imgs, out, im1_list, im2_list
    return bgr_list

def main():
    import numpy as np  # local import to avoid global dependency if not needed
    parser = argparse.ArgumentParser(description="Streaming RAFT-large optical flow → FFV1 MKV with progress.")
    parser.add_argument("--color_video", required=True, help="Path to input color video.")
    parser.add_argument("--batch", type=int, default=4, help="Pairs per forward pass (streamed, no full preload).")
    parser.add_argument("--last_mode", choices=["prev", "self", "none"], default="prev",
                        help="How to produce a flow for the last input frame.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast (fp16).")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last on GPU.")
    args = parser.parse_args()

    # Video IO
    cap = cv2.VideoCapture(args.color_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {args.color_video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0

    out_path = build_output_path(args.color_video)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {out_path}")

    # Model
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Progress
    total_out_plan = planned_total_out_frames(total_in, args.last_mode)
    written = 0
    start_time = time.perf_counter()
    prev_frame_end = start_time
    last_flow_bgr = None  # to duplicate if last_mode=prev
    last_pil = None       # to compute self-pair if last_mode=self

    # Streaming batch loop (sliding window)
    B = max(1, args.batch)
    frames_pil = []  # sliding buffer of PIL frames
    eof = False

    # Prime with first frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise RuntimeError("No frames in input.")
    frames_pil.append(cv2_to_pil(frame))
    last_pil = frames_pil[-1]

    try:
        with torch.inference_mode():
            while True:
                # read up to B new frames to have B+1 total (=> B pairs)
                new_count = 0
                while len(frames_pil) < B + 1:
                    ret, frame = cap.read()
                    if not ret:
                        eof = True
                        break
                    frames_pil.append(cv2_to_pil(frame))
                    last_pil = frames_pil[-1]
                    new_count += 1

                # if no pair available and EOF -> break
                if len(frames_pil) < 2:
                    break

                # Build pairs from the sliding buffer: (0,1),(1,2),...,(n-2,n-1)
                pair_pils = [(frames_pil[i], frames_pil[i+1]) for i in range(len(frames_pil) - 1)]

                # Try processing whole chunk; if OOM, split and retry
                chunk_B = len(pair_pils)
                start_idx = 0
                while start_idx < chunk_B:
                    end_idx = min(start_idx + B, chunk_B)
                    sub_pairs = pair_pils[start_idx:end_idx]

                    try:
                        bgr_list = process_pairs_batch(model, device, weights, sub_pairs,
                                                       amp=args.amp, channels_last=args.channels_last)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and device.type == "cuda":
                            # halve batch and retry this subrange
                            B = max(1, B // 2)
                            print(f"\n[OOM] Reducing batch to {B} and retrying...")
                            free_cuda()
                            continue
                        else:
                            raise

                    # Write outputs for this sub-batch
                    for bgr in bgr_list:
                        frame_start = time.perf_counter()
                        writer.write(bgr)
                        last_flow_bgr = bgr  # remember last (for last_mode=prev)
                        written += 1

                        pct = (written / total_out_plan) * 100 if total_out_plan > 0 else 0
                        avg_per_frame = (frame_start - start_time) / written if written > 0 else 0
                        rem_seconds = max(0.0, avg_per_frame * (max(0, total_out_plan - written)))
                        print(f"[{pct:5.1f}%] Frame #{written:4d}/{total_out_plan or written} "
                              f"| Remaining: {(int(rem_seconds)//60)}min{(int(rem_seconds)%60):02d}s "
                              f"| Last frame rendered in {(frame_start - prev_frame_end):6.3f}s",
                              end="\r")
                        prev_frame_end = time.perf_counter()

                    start_idx = end_idx
                    gc.collect()
                    if device.type == "cuda":
                        free_cuda()

                # Slide window: keep only the last frame as start for next chunk
                frames_pil = frames_pil[-1:]

                if eof:
                    break

            # Handle last frame per flag
            if args.last_mode == "prev" and last_flow_bgr is not None:
                # Duplicate the last computed flow as the last frame's flow
                writer.write(last_flow_bgr)
                written += 1
            elif args.last_mode == "self" and last_pil is not None:
                # Compute flow(last, last) and write
                try:
                    bgr_list = process_pairs_batch(model, device, weights, [(last_pil, last_pil)],
                                                   amp=args.amp, channels_last=args.channels_last)
                    writer.write(bgr_list[0])
                    written += 1
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device.type == "cuda":
                        free_cuda()
                        # Try plain black frame if even self-pair OOMs (unlikely)
                        writer.write(cv2.cvtColor(cv2.imread(str(build_output_path(args.color_video))) * 0, cv2.COLOR_BGR2BGR))
                    else:
                        raise

    finally:
        cap.release()
        writer.release()

    # Final newline
    print()
    print(f"Done. Wrote: {out_path}")
    if total_in > 0:
        print(f"Input frames: {total_in} | Output flow frames: {written} | last_mode: {args.last_mode}")
    else:
        print(f"Output flow frames: {written} (input frame count unknown) | last_mode: {args.last_mode}")

if __name__ == "__main__":
    # Optional env for big vids:
    #   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
    #   export CUDA_LAUNCH_BLOCKING=1
    main()

