import argparse
import numpy as np
import os
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from scipy.ndimage import binary_dilation
from stereo_rerender import infill_using_normals, masked_blur
import depth_frames_helper


import sys
sys.path.append("C:\\Users\\calle\\projects\\stereo_dissoclusion_net")
import inferance

# -----------------------
# Config / Globals
# -----------------------
num_inference_steps = None  # More steps look better but slowe set by arg
black = np.array([0, 0, 0], dtype=np.uint8)
blue = np.array([0, 0, 255], dtype=np.uint8)
black_color = np.array([0,0,0])
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
    
def blur_under_mask(img, bool_mask, ksize=(6,6), sigma=0):
    """
    Gaussian-blur only the area where bool_mask == True.
    The blur uses only masked pixels in the convolution window.
    Areas outside mask remain unchanged.

    img:        H×W×C uint8 BGR
    bool_mask:  H×W boolean (True = blur)
    ksize:      kernel size for Gaussian
    sigma:      Gaussian sigma
    """
    # 1) Gaussian kernel
    g1d = cv2.getGaussianKernel(ksize[0], sigma)
    kernel = g1d @ g1d.T

    # 2) prepare float32
    img_f = img.astype(np.float32)

    # Mask as float (1 = included in blur, 0 = excluded)
    m = bool_mask.astype(np.float32)

    # 3) Convolve masked image and the mask separately
    # weighted sum of pixels
    blurred_sum = cv2.filter2D(img_f * m[..., None], -1, kernel, borderType=cv2.BORDER_ISOLATED)
    
    # weighted sum of mask values
    weight_sum = cv2.filter2D(m, -1, kernel, borderType=cv2.BORDER_ISOLATED)

    # 4) Normalize (avoid division by zero)
    w = weight_sum[..., None]
    w_safe = np.where(w == 0, 1.0, w)

    blurred = blurred_sum / w_safe

    # 5) Combine:
    # - where mask is True → blurred value
    # - where mask is False → original
    out = img_f.copy()
    out[bool_mask] = blurred[bool_mask]

    return np.clip(out, 0, 255).astype(np.uint8)


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

def infill_using_sdiss(img, infill_mask, depth_rgb):
    
    depth_percent = depth_frames_helper.decode_rgb_depth_frame(depth_rgb, 1.0, True)
    predicted_img = inferance.infer(img, infill_mask, depth_percent)
    
    return predicted_img

def sdiss_infill(img, infill_mask, depth_rgb):
    bg_mask = np.all(infill_mask != black_color, axis=-1)
    
    
    #Fill in using the normals
    filled_in_img = infill_using_sdiss(img, infill_mask, depth_rgb)
    
    #Blur image and specifically the filled in parts as that looks better
    blured_filled_in_img = blurred = cv2.blur(filled_in_img, (4, 4))
    
    #write the now blured infill to the black areas
    img[bg_mask] = blured_filled_in_img[bg_mask]
    
    
    # Find the lower side of infill areas
    backedge_mask_blue = mark_lower_side(infill_mask)
    backedge_mask = np.all(backedge_mask_blue == blue, axis=-1)

    #make mask surounding that lower edge
    expanded_backedge_mask  = binary_dilation(backedge_mask,  iterations=6)
    
    #blur under the lower edge mask to hide the halos
    img = blur_under_mask(img, expanded_backedge_mask)
    return img
    
def process_pair(sbs_color_video_path: str, sbs_mask_video_path: str, depth_video_path: str, args):
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
    
    depth_video = cv2.VideoCapture(depth_video_path)
    d_frame_width  = int(depth_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    d_frame_height = int(depth_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    assert frame_width == m_frame_width and frame_height == m_frame_height and d_frame_width == m_frame_width and d_frame_height == m_frame_height, "mask and color video not same resolution"

    

    output_tmp_video_file = sbs_color_video_path + "_tmp_infilled.mkv"
    output_video_file = sbs_color_video_path + "_infilled.mkv"
    codec = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(output_tmp_video_file, codec, fps, out_size)

    frame_buffer = []
    first_chunk = True
    last_chunk = False
    frame_n = 0
    frames_chunk = 25
    
    pic_width = int(frame_width // 2)
    
    
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
            
            ret_depth, depth_frame = depth_video.read()
            if not ret_depth:
                # If mask video ended early, assume blank mask remainder
                mask_frame = np.zeros_like(raw_frame)
                
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
            
            # Right mask
            right_img_mask = mask_frame[:frame_height, pic_width:]
            
            # Right image
            right_org_img = rgb[:frame_height, pic_width:]
            
            # Right depth
            right_org_depth = depth_frame[:frame_height, pic_width:]

            # Left mask
            left_org_img_mask = mask_frame[:frame_height, :pic_width]
            
            # Left image
            left_org_img = rgb[:frame_height, :pic_width]
            
            # Left depth
            left_org_depth = depth_frame[:frame_height, :pic_width]
            
            left_img = sdiss_infill(left_org_img, left_org_img_mask, left_org_depth)
            
            
            right_img = sdiss_infill(right_org_img, right_img_mask, right_org_depth)
            
            out_image = cv2.hconcat([left_img, right_img])
            out_image_uint8 = np.clip(out_image, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(out_image_uint8, cv2.COLOR_RGB2BGR))

            if args.max_frames != -1 and frame_n >= args.max_frames:
                break
    finally:
        raw_video.release()
        mask_video.release()
        out.release()

    depth_frames_helper.verify_and_move(output_tmp_video_file, frame_n, output_video_file)

    print(f"Done. Wrote: {output_video_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo disoclusion infill script')
    parser.add_argument('--sbs_color_video', type=str, required=True, help='side by side stereo video renderd with point clouds in the masked area')
    parser.add_argument('--sbs_mask_video', type=str, required=True, help='side by side stereo video mask')
    parser.add_argument('--sbs_depth_video', type=str, required=True, help='side by side stereo depth video')
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    args = parser.parse_args()


    # -----------------------
    # Single vs Batch logic
    # -----------------------
    if _is_txt(args.sbs_color_video):
        if not _is_txt(args.sbs_mask_video):
            raise ValueError("If --sbs_color_video is a .txt file, then --sbs_mask_video must also be a .txt file.")

        color_list = _read_list_file(args.sbs_color_video)
        mask_list  = _read_list_file(args.sbs_mask_video)
        depth_list  = _read_list_file(args.sbs_depth_video)

        if len(color_list) != len(mask_list) or len(color_list) != len(depth_list):
            raise ValueError(
                f"List length mismatch: {args.sbs_color_video} has {len(color_list)} entries, "
                f"{args.sbs_mask_video} has {len(mask_list)} entries."
            )

        print(f"Batch mode: {len(color_list)} pairs")
        # Run up to 2 clips in parallel. GPU sections are serialized by _GPU_GATE.
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(process_pair, c_path, m_path, args)
                       for (c_path, m_path) in zip(color_list, mask_list, depth_list)]
            # Consume as they finish to keep the pool busy
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    # Surface errors but keep other jobs running
                    print(f"[ERROR] A clip failed: {e}")

    else:
        # Single-file mode (original behavior)
        process_pair(args.sbs_color_video, args.sbs_mask_video, args.sbs_depth_video, args)
