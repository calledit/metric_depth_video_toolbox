import argparse
import numpy as np
import os
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

#from transformers import CLIPVisionModelWithProjection
#from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

#from StereoCrafter.pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid

from omegaconf import OmegaConf

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"m2svid")
sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"m2svid/third_party/Hi3D-Official/")

from sgm.util import instantiate_from_config

from scipy.ndimage import binary_dilation

import depth_frames_helper
from infill_common import mark_lower_side

# -----------------------
# Config / Globals
# -----------------------
num_inference_steps = None  # More steps look better but slowe set by arg
black = np.array([0, 0, 0], dtype=np.uint8)
blue = np.array([0, 0, 255], dtype=np.uint8)
pipeline = None
apply_edge_blending = False

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

def generate_infilled_frames(input_frames, input_masks, org_imgs, fps: float):
    """GPU-heavy section — guarded by a global semaphore so only one thread runs at once."""
    global denoising_model
    # Pre-format on CPU
    org_frames = (
        torch.tensor(org_imgs)
        .permute(0, 3, 1, 2)   # [t,h,w,c] → [t,c,h,w]
        .float() / 255.0
        * 2 - 1
    )
    org_frames = org_frames.permute(1, 0, 2, 3)
    
    input_frames = (
        torch.tensor(input_frames)
        .permute(0, 3, 1, 2)
        .float() / 255.0
        * 2 - 1
    )
    input_frames = input_frames.permute(1, 0, 2, 3)

    frames_mask = (
        torch.tensor(input_masks)
        .float()
        .unsqueeze(1)          # [t, 1, h, w]
        .permute(1, 0, 2, 3)   # [1, t, h, w]
        / 255.0
        * 2 - 1
    )

    with _GPU_GATE:
        # Everything under this lock may use lots of VRAM.
        cuda_org_frames = org_frames[None].cuda()
        input_batch = {
            'video': cuda_org_frames,
            'video_2nd_view': cuda_org_frames,
            'reprojected_video': input_frames[None].cuda(),
            'reprojected_mask': frames_mask[None].cuda(),
            'fps_id': torch.tensor([fps]).cuda(),
            'caption': [""],
            "motion_bucket_id": torch.tensor([127]).cuda()
        }


        with torch.inference_mode():
            video_frames = denoising_model.generate(input_batch)['generated-video']
        
        del input_batch
        video_frames = ((video_frames[0]+1.0) /2.0).clip(0, 1).cpu().permute(1, 2, 3, 0).numpy()
        
        
        
        # Proactively free/cycle memory between clips
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return (video_frames * 255).astype(np.uint8)

def transfer_lhm_video_refmask(
    video: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray | None = None,   # (H,W) or (T,H,W); 0 = include
    single_precision: bool = True,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Linear Histogram Matching (per-frame) from a reference (image or video) to a video,
    where ONLY the reference statistics are sampled using a mask (black==0).
    Content statistics are computed per frame on the full image (no masking).
    """
    assert video.ndim == 4, "video must be (T,H,W,C)"
    T, H, W, C = video.shape
    dtype = np.float32 if single_precision else np.float64
    N = H * W

    X = video.reshape(T, N, C).astype(dtype, copy=False)

    if reference.ndim == 3:
        ref_is_video = False
        R_all = reference.astype(dtype, copy=False)
    elif reference.ndim == 4:
        ref_is_video = True
        assert reference.shape[0] == T, "reference video must have same T"
        R_all = reference.astype(dtype, copy=False)
    else:
        raise ValueError("reference must be (H,W,C) or (T,H,W,C)")

    if reference_mask is None:
        mask_T = None
    else:
        if reference_mask.ndim == 2:
            assert reference_mask.shape == (H, W)
            mask_T = np.broadcast_to(reference_mask[None, ...], (T, H, W))
        elif reference_mask.ndim == 3:
            assert reference_mask.shape == (T, H, W), "mask video must match (T,H,W)"
            mask_T = reference_mask
        else:
            raise ValueError("reference_mask must be (H,W) or (T,H,W)")
        mask_T = (mask_T == 0)  # include where == 0

    # Content stats (per frame, full image)
    mu_x = X.mean(axis=1)                                  # (T, C)
    Xc = X - mu_x[:, None, :]                              # (T, N, C)
    cov_x = np.matmul(np.transpose(Xc, (0, 2, 1)), Xc) / max(N - 1, 1)
    cov_x = 0.5 * (cov_x + np.transpose(cov_x, (0, 2, 1)))
    cov_x[:, np.arange(C), np.arange(C)] += eps

    eval_x, evec_x = np.linalg.eigh(cov_x)
    invsqrt_vals = 1.0 / np.sqrt(np.clip(eval_x, eps, None))
    tmp = evec_x * invsqrt_vals[:, None, :]
    invsqrt_x = tmp @ np.transpose(evec_x, (0, 2, 1))      # (T, C, C)

    # Reference stats (per frame, masked)
    mu_r_list, sqrt_r_list = [], []
    for t in range(T):
        R_t = R_all[t] if ref_is_video else R_all          # (H, W, C)
        Rt = R_t.reshape(-1, C)
        keep = np.ones(N, dtype=bool) if mask_T is None else mask_T[t].reshape(-1)
        if keep.sum() < C:
            keep = np.ones(N, dtype=bool)
        Rt_sel = Rt[keep]
        mu_r_t = Rt_sel.mean(axis=0)
        Rc = Rt_sel - mu_r_t
        cov_r_t = (Rc.T @ Rc) / max(len(Rt_sel) - 1, 1)
        cov_r_t = 0.5 * (cov_r_t + cov_r_t.T)
        cov_r_t[np.diag_indices(C)] += eps

        eval_r, evec_r = np.linalg.eigh(cov_r_t)
        sqrt_r_t = (evec_r * np.sqrt(np.clip(eval_r, 0, None))) @ evec_r.T

        mu_r_list.append(mu_r_t)
        sqrt_r_list.append(sqrt_r_t)

    mu_r = np.stack(mu_r_list, axis=0)            # (T, C)
    sqrt_r = np.stack(sqrt_r_list, axis=0)        # (T, C, C)

    # Apply transform
    A = np.matmul(sqrt_r, invsqrt_x)              # (T, C, C)
    Yc = np.matmul(X - mu_x[:, None, :], np.transpose(A, (0, 2, 1)))  # (T, N, C)
    Y = Yc + mu_r[:, None, :]                     # <<< fixed broadcasting here

    Y = np.clip(np.round(Y), 0, 255).astype(np.uint8).reshape(T, H, W, C)
    return Y

def apply_closing(tensor, kernel=11):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    for frame in range(tensor.shape[0]):
        img = tensor[frame, 0].numpy()
        closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        tensor[frame, 0] = torch.from_numpy(closed_img)
    tensor = (tensor > 0.5).to(tensor.dtype)
    return tensor

def deal_with_frame_chunk(keep_first_three, chunk, out, keep_last_three, frame_width, frame_height, fps):
    pic_width = int(frame_width // 2)

    # Reasonable working size for the diffusion model decode
    new_width = 512
    new_height = 512
    
    mask_size = 64

    right_input, left_input = [], []
    right_mask_input, left_mask_input = [], []
    right_col_input, left_col_input = [], []

    for img_and_mask in chunk:
        # Right mask
        org_img_mask = img_and_mask[1][:frame_height, pic_width:]
        img_mask_true_paralax = ~np.all(org_img_mask == black, axis=-1)
        img_mask_resized = np.array(
            cv2.resize(img_mask_true_paralax.astype(np.uint8) * 255, (mask_size, mask_size)) > 0
        ).astype(np.uint8) * 255
        right_mask_input.append(img_mask_resized)

        # Right image
        org_img = img_and_mask[0][:frame_height, pic_width:]
        img_resized = cv2.resize(org_img, (new_width, new_height))
        right_input.append(img_resized)
        
        # Right org image
        org_img = img_and_mask[2]
        img_resized = cv2.resize(org_img, (new_width, new_height))
        right_col_input.append(img_resized)

        # Left mask (fliplr)
        org_img_mask = np.fliplr(img_and_mask[1][:frame_height, :pic_width])
        img_mask_true_paralax = ~np.all(org_img_mask == black, axis=-1)
        
        
        img_mask_resized = np.array(
            cv2.resize(img_mask_true_paralax.astype(np.uint8) * 255, (mask_size, mask_size)) > 0
        ).astype(np.uint8) * 255
        left_mask_input.append(img_mask_resized)

        # Left image (fliplr)
        org_img = np.fliplr(img_and_mask[0][:frame_height, :pic_width])
        img_resized = cv2.resize(org_img, (new_width, new_height))
        left_input.append(img_resized)
        
        # Left org image (fliplr)
        org_img = np.fliplr(img_and_mask[2])
        img_resized = cv2.resize(org_img, (new_width, new_height))
        left_col_input.append(img_resized)

    right_mask_input = np.array(right_mask_input)
    left_mask_input  = np.array(left_mask_input)
    right_input      = np.array(right_input)
    left_input       = np.array(left_input)
    right_col_input      = np.array(right_col_input)
    left_col_input       = np.array(left_col_input)

    print("generating left side images")
    if np.all(left_mask_input == 0):
        left_frames = left_input
    else:
        left_frames = generate_infilled_frames(left_input, left_mask_input, left_col_input, fps)
        #left_frames = transfer_lhm_video_refmask(left_frames, left_input, left_mask_input)
    #print(left_frames)
    #show_imgs([left_frames[0], left_input[0], left_mask_input[0], left_col_input[0]])
    #exit()
    print("generating right side images")
    if np.all(right_mask_input == 0):
        right_frames = right_input
    else:
        right_frames = generate_infilled_frames(right_input, right_mask_input, right_col_input, fps)
        #right_frames = transfer_lhm_video_refmask(right_frames, right_input, right_mask_input)

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
        
        if apply_edge_blending:
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
        else:
            out_image_uint8 = basic_out_image_uint8
        out.write(cv2.cvtColor(out_image_uint8, cv2.COLOR_RGB2BGR))

    return proccessed_frames


def show_imgs(list_of_imgs, titles=None, cols=3, figsize=(12, 8)):
    import matplotlib.pyplot as plt
    import math
    n_images = len(list_of_imgs)
    cols = min(cols, n_images)
    rows = math.ceil(n_images / cols)

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array(axs).reshape(-1)

    for i, img in enumerate(list_of_imgs):
        # Convert PIL → NumPy
        if not hasattr(img, "ndim"):
            img = np.array(img)

        if img.ndim == 2:
            axs[i].imshow(img, cmap="gray")
        else:
            axs[i].imshow(img)

        if titles:
            axs[i].set_title(titles[i])

        axs[i].axis("off")

    # Hide empty axes
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

def process_pair(sbs_color_video_path: str, sbs_mask_video_path: str, color_video_path: str, args):
    if not os.path.isfile(sbs_color_video_path):
        raise Exception(f"input sbs_color_video does not exist: {sbs_color_video_path}")

    if not os.path.isfile(sbs_mask_video_path):
        raise Exception(f"input sbs_mask_video does not exist: {sbs_mask_video_path}")
    
    if not os.path.isfile(color_video_path):
        raise Exception(f"input sbs_mask_video does not exist: {color_video_path}")
    
    print(f"Processing: {sbs_color_video_path}")
    raw_video = cv2.VideoCapture(sbs_color_video_path)
    frame_width  = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = raw_video.get(cv2.CAP_PROP_FPS)
    out_size     = (frame_width, frame_height)

    mask_video = cv2.VideoCapture(sbs_mask_video_path)
    m_frame_width  = int(mask_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    m_frame_height = int(mask_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    org_video = cv2.VideoCapture(color_video_path)

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
            
            ret_col, col_frame = org_video.read()
            if not ret_col:
                # If mask video ended early, assume blank mask remainder
                raise Exception("org color ended early")
            col_frame = cv2.cvtColor(col_frame, cv2.COLOR_BGR2RGB)

            frame_buffer.append([rgb, mask_frame, col_frame])

            if len(frame_buffer) >= frames_chunk:
                proccessed_frames = deal_with_frame_chunk(
                    first_chunk, frame_buffer, out, last_chunk,
                    frame_width, frame_height, fps
                )
                

                if first_chunk:
                    first_chunk = False
                frame_buffer = [
                    (proccessed_frames[-6], frame_buffer[-6][1], frame_buffer[-6][2]),
                    (proccessed_frames[-5], frame_buffer[-5][1], frame_buffer[-5][2]),
                    (proccessed_frames[-4], frame_buffer[-4][1], frame_buffer[-4][2]),
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
    parser.add_argument('--color_video', type=str, required=True, help='Original input video')
    parser.add_argument('--sbs_color_video', type=str, required=True, help='side by side stereo video renderd with point clouds in the masked area')
    parser.add_argument('--sbs_mask_video', type=str, required=True, help='side by side stereo video mask')
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--num_inference_steps', default=5, type=int, help='Numer of defussion steps. More look better but is slower', required=False)
    parser.add_argument('--apply_edge_blending', action='store_true', help='applies blending of the downward facing side of edges to reduce halo effect', required=False)
    args = parser.parse_args()

    num_inference_steps = args.num_inference_steps

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    apply_edge_blending = args.apply_edge_blending
    

    # -----------------------
    # Load pipeline once (shared)
    # -----------------------
    
    config = OmegaConf.load("m2svid/configs/m2svid.yaml")
    denoising_model = instantiate_from_config(config.model).cpu()
    denoising_model.init_from_ckpt("ckpts/m2svid_weights.pt")
    denoising_model = denoising_model.cuda().half().eval()
    

    # -----------------------
    # Single vs Batch logic
    # -----------------------
    if _is_txt(args.sbs_color_video):
        if not _is_txt(args.sbs_mask_video):
            raise ValueError("If --sbs_color_video is a .txt file, then --sbs_mask_video must also be a .txt file.")

        color_list = _read_list_file(args.sbs_color_video)
        mask_list  = _read_list_file(args.sbs_mask_video)
        org_list  = _read_list_file(args.color_video)

        if len(color_list) != len(mask_list) or len(color_list) != len(org_list):
            raise ValueError(
                f"List length mismatch: {args.sbs_color_video} has {len(color_list)} entries, "
                f"{args.sbs_mask_video} has {len(mask_list)} entries,"
                f"{args.color_video} has {len(org_list)} entries."
            )

        print(f"Batch mode: {len(color_list)} pairs")
        # Run up to 2 clips in parallel. GPU sections are serialized by _GPU_GATE.
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(process_pair, c_path, m_path, o_path, args)
                       for (c_path, m_path, o_path) in zip(color_list, mask_list, org_list)]
            # Consume as they finish to keep the pool busy
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    # Surface errors but keep other jobs running
                    print(f"[ERROR] A clip failed: {e}")

    else:
        # Single-file mode (original behavior)
        process_pair(args.sbs_color_video, args.sbs_mask_video, args.color_video, args)
