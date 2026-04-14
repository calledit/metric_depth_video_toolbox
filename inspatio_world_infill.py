import argparse
import gc
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from threading import Semaphore
from omegaconf import OmegaConf
from safetensors.torch import load_file
from scipy.ndimage import binary_dilation

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "inspatio-world"))

from pipeline import CausalInferencePipeline
from demo_utils.memory import DynamicSwapInstaller
from utils.render_warper import convert_mask_video

import depth_frames_helper

# -----------------------
# Config / Globals
# -----------------------
black = np.array([0, 0, 0], dtype=np.uint8)
blue = np.array([0, 0, 255], dtype=np.uint8)
pipeline = None

# Generic prompt — inspatio-world accepts text conditioning but content works
# without a scene-specific description.
TEXT_PROMPT = "The image shows a scene from a video"

# Model expects inputs at this resolution (H x W).  Inputs are resized here.
TARGET_HEIGHT = 480
TARGET_WIDTH = 832

# Frames per pipeline call.  Must give T_lat = (frames_chunk+3)//4 divisible by
# num_frame_per_block (3).  225 -> T_lat=57 (19 blocks of 3); 225 = 4*57-3 so
# VAE decode recovers exactly 225 frames (no frame loss at chunk boundaries).
FRAMES_CHUNK = 225

CHECKPOINT_PATH = os.path.join(
    _script_dir, "inspatio-world", "checkpoints",
    "InSpatio-World-1.3B", "InSpatio-World-1.3B.safetensors"
)
CONFIG_PATH = os.path.join(_script_dir, "inspatio-world", "configs", "inference_1.3b.yaml")
DEFAULT_CONFIG_PATH = os.path.join(_script_dir, "inspatio-world", "configs", "default_config.yaml")

# Allow only ONE generate_infilled_frames on GPU at a time.
_GPU_GATE = Semaphore(1)

# -----------------------
# RAFT + flow-completion globals
# -----------------------
_RAFT_CKPT_PATH = os.path.join(_script_dir, "ckpts", "raft-things.pth")
_RFC_CKPT_PATH  = os.path.join(_script_dir, "ckpts", "recurrent_flow_completion.pth")
_RAFT_ALIGN_H = 480   # both RAFT and RFC run at this resolution (divisible by 8)
_RAFT_ALIGN_W = 832
_raft_model = None
_rfc_model  = None


# -----------------------
# RAFT + flow-completion helpers
# -----------------------
def _ensure_spp_path():
    """Add StereoProPainter to sys.path so its sub-packages can be imported."""
    spp = os.path.join(_script_dir, "StereoProPainter")
    if spp not in sys.path:
        sys.path.insert(0, spp)


def _load_raft(device):
    """Lazy-load the RAFT model; cached after first call."""
    global _raft_model
    if _raft_model is not None:
        return _raft_model

    _ensure_spp_path()
    from RAFT import RAFT as _RAFT  # relative imports handled inside the package

    _ns = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
    _m = torch.nn.DataParallel(_RAFT(_ns))
    _m.load_state_dict(torch.load(_RAFT_CKPT_PATH, map_location="cpu"))
    _m = _m.module
    _m.to(device)
    _m.eval()
    _raft_model = _m
    print(f"RAFT loaded from {_RAFT_CKPT_PATH}")
    return _raft_model


def _raft_prep(img_np, h, w, device):
    """(H,W,3) uint8 RGB -> (1,3,h,w) float32 in [-1,1], resized to (h,w)."""
    t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if t.shape[2] != h or t.shape[3] != w:
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return (t * 2.0 - 1.0).to(device)


def _load_rfc(device):
    """Lazy-load RecurrentFlowCompleteNet; cached after first call."""
    global _rfc_model
    if _rfc_model is not None:
        return _rfc_model
    _ensure_spp_path()
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    m = RecurrentFlowCompleteNet(_RFC_CKPT_PATH)
    for p in m.parameters():
        p.requires_grad = False
    m.to(device)
    m.eval()
    _rfc_model = m
    print(f"RecurrentFlowCompleteNet loaded from {_RFC_CKPT_PATH}")
    return _rfc_model


def _align_infilled_to_render(render_frames, infilled_frames, hole_masks, device):
    """
    Per-pixel warp to correct the latent-space drift introduced by the VAE
    round-trip.

    Strategy
    --------
    1. Pre-fill the render's holes with infilled content before running RAFT so
       that RAFT never sees artificial black regions (which produce garbage flow
       at hole boundaries).  After filling, hole areas are pixel-identical in
       both images → flow ≈ 0 there; non-hole areas differ only by the small
       VAE drift → RAFT cleanly measures that drift.
    2. Dilate the hole mask slightly so that boundary pixels (where the two
       images still differ slightly due to the seam) are also treated as
       unknown before flow completion.
    3. Run RecurrentFlowCompleteNet (ProPainter's learned flow-completion model)
       on the full temporal sequence of masked drift flows.  It uses forward +
       backward temporal context across all frames to fill holes with smooth,
       plausible drift estimates — far better than geometric extrapolation.
    4. Apply the completed dense flow per frame via cv2.remap with an OOB
       fallback (pixels whose corrected sample lands outside the frame keep
       their original infilled value instead of replicating the edge).

    Parameters
    ----------
    render_frames   : (T, H, W, 3) uint8 – per-eye render (with black holes)
    infilled_frames : (T, H, W, 3) uint8 – model output (possibly drifted)
    hole_masks      : (T, H, W)    uint8 – 255 = hole, 0 = valid surrounding area
    device          : torch.device

    Returns
    -------
    (T, H, W, 3) uint8 – aligned infilled frames
    """
    if not os.path.isfile(_RAFT_CKPT_PATH) or not os.path.isfile(_RFC_CKPT_PATH):
        missing = [p for p in (_RAFT_CKPT_PATH, _RFC_CKPT_PATH) if not os.path.isfile(p)]
        print(f"[align] Missing checkpoints {missing}, skipping alignment.")
        return infilled_frames

    raft = _load_raft(device)
    rfc  = _load_rfc(device)

    T, H, W = render_frames.shape[:3]
    rh, rw = _RAFT_ALIGN_H, _RAFT_ALIGN_W

    # ---- Step 1: compute RAFT drift flow per frame at reduced resolution ----
    flows_raft  = []   # list of (rh, rw, 2) float32
    masks_raft  = []   # list of (rh, rw) float32  1=unknown  0=known

    # Mask convention coming in: 255 = valid rendered pixel, 0 = hole to fill
    for i in range(T):
        has_valid = np.any(hole_masks[i] > 0)   # any 255 = valid pixel exists

        # Resize to RAFT resolution and convert to RFC convention (1=hole, 0=valid)
        mask_r = cv2.resize(hole_masks[i], (rw, rh), interpolation=cv2.INTER_NEAREST)
        mask_r_f = (mask_r == 0).astype(np.float32)              # 1 = hole/unknown (RFC convention)

        if not has_valid:
            flows_raft.append(np.zeros((rh, rw, 2), dtype=np.float32))
            masks_raft.append(np.ones((rh, rw), dtype=np.float32))
            continue

        # Fill render holes (0) with infilled content so RAFT never sees black
        render_filled = render_frames[i].copy()
        render_filled[hole_masks[i] == 0] = infilled_frames[i][hole_masks[i] == 0]

        with torch.no_grad():
            img1 = _raft_prep(render_filled,      rh, rw, device)
            img2 = _raft_prep(infilled_frames[i], rh, rw, device)
            _, flow_t = raft(img1, img2, iters=12, test_mode=True)

        flow_np = flow_t[0].permute(1, 2, 0).cpu().numpy()  # (rh, rw, 2)

        # Zero flow in hole pixels before passing to RFC
        flow_np[mask_r_f > 0] = 0.0

        flows_raft.append(flow_np)
        masks_raft.append(mask_r_f)

    # ---- Step 2: flow completion with RecurrentFlowCompleteNet ----
    # Pack to tensors: (1, T, 2, rh, rw) and (1, T, 1, rh, rw)
    flows_np = np.stack(flows_raft, axis=0)          # (T, rh, rw, 2)
    masks_np = np.stack(masks_raft, axis=0)          # (T, rh, rw)

    flows_t = torch.from_numpy(
        flows_np.transpose(0, 3, 1, 2)).float().unsqueeze(0).to(device)   # (1,T,2,rh,rw)
    masks_t = torch.from_numpy(
        masks_np[:, None]).float().unsqueeze(0).to(device)                # (1,T,1,rh,rw)

    masked_flows = flows_t * (1.0 - masks_t)   # zero in holes

    # Process in sub-clips so VRAM stays manageable (mirrors ProPainter's approach)
    sub_len, pad = 20, 3
    completed_parts = []
    for s in range(0, T, sub_len):
        s0 = max(0, s - pad)
        e0 = min(T, s + sub_len + pad)
        trim_s = s - s0
        trim_e = e0 - min(T, s + sub_len)

        with torch.no_grad():
            pred, _ = rfc.forward(masked_flows[:, s0:e0], masks_t[:, s0:e0])

        # In known regions keep original; in holes use predicted
        sub_masks = masks_t[:, s0:e0]
        sub_mf    = masked_flows[:, s0:e0]
        combined  = pred * sub_masks + sub_mf * (1.0 - sub_masks)

        # Trim padding overlap
        end_trim = combined.shape[1] - trim_e if trim_e > 0 else combined.shape[1]
        completed_parts.append(combined[:, trim_s:end_trim])

    completed = torch.cat(completed_parts, dim=1)   # (1, T, 2, rh, rw)

    # ---- Step 3: apply completed flow per frame ----
    aligned = infilled_frames.copy()
    for i in range(T):
        flow_np = completed[0, i].permute(1, 2, 0).cpu().numpy()  # (rh, rw, 2)

        # Scale to full frame resolution
        flow_full = cv2.resize(flow_np, (W, H), interpolation=cv2.INTER_LINEAR)
        flow_full[:, :, 0] *= W / rw
        flow_full[:, :, 1] *= H / rh

        ys_g, xs_g = np.mgrid[0:H, 0:W].astype(np.float32)
        map_x = (xs_g + flow_full[:, :, 0]).astype(np.float32)
        map_y = (ys_g + flow_full[:, :, 1]).astype(np.float32)

        remapped = cv2.remap(infilled_frames[i], map_x, map_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # OOB fallback: if drift pushes sample outside frame, keep original pixel
        oob = (map_x < 0) | (map_x > W - 1) | (map_y < 0) | (map_y > H - 1)
        remapped[oob] = infilled_frames[i][oob]
        aligned[i] = remapped

    return aligned


# -----------------------
# Helpers for batch mode
# -----------------------
def _is_txt(path: str) -> bool:
    return isinstance(path, str) and path.lower().endswith(".txt")


def _read_list_file(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


# -----------------------
# Latent frame helpers
# -----------------------
def _latent_frames(T: int) -> int:
    """Number of latent frames produced by Wan VAE for T pixel frames."""
    return (T + 3) // 4


def _pad_to_valid_T(T: int, num_frame_per_block: int = 3) -> int:
    """Return smallest T' >= T such that:
    1. _latent_frames(T') is divisible by num_frame_per_block, AND
    2. T' is of the form 4k-3 so VAE decode recovers exactly T' frames.
    """
    T_lat = _latent_frames(T)
    # Find smallest multiple of num_frame_per_block >= T_lat
    # such that (candidate * 4 - 3) >= T.
    candidate = ((T_lat + num_frame_per_block - 1) // num_frame_per_block) * num_frame_per_block
    while candidate * 4 - 3 < T:
        candidate += num_frame_per_block
    return candidate * 4 - 3


# -----------------------
# Tensor conversion helpers
# -----------------------
def _to_bcthw(frames_nhwc: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (1,3,T,H,W) float32 in [-1,1]."""
    t = torch.from_numpy(frames_nhwc.copy()).float() / 127.5 - 1.0
    return t.permute(3, 0, 1, 2).unsqueeze(0)


def _mask_to_bcthw(masks_thw: np.ndarray) -> torch.Tensor:
    """(T,H,W) uint8 -> (1,3,T,H,W) float32 in [-1,1].  First channel used by convert_mask_video."""
    t = torch.from_numpy(masks_thw.copy()).float() / 127.5 - 1.0  # (T,H,W)
    t = t.unsqueeze(0).unsqueeze(0)  # (1,1,T,H,W)
    return t.expand(-1, 3, -1, -1, -1)  # (1,3,T,H,W)


# -----------------------
# Core inference
# -----------------------
def generate_infilled_frames(
    source_frames: np.ndarray,
    render_frames: np.ndarray,
    mask_frames: np.ndarray,
    fps: float,
    ref_latent: torch.Tensor = None,
) -> np.ndarray:
    """
    GPU-heavy inference — guarded by _GPU_GATE.

    source_frames : (T, H, W, 3) uint8 — original source video
    render_frames : (T, H, W, 3) uint8 — warped/rendered frames (render view)
    mask_frames   : (T, H, W)    uint8 — 255 = hole to fill, 0 = keep source
    ref_latent    : pre-encoded source latent (optional); if given, skips encoding source.
    Returns       : (T, H, W, 3) uint8 infilled frames at same resolution as input
    """
    global pipeline

    T_orig, H, W = source_frames.shape[:3]

    # Resize to model's expected resolution
    def _resize_frames(arr_nhwc):
        if arr_nhwc.shape[1] == TARGET_HEIGHT and arr_nhwc.shape[2] == TARGET_WIDTH:
            return arr_nhwc
        return np.stack(
            [cv2.resize(f, (TARGET_WIDTH, TARGET_HEIGHT)) for f in arr_nhwc], axis=0
        )

    def _resize_masks(arr_nhw):
        if arr_nhw.shape[1] == TARGET_HEIGHT and arr_nhw.shape[2] == TARGET_WIDTH:
            return arr_nhw
        return np.stack(
            [cv2.resize(f, (TARGET_WIDTH, TARGET_HEIGHT)) for f in arr_nhw], axis=0
        )

    rnd_r = _resize_frames(render_frames)
    msk_r = _resize_masks(mask_frames)

    # Pad to a frame count whose latent count is divisible by num_frame_per_block
    T_padded = _pad_to_valid_T(T_orig)
    if T_padded > T_orig:
        pad = T_padded - T_orig
        rnd_r = np.concatenate([rnd_r, np.repeat(rnd_r[-1:], pad, axis=0)], axis=0)
        msk_r = np.concatenate([msk_r, np.repeat(msk_r[-1:], pad, axis=0)], axis=0)

    with _GPU_GATE:
        device = next(pipeline.generator.parameters()).device
        dtype = torch.bfloat16

        # Encode render and mask — VAE lives permanently on GPU
        render_latent = pipeline.vae.encode_to_latent(
            _to_bcthw(rnd_r).to(device, dtype)).to(dtype)
        pipeline.vae.model.clear_cache()

        mask_latent = convert_mask_video(
            _mask_to_bcthw(msk_r).to(device, dtype)).to(dtype)

        # Use pre-encoded ref_latent if provided, otherwise encode source now
        if ref_latent is None:
            src_r = _resize_frames(source_frames)
            if T_padded > T_orig:
                src_r = np.concatenate([src_r, np.repeat(src_r[-1:], T_padded - T_orig, axis=0)], axis=0)
            ref_latent = pipeline.vae.encode_to_latent(
                _to_bcthw(src_r).to(device, dtype)).to(dtype)
            pipeline.vae.model.clear_cache()

        T_lat = ref_latent.shape[1]
        lat_h = ref_latent.shape[3]
        lat_w = ref_latent.shape[4]

        noise = torch.randn([1, T_lat, 16, lat_h, lat_w], device=device, dtype=dtype)

        result = pipeline.inference(
            noise=noise,
            text_prompts=[TEXT_PROMPT],
            ref_latent=ref_latent,
            render_latent=render_latent,
            mask_latent=mask_latent,
            decode=False,  # decode separately so VAE can be offloaded
        )
        # result: (1, T_lat, 16, lat_h, lat_w) raw latents

        # Decode — VAE is permanently on GPU
        video = pipeline.vae.decode_to_pixel(result, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        pipeline.vae.model.clear_cache()

        frames_model = video[0].permute(0, 2, 3, 1).cpu().numpy()  # (T_decoded, H_model, W_model, 3)
        frames_model = np.clip(frames_model * 255, 0, 255).astype(np.uint8)
        frames_model = frames_model[:T_orig]  # trim padding

    # GPU tensors (result, video, noise, render_latent, mask_latent) are now out
    # of scope — CPython ref-counting has freed them.  Release cached CUDA memory.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Resize back to original per-eye resolution
    if H != TARGET_HEIGHT or W != TARGET_WIDTH:
        frames_model = np.stack(
            [cv2.resize(f, (W, H)) for f in frames_model], axis=0
        )

    return frames_model


# -----------------------
# Chunk processing
# -----------------------
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


def deal_with_frame_chunk(
    keep_first_three, chunk, out, keep_last_three,
    frame_width, frame_height, fps, apply_edge_blending
):
    """
    chunk entries: [sbs_render_rgb, sbs_mask_rgb, org_color_rgb]
      sbs_render_rgb : (H, W_full, 3) — full SBS rendered frame (left|right)
      sbs_mask_rgb   : (H, W_full, 3) — full SBS mask frame
      org_color_rgb  : (H, W_eye,  3) — original source frame (single eye width)
    """
    pic_width = int(frame_width // 2)

    right_source, left_source = [], []
    right_render, left_render = [], []
    right_mask, left_mask = [], []

    for sbs_render, sbs_mask, org_color in chunk:
        H = frame_height

        # --- Right eye ---
        rnd_right = sbs_render[:H, pic_width:]
        msk_right_rgb = sbs_mask[:H, pic_width:]
        msk_right = (np.all(msk_right_rgb == black, axis=-1)).astype(np.uint8) * 255
        right_render.append(rnd_right)
        right_mask.append(msk_right)
        right_source.append(org_color[:H])

        # --- Left eye ---
        rnd_left = sbs_render[:H, :pic_width]
        msk_left_rgb = sbs_mask[:H, :pic_width]
        msk_left = (np.all(msk_left_rgb == black, axis=-1)).astype(np.uint8) * 255
        left_render.append(rnd_left)
        left_mask.append(msk_left)
        left_source.append(org_color[:H])

    right_source = np.array(right_source)
    left_source  = np.array(left_source)
    right_render = np.array(right_render)
    left_render  = np.array(left_render)
    right_mask   = np.array(right_mask)
    left_mask    = np.array(left_mask)

    # Encode source once — left and right eyes share the same source frames
    shared_ref_latent = None
    if not np.all(left_mask == 255) or not np.all(right_mask == 255):
        device = next(pipeline.generator.parameters()).device
        dtype = torch.bfloat16
        src_arr = np.array(left_source)  # same as right_source
        T_orig_src = src_arr.shape[0]
        src_r = src_arr
        if src_r.shape[1] != TARGET_HEIGHT or src_r.shape[2] != TARGET_WIDTH:
            src_r = np.stack([cv2.resize(f, (TARGET_WIDTH, TARGET_HEIGHT)) for f in src_r], axis=0)
        T_padded = _pad_to_valid_T(T_orig_src)
        if T_padded > T_orig_src:
            src_r = np.concatenate([src_r, np.repeat(src_r[-1:], T_padded - T_orig_src, axis=0)], axis=0)
        shared_ref_latent = pipeline.vae.encode_to_latent(
            _to_bcthw(src_r).to(device, dtype)).to(dtype)
        pipeline.vae.model.clear_cache()

    print("generating left side images")
    if np.all(left_mask == 255):
        left_frames = left_render
    else:
        left_frames = generate_infilled_frames(left_source, left_render, left_mask, fps, ref_latent=shared_ref_latent)

    print("generating right side images")
    if np.all(right_mask == 255):
        right_frames = right_render
    else:
        right_frames = generate_infilled_frames(right_source, right_render, right_mask, fps, ref_latent=shared_ref_latent)

    # Align infilled frames to render coordinate space to correct VAE drift.
    # RAFT flow is estimated in the non-hole (surrounding) pixels only; an
    # affine is then fitted via RANSAC and applied to the full frame.
    device = next(pipeline.generator.parameters()).device
    print("aligning left infilled frames")
    left_frames  = _align_infilled_to_render(left_render,  left_frames,  left_mask,  device)
    print("aligning right infilled frames")
    right_frames = _align_infilled_to_render(right_render, right_frames, right_mask, device)

    start = 0 if keep_first_three else 3
    end = len(left_frames) if keep_last_three else len(left_frames) - 3

    processed_frames = []
    for j in range(start, end):
        left_img  = cv2.resize(left_frames[j], (pic_width, frame_height))
        right_img = cv2.resize(right_frames[j], (pic_width, frame_height))

        # Composite: source pixels where mask is black, model output elsewhere
        sbs_render_j, sbs_mask_j, _ = chunk[j]
        right_org = sbs_render_j[:frame_height, pic_width:].copy()
        left_org  = sbs_render_j[:frame_height, :pic_width].copy()
        right_m   = sbs_mask_j[:frame_height, pic_width:]
        left_m    = sbs_mask_j[:frame_height, :pic_width]

        right_hole = ~np.all(right_m == black, axis=-1)
        left_hole  = ~np.all(left_m == black, axis=-1)

        left_org[left_hole]   = left_img[left_hole]
        right_org[right_hole] = right_img[right_hole]

        basic_out = cv2.hconcat([left_org, right_org])
        basic_out = np.clip(basic_out, 0, 255).astype(np.uint8)
        processed_frames.append(basic_out)

        if apply_edge_blending:
            right_blue = mark_lower_side(right_m)
            right_edge = binary_dilation(np.all(right_blue == blue, axis=-1), iterations=3)
            left_blue  = mark_lower_side(left_m)
            left_edge  = binary_dilation(np.all(left_blue == blue, axis=-1), iterations=3)

            right_alpha = cv2.GaussianBlur(right_edge.astype(np.float32), (5, 5), 0)[..., np.newaxis]
            left_alpha  = cv2.GaussianBlur(left_edge.astype(np.float32),  (5, 5), 0)[..., np.newaxis]

            blended_left  = left_alpha  * left_img  + (1 - left_alpha)  * left_org
            blended_right = right_alpha * right_img + (1 - right_alpha) * right_org

            out_image = cv2.hconcat([blended_left, blended_right])
            out_image = np.clip(out_image, 0, 255).astype(np.uint8)
        else:
            out_image = basic_out

        out.write(cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))

    return processed_frames


# -----------------------
# Per-file processing
# -----------------------
def process_pair(
    sbs_color_video_path: str,
    sbs_mask_video_path: str,
    color_video_path: str,
    args,
):
    for path, label in [
        (sbs_color_video_path, "sbs_color_video"),
        (sbs_mask_video_path,  "sbs_mask_video"),
        (color_video_path,     "color_video"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")

    print(f"Processing: {sbs_color_video_path}")

    raw_video  = cv2.VideoCapture(sbs_color_video_path)
    mask_video = cv2.VideoCapture(sbs_mask_video_path)
    org_video  = cv2.VideoCapture(color_video_path)

    frame_width  = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = raw_video.get(cv2.CAP_PROP_FPS)
    out_size     = (frame_width, frame_height)

    m_w = int(mask_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    m_h = int(mask_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert frame_width == m_w and frame_height == m_h, "mask and SBS color video have different resolutions"

    output_tmp  = sbs_color_video_path + "_tmp_infilled.mkv"
    output_file = sbs_color_video_path + "_infilled.mkv"
    codec = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(output_tmp, codec, fps, out_size)

    frame_buffer = []
    first_chunk = True
    last_chunk  = False
    frame_n = 0

    try:
        while raw_video.isOpened():
            print(f"Frame: {frame_n}  ({frame_n / max(fps, 1e-6):.2f}s)")

            ret, raw_frame = raw_video.read()
            if not ret:
                break
            frame_n += 1

            sbs_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

            ret_m, mask_frame = mask_video.read()
            if not ret_m:
                mask_frame = np.zeros_like(raw_frame)
            mask_rgb = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)

            ret_c, col_frame = org_video.read()
            if not ret_c:
                raise RuntimeError("Original color video ended before SBS video")
            col_rgb = cv2.cvtColor(col_frame, cv2.COLOR_BGR2RGB)

            frame_buffer.append([sbs_rgb, mask_rgb, col_rgb])

            if len(frame_buffer) >= FRAMES_CHUNK:
                processed = deal_with_frame_chunk(
                    first_chunk, frame_buffer, out, last_chunk,
                    frame_width, frame_height, fps,
                    getattr(args, "apply_edge_blending", False),
                )
                if first_chunk:
                    first_chunk = False

                # Keep 6-frame overlap: last 3 as processed output + last 3 as originals
                frame_buffer = [
                    (processed[-6], frame_buffer[-6][1], frame_buffer[-6][2]),
                    (processed[-5], frame_buffer[-5][1], frame_buffer[-5][2]),
                    (processed[-4], frame_buffer[-4][1], frame_buffer[-4][2]),
                    frame_buffer[-3],
                    frame_buffer[-2],
                    frame_buffer[-1],
                ]

            if args.max_frames != -1 and frame_n >= args.max_frames:
                break

        last_chunk = True
        deal_with_frame_chunk(
            first_chunk, frame_buffer, out, last_chunk,
            frame_width, frame_height, fps,
            getattr(args, "apply_edge_blending", False),
        )
    finally:
        raw_video.release()
        mask_video.release()
        org_video.release()
        out.release()

    depth_frames_helper.verify_and_move(output_tmp, frame_n, output_file)
    print(f"Done.  Wrote: {output_file}")


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InSPAtio-World stereo infill script")
    parser.add_argument("--color_video",      type=str, required=True,
                        help="Original source video (single-eye width)")
    parser.add_argument("--sbs_color_video",  type=str, required=True,
                        help="Side-by-side rendered video (left|right) with point-cloud fill")
    parser.add_argument("--sbs_mask_video",   type=str, required=True,
                        help="Side-by-side mask video (white = hole to fill)")
    parser.add_argument("--max_frames",       type=int, default=-1,
                        help="Stop after this many frames (-1 = all)")
    parser.add_argument("--apply_edge_blending", action="store_true",
                        help="Blend edges to reduce halo artifacts")
    parser.add_argument("--text_prompt",      type=str, default=TEXT_PROMPT,
                        help="Text description of the scene (generic prompt works fine)")
    args = parser.parse_args()

    if args.text_prompt != TEXT_PROMPT:
        TEXT_PROMPT = args.text_prompt

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Download weights if missing
    # -----------------------
    _wan_dir = os.path.join(_script_dir, "inspatio-world", "checkpoints", "Wan2.1-T2V-1.3B")
    _weights_needed = [
        CHECKPOINT_PATH,
        os.path.join(_wan_dir, "Wan2.1_VAE.pth"),
        os.path.join(_wan_dir, "models_t5_umt5-xxl-enc-bf16.safetensors"),
    ]
    if any(not os.path.exists(p) for p in _weights_needed):
        print("inspatio-world weights not found, downloading...")
        import subprocess as _sp
        _sp.run([sys.executable, os.path.join(_script_dir, "download_weights.py"), "inspatio_world"], check=True)

    if not os.path.exists(_RAFT_CKPT_PATH) or not os.path.exists(_RFC_CKPT_PATH):
        print("RAFT / flow-completion checkpoints not found, downloading...")
        import subprocess as _sp
        _sp.run([sys.executable, os.path.join(_script_dir, "download_weights.py"), "raft"], check=True)

    # -----------------------
    # Load pipeline
    # -----------------------
    # WanDiffusionWrapper checks dist.is_initialized() and calls get_rank() or
    # init_process_group depending on the result.  Initialize it here so both
    # paths work correctly for single-process use.
    # WanDiffusionWrapper checks dist.is_initialized()/get_rank() or calls
    # init_process_group — both paths fail on this PyTorch build for single-process
    # use.  Patch the module so it looks already initialized as rank 0.
    import torch.distributed as dist
    dist.is_initialized = lambda: True
    dist.get_rank = lambda *args, **kwargs: 0
    dist.get_world_size = lambda *args, **kwargs: 1
    dist.barrier = lambda *args, **kwargs: None

    print("Loading inspatio-world pipeline...")
    config = OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG_PATH), OmegaConf.load(CONFIG_PATH))
    inspatio_dir = os.path.join(_script_dir, "inspatio-world")
    config.wan_model_folder = os.path.join(inspatio_dir, "checkpoints", "Wan2.1-T2V-1.3B")
    for w in config.get("generator", {}).get("weight_list", []):
        if w.get("path", "").startswith("./"):
            w["path"] = os.path.join(inspatio_dir, w["path"][2:])

    pipeline = CausalInferencePipeline(config, device=DEVICE)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    state_dict = load_file(CHECKPOINT_PATH)
    mismatch, missing = pipeline.generator.load_state_dict(state_dict, strict=False)
    print(f"  Mismatch keys: {len(mismatch)}  Missing keys: {len(missing)}")

    pipeline = pipeline.to(dtype=torch.bfloat16)
    # Text encoder (T5-XXL, ~7 GB) stays on CPU; DynamicSwapInstaller creates
    # temporary GPU copies of each layer only during the forward pass.
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=DEVICE)
    pipeline.vae.to(device=DEVICE)
    pipeline.generator.to(device=DEVICE)

    # Disable gradient tracking globally — inference only, no gradients needed.
    # This prevents the computation graph from accumulating during the last
    # denoising step of each block, which saves significant VRAM.
    torch.set_grad_enabled(False)

    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=DEVICE)
    print("Pipeline loaded.")

    # -----------------------
    # Single vs batch mode
    # -----------------------
    if _is_txt(args.sbs_color_video):
        if not _is_txt(args.sbs_mask_video) or not _is_txt(args.color_video):
            raise ValueError(
                "When --sbs_color_video is a .txt list, "
                "--sbs_mask_video and --color_video must also be .txt lists."
            )
        color_list  = _read_list_file(args.sbs_color_video)
        mask_list   = _read_list_file(args.sbs_mask_video)
        org_list    = _read_list_file(args.color_video)
        if len(color_list) != len(mask_list) or len(color_list) != len(org_list):
            raise ValueError(
                f"List length mismatch: {len(color_list)} color, "
                f"{len(mask_list)} mask, {len(org_list)} org."
            )
        print(f"Batch mode: {len(color_list)} pairs")
        for c, m, o in zip(color_list, mask_list, org_list):
            try:
                process_pair(c, m, o, args)
            except Exception as e:
                print(f"[ERROR] A clip failed: {e}")
    else:
        process_pair(args.sbs_color_video, args.sbs_mask_video, args.color_video, args)
