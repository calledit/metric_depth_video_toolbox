import argparse, os, json, cv2, torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append("mvsanywhere")

# your helpers (same as UniDepth script)
import depth_frames_helper
import depth_map_tools

# MVSA imports (same structure as run_demo.py)
from mvsanywhere.utils.model_utils import get_model_class, load_model_inference

# ---------- hard-coded MVSA paths (EDIT THESE) ----------
MVSA_CONFIG_PATH = "mvsanywhere/configs/models/mvsanywhere_model.yaml"
MVSA_WEIGHTS_PATH = "mvsanywhere/weights/mvsanywhere_hero.ckpt"
# --------------------------------------------------------


def load_transforms(json_path, n_frames):
    """Return [N,4,4] torch.float32 cam_T_world; identity if file is None/missing."""
    T_list = [np.eye(4, dtype=np.float32) for _ in range(n_frames)]
    if not json_path:
        return torch.from_numpy(np.stack(T_list, 0))
    with open(json_path, "r") as f:
        data = json.load(f)

    def to44(x):
        arr = np.array(x, dtype=np.float32)
        if arr.size == 16:
            arr = arr.reshape(4, 4)
        assert arr.shape == (4, 4)
        return arr

    if isinstance(data, list):
        for i, elem in enumerate(data[:n_frames]):
            if isinstance(elem, dict):
                if "cam_T_world" in elem:
                    T_list[i] = to44(elem["cam_T_world"])
                elif "Tcw" in elem:
                    T_list[i] = to44(elem["Tcw"])
            elif isinstance(elem, (list, tuple, np.ndarray)):
                T_list[i] = to44(elem)
    elif isinstance(data, dict):
        for k, v in data.items():
            try:
                idx = int(k)
            except:
                continue
            if 0 <= idx < n_frames:
                if isinstance(v, dict) and "cam_T_world" in v:
                    T_list[idx] = to44(v["cam_T_world"])
                else:
                    T_list[idx] = to44(v)
    return torch.from_numpy(np.stack(T_list, 0))


def to_b3hw(img_uint8):
    # BGR uint8 -> RGB float in [0,1], shape [1,3,H,W]
    rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description=( 
            'Convert a video together with camera poses into metric depth video using MVSAnywhere'
        ))
    ap.add_argument("--color_video", required=True, type=str)
    ap.add_argument("--xfov", type=float, required=False)
    ap.add_argument("--yfov", type=float, required=False)
    ap.add_argument("--transformation_file", type=str, default=None)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--window", type=int, default=7, help="ref count around target (odd recommended)")
    ap.add_argument("--resize_w", type=int, default=1024, help="model input width; keeps aspect")
    ap.add_argument("--fast_cost_volume", action="store_true")
    ap.add_argument("--max_depth", type=int, default=100)
    args = ap.parse_args()

    assert os.path.isfile(args.color_video), "input video missing"

    # ---- read video & metadata ----
    cap = cv2.VideoCapture(args.color_video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if args.max_frames > 0 and len(frames) >= args.max_frames:
            break
    cap.release()
    N = len(frames)
    assert N > 0, "No frames read"

    # ---- intrinsics via your helper (same as UniDepth script) ----
    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, W, H)
    K = torch.from_numpy(cam_matrix).float()  # [3,3]
    # promote to 4x4 for MVSA keys
    K44 = torch.eye(4, dtype=torch.float32)
    K44[:3, :3] = K
    invK44 = torch.inverse(K44)

    # ---- extrinsics per frame ----
    cam_T_world_all = load_transforms(args.transformation_file, N)  # [N,4,4], float32

    # ---- load MVSA model (hard-coded config/weights) ----
    from types import SimpleNamespace

    opts = SimpleNamespace(
        # REQUIRED so get_model_class(opts) knows which class to return
        model_type="depth_model",

        # paths you hard-coded at the top
        config_file=MVSA_CONFIG_PATH,
        load_weights_from_checkpoint=MVSA_WEIGHTS_PATH,

        # sensible defaults frequently referenced by MVSA loaders
        precision=16,                  # or 32 if you prefer full FP32
        fast_cost_volume=False,        # can be overridden by CLI later if you want
        name="mvsa_video_infer",       # harmless default

        prediction_scale=1.0,       # default scale factor
        prediction_num_scales=1,    # number of pyramid scales to predict
    )
    model_class = get_model_class(opts)
    model = load_model_inference(opts, model_class).cuda().eval()

    
    # ---- processing loop ----
    scale = args.resize_w / float(W)
    newW = int(round(W * scale))
    newH = int(round(H * scale))
    S = torch.tensor([[scale, 0, 0, 0],
                      [0, scale, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=torch.float32)
    K44_s = (S @ K44)
    invK44_s = torch.inverse(K44_s)

    half_w = max(1, args.window // 2)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    depths = []  # we’ll save via depth_frames_helper just like UniDepth

    for i in tqdm(range(N), desc="MVSA depth"):
        # reference indices
        ref_idx = []
        for d in range(-half_w, half_w + 1):
            j = i + d
            if j < 0 or j >= N or j == i:
                continue
            ref_idx.append(j)
        if len(ref_idx) == 0:
            j = min(N - 1, max(0, i + 1))
            ref_idx = [j]

        # current frame
        img_t = to_b3hw(frames[i])                                  # [1,3,H,W]
        img_rs = F.interpolate(img_t, size=(newH, newW), mode="bilinear", align_corners=False)
        
        # existing
        cur_T_cw = cam_T_world_all[i].to(torch.float32)               # [4,4]
        cur_W_tc = torch.inverse(cur_T_cw)                            # [4,4]

        cur_data = {
            "image_b3hw": img_rs.to(DEVICE, non_blocking=True),       # [1,3,h,w]

            # prediction-scale intrinsics
            "K_s0_b44": K44_s.unsqueeze(0).to(DEVICE),
            "invK_s0_b44": invK44_s.unsqueeze(0).to(DEVICE),

            # matching-scale intrinsics (MVSA expects these)
            "K_matching_b44": K44_s.unsqueeze(0).to(DEVICE),
            "invK_matching_b44": invK44_s.unsqueeze(0).to(DEVICE),

            # full-res intrinsics
            "K_full_depth_b44": K44.unsqueeze(0).to(DEVICE),
            "invK_full_depth_b44": invK44.unsqueeze(0).to(DEVICE),

            # poses (both directions)
            "cam_T_world_b44": cur_T_cw.unsqueeze(0).to(DEVICE),
            "world_T_cam_b44": cur_W_tc.unsqueeze(0).to(DEVICE),

            "frame_id_string": [f"{i:06d}"],
            "full_res_depth_b1hw": torch.zeros(1, 1, H, W, device=DEVICE),
        }


        # reference stack
        # build ref tensors as before...
        src_imgs, src_K, src_invK, src_Tcw_list = [], [], [], []
        for j in ref_idx:
            img_j = to_b3hw(frames[j])
            img_j_rs = F.interpolate(img_j, size=(newH, newW), mode="bilinear", align_corners=False)
            src_imgs.append(img_j_rs)
            src_K.append(K44_s.unsqueeze(0))
            src_invK.append(invK44_s.unsqueeze(0))
            src_Tcw_list.append(cam_T_world_all[j].unsqueeze(0))

        src_T_cw = torch.cat(src_Tcw_list, dim=0).to(torch.float32)   # [R,4,4]
        src_W_tc = torch.inverse(src_T_cw)                            # [R,4,4]
        R = src_T_cw.shape[0]

        # R = number of reference views
        src_imgs_cat = torch.cat(src_imgs, dim=0).to(DEVICE, non_blocking=True)      # [R,3,h,w]
        src_K_cat    = torch.cat(src_K,   dim=0).to(DEVICE)                           # [R,4,4]
        src_iK_cat   = torch.cat(src_invK,dim=0).to(DEVICE)                           # [R,4,4]
        src_T_cw     = torch.cat(src_Tcw_list, dim=0).to(torch.float32).to(DEVICE)    # [R,4,4]
        src_W_tc     = torch.inverse(src_T_cw)                                        # [R,4,4]

        # ★ Add batch dimension so shapes become [1,R,...]
        src_data = {
            "image_b3hw":            src_imgs_cat.unsqueeze(0),                        # [1,R,3,h,w]  ← was [R,3,h,w]
            "K_s0_b44":              src_K_cat.unsqueeze(0),                           # [1,R,4,4]
            "invK_s0_b44":           src_iK_cat.unsqueeze(0),                          # [1,R,4,4]
            "K_matching_b44":        src_K_cat.unsqueeze(0),                           # [1,R,4,4]
            "invK_matching_b44":     src_iK_cat.unsqueeze(0),                          # [1,R,4,4]
            "K_full_depth_b44":      K44.unsqueeze(0).expand(R,4,4).contiguous()
                                           .unsqueeze(0).to(DEVICE),                   # [1,R,4,4]
            "invK_full_depth_b44":   invK44.unsqueeze(0).expand(R,4,4).contiguous()
                                           .unsqueeze(0).to(DEVICE),                   # [1,R,4,4]
            "cam_T_world_b44":       src_T_cw.unsqueeze(0),                            # [1,R,4,4]
            "world_T_cam_b44":       src_W_tc.unsqueeze(0),                            # [1,R,4,4]
            "frame_id_string":       [f"{j:06d}" for j in ref_idx],                    # list is fine
        }

                
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(
                phase="test",
                cur_data=cur_data,
                src_data=src_data,
                unbatched_matching_encoder_forward=(not args.fast_cost_volume),
                return_mask=True,
                num_refinement_steps=1,
            )

        # depth at s0, upsample back to original resolution (H, W)
        #depth_s0 = outputs["depth_pred_s0_b1hw"]                         # [1,1,h,w]
        #depth_full = F.interpolate(depth_s0, size=(H, W), mode="nearest").squeeze(0).squeeze(0)

        # --- pure MVS depth ---
        #d_cv_1hw = outputs["lowest_cost_bhw"].unsqueeze(1).to(torch.float32)     # [1,1,h,w]
        # optional: keep only pixels with MVS support
        #if "overall_mask_bhw" in outputs:
        #    m_bhw = outputs["overall_mask_bhw"].to(torch.bool)                   # [1,h,w]
        #    d_cv_1hw[~m_bhw.unsqueeze(1)] = -1.0                                  # mark invalid

        #depth_full = torch.nn.functional.interpolate(d_cv_1hw, size=(H, W), mode="nearest")\
        #               .squeeze(0).squeeze(0)                                     # [H,W]


        #Not sure if this is right rescaling after the lowest cost volume
        # --- depths at their native resolutions ---
        d_ref_1hw = outputs["depth_pred_s0_b1hw"].to(torch.float32).detach().clone()      # [1,1,h_ref,w_ref]
        d_cv_1hw  = outputs["lowest_cost_bhw"].unsqueeze(1).to(torch.float32).detach()    # [1,1,h_cv,w_cv]

        # --- pick canonical resolution = cost-volume (h_cv,w_cv) ---
        h_cv, w_cv = d_cv_1hw.shape[-2], d_cv_1hw.shape[-1]

        # resample refined depth to cost-volume size
        d_ref_at_cv = F.interpolate(d_ref_1hw, size=(h_cv, w_cv), mode="nearest")         # [1,1,h_cv,w_cv]

        # get (and align) the MVS support mask, else fall back to positive depths
        if "overall_mask_bhw" in outputs:
            m_1hw = outputs["overall_mask_bhw"].to(torch.bool).unsqueeze(1)               # [1,1,hm,wm]
        elif "overall_mask_b1hw" in outputs:
            m_1hw = outputs["overall_mask_b1hw"].to(torch.bool)                           # [1,1,hm,wm]
        else:
            m_1hw = (d_cv_1hw > 0)

        # resize mask to (h_cv,w_cv) if needed
        if m_1hw.shape[-2:] != (h_cv, w_cv):
            m_1hw = F.interpolate(m_1hw.float(), size=(h_cv, w_cv), mode="nearest").to(torch.bool)

        # also require finite & positive depths
        valid = m_1hw & torch.isfinite(d_cv_1hw) & torch.isfinite(d_ref_at_cv) & (d_cv_1hw > 0) & (d_ref_at_cv > 0)

        # compute robust per-frame scale at CV resolution
        eps = 1e-6
        num   = d_cv_1hw[valid]
        den   = d_ref_at_cv[valid] + eps
        if num.numel() > 0:
            ratio = (num / den).clamp_min(0.0)
            s = torch.median(ratio).item()
        else:
            s = 1.0

        #print(s)
        # apply scale to refined depth (at CV res), then upsample to original (H,W) for saving
        d_ref_scaled_1hw = d_ref_at_cv# * s                                                  # [1,1,h_cv,w_cv]
        depth_full = F.interpolate(d_ref_scaled_1hw, size=(H, W), mode="nearest").squeeze(0).squeeze(0)
        depth_np = depth_full.cpu().numpy()
        depths.append(depth_np)

    # ---- save EXACTLY like UniDepth script ----
    out_path = args.color_video + "_depth.mkv"
    depth_frames_helper.save_depth_video(
        depths, out_path, fps, args.max_depth, W, H
    )
    print(f"Saved depth video to: {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

