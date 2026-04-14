import numpy as np


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

    Processes one frame at a time to keep peak memory low.
    """
    assert video.ndim == 4, "video must be (T,H,W,C)"
    T, H, W, C = video.shape
    dtype = np.float32 if single_precision else np.float64
    N = H * W

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

    diag = np.arange(C)
    out = np.empty_like(video)

    for t in range(T):
        X = video[t].reshape(N, C).astype(dtype)

        # Content stats
        mu_x = X.mean(axis=0)
        Xc = X - mu_x
        cov_x = (Xc.T @ Xc) / max(N - 1, 1)
        cov_x = 0.5 * (cov_x + cov_x.T)
        cov_x[diag, diag] += eps
        eval_x, evec_x = np.linalg.eigh(cov_x)
        invsqrt_x = (evec_x * (1.0 / np.sqrt(np.clip(eval_x, eps, None)))) @ evec_x.T

        # Reference stats (masked)
        R_t = R_all[t] if ref_is_video else R_all
        Rt = R_t.reshape(-1, C)
        keep = np.ones(N, dtype=bool) if mask_T is None else mask_T[t].reshape(-1)
        if keep.sum() < C:
            keep = np.ones(N, dtype=bool)
        Rt_sel = Rt[keep]
        mu_r = Rt_sel.mean(axis=0)
        Rc = Rt_sel - mu_r
        cov_r = (Rc.T @ Rc) / max(len(Rt_sel) - 1, 1)
        cov_r = 0.5 * (cov_r + cov_r.T)
        cov_r[diag, diag] += eps
        eval_r, evec_r = np.linalg.eigh(cov_r)
        sqrt_r = (evec_r * np.sqrt(np.clip(eval_r, 0, None))) @ evec_r.T

        # Apply transform: Y = (X - mu_x) @ A.T + mu_r
        A = sqrt_r @ invsqrt_x
        Y = Xc @ A.T + mu_r

        out[t] = np.clip(np.round(Y), 0, 255).astype(np.uint8).reshape(H, W, C)

    return out
