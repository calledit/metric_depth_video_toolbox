import argparse
import numpy as np
import os
import torch
import cv2
import json

import sys
sys.path.append("UniK3D")
from unik3d.models import UniK3D

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

def estimate_focal_lengths(projected_points: torch.Tensor, width: int, height: int):
    """
    Estimates the camera focal lengths f_x and f_y from a projected point cloud.

    The point cloud is assumed to be generated via the projection:
        X = (u - c_x) * Z / f_x,
        Y = (v - c_y) * Z / f_y,
        Z = depth,
    with the principal point (c_x, c_y) assumed to be in the middle of the image,
    i.e., (width/2, height/2).

    This function supports input in either [B, H, W, 3] (channel-last)
    or [B, 3, H, W] (channel-first) formats.

    Args:
        projected_points (torch.Tensor): Tensor containing the 3D points.
        width (int): The width of the projected image.
        height (int): The height of the projected image.

    Returns:
        fx (torch.Tensor): Estimated focal length in the x-direction (scalar).
        fy (torch.Tensor): Estimated focal length in the y-direction (scalar).
    """
    # Convert input to channel-last format: [B, H, W, 3]
    if projected_points.ndim == 4:
        if projected_points.shape[-1] == 3:
            # Already in [B, H, W, 3]
            pass
        elif projected_points.shape[1] == 3:
            # Convert from [B, 3, H, W] to [B, H, W, 3]
            projected_points = projected_points.permute(0, 2, 3, 1)
        else:
            raise ValueError("Unsupported 4D tensor shape for projected_points")
    elif projected_points.ndim == 3:
        # Assume shape is [H, W, 3] and add batch dimension.
        projected_points = projected_points.unsqueeze(0)
    else:
        raise ValueError("projected_points must be a 3D or 4D tensor")

    B, H, W, _ = projected_points.shape
    device = projected_points.device
    dtype = projected_points.dtype

    # Assume the principal point is in the middle of the image.
    cx = width / 2.0
    cy = height / 2.0

    # Create a grid of pixel coordinates.
    # grid_y will have shape [H, W] with row indices (v)
    # grid_x will have shape [H, W] with column indices (u)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij'
    )
    # Expand to batch dimension: shape [B, H, W]
    grid_x = grid_x.unsqueeze(0).expand(B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, H, W)

    # Extract the 3D coordinates.
    X = projected_points[..., 0]  # shape [B, H, W]
    Y = projected_points[..., 1]  # shape [B, H, W]
    Z = projected_points[..., 2]  # shape [B, H, W]

    # Avoid division by zero.
    eps = 1e-6
    valid_mask_x = (X.abs() > eps) & (Z.abs() > eps)
    valid_mask_y = (Y.abs() > eps) & (Z.abs() > eps)

    # Estimate focal lengths using the pinhole camera model:
    # u = X / Z * f_x + c_x  -->  f_x = (u - c_x) * Z / X
    # v = Y / Z * f_y + c_y  -->  f_y = (v - c_y) * Z / Y
    est_fx = torch.where(valid_mask_x, (grid_x - cx) * (Z / X), torch.zeros_like(X))
    est_fy = torch.where(valid_mask_y, (grid_y - cy) * (Z / Y), torch.zeros_like(Y))

    # Compute the mean focal length from all valid pixels.
    fx = est_fx[valid_mask_x].mean() if valid_mask_x.sum() > 0 else torch.tensor(0., device=device)
    fy = est_fy[valid_mask_y].mean() if valid_mask_y.sum() > 0 else torch.tensor(0., device=device)

    return fx, fy

def compute_camera_matrix(fov_horizontal_deg, fov_vertical_deg, image_width, image_height):

    #We need one or the other
    if fov_horizontal_deg is not None:
        # Convert FoV from degrees to radians
        fov_horizontal_rad = np.deg2rad(fov_horizontal_deg)

        # Compute the focal lengths in pixels
        fx = image_width /  (2 * np.tan(fov_horizontal_rad / 2))

    if fov_vertical_deg is not None:
        # Convert FoV from degrees to radians
        fov_vertical_rad = np.deg2rad(fov_vertical_deg)

        # Compute the focal lengths in pixels
        fy = image_height /  (2 * np.tan(fov_vertical_rad / 2))

    if fov_vertical_deg is None:
        fy = fx

    if fov_horizontal_deg is None:
        fx = fy

    # Assume the principal point is at the image center
    cx = image_width / 2
    cy = image_height / 2

    # Construct the camera matrix
    camera_matrix = np.array([[fx,  0, cx],
                              [ 0, fy, cy],
                              [ 0,  0,  1]], dtype=np.float64)

    return camera_matrix

def fov_from_camera_matrix(mat):
    w = mat[0][2]*2
    h = mat[1][2]*2
    fx = mat[0][0]
    fy = mat[1][1]

    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return fov_x, fov_y

def save_24bit(frames, output_video_path, fps, max_depth_arg):
    """
    Saves depth maps encoded in the R, G and B channels of a video (to increse accuracy as when compared to gray scale)
    """
    height = frames.shape[1]
    width = frames.shape[2]

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))

    max_depth = frames.max()
    print("max metric depth: ", max_depth)

    MODEL_maxOUTPUT_depth = max_depth_arg ### pick a value slitght above max metric depth to save the depth in th video file nicly
    # if you pick a high value you will lose resolution

    # incase you did not pick a absolute value we max out (this mean each video will have depth relative to max_depth)
    # (if you want to use the video as a depth souce a absolute value is prefrable)
    if MODEL_maxOUTPUT_depth < max_depth:
        print("warning: output depth is deeper than max_depth. The depth will be clipped")

    for i in range(frames.shape[0]):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDVT UniK3D video converter')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)

    args = parser.parse_args()

    use_fov = True
    if args.xfov is None and args.yfov is None:
        use_fov = False

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")

    MODEL_maxOUTPUT_depth = args.max_depth

    raw_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

    cam_matrix_torch = None
    if use_fov:
        cam_matrix = compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height).astype(np.float32)
        cam_matrix_torch = torch.from_numpy(cam_matrix)

    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")
    
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"
    
    model = model.to(DEVICE).eval()

    depths = []

    output_video_path = args.color_video+'_depth.mkv'
    out_xfov_file = output_video_path + "_xfovs.json"
    xfovs = []

    frame_n = 0
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        frame_n += 1
        print("--- frame ",frame_n," ----")

        if args.max_frames < frame_n and args.max_frames != -1:
            break

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        predictions = model.infer(rgb_torch, cam_matrix_torch)
        depths.append(predictions["depth"].squeeze().cpu().numpy())

        fx, fy = estimate_focal_lengths(predictions['points'], frame_width, frame_height)

        cam = compute_camera_matrix(90, None, frame_width, frame_height)

        cam[0][0] = fx
        cam[1][1] = fy

        fovx, fovy = fov_from_camera_matrix(cam)
        print("fovx:", fovx, "fovy:", fovy)
        xfovs.append(float(fovx))

    with open(out_xfov_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(xfovs, cls=NumpyEncoder))

    save_24bit(np.array(depths), output_video_path, frame_rate, args.max_depth)