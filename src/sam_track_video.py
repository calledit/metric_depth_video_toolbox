import argparse
import numpy as np
import os
import torch
import json
import cv2

import sys
from types import SimpleNamespace

import torch.nn.functional as F
sys.path.append("base/droid_slam")
from droid import Droid
from lietorch import SE3

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mega-sam camera tracker')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--depth_video', type=str, help='depth video', required=True)
    parser.add_argument('--mask_video', type=str, help='black and white mask video for thigns that should not be tracked', required=False)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)

    args = parser.parse_args()

    if args.xfov is None and args.yfov is None:
        print("Either --xfov or --yfov is required.")
        exit(0)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    out_file = args.depth_video + "_transformations.json"

    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")

    if not os.path.isfile(args.depth_video):
        raise Exception("input depth_video does not exist")

    MODEL_maxOUTPUT_depth = args.max_depth

    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

    color_video = cv2.VideoCapture(args.color_video)

    mask_video = None
    if args.mask_video is not None:
        if not os.path.isfile(args.mask_video):
            raise Exception("input mask_video does not exist")
        mask_video = cv2.VideoCapture(args.mask_video)

    cam_matrix = compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height).astype(np.float32)


    depths = []

    next_frame = None
    next_color_frame = None
    fr_n = 0
    final_frame = False
    stream = []
    while raw_video.isOpened():

        #we need to know what frame is the last so we buffer one frame
        if next_frame is None:
            ret, next_frame = raw_video.read()
            if not ret:
                break
            ret, next_color_frame = color_video.read()
            if not ret:
                break
            if mask_video is not None:
                ret, next_mask_frame = mask_video.read()
                if not ret:
                    break


		#make last next frame this frame
        this_frame = next_frame
        this_color_frame = next_color_frame
        if mask_video is not None:
            this_mask_frame = next_mask_frame

		#Read next frame
        ret, next_frame = raw_video.read()
        if not ret and final_frame:
                break
        if not ret:
            final_frame = True

        ret, next_color_frame = color_video.read()
        if not ret and final_frame:
                break

        if mask_video is not None:
            ret, next_mask_frame = mask_video.read()
            if not ret and final_frame:
                break

		#start processing of this frame
        print("--- frame ",fr_n+1," ----")

        if args.max_frames < fr_n and args.max_frames != -1:
            if final_frame:
                break
            final_frame = True

        rgb_color = cv2.cvtColor(this_color_frame, cv2.COLOR_BGR2RGB)
        rgb_depth = cv2.cvtColor(this_color_frame, cv2.COLOR_BGR2RGB)

        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((rgb_depth[..., 0].astype(np.uint32) + rgb_depth[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb_depth[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)

        #Down scale input images by 8
        h0, w0, _ = this_frame.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        cam_matrix_torch = torch.as_tensor([cam_matrix[0,0], cam_matrix[1,1], cam_matrix[0,2], cam_matrix[1,2]])
        cam_matrix_torch[0::2] *= w1 / w0
        cam_matrix_torch[1::2] *= h1 / h0


        image = cv2.resize(rgb_color, (w1, h1), interpolation=cv2.INTER_AREA)
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]

        image = torch.as_tensor(image).permute(2, 0, 1)

        image = image[None]

        depth = torch.as_tensor(depth)
        depth = F.interpolate(
            depth[None, None], (h1, w1), mode="nearest-exact"
        ).squeeze()
        depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

        if mask_video is not None:
            mask = torch.as_tensor(cv2.cvtColor(this_mask_frame, cv2.COLOR_BGR2GRAY)).float()/255.0
            mask = F.interpolate(
                mask[None, None], (h1, w1), mode="nearest-exact"
            ).squeeze()
            mask = mask[: h1 - h1 % 8, : w1 - w1 % 8]
        else:
            mask = torch.ones_like(depth)

        if fr_n == 0:
            droid_args = {
                'image_size': [image.shape[2], image.shape[3]],
                'weights': "checkpoints/megasam_final.pth",
                'disable_vis': True,
                'stereo': False,
                'upsample': False,
                'buffer': 1024,
                'beta': 0.3,
                'filter_thresh': 2.0,
                'warmup': 8,
                'keyframe_thresh': 2.0,
                'frontend_thresh': 12.0,
                'frontend_window': 25,
                'frontend_radius': 2,
                'frontend_nms': 1,
                'backend_thresh': 16.0,
                'backend_radius': 2,
                'backend_nms': 3,
            }
            droid = Droid(SimpleNamespace(**droid_args))

        droid_input = (fr_n, image, depth, cam_matrix_torch, mask)
        stream.append(droid_input)
        if final_frame:
            droid.track_final(*droid_input)
        else:
            droid.track(*droid_input)

        fr_n += 1

    traj_est, depth_est, motion_prob = droid.terminate(
        iter(stream), #Need to make a steam here
        _opt_intr=True,
        full_ba=True,
        scene_name='output_scene',
    )

    t = traj_est.shape[0]

    estimated_intrinsics = droid.video.intrinsics[:t].cpu().numpy() * 8 #images are rescaled by 8 so intrinsics need to be upscaled again

    estimated_intrinsic = estimated_intrinsics[0]
    est_cam_matrix = np.eye(3)
    est_cam_matrix[0,0] = estimated_intrinsic[0]
    est_cam_matrix[1,1] = estimated_intrinsic[1]
    est_cam_matrix[0,2] = estimated_intrinsic[2]
    est_cam_matrix[1,2] = estimated_intrinsic[3]


    depths_o = []
    for out_depth in depth_est:
        depth = F.interpolate(
                torch.as_tensor(out_depth)[None, None], (frame_height, frame_width), mode="nearest-exact"
            ).squeeze().numpy()
        #print("depth_est:", depth.shape)
        depths_o.append(depth)

    save_24bit(np.array(depths_o), args.depth_video + '_megasam.mkv', frame_rate, MODEL_maxOUTPUT_depth)
    #print("motion_prob:", motion_prob)

    poses_th = torch.as_tensor(traj_est, device="cpu")
    cam_c2w = SE3(poses_th).inv().matrix().numpy()

    with open(out_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(cam_c2w, cls=NumpyEncoder))


    fovx, fovy = fov_from_camera_matrix(est_cam_matrix)
    print("Estimated intrinsics:", "fovx:", fovx, "fovy", fovy)