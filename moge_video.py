import argparse
import numpy as np
import os
import torch
import cv2

import json

import sys
sys.path.append("MoGe")


from moge.model import MoGeModel

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

def fov_from_camera_matrix(mat):
    w = mat[0][2]*2
    h = mat[1][2]*2
    fx = mat[0][0]
    fy = mat[1][1]

    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return fov_x, fov_y


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
    parser = argparse.ArgumentParser(description='Video Depth Anything Unidepth video converter')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--xfov', type=float, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=float, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)

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

    fovx = None
    if use_fov:
        cam_matrix = compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height).astype(np.float32)
        fovx, fovy = fov_from_camera_matrix(cam_matrix)

    model = MoGeModel.from_pretrained('Ruicheng/moge-vitl').to(DEVICE).eval()


    output_video_path = args.color_video+'_depth.mkv'
    out_xfov_file = output_video_path + "_xfovs.json"

    xfovs = []

    depths = []

    frame_n = 0
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        frame_n += 1
        print("--- frame ",frame_n," ----")

        if args.max_frames < frame_n and args.max_frames != -1:
            break

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_tensor = torch.tensor(rgb / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

        output = model.infer(image_tensor, fov_x=fovx)


        estimted_cam_matrix = output['intrinsics'].cpu().numpy()
        estimated_fovx, estimated_fovy = fov_from_camera_matrix(estimted_cam_matrix)
        xfovs.append(float(estimated_fovx))
        print("fovx, fovy:", estimated_fovx, estimated_fovy)

        dpth = output['depth'].cpu().numpy()
        dpth = np.nan_to_num(dpth, nan=args.max_depth)#nan is the sky or somthing like that
        depths.append(dpth)

    with open(out_xfov_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(xfovs, cls=NumpyEncoder))

    save_24bit(np.array(depths), output_video_path, frame_rate, args.max_depth)