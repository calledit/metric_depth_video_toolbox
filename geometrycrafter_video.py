import gc
import os
import numpy as np
import torch

from diffusers.training_utils import set_seed
import argparse
import cv2
import json

import sys
sys.path.append("GeometryCrafter")
from geometrycrafter import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid
)
from decord import VideoReader, cpu

np.set_printoptions(suppress=True, precision=4)


def project_depth_maps(depth_maps: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Projects depth maps into 3D points using the camera intrinsic matrix.

    Args:
        depth_maps (torch.Tensor): A tensor of shape [B, H, W] containing depth values.
        intrinsic (torch.Tensor): A camera intrinsic matrix. Can be of shape [3, 3] or [B, 3, 3].
                                  Expected to have fx at [0,0], fy at [1,1], cx at [0,2], cy at [1,2].

    Returns:
        projected_points (torch.Tensor): A tensor of shape [B, 3, H, W] containing the 3D points.
            For each pixel, the channels represent [X, Y, Z] in the camera coordinate system.
    """
    # Determine batch size and spatial dimensions
    if depth_maps.ndim == 3:
        B, H, W = depth_maps.shape
    else:
        raise ValueError("depth_maps must have shape [B, H, W]")

    device = depth_maps.device
    dtype = depth_maps.dtype

    # Create a meshgrid of pixel coordinates (u, v)
    # u corresponds to the horizontal pixel coordinate [0, W-1]
    # v corresponds to the vertical pixel coordinate [0, H-1]
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing='xy'
    )
    # Both grid_x and grid_y have shape [H, W]
    # Stack them so that each pixel has a (u, v) coordinate
    pixel_coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    # Expand to batch size: [B, H, W, 2]
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, H, W, 2)

    # Extract intrinsic parameters.
    # If intrinsic is a single matrix (3,3), expand to batch.
    if intrinsic.ndim == 2:
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        # Reshape scalars to allow broadcasting with pixel_coords: shape [1, 1, 1]
        fx = fx.view(1, 1, 1)
        fy = fy.view(1, 1, 1)
        cx = cx.view(1, 1, 1)
        cy = cy.view(1, 1, 1)
    elif intrinsic.ndim == 3:
        # Assuming intrinsic shape is [B, 3, 3]
        fx = intrinsic[:, 0, 0].view(B, 1, 1)
        fy = intrinsic[:, 1, 1].view(B, 1, 1)
        cx = intrinsic[:, 0, 2].view(B, 1, 1)
        cy = intrinsic[:, 1, 2].view(B, 1, 1)
    else:
        raise ValueError("intrinsic must have shape [3, 3] or [B, 3, 3]")

    # Convert pixel coordinates to normalized camera coordinates.
    # x = (u - cx) / fx, y = (v - cy) / fy.
    x_norm = (pixel_coords[..., 0] - cx) / fx  # shape [B, H, W]
    y_norm = (pixel_coords[..., 1] - cy) / fy  # shape [B, H, W]

    # Back-project the depth values:
    # X = x_norm * depth, Y = y_norm * depth, Z = depth.
    X = x_norm * depth_maps
    Y = y_norm * depth_maps
    Z = depth_maps

    # Stack the 3D coordinates and rearrange to channel-first format: [B, 3, H, W]
    projected_points = torch.stack([X, Y, Z], dim=-1)  # [B, H, W, 3]
    #projected_points = projected_points.permute(0, 3, 1, 2)  # [B, 3, H, W]

    return projected_points

max_depth_arg = 0
cam_matrix = None
xfovs = None
image_id = 0
LoadMoge = False
depth_ref_frames = []
class MoGe(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        #to make prior model work
        if LoadMoge:
            sys.path.append("MoGe")
            from moge.model import MoGeModel
            self.model = MoGeModel.from_pretrained('Ruicheng/moge-vitl').eval()


    @torch.no_grad()
    def forward_image(self, image: torch.Tensor, **kwargs):
        global image_id
        nr_images = len(image)

        # Originally i built this to stablize any depth video but it seams like the geometrycrafter net depends hevily on the exact
        # nuances of the points returned by the network and projecting new ones does not work very well.
        # But there whould be no nuances since MoGe has force_projection set to true by default
        if len(depth_ref_frames) != 0:
            if len(depth_ref_frames)+nr_images < image_id:
                raise ValueError("requested depth image "+ str(image_id)+" not loaded")
            depth_images = depth_ref_frames[image_id:image_id+nr_images]
            #print(image.shape, depth_images.shape)
            masks = depth_images != max_depth_arg
            #print("get image:", image_id, nr_images)
            #Now we have the image we need to project it and create a masks so we can send it to geometrycrafter
            if xfovs is not None:
                intr = []
                for i in range(nr_images):
                    xfov = xfovs[image_id+i]
                    cam_matrix = compute_camera_matrix(xfov, None, original_width, original_height).astype(np.float32)
                    intr.append(cam_matrix)
                intrinsics = torch.tensor(np.array(intr))
            else:
                intrinsics = torch.tensor(cam_matrix)
            points = project_depth_maps(depth_images, intrinsics)
            #print("saved_points:", points[0].cpu().numpy()[0])

        else:
            #image = (image*2) -1 #If you want to purify the input that goes in to MoGe  you keep this line see:
            # https://github.com/TencentARC/GeometryCrafter/issues/2
            output = self.model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
            points = output['points'] # b,h,w,3
            #print("model_points:", points[0].cpu().numpy()[0])
            #exit(0)
            masks = output['mask'] # b,h,w
        image_id += nr_images

        # the fact that we move all frame to cuda is dumb and a waste of cuda memmory. This neeed to be fixed in geometrycrafter fix is here:
        # https://github.com/TencentARC/GeometryCrafter/pull/1
        return points.to("cpu"), masks.to("cpu")

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



    MODEL_maxOUTPUT_depth = max_depth_arg ### pick a value slitght above max metric depth to save the depth in th video file nicly
    # if you pick a high value you will lose resolution

    if isinstance(frames, np.ndarray):
        height = frames.shape[1]
        width = frames.shape[2]
        max_depth = frames.max()
        print("max metric depth: ", max_depth)
        # incase you did not pick a absolute value we max out (this mean each video will have depth relative to max_depth)
        # (if you want to use the video as a depth souce a absolute value is prefrable)
        if MODEL_maxOUTPUT_depth < max_depth:
            print("warning: output depth is deeper than max_depth. The depth will be clipped")
        nr_frames = frames.shape[0]
    else:
        nr_frames = len(frames)
        height = frames[0].shape[0]
        width = frames[0].shape[1]

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))

    for i in range(nr_frames):
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


def compute_scale_and_shift_full(prediction, target, mask = None):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    if mask is None:
        mask = np.ones_like(target)==1
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geometrycrafter depth stablizer')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--depth_video', type=str, required=False, help='reference metric depth video used to obtain conversion constants, can be created with any image depth model')
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--max_res', type=int, default=768)
    parser.add_argument('--xfov', type=float, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--xfov_file', type=str, help='alternative to xfov and yfov, json file with one xfov for each frame', required=False)
    parser.add_argument('--yfov', type=float, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)

    args = parser.parse_args()


    use_fov = True
    if args.xfov is None and args.yfov is None:
        use_fov = False

    if args.xfov_file is not None:
        if not os.path.isfile(args.xfov_file):
            raise Exception("input xfov_file does not exist")
        with open(args.xfov_file) as json_file_handle:
            xfovs = json.load(json_file_handle)
        use_fov = True


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_maxOUTPUT_depth = args.max_depth
    max_depth_arg = args.max_depth

    print("load Geometrycrafter net")
    set_seed(42)
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='unet_diff',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    ).requires_grad_(False).to("cuda", dtype=torch.float16)

    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    ).requires_grad_(False).to("cuda", dtype=torch.float32)

    prior_model = MoGe().requires_grad_(False).to('cuda', dtype=torch.float32)

    # load weights of other components from the provided checkpoint
    pipe = GeometryCrafterDiffPipeline.from_pretrained(
		"stabilityai/stable-video-diffusion-img2vid-xt",
		unet=unet,
		torch_dtype=torch.float16,
		variant="fp16"
	).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()


    print("load color frames")
    col_video = cv2.VideoCapture(args.color_video)
    frame_rate = col_video.get(cv2.CAP_PROP_FPS)
    col_video.release()

    vid = VideoReader(args.color_video, ctx=cpu(0))
    original_height, original_width = vid.get_batch([0]).shape[1:3]

    fovx = None
    if use_fov and xfovs is None:
        cam_matrix = compute_camera_matrix(args.xfov, args.yfov, original_width, original_height).astype(np.float32)
        fovx, fovy = fov_from_camera_matrix(cam_matrix)

    craft_width = 640
    craft_height = 384

    overlap = 5
    window_size = 110

    frames_idx = list(range(0, len(vid), 1))
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32) / 255.0
    if args.max_frames > 0:
        args.max_frames = min(args.max_frames, len(frames))
        frames = frames[:args.max_frames]
    else:
        args.max_frames = len(frames)
    window_size = min(window_size, args.max_frames)
    if window_size == args.max_frames:
        overlap = 0
    frames_tensor = torch.tensor(frames.astype("float32"), device='cuda').float().permute(0, 3, 1, 2)

    if args.depth_video is not None:

        if not use_fov:
            raise ValueError("some type of FOV is needed when using reference material, either FOV from a file or as a argument")

        print("load reference depth frames")
        raw_video = cv2.VideoCapture(args.depth_video)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

        depth_ref_frames = []

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

            # Decode video depth
            depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
            depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
            depth_unit[..., 3] = rgb[..., 0].astype(np.uint32)
            depth_unit[..., 2] = rgb[..., 2]
            depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
            # resized_depth_map = cv2.resize(depth, (craft_width, craft_height), interpolation=cv2.INTER_LINEAR)
            depth_ref_frames.append(depth)

        if raw_video is not None:
            raw_video.release()

        depth_ref_frames = torch.tensor(np.array(depth_ref_frames))


    print("Run the geometrycrafter net")
    # inference the depth map using the DepthCrafter pipeline
    with torch.inference_mode():
        image_id = 0
        point_maps, valid_masks = pipe(
            frames_tensor,
            point_map_vae,
            prior_model,
            height=craft_height,
            width=craft_width,
            num_inference_steps=5,
            guidance_scale=1.0,
            window_size=window_size,
            decode_chunk_size=8,
            overlap=overlap,
            force_projection=True,
            force_fixed_focal=True,
            use_extract_interp=False,
            track_time=False
        )

        depths = point_maps[..., 2].cpu().numpy()

    print("Save depth file")
    output_video_path = args.color_video + "_depth.mkv"
    save_24bit(depths, output_video_path, frame_rate, args.max_depth)