import argparse
import cv2
import numpy as np
import os
import copy
import time
import json
import math
import depth_frames_helper

import open3d as o3d
import depth_map_tools
from contextlib import contextmanager

@contextmanager
def timer(name = 'not named'):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.6f} seconds")

# Set numpy print options for better readability
np.set_printoptions(suppress=True, precision=4)

def convert_to_equirectangular(image, input_fov=100):
    """
    Maps an input rectilinear image rendered at a limited FOV (e.g., 100°)
    into a 180° equirectangular image while keeping the output size the same as the input.
    The valid image (representing the central 100°) is centered,
    with black padding on the sides and top/bottom.

    Parameters:
      image: Input image (H x W x 3, np.uint8)
      input_fov: Field of view (in degrees) of the input image. Default is 100.

    Returns:
      A new image (np.uint8) of the same shape as input, representing a 180° equirectangular projection.
    """
    # Get image dimensions.
    H, W = image.shape[:2]
    # Center coordinates of the input image.
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # For the output, we want to cover a horizontal range of [-90, 90] degrees.
    # Create a grid of pixel coordinates for the output image.
    x_coords = np.linspace(0, W - 1, W)
    y_coords = np.linspace(0, H - 1, H)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Map output pixel positions to spherical angles.
    # Horizontal: x -> theta in [-pi/2, pi/2]
    theta = (grid_x - cx) / cx * (np.pi / 2)
    # Vertical: y -> phi in [-pi/2, pi/2]
    phi = (grid_y - cy) / cy * (np.pi / 2)

    # The input image covers only a limited field of view.
    half_input_fov = np.radians(input_fov / 2.0)  # e.g. 50° in radians.

    # For a pinhole model, the relationship is: u = f * tan(theta) + cx.
    # Compute the effective focal lengths (assuming symmetric FOV horizontally and vertically).
    f_x = cx / np.tan(half_input_fov)
    f_y = cy / np.tan(half_input_fov)

    # Create a mask: valid if the output angle is within the input's FOV.
    valid_mask = (np.abs(theta) <= half_input_fov) & (np.abs(phi) <= half_input_fov)

    # For valid pixels, compute the corresponding input coordinates.
    # (These equations invert the pinhole projection: theta = arctan((u-cx)/f))
    map_x = f_x * np.tan(theta) + cx
    map_y = f_y * np.tan(phi) + cy

    # For invalid pixels (outside the input FOV), assign dummy values.
    # We'll set them to -1 so that cv2.remap (with BORDER_CONSTANT) returns black.
    map_x[~valid_mask] = -1
    map_y[~valid_mask] = -1

    # Convert mapping arrays to float32 (required by cv2.remap).
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Remap the image. Pixels with mapping -1 will be filled with borderValue.
    equirect_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return equirect_img


def make_infill_mask(boolean_mask, normals):
    # Placeholder stub: user to implement infill mask generator
    return None


def convergence_angle(distance, pupillary_distance):
    """
    Calculate the convergence angles for both eyes to point to an object.

    Parameters:
        distance (float): The distance to the object.
        pupillary_distance (float): The distance between the centers of the two pupils.

    Returns:
            - angle_per_eye: The angle each eye must rotate inward from the midline.
    """
    if distance == 0:
        raise ValueError("Distance must be non-zero to compute a valid angle.")

    # Calculate the angle for one eye in radians using arctan((pupillary_distance/2) / distance)
    angle_per_eye = math.atan((pupillary_distance / 2) / distance)


    return angle_per_eye

def masked_blur(img, ksize=(6,6), sigma=0):
    """
    Gaussian‑blurs `img` but ignores pure black pixels when computing each output pixel.
    Black pixels (0,0,0) act like “transparent” in the kernel.
    
    img:     H×W×C uint8 BGR image
    ksize:   blur kernel size
    sigma:   gaussian sigma (0 = auto)
    """
    # 1) Build your Gaussian kernel (1D then outer‑product → 2D)
    g1d = cv2.getGaussianKernel(ksize[0], sigma)
    kernel = g1d @ g1d.T

    # 2) Make a mask of “valid” pixels (1 where img != black)
    #    For color: consider black only if all channels are zero
    black_mask = np.all(img == 0, axis=2)
    valid_mask = ~black_mask
    valid_mask = valid_mask.astype(np.float32)

    # 3) Convolve image and mask separately
    #    We need float32 so sums don’t wrap / clip
    img_f = img.astype(np.float32)
    # sum of weighted pixel values
    blurred_sum = cv2.filter2D(img_f,   -1, kernel, borderType=cv2.BORDER_ISOLATED)
    # sum of weights where mask=1
    weight_sum  = cv2.filter2D(valid_mask, -1, kernel, borderType=cv2.BORDER_ISOLATED)

    # 4) Normalize: for each channel, divide by weight_sum
    #    Prevent divide‑by‑zero: wherever weight_sum==0, leave as black (or original)
    #    Expand weight_sum to H×W×1 so it broadcasts over channels
    w = weight_sum[..., None]
    # avoid zeros
    w_safe = np.where(w==0, 1.0, w)

    out = blurred_sum / w_safe
    # in “holes” where w was zero, force black
    out[weight_sum == 0] = 0
    out[black_mask] = 0

    return np.clip(out, 0, 255).astype(np.uint8)
    
def infill_using_normals(color_img, hole_mask, normal_map, max_steps=30):
    """
    Fill black "holes" in `color_img` by marching along surface normals
    obtained from `normal_map`. Stops after `max_steps` or upon hitting
    valid pixels.

    Args:
        color_img:   H×W×3 uint8 array of RGB colors.
        hole_mask:   H×W bool array where True indicates a hole (to fill).
        normal_map:  H×W×3 float array with normals in [0,1] or [-1,1] encoding;
                     XY components give fill directions.
        max_steps:   Maximum ray-march steps.
    Returns:
        H×W×3 uint8 array with holes filled.
    """
    H, W = hole_mask.shape
    # Copy colors for output
    out = color_img.copy()

    # 1) Decode and normalize XY directions
    dirs = normal_map[..., :2].astype(np.float32)
    norms = np.linalg.norm(dirs, axis=-1)
    valid = norms > 1e-6  # pixels with a valid normal
    dirs[valid] /= norms[valid][..., None]

    # 2) Identify hole pixels to fill (exclude any green-coded normals if present)
    green = np.all(normal_map == np.array([0.,1.,0.]), axis=-1)
    to_fill = hole_mask & valid & ~green
    ys, xs = np.nonzero(to_fill)
    if ys.size == 0:
        return out

    # Flatten pixel positions and directions
    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # N×2
    vecs = dirs[ys, xs]                                  # N×2
    N = pts.shape[0]

    # Arrays to track active rays and hit coordinates
    alive = np.ones(N, dtype=bool)
    hits = -np.ones((N, 2), dtype=int)

    # 3) Ray-march all holes in lockstep
    for t in range(1, max_steps+1):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break

        # Sample positions for this step
        sample = pts[idx] + vecs[idx] * t
        xi = np.rint(sample[:,0]).astype(int)
        yi = np.rint(sample[:,1]).astype(int)

        # In-bounds check
        inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx_in = idx[inb]
        if idx_in.size == 0:
            alive[idx] = False
            continue

        xi_in = xi[inb]; yi_in = yi[inb]
        # Check for exiting the hole
        not_hole = ~hole_mask[yi_in, xi_in]
        hit_ids = idx_in[not_hole]
        if hit_ids.size > 0:
            # For each hit, choose best fill source: t+2, t+1, else t
            for rid in hit_ids:
                # try two steps ahead
                for dt in (2, 1, 0):
                    off = t + dt
                    p2 = pts[rid] + vecs[rid] * off
                    x2 = int(round(p2[0])); y2 = int(round(p2[1]))
                    if 0 <= x2 < W and 0 <= y2 < H and not hole_mask[y2, x2]:
                        hits[rid] = (x2, y2)
                        break
            alive[hit_ids] = False

        # Rays that went out of bounds or still in hole remain or die
        alive[idx[~inb]] = False

    # 4) Scatter filled colors into output
    filled = hits[:,0] >= 0
    xs0 = xs[filled]; ys0 = ys[filled]
    xs1 = hits[filled, 0]; ys1 = hits[filled, 1]
    out[ys0, xs0] = color_img[ys1, xs1]

    return out

if __name__ == '__main__':

    # Setup arguments
    parser = argparse.ArgumentParser(
        description=(
            'Convert an RGB-encoded depth video and optional color video '
            'into a stereoscopic 3D side-by-side output for VR/3D TVs.'
        )
    )

    parser.add_argument('--depth_video', type=str, help='Path to input depth-encoded video file', required=True)
    parser.add_argument('--color_video', type=str, help='Path to input color video file', required=False)
    parser.add_argument('--xfov', type=float, help='Horizontal FOV in degrees', required=False)
    parser.add_argument('--yfov', type=float, help='Vertical FOV in degrees, calculated from aspectratio and xfov if not given', required=False)
    parser.add_argument('--xfov_file', type=str, help='JSON file specifying xfov per frame', required=False)
    parser.add_argument('--max_depth', default=100, type=int, help='Maximum depth encoded in video', required=False)
    parser.add_argument('--transformation_file', type=str, help='file with scene transformations from the aligner', required=False)
    parser.add_argument('--transformation_lock_frame', default=0, type=int, help='the frame that the transfomrmation will use as a base', required=False)
    parser.add_argument('--pupillary_distance', default=63, type=int, help='pupillary distance in mm', required=False)
    parser.add_argument('--max_frames', default=-1, type=int, help='Stop after processing this many frames', required=False)
    parser.add_argument('--touchly0', action='store_true', help='Render in touchly0 format (stereo 3D)', required=False)
    parser.add_argument('--vr180', action='store_true', help='Render in VR180 180° side-by-side', required=False)
    parser.add_argument('--render_as_pointcloud', action='store_true', help='Render output as point cloud', required=False)

    parser.add_argument('--convergence_file', type=str, help='json file with convergence data for each frame.', required=False)

    parser.add_argument('--dont_place_points_in_edges', action='store_true', help='Skip adding edge points for infill', required=False)

    parser.add_argument('--do_basic_infill', action='store_true', help='Use basic in-house infill algorithm.', required=False)
    parser.add_argument('--touchly1', action='store_true', help='Render in touchly1 format (mono+depth)', required=False)
    parser.add_argument('--touchly_max_depth', default=5, type=float, help='the max depth that touchly is cliped to', required=False)
    parser.add_argument('--compressed', action='store_true', help='Compress output video (lower quality)', required=False)
    parser.add_argument('--infill_mask', action='store_true', help='Save infill masks alongside output', required=False)
    parser.add_argument('--remove_edges', action='store_true', help='Remove mesh edges not visible in input frames', required=False)
    parser.add_argument('--mask_video', type=str, help='video file to use as mask input to filter out the forground and generate a background version of the mesh that can be used as infill. Requires non moving camera or very good tracking.', required=False)
    parser.add_argument('--save_background', action='store_true', help='Save the compound background as a file. To be ussed as infill.', required=False)
    parser.add_argument('--load_background', help='Load the compound background as a file. To be used as infill.', required=False)


    args = parser.parse_args()

    # ------------------ Input Validation ------------------
    if args.xfov is None and args.yfov is None and args.xfov_file is None:
        raise ValueError("Error: Either --xfov_file, --xfov or --yfov must be provided.")

    MODEL_maxOUTPUT_depth = args.max_depth

    if not os.path.isfile(args.depth_video):
        raise FileNotFoundError(f"Depth video not found: {args.depth_video}")
    depth_video = cv2.VideoCapture(args.depth_video)

    color_video = None
    if args.color_video:
        if not os.path.isfile(args.color_video):
            raise FileNotFoundError(f"Color video not found: {args.color_video}")
        color_video = cv2.VideoCapture(args.color_video)

    mask_video = None
    if args.mask_video:
        if not os.path.isfile(args.mask_video):
            raise FileNotFoundError(f"Mask video not found: {args.mask_video}")
        mask_video = cv2.VideoCapture(args.mask_video)

    convergence_depths = None
    if args.convergence_file:
        if not os.path.isfile(args.convergence_file):
            raise FileNotFoundError(f"Convergence file not found: {args.convergence_file}")
        with open(args.convergence_file) as json_file_handle:
            convergence_depths = json.load(json_file_handle)

    xfovs = None
    if args.xfov_file:
        if not os.path.isfile(args.xfov_file):
            raise FileNotFoundError(f"XFOV file not found: {args.xfov_file}")
        with open(args.xfov_file) as jf:
            xfovs = json.load(jf)
            # assert xfovs is an array
            if not isinstance(xfovs, list) or not all(isinstance(x, (int, float)) for x in xfovs):
                raise ValueError("XFOV file must contain a list of numbers.")


    transformations = None
    if args.transformation_file is not None:
        if not os.path.isfile(args.transformation_file):
            raise Exception("input transformation_file does not exist")
        with open(args.transformation_file) as json_file_handle:
            transformations = json.load(json_file_handle)

        if args.transformation_lock_frame != 0:
            ref_frame = transformations[args.transformation_lock_frame]
            ref_frame_inv_trans = np.linalg.inv(ref_frame)
            for i, transformation in enumerate(transformations):
                transformations[i] = transformation @ ref_frame_inv_trans

    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

    out_width , out_height = frame_width, frame_height

    # check color video / depth video / mask video have the same size and same number of frames
    if color_video is not None:
        color_width  = int(color_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        color_height = int(color_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        color_rate   = round(color_video.get(cv2.CAP_PROP_FPS), 2)

        if frame_width != color_width or frame_height != color_height:
            raise ValueError(f"Depth video and Color video must have the same dimensions (Depth: {frame_width}x{frame_height} vs Color {color_width}x{color_height}).")
        if round(frame_rate, 2) != color_rate:
            raise ValueError(f"Color video and depth video must have the same frame rate (Depth={frame_rate} vs Color={color_rate}).")
    
    if mask_video is not None:
        mask_width  = int(mask_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        mask_height = int(mask_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask_rate   = round(mask_video.get(cv2.CAP_PROP_FPS), 2)

        if frame_width != mask_width or frame_height != mask_height:
            raise ValueError("Mask video and depth video must have the same dimensions (Depth: {frame_width}x{frame_height} vs Mask {mask_width}x{mask_height}).")
        if round(frame_rate, 2) != mask_rate:
            raise ValueError("Mask video and depth video must have the same frame rate (Depth={round(frame_rate, 2)} vs Mask={mask_rate}).")

    if xfovs is not None:
        if len(xfovs) != int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError(f"XFOV file must have the same number of frames as the input video ({int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))} vs xfov={len(xfov)}).")

    if args.touchly0:
        args.vr180 = True

    out_size = None
    # Determine side-by-side output size
    if args.touchly1:
        output_file = args.depth_video + "_Touchly1."
        out_size = (out_width, out_height*2)
    elif args.touchly0:
        output_file = args.depth_video + "_Touchly0."
        out_size = (out_width*3, out_height)
    else:
        output_file = args.depth_video + "_stereo."
        out_size = (out_width*2, out_height)

    # avc1 seams to be required for Quest 2 if linux complains use mp4v but those video files wont work on Quest 2
    # Read this to install avc1 codec from source https://swiftlane.com/blog/generating-mp4s-using-opencv-python-with-the-avc1-codec/
    # generally it is better to render without compression then Add compression at a later stage with a better compresser like FFMPEG.

    if args.compressed:
        output_file += "mp4"
        codec = cv2.VideoWriter_fourcc(*"avc1")
    else:
        output_file += "mkv"
        codec = cv2.VideoWriter_fourcc(*"FFV1")

    out = cv2.VideoWriter(output_file, codec, frame_rate, out_size)

    infill_mask_video = None
    if args.infill_mask:
        infill_mask_video = cv2.VideoWriter(output_file+"_infillmask.mkv", cv2.VideoWriter_fourcc(*"FFV1"), frame_rate, out_size)

    if mask_video is not None:
        # Create background "sphere"
        bg_cloud = o3d.geometry.PointCloud()
        bg_points = np.asarray(bg_cloud.points)
        bg_point_colors = np.asarray(bg_cloud.colors)

        if args.load_background:
            loaded_bg = np.load(args.load_background)
            bg_points = loaded_bg[0]
            bg_point_colors = loaded_bg[1]

    # Precompute eye shifts in world units (meters)
    left_shift = -(args.pupillary_distance/1000)/2
    right_shift = +(args.pupillary_distance/1000)/2

    draw_mesh = None
    frame_n = 0
    last_mesh = None

    start_time = time.time()
    prev_frame_end  = start_time
    # Determine how many frames we’ll process
    total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT)) if args.max_frames < 0 else args.max_frames


    while raw_video.isOpened():
        frame_n += 1

        ############# TIMING PART #############
        # Mark the start of *this* frame
        frame_start = time.time()
        if frame_n == 1:
            print(f"[     %] Frame #{frame_n:4d}/{total_frames}", end='\r')
        else:
            pct = (frame_n / total_frames) * 100 if total_frames > 0 else 0
            avg_per_frame = (frame_start - start_time) / frame_n if frame_n > 0 else 0
            rem_seconds   = avg_per_frame * (total_frames - frame_n)
            print(f"[{pct:5.1f}%] Frame #{frame_n:4d}/{total_frames}, "
                f"Remaining: {(int(rem_seconds) // 60)}min{(int(rem_seconds) % 60):02d}s | "
                f"Last frame rendered in {(frame_start - prev_frame_end):6.3f}s", end='\r')
            prev_frame_end = frame_start
        ############# TIMING PART #############

        ret, raw_frame = raw_video.read()
        if not ret:
            break

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

        color_frame = None
        if color_video is not None:
            ret, color_frame = color_video.read()

            if not ret:
                print("Warning: cannot read color video frame. Skipping it until the end.")
                color_video.release()
                color_video = None
                continue

            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

            assert color_frame.shape == rgb.shape, "color image and depth image need to have same width and height" #potential BUG here with mono depth videos
        else:
            color_frame = rgb

        # Decode video depth
        depth = depth_frames_helper.decode_rgb_depth_frame(rgb, MODEL_maxOUTPUT_depth, True)

        # Compute camera intrinsics per frame
        if xfovs is not None:
            xf = xfovs[frame_n-1]
            yf = None
        else:
            xf = args.xfov
            yf = args.yfov

        cam_matrix = depth_map_tools.compute_camera_matrix(xf, yf, frame_width, frame_height)
        render_cam_matrix = cam_matrix
        if args.vr180:
            out_width , out_height = 1920, 1920
            fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)
            max_fov = max(fovx, fovy)
            if max_fov >= 180:
                raise ValueError("fov cant be 180 or over, the tool is not built to handle fisheye distorted input video")
            render_fov = max(75, max_fov)
            render_cam_matrix = depth_map_tools.compute_camera_matrix(render_fov, render_fov, out_width, out_height)



        edge_pcd = None

        if transformations is None and args.touchly1: #Fast path we can skip the full render pass
            depth8bit = np.rint(np.minimum(depth, args.touchly_max_depth)*(255/args.touchly_max_depth)).astype(np.uint8)
            touchly_depth = np.repeat(depth8bit[..., np.newaxis], 3, axis=-1)
            touchly_depth = 255 - touchly_depth #Touchly uses reverse depth
            out_image = cv2.vconcat([color_frame, touchly_depth])
        else:

            bg_color = np.array([0.0, 0.0, 0.0])
            if infill_mask_video is not None:
                bg_color = np.array([0.0, 1.0, 0.0])
                bg_color_infill_detect = np.array([0, 255, 0], dtype=np.uint8)
                black = np.array([0, 0, 0], dtype=np.uint8)



            if transformations is not None:
                transform_to_zero = np.array(transformations[frame_n-1])
            else:
                transform_to_zero = np.eye(4)

            remove_edges = False
            if args.infill_mask or args.remove_edges or args.do_basic_infill:
                remove_edges = True

            of_by_one = True
            if args.render_as_pointcloud:
                if args.infill_mask:
                    print("--infill_mask and --render_as_pointcloud dont work great together")
                    #TODO: add a feature so you can get a pointcloud render as a normal render
                of_by_one = False


            mesh, unused_indices, removed_normals = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, last_mesh, remove_edges = remove_edges, of_by_one = of_by_one, return_normals_of_removed = True)
            last_mesh = mesh


            # If there are not points in the infill areas the infill models get confused.
            # So we add points in the infill area
            if not args.dont_place_points_in_edges and remove_edges:
                vertextes_in_edge = np.zeros(len(mesh.vertices), dtype=bool)
                
                vertextes_in_edge[unused_indices] = True
                
                edge_points = np.asarray(mesh.vertices)[vertextes_in_edge]
                edge_colors = np.asarray(mesh.vertex_colors)[vertextes_in_edge]
                world_space_edge_normals = removed_normals + edge_points

                #Undo off by one fix
                edge_points[:, 0] *= (frame_width-1)/frame_width
                edge_points[:, 1] *= (frame_height-1)/frame_height
                

                #Only draw edge points if there is more than 1 of them
                if len(edge_points) > 1:
                    edge_pcd = depth_map_tools.pts_2_pcd(edge_points)
                    edge_normal_pcd = depth_map_tools.pts_2_pcd(world_space_edge_normals)


            if args.render_as_pointcloud:
                draw_mesh = depth_map_tools.convert_mesh_to_pcd(mesh, unused_indices, draw_mesh)
                #TODO:move points that is vertices back to their real position
            else:
                draw_mesh = mesh

            if transformations is not None:
                draw_mesh.transform(transform_to_zero)
                if edge_pcd is not None:
                    edge_pcd.transform(transform_to_zero)
                    edge_normal_pcd.transform(transform_to_zero)

            if mask_video is not None:

                ret, mask_frame = mask_video.read()
                if not ret:
                    print("Warning: cannot read mask video frame. Skipping it until the end.")
                    mask_video.release()
                    mask_video = None
                    continue
                mask_img = np.array(cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY))

                #find all black pixels
                mask_img1d = mask_img.reshape(-1)
                bg_mask = np.where(mask_img1d < 128)[0]
                
                used_indices_mask = np.ones(len(mesh.vertices), dtype=bool)
                used_indices_mask[unused_indices] = False
                
                used_indices = np.where(used_indices_mask)[0]
                
                # intersect the mask pixels with the pizels that are not edges
                points_2_keep = np.intersect1d(used_indices, bg_mask)

                new_points = np.asarray(mesh.vertices)[points_2_keep]
                new_colors = np.asarray(mesh.vertex_colors)[points_2_keep]


                bg_points  = np.concatenate((bg_points, new_points), axis=0)
                bg_point_colors  = np.concatenate((bg_point_colors, new_colors), axis=0)

                bg_cloud = depth_map_tools.pts_2_pcd(bg_points, bg_point_colors)

                #clear up the point clouds every so often
                if frame_n % 10 == 0:
                    print("clearing up pointcloud")

                    # perspective_aware_down_sample makes sense when you are looking in the same direction, techically a normal down_sample function would be better. But it is to slow.
                    bg_cloud = depth_map_tools.perspective_aware_down_sample(bg_cloud, 0.003)

                    bg_points  = np.asarray(bg_cloud.points)
                    bg_point_colors = np.asarray(bg_cloud.colors)
                    bg_cloud = copy.deepcopy(bg_cloud)



            if args.save_background:
                # We need to check this here so that the continue dont skip the check
                if args.max_frames < frame_n and args.max_frames != -1:
                    break
                continue



            #Only render the background
            if args.mask_video is not None:
                mesh = bg_cloud

            if args.touchly1:
                to_draw = [draw_mesh]




                color_transformed, touchly_depth = depth_map_tools.render(to_draw, render_cam_matrix, -2, bg_color = bg_color)
                color_transformed = (color_transformed*255).astype(np.uint8)


                touchly_depth8bit = np.rint(np.minimum(touchly_depth, args.touchly_max_depth)*(255/args.touchly_max_depth)).astype(np.uint8)
                touchly_depth8bit[touchly_depth8bit == 0] = 255 # Any pixel at zero depth needs to move back as it is part of the render viewport background and not the mesh
                touchly_depth8bit = 255 - touchly_depth8bit #Touchly uses reverse depth
                touchly_depth = np.repeat(touchly_depth8bit[..., np.newaxis], 3, axis=-1)

                out_image = cv2.vconcat([color_transformed, touchly_depth])

                if infill_mask_video is not None:
                    bg_mask = np.all(color_transformed == bg_color_infill_detect, axis=-1)
                    img_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    img_mask[bg_mask] = 255

                    zero = np.zeros((frame_height, frame_width), dtype=np.uint8)

                    out_mask_image = cv2.vconcat([img_mask, zero])
                    infill_mask_video.write(cv2.cvtColor(out_mask_image, cv2.COLOR_RGB2BGR))

            else:

                #move mesh for left eye render
                convergence_distance = None
                if convergence_depths is not None:
                    convergence_distance = float(convergence_depths[frame_n-1]) #np.mean(depth)#Testing set convergence to scene frame average depth
                    if convergence_distance == 0:
                        print("Convergence distance is zero, skipping convergence")
                        convergence_distance = None
                    else:
                        convergence_angle_rad = convergence_angle(convergence_distance, args.pupillary_distance/1000)
                        convergence_rotation_plus = mesh.get_rotation_matrix_from_xyz((0, convergence_angle_rad, 0))
                        convergence_rotation_minus = mesh.get_rotation_matrix_from_xyz((0, -convergence_angle_rad, 0))

                if convergence_distance is not None:
                    draw_mesh.rotate(convergence_rotation_minus, center=(0, 0, 0))
                draw_mesh.translate([-left_shift, 0.0, 0.0])
                to_draw = [draw_mesh]
                if edge_pcd is not None:
                    if convergence_distance is not None:
                        edge_pcd.rotate(convergence_rotation_minus, center=(0, 0, 0))
                        edge_normal_pcd.rotate(convergence_rotation_minus, center=(0, 0, 0))
                    edge_pcd.translate([-left_shift, 0.0, 0.0])
                    edge_normal_pcd.translate([-left_shift, 0.0, 0.0])
                    unprojected_normals = np.asarray(edge_normal_pcd.points) - np.asarray(edge_pcd.points)
                    points_3d = np.asarray(edge_pcd.points)
                    points_2d = depth_map_tools.project_3d_points_to_2d(points_3d, render_cam_matrix)
                
                
                left_image, left_depth = depth_map_tools.render(to_draw, render_cam_matrix, depth = -2, bg_color = bg_color)
                
                
                bg_mask = np.all(left_image == bg_color, axis=-1)
                

                    

                if edge_pcd is not None:
                    points_int = np.round(points_2d).astype(int)
                    valid_mask = (
                        (points_int[:, 0] >= 0) & (points_int[:, 0] < frame_width) &
                        (points_int[:, 1] >= 0) & (points_int[:, 1] < frame_height)
                    )
                    points_3d = points_3d[valid_mask]
                    depth_order = np.argsort(points_3d[:, 2])[::-1]
                    valid_points = points_int[valid_mask][depth_order]
                    valid_colors = edge_colors[valid_mask][depth_order]
                    valid_unprojected_normals = unprojected_normals[valid_mask][depth_order]
                    
                    ####valid_normals = [valid_mask] #What are the normals after rotation?
                    # valid_colors dows not change after rotation but the normals in screen space should change.
                    # A Truth is that the normals will change but not that much, we might get away with just leaving them as they are.
                    # NOT is there is insane convergence or large tranformations from a tranformation file.
                    # Okay... One way to deal with this is to project the normals in to world space by doing:
                    # projected_normals = edge_normals + np.asarray(edge_pcd.points)
                    # then applying rotations, then un projecting:
                    # unprojected_normals = projected_normals - np.asarray(edge_pcd.points)
                    
                    # Okay this works now, But there is still an issue:
                    # When there is a big triangle there will be green in the middle since there is no vertexes there and i only do
                    # color pixels that have vertexes.
                    # One way to solve this is to flood fill these areas, with normals from surounding areas.
                    # another is to acctually render the infill as triangles. (make a second mesh for the infill areas that is colored
                    # by its unprojected_normals). This will require a second render. And will not work great for back facing triangles.
                    # As they will be covered by front facing triangles.
                    # I just realized that you probably accutally want to do infill from the raw image/model not the rotated one.
                    # As the area you want to infill from might be coverd in the rotated translated render.
                    
                    mask = np.all(left_image[valid_points[:, 1], valid_points[:, 0]] == bg_color, axis=-1)
                    
                    normalized = valid_unprojected_normals[mask]
                    lengths = np.linalg.norm(normalized, axis=1, keepdims=True)
                    normalized = normalized / lengths
                
                
                
                    
                    
                    
                left_img_mask = np.zeros((frame_height, frame_width, 3), dtype=np.float64)

                if edge_pcd is not None:
                    
                    
                    left_img_mask[bg_mask] = bg_color # We simply hope that there is no normal that is perfectly green when we use green as bg
                    left_image[bg_mask] = np.array([.0,.0,.0])
                    left_img_mask[:, 0][np.all(left_img_mask[:, 0, :]  == bg_color, axis=1)] = np.array([1., 0.5, 0.5])
                    
                    
                    left_img_mask[valid_points[mask, 1], valid_points[mask, 0]] = (normalized+1)/2
                    green_left = np.all(left_img_mask == bg_color, axis=-1)
                    green_and_black = green_left | np.all(left_img_mask == np.array([.0,.0,.0]), axis=-1)
                    infill_area_mask = (green_and_black*255).astype('uint8')
                    left_img_mask_infilled = cv2.inpaint((left_img_mask*255).astype('uint8'), infill_area_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    left_img_mask[green_left] = left_img_mask_infilled[green_left].astype('float32')/255.0
                    left_img_mask = masked_blur((left_img_mask*255).astype('uint8')).astype('float32')/255.0
                    
                    if args.do_basic_infill:
                        left_img_mask_minus = (left_img_mask*2)-1
                        left_image = infill_using_normals(left_image, bg_mask, left_img_mask_minus)
                    else:
                        left_image[valid_points[mask, 1], valid_points[mask, 0]] = valid_colors[mask]
                    
                if infill_mask_video is not None:
                    left_img_mask = (left_img_mask*255).astype(np.uint8)

                left_image = (left_image*255).astype(np.uint8)



                touchly_left_depth = None
                #Touchly1 requires a left eye depthmap XXX use dual rendering here to speed things upp
                if args.touchly0:
                    left_depth8bit = np.rint(np.minimum(left_depth, args.touchly_max_depth)*(255/args.touchly_max_depth)).astype(np.uint8)
                    left_depth8bit[left_depth8bit == 0] = 255 # Any pixel at zero depth needs to move back is is non rendered depth buffer(ie things on the side of the mesh)
                    left_depth8bit = 255 - left_depth8bit #Touchly uses reverse depth
                    touchly_left_depth = np.repeat(left_depth8bit[..., np.newaxis], 3, axis=-1)

                #Move mesh back to center and move mesh for right eye render
                draw_mesh.translate([left_shift, 0.0, 0.0])
                if convergence_distance is not None:
                    draw_mesh.rotate(convergence_rotation_plus, center=(0, 0, 0))
                    draw_mesh.rotate(convergence_rotation_plus, center=(0, 0, 0))
                draw_mesh.translate([-right_shift, 0.0, 0.0])
                to_draw = [draw_mesh]
                if edge_pcd is not None:
                    edge_pcd.translate([left_shift, 0.0, 0.0])
                    edge_normal_pcd.translate([left_shift, 0.0, 0.0])
                    if convergence_distance is not None:
                        edge_pcd.rotate(convergence_rotation_plus, center=(0, 0, 0))
                        edge_pcd.rotate(convergence_rotation_plus, center=(0, 0, 0))
                        edge_normal_pcd.rotate(convergence_rotation_plus, center=(0, 0, 0))
                        edge_normal_pcd.rotate(convergence_rotation_plus, center=(0, 0, 0))
                    edge_pcd.translate([-right_shift, 0.0, 0.0])
                    edge_normal_pcd.translate([-right_shift, 0.0, 0.0])
                    unprojected_normals = np.asarray(edge_normal_pcd.points) - np.asarray(edge_pcd.points)
                    points_3d = np.asarray(edge_pcd.points)
                    points_2d = depth_map_tools.project_3d_points_to_2d(points_3d, render_cam_matrix)
                
                right_image, right_depth = depth_map_tools.render(to_draw, render_cam_matrix, depth = -2, bg_color = bg_color)

                bg_mask = np.all(right_image == bg_color, axis=-1)


                if edge_pcd is not None:
                    points_int = np.round(points_2d).astype(int)
                    valid_mask = (
                        (points_int[:, 0] >= 0) & (points_int[:, 0] < frame_width) &
                        (points_int[:, 1] >= 0) & (points_int[:, 1] < frame_height)
                    )
                    points_3d = points_3d[valid_mask]
                    depth_order = np.argsort(points_3d[:, 2])[::-1]
                    valid_points = points_int[valid_mask][depth_order]
                    valid_colors = edge_colors[valid_mask][depth_order]
                    valid_unprojected_normals = unprojected_normals[valid_mask][depth_order]
                    mask = np.all(right_image[valid_points[:, 1], valid_points[:, 0]] == bg_color, axis=-1)
                    
                    normalized = valid_unprojected_normals[mask]
                    lengths = np.linalg.norm(normalized, axis=1, keepdims=True)
                    normalized = normalized / lengths
                
                right_img_mask = np.zeros((frame_height, frame_width, 3), dtype=np.float64)
                
                if edge_pcd is not None:
                    
                    right_img_mask[bg_mask] = bg_color # We simply hope that there is no normal that is perfectly green when we use green as bg
                    right_image[bg_mask] = np.array([.0,.0,.0])
                    right_img_mask[:, -1][np.all(right_img_mask[:, -1, :]  == bg_color, axis=1)] = np.array([0., 0.5, 0.5])

                
                    
                    #if the left most pixels are all green set their normals so they get infilled
                    
                    
                    #What hapens when two vertexes are antop of eachother?
                    right_img_mask[valid_points[mask, 1], valid_points[mask, 0]] = (normalized+1)/2
                    green_left = np.all(right_img_mask == bg_color, axis=-1)
                    green_and_black = green_left | np.all(right_img_mask == np.array([.0,.0,.0]), axis=-1)
                    infill_area_mask = (green_and_black*255).astype('uint8')
                    right_img_mask_infilled = cv2.inpaint((right_img_mask*255).astype('uint8'), infill_area_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    right_img_mask[green_left] = right_img_mask_infilled[green_left].astype('float32')/255.0
                    right_img_mask = masked_blur((right_img_mask*255).astype('uint8')).astype('float32')/255.0
                    
                    if args.do_basic_infill:
                        right_img_mask_minus = (right_img_mask*2)-1
                        right_image = infill_using_normals(right_image, bg_mask, right_img_mask_minus)
                    else:
                        right_image[valid_points[mask, 1], valid_points[mask, 0]] = valid_colors[mask]
                
                if infill_mask_video is not None:
                    right_img_mask = (right_img_mask*255).astype(np.uint8)

                right_image = (right_image*255).astype(np.uint8)


                imgs = [left_image, right_image]
                if touchly_left_depth is not None:
                    imgs.append(touchly_left_depth)

                if args.vr180:
                    for i, img in enumerate(imgs):
                        imgs[i] = convert_to_equirectangular(img, input_fov = render_fov)

                out_image = cv2.hconcat(imgs)


                if infill_mask_video is not None:
                    imgs = [left_img_mask, right_img_mask]
                    if touchly_left_depth is not None:
                        zero = np.zeros((frame_height, frame_width), dtype=np.uint8)
                        imgs.append(zero)

                    out_mask_image = cv2.hconcat(imgs)
                    infill_mask_video.write(cv2.cvtColor(out_mask_image, cv2.COLOR_RGB2BGR))


        out.write(cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))

        if args.max_frames < frame_n and args.max_frames != -1:
            break

    if args.save_background:
        np.save(args.depth_video + '_background.npy', np.array([bg_points, bg_point_colors]))

    # Release resources
    raw_video.release()
    out.release()
    if mask_video is not None:
        mask_video.release()
    
    if color_video is not None:
        color_video.release()

    if infill_mask_video is not None:
        infill_mask_video.release()

    print(f"Processing complete. Output saved to: {output_file}")
