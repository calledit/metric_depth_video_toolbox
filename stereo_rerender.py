import argparse
import cv2
import numpy as np
import os
import copy
import sys
import time
import json

import open3d as o3d
import depth_map_tools
from contextlib import contextmanager
import time

@contextmanager
def timer(name = 'not named'):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.6f} seconds")

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


def infill_from_deep_side(image, mask, sample_side='right'):
    """
    For each row in the image, infill segments marked in the mask with the color sampled
    from the deeper side (blue side). The mask is assumed to have:
      - Blue pixels: [0, 0, 255] marking the deeper edge.
      - Red  ([255, 0, 0]) and Green ([0, 255, 0]) marking the infill region.
      - Black: [0, 0, 0] elsewhere.
    
    For each contiguous segment (where mask is non-black) in a row, the function:
      1. Searches for blue marker pixels in the segment.
      2. If blue markers exist, uses their position to decide the infill candidate.
         - If a blue marker is at the left or right boundary, the candidate color is
           sampled from the pixel just outside the segment.
         - Otherwise, the first blue marker is used (and its right neighbor is chosen).
      3. If no blue marker exists (i.e. only a single green pixel), the candidate color
         is sampled from the right or left side of the segment, depending on the `sample_side`
         argument.
      4. Fills the segment with the candidate color.
    
    Parameters:
      image (np.ndarray): Input image (H, W, 3) of type uint8.
      mask (np.ndarray): Marking mask (H, W, 3) of type uint8.
      sample_side (str): "right" (default) or "left". Used when no blue pixel is found.
      
    Returns:
      np.ndarray: The resulting image after infilling.
    """
    result = image.copy()
    H, W, _ = image.shape

    # Define colors.
    black = np.array([0, 0, 0], dtype=np.uint8)
    blue  = np.array([0, 0, 255], dtype=np.uint8)

    # Create a binary infill region: True where the mask is not black.
    infill_region = np.any(mask != 0, axis=-1)  # shape (H, W)

    for i in range(H):
        row_infill = infill_region[i]
        if not np.any(row_infill):
            continue

        # Compute contiguous segments in the row.
        row_int = row_infill.astype(np.int32)
        d = np.diff(row_int)  # length W-1

        # A segment starts where diff == 1 (and at index 0 if the row starts True).
        seg_starts = list(np.where(d == 1)[0] + 1)
        if row_infill[0]:
            seg_starts = [0] + seg_starts

        # A segment ends where diff == -1 (and at W-1 if the row ends True).
        seg_ends = list(np.where(d == -1)[0])
        if row_infill[-1]:
            seg_ends = seg_ends + [W - 1]

        # Process each segment.
        for start, end in zip(seg_starts, seg_ends):
            # Extract this segment from the mask.
            segment_mask = mask[i, start:end+1]  # shape (segment_length, 3)
            # Identify blue pixels in the segment.
            is_blue = np.all(segment_mask == blue, axis=-1)
            blue_indices = np.where(is_blue)[0]
            
            if blue_indices.size == 0:
                # No blue pixel exists, so sample from the specified side.
                if sample_side == 'right':
                    candidate_idx = end + 1 if end < W - 1 else end
                else:  # sample_side == 'left'
                    candidate_idx = start - 1 if start > 0 else start
            else:
                # Use blue markers if present.
                # Check if the blue marker is at one of the boundaries.
                if 0 in blue_indices:
                    # Blue is at the left edge of the segment.
                    candidate_idx = start - 1 if start > 0 else start
                elif (end - start) in blue_indices:
                    # Blue is at the right edge.
                    candidate_idx = end + 1 if end < W - 1 else end
                else:
                    # Otherwise, choose the first blue marker in the segment.
                    b_rel = blue_indices[0]  # relative index inside segment
                    candidate_idx = start + b_rel + 1
                    if candidate_idx >= W:
                        candidate_idx = start + b_rel - 1 if (start + b_rel - 1) >= 0 else start + b_rel

            # Sample the candidate color from the original image.
            candidate_color = image[i, candidate_idx]
            # Fill the entire segment with the candidate color.
            result[i, start:end+1] = candidate_color

    return result


def mark_depth_transitions(boolean_mask, depth_map):
    """
    This function tries to determine what side of an edge is closer or further away from the camera.
    Which is usefull info to have when doing paralax infill.
    This is something that could be calculated exactly but, that would probably require more processing power.
    Or use of a diffrent 3D library than open3D. So this function asumes that all pixels only move in x direction due to paralax.
    Which is true for the exact height center of the image as that is how we move the camera but it is a rough estimation of pixels
    Further up or down....
    
    For each row in the boolean mask, finds segments where the mask is True.
    For each segment, compares depth values at points slightly outside the segment edges:
      - Closer edge is marked red (255, 0, 0)
      - Further edge is marked blue (0, 0, 255)
      - The interior of infill areas is marked green (0, 255, 0)
      - Non infill pixels are black (0, 0, 0)
    
    Parameters:
      boolean_mask (np.ndarray): 2D boolean array, shape (H, W)
      depth_map (np.ndarray): 2D float array, shape (H, W) with depth values.
      
    Returns:
      np.ndarray: Color image (H, W, 3) with the markings.
    """
    H, W = boolean_mask.shape

    # Define colors (RGB)
    red   = np.array([255, 0, 0], dtype=np.uint8)
    blue  = np.array([0, 0, 255], dtype=np.uint8)
    green = np.array([0, 255, 0], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)
    
    # Start with a black output image.
    result = np.zeros((H, W, 3), dtype=np.uint8)
    # Mark all True pixels as green.
    result[boolean_mask] = green

    # Convert mask to integer (0/1) and compute diff along each row.
    mask_int = boolean_mask.astype(np.int8)
    diff = np.diff(mask_int, axis=1)  # shape (H, W-1)

    # Find indices of transitions:
    # diff == 1: transition from False->True, so segment start (actual index = col + 1)
    start_rows, start_cols = np.where(diff == 1)
    start_cols = start_cols + 1  # adjust to actual start

    # diff == -1: transition from True->False, so segment end (actual index = col)
    end_rows, end_cols = np.where(diff == -1)

    # Rows starting with True: add (row, 0) as start.
    first_true = np.where(boolean_mask[:, 0])[0]
    start_rows = np.concatenate([first_true, start_rows])
    start_cols = np.concatenate([np.zeros(first_true.shape, dtype=int), start_cols])

    # Rows ending with True: add (row, W-1) as end.
    last_true = np.where(boolean_mask[:, -1])[0]
    end_rows = np.concatenate([end_rows, last_true])
    end_cols = np.concatenate([end_cols, np.full(last_true.shape, W - 1, dtype=int)])

    # To pair transitions per row, group by row.
    # Process only rows that have at least one transition.
    unique_rows = np.unique(np.concatenate([start_rows, end_rows]))
    for r in unique_rows:
        # Get all start and end indices for this row.
        s_mask = start_rows == r
        e_mask = end_rows == r
        s_cols = start_cols[s_mask]
        e_cols = end_cols[e_mask]
        # Only process if we have a matching pair per segment.
        n_segments = min(len(s_cols), len(e_cols))
        if n_segments == 0:
            continue
        s_cols = s_cols[:n_segments]
        e_cols = e_cols[:n_segments]
        
        # Compute check positions: 2 pixels to the left of s (clamped to 0) 
        # and 2 pixels to the right of e (clamped to W-1)
        s_checks = np.maximum(s_cols - 2, 0)
        e_checks = np.minimum(e_cols + 2, W - 1)
        
        # Get depth values at the check positions.
        depth_s = depth_map[r, s_checks]
        depth_e = depth_map[r, e_checks]
        
        # Compare depths: lower depth is considered closer.
        # For each segment, mark the closer edge red and the further edge blue.
        closer_at_start = depth_s < depth_e
        # For segments where start is closer:
        result[r, s_cols[closer_at_start]] = red
        result[r, e_cols[closer_at_start]] = blue
        # For segments where end is closer:
        result[r, s_cols[~closer_at_start]] = blue
        result[r, e_cols[~closer_at_start]] = red

    return result


if __name__ == '__main__':
    
    # Setup arguments
    parser = argparse.ArgumentParser(description='Take a rgb encoded depth video and a color video, and render them it as a steroscopic 3D video.'+
        'that can be used on 3d tvs and vr headsets.')
    
    parser.add_argument('--depth_video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=False)
    parser.add_argument('--xfov', type=float, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=float, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the input video uses', required=False)
    parser.add_argument('--transformation_file', type=str, help='file with scene transformations from the aligner', required=False)
    parser.add_argument('--transformation_lock_frame', default=0, type=int, help='the frame that the transfomrmation will use as a base', required=False)
    parser.add_argument('--pupillary_distance', default=63, type=int, help='pupillary distance in mm', required=False)
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--touchly0', action='store_true', help='Render as touchly0 format. ie. stereo video with 3d ', required=False)
    parser.add_argument('--vr180', action='store_true', help='Render as vr180 format. ie. stereo video at 180 deg ', required=False)
    parser.add_argument('--render_as_pointcloud', action='store_true', help='Render as point cloud instead of as mesh', required=False)
    
    parser.add_argument('--dont_place_points_in_edges', action='store_true', help='Dont put point cloud points in the removed edges', required=False)
    
    parser.add_argument('--touchly1', action='store_true', help='Render as touchly1 format. ie. mono video with 3d', required=False)
    parser.add_argument('--touchly_max_depth', default=5, type=float, help='the max depth that touchly is cliped to', required=False)
    parser.add_argument('--compressed', action='store_true', help='Render the video in a compressed format. Reduces file size but also quality.', required=False)
    parser.add_argument('--infill_mask', action='store_true', help='Save infill mask video.', required=False)
    parser.add_argument('--remove_edges', action='store_true', help='Tries to remove edges that was not visible in image', required=False)
    parser.add_argument('--mask_video', type=str, help='video file to use as mask input to filter out the forground and generate a background version of the mesh that can be used as infill. Requires non moving camera or very good tracking.', required=False)
    parser.add_argument('--save_background', action='store_true', help='Save the compound background as a file. To be ussed as infill.', required=False)
    parser.add_argument('--load_background', help='Load the compound background as a file. To be used as infill.', required=False)
    
    
    args = parser.parse_args()
    
    if args.xfov is None and args.yfov is None:
        print("Either --xfov or --yfov is required.")
        exit(0)
    
    
   
    MODEL_maxOUTPUT_depth = args.max_depth
    
    # Verify input file exists
    if not os.path.isfile(args.depth_video):
        raise Exception("input video does not exist")
    
    color_video = None
    if args.color_video is not None:
        if not os.path.isfile(args.color_video):
            raise Exception("input color_video does not exist")
        color_video = cv2.VideoCapture(args.color_video)
        
    mask_video = None
    if args.mask_video is not None:
        if not os.path.isfile(args.mask_video):
            raise Exception("input mask_video does not exist")
        mask_video = cv2.VideoCapture(args.mask_video)
    
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
        
    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)
    fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)
    render_cam_matrix = cam_matrix
    out_width , out_height = frame_width, frame_height
    
    if args.touchly0:
        args.vr180 = True
        
    if args.vr180:
        out_width , out_height = 1920, 1920
        max_fov = max(fovx, fovy)
        if max_fov >= 180:
            raise ValueError("fov cant be 180 or over, the tool is not built to handle fisheye distorted input video")
        render_fov = max(75, max_fov)
        render_cam_matrix = depth_map_tools.compute_camera_matrix(render_fov, render_fov, out_width, out_height)
        
    out_size = None
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
    
    
    left_shift = -(args.pupillary_distance/1000)/2
    right_shift = +(args.pupillary_distance/1000)/2
    
    draw_mesh = None
    frame_n = 0
    last_mesh = None
    while raw_video.isOpened():
        
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        frame_n += 1
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        
        color_frame = None
        if color_video is not None:
            ret, color_frame = color_video.read()
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            
            assert color_frame.shape == rgb.shape, "color image and depth image need to have same width and height" #potential BUG here with mono depth videos
        else:
            color_frame = rgb

        # Decode video depth
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((rgb[..., 0].astype(np.uint32) + rgb[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        
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
            if args.infill_mask or args.remove_edges:
                remove_edges = True
            
            of_by_one = True
            if args.render_as_pointcloud:
                if args.infill_mask:
                    print("--infill_mask and --render_as_pointcloud dont work great together")
                    #TODO: add a feature so you can get a pointcloud render as a normal render
                of_by_one = False
            
                
            mesh, used_indices = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, last_mesh, remove_edges = remove_edges, of_by_one = of_by_one)
            last_mesh = mesh
            
            
            # If there are not points in the infill areas the infill models get confused.
            # So we add points in the infill area
            if not args.dont_place_points_in_edges and remove_edges:
                vertextes_in_edge = np.ones(len(mesh.vertices), dtype=bool)
                vertextes_in_edge[used_indices] = False
                edge_points = np.asarray(mesh.vertices)[vertextes_in_edge]
                edge_colors = np.asarray(mesh.vertex_colors)[vertextes_in_edge]
                
                #Undo off by one fix
                edge_points[:, 0] *= (frame_width-1)/frame_width
                edge_points[:, 1] *= (frame_height-1)/frame_height
                
                #Only draw edge points if there is more than 1 of them
                if len(edge_points) > 1:
                    edge_pcd = depth_map_tools.pts_2_pcd(edge_points, edge_colors)
                
            
            if args.render_as_pointcloud:
                draw_mesh = depth_map_tools.convert_mesh_to_pcd(mesh, used_indices, draw_mesh)
                #TODO:move points that is vertices back to their real position
            else:
                draw_mesh = mesh
            
            if transformations is not None:
                draw_mesh.transform(transform_to_zero)
                if edge_pcd is not None:
                    edge_pcd.transform(transform_to_zero)
            
            if mask_video is not None:
                
                ret, mask_frame = mask_video.read()
                mask_img = np.array(cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY))
                
                #find all black pixels 
                mask_img1d = mask_img.reshape(-1)
                bg_mask = np.where(mask_img1d < 128)[0]
                
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
                if args.max_frames < frame_n and args.max_frames != -1: #We ceed to check this here so that the continue dont skip the check
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
                draw_mesh.translate([-left_shift, 0.0, 0.0])
                to_draw = [draw_mesh]
                if edge_pcd is not None:
                    edge_pcd.translate([-left_shift, 0.0, 0.0])
                    points_2d = depth_map_tools.project_3d_points_to_2d(np.asarray(edge_pcd.points), render_cam_matrix)
                
                left_image, left_depth = depth_map_tools.render(to_draw, render_cam_matrix, depth = -2, bg_color = bg_color)
                
                if infill_mask_video is not None:
                    bg_mask = np.all(left_image == bg_color, axis=-1)
                    left_img_mask = mark_depth_transitions(bg_mask, left_depth)
                
                
                if edge_pcd is not None:
                    points_int = np.round(points_2d).astype(int)
                    valid_mask = (
                        (points_int[:, 0] >= 0) & (points_int[:, 0] < frame_width) &
                        (points_int[:, 1] >= 0) & (points_int[:, 1] < frame_height)
                    )
                    valid_points = points_int[valid_mask]
                    valid_colors = edge_colors[valid_mask]
                    mask = np.all(left_image[valid_points[:, 1], valid_points[:, 0]] == bg_color, axis=-1)
                    
                
                
                
                if infill_mask_video is not None:
                    left_image[bg_mask] = np.array([.0,.0,.0])
                    
                if edge_pcd is not None:
                    left_image[valid_points[mask, 1], valid_points[mask, 0]] = valid_colors[mask]
                
                left_image = (left_image*255).astype(np.uint8)
                
                #else:
                #    left_image = infill_from_deep_side(left_image, left_img_mask, 'left')
                    
            
                touchly_left_depth = None
                #Touchly1 requires a left eye depthmap XXX use dual rendering here to speed things upp
                if args.touchly0:
                    left_depth8bit = np.rint(np.minimum(left_depth, args.touchly_max_depth)*(255/args.touchly_max_depth)).astype(np.uint8)
                    left_depth8bit[left_depth8bit == 0] = 255 # Any pixel at zero depth needs to move back is is non rendered depth buffer(ie things on the side of the mesh)
                    left_depth8bit = 255 - left_depth8bit #Touchly uses reverse depth
                    touchly_left_depth = np.repeat(left_depth8bit[..., np.newaxis], 3, axis=-1)
            
                #Move mesh back to center and move mesh for right eye render
                draw_mesh.translate([left_shift-right_shift, 0.0, 0.0])
                to_draw = [draw_mesh]
                if edge_pcd is not None:
                    edge_pcd.translate([left_shift-right_shift, 0.0, 0.0])
                    points_2d = depth_map_tools.project_3d_points_to_2d(np.asarray(edge_pcd.points), render_cam_matrix)
                
                right_image, right_depth = depth_map_tools.render(to_draw, render_cam_matrix, depth = -2, bg_color = bg_color)
                
                if infill_mask_video is not None:
                    bg_mask = np.all(right_image == bg_color, axis=-1)
                    right_img_mask = mark_depth_transitions(bg_mask, right_depth)
                
                    
                
                if edge_pcd is not None:
                    points_int = np.round(points_2d).astype(int)
                    valid_mask = (
                        (points_int[:, 0] >= 0) & (points_int[:, 0] < frame_width) &
                        (points_int[:, 1] >= 0) & (points_int[:, 1] < frame_height)
                    )
                    valid_points = points_int[valid_mask]
                    valid_colors = edge_colors[valid_mask]
                    mask = np.all(right_image[valid_points[:, 1], valid_points[:, 0]] == bg_color, axis=-1)
                
                
                if infill_mask_video is not None:
                    right_image[bg_mask] = np.array([.0,.0,.0])
                
                if edge_pcd is not None:
                    right_image[valid_points[mask, 1], valid_points[mask, 0]] = valid_colors[mask]
                
                right_image = (right_image*255).astype(np.uint8)
                
                #else:
                    #right_image = infill_from_deep_side(right_image, right_img_mask, 'right')
            
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
    
    raw_video.release()
    out.release()
    
    if mask_video is not None:
        mask_video.release()
    
    if infill_mask_video is not None:
        infill_mask_video.release()

