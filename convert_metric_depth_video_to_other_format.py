import argparse
import cv2
import numpy as np
import os
import sys
import depth_map_tools
import json
import open3d as o3d

np.set_printoptions(suppress=True, precision=4)


def float_image_to_byte_image(float_image, max_value=10.0, scale=255, log_scale=5):
    # Ensure that no values are below a very small positive number to avoid log(0)
    epsilon = 0.0001
    float_image = np.clip(float_image, epsilon, max_value)
    
    # Apply logarithmic scaling
    transformed = np.log(float_image * log_scale + 1)
    max_log = np.log(max_value * log_scale + 1)
    
    # Normalize to fit into 0-255
    normalized = transformed / max_log * scale
    
    # Convert to integers and clip to ensure values stay in the 0-255 range
    byte_image = np.clip(normalized, 0, scale).astype(np.uint8)
    
    return byte_image


def best_intersection_point_vectorized(points, directions):
    """
    Finds the best intersecting point of many lines in 3D using a vectorized approach.
    
    Parameters:
        points (np.ndarray): An array of shape (N, 3) where each row is a point p_i.
        directions (np.ndarray): An array of shape (N, 3) where each row is a direction d_i.
    
    Returns:
        x (np.ndarray): The best intersecting point in 3D.
    """
    # Ensure directions are normalized
    d_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    N = points.shape[0]
    
    # Compute the sum of outer products d_i d_i^T
    # This can be done by (d_norm.T @ d_norm) since for each i, d_i d_i^T contributes to the sum.
    sum_outer = d_norm.T @ d_norm  # shape (3, 3)
    
    # Construct A: A = N*I - sum_i(d_i d_i^T)
    A = N * np.eye(3) - sum_outer
    
    # Compute b: for each i, b_i = p_i - d_i*(d_i^T p_i)
    # First compute dot products for each line.
    dp = np.sum(d_norm * points, axis=1, keepdims=True)  # shape (N, 1)
    b_individual = points - d_norm * dp  # shape (N, 3)
    
    # Sum over all lines to get b.
    b = np.sum(b_individual, axis=0)
    
    # Solve the linear system A x = b for the best intersection point.
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x

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


def find_nearby_points(points_3d, i, threshold=0.01, exclude_self=True):
    """
    Finds all points within a given threshold from points_3d[i].

    Parameters:
        points_3d (np.ndarray): Array of shape (N, 3) with 3D points.
        i (int): Index of the current point.
        threshold (float): Distance threshold.
        exclude_self (bool): Whether to exclude the point itself.
    
    Returns:
        np.ndarray: Indices of points within the threshold distance.
        np.ndarray: The corresponding 3D points.
    """
    current_point = points_3d[i]
    # Compute Euclidean distances from current_point to all points.
    distances = np.linalg.norm(points_3d - current_point, axis=1)
    
    # Create a mask for points within the threshold.
    mask = distances < threshold
    if exclude_self:
        mask[i] = False  # Optionally exclude the current point.
    
    nearby_indices = np.where(mask)[0]
    return nearby_indices
    
class UnionFind:
    def __init__(self, items):
        # Initialize each item as its own parent.
        self.parent = {item: item for item in items}

    def find(self, x):
        # Path compression.
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            # Attach root_b to root_a.
            self.parent[root_b] = root_a

def merge_global_points(global_3d_points, remaped_points):
    """
    Merge points in global_3d_points based on the remapping dictionary.
    This version uses union-find for faster merging.

    Parameters:
        global_3d_points (dict): Keys are point IDs and values are lists of 3 lists (e.g., x, y, z observations).
        remaped_points (dict): Keys are global IDs mapping to lists of IDs that should be merged into them.

    Returns:
        None: global_3d_points is modified in-place.
    """
    # Initialize union-find with all keys in global_3d_points.
    uf = UnionFind(global_3d_points.keys())

    # Union the points according to remaped_points.
    for global_id, pts in remaped_points.items():
        if global_id not in uf.parent:
            continue
        for rem_global_id in pts:
            if rem_global_id in uf.parent:
                uf.union(global_id, rem_global_id)

    # Group keys by their final representative (root).
    groups = {}
    for key in list(uf.parent.keys()):
        root = uf.find(key)
        groups.setdefault(root, []).append(key)

    # Merge all points in each group into the representative (the union-find root)
    for root, group_keys in groups.items():
        if len(group_keys) <= 1:
            continue  # nothing to merge
        # Merge data from every key into root.
        for key in group_keys:
            if key == root:
                continue
            # Extend the lists from key into root.
            global_3d_points[root][0].extend(global_3d_points[key][0])
            global_3d_points[root][1].extend(global_3d_points[key][1])
            global_3d_points[root][2].extend(global_3d_points[key][2])
            # Remove the merged key.
            del global_3d_points[key]


if __name__ == '__main__':
    
    # Setup arguments
    parser = argparse.ArgumentParser(description='Convert depth video other formats like .obj or .ply or greyscale video')
    
    parser.add_argument('--depth_video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--bit16', action='store_true', help='Convert depth video to a 16bit mono grayscale video file', required=False)
    parser.add_argument('--bit8', action='store_true', help='Convert depth video to a rgb grayscale video file', required=False)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    
    parser.add_argument('--save_ply', type=str, help='folder to save .ply pointcloud files in', required=False)
    parser.add_argument('--save_obj', type=str, help='folder to save .obj mesh files in', required=False)
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=False)
    parser.add_argument('--xfov', type=float, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=float, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--min_frames', default=-1, type=int, help='start convertion after nr of frames', required=False)
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--transformation_file', type=str, help='file with scene transformations from the aligner', required=False)
    parser.add_argument('--transformation_lock_frame', default=0, type=int, help='the frame that the transformation will use as a base', required=False)
    parser.add_argument('--remove_edges', action='store_true', help='Tries to remove edges that was not visible in image', required=False)
    
    parser.add_argument('--track_file', type=str, help='file with 2d point tracking data', required=False)
    parser.add_argument('--strict_mask', default=False, action='store_true', help='Remove any points that has ever been masked out even in frames where they are not masked', required=False)
    parser.add_argument('--mask_video', type=str, help='black and white mask video for thigns that should not be tracked', required=False)
    
    
    args = parser.parse_args()
    
   
    MODEL_maxOUTPUT_depth = args.max_depth
    
    # Verify input file exists
    if not os.path.isfile(args.depth_video):
        raise Exception("input video does not exist")

        
    output_file = args.depth_video + "_grey_depth.mkv"
    
    global_3d_points = None
    
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
    out = None
    if args.bit16:
        out = cv2.VideoWriter(
            filename=output_file,
            apiPreference=cv2.CAP_FFMPEG,
            fourcc=cv2.VideoWriter_fourcc(*"FFV1"),
            fps=frame_rate,
            frameSize=(frame_width, frame_height),
            params=[
                cv2.VIDEOWRITER_PROP_DEPTH,
                cv2.CV_16U,
                cv2.VIDEOWRITER_PROP_IS_COLOR,
                0,  # false
            ],
        )
    elif args.bit8:
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"FFV1"), frame_rate, (frame_width, frame_height))
    
    if args.save_ply is not None:
        if not os.path.exists(args.save_ply):
            os.makedirs(args.save_ply)
    
    if args.save_obj is not None:
        if not os.path.exists(args.save_obj):
            os.makedirs(args.save_obj)
            
    mask_video = None
    if args.mask_video is not None:
        if not os.path.isfile(args.mask_video):
            raise Exception("input mask_video does not exist")
        mask_video = cv2.VideoCapture(args.mask_video)
    
    frames = None
    if args.track_file is not None:
        with open(args.track_file) as json_track_file_handle:
            frames = json.load(json_track_file_handle)
    
    cam_matrix = None
    if args.save_ply is not None or args.save_obj is not None:
        if args.xfov is None and args.yfov is None:
            print("Either --xfov or --yfov is required.")
            exit(0)
    
    if args.xfov is not None or args.yfov is not None:
        cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)
        fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)
        print("Camera fovx: ", fovx, "fovy:", fovy)
        
    color_video = None
    if args.color_video is not None:
        if not os.path.isfile(args.color_video):
            raise Exception("input color_video does not exist")
        color_video = cv2.VideoCapture(args.color_video)
    
    transformations = None
    if args.transformation_file is not None:
        if not os.path.isfile(args.transformation_file):
            raise Exception("input transformation_file does not exist")
        with open(args.transformation_file) as json_file_handle:
            transformations = json.load(json_file_handle)
            transformations.insert(0, np.eye(4))#Mega sam bug, TODO fix in megsam script
    
        if args.transformation_lock_frame != 0:
            ref_frame = transformations[args.transformation_lock_frame]
            ref_frame_inv_trans = np.linalg.inv(ref_frame)
            for i, transformation in enumerate(transformations):
                transformations[i] = transformation @ ref_frame_inv_trans

    saved_depth_maps = None
    #Lets do 3d reconstruction
    if args.transformation_file is not None and args.track_file is not None and cam_matrix is not None:
        global_3d_points = {}
        saved_depth_maps = []
        output_file = args.depth_video + "_rescaled.mkv"
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"FFV1"), frame_rate, (frame_width, frame_height))
    
    remaped_points = {}
    frame_n = 0
    mesh = None
    while raw_video.isOpened():
        
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        
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
            
        if args.min_frames >= frame_n and args.min_frames != -1:
            frame_n += 1
            continue
            
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        
        depth_unit[..., 3] = ((rgb[..., 0].astype(np.uint32) + rgb[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb[..., 2]
        
        
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        
        if mask_video is not None and frames is not None:
            ret, mask = mask_video.read()
            if ret:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


                rem = []
                rem_global = []
                for i, point in enumerate(frames[frame_n]):
                    if point[1] >= frame_width or point[2] >= frame_height or mask[point[2], point[1]] > 0:
                        rem.append(i)
                        rem_global.append(point[0])

                if len(rem) > 0:
                    frames[frame_n] = np.delete(frames[frame_n], rem, axis=0)

                if args.strict_mask:
                    for global_id in rem_global:
                        for frame_id, frame in enumerate(frames):
                            rem = []
                            for i, point in enumerate(frames[frame_n]):
                                if global_id == point[0]:
                                    rem.append(i)
                            if len(rem) > 0:
                                frames[frame_id] = np.delete(frames[frame_id], rem, axis=0)
            else:
                print("WARNING: mask video ended before other videos")
        
        
        if cam_matrix is not None:
            transform_to_zero = np.eye(4)
            if transformations is not None:
                transform_to_zero = np.array(transformations[frame_n])
            mesh_ret, used_indices = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, mesh, remove_edges = args.remove_edges)
            
            if transformations is not None:
                mesh_ret.transform(transform_to_zero)
            
                #frames is a bad name but it a var that holds all frames with 2d tracking points
                if frames is not None:
                    
                    saved_depth_maps.append(depth)
                    
                    point_ids_in_this_frame = frames[frame_n][:,0]
                    points_2d = frames[frame_n][:, 1:3]
                    points_3d = depth_map_tools.project_2d_points_to_3d(points_2d, depth, cam_matrix)
                    transform_to_zero_rot = transform_to_zero.copy()
                    transform_to_zero_rot[:3, 3] = 0.0
                    points_3d = depth_map_tools.transform_points(points_3d, transform_to_zero_rot)
                    cam_pos = transform_to_zero[:3, 3]
                    
                    for i, global_id in enumerate(point_ids_in_this_frame):
                        if global_id not in global_3d_points:
                            global_3d_points[global_id] = [[],[],[],[]]
                        nearby_points = find_nearby_points(points_3d, i)
                        for pt in nearby_points:
                            if global_id not in remaped_points:
                                remaped_points[global_id] = []
                            remaped_points[global_id].append(point_ids_in_this_frame[pt])
                        
                        global_3d_points[global_id][0].append(cam_pos)
                        global_3d_points[global_id][1].append(points_3d[i])
                        global_3d_points[global_id][2].append(np.array(color_frame[points_2d[i][1], points_2d[i][0]], dtype=np.float32)/255)
                        global_3d_points[global_id][3].append(frame_n)
                
            if args.save_obj is not None:
                file_name = args.save_obj+f"/{frame_n:07d}"+".obj"
                # TODO Remove the unused vertices that represent removed edges If we just remove the vertices
                # the "order" will chnange and the tringles will point to the wrong vertex
                write_mesh = o3d.geometry.TriangleMesh()
                triangles = np.asarray(mesh_ret.triangles)
                no_removed_triangles = ~np.all(triangles == 0, axis=1)
                write_mesh.triangles = o3d.utility.Vector3iVector(triangles[no_removed_triangles])
                write_mesh.vertices = mesh_ret.vertices
                write_mesh.vertex_colors = mesh_ret.vertex_colors
                o3d.io.write_triangle_mesh(file_name, write_mesh)
            if args.save_ply is not None:
                file_name = args.save_ply+f"/{frame_n:07d}"+".ply"
                points = np.asarray(mesh_ret.vertices)[used_indices]
                colors = np.asarray(mesh_ret.vertex_colors)[used_indices]
                
                pcd = depth_map_tools.pts_2_pcd(points, colors)
                o3d.io.write_point_cloud(file_name, pcd)
            
            mesh = mesh_ret
        if args.bit16:
            depth = depth*((255**2)/MODEL_maxOUTPUT_depth)
            depth = np.rint(depth).astype(np.uint16)
            out.write(depth)
        elif args.bit8:
            depth = depth*(255/MODEL_maxOUTPUT_depth)
            depth = np.rint(depth).astype(np.uint8)
            vid_depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            out.write(vid_depth)
        
        
        
        
        if args.max_frames < frame_n and args.max_frames != -1:
            break
            
        frame_n += 1
    
    if global_3d_points is not None:
        
        merge_global_points(global_3d_points, remaped_points)
        
        
        points = []
        colors = []
        messured_points = {}
        messured_points_3d = {}
        for global_id in global_3d_points:
            com_poses = np.array(global_3d_points[global_id][0])
            
            if len(com_poses) > 20:
                line_directions = np.array(global_3d_points[global_id][1])
                #print("global_id", global_id, "poses:", com_poses, "dirs:", line_directions)
                #exit(0)
                intersection_point = best_intersection_point_vectorized(com_poses, line_directions)
                if intersection_point is None:
                    continue
            
                print("Global id:", global_id," nr observations:", len(com_poses), "best Iintersection point:", intersection_point)
                points.append(intersection_point)
                
                for frame_n in global_3d_points[global_id][3]:
                    if frame_n not in messured_points:
                        messured_points[frame_n] = []
                        messured_points_3d[frame_n] = []
                    messured_points[frame_n].append(global_id)
                    messured_points_3d[frame_n].append(intersection_point)
                
                rgb = np.array(global_3d_points[global_id][2])
                colors.append(np.mean(rgb, axis=0))
            
        pcd = depth_map_tools.pts_2_pcd(points, colors)
        depth_map_tools.draw([pcd])
        target = []
        source = []
        print("rescaling depthmap based on triangulated depth")
        for frame_n, depth in enumerate(saved_depth_maps):
            
            
            global_points_in_frame = []
            global_points_3d_in_frame = []
            if frame_n in messured_points:
                global_points_in_frame = messured_points[frame_n]
                global_points_3d_in_frame = messured_points_3d[frame_n]
                    
            #global_points_3d_in_frame = np.array(global_points_3d_in_frame)
            
            if len(global_points_in_frame) == 0:
                continue
            
            #print(global_points_3d_in_frame)
            global_points_3d_in_frame = np.array(global_points_3d_in_frame)
            
            transform_from_zero = np.linalg.inv(np.array(transformations[frame_n]))
            
            point_ids_in_this_frame = frames[frame_n][:,0]
            cur_mask = np.isin(point_ids_in_this_frame, global_points_in_frame)
            points_2d = frames[frame_n][cur_mask][:, 1:3]
            points_3d = depth_map_tools.project_2d_points_to_3d(points_2d, depth, cam_matrix)
            
            ref_points_3d = depth_map_tools.transform_points(global_points_3d_in_frame, transform_from_zero)
            
            target.append(1/ref_points_3d[:,2])
            source.append(1/points_3d[:, 2])
            
            #scale = np.mean(points_3d[:, 2]/global_points_3d_in_frame[:,2])
            
        #scale = np.mean(np.concatenate(source)/ np.concatenate(target))
        scale, shift = compute_scale_and_shift_full(np.concatenate(source), np.concatenate(target))
            
        for frame_n, depth in enumerate(saved_depth_maps):
            
            target = []
            source = []
            
            inv_depth = 1/depth
            inverse_reconstructed_metric_depth = (inv_depth * scale) + shift
            
            fixed_depth = 1/inverse_reconstructed_metric_depth
            #fixed_depth = depth / scale
            
            scaled_depth = (((255**4)/MODEL_maxOUTPUT_depth)*fixed_depth.astype(np.float64)).astype(np.uint32)

            # View the depth as raw bytes: shape (H, W, 4)
            depth_bytes = scaled_depth.view(np.uint8).reshape(frame_height, frame_width, 4)


            R = (depth_bytes[:, :, 3]) # Most significant bits in R and G channel (duplicated to reduce compression artifacts)
            G = (depth_bytes[:, :, 3])
            B = (depth_bytes[:, :, 2]) # Least significant bit in blue channel
            bgr24bit = np.dstack((B, G, R))

            out.write(bgr24bit)
        
    raw_video.release()
    if out is not None:
        out.release()
