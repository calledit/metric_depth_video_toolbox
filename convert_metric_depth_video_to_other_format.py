import argparse
import cv2
import numpy as np
import os
import sys
import depth_map_tools
import json
import copy
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


def compute_weights(directions):
    """
    Compute weights for each direction based on its average angular difference
    with all other directions. The measure used is 1 - |dot product|.
    
    Parameters:
        directions (np.ndarray): Array of shape (N, 3) of normalized direction vectors.
    
    Returns:
        weights (np.ndarray): Array of shape (N,) with computed weights.
    """
    N = directions.shape[0]
    # Compute the pairwise dot products
    dot_products = directions @ directions.T  # shape (N, N)
    # Take absolute value so that parallel and anti-parallel are both considered similar
    abs_dot = np.abs(dot_products)
    # Compute difference measure: 1 - |dot|
    diff = 1 - abs_dot
    # Exclude the diagonal (self comparison) which is 0 since |dot(d_i,d_i)| = 1.
    # Average the differences over the other directions.
    weights = np.sum(diff, axis=1) / (N - 1)
    return weights

def best_intersection_point_vectorized_weighted(points, directions, weights=None):
    """
    Finds the best intersecting point of many 3D lines in a weighted least-squares sense.
    
    Each line is defined by a point and a normalized direction vector.
    The goal is to find x that minimizes:
    
        J(x) = sum_i w_i * || (I - d_i d_i^T)(x - p_i) ||^2
    
    If weights are not provided, they are computed based on the average angular difference.
    
    Parameters:
        points (np.ndarray): Array of shape (N, 3) with each row as the line's origin.
        directions (np.ndarray): Array of shape (N, 3) with each row as the line's direction.
        weights (np.ndarray, optional): Array of shape (N,) of weights for each line.
            If None, weights are computed from the directions.
    
    Returns:
        x (np.ndarray): The best intersection point in 3D (shape (3,)).
    """
    # Normalize direction vectors
    d_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    N = points.shape[0]
    
    # If no weights are provided, compute them based on angular differences.
    if weights is None:
        weights = compute_weights(d_norm)
    # Ensure weights is a column vector for broadcasting.
    weights = weights.reshape(-1, 1)
    
    # Instead of computing the full projection matrix per line,
    # note that for each line:
    #   (I - d d^T) p = p - d (d^T p)
    # Also, the sum of projection matrices:
    #   A = sum_i w_i * (I - d_i d_i^T)
    # can be computed as:
    #   A = (sum_i w_i)*I - sum_i w_i * (d_i d_i^T)
    
    # Compute weighted sum of outer products d_i d_i^T:
    weighted_outer = (d_norm * weights)  # shape (N, 3)
    weighted_outer = weighted_outer.T @ d_norm  # shape (3, 3)
    
    # Sum of weights:
    sum_weights = np.sum(weights)
    
    # Form the matrix A
    A = sum_weights * np.eye(3) - weighted_outer
    
    # Compute the weighted sum for b:
    # For each line, b_i = w_i * (p_i - d_i * (d_i dot p_i))
    dp = np.sum(d_norm * points, axis=1, keepdims=True)  # shape (N, 1)
    b_individual = points - d_norm * dp  # shape (N, 3)
    # Apply weights and sum over all lines:
    b = np.sum(weights * b_individual, axis=0)
    
    # Solve the linear system A x = b (using lstsq to be robust in case A is singular)
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x


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
            
def add_open3d_mesh(o3d_mesh, object_name="ImportedMesh"):
    """
    Converts an Open3D TriangleMesh to a Blender mesh object and adds it to the current collection.
    """
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    mesh_data = bpy.data.meshes.new(object_name + "Mesh")
    obj = bpy.data.objects.new(object_name, mesh_data)
    bpy.context.collection.objects.link(obj)
    
    vertices_list = [tuple(v) for v in vertices]
    faces_list = [tuple(face) for face in faces]
    
    mesh_data.from_pydata(vertices_list, [], faces_list)
    mesh_data.update()
    
    if o3d_mesh.has_vertex_colors():
        vertex_colors = np.asarray(o3d_mesh.vertex_colors)
        if len(vertex_colors) != len(vertices):
            raise ValueError("Number of vertex colors must match number of vertices.")
        color_attr = mesh_data.color_attributes.new(name="Col", domain='POINT', type='FLOAT_COLOR')
        for i, col in enumerate(vertex_colors):
            if len(col) == 3:
                color_attr.data[i].color = (col[0], col[1], col[2], 1.0)
            else:
                color_attr.data[i].color = tuple(col)
    return obj

def add_point_cloud(point_cloud, point_colors=None, object_name="PointCloud"):
    """
    Creates a Blender mesh object from a point cloud (Nx3 numpy array) with vertices only.
    Optionally assigns per-vertex colors.
    """
    mesh = bpy.data.meshes.new(object_name + "Mesh")
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)
    
    vertices = [tuple(pt) for pt in point_cloud]
    mesh.from_pydata(vertices, [], [])
    mesh.update()
    
    if point_colors is not None:
        if len(point_colors) != len(vertices):
            raise ValueError("Number of point colors must match number of vertices.")
        color_attr = mesh.color_attributes.new(name="Col", domain='POINT', type='FLOAT_COLOR')
        for i, col in enumerate(point_colors):
            if len(col) == 3:
                color_attr.data[i].color = (col[0], col[1], col[2], 1.0)
            else:
                color_attr.data[i].color = tuple(col)
    return obj

def assign_vertex_color_material(obj, vcol_name="Col"):
    """
    Creates and assigns a simple material that uses the given vertex color attribute.
    """
    mat = bpy.data.materials.new(name="PointCloudMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes.
    for node in nodes:
        nodes.remove(node)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse_node = nodes.new(type="ShaderNodeBsdfDiffuse")
    vcol_node = nodes.new(type="ShaderNodeVertexColor")
    vcol_node.layer_name = vcol_name

    links.new(vcol_node.outputs["Color"], diffuse_node.inputs["Color"])
    links.new(diffuse_node.outputs["BSDF"], output_node.inputs["Surface"])

    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
        


def create_camera_alembic(transforms, output_file, fps=24.0, camera_name="TrackedCamera",
                           intrinsic_matrix=None, resolution=(1920, 1080),
                           point_cloud_points=None, point_cloud_colors=None,
                           point_cloud_name="PointCloud",
                           open3d_mesh=None, open3d_mesh_name="ImportedMesh",
                           blend_filepath=None):
    """
    Creates an animated camera in Blender from a list of 4x4 transformation matrices,
    optionally adds a colored point cloud and an Open3D mesh, then exports the scene
    as an Alembic (.abc) file. Optionally, the Blender file can also be saved.
    
    Parameters:
        transforms (list or np.ndarray):
            A list of 4x4 NumPy arrays representing world transforms for each frame.
        output_file (str):
            The file path for the output Alembic file.
        fps (float):
            Desired frames per second (e.g., 29.97, 24, 30).
        camera_name (str):
            Name for the created camera object.
        intrinsic_matrix (np.ndarray):
            A 3x3 camera intrinsic matrix:
                [ fx   0  cx ]
                [  0  fy  cy ]
                [  0   0   1 ]
            If provided, it is used to set the camera’s focal length and sensor size.
        resolution (tuple):
            The image resolution as (width, height) in pixels.
        point_cloud_points (np.ndarray, optional):
            An Nx3 array of 3D points for a point cloud.
        point_cloud_colors (np.ndarray, optional):
            An Nx3 or Nx4 array of per-vertex colors (RGB or RGBA, values 0-1).
        point_cloud_name (str):
            Name for the point cloud object.
        open3d_mesh (open3d.geometry.TriangleMesh, optional):
            An Open3D mesh to import into the Blender scene.
        open3d_mesh_name (str):
            Name for the imported Open3D mesh object.
        blend_filepath (str, optional):
            If provided, the Blender file (.blend) will be saved to this path.
    """
    # Create a new camera.
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = camera_name
    cam_data = camera.data

    # Set camera intrinsics if provided.
    if intrinsic_matrix is not None:
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        
        sensor_width = 36.0  
        image_width, image_height = resolution
        computed_lens = fx * (sensor_width / image_width)
        sensor_height = sensor_width * image_height / image_width
        
        cam_data.lens = computed_lens
        cam_data.sensor_width = sensor_width
        cam_data.sensor_height = sensor_height
        
        print(f"Using intrinsic matrix: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"Computed lens: {computed_lens} mm, sensor size: {sensor_width}x{sensor_height} mm")
    else:
        cam_data.lens = 35.0
        cam_data.sensor_width = 36.0
        cam_data.sensor_height = 24.0

    # Create animation data for the camera.
    if camera.animation_data is None:
        camera.animation_data_create()
    camera.animation_data.action = bpy.data.actions.new(name=f"{camera_name}_Action")
    
    fps_fraction = Fraction(fps).limit_denominator(1001)
    bpy.context.scene.render.fps = fps_fraction.numerator
    bpy.context.scene.render.fps_base = fps_fraction.denominator
    
    num_frames = len(transforms)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    
    # Animate the camera: apply each transformation as a keyframe.
    for frame, mat in enumerate(transforms, start=1):
        bpy.context.scene.frame_set(frame)
        # Convert the transformation matrix and apply correction.
        bl_matrix = mathutils.Matrix(mat.tolist())
        # Correct the orientation: rotate 180° about the Y axis.
        correction = mathutils.Matrix.Rotation(np.pi, 4, 'X')
        bl_matrix = bl_matrix @ correction
        camera.matrix_world = bl_matrix
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
        camera.keyframe_insert(data_path="scale", frame=frame)
    
    # Optionally, add a point cloud.
    point_cloud_obj = None
    if point_cloud_points is not None:
        point_cloud_obj = add_point_cloud(point_cloud_points, point_colors=point_cloud_colors, object_name=point_cloud_name)
        assign_vertex_color_material(point_cloud_obj, vcol_name="Col")
    
    # Optionally, add an Open3D mesh.
    open3d_mesh_obj = None
    if open3d_mesh is not None:
        open3d_mesh_obj = add_open3d_mesh(open3d_mesh, object_name=open3d_mesh_name)
        assign_vertex_color_material(open3d_mesh_obj, vcol_name="Col")
        
        
    # --- Global Coordinate Conversion ---
    # Create an empty object that applies the coordinate conversion from external (Y-up)
    # to Blender (Z-up). For example, if your external system has the floor in the Z-X plane (Y up),
    # then swapping Y and Z will convert it to Blender's XY floor.
    # The conversion matrix below swaps Y and Z:
    conversion_matrix = mathutils.Matrix((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1)
    ))
    bpy.ops.object.empty_add(type='PLAIN_AXES')
    global_empty = bpy.context.active_object
    global_empty.name = "GlobalCorrection"
    global_empty.matrix_world = conversion_matrix

    # Parent your scene objects to the global empty.
    for obj in (camera, point_cloud_obj, open3d_mesh_obj):
        if obj is not None:
            obj.parent = global_empty
            
            
    bpy.ops.wm.save_as_mainfile(filepath=output_file+".blend")

    # Select the global empty and all its children.
    bpy.ops.object.select_all(action='DESELECT')
    global_empty.select_set(True)
    for child in global_empty.children_recursive:
        child.select_set(True)
    bpy.context.view_layer.objects.active = global_empty

    # Export the global empty (with children) to Alembic.
    bpy.ops.wm.alembic_export(
        filepath=output_file,
        selected=True,
        start=bpy.context.scene.frame_start,
        end=bpy.context.scene.frame_end,
        vcolors=True,
        flatten=False  # Do not flatten to preserve hierarchy
    )
    print("Alembic export complete:", output_file)


def estimate_scale_shift(depth, depth_target):
    """
    Estimates the scale and shift constants to map 'depth' to 'depth_target'
    using the model:
    
        1/depth_target = scale * (1/depth) + shift
    
    Parameters:
        depth (np.ndarray): 1D array of original depth values.
        depth_target (np.ndarray): 1D array of target depth values.
        
    Returns:
        scale (float): Estimated scale constant.
        shift (float): Estimated shift constant.
    """
    # Ensure no zeros to avoid division issues:
    valid = (depth > 0) & (depth_target > 0)
    d = depth[valid]
    d_t = depth_target[valid]
    
    # Compute the transformed variables.
    x = 1.0 / d
    y = 1.0 / d_t
    
    # Build the design matrix for the linear model y = a*x + b
    X = np.vstack([x, np.ones_like(x)]).T
    
    # Solve the least-squares problem
    solution, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    scale, shift = solution
    
    return scale, shift


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
    
    parser.add_argument('--show_scene_point_clouds', action='store_true', help='Opens window and shows the resulting pointclouds', required=False)
    parser.add_argument('--show_both_point_clouds', action='store_true', help='If the viewer should show both pointclouds overlapping', required=False)
    
    
    parser.add_argument('--save_alembic', action='store_true', help='Save data to a alembic file', required=False)
    parser.add_argument('--use_triangulated_points', action='store_true', help='If the triangulated points should be used', required=False)
    
    parser.add_argument('--save_rescaled_depth', action='store_true', help='Saves a video with rescaled depth', required=False)
    parser.add_argument('--global_align', action='store_true', help='Aligns the depth video to the triangulated depth', required=False)
    
    
    
    args = parser.parse_args()
    
    if args.global_align:
        args.use_triangulated_points = True
    
   
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
            
        if args.transformation_lock_frame != 0:
            ref_frame = transformations[args.transformation_lock_frame]
            ref_frame_inv_trans = np.linalg.inv(ref_frame)
            for i, transformation in enumerate(transformations):
                transformations[i] = transformation @ ref_frame_inv_trans

    
    alembic_mesh = None
    saved_depth_maps = None
    #Lets do 3d reconstruction
    if args.transformation_file is not None and args.track_file is not None and cam_matrix is not None:
        global_3d_points = {}
        saved_depth_maps = []
        output_file = args.depth_video + "_rescaled.mkv"
        if args.save_rescaled_depth:
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
        
        #Test values for crop_dancing
        #depth = 1/depth
        #depth *= 0.55 #decrese to move back out from camera
        #depth = 1/(depth+0.25) #increse to compress the scene
        
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
                
                if alembic_mesh is None:
                    alembic_mesh = copy.deepcopy(mesh_ret)
            
                #frames is a bad name but it a var that holds all frames with 2d tracking points
                if frames is not None:
                    
                    saved_depth_maps.append(depth)
                    
                    point_ids_in_this_frame = frames[frame_n][:,0]
                    points_2d = frames[frame_n][:, 1:3]
                    points_3d = depth_map_tools.project_2d_points_to_3d(points_2d, depth, cam_matrix)
                    transform_to_zero_rot = transform_to_zero.copy()
                    transform_to_zero_rot[:3, 3] = 0.0
                    points_3d_rot = depth_map_tools.transform_points(points_3d, transform_to_zero_rot)
                    points_3d_trans = depth_map_tools.transform_points(points_3d, transform_to_zero)
                    cam_pos = transform_to_zero[:3, 3]
                    
                    for i, global_id in enumerate(point_ids_in_this_frame):
                        if global_id not in global_3d_points:
                            global_3d_points[global_id] = [[],[],[],[],[]]
                            
                        #Disable Temporarily
                        nearby_points = find_nearby_points(points_3d_rot, i, 0.005)
                        for pt in nearby_points:
                            if global_id not in remaped_points:
                                remaped_points[global_id] = []
                            remaped_points[global_id].append(point_ids_in_this_frame[pt])
                        
                        global_3d_points[global_id][0].append(cam_pos)
                        global_3d_points[global_id][1].append(points_3d_rot[i])
                        global_3d_points[global_id][2].append(np.array(color_frame[points_2d[i][1], points_2d[i][0]], dtype=np.float32)/255)
                        global_3d_points[global_id][3].append(frame_n)
                        global_3d_points[global_id][4].append(points_3d_trans[i])
                
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
    
    alembic_point_cloud = None
    alembic_point_cloud_colors = None
    
    if global_3d_points is not None:
        
        #Merge points that overlap at some point
        merge_global_points(global_3d_points, remaped_points)
        
        
        points = []
        points_i = []
        colors = []
        messured_points = {}
        messured_points_3d = {}
        dist_m_2_a = {}
        avg_mon_points_3d = {}
        raw_points_3d = {}
        #Find intersecting rays from the camera and use that to determine distance and make a point cloud
        #Also take averages of the depth from the depth map to make a point cloud
        for global_id in global_3d_points:
            com_poses = np.array(global_3d_points[global_id][0])
            
            if len(com_poses) > 2:
                line_directions = np.array(global_3d_points[global_id][1])
                #print("global_id", global_id, "poses:", com_poses, "dirs:", line_directions)
                #exit(0)
                intersection_point = np.array([0.0,0.0,0.0])
                if args.use_triangulated_points:
                    intersection_point = best_intersection_point_vectorized_weighted(com_poses, line_directions)
                    if intersection_point is None:
                        continue
                
                mon_dep = np.mean(global_3d_points[global_id][4], axis=0)
            
                print("Global id:", global_id," nr observations:", len(com_poses), "best intersection point:", intersection_point, "mono:", mon_dep)
                points.append(mon_dep)
                points_i.append(intersection_point)
                
                for k, frame_n in enumerate(global_3d_points[global_id][3]):
                    if frame_n not in messured_points:
                        messured_points[frame_n] = []
                        messured_points_3d[frame_n] = []
                        avg_mon_points_3d[frame_n] = []
                        dist_m_2_a[frame_n] = []
                        raw_points_3d[frame_n] = []
                    messured_points[frame_n].append(global_id)
                    messured_points_3d[frame_n].append(intersection_point)
                    avg_mon_points_3d[frame_n].append(mon_dep)
                    dist_m_2_a[frame_n].append(np.linalg.norm(mon_dep-intersection_point))
                    raw_points_3d[frame_n].append(global_3d_points[global_id][4][k])
                
                rgb = np.array(global_3d_points[global_id][2])
                colors.append(np.mean(rgb, axis=0))
        
        
        if args.use_triangulated_points:
            alembic_point_cloud = points_i
        else:
            alembic_point_cloud = points
        alembic_point_cloud_colors = colors
        
        print("Creating and saving pcd files")
        pcd = depth_map_tools.pts_2_pcd(np.array(points), colors)
        o3d.io.write_point_cloud(args.depth_video + "_avgmonodepth.ply", pcd)
        pcd_i = None
        if args.use_triangulated_points:
            pcd_i = depth_map_tools.pts_2_pcd(np.array(points_i), colors)
            o3d.io.write_point_cloud(args.depth_video + "_triangulated.ply", pcd_i)

        
        
        if args.show_scene_point_clouds:
            print("Showing pointclouds")
            if args.show_both_point_clouds and args.use_triangulated_points:
                depth_map_tools.draw([pcd, pcd_i])
            else:
                depth_map_tools.draw([depth_map_tools.pts_2_pcd(np.array(alembic_point_cloud), colors)])
        
        target = []
        source = []
        
        if args.save_rescaled_depth:
            # this sort of works but you get better shift and scale values by selecting values yourself (but that is manual work)
            # i have observed that you get better output if you run this iteravly using the rescaled output as input 2-3 times.
            # dont know why or how. But it is an observation.
            print("rescaling depthmap based on triangulated depth (run in iterations for better result)")
            for frame_n, depth in enumerate(saved_depth_maps):
        
        
                global_points_in_frame = []
                global_points_3d_in_frame = []
                if frame_n in messured_points:
                    global_points_in_frame = messured_points[frame_n]
                    mes_ref_points_3d = np.array(messured_points_3d[frame_n])
                    points_3d_avg = np.array(avg_mon_points_3d[frame_n])
                    frame_points = np.array(raw_points_3d[frame_n])
                
                if len(global_points_in_frame) == 0:
                    continue
        
                transform_from_zero = np.linalg.inv(np.array(transformations[frame_n]))
                
                
                # There are multiple ways of scaling here.
                # One is to scale the average points to the messured points (over all frames) this
                # will give you a global scaling factor that tries to globally align the video to the messured values
                # Another is to scale each frame indevidually to the average points. This should give better internal consistency within the video
        
                points_3d = depth_map_tools.transform_points(frame_points, transform_from_zero)
                ref_points_3d = depth_map_tools.transform_points(points_3d_avg, transform_from_zero)
                
                
                scale = np.mean(ref_points_3d[:,2]/points_3d[:, 2])
                
                # apply self alignment
                saved_depth_maps[frame_n] = depth*scale
                #points_3d[:, 2] *= scale
        
                
                if args.global_align:
                    
                    points_3d = ref_points_3d
                    ref_points_3d = depth_map_tools.transform_points(mes_ref_points_3d, transform_from_zero)
                    
                    # filter away ponts that are to far from eachother, can good for global alignment in certain cases like
                    # when the triangulated truth is very difrent from the ML model output. Adding more tracking points is often
                    # a better alternative than doing this filtering
                    #dists = np.array(dist_m_2_a[frame_n])
                    #m_d = np.mean(dists)
                    #mask = dists < m_d
                    
                    #Global need this since we need to account for shift
                    scale, shift = estimate_scale_shift(points_3d[:, 2], ref_points_3d[:,2])
            
                    #we filter away extreme values. Dont know exatly why those values apear anyway filtering gives better result
                    if abs(shift) > 1 or abs(scale) > 3:
                        print("ignoring bad align: frame shift:", shift, "scale:", scale)
                        continue
            
                    target.append(ref_points_3d[:,2])
                    source.append(points_3d[:, 2])
                    
        
        
            scale, shift = estimate_scale_shift(np.concatenate(source), np.concatenate(target))
            print("global alignment scale:", scale, "shift:", shift)
            target = []
            source = []
    
            for frame_n, depth in enumerate(saved_depth_maps):
        
   
                if args.global_align:
                    inv_depth = 1/depth
                    inverse_reconstructed_metric_depth = (inv_depth * scale) + shift
                    fixed_depth = 1/inverse_reconstructed_metric_depth
                else:
                    fixed_depth = depth
        
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
        
    if args.save_alembic:
        try:
            import bpy
            from fractions import Fraction
            import mathutils
            create_camera_alembic(np.array(transformations), args.depth_video + "_alembic.abc", fps=frame_rate,
            intrinsic_matrix=cam_matrix, resolution=(frame_width, frame_height), point_cloud_points=alembic_point_cloud,
            point_cloud_colors=alembic_point_cloud_colors, open3d_mesh=alembic_mesh)
        except ImportError:
            print("Error: 'bpy' module is not installed and is required to export to alembic. Install with: pip install bpy")
