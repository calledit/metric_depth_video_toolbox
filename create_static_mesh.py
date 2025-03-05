import argparse
import cv2
import numpy as np
import os
import json
import sys
import time
import copy

import open3d as o3d
import depth_map_tools
from skimage import measure
from scipy.spatial import cKDTree

np.set_printoptions(suppress=True, precision=4)
    
def remove_small_clusters(mesh, min_triangles=10):
    """
    Removes small, isolated connected components (clusters) from the mesh.
    Clusters with fewer than min_triangles triangles are discarded.
    
    Parameters:
      - mesh: an instance of o3d.geometry.TriangleMesh.
      - min_triangles: minimum number of triangles a cluster must have to be kept.
    
    Returns:
      - mesh: The input mesh with small clusters removed.
    """
    # Cluster connected triangles.
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    # Find clusters that are large enough.
    cluster_counts = np.bincount(triangle_clusters)
    valid_clusters = np.where(cluster_counts >= min_triangles)[0]
    valid_triangle_mask = np.isin(triangle_clusters, valid_clusters)
    
    # Filter triangles.
    triangles = np.asarray(mesh.triangles)
    new_triangles = triangles[valid_triangle_mask]
    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    #mesh.remove_unreferenced_vertices()
    
    return mesh


def remove_visible_triangles_by_image_mask(mesh, image_mask, intrinsics, extrinsics=None, rendered_depth=None, min_cluster_triangles=10, just_return_index=False):
    """
    Removes from the mesh only the closest (visible) triangle per pixel that falls
    in the masked region (i.e. where image_mask == 255). Optionally, if a rendered depth map 
    is provided, the candidate triangle whose centroid depth best matches the rendered depth is chosen.
    After removal, small isolated clusters (with fewer than min_cluster_triangles) are discarded.
    
    Parameters:
      - mesh: An Open3D TriangleMesh.
      - image_mask: A 2D uint8 numpy array (H x W) where 255 indicates the area to remove.
      - intrinsics: A 3x3 camera intrinsic matrix.
      - extrinsics: A 4x4 camera extrinsic matrix (world-to-camera). If None, identity is used.
      - rendered_depth: (Optional) A 2D numpy array (H x W) with depth values from a previous render.
      - min_cluster_triangles: Minimum number of triangles a connected cluster must have to be kept.
      
    Returns:
      - new_mesh: A new TriangleMesh with the selected visible triangles in masked pixels removed and
                  small isolated clusters discarded.
    """
    # Get mesh data.
    vertices = np.asarray(mesh.vertices)    # shape (N, 3)
    triangles = np.asarray(mesh.triangles)    # shape (M, 3)
    H, W = image_mask.shape

    if extrinsics is None:
        extrinsics = np.eye(4)

    # --- Compute triangle centroids and project them ---
    centroids = np.mean(vertices[triangles], axis=1)  # (M, 3)
    centroids_h = np.hstack([centroids, np.ones((centroids.shape[0], 1))])  # (M, 4)

    # Transform centroids into camera coordinates.
    centroids_cam = (extrinsics @ centroids_h.T).T[:, :3]  # (M, 3)
    # Project into image plane.
    proj = (intrinsics @ centroids_cam.T).T  # (M, 3)
    pixel_coords = proj[:, :2] / proj[:, 2:3]  # (M, 2)
    pixel_coords_int = np.round(pixel_coords).astype(int)  # (M, 2)
    # Depth from the camera for each centroid.
    candidate_depths = centroids_cam[:, 2]

    # --- Determine which triangles project inside the image and hit the mask ---
    valid = (pixel_coords_int[:, 0] >= 0) & (pixel_coords_int[:, 0] < W) & \
            (pixel_coords_int[:, 1] >= 0) & (pixel_coords_int[:, 1] < H)
    masked = np.zeros(triangles.shape[0], dtype=bool)
    valid_idx = np.where(valid)[0]
    masked[valid_idx] = image_mask[pixel_coords_int[valid_idx, 1],
                                       pixel_coords_int[valid_idx, 0]]

    # --- Candidate triangles (those in masked pixels) ---
    candidate_idx = np.where(masked)[0]
    if candidate_idx.size == 0:
        # No candidate triangles to remove.
        new_mesh = mesh
    else:
        # Get candidate pixel coordinates and depths.
        candidate_pixels = pixel_coords_int[candidate_idx]  # (n_candidates, 2)
        candidate_depths = candidate_depths[candidate_idx]    # (n_candidates,)
        # Compute a 1D index for grouping: index = v * W + u.
        candidate_pixel_index = candidate_pixels[:, 1] * W + candidate_pixels[:, 0]

        # Compute per-candidate error.
        if rendered_depth is not None:
            # For each candidate, look up the rendered depth at its pixel.
            candidate_rendered_depth = rendered_depth[candidate_pixels[:, 1], candidate_pixels[:, 0]]
            candidate_error = np.abs(candidate_depths - candidate_rendered_depth)
        else:
            candidate_error = candidate_depths

        # --- Group candidates by pixel and select the one with minimum error (vectorized) ---
        sorted_order = np.argsort(candidate_pixel_index)
        sorted_pixels = candidate_pixel_index[sorted_order]
        sorted_error = candidate_error[sorted_order]
        sorted_candidate_idx = candidate_idx[sorted_order]

        # Group candidates by unique pixel index.
        unique_pixels, group_start, group_counts = np.unique(sorted_pixels, return_index=True, return_counts=True)
        # Compute the minimum error per group.
        min_errors = np.minimum.reduceat(sorted_error, group_start)
        # Expand the min error to each candidate within the group.
        group_ids = np.repeat(np.arange(len(group_start)), group_counts)
        min_errors_per_candidate = min_errors[group_ids]
        # Flag candidates whose error equals the group's min error.
        is_min = sorted_error == min_errors_per_candidate
        # Compute offsets within each group.
        group_offsets = np.arange(sorted_error.shape[0]) - np.repeat(group_start, group_counts)
        group_offsets_masked = np.where(is_min, group_offsets, np.inf)
        first_offsets = np.minimum.reduceat(group_offsets_masked, group_start)
        selected_sorted_indices = (group_start + first_offsets).astype(int)
        # Map back to the original triangle indices.
        remove_indices = sorted_candidate_idx[selected_sorted_indices]
        
        #returns a vertex index
        if just_return_index:
            ret_triangle_mask = np.zeros(triangles.shape[0], dtype=bool)
            ret_triangle_mask[remove_indices] = True
            return np.unique(triangles[ret_triangle_mask])

        # --- Rebuild the mesh by removing only the selected triangles ---
        keep_triangle_mask = np.ones(triangles.shape[0], dtype=bool)
        keep_triangle_mask[remove_indices] = False
        new_triangles = triangles[keep_triangle_mask]

        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = mesh.vertices  # vertices remain unchanged
        new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        if len(mesh.vertex_colors) == vertices.shape[0]:
            new_mesh.vertex_colors = mesh.vertex_colors

    # --- Final Step: Remove small isolated clusters ---
    new_mesh = remove_small_clusters(new_mesh, min_triangles=min_cluster_triangles)
    
    return new_mesh

def remove_vertices_by_mask(mesh, vertex_mask):
    """
    Removes vertices from an Open3D TriangleMesh using a boolean vertex mask,
    and returns an index map that maps original vertex indices to the new indices.
    
    Parameters:
      - mesh: an instance of o3d.geometry.TriangleMesh.
      - vertex_mask: A 1D boolean numpy array of length (num_vertices,). 
                     True indicates the vertex should be kept.
    
    Returns:
      - new_mesh: A new TriangleMesh with vertices and triangles updated so that only
                  vertices with True in vertex_mask remain. Triangles referencing any 
                  removed vertex are discarded.
      - index_map: A NumPy array of shape (num_vertices,) where each entry is the new
                   vertex index for kept vertices, or -1 for removed vertices.
    """
    # Convert mesh data to NumPy arrays.
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Optionally, get vertex colors if they exist.
    has_colors = (len(mesh.vertex_colors) == len(mesh.vertices))
    if has_colors:
        colors = np.asarray(mesh.vertex_colors)
    
    # Create a mapping from old vertex indices to new ones.
    index_map = np.full(vertices.shape[0], -1, dtype=int)
    index_map[vertex_mask] = np.arange(np.count_nonzero(vertex_mask))
    
    # Filter the vertices (and colors if they exist).
    new_vertices = vertices[vertex_mask]
    if has_colors:
        new_colors = colors[vertex_mask]
    
    # Filter triangles: keep only triangles where all three vertex indices are kept.
    valid_triangle_mask = vertex_mask[triangles].all(axis=1)
    valid_triangles = triangles[valid_triangle_mask]
    
    # Update the triangle indices using the mapping.
    new_triangles = index_map[valid_triangles]
    
    # Create a new mesh with the filtered vertices, triangles, and optionally colors.
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    if has_colors:
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    
    return new_mesh, index_map

    
def remove_vertices_by_color_exact(mesh, target_color):
    """
    Removes vertices from an Open3D TriangleMesh that exactly match the target color.
    
    Parameters:
      - mesh: an instance of o3d.geometry.TriangleMesh that has vertex_colors.
      - target_color: a 3-element array-like (values in [0,1]) representing the color to remove.
    
    Returns:
      - new_mesh: a new TriangleMesh with vertices (and associated triangles) removed.
    """
    # Convert mesh data to NumPy arrays.
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    
    # Create a boolean mask for vertices to keep.
    # A vertex is kept if its color is not exactly equal to target_color.
    target_color = np.array(target_color)
    keep_mask = ~np.all(colors == target_color, axis=1)
    
    # Create a mapping from old vertex indices to new indices.
    new_indices = np.full(keep_mask.shape, -1, dtype=np.int64)
    new_indices[keep_mask] = np.arange(np.count_nonzero(keep_mask))
    
    # Filter the vertices and their colors.
    new_vertices = vertices[keep_mask]
    new_colors = colors[keep_mask]
    
    # Filter triangles: keep only triangles where all vertices are kept.
    valid_triangles_mask = keep_mask[triangles].all(axis=1)
    valid_triangles = triangles[valid_triangles_mask]
    
    # Update the triangle indices using the mapping.
    new_triangles = new_indices[valid_triangles]
    
    # Create a new mesh with the updated vertices, colors, and triangles.
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    return new_mesh
def find_edge_pixels(mask):
    """
    Find edge pixels in a 2D boolean mask using 4-connected neighbors.
    An edge pixel is defined as a True pixel that borders at least one False pixel.
    
    Parameters:
        mask (np.ndarray): A 2D boolean array.
    
    Returns:
        edge_mask (np.ndarray): A boolean array with True at outer edge positions.
    """
    edge_mask = np.zeros_like(mask, dtype=bool)
    edge_mask[1:, :] |= mask[1:, :] & ~mask[:-1, :]  # Compare with pixel above.
    edge_mask[:-1, :] |= mask[:-1, :] & ~mask[1:, :]  # Compare with pixel below.
    edge_mask[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]    # Compare with pixel to the left.
    edge_mask[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]    # Compare with pixel to the right.
    return edge_mask

def deduplicate_line(line):
    """
    Remove consecutive duplicate coordinates from a list of points.
    
    Parameters:
        line (np.ndarray): An array of (row, col) coordinates.
    
    Returns:
        np.ndarray: Deduplicated array of coordinates.
    """
    if len(line) == 0:
        return line
    dedup = [line[0]]
    for pt in line[1:]:
        if not np.array_equal(pt, dedup[-1]):
            dedup.append(pt)
    return np.array(dedup)

def extract_and_snap_contours(mask):
    """
    Extracts a contour from a binary mask using skimage's find_contours, then snaps
    each subpixel contour coordinate to the nearest outer True pixel in the mask.
    
    This is useful if you want an ordered line that lies on the outer True pixels.
    
    Parameters:
        mask (np.ndarray): A 2D binary (boolean) array.
    
    Returns:
        snapped_lines (list of np.ndarray): A list of ordered arrays of (row, col)
            coordinates representing the snapped outer edge lines.
    """
    # First, get subpixel contours at the interface between 0 and 1.
    contours = measure.find_contours(mask.astype(float), level=0.5)
    
    # Compute the edge mask (outer True pixels) using fast numpy slicing.
    edge_mask = find_edge_pixels(mask)
    edge_coords = np.argwhere(edge_mask)
    
    # Build a KDTree for fast nearest-neighbor lookup.
    tree = cKDTree(edge_coords)
    
    snapped_lines = []
    for contour in contours:
        snapped = []
        for point in contour:
            # Query the nearest edge pixel; point is in (row, col) format.
            _, idx = tree.query(point)
            snapped.append(edge_coords[idx])
        # Convert to numpy array and remove any consecutive duplicates.
        snapped_line = deduplicate_line(np.array(snapped))
        snapped_lines.append(snapped_line)
    
    return snapped_lines, edge_mask
    

def fill_seam(mesh, no_bg_mesh, edge_lines, edge_vertex_indexmesh, max_distance=0.2, max_usage=2, k_candidates=5):
    """
    Fill the seam between the inserted mesh (no_bg_mesh) and the original mesh using only
    candidate vertices specified by edge_vertex_indexmesh (indices into mesh).
    
    Parameters:
      mesh (o3d.geometry.TriangleMesh): The original mesh.
      no_bg_mesh (o3d.geometry.TriangleMesh): The inserted mesh.
      edge_lines (list of np.ndarray): List of arrays of indices into no_bg_mesh.vertices that define the boundary.
      edge_vertex_indexmesh (np.array): Array of vertex indices in 'mesh' that are allowed to be used.
      max_distance (float): Maximum allowed 3D distance from the segment midpoint to a candidate.
      max_usage (int): Maximum allowed usage count per candidate vertex.
      k_candidates (int): Number of nearest candidates to consider per query.
      
    Returns:
      o3d.geometry.TriangleMesh: The new mesh with the seam filled.
    """
    # Number of vertices from the inserted (no_bg_mesh)
    inserted_vertices_len = len(np.asarray(no_bg_mesh.vertices))
    
    print("before:", inserted_vertices_len, len(np.asarray(mesh.vertices)), inserted_vertices_len+ len(np.asarray(mesh.vertices)))
    
    # Merge meshes; note that no_bg_mesh vertices come first, then mesh vertices.
    new_mesh = no_bg_mesh + mesh
    merged_vertices = np.asarray(new_mesh.vertices)
    
    print("after:", len(merged_vertices))
    
    # The candidate vertices are only those specified by edge_vertex_indexmesh in the original mesh.
    # In new_mesh, the original mesh vertices start at index 'inserted_vertices_len',
    # so adjust the candidate indices accordingly.
    candidate_indices_in_merged = edge_vertex_indexmesh + inserted_vertices_len
    candidate_vertices = merged_vertices[candidate_indices_in_merged]
    
    # Build a KDTree only on the candidate vertices.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(candidate_vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    new_triangles = []
    # Create a usage counter for each candidate vertex.
    usage_counts = np.zeros(len(candidate_vertices), dtype=int)
    
    # Process each boundary line.
    for line in edge_lines:
        # Convert to a flat 1D numpy array (these indices refer to vertices in no_bg_mesh).
        line = np.asarray(line).flatten()
        # List to hold the selected candidate (original mesh) vertex for each edge segment.
        candidate_selected = []
        
        # First pass: for each consecutive pair along the no_bg_mesh boundary,
        # compute the midpoint and choose a candidate from the candidate set.
        for i in range(len(line) - 1):
            idx_no_bg1 = int(line[i])
            idx_no_bg2 = int(line[i+1])
            
            v1 = merged_vertices[idx_no_bg1]
            v2 = merged_vertices[idx_no_bg2]
            midpoint = (v1 + v2) / 2.0
            
            # Search for up to k_candidates nearest neighbors from the candidate vertices.
            [k, idxs, dists] = kdtree.search_knn_vector_3d(midpoint, k_candidates)
            chosen_candidate = None
            for j in range(k):
                candidate_relative_index = idxs[j]  # Index in candidate_vertices.
                candidate_distance = np.sqrt(dists[j])
                # Accept candidate if within max_distance and not overused.
                if candidate_distance < max_distance and usage_counts[candidate_relative_index] < max_usage:
                    # Map back to the merged mesh index.
                    chosen_candidate = candidate_indices_in_merged[candidate_relative_index]
                    usage_counts[candidate_relative_index] += 1
                    break
            
            candidate_selected.append(chosen_candidate)
            if chosen_candidate is not None:
                # Create a triangle with the two no_bg_mesh vertices and the candidate.
                new_triangles.append([idx_no_bg1, idx_no_bg2, chosen_candidate])
        
        # Second pass: For every pair of consecutive segments,
        # add extra triangles that use the candidate vertices so that an edge from the original mesh is also used.
        for i in range(len(candidate_selected) - 1):
            cand1 = candidate_selected[i]
            cand2 = candidate_selected[i+1]
            shared_boundary = int(line[i+1])  # The vertex shared by both segments.
            if cand1 is not None and cand2 is not None:
                # Add extra triangles bridging the gap.
                new_triangles.append([shared_boundary, cand1, cand2])
                new_triangles.append([int(line[i]), cand1, cand2])
    
    # Combine the new triangles with any existing triangles in new_mesh.
    if len(new_mesh.triangles) > 0:
        existing_triangles = np.asarray(new_mesh.triangles)
        combined_triangles = np.vstack((existing_triangles, np.array(new_triangles)))
    else:
        combined_triangles = np.array(new_triangles)
    
    new_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles.tolist())
    
    # Recompute normals after modifying the mesh.
    new_mesh.compute_vertex_normals()
    new_mesh.compute_triangle_normals()
    
    return new_mesh

if __name__ == '__main__':
    
    # Setup arguments
    parser = argparse.ArgumentParser(description='Take a rgb encoded depth video and a color video, and view it/render as 3D')
    
    parser.add_argument('--depth_video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=False)
    parser.add_argument('--xfov', type=float, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=float, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--render', action='store_true', help='Render to video insted of GUI', required=False)
    parser.add_argument('--remove_edges', action='store_true', help='Tries to remove edges that was not visible in image(it is a bit slow)', required=False)
    parser.add_argument('--show_camera', action='store_true', help='Shows lines representing the camera frustrum', required=False)
    parser.add_argument('--background_ply', type=str, help='PLY file that will be included in the scene', required=False)
    parser.add_argument('--mask_video', type=str, help='Mask video to filter out back or forground', required=False)
    parser.add_argument('--invert_mask', action='store_true', help='Remove the baground(black) instead of the forground(white)', required=False)
    parser.add_argument('--downscale', default=1, type=int, help='if we are to downscale the image for faster proccesing', required=False)
    
    parser.add_argument('--compressed', action='store_true', help='Render the video in a compressed format. Reduces file size but also quality.', required=False)
    parser.add_argument('--draw_frame', default=-1, type=int, help='open gui with specific frame', required=False)
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--transformation_file', type=str, help='file with scene transformations from the aligner', required=False)
    parser.add_argument('--transformation_lock_frame', default=0, type=int, help='the frame that the transfomrmation will use as a base', required=False)
    
    parser.add_argument('--x', default=2.0, type=float, help='set position of cammera x cordicate in meters', required=False)
    parser.add_argument('--y', default=2.0, type=float, help='set position of cammera y cordicate in meters', required=False)
    parser.add_argument('--z', default=-4.0, type=float, help='set position of cammera z cordicate in meters', required=False)
    parser.add_argument('--tx', default=-99.0, type=float, help='set poistion of camera target x cordinate in meters', required=False)
    parser.add_argument('--ty', default=-99.0, type=float, help='set poistion of camera target y cordinate in meters', required=False)
    parser.add_argument('--tz', default=-99.0, type=float, help='set poistion of camera target z cordinate in meters', required=False)
    
    
    
    
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
    
    background_obj = None
    if args.background_ply is not None:
        background_obj = o3d.io.read_point_cloud(args.background_ply)
        
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
        
    if args.downscale != 1:
        frame_width //= args.downscale
        frame_height //= args.downscale
        
    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)
    fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)
    print("Camera fovx: ", fovx, "fovy:", fovy)

    out = None
    if args.draw_frame == -1:
        if not args.render:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.clear_geometries()
            rend_opt = vis.get_render_option()
            ctr = vis.get_view_control()
            ctr.set_lookat([0, 0, 1])
            ctr.set_up([0, -1, 0])
            ctr.set_front([0, 0, -1])
            ctr.set_zoom(1)
            vis.update_renderer()
            params = ctr.convert_to_pinhole_camera_parameters()
        else:
            
            # avc1 seams to be required for Quest 2 if linux complains use mp4v those video files that wont work on Quest 2
            # Read this to install avc1 codec from source https://swiftlane.com/blog/generating-mp4s-using-opencv-python-with-the-avc1-codec/
            # generally it is better to render without compression then Add compression at a later stage with a better compresser like FFMPEG.
            if args.compressed:
                output_file = args.depth_video + "_render.mp4"
                codec = cv2.VideoWriter_fourcc(*"avc1")
            else:
                output_file = args.depth_video + "_render.mkv"
                codec = cv2.VideoWriter_fourcc(*"FFV1")
            out = cv2.VideoWriter(output_file, codec, frame_rate, (frame_width, frame_height))
    mesh = None
    projected_mesh = None
    
    cameraLines, LastcameraLines = None, None
    frame_n = 0
    last30_max_depth = []
    while raw_video.isOpened():
        
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        frame_n += 1
        ret, raw_frame = raw_video.read()
        if not ret:
            break
            
        
        
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        if args.downscale != 1:
            rgb = cv2.resize(rgb, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        
        color_frame = None
        if color_video is not None:
            ret, color_frame = color_video.read()
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            if args.downscale != 1:
                color_frame = cv2.resize(color_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            
            assert color_frame.shape == rgb.shape, "color image and depth image need to have same width and height" #potential BUG here with mono depth videos
        else:
            color_frame = rgb
            
        mask = None
        if mask_video is not None:
            ret, mask = mask_video.read()
            if ret:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if args.downscale != 1:
                    mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                if args.invert_mask:
                    mask = 255-mask
        
        if args.draw_frame != -1 and args.draw_frame != frame_n:
            continue
            
        if not ((frame_n-1) % 4 == 0):
            continue

        # Decode video depth
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((rgb[..., 0].astype(np.uint32) + rgb[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        
        transform_to_zero = np.eye(4)
        if transformations is not None:
            transform_to_zero = np.array(transformations[frame_n-1])
        
        if args.show_camera:
            last30_max_depth.append(depth.max())
            roll_depth = sum(last30_max_depth)/len(last30_max_depth)
            if len(last30_max_depth) > 30:
                last30_max_depth.pop(0)
            cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=frame_width, view_height_px=frame_height, intrinsic=cam_matrix, extrinsic=np.eye(4), scale=roll_depth)
            cameraLines.transform(transform_to_zero)
        
        invalid_color = np.array([0.0,1.0,0.0])
        
        
        
        green_mesh, _ = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, projected_mesh, remove_edges = True, mask = mask, invalid_color = invalid_color)
        #clean_mesh, _ = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, projected_mesh, remove_edges = True, mask = mask)
        
        if transformations is not None:
            green_mesh.transform(transform_to_zero)
            #clean_mesh.transform(transform_to_zero)
        
        if mesh is None:
            mesh = copy.deepcopy(green_mesh)
            if not args.render and args.draw_frame == -1:
                vis.add_geometry(mesh)
            if background_obj is not None:
                vis.add_geometry(background_obj)
                
        else:
            vis.remove_geometry(mesh, reset_bounding_box = False)
            
            transform_from_zero = np.linalg.inv(transform_to_zero)
            
            
            
            
            
            
            
            #parts of the mesh that are invalid will be invalid_color as well as parts in the view that has need been seen before (ie the background)
            pixels_to_where_vertex_needed, removal_depth_map = depth_map_tools.render([mesh], cam_matrix, depth = -2, extrinsic_matric = transform_from_zero, bg_color = invalid_color)
            
            #print("rendered_shape:", pixels_to_where_vertex_needed.shape, "depth_shape", depth.shape)
            
            
            #Create a mask for all vertextes that are green in pixels_to_replace
            pixels_to_where_vertexes_needed_mask = np.all(pixels_to_where_vertex_needed == invalid_color, axis=-1)
            #model_mask = np.full((frame_height, frame_width), 255, dtype=np.uint8)
            #model_mask[pixels_to_replace_mask] = 0
            
            #print(pixels_to_where_vertexes_needed_mask)
            
            #bg_col = np.array([0,0,255], dtype=np.uint8)
            
            #_, edge_coords = find_edge_pixels(pixels_to_where_vertexes_needed_mask)
            
            #edge_lines = trace_lines_from_edge_coords(edge_coords)
            
            
            
            
            print("rendered_shape:", pixels_to_where_vertex_needed.shape, "depth_shape", depth.shape, "mask:", pixels_to_where_vertexes_needed_mask.shape)
            
            #print(edge_lines)
            
            
            #color_frame[pixels_to_replace_mask] = bg_col
            
            green_mesh_2, _ = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, projected_mesh, remove_edges = True, mask = mask, invalid_color = invalid_color)
            green_mesh_2.transform(transform_to_zero)
            
            
            edge_lines, edge_mask = extract_and_snap_contours(pixels_to_where_vertexes_needed_mask)
            
            
            
            
            
            no_bg_mesh, index_map = remove_vertices_by_mask(green_mesh_2, pixels_to_where_vertexes_needed_mask.reshape(-1))
            
            
            #Test to show that we can draw all edge lines red
            edge_lines_vert = []
            for line in edge_lines:
                vert_indexs = line[:, 0] * pixels_to_where_vertexes_needed_mask.shape[1] + line[:, 1]
                edge_lines_vert.append(np.array(index_map[vert_indexs]))
            
            # Before we add the new mesh we need to remove the stuff it replaces
            
            edge_vertex_index = remove_visible_triangles_by_image_mask(mesh, edge_mask, cam_matrix, extrinsics=transform_from_zero, rendered_depth=removal_depth_map, just_return_index=True)
            
            
            print(edge_vertex_index)
            #edge_vertex_index += len(np.asarray(no_bg_mesh.vertices))
            
            
            
            
            
            mesh = remove_visible_triangles_by_image_mask(mesh, pixels_to_where_vertexes_needed_mask, cam_matrix, extrinsics=transform_from_zero, rendered_depth=removal_depth_map, min_cluster_triangles=100)
            
            
            mesh = fill_seam(mesh, no_bg_mesh, edge_lines_vert, edge_vertex_index)
            
            #mesh += no_bg_mesh 
            
            #mesh = fill_seam_with_fan(mesh, edge_lines, pixels_to_where_vertexes_needed_mask.shape)
            
            redner_of_replacement_mesh = depth_map_tools.render([mesh], cam_matrix, extrinsic_matric = transform_from_zero, bg_color = np.array([0.0,0.0,0.0]))
            
            
            #for line in edge_lines:
            #    redner_of_replacement_mesh[line[:, 0], line[:, 1]] = np.array([1.,0.,0.])
            
            #image = (redner_of_replacement_mesh*255).astype(np.uint8)
            
            #out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            #break
            
            #merge_meshes_with_vertex_mapping(mesh, projected_mesh, vertex_maping)
            vis.add_geometry(mesh, reset_bounding_box = False)
            #print(vertex_maping)
            
        projected_mesh = green_mesh
        #mesh = mesh_ret
        
        #if mesh is None:
        #    mesh = copy.deepcopy(projected_mesh)
        
        
        
        
        to_draw = [mesh]
        if cameraLines is not None:
            to_draw.append(cameraLines)
            
        if background_obj is not None:
            to_draw.append(background_obj)
            
        
        if args.draw_frame == frame_n:
            depth_map_tools.draw(to_draw)
            exit(0)
        
        
        if not args.render and args.draw_frame == -1:
            if cameraLines is not None:
                if LastcameraLines is not None:
                    vis.remove_geometry(LastcameraLines, reset_bounding_box = False)
                vis.add_geometry(cameraLines, reset_bounding_box = False)
                LastcameraLines = cameraLines
            vis.update_geometry(mesh)
        
        
        # Set Camera position
        lookat = mesh.get_center()
        if args.tx != -99.0:
            lookat[0] = args.tx
        if args.ty != -99.0:
            lookat[1] = args.ty
        if args.tz != -99.0:
            lookat[2] = args.tz
            
        cam_pos = np.array([args.x, args.y, args.z]).astype(np.float32)
        ext = depth_map_tools.cam_look_at(cam_pos, lookat)
        
        if not args.render:
            if  args.draw_frame == -1:
                if frame_n <= 1:#We set the camera position the first frame
                    params.extrinsic = ext
                    params.intrinsic.intrinsic_matrix = cam_matrix
                    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        
                start_time = time.time()
                while time.time() - start_time < 1/frame_rate: #should be (1/frame_rate) but we dont rach that speed anyway
                    vis.poll_events()
                    vis.update_renderer()
        else:
            image = (depth_map_tools.render(to_draw, cam_matrix, extrinsic_matric = ext, bg_color = np.array([1.0,1.0,1.0]))*255).astype(np.uint8)
            #out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if args.max_frames < frame_n and args.max_frames != -1:
            break
    
    raw_video.release()
    if args.render:
        out.release()
    
