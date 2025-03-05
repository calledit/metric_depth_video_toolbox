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
from scipy.ndimage import binary_dilation

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


def remove_visible_triangles_by_image_mask(mesh, image_mask, intrinsics, extrinsics=None, rendered_depth=None):
    import numpy as np
    import open3d as o3d

    # Get mesh data.
    vertices = np.asarray(mesh.vertices)    # shape (N, 3)
    triangles = np.asarray(mesh.triangles)    # shape (M, 3)
    H, W = image_mask.shape

    if extrinsics is None:
        extrinsics = np.eye(4)

    # --- Compute triangle centroids and project them ---
    centroids = np.mean(vertices[triangles], axis=1)  # (M, 3)
    centroids_h = np.hstack([centroids, np.ones((centroids.shape[0], 1))])  # (M, 4)
    centroids_cam = (extrinsics @ centroids_h.T).T[:, :3]  # (M, 3)
    proj = (intrinsics @ centroids_cam.T).T  # (M, 3)
    pixel_coords = proj[:, :2] / proj[:, 2:3]  # (M, 2)
    pixel_coords_int = np.round(pixel_coords).astype(int)  # (M, 2)
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
        # No triangles to remove. Return the original mesh and an identity mapping.
        new_mesh = mesh
        removed_boundry = np.array([])
        vertex_mapping = np.arange(vertices.shape[0])
    else:
        candidate_pixels = pixel_coords_int[candidate_idx]  # (n_candidates, 2)
        candidate_depths_candidates = candidate_depths[candidate_idx]    # (n_candidates,)
        candidate_pixel_index = candidate_pixels[:, 1] * W + candidate_pixels[:, 0]

        if rendered_depth is not None:
            candidate_rendered_depth = rendered_depth[candidate_pixels[:, 1], candidate_pixels[:, 0]]
            candidate_error = np.abs(candidate_depths_candidates - candidate_rendered_depth)
        else:
            candidate_error = candidate_depths_candidates
            
        remove_indices = candidate_idx
        removed_triangle_pixels = candidate_pixels
        
        # Determine which vertices are in the removed triangles.
        removed_triangles = triangles[remove_indices]  # shape (R, 3)
        
        removed_map = np.column_stack((
            np.repeat(remove_indices, 3),
            np.repeat(removed_triangle_pixels[:, 0], 3),
            np.repeat(removed_triangle_pixels[:, 1], 3),
            removed_triangles.reshape(-1)
        ))
        
        # This function is assumed to be defined elsewhere:
        verexes_of_removed_triangels = np.unique(removed_triangles)
        boundry_vertexes = get_still_used_vertices(mesh, verexes_of_removed_triangels)
        
        mask = np.isin(removed_map[:, 3], boundry_vertexes)
        removed_boundry = removed_map[mask]
        
        # --- Rebuild the mesh by removing only the selected triangles ---
        keep_triangle_mask = np.ones(triangles.shape[0], dtype=bool)
        keep_triangle_mask[remove_indices] = False
        new_triangles = triangles[keep_triangle_mask]

        # --- Create a mapping from old vertices to new vertices ---
        # Identify which vertices are used in the new triangles.
        used_vertices = np.unique(new_triangles)
        # Build a mapping: for each vertex in the original mesh, assign a new index if used, else -1.
        vertex_mapping = -np.ones(vertices.shape[0], dtype=int)
        for new_idx, old_idx in enumerate(used_vertices):
            vertex_mapping[old_idx] = new_idx

        # Re-index the triangles using the mapping.
        new_triangles_reindexed = np.array([[vertex_mapping[idx] for idx in tri] for tri in new_triangles])
        new_vertices = vertices[used_vertices]
        
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles_reindexed)
        if len(mesh.vertex_colors) == vertices.shape[0]:
            colors = np.asarray(mesh.vertex_colors)[used_vertices]
            new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return new_mesh, removed_boundry, vertex_mapping

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
    Returns a tuple:
      - first element: a boolean mask for True pixels that border a False pixel,
      - second element: a boolean mask for False pixels that border a True pixel.
    
    Parameters:
        mask (np.ndarray): A 2D boolean array.
    
    Returns:
        (true_edge, false_edge): tuple of two boolean arrays.
    """
    true_edge = np.zeros_like(mask, dtype=bool)
    false_edge = np.zeros_like(mask, dtype=bool)
    
    # For True pixels: check if they border a False pixel.
    true_edge[1:, :] |= mask[1:, :] & ~mask[:-1, :]  # Compare with pixel above.
    true_edge[:-1, :] |= mask[:-1, :] & ~mask[1:, :]  # Compare with pixel below.
    true_edge[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]    # Compare with pixel to the left.
    true_edge[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]    # Compare with pixel to the right.
    
    # For False pixels: check if they border a True pixel.
    false_edge[1:, :] |= ~mask[1:, :] & mask[:-1, :]   # Compare with pixel above.
    false_edge[:-1, :] |= ~mask[:-1, :] & mask[1:, :]   # Compare with pixel below.
    false_edge[:, 1:] |= ~mask[:, 1:] & mask[:, :-1]     # Compare with pixel to the left.
    false_edge[:, :-1] |= ~mask[:, :-1] & mask[:, 1:]     # Compare with pixel to the right.
    
    return true_edge, false_edge

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

def find_corresponding_false_pixel(pt, false_edge_mask, tangent):
    """
    For a given true edge pixel pt (row, col) and a tangent vector at that point,
    return an adjacent false edge pixel that lies to the right of the tangent.
    
    The right-hand normal is computed as (tangent[1], -tangent[0]) (when tangent is (dr, dc)).
    Then, among the 8-connected neighbors of pt, the function picks the neighbor
    for which the vector (neighbor - pt) has the highest positive dot product with
    the right-hand normal. If no neighbor lies to the right (i.e. positive dot product),
    None is returned.
    
    Parameters:
        pt (array-like): (row, col) coordinates of a true pixel.
        false_edge_mask (np.ndarray): 2D boolean array indicating false edge pixels.
        tangent (array-like): Tangent vector at pt (derived from the contour order).
    
    Returns:
        np.ndarray or None: (row, col) coordinate of the chosen false edge pixel, or None if not found.
    """
    r, c = int(round(pt[0])), int(round(pt[1]))
    H, W = false_edge_mask.shape
    tangent = np.array(tangent)
    if np.linalg.norm(tangent) == 0:
        return None
    # Compute right-hand normal: (dcol, -drow) for tangent (drow, dcol)
    right_normal = np.array([tangent[1], -tangent[0]])
    right_normal = right_normal / np.linalg.norm(right_normal)
    
    # Define 8-connected neighbor offsets.
    offsets = [(-1, 0), (0, 1), (1, 0), (0, -1),
               (-1, -1), (-1, 1), (1, 1), (1, -1)]
    
    best_dot = -np.inf
    best_neighbor = None
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and false_edge_mask[nr, nc]:
            vec = np.array([dr, dc])
            dot = np.dot(vec, right_normal)
            if dot > best_dot and dot > 0:
                best_dot = dot
                best_neighbor = np.array([nr, nc])
    return best_neighbor

def extract_and_snap_contours(mask):
    """
    Extracts a contour from a binary mask using skimage's find_contours, then snaps
    each subpixel contour coordinate to the nearest outer True pixel in the mask.
    Also, for each snapped true edge pixel, it finds an adjacent false edge pixel
    that lies on the right side of the contour's direction.
    
    Parameters:
        mask (np.ndarray): A 2D binary (boolean) array.
    
    Returns:
        snapped_lines (list of np.ndarray): A list of ordered arrays of (row, col)
            coordinates representing the snapped outer edge lines.
        false_mapping_lines (list of np.ndarray): A list (same length as snapped_lines) of arrays of (row, col)
            coordinates representing the corresponding false edge pixel for each snapped true edge pixel.
        edge_mask (np.ndarray): Boolean mask with True at outer edge positions.
        false_edge_mask (np.ndarray): Boolean mask with True for false edge pixels.
    """
    # Get subpixel contours.
    contours = measure.find_contours(mask.astype(float), level=0.5)
    
    # Compute edge masks.
    edge_mask, false_edge_mask = find_edge_pixels(mask)
    # Get coordinates of true edge pixels (for snapping).
    edge_coords = np.argwhere(edge_mask)
    # Build KDTree for fast nearest-neighbor lookup.
    tree = cKDTree(edge_coords)
    
    snapped_lines = []
    false_mapping_lines = []
    
    for contour in contours:
        snapped = []
        false_map = []
        N = len(contour)
        for i, point in enumerate(contour):
            # Snap the subpixel point to a true edge pixel.
            _, idx = tree.query(point)
            snap_pt = edge_coords[idx]
            snapped.append(snap_pt)
            # Compute the tangent vector using the contour order.
            if i < N - 1:
                tangent = contour[i+1] - point
            elif i > 0:
                tangent = point - contour[i-1]
            else:
                tangent = np.array([0, 0])
            # Find a false edge neighbor on the right side.
            false_pt = find_corresponding_false_pixel(snap_pt, false_edge_mask, tangent)
            false_map.append(false_pt if false_pt is not None else [-1, -1])
        snapped_line = deduplicate_line(np.array(snapped))
        false_line = deduplicate_line(np.array(false_map))
        snapped_lines.append(snapped_line)
        false_mapping_lines.append(false_line)
    
    return snapped_lines, false_mapping_lines, edge_mask, false_edge_mask

def project_to_2d(points):
    """
    Vectorized projection of 3D points (N,3) to 2D using perspective division.
    If a point has z == 0, it is left unmodified.
    """
    # Avoid division by zero:
    z = points[:, 2:3]
    z_safe = np.where(z == 0, 1, z)
    return points[:, :2] / z_safe

def fill_seam(mesh, no_bg_mesh, edge_lines, edge_vertex_indexmesh, new_vert_to_old):

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
            chosen_candidate1 = None
            if idx_no_bg1 in new_vert_to_old:
                chosen_candidate1 = new_vert_to_old[idx_no_bg1][0] + inserted_vertices_len
            chosen_candidate2 = None
            
            if idx_no_bg2 in new_vert_to_old:
                chosen_candidate2 = new_vert_to_old[idx_no_bg2][0] + inserted_vertices_len
            one_or_two = True
            if chosen_candidate1 is not None:
                # Create a triangle with the two no_bg_mesh vertices and the candidate.
                tri = choose_candidate_facing_camera(idx_no_bg1, idx_no_bg2, chosen_candidate1, merged_vertices)
                new_triangles.append(tri)
                #new_triangles.append([idx_no_bg1, idx_no_bg2, chosen_candidate1])
                #new_triangles.append([idx_no_bg2, idx_no_bg1, chosen_candidate1])
            elif chosen_candidate2 is not None:
                # Create a triangle with the two no_bg_mesh vertices and the candidate.
                tri = choose_candidate_facing_camera(idx_no_bg1, idx_no_bg2, chosen_candidate2, merged_vertices)
                new_triangles.append(tri)
                one_or_two = False
            
            candidate_selected.append(chosen_candidate2)
            if chosen_candidate2 is not None and chosen_candidate1 is not None:
                if one_or_two:
                    tri = choose_candidate_facing_camera(chosen_candidate2, chosen_candidate1, idx_no_bg1, merged_vertices)
                    new_triangles.append(tri)
                    #new_triangles.append([chosen_candidate2, chosen_candidate1, idx_no_bg2])
                else:
                    tri = choose_candidate_facing_camera(chosen_candidate2, chosen_candidate1, idx_no_bg2, merged_vertices)
                    new_triangles.append(tri)
                    #new_triangles.append([chosen_candidate1, chosen_candidate2, idx_no_bg1])
        
    # Combine the new triangles with any existing triangles in new_mesh.
    if len(new_mesh.triangles) > 0:
        existing_triangles = np.asarray(new_mesh.triangles)
        combined_triangles = np.vstack((existing_triangles, np.array(new_triangles)))
    else:
        combined_triangles = np.array(new_triangles)
    
    new_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles.tolist())
    
    return new_mesh
    
def choose_candidate_facing_camera(v1_idx, v2_idx, candidate_idx, vertices):
    """
    Given three vertex indices (v1_idx, v2_idx, candidate_idx) and a vertex list,
    returns a tuple of indices for the triangle that is oriented to face the camera
    (assumed to be at (0,0,0)).
    
    The function:
      1. Retrieves the 3D coordinates from the vertex list.
      2. Computes the centroid of the triangle.
      3. Computes the triangle's normal using the cross product of (v2 - v1) and (candidate - v1).
      4. Normalizes the normal.
      5. Computes the vector from the centroid to the camera (i.e. -centroid) and normalizes it.
      6. Checks the dot product between the normal and the vector to the camera.
         If the dot product is positive, the triangle is facing the camera; otherwise, 
         the function swaps v2 and candidate to flip the normal.
    
    Parameters:
      v1_idx, v2_idx, candidate_idx (int): Vertex indices for the triangle.
      vertices (np.ndarray): An array of shape (N, 3) containing the 3D coordinates for each vertex.
      
    Returns:
      tuple: A tuple of three vertex indices representing the triangle, ordered so that the face is front‐facing.
    """
    v1 = vertices[v1_idx]
    v2 = vertices[v2_idx]
    candidate = vertices[candidate_idx]
    
    # Form the triangle and compute its centroid.
    tri = np.array([v1, v2, candidate])
    centroid = np.mean(tri, axis=0)
    
    # Compute the triangle's normal using the given order.
    normal = np.cross(v2 - v1, candidate - v1)
    norm_val = np.linalg.norm(normal)
    if norm_val != 0:
        normal = normal / norm_val
    else:
        # Degenerate triangle; return the original ordering.
        return (v1_idx, v2_idx, candidate_idx)
    
    # Compute the vector from the centroid to the camera (camera at origin).
    to_camera = -centroid
    to_camera_norm = np.linalg.norm(to_camera)
    if to_camera_norm != 0:
        to_camera = to_camera / to_camera_norm
    else:
        # If centroid is at the origin, the orientation is ambiguous.
        return (v1_idx, v2_idx, candidate_idx)
    
    dot = np.dot(normal, to_camera)
    if dot > 0:
        # The triangle is already facing the camera.
        return (v1_idx, v2_idx, candidate_idx)
    else:
        # Swap the order of v2 and candidate to flip the normal.
        return (v1_idx, candidate_idx, v2_idx)
    
def get_still_used_vertices(mesh, vertex_list):
    """
    Given a mesh and a list/array of vertex indices, returns the subset of those
    vertices that are still used in any triangle of the mesh.
    
    Parameters:
      mesh (o3d.geometry.TriangleMesh): The mesh to check.
      vertex_list (array-like): List or array of vertex indices to check.
    
    Returns:
      numpy.ndarray: Array of vertex indices from vertex_list that are still used.
    """
    # Get all triangles from the mesh (each triangle is a triplet of vertex indices).
    triangles = np.asarray(mesh.triangles)
    
    # Get unique vertex indices that appear in any triangle.
    used_vertex_indices = np.unique(triangles)
    
    # Find the intersection between used vertices and the given vertex list.
    still_used = np.intersect1d(vertex_list, used_vertex_indices)
    
    return still_used
    
    

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
    invalid_colors = None
    invalid_indexs = None
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
        
        
        
        green_mesh, invalid_vertex_indices = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, projected_mesh, remove_edges = True, mask = mask, invalid_color = invalid_color)
        
        ref_to_all_col = np.asarray(green_mesh.vertex_colors)
        org_colors = ref_to_all_col.copy()
        ref_to_all_col[invalid_vertex_indices] = np.array(invalid_color)
        #clean_mesh, _ = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, projected_mesh, remove_edges = True, mask = mask)
        
        if transformations is not None:
            green_mesh.transform(transform_to_zero)
            #clean_mesh.transform(transform_to_zero)
        
        if mesh is None:
            mesh = copy.deepcopy(green_mesh)
            invalid_colors = org_colors
            invalid_indexs = invalid_vertex_indices
            if not args.render and args.draw_frame == -1:
                vis.add_geometry(mesh)
            if background_obj is not None:
                vis.add_geometry(background_obj)
                
        else:
            vis.remove_geometry(mesh, reset_bounding_box = False)
            
            transform_from_zero = np.linalg.inv(transform_to_zero)
            
            
            
            
            
            
            
            #parts of the mesh that are invalid will be invalid_color as well as parts in the view that has need been seen before (ie the background)
            
            mesh.transform(transform_to_zero) #Transform to take a picture from the current camera pos
            pixels_to_where_vertex_needed, removal_depth_map = depth_map_tools.render([mesh], cam_matrix, depth = -2, bg_color = invalid_color)
             
            #print("rendered_shape:", pixels_to_where_vertex_needed.shape, "depth_shape", depth.shape)
            
            
            #Create a mask for all vertextes that are green in pixels_to_replace
            pixels_to_where_vertexes_needed_mask = np.all(pixels_to_where_vertex_needed == invalid_color, axis=-1)
            
            
            #Find "Cut" lines (not sure we need this anymore)
            edge_lines, false_mapping_lines, edge_mask, false_edge_mask  = extract_and_snap_contours(pixels_to_where_vertexes_needed_mask)
            
            
            
            #model_mask = np.full((frame_height, frame_width), 255, dtype=np.uint8)
            #model_mask[pixels_to_replace_mask] = 0
            
            #print(pixels_to_where_vertexes_needed_mask)
            
            #bg_col = np.array([0,0,255], dtype=np.uint8)
            
            #_, edge_coords = find_edge_pixels(pixels_to_where_vertexes_needed_mask)
            
            #edge_lines = trace_lines_from_edge_coords(edge_coords)
            
            
            
            
            print("rendered_shape:", pixels_to_where_vertex_needed.shape, "depth_shape", depth.shape, "mask:", pixels_to_where_vertexes_needed_mask.shape)
            
            #print(edge_lines)
            
            
            #color_frame[pixels_to_replace_mask] = bg_col
            
            #Generate mesh from current frame
            #green_mesh_2, _ = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, projected_mesh, remove_edges = True, )
            #green_mesh_2.transform(transform_to_zero)
            
            #Mesh from current frame in zero transform state green_mesh
            green_mesh
            
            
            
            
            #structuring_element = np.ones((3, 3), dtype=bool)

            # Perform binary dilation
            #expanded_edge_mask = binary_dilation(edge_mask, structure=structuring_element)
            
            
            
            #Remove all vertexes that are not in the pixels that we are intressted in
            no_bg_mesh, index_map = remove_vertices_by_mask(green_mesh, pixels_to_where_vertexes_needed_mask.reshape(-1))
            
            #depth_map_tools.draw([no_bg_mesh])
            #exit(0)
            
            
            #Test to show that we can draw all edge lines red
            edge_lines_vert = []
            for line in edge_lines:
                vert_indexs = line[:, 0] * pixels_to_where_vertexes_needed_mask.shape[1] + line[:, 1]
                edge_lines_vert.append(np.array(index_map[vert_indexs]))
            
            # Before we add the new mesh we need to remove the stuff it replaces
            
            #edge_vertex_index = remove_visible_triangles_by_image_mask(mesh, false_edge_mask, cam_matrix, extrinsics=transform_from_zero, rendered_depth=removal_depth_map, just_return_index=True)
            
            
            #print(edge_vertex_index)
            #edge_vertex_index += len(np.asarray(no_bg_mesh.vertices))
            
            #mesh.transform(transform_to_zero)
            #no_bg_mesh.transform(transform_to_zero)
            
            #Remove triangles that will be replaced by the new mesh
            mesh, removed_boundry, index_map = remove_visible_triangles_by_image_mask(mesh, ~pixels_to_where_vertexes_needed_mask, cam_matrix)
            
            mesh.transform(transform_from_zero) #Untranform from current camera pos back in to zero
            
            depth_map_tools.draw([mesh])
            exit(0)
            
            #print(boundary_to_pixels)
            
            #still_used_edge_vertex_list = get_still_used_vertices(mesh, edge_vertex_index)
            #pixels2_boundry_vert = invert_boundary_mapping(boundary_to_pixels, edge_mask)
            #boundry_vertexes = np.unique(removed_boundry[:, 3])
            #new_vert_to_old = {}
            #for row in removed_boundry:
            #    insert_vert_indexs = row[1] * pixels_to_where_vertexes_needed_mask.shape[1] + row[2]
            #    if insert_vert_indexs in index_map:
            #        if insert_vert_indexs not in new_vert_to_old:
            #            new_vert_to_old[index_map[insert_vert_indexs]] = []
            #        new_vert_to_old[index_map[insert_vert_indexs]].append(row[3])
                    
            #for bv, pixels in boundary_to_pixels.items():
            #    for p in pixels:
            #        insert_vert_indexs = p[0] * pixels_to_where_vertexes_needed_mask.shape[1] + p[1]
            #        if insert_vert_indexs in index_map:
            #            if insert_vert_indexs not in new_vert_to_old:
            #                new_vert_to_old[index_map[insert_vert_indexs]] = []
            #            new_vert_to_old[index_map[insert_vert_indexs]].append(bv)
            #            #print(p, "=", bv)
            
            #print(pixels2_boundry_vert)
            
            ofset = 0 #+len(no_bg_mesh.vertices)
            
            
            #mesh = fill_seam(mesh, no_bg_mesh, edge_lines_vert, boundry_vertexes, new_vert_to_old)
            
            #mesh, index_map = remove_vertices_by_mask(mesh, pixels_to_where_vertexes_needed_mask.reshape(-1))
            
            #mesh.transform(transform_from_zero)
            
            #mesh += no_bg_mesh 
            
            #mesh = fill_seam_with_fan(mesh, edge_lines, pixels_to_where_vertexes_needed_mask.shape)
            
            
            #invalid_colors = ref_to_all_col.copy()
            #invalid_indexs = invalid_vertex_indices
            
            new_indices = index_map[invalid_indexs-ofset]

            # Filter out any entries that are -1 (indicating the vertex was removed).
            valid_mask = new_indices >= 0
            new_indices_valid = new_indices[valid_mask]

            # Also, get the corresponding colors for the kept vertices.
            invalid_colors_valid = invalid_colors[invalid_indexs[valid_mask]-ofset]

            # Update the new mesh's vertex colors.
            ref_to_all_col = np.asarray(mesh.vertex_colors)
            ref_to_all_col[new_indices_valid] = invalid_colors_valid
            
            redner_of_replacement_mesh = depth_map_tools.render([mesh], cam_matrix, extrinsic_matric = transform_from_zero, bg_color = np.array([0.0,0.0,0.0]))
            
            
            depth_map_tools.draw([mesh])
            exit(0)
            
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
    
