import numpy as np
import open3d as o3d
import copy
import cv2
from contextlib import contextmanager
import time
import ctypes
from ctypes import wintypes
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw

@contextmanager
def timer(name = 'not named'):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.6f} seconds")

def calculate_normals(depth, K):
    H, W = depth.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # ------------------------------------------------------------------
    # 0) Per-pixel normals (unchanged)
    # ------------------------------------------------------------------
    Zc = depth  # (H, W)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)

    Xc = (u_grid - cx) / fx * Zc
    Yc = (cy - v_grid) / fy * Zc
    Zc_cam = Zc

    P = np.stack([Xc, Yc, Zc_cam], axis=-1)

    P_x1 = np.empty_like(P)
    P_x1[:, :-1, :] = P[:, 1:, :]
    P_x1[:, -1,  :] = P[:, -1, :]

    P_y1 = np.empty_like(P)
    P_y1[:-1, :, :] = P[1:, :, :]
    P_y1[-1,  :, :] = P[-1, :, :]

    v1 = P_x1 - P
    v2 = P_y1 - P

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    normal = normal / norm
    
    # --- DirectX conversion ---
    normal[..., 1] *= -1  # flip Y
    normal[..., 2] *= -1  # flip Z
    return normal

def open_cv_w2c_to_gl_view(transform_to_ref):
    open_cv_w2c = np.linalg.inv(transform_to_ref)   # this is your original w2c again

    # 2) Axis conversion CV -> OpenGL
    A = np.diag([1, -1, -1, 1]).astype(np.float32)

    # world_gl -> camera_gl (still row-major numbers)
    V_gl_row = A @ open_cv_w2c @ A

    # 3) Convert to column-major for OpenGL
    base_pos = V_gl_row
    
    # 4. OpenGL view matrix is inverse of camera pose
    return np.linalg.inv(base_pos)

# ============================================================
# Build frustum planes (6 planes) from K + c2w
# Each plane: (normal n, scalar d) so that n·X + d >= 0 means "inside"
# ============================================================

def frustum_planes(K, c2w, near=0.1, far=100.0):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    W = int(round(2 * cx))
    H = int(round(2 * cy))

    # Directions of rays for 4 corners in camera space
    invK = np.linalg.inv(K)
    corners_px = [(0,0), (W-1,0), (W-1,H-1), (0,H-1)]
    rays = []
    for u,v in corners_px:
        d = invK @ np.array([u,v,1.0],dtype=np.float32)
        rays.append(d / np.linalg.norm(d))
    rays = np.array(rays)

    # Camera center & rotation
    R = c2w[:3,:3]
    t = c2w[:3,3]
    C = t  # camera center in world

    # Convert rays to world coords
    rays_world = (R @ rays.T).T

    # --- Build 6 planes in world space ---
    planes = []

    # Near plane
    n_near = rays_world.mean(axis=0)  # approximate forward
    n_near = n_near / np.linalg.norm(n_near)
    Pn = C + n_near * near
    planes.append(( n_near, -np.dot(n_near, Pn) ))

    # Far plane
    Pf = C + n_near * far
    planes.append(( -n_near,  np.dot(n_near, Pf) ))

    # Side planes (each edge defines a plane through C)
    # For each edge (i -> i+1)
    for i in range(4):
        a = rays_world[i]
        b = rays_world[(i+1)%4]
        n = np.cross(a, b)
        if np.linalg.norm(n) < 1e-9: continue
        n = n / np.linalg.norm(n)
        d = -np.dot(n, C)
        planes.append((n, d))

    return planes  # list of (n, d)


# ============================================================
# SAT / Half-space intersection test for two frusta
# ============================================================

def frusta_intersect(K, c2w1, c2w2, near=0.1, far=10000.0):
    P1 = frustum_planes(K, c2w1, near, far)
    P2 = frustum_planes(K, c2w2, near, far)

    # For convex polyhedra A and B:
    # Check if all vertices of A lie outside any plane of B OR vice versa.
    # Instead of computing vertices, sample along rays (exact for pyramids).
    
    # Get 8 corner rays
    cx, cy = K[0,2], K[1,2]
    W = int(round(2*cx))
    H = int(round(2*cy))
    invK = np.linalg.inv(K)

    def corner_rays():
        pts=[]
        for u,v in [(0,0),(W-1,0),(W-1,H-1),(0,H-1)]:
            d = invK @ np.array([u,v,1.0])
            pts.append(d/np.linalg.norm(d))
        return np.array(pts)

    cr = corner_rays()

    # CAM CENTERS
    C1 = c2w1[:3,3]
    C2 = c2w2[:3,3]
    R1 = c2w1[:3,:3]
    R2 = c2w2[:3,:3]

    # Build the 8 vertices for each frustum exactly
    def frustum_vertices(c2w):
        C = c2w[:3,3]
        R = c2w[:3,:3]
        out=[]
        for z in (near, far):
            for d in cr:
                p_cam = d * z
                out.append(R @ p_cam + C)
        return np.array(out)

    V1 = frustum_vertices(c2w1)
    V2 = frustum_vertices(c2w2)

    # Test each frustum against the other's planes
    def outside_all(vertices, planes):
        # If all vertices lie OUTSIDE a single plane → separated
        for (n, d) in planes:
            if np.all(np.dot(vertices, n) + d < 0):
                return True
        return False

    # If either frustum is fully outside any plane → no intersection
    if outside_all(V1, frustum_planes(K, c2w2, near, far)):
        return False
    if outside_all(V2, frustum_planes(K, c2w1, near, far)):
        return False

    return True

def apply_side_view_to_paralax_mask(parallax_mask, normals, right):
    right_dot = normals[..., 0]
    
    mask_normal_thrsehold_deg = 90.0
    cos_threshold = np.cos(np.deg2rad(mask_normal_thrsehold_deg))
    if right:
        mask_normal = (right_dot > cos_threshold)
    else:
        mask_normal = (right_dot < cos_threshold)
    
    side_view_mask = parallax_mask & mask_normal
    
    return side_view_mask

def rotation_y(angle_rad):
    #print(np.cos)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [ c, 0,  s, 0],
        [ 0, 1,  0, 0],
        [-s, 0,  c, 0],
        [ 0, 0,  0, 1]
    ], dtype=np.float32)


def translation_matrix(x, y, z):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [x, y, z]
    return T

def get_cam_view(side_offset, convergence_angle_rad=0.0, reverse=False):
    eye    = np.array([0, 0, 0], dtype=np.float32)
    target = eye + np.array([0, 0, -1], dtype=np.float32)
    up     = np.array([0, 1, 0], dtype=np.float32)

    base_view = gl_look_at(eye, target, up)

    if not reverse:
        # ----- Forward transform -----
        T = translation_matrix(side_offset, 0, 0)         # world translation
        R = rotation_y(convergence_angle_rad)             # inward rotation
        return R @ T @ base_view

    else:
        # ----- Proper reverse transform -----
        R_inv = rotation_y(-convergence_angle_rad)   # undo rotation
        T_inv = translation_matrix(-side_offset, 0, 0)  # undo translation

        # Reverse the order: T_inv then R_inv
        return T_inv @ R_inv @ base_view

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
    return np.atan((pupillary_distance / 2) / distance)


def mesh_from_depth_and_rgb(
    depth,
    rgb_image,
    K
):

    depth = np.asarray(depth).astype(np.float32)
    rgb_image = np.asarray(rgb_image).astype(np.float32)/255

    H, W = depth.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    Zc = depth  # (H, W)
    
    normal = calculate_normals(Zc, K)

    #right_dot = normal[..., 0]
    #
    #mask_normal_thrsehold_deg = 90.0
    #cos_threshold = np.cos(np.deg2rad(mask_normal_thrsehold_deg))
    #if mask_for_right:
    #    mask_normal = (right_dot > cos_threshold)
    #else:
    #    mask_normal = (right_dot < cos_threshold)

    # ----------------------------------------------------
    # Expand pixel-center normals to per-corner normals
    # Each pixel has 4 vertices → duplicate the normal
    # resulting shape: (H, W, 4, 3)
    # ----------------------------------------------------
    


    c = Zc
    up    = np.pad(Zc, ((1, 0), (0, 0)), mode='edge')[:-1, :]
    down  = np.pad(Zc, ((0, 1), (0, 0)), mode='edge')[1:, :]
    left  = np.pad(Zc, ((0, 0), (1, 0)), mode='edge')[:, :-1]
    right = np.pad(Zc, ((0, 0), (0, 1)), mode='edge')[:, 1:]

    up_left    = np.pad(Zc, ((1, 0), (1, 0)), mode='edge')[:-1, :-1]
    up_right   = np.pad(Zc, ((1, 0), (0, 1)), mode='edge')[:-1, 1:]
    down_left  = np.pad(Zc, ((0, 1), (1, 0)), mode='edge')[1:, :-1]
    down_right = np.pad(Zc, ((0, 1), (0, 1)), mode='edge')[1:, 1:]

    
    
    # ------------------------------------------------------------------
    # 3) Unclamped corners (no threshold), used when apply_clampin_to_mesh=False
    # ------------------------------------------------------------------
    z0_u = mesh_maker_helper_make_corner_unclamped(c, up,   left,  up_left)
    z1_u = mesh_maker_helper_make_corner_unclamped(c, up,   right, up_right)
    z2_u = mesh_maker_helper_make_corner_unclamped(c, down, left,  down_left)
    z3_u = mesh_maker_helper_make_corner_unclamped(c, down, right, down_right)

    #z_corners_unclamped = np.stack([z0_u, z1_u, z2_u, z3_u], axis=-1)  # (H,W,4)


     # ------------------------------------------------------------------
    # NEW: enforce consistency of corner depths across neighboring pixels
    # so that shared logical corners have exactly the same Z value.
    # ------------------------------------------------------------------
    corner_sum  = np.zeros((H + 1, W + 1), dtype=np.float32)
    corner_count = np.zeros((H + 1, W + 1), dtype=np.float32)

    # Each pixel (y, x) contributes its 4 corners:
    #  corner 0 (TL) -> (y,   x)
    #  corner 1 (TR) -> (y,   x+1)
    #  corner 2 (BL) -> (y+1, x)
    #  corner 3 (BR) -> (y+1, x+1)

    # Top-left corners
    corner_sum[0:H,   0:W   ] += z0_u
    corner_count[0:H, 0:W   ] += 1.0

    # Top-right corners
    corner_sum[0:H,   1:W+1 ] += z1_u
    corner_count[0:H, 1:W+1 ] += 1.0

    # Bottom-left corners
    corner_sum[1:H+1, 0:W   ] += z2_u
    corner_count[1:H+1, 0:W ] += 1.0

    # Bottom-right corners
    corner_sum[1:H+1, 1:W+1 ] += z3_u
    corner_count[1:H+1, 1:W+1] += 1.0

    # Avoid divide-by-zero just in case
    corner_count = np.maximum(corner_count, 1.0)
    Z_corners_shared = corner_sum / corner_count  # shape: (H+1, W+1)

    # Rebuild per-pixel 4-corner depths from this *shared* grid
    Z0 = Z_corners_shared[0:H,   0:W   ]  # TL
    Z1 = Z_corners_shared[0:H,   1:W+1 ]  # TR
    Z2 = Z_corners_shared[1:H+1, 0:W   ]  # BL
    Z3 = Z_corners_shared[1:H+1, 1:W+1 ]  # BR

    Z_corners_consistent = np.stack([Z0, Z1, Z2, Z3], axis=-1)  # (H, W, 4)
    
    # FINAL MASK: clamping AND normal-direction
    #if mask_for_right is None:
    #    mask_img = parallax_mask
    #else:
    #    mask_img = parallax_mask & mask_normal

    # ------------------------------------------------------------------
    # 6) Corner coordinates in pixel space
    # ------------------------------------------------------------------
    u_corners = np.arange(W +1, dtype=np.float32)
    v_corners = np.arange(H +1, dtype=np.float32)
    u_corner_grid, v_corner_grid = np.meshgrid(u_corners, v_corners)

    u0 = u_corner_grid[0:H,   0:W]
    u1 = u_corner_grid[0:H,   1:W+1]
    u2 = u_corner_grid[1:H+1, 0:W]
    u3 = u_corner_grid[1:H+1, 1:W+1]

    v0 = v_corner_grid[0:H,   0:W]
    v1 = v_corner_grid[0:H,   1:W+1]
    v2 = v_corner_grid[1:H+1, 0:W]
    v3 = v_corner_grid[1:H+1, 1:W+1]

    u_pix = np.stack([u0, u1, u2, u3], axis=-1)
    v_pix = np.stack([v0, v1, v2, v3], axis=-1)

    # ------------------------------------------------------------------
    # 7) Choose which corner depths to use for geometry
    # ------------------------------------------------------------------
    make_flat = False
    if make_flat:
        # Zc shape is (H, W)
        # We want (H, W, 4) where all four corners use the same depth
        Z = np.repeat(Zc[:, :, None], 4, axis=2)  # flat quads
    else:
        # old behavior
        Z = Z_corners_consistent
    

    Z = Z.astype(np.float32)

    X = (u_pix - cx) / fx * Z
    Y = (cy - v_pix) / fy * Z
    Z = -Z

    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    Z_flat = Z.reshape(-1)
    positions = np.stack([X_flat, Y_flat, Z_flat], axis=-1)

    # ------------------------------------------------------------------
    # 8) Colors (optionally black out masked pixels)
    # ------------------------------------------------------------------
    
    
    
    #rgb = rgb_image.astype(np.float32) / 255.0
    
    # ------------------------------------------------------------------
    # DEBUG TEXTURE: use normals as RGB
    # ------------------------------------------------------------------
    #
    # Map normals from [-1,1] to [0,1] then to [0,255]
    #normal_rgb = (normal * 0.5 + 0.5).clip(0.0, 1.0)
    #rgb = normal_rgb
    rgb_4 = np.repeat(rgb_image.reshape(H * W, 3), 4, axis=0)

    pixel_ids = np.repeat(np.arange(H*W, dtype=np.uint32)+1, 4)
    pixel_ids = pixel_ids.reshape(-1, 1).astype(np.float32)
    
    # Flatten normals the same way as positions
    normal_expanded = np.repeat(normal[:, :, None, :], 4, axis=2)
    normals_flat = normal_expanded.reshape(-1, 3).astype(np.float32)

    vertices = np.concatenate([
        positions,      # (N,3)
        normals_flat,   # (N,3)
        rgb_4,          # (N,3)
        pixel_ids       # (N,1)
    ], axis=-1)

    # ------------------------------------------------------------------
    # 9) Indices
    # ------------------------------------------------------------------
    num_pixels = H * W
    base_indices = (np.arange(num_pixels, dtype=np.uint32) * 4)[:, None]
    local = base_indices + np.array([0, 1, 2, 3], dtype=np.uint32)[None, :]

    v0_i = local[:, 0]
    v1_i = local[:, 1]
    v2_i = local[:, 2]
    v3_i = local[:, 3]

    tris = np.stack([
        np.stack([v0_i, v2_i, v1_i], axis=-1),
        np.stack([v2_i, v3_i, v1_i], axis=-1),
    ], axis=1)

    indices = tris.reshape(-1).astype(np.uint32)

    return (vertices, indices), normal

def mesh_maker_helper_make_corner_unclamped(c, n1, n2, n3):
    # Just average center+neighbors (what you conceptually want as baseline)
    return (c + n1 + n2 + n3) * 0.25

def mesh_maker_helper_make_corner_with_mask(c, n1, n2, n3, thr):
    sum_z = c.copy()
    cnt = np.ones_like(c, dtype=np.float32)
    rejected = np.zeros_like(c, dtype=bool)

    for n in (n1, n2, n3):
        diff = np.abs(n - c)
        accept = diff <= thr
        rejected |= ~accept

        sum_z += np.where(accept, n, 0.0)
        cnt += accept.astype(np.float32)

    return (sum_z / cnt), rejected

def remap_ids_to_img(rgb_image, id_maps, invalid_color=(0,0,0)):
    """
    Used to go back throgh the redering pipline and get the source data
    id_maps = [ids1, ids2, ..., idsN]
    idsN has output resolution
    idsN → ids(N-1) → ... → ids1 → rgb_image
    """

    # Store output shape BEFORE flattening
    final_shape = id_maps[-1].shape

    # Flatten maps
    flat_maps = [m.reshape(-1) for m in id_maps]

    # Start with the last map (idsN)
    current_ids = flat_maps[-1].copy()
    N_final = current_ids.size

    # Prepare validity mask
    valid = np.ones(N_final, dtype=bool)

    # Process from ids(N-1) down to ids1
    # IMPORTANT: skip flat_maps[-1] — do NOT dereference the last map
    for stage in reversed(range(len(id_maps)-1)):
        ids = flat_maps[stage]
        Ns = ids.size

        stage_valid = (current_ids >= 0) & (current_ids < Ns)
        valid &= stage_valid

        # remap only valid entries
        next_ids = np.zeros_like(current_ids)
        idx = valid & stage_valid
        next_ids[idx] = ids[current_ids[idx]]

        current_ids = next_ids

    # Now current_ids indexes directly into the rgb_image
    H0, W0, _ = rgb_image.shape
    N0 = H0 * W0

    final_valid = valid & (current_ids >= 0) & (current_ids < N0)

    out = np.zeros((N_final, 3), dtype=rgb_image.dtype)
    out[:] = invalid_color

    ids0 = current_ids[final_valid]
    ys = ids0 // W0
    xs = ids0 %  W0

    out[final_valid] = rgb_image[ys, xs]

    return out.reshape(final_shape[0], final_shape[1], 3)

def steep_disparity_lr(depth, K, parallax_shift=0.0351, threshold=0.1):
    """
    depth  : (H,W) depth map
    K      : 3x3 intrinsic matrix
    parallax_shift : stereo baseline / shift scale
    threshold : disparity magnitude threshold
    
    Returns:
        left_mask, right_mask
        (each boolean (H,W) array)
    """

    Zc = depth
    fx = float(K[0, 0])

    # Neighbors
    left_Z  = np.pad(Zc, ((0,0),(1,0)), mode='edge')[:, :-1]
    right_Z = np.pad(Zc, ((0,0),(0,1)), mode='edge')[:, 1:]

    # Compute horizontal disparity change
    du_L = fx * parallax_shift * (1.0/Zc - 1.0/left_Z)   # steep toward left neighbor
    du_R = fx * parallax_shift * (1.0/Zc - 1.0/right_Z)  # steep toward right neighbor

    # Two directional steepness masks
    #left_mask  = np.abs(du_L) > threshold
    #right_mask = np.abs(du_R) > threshold
    
    left_mask  = (du_L >  threshold) | (du_R < -threshold)
    right_mask = (du_R >  threshold) | (du_L < -threshold)

    return left_mask, right_mask

def steep_mask_disparity(depth, K, parallax_shift=0.0351, threshold=0.1):
    """
    depth : (H, W) depth map
    K     : 3x3 camera intrinsic matrix
    parallax_shift : baseline / relative camera shift scaling
    threshold : disparity gradient threshold for steepness
    
    Returns:
        Boolean mask where steep / visually-foreshortened pixels are marked True.
    """

    Zc = depth

    # Extract focal length fx (pixel units)
    fx = float(K[0, 0])

    # Build neighbor depth maps
    left  = np.pad(Zc, ((0,0),(1,0)), mode='edge')[:, :-1]
    right = np.pad(Zc, ((0,0),(0,1)), mode='edge')[:, 1:]
    up    = np.pad(Zc, ((1,0),(0,0)), mode='edge')[:-1, :]
    down  = np.pad(Zc, ((0,1),(0,0)), mode='edge')[1:, :]

    # disparity gradient (inverse depth change scaled by fx & baseline)
    du_l = fx * parallax_shift * (1.0/Zc - 1.0/left)
    du_r = fx * parallax_shift * (1.0/Zc - 1.0/right)
    du_u = fx * parallax_shift * (1.0/Zc - 1.0/up)
    du_d = fx * parallax_shift * (1.0/Zc - 1.0/down)

    # steep if any neighbor parallax exceeds threshold
    mask = (
        (np.abs(du_l) > threshold) |
        (np.abs(du_r) > threshold) |
        (np.abs(du_u) > threshold) |
        (np.abs(du_d) > threshold)
    )

    return mask

def generate_normal_bg_image(width, height):
    """
    Create an X-shaped normal background using your four normals.
    The X intersects EXACTLY at the rectangle center, even if W != H.
    Includes diagonal pixels (no gaps).
    """

    W, H = width, height
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Your original normals
    n_left   = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    n_right  = np.array([1.0, 0.5, 0.5], dtype=np.float32)
    n_top    = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    n_bottom = np.array([0.5, 0.5, 1.0], dtype=np.float32)

    # Grid
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    # Center
    cx = W / 2.0
    cy = H / 2.0

    # Diagonals (scaled)
    main_diag = (yy - cy) * W + (xx - cx) * H
    anti_diag = (yy - cy) * W - (xx - cx) * H

    # Region masks WITH diagonals included (<= instead of <)
    mask_top    = (main_diag <= 0) & (anti_diag <= 0)
    mask_right  = (main_diag <= 0) & (anti_diag >= 0)
    mask_bottom = (main_diag >= 0) & (anti_diag >= 0)
    mask_left   = (main_diag >= 0) & (anti_diag <= 0)

    img[mask_top]    = n_top
    img[mask_bottom] = n_bottom
    img[mask_left]   = n_left
    img[mask_right]  = n_right

    return img



glwindow = None
render_shader = None

def gl_render(vertices_and_indices, mvp, width, height, near, far, bg_color = [0.0,0.0,0.0]):
    global glwindow, render_shader
    
    if glwindow is None:
        
        if not glfw.init():
            raise RuntimeError("Could not init GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # Hide the window
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
        
        #Create hidden window to be able to set context
        glwindow = glfw.create_window(1, 1, "", None, None)
        glfw.make_context_current(glwindow)
        glEnable(GL_DEPTH_TEST)
        
        # ---------- SHADERS ----------
        VERT_SHADER = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec3 aColor;
        layout(location = 3) in float aID;

        out vec3 vColor;
        out vec3 vNormal;
        flat out float vID;

        uniform mat4 uMVP;

        void main()
        {
            vColor  = aColor;
            vNormal = aNormal;
            vID     = aID;

            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """

        FRAG_SHADER = """
        #version 330 core

        in vec3 vColor;
        in vec3 vNormal;
        flat in float vID;

        layout(location = 0) out vec4 FragColor;
        layout(location = 1) out vec3 FragNormal;
        layout(location = 2) out int FragID;

        void main()
        {
            FragColor  = vec4(vColor, 1.0);
            FragNormal = vNormal;//normalize(vNormal);
            FragID     = int(vID);
        }

        """

        render_shader = compileProgram(
            compileShader(VERT_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAG_SHADER, GL_FRAGMENT_SHADER)
        )
    
    # ---------- CREATE FBO ----------
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # --- RGB ---
    tex_color = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_color)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8,
                 width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex_color, 0)

    # --- NORMALS ---
    tex_normals = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_normals)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F,
                 width, height, 0, GL_RGB, GL_FLOAT, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                           GL_TEXTURE_2D, tex_normals, 0)

    # --- IDS ---
    tex_ids = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_ids)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I,
                 width, height, 0, GL_RED_INTEGER, GL_INT, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,
                           GL_TEXTURE_2D, tex_ids, 0)

    # --- DEPTH ---
    tex_depth = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_depth)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
                 width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, tex_depth, 0)

    # ---------- draw buffers ----------
    buffers = (OpenGL.raw.GL.VERSION.GL_1_0.GLenum * 3)(
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2
    )
    glDrawBuffers(3, buffers)

    vertices, indices = vertices_and_indices

    # ---------- VAO ----------
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # NEW: 10 floats per vertex (3 pos, 3 normal, 3 color, 1 id)
    stride = 10 * 4

    # aPos (location=0)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

    # aNormal (location=1) → offset 12 bytes
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

    # aColor (location=2) → offset 24 bytes
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

    # aID (location=3) → offset 36 bytes
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))


    # ---------- DRAW ----------
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, width, height)
    glClearColor(bg_color[0], bg_color[1], bg_color[2], 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearBufferiv(GL_COLOR, 2, np.array([0], dtype=np.int32)) # set ids to zero

    glUseProgram(render_shader)
    loc_mvp = glGetUniformLocation(render_shader, "uMVP")
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.T)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # ---------- READ RGB ----------
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    rgb = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)[::-1]

    # ---------- READ DEPTH (IMPORTANT FIX) ----------
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)  # <-- REQUIRED for NVIDIA
    glReadBuffer(GL_NONE)

    buf = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_raw = np.frombuffer(buf, dtype=np.float32).reshape(height, width)[::-1]

    # ---------- LINEARIZE ----------
    z_ndc = depth_raw * 2.0 - 1.0
    linear_depth = (2.0 * near * far) / (far + near - z_ndc * (far - near))

    # ---------- READ NORMALS ----------
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
    glReadBuffer(GL_COLOR_ATTACHMENT1)
    buf = glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT)
    normals = np.frombuffer(buf, dtype=np.float32).reshape(height, width, 3)[::-1]
    
    normals = (normals * 0.5 + 0.5).clip(0.0, 1.0)

    # ---------- READ ID ----------
    glReadBuffer(GL_COLOR_ATTACHMENT2)
    buf = glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_INT)
    ids = np.frombuffer(buf, dtype=np.int32).reshape(height, width)[::-1]
    
    # ---------- CLEANUP ----------
    glDeleteFramebuffers(1, [fbo])
    glDeleteTextures(1, [tex_color])
    glDeleteTextures(1, [tex_normals])
    glDeleteTextures(1, [tex_ids])
    glDeleteTextures(1, [tex_depth])

    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [ebo])
    glDeleteVertexArrays(1, [vao])
    
    return rgb, linear_depth, normals, ids

def open_gl_projection_from_camera_matrix(K, near, far):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Convert intrinsics to OpenGL-style NDC projection
    # Note: OpenGL NDC x,y range is [-1, 1]
    #
    # x_ndc =  (x * 2 / width)  - 1
    # y_ndc =  1 - (y * 2 / height)
    
    width = cx * 2
    height = cy * 2
    

    A11 =  2 * fx / width
    A22 =  2 * fy / height

    A13 =  2 * (cx / width) - 1
    A23 =  1 - 2 * (cy / height)

    # Depth terms same as your perspective() version
    A33 = -(far + near) / (far - near)
    A34 = -(2 * far * near) / (far - near)

    proj = np.array([
        [A11, 0,   A13,  0],
        [0,   A22, A23,  0],
        [0,   0,   A33,  A34],
        [0,   0,  -1,    0]
    ], dtype=np.float32)

    return proj

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


def svd(source_points, target_points, ZeroCentroid = False):
    # Compute the centroid of each set of points
    if ZeroCentroid: #If we only care about rotation. ie the camera is locked in place
        z = np.array([0.0,0.0,0.0])
        centroid_source = z
        centroid_target = z
    else:
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)

    #print(source_points, target_points)
    # Center the points around the centroid
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Compute the covariance matrix
    H = np.dot(source_centered.T, target_centered)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    Rot = np.dot(Vt.T, U.T)

    #Special reflection case handling
    if np.linalg.det(Rot) < 0:
        Vt[2, :] *= -1
        Rot = np.dot(Vt.T, U.T)

    


    # Form the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = Rot
    # Compute the translation vector
    transformation_matrix[:3, 3] = centroid_target - np.dot(Rot, centroid_source)#original function

    return transformation_matrix

def transform_points(points, transform):
    """
    Transform a set of 3D points using a 4x4 homogeneous transform.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 3)
        Input 3D points.
    transform : numpy.ndarray, shape (4, 4)
        4x4 homogeneous transformation matrix.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Transformed 3D points.
    """
    # 1. Convert Nx3 points to Nx4 homogeneous coordinates by appending a column of 1s.
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_hom = np.hstack([points, ones])  # Now shape is (N, 4)

    # 2. Multiply by the 4x4 transformation matrix
    # Note: We use transform.T for correct multiplication with row vectors
    transformed_hom = points_hom @ transform.T  # Still (N, 4)

    # 3. Convert back to Nx3 by dropping the last column (the 'w' component)
    transformed_points = transformed_hom[:, :3]

    return transformed_points

def pnpSolve_ransac(t3d_points_new_frame, mkpts2, cam_mat, distCoeffs = None, refine = False):
    """
        returns a transformation matrix
    """
    if distCoeffs is None:
        distCoeffs = np.array([0, 0, 0, 0], dtype=np.float64)  # distortion coefficients

    #mkpts2 = cv.undistortPoints(mkpts2.reshape(-1, 1, 2), cam_mat, distCoeffs).squeeze()
    #mkpts2 = np.dot(mkpts2, cam_mat[:2, :2].T) + cam_mat[:2, 2 ]
    #if you set the reprojectionError to low the algorithm goes to shit
    reperr = 6
    if refine:
        reperr = 6
    success, rvec, tvec, inliers = cv2.solvePnPRansac(t3d_points_new_frame, np.array(mkpts2,dtype=np.float64), cam_mat, distCoeffs, reprojectionError=reperr, iterationsCount=100000)
    matrix = np.eye(4)
    if success:
        tv = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
        mat, jac = cv2.Rodrigues(rvec)

        if refine:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20000, 1e-32) 
            rvec2, tvec2 = cv2.solvePnPRefineVVS(np.array(t3d_points_new_frame[inliers],dtype=np.float64), np.array(mkpts2[inliers],dtype=np.float64), cam_mat, distCoeffs, rvec, tvec)
            tv = np.array([tvec2[0][0], tvec2[1][0], tvec2[2][0]])
            mat, jac = cv2.Rodrigues(rvec2)

        matrix[:3, :3] = mat
        matrix[:3, 3] = tv
        return matrix
    print("solvePnP FAIL")
    return None

def reject_outliers(data, m=1):
    return abs(data - np.mean(data)) < m * np.std(data)

def pts_2_pcd(points, colors = None, ids = None, normals = None):
    if ids is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(points, dtype=o3d.core.float32)
        if colors is not None:
            pcd.point["colors"] = o3d.core.Tensor((np.array(colors) * 255).astype(np.uint8), dtype=o3d.core.uint8) # o3d.core.Tensor(colors, dtype=o3d.core.float32)
        if ids is not None:
            pcd.point["ids"] = o3d.core.Tensor(np.array(ids).reshape(-1, 1), dtype=o3d.core.int32)
    return pcd

def project_3d_points_to_2d(t3d_points, cam_mat, distCoeffs = np.array([0,0,0,0])):
    mkpts, jacobian = cv2.projectPoints(t3d_points.reshape(1, -1, 3), np.array([[[0., 0., 0.]]]), np.array([[[0., 0., 0.]]]), cam_mat.astype(np.float32), distCoeffs.astype(np.float32))
    mkpts = mkpts.squeeze()
    return mkpts

def project_2d_points_to_3d(points, depth, camera_matrix, distCoeffs = None):
    
    xx = points[:,0]
    yy = points[:,1]
    z = depth[points[:,1].astype(np.int32), points[:,0].astype(np.int32)]
    
    if distCoeffs is None:
        distCoeffs = np.array([0, 0, 0, 0], dtype=np.float64)  # distortion coefficients

    # Step 1: Prepare 2D points in the format (N, 1, 2) for OpenCV
    points_2d = np.array([[[x, y]] for x, y in zip(xx, yy)], dtype=np.float64)

    # Step 2: Undistort the 2D points using distCoeffs
    undistorted_points = cv2.undistortPoints(points_2d, camera_matrix, distCoeffs)

    u = undistorted_points[:, 0, 0]
    v = undistorted_points[:, 0, 1]

    # Use numpy to perform element-wise multiplication and stacking
    points_3d = np.column_stack((u * z, v * z, z))

    # Convert the result to a numpy array for easier use
    return np.array(points_3d)

def convert_mesh_to_pcd(mesh, points_to_remove, input_pcd):
    ref_to_mesh_vert = np.asarray(mesh.vertices)
    ref_to_mesh_cols = np.asarray(mesh.vertex_colors)
    
    #We simply move non visble vertexs away since removing them is so slow in open3d
    ref_to_mesh_vert[points_to_remove] = np.array([-0.2, -0.2, -0.2])
    
    if input_pcd is None:
        input_pcd = pts_2_pcd(ref_to_mesh_vert, ref_to_mesh_cols)
    else:
        
        ref_to_pcd_vert = np.asarray(input_pcd.points)
        ref_to_pcd_cols = np.asarray(input_pcd.colors)
        ref_to_pcd_vert[:] = ref_to_mesh_vert[:]
        ref_to_pcd_cols[:] = ref_to_mesh_cols[:]
    
    return input_pcd

def get_mesh_from_depth_map(depth_map, cam_mat, color_frame = None, inp_mesh = None, remove_edges = False, mask = None,
                                 invalid_color=None, of_by_one = True, return_normals_of_removed = False):
    points, height, width = create_point_cloud_from_depth(depth_map, cam_mat, of_by_one)

    # Create mesh from point cloud
    ret = create_mesh_from_point_cloud(points, height, width, color_frame, inp_mesh, remove_edges, mask = mask, invalid_color = invalid_color, return_normals_of_removed = return_normals_of_removed)
    return ret

def create_point_cloud_from_depth(depth_image, intrinsics, of_by_one = False):
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Here we fix a of by one error caused by the fact that this function fills in the area betwen each vertex
    if of_by_one:
        # should probably solve in a better way. Should probably just use openGL or whatever.
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x *= (width+1)/width
        y *= (height+1)/height



    z = depth_image 
    x3d = (x - intrinsics[0][2]) * z / intrinsics[0][0]  # (x - cx) * z / fx
    y3d = (y - intrinsics[1][2]) * z / intrinsics[1][1]  # (y - cy) * z / fy

    points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)


    return points, height, width


def perspective_aware_down_sample(pcd, voxel_size_norm):
    """
    Downsamples a point cloud in a perspective-aware manner.
    
    The function assumes that the input points are in camera coordinates,
    for example as produced by create_point_cloud_from_depth:
    
        x = (u - cx)*z/fx,   y = (v - cy)*z/fy,   z = depth.
    
    It transforms the points into a warped space where lateral coordinates
    do not depend on depth (i.e. x_norm = x/z, y_norm = y/z), performs voxel
    downsampling in that space, and then transforms the points back.
    
    Args:
        points (np.ndarray): Nx3 array of points in camera space.
        voxel_size_norm (float): Voxel size to use in the warped (normalized)
                                 space. Its units are in “normalized” coordinates.
                                 (For example, if you wish to merge points within
                                 0.005 units in normalized space, set voxel_size_norm=0.005.)
    
    Returns:
        np.ndarray: Downsampled Nx3 array of points in camera space.
    """
    
    points = np.asarray(pcd.points)
    # --- Warp points to remove perspective scaling ---
    # (x, y, z) -> (x/z, y/z, z)
    # (Note: since x = (u-cx)*z/fx, x/z = (u-cx)/fx, and similarly for y.)
    x_norm = points[:, 0] / points[:, 2]
    y_norm = points[:, 1] / points[:, 2]
    warped = np.stack([x_norm, y_norm, points[:, 2]], axis=1)
    
    # --- Create a temporary Open3D point cloud in warped space and downsample ---
    pcd.points = o3d.utility.Vector3dVector(warped)
    pcd_down_warped = pcd.voxel_down_sample(voxel_size_norm)
    warped_down = np.asarray(pcd_down_warped.points)
    
    # --- Unwarp: transform back to original camera coordinates ---
    # For each point, x = (x/z)*z, y = (y/z)*z, and z remains the same.
    x_down = warped_down[:, 0] * warped_down[:, 2]
    y_down = warped_down[:, 1] * warped_down[:, 2]
    z_down = warped_down[:, 2]
    points_down = np.stack([x_down, y_down, z_down], axis=1)
    
    pcd_down_warped.points = o3d.utility.Vector3dVector(points_down)
    
    return pcd_down_warped


zero_identity_matrix = np.identity(4)
def create_mesh_from_point_cloud(points, height, width,
                                 image_frame=None,
                                 inp_mesh=None,
                                 remove_edges=False,
                                 mask=None,
                                 angle_threshold_deg=89.0,
                                 invalid_color=None,
                                 background_edge_mask_expandansions=0,
                                 return_normals_of_removed = False):
    """
    Creates an Open3D TriangleMesh from a grid-organized point cloud while
    filtering out triangles whose orientation relative to the camera is too oblique.
    
    The function assumes the 3D points are in camera coordinates (i.e. the camera is at the origin).
    
    Parameters:
      - points: A numpy array that can be reshaped to (-1, 3) containing the 3D points.
      - height: The number of rows in the grid.
      - width: The number of columns in the grid.
      - image_frame: (Optional) An image whose colors will be mapped to the mesh vertices.
      - inp_mesh: (Optional) An existing mesh to update.
      - remove_edges: If True, triangles with normals that deviate too far from the view vector are removed.
      - mask: (Optional) A mask image; when provided, it is used to filter out cells.
      - angle_threshold_deg: The maximum allowed angle (in degrees) between a triangle’s normal and 
                             the view vector. Triangles with an angle larger than this threshold are discarded.
      - invalid_color: (Optional) If provided (as a 3-element color in [0,1]), triangles failing the
                       edge/mask tests are not removed. Instead, the vertices belonging to these
                       triangles are colored with this value.
    
    Returns:
      - mesh: The resulting Open3D TriangleMesh.
      - used_indices: The indices of vertices that are used in valid triangles. In the case when
                      invalid_color is provided, all vertices are considered used.
    """
    # Reshape points into a (N, 3) array of vertices.
    vertices = points.reshape(-1, 3)
    
    used_indices = []
    
    # Optionally, get vertex colors.
    colors = None
    if image_frame is not None:
        colors = np.array(image_frame).reshape(-1, 3) / 255.0

    # If no mesh exists or if we need to remove edges or apply the mask, compute the triangles.
    if inp_mesh is None or remove_edges or mask is not None:
        if inp_mesh is None:
            mesh = o3d.geometry.TriangleMesh()
        else:
            mesh = inp_mesh
            mesh.transform(zero_identity_matrix)
        
        # --- Generate candidate triangles via the grid layout ---
        # For each grid cell at (i, j) with i in [0, height-2] and j in [0, width-2],
        # we define two triangles:
        #    tri1: (i, j), (i+1, j), (i+1, j+1)
        #    tri2: (i, j), (i+1, j+1), (i, j+1)
        grid_i, grid_j = np.meshgrid(np.arange(height - 1), np.arange(width - 1), indexing='ij')
        grid_i = grid_i.ravel()
        grid_j = grid_j.ravel()
    
        idx1 = grid_i * width + grid_j
        idx2 = (grid_i + 1) * width + grid_j
        idx3 = (grid_i + 1) * width + (grid_j + 1)
        idx4 = grid_i * width + (grid_j + 1)
    
        tri1 = np.stack([idx1, idx2, idx3], axis=1)
        tri2 = np.stack([idx1, idx3, idx4], axis=1)
        triangles_all = np.vstack([tri1, tri2])
        
        if inp_mesh is None:
            mesh.triangles = o3d.utility.Vector3iVector(triangles_all)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            used_indices = np.arange(vertices.shape[0])
        else:
            ref_to_all_tri = np.asarray(mesh.triangles)
            ref_to_all_vert = np.asarray(mesh.vertices)
            ref_to_all_tri[:] = triangles_all[:]
            ref_to_all_vert[:] = vertices[:]
            if colors is not None:
                ref_to_all_col = np.asarray(mesh.vertex_colors)
                ref_to_all_col[:] = colors[:]
        
        # Get references to underlying arrays.
        ref_to_all_tri = np.asarray(mesh.triangles)
        ref_to_all_vert = np.asarray(mesh.vertices)
        if colors is not None:
            ref_to_all_col = np.asarray(mesh.vertex_colors)
        else:
            ref_to_all_col = None

        # --- Filter triangles based on the triangle angle relative to the camera ---
        if remove_edges or mask is not None:
            invalid_mask = None
            if remove_edges:
                v1 = vertices[triangles_all[:, 0]]
                v2 = vertices[triangles_all[:, 1]]
                v3 = vertices[triangles_all[:, 2]]
                cos_threshold = np.cos(np.radians(angle_threshold_deg))
            
                normals = np.cross(v2 - v1, v3 - v1)            # shape (N, 3)
                view    = - (v1 + v2 + v3) / 3.0                  # centers of triangles
                dot     = np.einsum('ij,ij->i', normals, view)
                len_n   = np.sqrt(np.einsum('ij,ij->i', normals, normals))
                len_v   = np.sqrt(np.einsum('ij,ij->i', view, view))
                cosines = dot / (len_n * len_v + 1e-15)
                invalid_mask = (cosines < cos_threshold)
                
                # Here we remove extra triangles on the lower side of each edge. Should in theory
                # help with colors from the forground leaching in to the background. (an issue that is visible as
                # a halo around forground objects when using ML infill). In practise it causes to much glitching so is disabled.
                for it in range(background_edge_mask_expandansions):
                    # For the invalid triangles, extract the depth (z-coordinate) of each vertex.
                    depth_v1 = v1[invalid_mask][:, 2]
                    depth_v2 = v2[invalid_mask][:, 2]
                    depth_v3 = v3[invalid_mask][:, 2]

                    # Stack the depths so that each row corresponds to one invalid triangle.
                    depths = np.stack([depth_v1, depth_v2, depth_v3], axis=1)

                    # Determine which vertex (0, 1, or 2) in each invalid triangle has the maximum depth.
                    furthest_vertex_per_triangle = np.argmax(depths, axis=1)

                    # Map back to the global vertex indices using triangles_all.
                    invalid_indices = np.nonzero(invalid_mask)[0]  # indices into triangles_all for invalid triangles
                    furthest_vertex_indices = triangles_all[invalid_indices, furthest_vertex_per_triangle]

                    # --- New Step: Mark any triangle that uses any of these furthest vertices as invalid ---
                    # Create a set of unique furthest vertex indices.
                    furthest_vertices_set = np.unique(furthest_vertex_indices)

                    # Mark all triangles that use any vertex in furthest_vertices_set.
                    additional_invalid_mask = np.isin(triangles_all, furthest_vertices_set).any(axis=1)

                    # Optionally, combine the original invalid_mask with the additional mask.
                    invalid_mask = invalid_mask | additional_invalid_mask
                
            
            if mask is not None:
                mask = mask > 128
                cell_mask = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
                cell_mask_flat = cell_mask.ravel()
                triangle_mask = np.concatenate([cell_mask_flat, cell_mask_flat])
                if invalid_mask is not None:
                    invalid_mask = invalid_mask | (~triangle_mask)
                else:
                    invalid_mask = ~triangle_mask

            if invalid_color is None:
                
                
                invalid_vertexes = triangles_all[invalid_mask].ravel()
                
                num_vertices = ref_to_all_vert.shape[0]
                is_un_used = np.zeros(num_vertices, dtype=bool)
                is_un_used[invalid_vertexes] = True
                un_used_indices = np.where(is_un_used)[0]
                
                area2 = np.linalg.norm(normals, axis=1)
                
                triangle_normals = np.divide(
                    normals, 
                    area2[:, None],
                    out=np.ones_like(normals), ##some normals are invalid if depth is zero so then we just set those normals to one
                    where=area2[:, None] > 0
                )
                
                # assume triangles_all is (n_triangles, 3), triangle_normals is (n_triangles, 3)
                n_vertices = vertices.shape[0]

                # 1. Flatten the triangle‐to‐vertex index list:
                flat_vids = triangles_all.reshape(-1)            # shape (n_triangles * 3,)

                # 2. Repeat each triangle normal 3× so it lines up with flat_vids:
                repeated_normals = np.repeat(triangle_normals, 3, axis=0)  # shape (n_triangles * 3, 3)

                # 3. Create the output array and assign:
                normals_of_vertexes = np.zeros((n_vertices, 3), dtype=triangle_normals.dtype)
                normals_of_vertexes[flat_vids] = repeated_normals
                
                normals_of_removed_vertexes = normals_of_vertexes[un_used_indices]
                
                    
                # Old behavior: remove invalid triangles by setting their indices to [0,0,0]
                ref_to_all_tri[invalid_mask] = np.array([0, 0, 0])
                
                
                if return_normals_of_removed:
                    return mesh, un_used_indices, normals_of_removed_vertexes
                
                # Compute used indices from the valid triangles only.
                valid_mask = np.logical_not(invalid_mask)
                num_vertices = ref_to_all_vert.shape[0]
                is_used = np.zeros(num_vertices, dtype=bool)
                valid_vertexes = ref_to_all_tri[valid_mask].ravel()
                is_used[valid_vertexes] = True
                used_indices = np.where(is_used)[0]
                
            else:
                # New behavior: keep all triangles, but return them vertices of invalid triangles.
                # Ensure a vertex_colors array exists.
                if ref_to_all_col is None:
                    # Initialize vertex colors to white if not provided.
                    ref_to_all_col = np.ones((ref_to_all_vert.shape[0], 3))
                    mesh.vertex_colors = o3d.utility.Vector3dVector(ref_to_all_col)
                # Get indices of vertices in invalid triangles.
                invalid_triangles = ref_to_all_tri[invalid_mask]
                invalid_vertex_indices = np.unique(invalid_triangles)
                if return_normals_of_removed:
                    return mesh, invalid_vertex_indices, []
                return mesh, invalid_vertex_indices
    else:
        # If we already have an input mesh and we are not removing edges, simply update vertices.
        mesh = inp_mesh
        ref_to_all_vert = np.asarray(mesh.vertices)
        ref_to_all_vert[:] = vertices[:]
        if colors is not None:
            ref_to_all_col = np.asarray(mesh.vertex_colors)
            ref_to_all_col[:] = colors[:]
        used_indices = np.arange(vertices.shape[0])
    if return_normals_of_removed:
        num_vertices = ref_to_all_vert.shape[0]
        is_un_used = np.ones(num_vertices, dtype=bool)
        is_un_used[used_indices] = False
        un_used_indices = np.where(is_un_used)[0]
        return mesh, un_used_indices, []
    return mesh, used_indices


vis = None
v_h = None
use_ofscreen = True
v_w = None
rend = None
def render(objects, cam_mat, depth = False, w = None, h = None, extrinsic_matric = np.eye(4), bg_color = np.array([0, 0, 0])):
    global vis, v_h, v_w, use_ofscreen, rend

    if w is None:
        w = cam_mat[0][2]*2
        h = cam_mat[1][2]*2
    if v_h != h or v_w != w:
        if vis is not None:
            #print(dir(vis))
            #print(vis)
            vis.destroy_window()
            vis.close()
            vis = None
        rend = None
        v_h = h
        v_w = w


    #We set use_ofscreen to False to disable OffscreenRenderer cause it is bugged and is missing required API's
    use_ofscreen = False
    if use_ofscreen:
        if rend is None:
            try:
                rend = o3d.visualization.rendering.OffscreenRenderer(int(w), int(h))
            except:
                use_ofscreen = False



    if rend is None:

        if vis is None:
            vis = o3d.visualization.Visualizer()
            wind_name = "MDVT_render"
            vis.create_window(window_name=wind_name, width=int(w), height=int(h), visible=False) #works for me with False, on some systems needs to be true
            
            
            
            #Insane shit to fix bug in open3d where it wont set the size of the window properly
            user32 = ctypes.windll.user32
            FindWindowW = user32.FindWindowW
            FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
            FindWindowW.restype = wintypes.HWND
            
            MoveWindow = user32.MoveWindow
            MoveWindow.argtypes = [wintypes.HWND, wintypes.INT, wintypes.INT,
                                   wintypes.INT, wintypes.INT, wintypes.BOOL]
            MoveWindow.restype = wintypes.BOOL

            # Try the default Open3D title:
            hwnd = FindWindowW(None, wind_name)
            
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))

            x = 0
            y = 0
            window_width_with_borders = int(w)
            window_height_with_borders = int(h)
            
            desired_w = int(w)
            desired_h = int(h)
            
            print("desired size:", desired_w, desired_h)
            render_height, render_width, _ = np.asarray(vis.capture_screen_float_buffer(do_render=False)).shape
            rounds = 0
            while desired_w != render_width or desired_h != render_height:
                
                MoveWindow(hwnd, x, y, int(window_width_with_borders), int(window_height_with_borders), True)
                print("set size:", window_width_with_borders, window_height_with_borders)
                vis.poll_events()
                render_height, render_width, _ = np.asarray(vis.capture_screen_float_buffer(do_render=False)).shape
                print("repported size:", render_width, render_height)
                window_width_with_borders += window_width_with_borders - render_width
                window_height_with_borders += window_height_with_borders - render_height
                rounds += 1
                if rounds > 2:
                    vis.destroy_window()
                    vis.close()
                    vis = None
                    raise ValueError(f"open3d cant render the desired resolution: {w}x{h}")
            
            
        vis.clear_geometries()

        rend_opt = vis.get_render_option()
        #print("view_status:", vis.get_view_status())
        rend_opt.background_color = bg_color
        rend_opt.point_size = 1.0
        #rend_opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
        #rend_opt.point_show_normal = True
        
        ctr = vis.get_view_control()

        ctr.set_lookat([0, 0, 1])
        ctr.set_up([0, -1, 0])
        ctr.set_front([0, 0, -1])
        ctr.set_zoom(1)
        ctr.set_constant_z_near(0.0001) #you cant set near to 0 with open 3d this close to zero this seams good enogh


        params = ctr.convert_to_pinhole_camera_parameters()

        #print("pos", params.extrinsic, params.intrinsic)
        params.extrinsic = extrinsic_matric
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        #There is a bug in open3d where focaly is not used
        #https://github.com/isl-org/Open3D/issues/1343

        #Bug workaround where we scale the geometry insted of the viewport
        scale_up_factor = cam_mat[1][1]/cam_mat[0][0]
        
        
        
        for obj in objects:
            obj2 = copy.deepcopy(obj)
            if hasattr(obj2, 'points'):
                np.asarray(obj2.points)[:,1] *= scale_up_factor
            else:
                np.asarray(obj2.vertices)[:,1] *= scale_up_factor

            vis.add_geometry(obj2)
            vis.update_geometry(obj2)


        intrinsic.intrinsic_matrix = np.array([
            [999999, 0.          , cam_mat[0][2]     ],#99999 should be focalx This is reversed from a normal cam_matrix but this is a hack and it works.. dont ask se above bug
            [  0.  , cam_mat[0][0]      , cam_mat[1][2]     ],
            [  0.  , 0.          , 1.        ]])
        params.intrinsic = intrinsic
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.update_renderer()


        rend_opt.light_on = False
        vis.poll_events()
        vis.update_renderer()


        #For some reason using capture_depth_float_buffer is very slow taking about a tenth of a second while capture_screen_float_buffer is like 100 times faster
        #Probably due to this https://github.com/isl-org/Open3D/blob/c6d474b3fa0b47adbcff51219f5928855c3bb806/cpp/open3d/visualization/visualizer/VisualizerRender.cpp#L286
        if depth == -2:
            ret = (np.asarray(vis.capture_screen_float_buffer(do_render=True)), np.asarray(vis.capture_depth_float_buffer(do_render=False)))
            assert ret[0].shape[0] == h and ret[0].shape[1] == w and ret[1].shape[0] == h and ret[1].shape[1] == w, f"Render output is not the correct width ({w} != {ret[0].shape[1]}) and height ({h} !=  {ret[0].shape[0]})"
        if depth == False:
            ret = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            assert ret.shape[0] == h and ret.shape[1] == w, "Render output is not the correct width and height"
        if depth == True:
            ret = np.asarray(vis.capture_depth_float_buffer(do_render=True))
            assert ret.shape[0] == h and ret.shape[1] == w, "Render output is not the correct width and height"
        return ret
    else:

        scene = rend.scene
        scene.clear_geometry()
        scene.set_background([1.0, 1.0, 1.0, 1.0]) #white
        scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
                           np.array([0.0, 0.0, 0.0]))

        scene.camera.set_view_proj(extrinsic_matric, cam_mat)
        scene.camera.set_projection(cam_mat)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        
        for obj in objects:
            scene.add_geometry("mesh", obj, mat)

        if depth:
            image = rend.render_to_depth_image()
        else:
            image = rend.render_to_image()

    ret = np.asarray(image)
    assert ret.shape[0] == h and ret.shape[1] == w, "Render output is not the correct width and height"
    return ret

def gl_look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)

    s = np.cross(f, up)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye

    return M @ T

def cam_look_at(cam_pos, target, up = np.array([0.0, 1.0, 0.0])):

    f = target - cam_pos
    f /= np.linalg.norm(f)

    # 2) Right vector: cross(Up, Forward)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)

    # 3) Actual up vector: cross(Forward, Right)
    u = np.cross(f, r)

    # 4) Build the view matrix in row-major form
    mat = np.array([
        [r[0],   u[0],   f[0],   cam_pos[0]],
        [r[1],   u[1],   f[1],   cam_pos[1]],
        [r[2],   u[2],   f[2],   -cam_pos[2]],
        [-np.dot(r, target), -np.dot(u, target), -np.dot(f, target), 1.0]
    ], dtype=float)

    return mat

def fov_from_camera_matrix(mat):
    w = mat[0][2]*2
    h = mat[1][2]*2
    fx = mat[0][0]
    fy = mat[1][1]

    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return fov_x, fov_y


def draw(what):
    lookat = what[0].get_center()
    lookat[2] = 1
    lookat[1] = 0
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    what.append(mesh)
    o3d.visualization.draw_geometries(what, front=[ 0.0, 0.23592114315107779, -1.0 ], lookat=lookat,up=[ 0, -1, 0 ], zoom=0.53199999999999981)