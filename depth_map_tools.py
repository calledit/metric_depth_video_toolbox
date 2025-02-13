import numpy as np
import open3d as o3d
import copy
import cv2
from contextlib import contextmanager
import time

@contextmanager
def timer(name = 'not named'):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.6f} seconds")


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

def pts_2_pcd(points, colors = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
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


def get_mesh_from_depth_map(depth_map, cam_mat, color_frame = None, inp_mesh = None, remove_edges = False):
    points, height, width = create_point_cloud_from_depth(depth_map, cam_mat, True)

    # Create mesh from point cloud
    mesh, used_indices = create_mesh_from_point_cloud(points, height, width, color_frame, inp_mesh, remove_edges)
    return mesh, used_indices

def create_point_cloud_from_depth(depth_image, intrinsics, of_by_one = False):
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    #Here we fix a of by one error caused by the fact that this function fills in the area betwen each vertex
    if of_by_one:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x *= (width+1)/width
        y *= (height+1)/height


    z = depth_image  # Assuming depth is in millimeters
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
                                 angle_threshold_deg=85):
    """
    Creates an Open3D TriangleMesh from a grid-organized point cloud while
    filtering out triangles whose orientation relative to the camera is too oblique.
    
    The function assumes the 3D points are in camera coordinates (i.e. the camera is at the origin).
    
    Parameters:
      - points: A numpy array that can be reshaped to (-1, 3) containing the 3D points.
      - height: The number of rows in the grid.
      - width: The number of columns in the grid.
      - image_frame: (Optional) An image whose colors will be mapped to the mesh vertices.
      - camera_intrinsic_matrix: (Not used here; points are assumed to be already projected.)
      - inp_mesh: (Optional) An existing mesh to update.
      - remove_edges: If True, triangles with normals that deviate too far from the view vector are removed.
      - angle_threshold_deg: The maximum allowed angle (in degrees) between a triangle’s normal and 
                             the view vector. Triangles with an angle larger than this threshold are discarded.
    
    Returns:
      - mesh: The resulting Open3D TriangleMesh.
    """
    # Reshape points into a (N, 3) array of vertices.
    vertices = points.reshape(-1, 3)
    
    used_indices = []
    
    # If no mesh exists or if we need to remove edges, compute the triangles.
    if inp_mesh is None or remove_edges:
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
        grid_i = grid_i.ravel()  # Flatten to 1D arrays (num_cells,)
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
        
        ref_to_all_tri = np.asarray(mesh.triangles)
        ref_to_all_vert = np.asarray(mesh.vertices)
        if inp_mesh is not None:
            ref_to_all_tri[:] = triangles_all[:]
            ref_to_all_vert[:] = vertices[:]
        
        # --- Filter triangles based on the triangle angle relative to the camera ---
        if remove_edges:
            v1 = vertices[triangles_all[:, 0]]
            v2 = vertices[triangles_all[:, 1]]
            v3 = vertices[triangles_all[:, 2]]
            cos_threshold = np.cos(np.radians(angle_threshold_deg))
            
            normals = np.cross(v2 - v1, v3 - v1)            # shape (N, 3)
            view    = - (v1 + v2 + v3)/3.0                  # same shape (N, 3) as centers
            dot     = np.einsum('ij,ij->i', normals, view)  # dot products
            len_n   = np.sqrt(np.einsum('ij,ij->i', normals, normals))
            len_v   = np.sqrt(np.einsum('ij,ij->i', view, view))
            cosines = dot / (len_n * len_v + 1e-15)         # +1e-15 to avoid div-by-zero

            invalid_mask = (cosines < cos_threshold)
            ref_to_all_tri[invalid_mask] = np.array([0,0,0])
            
            # 1) Identify which rows are *not* all zero:
            valid_mask = np.logical_not(invalid_mask)

            # 2) Boolean mask to track used vertices
            num_vertices = ref_to_all_vert.shape[0]  # or known from your logic
            is_used = np.zeros(num_vertices, dtype=bool)

            # 3) Mark vertices in valid triangles
            is_used[ ref_to_all_tri[valid_mask].ravel() ] = True

            # 4) Extract the used indices
            used_indices = np.where(is_used)[0]
    
    # If we already have an input mesh and we are not removing edges, simply update vertices.
    else:
        mesh = inp_mesh
        ref_to_all_vert = np.asarray(mesh.vertices)
        ref_to_all_vert[:] = vertices[:]
    
    # Optionally, set vertex colors.
    if image_frame is not None:
        colors = np.array(image_frame).reshape(-1, 3) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    
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
            vis.create_window(width=int(w), height=int(h), visible=False) #works for me with False, on some systems needs to be true
        vis.clear_geometries()

        rend_opt = vis.get_render_option()
        rend_opt.background_color = bg_color
        rend_opt.point_size = 1.0
        
        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 1])
        ctr.set_up([0, -1, 0])
        ctr.set_front([0, 0, -1])
        ctr.set_zoom(1)
        


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
            return (np.asarray(vis.capture_screen_float_buffer(do_render=True)), np.asarray(vis.capture_depth_float_buffer(do_render=False)))
        if depth == False:
            return(np.asarray(vis.capture_screen_float_buffer(do_render=True)))
        if depth == True:
            return(np.asarray(vis.capture_depth_float_buffer(do_render=True)))
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


    return np.asarray(image)

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