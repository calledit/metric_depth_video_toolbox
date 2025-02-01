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
        centroid_source = np.array([0.0,0.0,0.0])
        centroid_target = np.array([0.0,0.0,0.0])
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

    # Compute the translation vector
    t = centroid_target - np.dot(Rot, centroid_source)#original function


    # Form the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = Rot
    transformation_matrix[:3, 3] = t

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
        reperr = 1
    success, rvec, tvec, inliers = cv2.solvePnPRansac(t3d_points_new_frame, np.array(mkpts2,dtype=np.float64), cam_mat, distCoeffs, reprojectionError=reperr, iterationsCount=100000)
    matrix = np.eye(4)
    if success:
        tv = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
        mat, jac = cv2.Rodrigues(rvec)

        if refine:
            rvec2, tvec2 = cv2.solvePnPRefineLM(np.array(t3d_points_new_frame[inliers],dtype=np.float64), np.array(mkpts2[inliers],dtype=np.float64), cam_mat, distCoeffs, rvec, tvec)
            tv = np.array([tvec2[0][0], tvec2[1][0], tvec2[2][0]])
            mat, jac = cv2.Rodrigues(rvec2)

        matrix[:3, :3] = mat
        matrix[:3, 3] = tv
        return matrix
    print("solvePnP FAIL")
    return None

def pts_2_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

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
    mesh = create_mesh_from_point_cloud(points, height, width, color_frame, inp_mesh, remove_edges)
    return mesh

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




def is_triangle_valid(v1, v2, v3, depth_threshold_scale):
    """
    Checks if the triangle defined by vertices v1, v2, and v3 is valid.
    Instead of a fixed threshold, the maximum allowed depth difference is 
    computed as depth_threshold_scale * average_depth of the triangle.
    
    Parameters:
      - v1, v2, v3: Each is a numpy array [x, y, z] representing a vertex.
      - depth_threshold_scale: A scaling factor applied to the triangle's 
                               average depth to determine the threshold.
    
    Returns:
      - True if the depth difference is within the computed threshold; False otherwise.
    """
    depths = np.array([v1[2], v2[2], v3[2]])
    avg_depth = depths.mean()
    # Compute the threshold as a linear function of the average depth.
    threshold = depth_threshold_scale * avg_depth
    return (depths.max() - depths.min()) <= threshold

zero_identity_matrix = np.identity(4)
def create_mesh_from_point_cloud(points, height, width,
                                 image_frame=None, inp_mesh=None, remove_edges=False):
    """
    Creates (or updates) an Open3D TriangleMesh from a grid-organized point cloud.
    If no input mesh is provided or if remove_edges is True, the triangles are computed.
    Otherwise (when inp_mesh is provided and remove_edges is False), the function
    simply updates the vertices (and optionally the vertex colors).

    Parameters:
      - points: a numpy array that can be reshaped to (-1, 3) containing the 3D points.
      - height: number of rows in the grid.
      - width: number of columns in the grid.
      - image_frame: (optional) image whose colors are mapped to the mesh vertices.
      - inp_mesh: (optional) an existing mesh to update.
      - remove_edges: if True, triangles with large depth gaps are filtered out.

    Returns:
      - mesh: an Open3D TriangleMesh with updated vertices (and triangles if computed).
    """
    
    # Reshape the points into a (N, 3) array of vertices.
    vertices = points.reshape(-1, 3)
    
    # Case 1: We need to compute triangles (either no input mesh exists, or we need to remove edges).
    if inp_mesh is None or remove_edges:
        # If there's no input mesh, create a new one; otherwise, update the provided mesh.
        if inp_mesh is None:
            mesh = o3d.geometry.TriangleMesh()
        else:
            mesh = inp_mesh
            mesh.transform(zero_identity_matrix)
        
        # Generate grid-based triangle indices using vectorized operations.
        # For each cell at (i, j) with i in [0, height-2] and j in [0, width-2]:
        #   - The four corner indices of the cell are computed.
        #   - Two triangles are formed:
        #         tri1: (i, j), (i+1, j), (i+1, j+1)
        #         tri2: (i, j), (i+1, j+1), (i, j+1)
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
        
        # If remove_edges is True, filter triangles based on a depth gap that scales with average depth.
        if remove_edges:
            depth_threshold_scale = 0.04
            # Gather vertices for each triangle; shape: (N_tri, 3, 3)
            tri_vertices = vertices[triangles_all]
            # Extract the z-values (depth) for each vertex.
            depths = tri_vertices[:, :, 2]
            # Compute the average depth for each triangle.
            avg_depth = depths.mean(axis=1)
            # Allowed depth gap is linear with average depth.
            allowed_thresholds = depth_threshold_scale * avg_depth
            # Compute the actual depth range for each triangle.
            depth_ranges = depths.max(axis=1) - depths.min(axis=1)
            # Only keep triangles where the depth range is within the allowed threshold.
            valid_mask = depth_ranges <= allowed_thresholds
            triangles_all = triangles_all[valid_mask]
        
        # Set the computed triangles and vertices on the mesh.
        mesh.triangles = o3d.utility.Vector3iVector(triangles_all)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Case 2: We already have an input mesh and we are not removing edges.
    # In this case, simply update the vertices (and leave the triangles unchanged).
    else:
        mesh = inp_mesh
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Optionally, assign vertex colors if an image frame is provided.
    if image_frame is not None:
        colors = np.array(image_frame).reshape(-1, 3) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    #mesh.compute_vertex_normals()
    return mesh

def create_mesh_from_point_cloud2(points, height, width,
                                 image_frame=None, inp_mesh=None, remove_edges=False):
    """
    Creates an Open3D TriangleMesh from a grid-organized point cloud while
    filtering out triangles that span large depth gaps. The allowable depth gap 
    scales linearly with the triangle's average depth.
    
    Returns:
      - mesh: the Open3D TriangleMesh with only the valid triangles.
    """
    
    if remove_edges:
        depth_threshold_scale = 0.04
                                 
    # Reshape the points into a list of vertices.
    vertices = points.reshape(-1, 3)

    # Create or update the Open3D mesh. (This is very sow so we only do it when we need to)
    if remove_edges or inp_mesh is None:
        if inp_mesh is None:
            mesh = o3d.geometry.TriangleMesh()
        else:
            mesh = inp_mesh
            mesh.transform(zero_identity_matrix)
        # Build triangles by connecting grid-adjacent points,
        # and only keep triangles that pass the depth continuity check.
        triangles = []
        for i in range(height - 1):
            for j in range(width - 1):
                # Indices for the four corners of the current grid cell.
                idx1 = i * width + j
                idx2 = (i + 1) * width + j
                idx3 = (i + 1) * width + (j + 1)
                idx4 = i * width + (j + 1)

                # Define two candidate triangles for the cell.
                tri1 = [idx1, idx2, idx3]
                tri2 = [idx1, idx3, idx4]

                if remove_edges:
                    # Check each triangle with a depth threshold that scales linearly with depth.
                    if is_triangle_valid(vertices[idx1], vertices[idx2], vertices[idx3], depth_threshold_scale):
                        triangles.append(tri1)
                    if is_triangle_valid(vertices[idx1], vertices[idx3], vertices[idx4], depth_threshold_scale):
                        triangles.append(tri2)
                else:
                    triangles.append(tri1)
                    triangles.append(tri2)

        
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    else:
        mesh = inp_mesh
        # Assume zero_identity_matrix is defined elsewhere if needed.
        mesh.transform(zero_identity_matrix)
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Optionally, assign vertex colors from an image frame.
    if image_frame is not None:
        colors = np.array(image_frame).reshape(-1, 3) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    #mesh.compute_vertex_normals()
    return mesh

vis = None
v_h = None
use_ofscreen = True
v_w = None
rend = None
def render(pcd, cam_mat, depth = False, w = None, h = None, extrinsic_matric = np.eye(4), bg_color = np.array([0, 0, 0])):
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
        pcd = copy.deepcopy(pcd)
        pcd.vertices = o3d.utility.Vector3dVector(np.asarray(pcd.vertices)*np.array([1.0, scale_up_factor, 1.0]))#scale


        vis.add_geometry(pcd)
        vis.update_geometry(pcd)


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

        scene.add_geometry("mesh", pcd, mat)

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