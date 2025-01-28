import numpy as np
import open3d as o3d
import copy

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

def get_mesh_from_depth_map(depth_map, cam_mat, color_frame = None):
    points, height, width = create_point_cloud_from_depth(depth_map, cam_mat, True)

    # Create mesh from point cloud
    mesh = create_mesh_from_point_cloud(points, height, width, color_frame)
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


def create_mesh_from_point_cloud(points, height, width, image_frame = None):
    # Create a mesh by connecting adjacent points
    vertices = points.reshape(-1, 3)

    # Create triangles based on the grid layout
    triangles = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Indices for the square formed by the current grid cell
            idx1 = i * width + j
            idx2 = (i + 1) * width + j
            idx3 = (i + 1) * width + (j + 1)
            idx4 = i * width + (j + 1)

            # Create two triangles for each square
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx1, idx3, idx4])

    # Convert triangles to a NumPy array
    triangles = np.array(triangles)

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    if image_frame is not None:
        color_image = image_frame
        colors = np.array(color_image).reshape(-1, 3) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    #mesh.compute_vertex_normals()

    return mesh

vis = None
v_h = None
use_ofscreen = True
v_w = None
rend = None
def render(pcd, cam_mat, depth = False, w = None, h = None, extrinsic_matric = np.eye(4)):
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


        if depth:
            image = vis.capture_depth_float_buffer(do_render=True)
        else:
            image = vis.capture_screen_float_buffer(do_render=True)
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