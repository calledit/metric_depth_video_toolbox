import numpy as np
import open3d as o3d

def compute_camera_matrix(fov_horizontal_deg, fov_vertical_deg, image_width, image_height):
    # Convert FoV from degrees to radians
    fov_horizontal_rad = np.deg2rad(fov_horizontal_deg)

    # Compute the focal lengths in pixels
    fx = image_width /  (2 * np.tan(fov_horizontal_rad / 2))

    if fov_vertical_deg is None:
        fy = fx
    else:
        fov_vertical_rad = np.deg2rad(fov_vertical_deg)
        fy = image_height / (2 * np.tan(fov_vertical_rad / 2))

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
    
    
    
    
def fov_from_camera_matrix(mat):
    w = mat[0][2]*2
    h = mat[1][2]*2
    fx = mat[0][0]
    fy = mat[1][1]
    
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return fov_x, fov_y