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
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--min_frames', default=-1, type=int, help='start convertion after nr of frames', required=False)
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--transformation_file', type=str, help='file with scene transformations from the aligner', required=False)
    parser.add_argument('--transformation_lock_frame', default=0, type=int, help='the frame that the transformation will use as a base', required=False)
    parser.add_argument('--remove_edges', action='store_true', help='Tries to remove edges that was not visible in image', required=False)
    
    
    
    args = parser.parse_args()
    
   
    MODEL_maxOUTPUT_depth = args.max_depth
    
    # Verify input file exists
    if not os.path.isfile(args.depth_video):
        raise Exception("input video does not exist")

        
    output_file = args.depth_video + "_grey_depth.mkv"
    
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
    
    cam_matrix = None
    if args.save_ply is not None or args.save_obj is not None:
        if args.xfov is None and args.yfov is None:
            print("Either --xfov or --yfov is required.")
            exit(0)
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
        
        if cam_matrix is not None:
            transform_to_zero = np.eye(4)
            if transformations is not None:
                transform_to_zero = np.array(transformations[frame_n-1])
            mesh_ret, used_indices = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, mesh, remove_edges = args.remove_edges)
            
            if transformations is not None:
                mesh_ret.transform(transform_to_zero)
                
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
        
    raw_video.release()
    if out is not None:
        out.release()
