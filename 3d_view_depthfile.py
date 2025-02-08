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

np.set_printoptions(suppress=True, precision=4)



if __name__ == '__main__':
    
    # Setup arguments
    parser = argparse.ArgumentParser(description='Take a rgb encoded depth video and a color video, and view it/render as 3D')
    
    parser.add_argument('--depth_video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--render', action='store_true', help='Render to video insted of GUI', required=False)
    parser.add_argument('--remove_edges', action='store_true', help='Tries to remove edges that was not visible in image(it is a bit slow)', required=False)
    
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
        
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
        
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
    

    frame_n = 0
    while raw_video.isOpened():
        
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        frame_n += 1
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
            
        if args.draw_frame != -1 and args.draw_frame != frame_n:
            continue

        # Decode video depth
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((rgb[..., 0].astype(np.uint32) + rgb[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        
        
        if transformations is not None:
            transform_to_zero = np.array(transformations[frame_n-1])
            
        
        #This is very slow needs optimizing (i think)
        mesh_ret, _ = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, mesh, remove_edges = args.remove_edges)
        
        if mesh is None:
            if not args.render and args.draw_frame == -1:
                vis.add_geometry(mesh_ret)
        mesh = mesh_ret
        
        if transformations is not None:
            transform_to_zero = np.array(transformations[frame_n-1])
            mesh.transform(transform_to_zero)
        
        if args.draw_frame == frame_n:
            to_draw = [mesh]

            depth_map_tools.draw(to_draw)
            exit(0)
        
        
        if not args.render and args.draw_frame == -1:
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
                params.extrinsic = ext
                params.intrinsic.intrinsic_matrix = cam_matrix
                ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        
                start_time = time.time()
                while time.time() - start_time < 0.1: #should be (1/frame_rate) but we dont rach that speed anyway
                    vis.poll_events()
                    vis.update_renderer()
        else:
            image = (depth_map_tools.render(mesh, cam_matrix, extrinsic_matric = ext, bg_color = np.array([1.0,1.0,1.0]))*255).astype(np.uint8)
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if args.max_frames < frame_n and args.max_frames != -1:
            break
    
    raw_video.release()
    if args.render:
        out.release()
    
