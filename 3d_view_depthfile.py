import argparse
import cv2
import numpy as np
import os
import json
import sys
import time
import copy
import depth_frames_helper

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
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--render', action='store_true', help='Render to video insted of GUI', required=False)
    parser.add_argument('--render_as_pointcloud', action='store_true', help='Render as point cloud instead of as mesh', required=False)
    parser.add_argument('--remove_edges', action='store_true', help='Tries to remove edges that was not visible in image(it is a bit slow)', required=False)
    parser.add_argument('--show_camera', action='store_true', help='Shows lines representing the camera frustrum', required=False)
    parser.add_argument('--background_ply', type=str, help='PLY file that will be included in the scene', required=False)
    parser.add_argument('--mask_video', type=str, help='Mask video to filter out back or forground', required=False)
    parser.add_argument('--invert_mask', action='store_true', help='Remove the baground(black) instead of the forground(white)', required=False)
    
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
    mesh, draw_mesh = None, None
    
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
        
        color_frame = None
        if color_video is not None:
            ret, color_frame = color_video.read()
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            
            assert color_frame.shape == rgb.shape, "color image and depth image need to have same width and height" #potential BUG here with mono depth videos
        else:
            color_frame = rgb
            
        mask = None
        if mask_video is not None:
            ret, mask = mask_video.read()
            if ret:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if args.invert_mask:
                    mask = 255-mask
            
        if args.draw_frame != -1 and args.draw_frame != frame_n:
            continue

        # Decode video depth
        depth = depth_frames_helper.decode_rgb_depth_frame(rgb, MODEL_maxOUTPUT_depth, True)
        
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
            
        of_by_one = True
        if args.render_as_pointcloud:
            of_by_one = False
        
        mesh_ret, included_points = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame, mesh, remove_edges = args.remove_edges, mask = mask, of_by_one = of_by_one)
        
        
        if transformations is not None:
            mesh_ret.transform(transform_to_zero)
        
        add_geom = False
        if mesh is None:
            add_geom = True

        mesh = mesh_ret
        
        
        if args.render_as_pointcloud:
            draw_mesh = depth_map_tools.convert_mesh_to_pcd(mesh, included_points, draw_mesh)
            #TODO:move points that is vertices back to their real position
        else:
            draw_mesh = mesh
        
        if add_geom:
            if not args.render and args.draw_frame == -1:
                vis.add_geometry(draw_mesh)
            if background_obj is not None:
                vis.add_geometry(background_obj)
        
        
        to_draw = [draw_mesh]
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
            vis.update_geometry(draw_mesh)
        
        
        # Set Camera position
        lookat = draw_mesh.get_center()
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
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if args.max_frames < frame_n and args.max_frames != -1:
            break
    
    raw_video.release()
    if args.render:
        out.release()
    
