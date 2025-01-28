import argparse
import cv2
import numpy as np
import os
import sys
import time

import open3d as o3d
import depth_map_tools

np.set_printoptions(suppress=True, precision=4)

if __name__ == '__main__':
    
    # Setup arguments
    parser = argparse.ArgumentParser(description='Generate a depth video in greyscale from a rgb encoded depth video')
    
    parser.add_argument('--video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction', required=True)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--max_depth', default=6, type=int, help='the max depth that the video uses', required=False)
    
    
    
    args = parser.parse_args()
    
   
    MODEL_maxOUTPUT_depth = args.max_depth
    
    # Verify input file exists
    if not os.path.isfile(args.video):
        raise Exception("input video does not exist")
    
    color_video = None
    if args.color_video is not None:
        if not os.path.isfile(args.color_video):
            raise Exception("input color_video does not exist")
        color_video = cv2.VideoCapture(args.color_video)
        
    raw_video = cv2.VideoCapture(args.video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
        
    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)

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
    
    org_mesh = None

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

        # Decode video depth
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((rgb[..., 0].astype(np.uint32) + rgb[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        
        #This is very slow needs optimizing (i think)
        mesh = depth_map_tools.get_mesh_from_depth_map(depth, cam_matrix, color_frame)
        
        if org_mesh is None:
            org_mesh = mesh
            vis.add_geometry(org_mesh)
            
        org_mesh.vertices = mesh.vertices
        org_mesh.vertex_colors = mesh.vertex_colors
        vis.update_geometry(org_mesh)
        params.extrinsic = [
            [ 0.8837,-0.1421,-0.4459,  1.0534],
            [-0.0598 , 0.9107, -0.4088 , 2.038 ],
            [ 0.4642 , 0.3879 , 0.7963 , 4.442 ],
            [ 0.    ,  0.     , 0.   ,   1.    ]
        ]
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        start_time = time.time()
        while time.time() - start_time < 0.1: #should be (1/frame_rate) but we dont rach that speed anyway
            vis.poll_events()
            vis.update_renderer()
        
        
        #depth_map_tools.draw([mesh])
        
        
    raw_video.release()



def draw(what):
    lookat = what[0].get_center()
    lookat[2] = 1
    lookat[1] = 0 
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    what.append(mesh)
    o3d.visualization.draw_geometries(what, front=[ 0.0, 0.23592114315107779, -1.0 ], lookat=lookat,up=[ 0, -1, 0 ], zoom=0.53199999999999981)