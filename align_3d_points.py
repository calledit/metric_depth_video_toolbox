import numpy as np
import os
import cv2
import json
import argparse
import depth_map_tools

    
    
def find_best_matching_frame(selected_frame, frames, used_frames):
    # Extract point IDs from the selected frame
    point_ids_in_selected_frame = {point[0] for point in selected_frame}  # Use a set for fast lookup
    
    frame_common_counts = []  # Store (frame_id, common points)

    for frame_id, frame in enumerate(frames):
        if frame_id in used_frames:  # Ignore already used frames
            continue

        # Extract point IDs from the current frame
        points_in_frame = {point[0] for point in frame}  
        
        # Find common points
        common_elements = list(point_ids_in_selected_frame & points_in_frame)  # Set intersection

        frame_common_counts.append((frame_id, common_elements))  # Store frame ID and common points

    # Sort by the number of common points in descending order
    frame_common_counts.sort(key=lambda x: len(x[1]), reverse=True)

    # Get the best frame ID and its common points (if available)
    if frame_common_counts:
        best_frame_id, best_common_points = frame_common_counts[0]
    else:
        best_frame_id, best_common_points = None, []

    return best_frame_id, best_common_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align 3D video based on depth video and a point tracking file')

    parser.add_argument('--track_file', type=str, help='file with 2d point tracking data', required=True)
    parser.add_argument('--mask_video', type=str, help='black and white mask video for thigns that should not be tracked', required=False)
    parser.add_argument('--strict_mask', default=False, action='store_true', help='Remove any points that has ever been masked out even in frames where they are not masked', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--depth_video', type=str, help='depth video', required=True)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--color_video', type=str, help='video file to use as color input only used when debuging', required=False)

    args = parser.parse_args()
    
    if args.xfov is None and args.yfov is None:
        print("Either --xfov or --yfov is required.")
        exit(0)

    if not os.path.isfile(args.track_file):
        raise Exception("input track_file does not exist")
        
    if not os.path.isfile(args.mask_video):
        raise Exception("input mask_video does not exist")
        
    if not os.path.isfile(args.depth_video):
        raise Exception("input depth_video does not exist")
    
    with open(args.track_file) as json_track_file_handle:
        frames = json.load(json_track_file_handle)
    
    MODEL_maxOUTPUT_depth = args.max_depth
        
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
        
    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)
    
    color_video = None
    if args.color_video is not None:
        if not os.path.isfile(args.color_video):
            raise Exception("input color_video does not exist")
        color_video = cv2.VideoCapture(args.color_video)
    
    mask_video = None
    if args.mask_video is not None:
        if not os.path.isfile(args.mask_video):
            raise Exception("input color_video does not exist")
        mask_video = cv2.VideoCapture(args.mask_video)
    
    for i, frame in enumerate(frames):
        frames[i] = np.array(frames[i])
    
    depth_frames = []
    rgb_frames = []
    fr_n = 0
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
            
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        
        if color_video is not None:
            ret, col_vid = color_video.read()
            col_vid = cv2.cvtColor(col_vid, cv2.COLOR_BGR2RGB)
            rgb_frames.append(col_vid)
        else:
            rgb_frames.append(raw_frame)
            
        if mask_video is not None:
            ret, mask = mask_video.read()
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            
            rem = []
            rem_global = []
            for i, point in enumerate(frames[fr_n]):
                if mask[point[2], point[1]] > 0:
                    rem.append(i)
                    rem_global.append(point[0])
            
            if len(rem) > 0:    
                frames[fr_n] = np.delete(frames[fr_n], rem, axis=0)
            
            if args.strict_mask:
                for global_id in rem_global:
                    for frame_id, frame in enumerate(frames):
                        rem = []
                        for i, point in enumerate(frames[fr_n]):
                            if global_id == point[0]:
                                rem.append(i)
                        if len(rem) > 0:
                            frames[frame_id] = np.delete(frames[frame_id], rem, axis=0)
        
        
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((raw_frame[..., 0].astype(np.uint32) + raw_frame[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = raw_frame[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        depth_frames.append(depth)
        
        #DEBUG: Only looking at 25 frames dont want to load entire video when DEBUGING
        if fr_n > 210:
            break
            
        fr_n += 1
    raw_video.release()
    if color_video is not None:
        color_video.release()
    
    used_frames = []
    
    #1. Pick the first frame
    frame_n = 0
    
    used_frames.append(frame_n)
    
    #for i in range(20): # DEBUG IF you want to see what happens with the alignment after more than 1 frame
    #    used_frames.append(i)
    
    #ref_mesh Is used to draw the DEBUG alignment window
    ref_mesh = depth_map_tools.get_mesh_from_depth_map(depth_frames[frame_n], cam_matrix, rgb_frames[frame_n])
    ref_mesh.paint_uniform_color([0, 0, 1])
    
    meshes = [ref_mesh]
    to_ref_zero = np.eye(4)
    while len(used_frames) < len(frames):
        print("--- frame ", frame_n, " ---")
        #2. Find most connected frame (tends to be the next frame)
        best_match_frame_no, best_common_points = find_best_matching_frame(frames[frame_n], frames, used_frames)
    
        print("match: ", best_match_frame_no, "nr_matches: ", len(best_common_points))
        
        #Current frame points
        point_ids_in_frame = frames[best_match_frame_no][:,0]
        cur_mask = np.isin(point_ids_in_frame, best_common_points)
        points_2d = frames[best_match_frame_no][cur_mask][:, 1:3]
        points_3d = depth_map_tools.project_2d_points_to_3d(points_2d, depth_frames[best_match_frame_no], cam_matrix)
        
        
        #Ref frame points
        point_ids_in_frame = frames[frame_n][:,0]
        cur_mask = np.isin(point_ids_in_frame, best_common_points)
        points_2d = frames[frame_n][cur_mask][:, 1:3]
        ref_points_3d = depth_map_tools.project_2d_points_to_3d(points_2d, depth_frames[frame_n], cam_matrix)
        
        
        ref_pcd = depth_map_tools.pts_2_pcd(ref_points_3d)
        frame_pcd = depth_map_tools.pts_2_pcd(points_3d)
        
        ref_pcd.paint_uniform_color([1, 0, 0])
        frame_pcd.paint_uniform_color([0, 1, 0])
        
        #Use SVD to find transformation from one frame to the next
        tranformation_to_ref = depth_map_tools.svd(points_3d, ref_points_3d)
        
        to_ref_zero @= tranformation_to_ref
        
        
        ref_pcd.transform(to_ref_zero)
        frame_pcd.transform(to_ref_zero)
        
        meshes.append(ref_pcd)
        meshes.append(frame_pcd)
        
        
        
        #Transform the mesh so that we can see how well it aligns in the GUI
        #mesh.transform(tranformation_to_ref)
        
        #if frame_n % 25 == 0:
        
        if frame_n == 200:
            mesh = depth_map_tools.get_mesh_from_depth_map(depth_frames[best_match_frame_no], cam_matrix, rgb_frames[best_match_frame_no])
            mesh.transform(to_ref_zero)
            meshes.append(mesh)
        
        
            depth_map_tools.draw(meshes)
        
        frame_n = best_match_frame_no
        used_frames.append(frame_n)
        
        
        