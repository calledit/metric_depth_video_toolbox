import numpy as np
import os
import cv2
import json
import argparse
import depth_map_tools
from itertools import islice

    
    
def find_best_matching_frame(selected_frame_id, frames, used_frames):
    selected_frame = frames[selected_frame_id]
    # Extract point IDs from the selected frame
    if len(selected_frame) == 0:
        print(selected_frame_id, "has zero registerd points")
        return
    point_ids_in_selected_frame = set(selected_frame[:, 0])  # Use a set for fast lookup
    
    frame_common_counts = []  # Store (frame_id, common points)
    
    
    
    start_index = max(0, selected_frame_id - 60)
    end_index = min(selected_frame_id + 60, len(frames))

    #for frame_id, frame in enumerate(frames):
    for frame_id, frame in enumerate(islice(frames, start_index, end_index), start=start_index):
        if frame_id in used_frames:  # Ignore already used frames
            continue
            
        if len(frame) == 0:
            continue
            
        # Extract point IDs from the current frame
        points_in_frame = set(frame[:, 0])
        
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
    
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align 3D video based on depth video and a point tracking file')

    parser.add_argument('--track_file', type=str, help='file with 2d point tracking data', required=True)
    parser.add_argument('--mask_video', type=str, help='black and white mask video for thigns that should not be tracked', required=False)
    parser.add_argument('--strict_mask', default=False, action='store_true', help='Remove any points that has ever been masked out even in frames where they are not masked', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument('--depth_video', type=str, help='depth video', required=True)
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--color_video', type=str, help='video file to use as color input only used when debuging', required=False)
    parser.add_argument('--assume_stationary_camera', action='store_true', help='Makes the algorithm assume the camera a stationary_camera, leads to better tracking.', required=False)
    

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
    
    
    out_file = args.depth_video + "_transformations.json"
    
    MODEL_maxOUTPUT_depth = args.max_depth
        
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
        
    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)
    
    color_video = None
    debug_video = None
    if args.color_video is not None:
        if not os.path.isfile(args.color_video):
            raise Exception("input color_video does not exist")
        color_video = cv2.VideoCapture(args.color_video)
        debug_video = cv2.VideoWriter(args.color_video+"_track_debug.mp4", cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (frame_width, frame_height))
    
    mask_video = None
    if args.mask_video is not None:
        if not os.path.isfile(args.mask_video):
            raise Exception("input color_video does not exist")
        mask_video = cv2.VideoCapture(args.mask_video)
    
    for i, frame in enumerate(frames):
        frames[i] = np.array(frames[i])
    
    
    used_frames = []
    
    #1. Pick the first frame
    frame_n = 0
    used_frames.append(frame_n)
    
    
    transformations = []
    to_ref_zero = np.eye(4)
    transformations.append(to_ref_zero.tolist())
    
    # Load frames
    depth_frames = []
    rgb_frames = []
    fr_n = 0
    while raw_video.isOpened():
        
        print("--- frame ", fr_n + 1, " ---")
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
            if ret:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            
                rem = []
                rem_global = []
                for i, point in enumerate(frames[fr_n]):
                    if point[1] >= frame_width or point[2] >= frame_height or mask[point[2], point[1]] > 0:
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
            else:
                print("WARNING: mask video ended before other videos")
        
        
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((raw_frame[..., 0].astype(np.uint32) + raw_frame[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = raw_frame[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        depth_frames.append(depth)
        
        #DEBUG: Only looking at 25 frames dont want to load entire video when DEBUGING
        #if fr_n > 210:
        #    break
        
        #Add online analyzing so we dont have to load everything in to memory
        if len(depth_frames) > 1:
            ref_frame_no = fr_n - 1
            this_frame_no = fr_n
            best_common_points = list(set(frames[ref_frame_no][:, 0]) & set(frames[this_frame_no][:, 0]))
            
            #Current frame points
            point_ids_in_frame = frames[this_frame_no][:,0]
            cur_mask = np.isin(point_ids_in_frame, best_common_points)
            points_2d = frames[this_frame_no][cur_mask][:, 1:3]
            points_3d = depth_map_tools.project_2d_points_to_3d(points_2d, depth_frames[1], cam_matrix)
            
            #depth in the distance is often wrong we incresse that to a value that is bascially infinity
            #points_to_keep = points_3d[:, 2] > 4.8
            
            #Move distant points
            #points_3d[points_to_keep][:, 2] = 40.0
            #points_2d = points_2d[points_to_keep]
            
    
    
            #Ref frame points
            point_ids_in_frame = frames[ref_frame_no][:,0]
            cur_mask = np.isin(point_ids_in_frame, best_common_points)
            ref_points_2d = frames[ref_frame_no][cur_mask][:, 1:3]
            ref_points_3d = depth_map_tools.project_2d_points_to_3d(ref_points_2d, depth_frames[0], cam_matrix)
    
            #ref_points_3d = ref_points_3d[points_to_keep]
            
            #Move distant points
            #ref_points_3d[points_to_keep][:, 2] = 40.0
            
            #ref_points_2d = ref_points_2d[points_to_keep]
            
            #Reject points that dont move like the others this should probably be done on a larger time scale instead of frame 2 frame
            movments = np.mean(np.abs(points_3d-ref_points_3d), axis=-1)
            mov = depth_map_tools.reject_outliers(movments, 2)
            
            
            points_3d = points_3d[mov]
            points_2d = points_2d[mov]
            
            ref_points_3d = ref_points_3d[mov]
            ref_points_2d = ref_points_2d[mov]
            #print(mov)

    
            #Use SVD to find transformation from one frame to the next
            tranformation_to_ref = depth_map_tools.svd(points_3d, ref_points_3d, True)
            
            if not args.assume_stationary_camera:
                transformed_points_3d = depth_map_tools.transform_points(points_3d, tranformation_to_ref)
                pnpTrans = depth_map_tools.pnpSolve_ransac(transformed_points_3d, ref_points_2d, cam_matrix, refine = True)
                tranformation_to_ref @= pnpTrans
            
            to_ref_zero @= tranformation_to_ref
            transformations.append(to_ref_zero.tolist())
            
            
            if debug_video is not None:
                
                max_nr = np.max(depth_frames[1])
                
                #print("max_nr: ", max_nr)
                
                debug_image = rgb_frames[1]
                
                x = points_2d[:,0]
                y = points_2d[:,1]
                debug_image[y, x] = np.array([255,0,0])
                debug_image[y, x+1] = np.array([255,0,0])
                debug_image[y, x-1] = np.array([255,0,0])
                
                debug_image[y-1, x] = np.array([255,0,0])
                debug_image[y-1, x+1] = np.array([255,0,0])
                debug_image[y-1, x-1] = np.array([255,0,0])
                
                debug_image[y+1, x] = np.array([255,0,0])
                debug_image[y+1, x+1] = np.array([255,0,0])
                debug_image[y+1, x-1] = np.array([255,0,0])
                
                debug_video.write(cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
            
            depth_frames.pop(0)
            if color_video is not None:
                rgb_frames.pop(0)
            
        fr_n += 1
        
        if args.max_frames < fr_n and args.max_frames != -1:
            break
        
    raw_video.release()
    if color_video is not None:
        color_video.release()
        
    if debug_video is not None:
        debug_video.release()
        
    
    
    #ref_mesh Is used to draw the DEBUG alignment window
    #ref_mesh = depth_map_tools.get_mesh_from_depth_map(depth_frames[frame_n], cam_matrix, rgb_frames[frame_n])
    #ref_mesh.paint_uniform_color([0, 0, 1])
    #meshes = [ref_mesh]
    
        
        
    with open(out_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(transformations, cls=NumpyEncoder))