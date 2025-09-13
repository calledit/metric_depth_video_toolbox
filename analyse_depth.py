import numpy as np
import os
import cv2
import json
import argparse
import depth_map_tools
from itertools import islice
import depth_frames_helper


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finds paterns in depth video')

    parser.add_argument('--track_file', type=str, help='file with 2d point tracking data', required=True)
    parser.add_argument('--transformation_file', type=str, help='file with scene transformations from the aligner', required=False)
    parser.add_argument('--depth_video', type=str, help='Dept Video file to analyse', required=True)
    parser.add_argument('--mask_video', type=str, help='black and white mask video for things that should not be tracked', required=False)
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)
    parser.add_argument('--xfov', type=int, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=int, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)

    args = parser.parse_args()

    if args.xfov is None and args.yfov is None:
        print("Either --xfov or --yfov is required.")
        exit(0)

    if not os.path.isfile(args.track_file):
        raise Exception("input track_file does not exist")
        
    if not os.path.isfile(args.depth_video):
        raise Exception("input color_video does not exist")
    
    mask_video = None
    if args.mask_video is not None:
        if not os.path.isfile(args.mask_video):
            raise Exception("input mask_video does not exist")    
        mask_video = cv2.VideoCapture(args.mask_video)
        
    with open(args.track_file) as json_track_file_handle:
        track_frames = json.load(json_track_file_handle)

    transformations = None
    if args.transformation_file is not None:
        if not os.path.isfile(args.transformation_file):
            raise Exception("input transformation_file does not exist")
        with open(args.transformation_file) as json_file_handle:
            transformations = json.load(json_file_handle)
    
    MODEL_maxOUTPUT_depth = args.max_depth
    
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
    
    
    cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, frame_width, frame_height)
    
    for i, frame in enumerate(track_frames):
        track_frames[i] = np.array(track_frames[i])
    
    
    
    used_frames = []
    
    #1. Pick the first frame
    frame_n = 0
    depth_frames = []
    depths = []
    d3_track_frames = []
    
    while raw_video.isOpened():
        
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        ret, raw_frame = raw_video.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        
        # Decode video depth
        depth = depth_frames_helper.decode_rgb_depth_frame(rgb, MODEL_maxOUTPUT_depth, True)
        depth_frames.append(depth)
        
        if mask_video is not None:
            ret, mask = mask_video.read()
            if ret:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            
                rem = []
                rem_global = []
                for i, point in enumerate(track_frames[frame_n]):
                    if mask[point[2], point[1]] > 0:
                        rem.append(i)
                        rem_global.append(point[0])
            
                if len(rem) > 0:
                    track_frames[frame_n] = np.delete(frames[frame_n], rem, axis=0)
            
                if args.strict_mask:
                    for global_id in rem_global:
                        for frame_id, frame in enumerate(track_frames):
                            rem = []
                            for i, point in enumerate(track_frames[frame_n]):
                                if global_id == point[0]:
                                    rem.append(i)
                            if len(rem) > 0:
                                track_frames[frame_id] = np.delete(track_frames[frame_id], rem, axis=0)

        transformation = np.eye(4)
        if transformations is not None:
            transformation = transformations[frame_n]

        transform_to_zero = np.array(transformations[frame_n])

        points_3d_with_id = np.array([])
        if len(track_frames[frame_n]) > 0:
            global_ids = track_frames[frame_n][:, 0]
            points_2d = track_frames[frame_n][:, 1:3]
            points_3d = depth_map_tools.project_2d_points_to_3d(points_2d, depth_frames[-1], cam_matrix)

            dist_cam = np.linalg.norm(points_3d, axis=1)

            points_3d_trans = depth_map_tools.transform_points(points_3d, transform_to_zero) #tracked points in 3d space
            points_3d_with_id = np.hstack((global_ids[:, None], points_3d_trans, dist_cam[:, None]))

        d3_track_frames.append(points_3d_with_id)

        #pcd = depth_map_tools.pts_2_pcd(points_3d_trans)
        #depth_map_tools.draw([pcd])
        
        if len(depth_frames) > 1 and False:
            points = track_frames[frame_n]
            
            ref_frame_no = frame_n - 1
            this_frame_no = frame_n
            best_common_points = list(set(track_frames[ref_frame_no][:, 0]) & set(track_frames[this_frame_no][:, 0]))
            
            
            #Current frame points
            point_ids_in_frame = track_frames[this_frame_no][:,0]
            cur_mask = np.isin(point_ids_in_frame, best_common_points)
            points_2d = track_frames[this_frame_no][cur_mask][:, 1:3]
            dpt_to_points = depth_frames[1][points_2d[:,1].astype(np.int32), points_2d[:,0].astype(np.int32)]
    
            #Ref frame points
            point_ids_in_frame = frames[ref_frame_no][:,0]
            cur_mask = np.isin(point_ids_in_frame, best_common_points)
            ref_points_2d = frames[ref_frame_no][cur_mask][:, 1:3]
            dpt_to_ref_points = depth_frames[0][ref_points_2d[:,1].astype(np.int32), ref_points_2d[:,0].astype(np.int32)]
            
            mean_depth = dpt_to_points.mean()
            std_depth = dpt_to_points.std()
            
            mean_depth_ref = dpt_to_ref_points.mean()
            std_depth_ref = dpt_to_ref_points.std()
            
            
            mean_depth = depth_frames[1].mean()
            std_depth = depth_frames[1].std()
            
            mean_depth_ref = depth_frames[0].mean()
            std_depth_ref = depth_frames[0].std()
            
            
            
            #cur_to_ref_multiplier = std_depth_ref/std_depth
            
            cur_align = mean_depth - mean_depth_ref
            
            #depth_frames[1] *= cur_to_ref_multiplier #This moves the mean in some way that i dont know if it is correct
            depth_frames[1] -= cur_align
            
            depths.append(depth_frames[1])
            
            print("mean_depth_ref:", mean_depth_ref, "std_depth_ref", std_depth_ref, "mean_depth:", mean_depth, "std_depth:", std_depth)
            
            depth_frames.pop(0)
        else:
            depths.append(depth_frames[0])
        
        frame_n += 1

        if args.max_frames < frame_n and args.max_frames != -1:
            break
        
    if raw_video is not None:
        raw_video.release()

    global_points = {}   # gid -> (N,3) XYZ in transformed/world frame
    global_camdist = {}  # gid -> (N,)  camera distance for each sample

    for frame in d3_track_frames:
        if len(frame) == 0:
            continue
        # frame rows: [gid, X, Y, Z, dist_cam]
        gids = frame[:, 0].astype(int)
        xyz  = frame[:, 1:4]
        dcam = frame[:, 4]

        for g, p, d in zip(gids, xyz, dcam):
            if g not in global_points:
                global_points[g] = []
                global_camdist[g] = []
            global_points[g].append(p)
            global_camdist[g].append(d)
    
    # --- 1) First-appearance frame per global_id
    id_first_frame = {}
    for f, arr in enumerate(d3_track_frames):
        if arr is None or len(arr) == 0:
            continue
        for gid in arr[:, 0].astype(int):
            if gid not in id_first_frame:
                id_first_frame[gid] = f


# -------- Reproject per-ID (one projection call per gid) --------
global_points_2d_first = {}     # gid -> (M,2) u,v in its first frame
global_distances_2d_first = {}  # gid -> total 2D path length

def total_path_length_2d(uv: np.ndarray) -> float:
    if not isinstance(uv, np.ndarray) or uv.ndim != 2 or uv.shape[0] < 2:
        return 0.0
    diffs = np.diff(uv, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)/len(diffs)))

for gid, pts0 in global_points.items():
    if gid not in id_first_frame:
        continue
    pts0 = np.asarray(pts0, dtype=float)  # (N,3) in frame-0/world coords
    if pts0.size == 0:
        continue

    f0 = id_first_frame[gid]
    T_0_to_f0 = np.linalg.inv(np.array(transformations[f0], dtype=float))  # 0 -> first-frame

    # Transform ALL samples for this gid into its first frame (vectorized)
    pts0_h = np.hstack([pts0, np.ones((pts0.shape[0], 1), dtype=float)])   # (N,4)
    pf0_h = pts0_h @ T_0_to_f0.T                                           # (N,4)
    pf0 = pf0_h[:, :3]                                                     # (N,3)

    # Keep only forward-facing (positive depth)
    valid = pf0[:, 2] > 1e-6
    if not np.any(valid):
        continue
    pf0_valid = pf0[valid]

    if len(pf0_valid) <= 1:
        continue

    # <<< ONE call per gid >>>
    uvs = depth_map_tools.project_3d_points_to_2d(pf0_valid, cam_matrix)   # (M,2)
    # Drop any non-finite projections
    finite = np.isfinite(uvs).all(axis=1)
    uvs = uvs[finite]
    if uvs.shape[0] == 0:
        continue

    global_points_2d_first[gid] = uvs
    global_distances_2d_first[gid] = total_path_length_2d(uvs)

# -------- Color by z-score of 2D distance, render average 3D position --------
gids_to_render = list(global_points_2d_first.keys())

dists = np.array([global_distances_2d_first[g] for g in gids_to_render], dtype=float)
if dists.size > 0:
    mu, sigma = float(np.nanmean(dists)), float(np.nanstd(dists))

    # dists: np.array of per-gid 2D travel distances (your first-frame reprojection metric)
    # gids_to_render: list of gids aligned with dists

    # --- Choose how to set the inflection point ---
    use_percentile_inflection = True   # set False to use z-score inflection

    if use_percentile_inflection:
        # Inflection at the P-th percentile distance (e.g., 80% -> only top 20% light up strongly)
        P = 75.0
        inflection_dist = np.percentile(dists, P) if dists.size else 0.0
        # scale by robust width (~ how quickly it turns on). Use std or IQR-based width.
        width = np.std(dists) if dists.size else 1.0
        x = (dists - inflection_dist) / (width if width > 0 else 1.0)
    else:
        # Inflection at a z-score (e.g., 1.0σ above mean)
        mu, sigma = float(np.nanmean(dists)) if dists.size else 0.0, float(np.nanstd(dists)) if dists.size else 1.0
        z = (dists - mu) / (sigma if sigma > 0 else 1.0)
        inflection_z = 1.0  # <-- control inflection location in σ units
        x = z - inflection_z

    # --- Sigmoid with adjustable slope (steepness) and output range ---
    slope = 3.0   # higher -> sharper transition around the inflection point
    def logistic(y): return 1.0 / (1.0 + np.exp(-y))

    red = logistic(slope * x)                 # maps to (0,1) around your chosen inflection
    red_floor, red_ceil = 0.0, 1.0            # optional output bounds
    red = red_floor + (red_ceil - red_floor) * red

    # Optional: hard clip low movers completely black until a soft threshold
    black_clip = 0.05                         # set to 0 for no extra clipping
    red[red < black_clip] = 0.0

    #z = (dists - mu) / (sigma if sigma > 0 else 1.0)
    #red = np.clip(z / 2.0, 0.0, 1.0)  # mean -> 0 red, +2σ -> full red
    # Apply gamma to control how fast values rise from black to red
    #gamma = 1.0   # tweak this
    #red = np.power(red, gamma)
else:
    red = np.array([], dtype=float)

avg_positions, colors_list = [], []
for gid, r in zip(gids_to_render, red):
    pts3d = np.asarray(global_points.get(gid, []), dtype=float)
    if pts3d.size == 0:
        continue
    avg3d = np.nanmean(pts3d, axis=0)
    if not np.all(np.isfinite(avg3d)):
        continue
    avg_positions.append(avg3d)
    colors_list.append([r, 0.0, 0.0])

if len(avg_positions) == 0:
    print("No points to render.")
else:
    points_3d = np.vstack(avg_positions).astype(float)
    colors    = np.vstack(colors_list).astype(float)
    pcd = depth_map_tools.pts_2_pcd(points_3d, colors)
    depth_map_tools.draw([pcd])
