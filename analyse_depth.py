import numpy as np
import os
import cv2
import json
import argparse
import depth_map_tools
from itertools import islice

def save_24bit(frames, output_video_path, fps, max_depth_arg):
    """
    Saves depth maps encoded in the R, G and B channels of a video (to increse accuracy as when compared to gray scale)
    """
    height = frames.shape[1]
    width = frames.shape[2]

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))

    max_depth = frames.max()
    print("max metric depth: ", max_depth)

    MODEL_maxOUTPUT_depth = max_depth_arg ### pick a value slitght above max metric depth to save the depth in th video file nicly
    # if you pick a high value you will lose resolution

    # incase you did not pick a absolute value we max out (this mean each video will have depth relative to max_depth)
    # (if you want to use the video as a depth souce a absolute value is prefrable)
    if MODEL_maxOUTPUT_depth < max_depth:
        print("warning: output depth is deeper than max_depth. The depth will be clipped")

    for i in range(frames.shape[0]):
        depth = frames[i]
        scaled_depth = (((255**4)/MODEL_maxOUTPUT_depth)*depth.astype(np.float64)).astype(np.uint32)

        # View the depth as raw bytes: shape (H, W, 4)
        depth_bytes = scaled_depth.view(np.uint8).reshape(height, width, 4)


        R = (depth_bytes[:, :, 3]) # Most significant bits in R and G channel (duplicated to reduce compression artifacts)
        G = (depth_bytes[:, :, 3])
        B = (depth_bytes[:, :, 2]) # Least significant bit in blue channel
        bgr24bit = np.dstack((B, G, R))
        out.write(bgr24bit)

    out.release()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finds paterns in depth video')

    parser.add_argument('--track_file', type=str, help='file with 2d point tracking data', required=True)
    parser.add_argument('--depth_video', type=str, help='Dept Video file to analyse', required=True)
    parser.add_argument('--mask_video', type=str, help='black and white mask video for things that should not be tracked', required=False)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)

    args = parser.parse_args()

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
        frames = json.load(json_track_file_handle)
    
    MODEL_maxOUTPUT_depth = args.max_depth
    
    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
    
    
    
    for i, frame in enumerate(frames):
        frames[i] = np.array(frames[i])
    
    
    
    used_frames = []
    
    #1. Pick the first frame
    frame_n = 0
    depth_frames = []
    depths = []
    
    while raw_video.isOpened():
        
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        ret, raw_frame = raw_video.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        
        # Decode video depth
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        depth_unit[..., 3] = ((rgb[..., 0].astype(np.uint32) + rgb[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb[..., 2]
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        depth_frames.append(depth)
        
        if mask_video is not None:
            ret, mask = mask_video.read()
            if ret:
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
        
        if len(depth_frames) > 1:
            points = frames[frame_n]
            
            ref_frame_no = frame_n - 1
            this_frame_no = frame_n
            best_common_points = list(set(frames[ref_frame_no][:, 0]) & set(frames[this_frame_no][:, 0]))
            
            
            #Current frame points
            point_ids_in_frame = frames[this_frame_no][:,0]
            cur_mask = np.isin(point_ids_in_frame, best_common_points)
            points_2d = frames[this_frame_no][cur_mask][:, 1:3]
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
        
    if raw_video is not None:
        raw_video.release()
        
    output_video_path = args.depth_video+'_corrected.mkv'
    save_24bit(np.array(depths), output_video_path, frame_rate, args.max_depth)