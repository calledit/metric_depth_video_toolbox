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
    parser = argparse.ArgumentParser(description='Tries to find video cuts based on tracking data')

    parser.add_argument('--track_file', type=str, help='file with 2d point tracking data', required=True)
    parser.add_argument('--color_video', type=str, help='Video file to extact framerate from', required=False)

    args = parser.parse_args()

    if not os.path.isfile(args.track_file):
        raise Exception("input track_file does not exist")
        
    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")
        
    with open(args.track_file) as json_track_file_handle:
        frames = json.load(json_track_file_handle)
    
    
    raw_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
    
    if raw_video is not None:
        raw_video.release()
    
    
    for i, frame in enumerate(frames):
        frames[i] = np.array(frames[i])
    
    
    
    used_frames = []
    
    #1. Pick the first frame
    frame_n = 0
    
    cut = False
    used_frames.append(frame_n)
    while len(used_frames) < len(frames):
        #2. Find most connected frame (tends to be the next frame)
        best_match_frame_no, best_common_points = find_best_matching_frame(frame_n, frames, used_frames)
        
        #print(best_match_frame_no, len(best_common_points))
        
        report_frame = False
        if not cut and len(best_common_points) < 100 and frame_n > 27*frame_rate:
            print("cut start", "less than 100 matching points")
            cut = True
            report_frame = True
        if cut and len(best_common_points) > 500:
            print("cut end", "more than 500 matching points")
            cut = False
            report_frame = True
            
        if report_frame:
            print("--- frame ", frame_n, (frame_n/frame_rate), " ---")
        
        frame_n = best_match_frame_no
        used_frames.append(frame_n)
        
    