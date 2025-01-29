import numpy as np
import os
import json
import argparse



cotracker = None

def convert_to_point_list(point_list, point_visibility):
    points = []
    save_masks = []
    for batch_no, batch in enumerate(point_list):
        for frame_no, frame in enumerate(batch):
            visibility_mask = point_visibility[batch_no][frame_no]
            for point_id, point in enumerate(frame):
                if point_id >= len(points):
                    points.append([])
                pt = None
                if visibility_mask[point_id]:
                    pt = [int(point[0]), int(point[1])]
                points[point_id].append(pt)

    for point in points:
        nr_none = 0
        for frame_point in point:
            if frame_point is None:
                nr_none += 1
        #print(len(points), "missing:", nr_none)
    return points


def process_clip(frames):
    global cotracker
    grid_size = 20
    video = torch.tensor(frames).to(DEVICE).permute(0, 3, 1, 2)[None].float()  # B T C H W

    if cotracker is None:
        print("load cotracker")
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEVICE)

    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size, backward_tracking=True) # B T N 2,  B T N 1
    return convert_to_point_list(pred_tracks, pred_visibility)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align 3D video based on depth video and a point tracking file')

    parser.add_argument('--track_file', type=str, help='file with 2d point tracking data', required=True)
    
    parser.add_argument('--depth_video', type=str, help='depth video', required=False)# Set to required


    args = parser.parse_args()

    if not os.path.isfile(args.track_file):
        raise Exception("input track_file does not exist")
        
    #if not os.path.isfile(args.depth_video):
    #    raise Exception("input depth_video does not exist")
    
    with open(args.track_file) as json_track_file_handle:
        tracking_points = json.load(json_track_file_handle)
    
    frames = []
    clip_start = 0
    global_point_id_start = 0
    for clip_id, clip in enumerate(tracking_points):
        if clip_id % 2 == 0:
            print("clip1")
        else:
            print("clip2")
        
        for point_id, point in enumerate(clip):
            for frame_id, frame_point in enumerate(point):
                frame_no = clip_start + frame_id
                if frame_no >= len(frames):
                    frames.append([])
                if frame_point is not None:
                    frames[frame_no].append([global_point_id_start+point_id, frame_point[0], frame_point[1]])
        
        global_point_id_start += len(clip)
        clip_start += 30
        
    print(frames)