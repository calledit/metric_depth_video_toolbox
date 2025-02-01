import torch
import numpy as np
import os
import json
import argparse
import cv2


DEVICE = 'cuda'

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
    parser = argparse.ArgumentParser(description='Generate a json tracking file from a video')

    parser.add_argument('--color_video', type=str, help='video file to use as input', required=True)


    args = parser.parse_args()

    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")

    out_file = args.color_video + "_tracking.json"

    raw_video = cv2.VideoCapture(args.color_video)

    nr_of_tracking_frames = 60#Has to be even #250 works to
    clip1 = []
    clip2 = []
    clip_tracking_points = []
    frame_n = 0
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        print("--- frame ",frame_n+1," ----")
        clip1.append(frame)
        if frame_n >= int(nr_of_tracking_frames/2):
            clip2.append(frame)

        if len(clip1) == nr_of_tracking_frames:
            print("process clip 1")
            clip_tracking_points.append(process_clip(np.array(clip1, dtype=np.int32)))
            clip1 = []
        if len(clip2) == nr_of_tracking_frames:
            print("process clip 2")
            clip_tracking_points.append(process_clip(np.array(clip2, dtype=np.int32)))
            clip2 = []

        frame_n += 1


    first_order = clip2
    second_order = clip1
    if len(clip1) > len(clip2):
        first_order = clip1
        second_order = clip2

    if len(first_order) != 0:
        clip_tracking_points.append(process_clip(np.array(first_order, dtype=np.int32)))
    if len(second_order) != 0:
        clip_tracking_points.append(process_clip(np.array(second_order, dtype=np.int32)))

    track_frames = []
    clip_start = 0
    global_point_id_start = 0
    for clip_id, clip in enumerate(clip_tracking_points):
        for point_id, point in enumerate(clip):
            for frame_id, frame_point in enumerate(point):
                frame_no = clip_start + frame_id
                if frame_no >= len(track_frames):
                    track_frames.append([])
                if frame_point is not None:
                    track_frames[frame_no].append([global_point_id_start+point_id, frame_point[0], frame_point[1]])

        global_point_id_start += len(clip)
        clip_start += int(nr_of_tracking_frames/2)

    with open(out_file, "w") as fp:
        fp.write(json.dumps(track_frames))