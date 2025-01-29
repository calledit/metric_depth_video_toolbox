import torch
import numpy as np
import os
import json
import argparse

import imageio.v3 as iio

DEVICE = 'cuda'

#pip install 'imageio[ffmpeg]'

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

    frames = iio.imread(args.color_video, plugin="FFMPEG")  # plugin="pyav"
    nr_of_tracking_frames = 60
    clip1 = []
    clip2 = []
    clip_tracking_points = []
    frame_n = 0
    for frame in frames:
        print("--- frame ",frame_n+1," ----")
        clip1.append(frame)
        if frame_n >= (nr_of_tracking_frames/2):
            clip2.append(frame)

        if len(clip1) == nr_of_tracking_frames:
            print("process clip 1")
            clip_tracking_points.append(process_clip(np.array(clip1)))
            clip1 = []
        if len(clip2) == nr_of_tracking_frames:
            print("process clip 2")
            clip_tracking_points.append(process_clip(np.array(clip2)))
            clip2 = []

        frame_n += 1


    if len(clip1) != 0:
        clip_tracking_points.append(process_clip(np.array(clip1)))
    if len(clip2) < (nr_of_tracking_frames/2):
        clip_tracking_points.append(process_clip(np.array(clip2)))

    fp = open(out_file, "w")
    fp.write(json.dumps(clip_tracking_points))
    fp.close()