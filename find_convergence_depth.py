import numpy as np
import os
import cv2
import json
import argparse
import depth_map_tools
from itertools import islice

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finds convergence depth in depth video')

    parser.add_argument('--depth_video', type=str, help='Dept Video file to analyse', required=True)
    parser.add_argument('--mask_video', type=str, help='black and white mask video for things that should not be tracked', required=False)
    parser.add_argument('--max_depth', default=20, type=int, help='the max depth that the video uses', required=False)

    args = parser.parse_args()

    if not os.path.isfile(args.depth_video):
        raise Exception("input color_video does not exist")

    mask_video = None
    if args.mask_video is not None:
        if not os.path.isfile(args.mask_video):
            raise Exception("input mask_video does not exist")
        mask_video = cv2.VideoCapture(args.mask_video)


    MODEL_maxOUTPUT_depth = args.max_depth

    raw_video = cv2.VideoCapture(args.depth_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)


    out_file = args.depth_video + "_convergence_depths.json"

    frame_n = 0
    convergence_depths = []

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

        if mask_video is not None:
            ret, mask = mask_video.read()
            if ret:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                where_forground = mask > 240

                mesured_pixels = depth[where_forground]
        else:
            mesured_pixels = depth

        if mesured_pixels.size != 0:
            convergence_depth = mesured_pixels.mean()
            convergence_depths.append(float(convergence_depth))

        frame_n += 1

    if raw_video is not None:
        raw_video.release()

    avg_convergence_depth = float(np.array(convergence_depths).mean())
    save_convergence_depths = []
    for i in range(frame_n):
        save_convergence_depths.append(avg_convergence_depth)

    with open(out_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(save_convergence_depths, cls=NumpyEncoder))