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
                    pt = [int(torch.round(point[0])), int(torch.round(point[1]))]
                points[point_id].append(pt)

    for point in points:
        nr_none = 0
        for frame_point in point:
            if frame_point is None:
                nr_none += 1
        #print(len(points), "missing:", nr_none)
    return points
    
def create_keypoint_mask(image, keypoints, radius=2):
    """
    Creates a binary mask from the given image and keypoints.
    All pixels within a circle of given radius around each keypoint are set to white (255),
    and all other pixels are black (0).
    
    Args:
        image (numpy.ndarray): The original image (used only to determine the shape).
        keypoints (list): List of keypoints (each with a .pt attribute).
        radius (int): Radius (in pixels) around each keypoint to set as white.
        
    Returns:
        mask (numpy.ndarray): A binary mask image.
    """
    # Create a black mask with the same height and width as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # For each keypoint, draw a filled circle (white) on the mask.
    for kp in keypoints:
        # Extract the keypoint coordinates and round them to the nearest integer.
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(mask, (x, y), radius, color=255, thickness=-1)
    
    return mask

def mask_from_orb_features(image, mask=None):
    """
    Detects ORB features and creates a image mask from that
    """
    # Create an ORB detector instance
    orb = cv2.ORB_create(nfeatures=9000000, edgeThreshold=2, patchSize=2, fastThreshold=2)

    # Detect keypoints and compute descriptors using the mask if provided
    keypoints, descriptors = orb.detectAndCompute(image, mask=mask)

    keypoint_mask = create_keypoint_mask(image, keypoints, radius=5)
    
    blurred_mask = cv2.GaussianBlur(keypoint_mask, (9, 9), 0)
    
    ret, keypoint_mask = cv2.threshold(blurred_mask, 15, 255, cv2.THRESH_BINARY)
    
    return keypoint_mask

def process_clip(frames):
    global cotracker
    grid_size = 30
    video = torch.tensor(frames).to(DEVICE).permute(0, 3, 1, 2)[None].float() # B T C H W
    
    #First we set up a grid of points to track
    grid_egde = 2
    width = frames.shape[2]
    height = frames.shape[1]
    
    width_step = (width - (grid_egde*2)) // grid_size
    height_step = (height - (grid_egde*2)) // grid_size
    
    x, y = np.meshgrid(np.arange(grid_egde, width-grid_egde, width_step), np.arange(grid_egde, height-grid_egde, height_step))
    track_frame = np.zeros(x.shape)
    
    track_2d_points = np.stack((track_frame, x, y), axis=-1).reshape(-1, 3)
    
    #Then we filter away points that probably cant be accuratly tracked (like large flat single colord surfaces or the sky)
    first_frame = frames[0].astype(np.uint8)
    mask = mask_from_orb_features(first_frame)
    
    track_points_x = track_2d_points[:, 1].astype(np.int32)
    track_points_y = track_2d_points[:, 2].astype(np.int32)

    # Check for each point: keep the point if the mask at that (y, x) location is white (>0)
    valid = mask[track_points_y, track_points_x] > 0
    filtered_points = track_2d_points[valid]

    if cotracker is None:
        print("load cotracker")
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEVICE)
        
    queries = torch.tensor(filtered_points, dtype=torch.float32).cuda()

    pred_tracks, pred_visibility = cotracker(video, queries=queries[None]) # B T N 2,  B T N 1
    return convert_to_point_list(pred_tracks, pred_visibility)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a json tracking file from a video')

    parser.add_argument('--color_video', type=str, help='video file to use as input', required=True)
    
    args = parser.parse_args()

    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")

    out_file = args.color_video + "_tracking.json"

    raw_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
                if frame_point is not None and frame_point[0] < frame_width and frame_point[1] < frame_height and frame_point[1] >= 0 and frame_point[0] >= 0:
                    track_frames[frame_no].append([global_point_id_start+point_id, frame_point[0], frame_point[1]])

        global_point_id_start += len(clip)
        clip_start += int(nr_of_tracking_frames/2)

    with open(out_file, "w") as fp:
        fp.write(json.dumps(track_frames))