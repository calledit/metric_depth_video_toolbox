import torch
import numpy as np
import os
import json
import argparse
import cv2
import gc


DEVICE = 'cuda'

cotracker = None

def convert_to_point_list(point_list, point_visibility, width, height,  global_ids, start_frame, skip_first = False):
    points = []
    save_masks = []
    final_frame_points = []
    final_batch_no = len(point_list)-1
    for batch_no, batch in enumerate(point_list):
        final_frame_no = len(batch)-1
        actual_frame_no = 0
        for frame_no, frame in enumerate(batch):
            if skip_first and frame_no == 0:
                continue
            visibility_mask = point_visibility[batch_no][frame_no]
            for point_id, point in enumerate(frame):
                if point_id >= len(points):
                    points.append([])
                pt = None
                if visibility_mask[point_id] and width > point[0] and 0 < point[0] and height > point[1] and 0 < point[1]:
                    pt = [global_ids[point_id], point[0], point[1], start_frame + actual_frame_no]
                points[point_id].append(pt)

            if final_batch_no == batch_no and final_frame_no == frame_no:
                final_frame_points = []
                for point_id, point in enumerate(frame):
                    if visibility_mask[point_id] and width > point[0] and 0 < point[0] and height > point[1] and 0 < point[1]:
                        final_frame_points.append([global_ids[point_id], point])
            actual_frame_no += 1

    return points, final_frame_points

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

def process_clip(frames, grid_size, global_id_start, start_frame, last_points, iteration, nr_iterations):
    global cotracker
    video = torch.tensor(frames).to(DEVICE).permute(0, 3, 1, 2)[None].float() # B T C H W

    #First we set up a grid of points to track
    grid_egde = 2
    width = frames.shape[2]
    height = frames.shape[1]

    if grid_egde + grid_size > width:
        grid_size = width - grid_egde*2

    if grid_egde + grid_size > height:
        grid_size = height - grid_egde*2

    width_step = (width - (grid_egde*2)) // grid_size
    height_step = (height - (grid_egde*2)) // grid_size

    int(float(width_step)/nr_iterations * iteration)

    x, y = np.meshgrid(np.arange(grid_egde, width-grid_egde, width_step), np.arange(grid_egde, height-grid_egde, height_step))

    print("iteration: ", iteration)
    #We move the points a bit each iteration
    x_per_iteration = float(width_step-1)/nr_iterations
    y_per_iteration = float(height_step-1)/nr_iterations
    x += int(x_per_iteration)*iteration
    y += int(y_per_iteration)*iteration


    track_frame = np.zeros(x.shape)

    track_2d_points = np.stack((track_frame, x, y), axis=-1).reshape(-1, 3)

    mask = (
        (track_2d_points[:, 1] >= 0) & (track_2d_points[:, 1] < width) &
        (track_2d_points[:, 2] >= 0) & (track_2d_points[:, 2] < height)
    )
    track_2d_points = track_2d_points[mask]

    global_ids = []
    for x, pts in enumerate(track_2d_points):
        global_ids.append(global_id_start+x)

    skip_first = False
    if last_points is not None:
        skip_first = True

        # Create an array of indices to keep track of which rows are available for replacement.
        available_indices = np.arange(track_2d_points.shape[0])

        for id_val, tensor_val in last_points:
            # Convert tensor to numpy array (ensuring it is on the CPU)
            x, y = tensor_val

            # Extract only the coordinate columns (columns 1 and 2) from the available rows
            available_coords = track_2d_points[available_indices, 1:3]

            # Calculate Euclidean distances between (x, y) and every available coordinate
            distances = np.linalg.norm(available_coords - np.array([x, y]), axis=1)

            # Find the index of the row with the minimum distance (local index within available_indices)
            local_idx = np.argmin(distances)

            # Map the local index to the actual row index in track_2d_points
            global_idx = available_indices[local_idx]

            global_ids[global_idx] = id_val

            # Replace the entire row with [id, x, y] from last_points
            track_2d_points[global_idx] = np.array([0., x, y])

            # Remove this index from available_indices so it won't be replaced again
            available_indices = np.delete(available_indices, local_idx)


    #Then we filter away points that probably cant be accuratly tracked (like large flat single colord surfaces or the sky)
    first_frame = frames[0].astype(np.uint8)
    mask = mask_from_orb_features(first_frame)

    track_points_x = track_2d_points[:, 1].astype(np.int32)
    track_points_y = track_2d_points[:, 2].astype(np.int32)

    # Check for each point: keep the point if the mask at that (y, x) location is white (>0)
    valid = mask[track_points_y, track_points_x] > 0
    filtered_points = track_2d_points[valid]

    global_idret = global_id_start + len(global_ids)
    #print("global_ids:", global_id_start, "len:", len(global_ids))
    global_ids = np.array(global_ids)[valid]


    if cotracker is None:
        print("load cotracker")
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEVICE)

    queries = torch.tensor(filtered_points, dtype=torch.float32).cuda()


    pred_tracks, pred_visibility = cotracker(video, queries=queries[None]) # B T N 2,  B T N 1

    pred_tracks = pred_tracks.cpu().numpy()
    pred_visibility = pred_visibility.cpu().numpy()

    gc.collect()
    torch.cuda.empty_cache()

    return (*convert_to_point_list(pred_tracks, pred_visibility, width, height, global_ids, start_frame, skip_first), global_idret)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a json tracking file from a video')

    parser.add_argument('--color_video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--downscale', type=int, default=2, help='how much to downscale the frames before tacking, presumably makes tracking faster?', required=False)
    parser.add_argument('--nr_iterations', type=int, default=1, help='how many times to do the tracking more times = more points', required=False)

    args = parser.parse_args()

    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")

    out_file = args.color_video + "_tracking.json"

    raw_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    downscaled_dimensions = (frame_width//args.downscale, frame_height//args.downscale)

    #30x250 works well for long shots on nvidia 3090

    #short: 100 points x 30 frames
    #medium: 46 points x 120 frames
    #long: 30 points x 250 frames

    #Default can be 120, but should let user set this with arg
    nr_of_tracking_frames = 120
    grid_size = np.min([42, downscaled_dimensions[0], downscaled_dimensions[1]])

    clip1 = []
    clip2 = []
    clip_tracking_points = []
    frame_n = 0
    clip1_precursor = 0
    clip2_precursor = 0
    clip1_final_points = None
    clip2_final_points = None
    global_id_start = 0

    total_nr_points = grid_size * grid_size

    clip_frame_start = []

    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, downscaled_dimensions, interpolation=cv2.INTER_AREA)
        print("--- frame ",frame_n+1," ----")
        clip1.append(frame)
        if frame_n >= int(nr_of_tracking_frames/2):
            clip2.append(frame)

        if len(clip1) == nr_of_tracking_frames+clip1_precursor:
            clip_start_frame = frame_n-((len(clip1)-clip1_precursor)-1)
            print("process clip 1:", clip_start_frame)
            for iteration in range(args.nr_iterations):
                pts, clip1_final_points, nr_new_points = process_clip(np.array(clip1, dtype=np.int32), grid_size, global_id_start, clip_start_frame, clip1_final_points, iteration, args.nr_iterations)
                clip_tracking_points.append(pts)
                global_id_start += nr_new_points
            clip1_precursor = 1
            clip1 = [clip1[-1]]
        if len(clip2) == nr_of_tracking_frames+clip2_precursor:
            clip_start_frame = frame_n - ((len(clip2)-clip2_precursor)-1)
            print("process clip 2:", clip_start_frame)
            for iteration in range(args.nr_iterations):
                pts, clip2_final_points, nr_new_points = process_clip(np.array(clip2, dtype=np.int32), grid_size, global_id_start, clip_start_frame, clip2_final_points, iteration, args.nr_iterations)
                clip_tracking_points.append(pts)
                global_id_start += nr_new_points
            clip2_precursor = 1
            clip2 = [clip2[-1]]


        frame_n += 1


    first_order = clip2, clip2_final_points, clip2_precursor
    second_order = clip1, clip1_final_points, clip1_precursor
    if len(clip1) > len(clip2):
        first_order = clip1, clip1_final_points, clip1_precursor
        second_order = clip2, clip2_final_points, clip2_precursor

    if len(first_order[0]) != 0:
        clip_start_frame = (frame_n-1) - ((len(first_order[0])-first_order[2])-1)
        print("process first ordered clip:", clip_start_frame, len(first_order[0]))
        for iteration in range(args.nr_iterations):
            pts, _, nr_new_points = process_clip(np.array(first_order[0], dtype=np.int32), grid_size, global_id_start, clip_start_frame, first_order[1], iteration, args.nr_iterations)
            global_id_start += nr_new_points
            clip_tracking_points.append(pts)

    if len(second_order[0]) != 0:
        clip_start_frame = (frame_n-1) - ((len(second_order[0])-second_order[2])-1)
        print("process second ordered clip:", clip_start_frame, len(second_order[0]))
        for iteration in range(args.nr_iterations):
            pts, _, nr_new_points = process_clip(np.array(second_order[0], dtype=np.int32), grid_size, global_id_start, clip_start_frame, second_order[1], iteration, args.nr_iterations)
            global_id_start += nr_new_points
            clip_tracking_points.append(pts)

    track_frames = []
    for clip_id, clip in enumerate(clip_tracking_points):
        for point_id, point in enumerate(clip):
            for frame_id, frame_point in enumerate(point):
                if frame_point is not None and frame_point[1] < downscaled_dimensions[0] and frame_point[2] < downscaled_dimensions[1] and frame_point[2] >= 0 and frame_point[1] >= 0:
                    frame_no = frame_point[3]
                    while frame_no >= len(track_frames):
                        track_frames.append([])
                    track_frames[frame_no].append([int(frame_point[0]), int(round(frame_point[1] * args.downscale)), int(round(frame_point[2] * args.downscale))])

    with open(out_file, "w") as fp:
        fp.write(json.dumps(track_frames))