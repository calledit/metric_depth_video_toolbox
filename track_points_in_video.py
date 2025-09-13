import torch
import numpy as np
import os
import json
import argparse
import cv2
import gc
import random


def visualize_mask(image, mask):
    """
    Shows orb image, mask, and overlay side by side.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    # Ensure mask is single-channel uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Make overlay (red mask on top of original image)
    overlay = image.copy()
    overlay[mask > 0] = (0, 0, 255)  # red where mask is nonzero

    # Show side by side
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()



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

def mask_from_orb_features(image, size=10, mask=None):
    """
    Detects ORB features and creates a image mask from that
    """
    # Create an ORB detector instance
    orb = cv2.ORB_create(nfeatures=9000000, edgeThreshold=2, patchSize=2, fastThreshold=2)

    # Detect keypoints and compute descriptors using the mask if provided
    keypoints, descriptors = orb.detectAndCompute(image, mask=mask)

    keypoint_mask = create_keypoint_mask(image, keypoints, radius=int(size/2))

    blurred_mask = cv2.GaussianBlur(keypoint_mask, (9, 9), 0)

    ret, keypoint_mask = cv2.threshold(blurred_mask, 15, 255, cv2.THRESH_BINARY)

    #visualize_mask(image, keypoint_mask)

    return keypoint_mask

def generate_grid(width, height, grid_edge, width_step, height_step, iteration, nr_iterations):
    """
    Generate a grid of points with a square spiral offset that uses separate maximum offsets for x and y.

    At iteration 0, the offset is (0, 0), so you get the original grid.
    Over iterations, the grid is shifted along a square path with:
      - max offset for x: width_step/2 - 1
      - max offset for y: height_step/2 - 1

    Parameters:
      width, height: dimensions of the area.
      grid_edge: distance from the edge to start the grid.
      width_step, height_step: spacing between grid points.
      iteration: current iteration (0-indexed).
      nr_iterations: total number of iterations.

    Returns:
      track_2d_points: an array of shape (-1, 3) with columns [frame, x, y].
    """
    # Create the base grid.
    x, y = np.meshgrid(
        np.arange(grid_edge, width - grid_edge, width_step),
        np.arange(grid_edge, height - grid_edge, height_step)
    )

    # Define maximum offsets for x and y.
    max_offset_x = width_step / 2
    max_offset_y = height_step / 2

    # Compute fraction f that scales from 0 (iteration 0) to 1 (final iteration)
    f = iteration / (max(nr_iterations, 2) - 1)
    
    random.seed(nr_iterations^iteration)
    
    random_x = (random.random()*2) - 1
    random_y = (random.random()*2) - 1
    
    #Zero offset on first frame
    if iteration == 0:
        random_x = 0
        random_y = 0

    # Current amplitude for each axis.
    offset_x = random_x * max_offset_x 
    offset_y = random_y * max_offset_y

    # Apply the computed square spiral offset to the entire grid.
    x_offset = x + offset_x
    y_offset = y + offset_y

    # Create a frame column (set to zeros in this example)
    track_frame = np.zeros_like(x)
    track_2d_points = np.stack((track_frame, x_offset, y_offset), axis=-1).reshape(-1, 3)

    # Filter out any points outside the boundaries.
    mask = (
        (track_2d_points[:, 1] >= 0) & (track_2d_points[:, 1] < width) &
        (track_2d_points[:, 2] >= 0) & (track_2d_points[:, 2] < height)
    )
    return track_2d_points[mask]

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

    print("iteration:", iteration)

    track_2d_points = generate_grid(width, height, grid_egde, width_step, height_step, iteration, nr_iterations)

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


    #Then we filter away points that probably cant be accuratly tracked (like large flat single colord surfaces or the sky) cotracker will try to track these but it does not do a good joob
    first_frame = frames[0].astype(np.uint8)

    #here we create a mask that is 0 where there are no features and 255 in areas where there are features
    mask = mask_from_orb_features(first_frame, size=(width_step+height_step)/2)

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

    # Memory allocated by tensors (in bytes)
    #allocated = torch.cuda.memory_allocated()
    #print(f"Allocated: {allocated / (1024**2):.2f} MB")

    # Memory reserved by the caching allocator (in bytes)
    #reserved = torch.cuda.memory_reserved()
    #print(f"Reserved: {reserved / (1024**2):.2f} MB")

    gc.collect()
    torch.cuda.empty_cache()

    return (*convert_to_point_list(pred_tracks, pred_visibility, width, height, global_ids, start_frame, skip_first), global_idret)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a json tracking file from a video')

    parser.add_argument('--color_video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--downscale', type=int, default=1, help='how much to downscale the frames before tacking, presumably makes tracking faster?', required=False)
    parser.add_argument('--nr_iterations', type=int, default=1, help='how many times to do the tracking more times = more points', required=False)
    parser.add_argument('--steps_bewtwen_track_init', type=int, default=60, help='how often to seek for new tracking points in nr of frames', required=False)
    parser.add_argument('--save_visulization_video', action='store_true', help='Save a video with the tracking points visualised', required=False)
    

    args = parser.parse_args()

    if not os.path.isfile(args.color_video):
        raise Exception("input color_video does not exist")

    out_file = args.color_video + "_tracking.json"

    raw_video = cv2.VideoCapture(args.color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

    downscaled_dimensions = (frame_width//args.downscale, frame_height//args.downscale)
    visualization_video = None
    visualization_frames = []
    if args.save_visulization_video:
        visualization_video = cv2.VideoWriter(args.color_video+"_track_visualization.mkv", cv2.VideoWriter_fourcc(*"FFV1"), frame_rate, (frame_width, frame_height))

    #30x250 works well for long shots on nvidia 3090

    #short: 100 points x 30 frames
    #medium: 46 points x 120 frames
    #long: 30 points x 250 frames

    #TIP reduce these if you are running you of memmory
    nr_of_tracking_frames = 120
    grid_size = np.min([36, downscaled_dimensions[0], downscaled_dimensions[1]])

    clip1 = []
    clip2 = []
    clip_tracking_points = []
    frame_n = 0
    clip1_precursor = 0
    clip2_precursor = 0
    clip1_final_points = None
    clip2_final_points = None
    global_id_start = 0


    last_points = {}
    nr_overlaps = nr_of_tracking_frames/args.steps_bewtwen_track_init
    
    assert nr_overlaps % 2 == 0, f"steps_bewtwen_track_init must evenly devide nr_of_tracking_frames"
    
    nr_overlaps = int(nr_overlaps)
    
    nr_overlaps = max(2, nr_overlaps) #never less than 2 overlaps
    
    steps_betwen_overlaps = nr_of_tracking_frames//nr_overlaps
    
    overlaps_offset = [0]
    
    for x in range(1, nr_overlaps):
        overlaps_offset.append(overlaps_offset[-1] + steps_betwen_overlaps)
    
    act_clips = {}
    for offset in overlaps_offset:
        act_clips[offset] = []
    
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        if visualization_video is not None:
            visualization_frames.append(frame)
        frame = cv2.resize(frame, downscaled_dimensions, interpolation=cv2.INTER_AREA)
        print("--- frame ",frame_n+1," ----")
        
            
        for offset in overlaps_offset:
            extra_overlap_frames = 1
            if frame_n <= offset+nr_of_tracking_frames:
                extra_overlap_frames = 0
            
            if frame_n >= offset:
                act_clips[offset].append(frame)

            if len(act_clips[offset]) == nr_of_tracking_frames+extra_overlap_frames:
                clip_start_frame = frame_n-((len(act_clips[offset])-extra_overlap_frames)-1)
                clip_nextstart_frame = clip_start_frame + nr_of_tracking_frames
                print("process clip with offset", offset,":", clip_start_frame)
                for iteration in range(args.nr_iterations):
                    if clip_start_frame not in last_points:
                        last_points[clip_start_frame] = []
                    if len(last_points[clip_start_frame]) >= iteration:
                        last_points[clip_start_frame].append(None)
                    pts, clip_final_points, global_id_start = process_clip(np.array(act_clips[offset], dtype=np.int32), grid_size, global_id_start, clip_start_frame, last_points[clip_start_frame][iteration], iteration, args.nr_iterations)
                    if clip_nextstart_frame not in last_points:
                        last_points[clip_nextstart_frame] = []
                    last_points[clip_nextstart_frame].append(clip_final_points)
                    clip_tracking_points.append(pts)
                act_clips[offset] = [act_clips[offset][-1]]


        frame_n += 1

    for offset in overlaps_offset:
        extra_overlap_frames = 1
        if frame_n <= offset+nr_of_tracking_frames:
            extra_overlap_frames = 0

        clip_start_frame = (frame_n-1) - ((len(act_clips[offset])-extra_overlap_frames)-1)
        clip_nextstart_frame = clip_start_frame + nr_of_tracking_frames
        print("process final clip with offset", offset,":", clip_start_frame)
        for iteration in range(args.nr_iterations):
            if clip_start_frame not in last_points:
                last_points[clip_start_frame] = []
            if len(last_points[clip_start_frame]) >= iteration:
                last_points[clip_start_frame].append(None)
            pts, clip_final_points, global_id_start = process_clip(np.array(act_clips[offset], dtype=np.int32), grid_size, global_id_start, clip_start_frame, last_points[clip_start_frame][iteration], iteration, args.nr_iterations)
            if clip_nextstart_frame not in last_points:
                last_points[clip_nextstart_frame] = []
            last_points[clip_nextstart_frame].append(clip_final_points)
            clip_tracking_points.append(pts)
        act_clips[offset] = [act_clips[offset][-1]]
            
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

    if visualization_video is not None:
        print("Creating visualization video")
        for frame_no, frame_points in enumerate(track_frames):
            image = visualization_frames[frame_no]
            if len(frame_points) > 0:
                points_2d = np.array(frame_points)
                x = np.clip(points_2d[:,1].astype(np.int32), 0, frame_width-2)
                y = np.clip(points_2d[:,2].astype(np.int32), 0, frame_height-2)
                red = np.array([255,0,0])
                image[y, x] = red
                image[y, x+1] = red
                image[y, x-1] = red

                image[y-1, x] = red
                image[y-1, x+1] = red
                image[y-1, x-1] = red

                image[y+1, x] = red
                image[y+1, x+1] = red
                image[y+1, x-1] = red

            visualization_video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        visualization_video.release()
