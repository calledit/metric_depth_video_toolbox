import argparse
import cv2
import numpy as np
import os
import sys

np.set_printoptions(suppress=True, precision=4)


def float_image_to_byte_image(float_image, max_value=10.0, scale=255, log_scale=5):
    # Ensure that no values are below a very small positive number to avoid log(0)
    epsilon = 0.0001
    float_image = np.clip(float_image, epsilon, max_value)
    
    # Apply logarithmic scaling
    transformed = np.log(float_image * log_scale + 1)
    max_log = np.log(max_value * log_scale + 1)
    
    # Normalize to fit into 0-255
    normalized = transformed / max_log * scale
    
    # Convert to integers and clip to ensure values stay in the 0-255 range
    byte_image = np.clip(normalized, 0, scale).astype(np.uint8)
    
    return byte_image


if __name__ == '__main__':
    
    # Setup arguments
    parser = argparse.ArgumentParser(description='Generate a depth video in greyscale from a rgb encoded depth video')
    
    parser.add_argument('--video', type=str, help='video file to use as input', required=True)
    parser.add_argument('--bit16', action='store_true', help='Store outut as 16bit file', required=False)
    parser.add_argument('--max_depth', default=6, type=int, help='the max depth that the video uses', required=False)
    
    
    
    args = parser.parse_args()
    
   
    MODEL_maxOUTPUT_depth = args.max_depth
    
    # Verify input file exists
    if not os.path.isfile(args.video):
        raise Exception("input video does not exist")

    # the Touchly1 format has the depth video underneeth the normal video
    output_file = args.video + "_grey_depth.mkv"
    
    raw_video = cv2.VideoCapture(args.video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
    out = None
    if args.bit16:
        out = cv2.VideoWriter(
            filename=output_file,
            apiPreference=cv2.CAP_FFMPEG,
            fourcc=cv2.VideoWriter_fourcc(*"FFV1"),
            fps=frame_rate,
            frameSize=(frame_width, frame_height),
            params=[
                cv2.VIDEOWRITER_PROP_DEPTH,
                cv2.CV_16U,
                cv2.VIDEOWRITER_PROP_IS_COLOR,
                0,  # false
            ],
        )
    else:
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"FFV1"), frame_rate, (frame_width, frame_height))

    
    frame_n = 0
    while raw_video.isOpened():
        
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        frame_n += 1
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            
        depth = np.zeros((frame_height, frame_width), dtype=np.uint32)
        
        depth_unit = depth.view(np.uint8).reshape((frame_height, frame_width, 4))
        
        depth_unit[..., 3] = ((rgb[..., 0].astype(np.uint32) + rgb[..., 1]).astype(np.uint32) / 2)
        depth_unit[..., 2] = rgb[..., 2]
        
        
        depth = depth.astype(np.float32)/((255**4)/MODEL_maxOUTPUT_depth)
        
        
        if args.bit16:
            depth = depth*((255**2)/MODEL_maxOUTPUT_depth)
            depth = np.rint(depth).astype(np.uint16)
            vid_depth = depth
        else:
            depth = depth*(255/MODEL_maxOUTPUT_depth)
            depth = np.rint(depth).astype(np.uint8)
            vid_depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        
        
        out.write(vid_depth)
        
    raw_video.release()
    out.release()
