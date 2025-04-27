import numpy as np
import cv2

def encode_depth_as_uint32(depth, max_depth):
    
    depth = np.clip(depth, a_max=max_depth, a_min=0.0)
    encoded_value = (((255**4)/max_depth)*depth.astype(np.float64)).astype(np.uint32)
    
    return encoded_value
    
def decode_uint32_as_depth(encoded_value, max_depth):
    """
    encoded_value: numpy array of dtype uint32 (or scalar)
    MODEL_maxOUTPUT_depth: the same max depth used in the encoder
    
    returns: numpy array of dtype float32 giving depth in metres
    """
    # cast up to float for the division
    e = encoded_value.astype(np.float32)
    depth = e * (max_depth / (255**4))
    return depth

#These values have been picked as they give reaonable resolution at max_depth 100
#The resolution 
C = 2.0
A = 16538.0
    
def encode_depth_as_uint32_log(depth, max_depth):
    depth = np.clip(depth, a_max=max_depth, a_min=0.0)
    encoded_value = np.round(A * np.log1p(depth / C)).astype(np.uint32)
    return encoded_value

def decode_uint32_log_as_depth(encoded_value, max_depth):
    """
    encoded_value: numpy array of dtype uint32 (or scalar)
    returns: numpy array of dtype float32 giving depth in metres
    """
    
    # promote to float for the expm1/division
    e = encoded_value.astype(np.float32)
    # invert the log1p mapping
    depth = C * np.expm1(e / A)
    return depth.astype(np.float32)

def encode_data_as_BGR(data, frame_width, frame_height, bit16 = False):
    # View the uint32 as raw bytes: shape (H, W, 4)
    save_bytes = data.view(np.uint8).reshape(frame_height, frame_width, 4)
    
    if bit16:
        R = (save_bytes[:, :, 3])# if 16 bit Most significant bits in R and G channel (duplicated in 16bit for visulization)
        G = (save_bytes[:, :, 3])
        B = (save_bytes[:, :, 2]) # Least significant bit in blue channel
    else:#24 bif format is absolute or mabye it depends on input format like int32 is one thing sanf float32 another
        R = (save_bytes[:, :, 2])
        G = (save_bytes[:, :, 1])
        B = (save_bytes[:, :, 0])
    
    return np.dstack((B, G, R))

def decode_rgb_as_data(rgb, frame_width, frame_height, bit16 = False):
    # View the uint32 as raw bytes: shape (H, W, 4)
    data = np.zeros((frame_height, frame_width), dtype=np.uint32)
    depth_unit = data.view(np.uint8).reshape((frame_height, frame_width, 4))
    if bit16:
        depth_unit[..., 3] = rgb[..., 0]
        depth_unit[..., 2] = rgb[..., 2]
    else:
        depth_unit[..., 0] = rgb[..., 2]
        depth_unit[..., 1] = rgb[..., 0]
        depth_unit[..., 2] = rgb[..., 1]
    
    return data

def decode_rgb_depth_frame(rgb, max_depth, bit16):
    frame_height = rgb.shape[0]
    frame_width = rgb.shape[1]
    encoded_value = decode_rgb_as_data(rgb, frame_width, frame_height, bit16)
    return decode_uint32_as_depth(encoded_value, max_depth)

def save_depth_video(frames, output_video_path, fps, max_depth_arg, rescale_width, rescale_height):
    """
    Saves depth maps encoded in the R, G and B channels of a video (to increse accuracy as when compared to gray scale)
    """


    MODEL_maxOUTPUT_depth = max_depth_arg ### pick a value slitght above max metric depth to save the depth in th video file nicly
    # if you pick a high value you will lose resolution

    if isinstance(frames, np.ndarray):
        height = frames.shape[1]
        width = frames.shape[2]
        max_depth = frames.max()
        print("max metric depth: ", max_depth)
        # incase you did not pick a absolute value we max out (this mean each video will have depth relative to max_depth)
        # (if you want to use the video as a depth souce a absolute value is prefrable)
        if MODEL_maxOUTPUT_depth < max_depth:
            print("warning: output depth is deeper than max_depth. The depth will be clipped")
        nr_frames = frames.shape[0]
    else:
        nr_frames = len(frames)
        height = frames[0].shape[0]
        width = frames[0].shape[1]

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (rescale_width, rescale_height))

    for i in range(nr_frames):
        if rescale_width != width or rescale_height != height:
            depth = cv2.resize(frames[i], (rescale_width, rescale_height), interpolation=cv2.INTER_LINEAR)
        else:
            depth = frames[i]
        
        encoded_depth = encode_depth_as_uint32(depth, MODEL_maxOUTPUT_depth)
        bgr24bit = encode_data_as_BGR(encoded_depth, rescale_width, rescale_height, bit16 = True)
        out.write(bgr24bit)

    out.release()