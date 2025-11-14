import torch
import numpy as np
import sys
sys.path.append("Depth-Anything-3/src")
from depth_anything_3.api import DepthAnything3
import cv2
import depth_frames_helper
import depth_map_tools
import argparse
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDVT UniK3D video converter')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--xfov', type=float, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=float, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument("--transformation_file", type=str, default=None)
    parser.add_argument('--xfov_file', type=str, help='alternative to xfov and yfov, json file with one xfov for each frame', required=False)

    args = parser.parse_args()


    extrinsics = None
    intrinsics = None

    use_fov = True
    if args.xfov is None and args.yfov is None:
        use_fov = False
    
    xfovs = None
    if args.xfov_file is not None:
        if not os.path.isfile(args.xfov_file):
            raise Exception("input xfov_file does not exist")
        with open(args.xfov_file) as json_file_handle:
            xfovs = json.load(json_file_handle)
        use_fov = True

    # Run inference on images

    output_tmp_video_path = args.color_video+'_tmp_depth.mkv'
    output_video_path = args.color_video+'_depth.mkv'
    out_xfov_file = output_video_path + "_xfovs.json"
    out_transformations_file = output_video_path + "_transformations.json"
    
    images, fps = depth_frames_helper.load_video_frames_from_path(args.color_video, max_frames = args.max_frames)  # List of image paths, PIL Images, or numpy arrays
    
    nr_images = len(images)
    H = images[0].shape[0]
    W = images[0].shape[1]
    
    #get xfovs from xfov or yfov
    if use_fov and xfovs is None:
        intrinsics = []
        fovx, fovy = fov_from_camera_matrix(cam_matrix)
        for i in range(nr_images):
            cam_matrix = depth_map_tools.compute_camera_matrix(args.xfov, args.yfov, W, H).astype(np.float32)
            intrinsics.append(cam_matrix)

    #convert xfovs to intrinsics
    if use_fov and xfovs is not None and intrinsics is None:
        intrinsics = []
        for i in range(nr_images):
            xfov = xfovs[i]
            cam_matrix = compute_camera_matrix(xfov, None, original_width, original_height).astype(np.float32)
            intrinsics.append(cam_matrix)
      
    
    if args.transformation_file is not None and intrinsics is not None:
        if not os.path.isfile(args.transformation_file):
            raise Exception("input transformation_file does not exist")
        with open(args.transformation_file) as json_file_handle:
            transformations = json.load(json_file_handle)
        extrinsics = []
        for i in range(nr_images):
            extrinsics.append(np.array(transformations[i]).astype(np.float32))
           
     
    
    
    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
    model = model.to(device=device)
    
    
    prediction = model.inference(
        images,
        extrinsics = extrinsics,
        intrinsics = intrinsics,
    )

    depth_frames_helper.save_depth_video(prediction.depth, output_tmp_video_path, fps, args.max_depth, W, H)
    depth_frames_helper.verify_and_move(output_tmp_video_path, len(prediction.depth), output_video_path)

    out_xfovs = []
    for intrin in prediction.intrinsics:
        fovx, fovy = depth_map_tools.fov_from_camera_matrix(intrin)
        out_xfovs.append(float(fovx))
        
    with open(out_xfov_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(out_xfovs))
    
    out_transformations = []
    for extrin in prediction.extrinsics:
        fixed_extrin = np.vstack([extrin, np.array([0, 0, 0, 1], dtype=extrin.dtype)])
        fixed_extrin = np.linalg.inv(fixed_extrin)#da3 outputs inverted transformations to waht we want
        out_transformations.append(fixed_extrin)
    
    with open(out_transformations_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(out_transformations, cls=NumpyEncoder)) 
    
    
    #DEBUG grayscale out
    max_depth = np.max(prediction.depth)
    depth_frames_helper.save_grayscale_video(prediction.depth, output_video_path+"_grayscale_depth.mkv", fps, max_depth, W, H) # prediction.depth.shape[2], prediction.depth.shape[1]
    
    
    # Access results
    #print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
    #print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
    #print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
    #print(prediction.intrinsics[0])   # Camera intrinsics: [N, 3, 3] float32

    #print(prediction.extrinsics[0])
