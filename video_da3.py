import torch
import numpy as np
import sys
sys.path.append("Depth-Anything-3/src")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.pose_align import align_poses_umeyama, _apply_sim3_to_poses
from depth_anything_3.utils.alignment import least_squares_scale_scalar
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


def _is_txt(path):
    return isinstance(path, str) and path.lower().endswith(".txt")

def _read_list_file(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items

model = None

def do_batch(list_of_ids, intrinsics, extrinsics, images, resolution):
    
    picked_imgs = []
    picked_extrs = []
    picked_intrs = []

    for i in list_of_ids:
        picked_imgs.append(images[i])
        if extrinsics is not None:
            picked_imgs.append(extrinsics[i])
        if intrinsics is not None:
            picked_imgs.append(intrinsics[i])
    
    if extrinsics is None:
        picked_extrs = None
    
    if intrinsics is None:
        picked_intrs = None
        
    prediction = model.inference(
        picked_imgs,
        extrinsics = picked_extrs,
        intrinsics = picked_intrs,
        process_res = resolution
    )
    return prediction


    
def _single_run(args, color_video_path):

    extrinsics = None
    intrinsics = None

    
    output_tmp_video_path = color_video_path+'_tmp_depth.mkv'
    output_video_path = color_video_path+'_depth.mkv'
    out_xfov_file = output_video_path + "_xfovs.json"
    out_transformations_file = output_video_path + "_transformations.json"
    
    #Load Video in to ram
    images, fps = depth_frames_helper.load_video_frames_from_path(color_video_path, max_frames = args.max_frames)  # List of image paths, PIL Images, or numpy arrays
    
    nr_images = len(images)
    H = images[0].shape[0]
    W = images[0].shape[1]
    
    # handle field of fiew arguments
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
    
    
    #Load cammera transformation input file
    if args.transformation_file is not None and intrinsics is not None:
        if not os.path.isfile(args.transformation_file):
            raise Exception("input transformation_file does not exist")
        with open(args.transformation_file) as json_file_handle:
            transformations = json.load(json_file_handle)
        extrinsics = []
        for i in range(nr_images):
            extrinsics.append(np.array(transformations[i]).astype(np.float32))
           
     
    # select reference frames
    reference_frame_ids = []
    if args.nr_of_ref_frames != 0:
        nth_frame = nr_images//args.nr_of_ref_frames
        
        for i in range(nr_images):
            if i % nth_frame == 0:
                reference_frame_ids.append(i)
    numer_of_refs = len(reference_frame_ids)
    
    # Create batches
    images_per_batch = args.images_per_batch - (args.nr_of_ref_frames+args.batch_overlap)
    batches = []
    for i in range(nr_images):
        if i % args.images_per_batch == 0:
            batches.append([])
        batches[-1].append(i)
    
    
    depth_out = []
    intrin_out = []
    extrin_out = []
    
    alingnment_depths = None
    alingnment_extrinsics = None
    last_frame = None
    last_frame_tranform = None
    last_frame_depth = None
    
    # do all batches on edy one
    for batch in batches:
        nr_in_batch = len(batch)
        if nr_in_batch != 0:
            to_batch = reference_frame_ids + []
            
            nr_used_refs = numer_of_refs
            # add last_frame from the last batch as reference frames in this batch
            if last_frame is not None:
                to_batch.extend(last_frame)
                nr_used_refs = len(to_batch)
                
            to_batch.extend(batch)
            
            #Compute the batch
            prediction = do_batch(to_batch, intrinsics, extrinsics, images, args.da3_resolution)
            
            ref_depths = torch.as_tensor(prediction.depth[:nr_used_refs])
            
            if alingnment_depths is None:
                alingnment_depths = ref_depths
            
            
            #Rescale output to match last batch
            if last_frame_depth is not None:
                batch_alignment_depths = torch.cat(
                    [alingnment_depths, last_frame_depth],
                    dim=0
                )
                scale_factor = least_squares_scale_scalar(batch_alignment_depths, ref_depths)
                
                #We asume that any differance in depth between batches depends on errors in the scale of the hole scene
                #So we correct for that scale differance herer
                prediction.extrinsics[:, :, 3] *= float(scale_factor)
                prediction.depth *= float(scale_factor)
            
            
            
            #Align this batch's predicted cammera tranfomrs to the last batch's transforms
            ref_extrinsics = prediction.extrinsics[:nr_used_refs]
            if alingnment_extrinsics is None:
                alingnment_extrinsics = ref_extrinsics
            
            if last_frame_tranform is not None:
                batch_alignment_extrinsics = np.concatenate([alingnment_extrinsics, np.array(last_frame_tranform)], axis=0)
                
                # align_poses_umeyama looks at multiple reference camera positions/rotations and compares them to the prediction
                # It then tries to correct for the differance
                try:
                    r, t, s = align_poses_umeyama(batch_alignment_extrinsics, ref_extrinsics, return_aligned=False)
                    aligned_extrinsics = _apply_sim3_to_poses( prediction.extrinsics[nr_used_refs:], r, t, s)
                    ref_extrinsics = _apply_sim3_to_poses( ref_extrinsics, r, t, s)
                    # prediction.depth /= float(s) # You could correct for the scale differance that is detected in the camera positions. But we are already correcting for depth using depth referances from the last batch you cant do both 
                except:
                    pass
                
                # if you are mostly interested in making the scene perceptioanlly smoth use_last_frame_in_batch_to_align is what you want
                # it is less accurate over the overall scene (so proably worse for 3d construction)but it bridges the gap between between frames beter
                use_last_frame_in_batch_to_align = True
                if use_last_frame_in_batch_to_align:
                    recerence_matrix_4x4 = np.vstack([batch_alignment_extrinsics[-1], np.array([0, 0, 0, 1], dtype=batch_alignment_extrinsics[-1].dtype)])
                    
                    predicted_matrix_4x4 = np.vstack([ref_extrinsics[-1], np.array([0, 0, 0, 1], dtype=ref_extrinsics[-1].dtype)])
                    inverted_prediction = np.linalg.inv(predicted_matrix_4x4)
                    diff = recerence_matrix_4x4 @ inverted_prediction
                    
                    for i, v in enumerate(aligned_extrinsics):
                        v_4x4 = np.vstack([v, np.array([0, 0, 0, 1], dtype=v.dtype)])
                        fixd = diff @ v_4x4
                        aligned_extrinsics[i] = fixd[:3, :]
                    
            else:
                aligned_extrinsics = prediction.extrinsics[nr_used_refs:] #We dont have anything to align to
            
            
            prediction.extrinsics[:nr_used_refs]
            depth_out.extend(prediction.depth[nr_used_refs:])
            intrin_out.extend(prediction.intrinsics[nr_used_refs:])
            extrin_out.extend(aligned_extrinsics)
        
            last_frame_tranform = aligned_extrinsics[-args.batch_overlap:]
            last_frame = batch[-args.batch_overlap:]
            last_frame_depth = torch.as_tensor(prediction.depth[-args.batch_overlap:])
        
    #Save depth video output
    depth_frames_helper.save_depth_video(depth_out, output_tmp_video_path, fps, args.max_depth, W, H)
    depth_frames_helper.verify_and_move(output_tmp_video_path, len(depth_out), output_video_path)

    #Save predicted Field of views
    out_xfovs = []
    for intrin in intrin_out:
        fovx, fovy = depth_map_tools.fov_from_camera_matrix(intrin)
        out_xfovs.append(float(fovx))
    with open(out_xfov_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(out_xfovs))
    
    #Save predicted camera transforms
    out_transformations = []
    for extrin in extrin_out:
        fixed_extrin = np.vstack([extrin, np.array([0, 0, 0, 1], dtype=extrin.dtype)])
        fixed_extrin = np.linalg.inv(fixed_extrin)#da3 outputs inverted transformations to waht we want
        out_transformations.append(fixed_extrin)
    
    with open(out_transformations_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(out_transformations, cls=NumpyEncoder)) 
    
    
    #DEBUG grayscale out
    #max_depth = np.max(depth_out)
    #depth_frames_helper.save_grayscale_video(depth_out, output_video_path+"_grayscale_depth.mkv", fps, max_depth, W, H) # # the model output resolution: prediction.depth.shape[2], prediction.depth.shape[1]
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDVT DepthAnythingV3 video converter')
    parser.add_argument('--color_video', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--max_depth', default=100, type=int, help='the max depth that the video uses', required=False)
    parser.add_argument('--xfov', type=float, help='fov in deg in the x-direction, calculated from aspectratio and yfov in not given', required=False)
    parser.add_argument('--yfov', type=float, help='fov in deg in the y-direction, calculated from aspectratio and xfov in not given', required=False)
    parser.add_argument("--transformation_file", type=str, default=None, help='File with cammera transforms for the video')
    parser.add_argument('--xfov_file', type=str, help='alternative to xfov and yfov, json file with one xfov for each frame', required=False)
    parser.add_argument('--images_per_batch', default=40, type=int, help='nr of images per batch; set depening on GPU memmory', required=False)
    parser.add_argument('--batch_overlap', default=6, type=int, help='nr of images that is overlaped from the last batch', required=False)
    parser.add_argument('--nr_of_ref_frames', default=6, type=int, help='nr of references that will be picked spaning the video', required=False)
    parser.add_argument('--da3_resolution', default=504, type=int, help='resolution width used in inference. 504 is default 252 good for longer videos and GPUs with less ram', required=False)

    args = parser.parse_args()
    
    
    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
    model = model.to(device=device)
    
    # -----------------------
    # Batch mode like video_metric_convert.py
    # -----------------------
    if _is_txt(args.color_video):
        color_list = _read_list_file(args.color_video)

        for idx, c_path in enumerate(color_list, start=1):
            print(f"\n##### [{idx}/{len(color_list)}] Processing {c_path} #####")
            # Re-invoke this script logic on each video
            _single_run(args, c_path)

    else:
        # Single input video behavior (original code path)
        _single_run(args, args.color_video)