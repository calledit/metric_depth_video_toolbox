import torch

from depth_anything_v2.dpt import DepthAnythingV2

depth_anything = None

def get_metric_depth(image, input_size=518):
    global depth_anything

    #Load model
    if depth_anything is None:
        depth_anything = DepthAnythingV2(**{**{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}, 'max_depth': 20})
        depth_anything.load_state_dict(torch.load('Video-Depth-Anything/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
        depth_anything = depth_anything.to('cuda').eval()

    depth = depth_anything.infer_image(image, input_size)
    return depth