import torch
import os 
import sys
sys.path.append("DiffuEraser_np_array")
import time
import argparse
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device
import cv2


device = None
last_ckpt = None
video_inpainting_sd = None
propainter = None

def run_infill_on_frames(frames_rgb, mask_frames, mask_dilation_iter = 8, ckpt = "2-Step", propainer_frames = None):
    global device, last_ckpt, video_inpainting_sd, propainter
    # PCM params
    if last_ckpt != ckpt:
        device = get_device()
        ckpt = "2-Step"
        last_ckpt = ckpt
        video_inpainting_sd = DiffuEraser(device, "stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/sd-vae-ft-mse", "lixiaowen/diffuEraser", ckpt=ckpt)

    H0, W0 = frames_rgb[0].shape[:2]

    if propainer_frames is None:
        if propainter is None:
            propainter = Propainter("ruffy369/propainter", device=device)

        propainer_frames = propainter.forward(frames_rgb, mask_frames, 
                        ref_stride=10, neighbor_length=10, subvideo_length=50,
                        mask_dilation = mask_dilation_iter) 



    ## diffueraser
    guidance_scale = None    # The default value is 0.  
    inpainted_frames = video_inpainting_sd.forward(frames_rgb, mask_frames, propainer_frames,
                                max_img_size = 960, mask_dilation_iter=mask_dilation_iter,
                                guidance_scale=None)

    for i, f in enumerate(inpainted_frames):
        if f.shape[0] != H0 or f.shape[1] != W0:
            inpainted_frames[i] = cv2.resize(f, (W0, H0))
    
    return inpainted_frames
    

