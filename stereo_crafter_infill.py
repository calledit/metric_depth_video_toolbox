import argparse
import numpy as np
import os
import torch
import cv2

from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from StereoCrafter.pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
from scipy.ndimage import binary_dilation

import cv2


black = np.array([0, 0, 0], dtype=np.uint8)
blue = np.array([0, 0, 255], dtype=np.uint8)
pipeline = None
frame_rate, frame_width, frame_height = None, None, None
def generate_infilled_frames(input_frames, input_masks):

    input_frames = torch.tensor(input_frames).permute(0, 3, 1, 2).float()/255.0
    frames_mask = torch.tensor(input_masks).permute(0, 1, 2).float()/255.0

    video_latents = pipeline(
        frames=input_frames.clone(),
        frames_mask=frames_mask,
        height=input_frames.shape[2],
        width=input_frames.shape[3],
        num_frames=len(input_frames),
        output_type="latent",
        min_guidance_scale=1.01,
        max_guidance_scale=1.01,
        decode_chunk_size=8,
        fps=frame_rate,
        motion_bucket_id=127,
        noise_aug_strength=0.0,
        num_inference_steps=8,
    ).frames[0]

    video_latents = video_latents.unsqueeze(0)
    if video_latents == torch.float16:
        pipeline.vae.to(dtype=torch.float16)

    video_frames = pipeline.decode_latents(video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2)
    video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="np")[0]

    return (video_frames*255).astype(np.uint8)

def deal_with_frame_chunk(keep_first_three, chunk, out, keep_last_three):

    ##where the side by side picture ends
    pic_width = int(frame_width//2)


    #Looks like shit at 512 x 512 but looks quite good at 1024 x 1024

    #some issues but looks ok (a bit faster is good)
    new_width = 1024
    new_height = 768



    input_frames_i_right = np.array([np.array(cv2.resize(row[0][:frame_height, pic_width:], (new_width, new_height))) for row in chunk])
    mask_frames_i_right = np.array([np.array(cv2.resize(np.all(row[1][:frame_height, pic_width:] != black, axis=-1).astype(np.uint8)*255, (new_width, new_height), interpolation = cv2.INTER_NEAREST)) for row in chunk])

    #The model has only been trained on right shifted images so it works better if we flip the left ones first so they look like they are right eye images
    input_frames_i_left = np.array([np.fliplr(np.array(cv2.resize(row[0][:frame_height, :pic_width], (new_width, new_height)))) for row in chunk])
    mask_frames_i_left = np.array([np.fliplr(np.array(cv2.resize(np.all(row[1][:frame_height, :pic_width] != black, axis=-1).astype(np.uint8)*255, (new_width, new_height), interpolation = cv2.INTER_NEAREST))) for row in chunk])

    print("generating left side images")
    left_frames = generate_infilled_frames(input_frames_i_left, mask_frames_i_left)
    print("generating right side images")
    right_frames = generate_infilled_frames(input_frames_i_right, mask_frames_i_right)

    sttart = 0
    if not keep_first_three:
        sttart = 3

    eend = len(left_frames)
    if not keep_last_three:
        eend -= 3

    proccessed_frames = []
    for j in range(sttart, eend):
        left_img = cv2.resize(np.fliplr(left_frames[j]), (pic_width, frame_height))
        right_img = cv2.resize(right_frames[j], (pic_width, frame_height))


        right_org_img = chunk[j][0][:frame_height, pic_width:].copy()
        left_org_img = chunk[j][0][:frame_height, :pic_width].copy()
        right_mask = chunk[j][1][:frame_height, pic_width:]
        left_mask = chunk[j][1][:frame_height, :pic_width]

        #we invert the mask here, originaly black is source material ie mask = True, white is area that needs infill ie mask = False
        right_black_mask = np.all(right_mask == black, axis=-1)
        left_black_mask = np.all(left_mask == black, axis=-1)

        #We update the org image so it contains the rigthpixels
        left_org_img[~left_black_mask] = left_img[~left_black_mask]
        right_org_img[~right_black_mask] = right_img[~right_black_mask]

        #We save this basic image witout blending for use as input to next batch
        basic_out_image = cv2.hconcat([left_org_img, right_org_img])
        basic_out_image_uint8 = np.clip(basic_out_image, 0, 255).astype(np.uint8)
        proccessed_frames.append(basic_out_image_uint8)

        # Apply edge blending
        # if we dont we get a uggly halo effect around forground objects
        right_backedge_mask = np.all(right_mask == blue, axis=-1)
        left_backedge_mask = np.all(left_mask == blue, axis=-1)

        right_backedge_mask = binary_dilation(right_backedge_mask, iterations = 6)
        left_backedge_mask = binary_dilation(left_backedge_mask, iterations = 6)

        right_mask_float = right_backedge_mask.astype(np.float32)
        left_mask_float = left_backedge_mask.astype(np.float32)


        # Choose a kernel size and sigma for the Gaussian blur (tweak as needed).
        kernel_size = (15, 15)
        sigma = 0  # let OpenCV choose based on kernel size


        # Apply Gaussian blur to get soft alpha masks.
        right_alpha = cv2.GaussianBlur(right_mask_float, kernel_size, sigma)
        left_alpha = cv2.GaussianBlur(left_mask_float, kernel_size, sigma)

        # Expand dimensions to match image shape (H, W, 1).
        right_alpha = right_alpha[..., np.newaxis]
        left_alpha = left_alpha[..., np.newaxis]

        # Now blend: use the soft alpha to mix the original image with the existing one.
        # When alpha is 1, original image takes full weight; when 0, the destination image is preserved.
        left_img = left_alpha * left_img + (1 - left_alpha) * left_org_img
        right_img = right_alpha * right_img + (1 - right_alpha) * right_org_img

        # Finally, concatenate the blended images.
        out_image = cv2.hconcat([left_img, right_img])

        out_image_uint8 = np.clip(out_image, 0, 255).astype(np.uint8)
        out.write(cv2.cvtColor(out_image_uint8, cv2.COLOR_RGB2BGR))

    return proccessed_frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Crafter infill script')
    parser.add_argument('--sbs_color_video', type=str, required=True, help='side by side stereo video')
    parser.add_argument('--sbs_mask_video', type=str, required=True, help='side by side stereo video mask')
    parser.add_argument('--max_frames', default=-1, type=int, help='quit after max_frames nr of frames', required=False)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    frames_chunk=25


    if not os.path.isfile(args.sbs_color_video):
        raise Exception("input sbs_color_video does not exist")

    if not os.path.isfile(args.sbs_mask_video):
        raise Exception("input sbs_mask_video does not exist")

    raw_video = cv2.VideoCapture(args.sbs_color_video)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
    out_size = (frame_width, frame_height)

    mask_video = cv2.VideoCapture(args.sbs_mask_video)

    output_video_file = args.sbs_color_video+"_infilled.mkv"

    codec = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(output_video_file, codec, frame_rate, (frame_width, frame_height))

    img2vid_path = 'weights/stable-video-diffusion-img2vid-xt-1-1'
    unet_path = 'StereoCrafter/weights/StereoCrafter'

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        img2vid_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        img2vid_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        # variant="fp16",
        torch_dtype=torch.float16
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        img2vid_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to("cuda")

    frame_buffer = []
    first_chunk = True
    last_chunk = False
    frame_n = 0
    while raw_video.isOpened():
        print(f"Frame: {frame_n} {frame_n/frame_rate}s")
        frame_n += 1
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

        ret, mask_frame = mask_video.read()
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)

        #bg_color_infill_detect = np.array([0, 255, 0], dtype=np.uint8)
        #bg_mask = np.all(rgb == bg_color_infill_detect, axis=-1)
        #img_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        #rgb[bg_mask] = black

        frame_buffer.append([rgb, mask_frame])

        if len(frame_buffer) >= frames_chunk:
            proccessed_frames = deal_with_frame_chunk(first_chunk, frame_buffer, out, last_chunk)

            # the first 3 frames are not used (unless this is the first chunk), and the last 3 frames are not used
            if first_chunk:
                #keep overlap
                first_chunk = False
            frame_buffer = [
                # have tried priming with previously generated frames: (proccessed_frames[-5], frame_buffer[-5][1])
                # It does not genrerate great results

                (proccessed_frames[-6], frame_buffer[-6][1]),# we prime the next round with some frames
                (proccessed_frames[-5], frame_buffer[-5][1]),
                (proccessed_frames[-4], frame_buffer[-4][1]),
                frame_buffer[-3],# the last 3 frames tend to be pretty bad so we dont prime with them
                frame_buffer[-2],
                frame_buffer[-1],
            ]#reset but keep overlapp

        if frame_n == args.max_frames:
            break

    last_chunk = True
    #Append final three frames or whatever is left
    deal_with_frame_chunk(first_chunk, frame_buffer, out, last_chunk)

    raw_video.release()
    mask_video.release()
    out.release()