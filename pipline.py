import subprocess
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Take a clip from a color video and make it in to a stereo 3D video')
    
    parser.add_argument('--color_video', type=str, help='video file to use as color input', required=True)
    parser.add_argument('--mask_video', type=str, help='Mask video file', required=True)
    parser.add_argument('--xfov', type=str, help='camera field of view in x direction', required=True)
    parser.add_argument('--mask_depth', type=str, default= "2.0", help='The depth in meters that is considerd background. (used for infill)', required=True)

    parser.add_argument('--clip_name', type=str, help='A name to give the clip', required=True)
    parser.add_argument('--clip_starttime', type=str, help='Clip start time given as mm:ss', required=True)
    parser.add_argument('--clip_len', type=str, help='Clip length time given as mm:ss', required=True)
    args = parser.parse_args()
    
    color_file = args.color_video
    
    mask_file = args.mask_video
    
    output_dir = "~/cuts"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    times = []
    times.append([args.clip_name, args.clip_starttime, args.clip_len])

    #times = [
    #        ['clip1', '1:24', '0:50'],
    #        ['clip2', '3:16', '0:20'],
    #        ['clip3', '5:07', '1:02'],
    #]

    for clip in times:
        print("clip", clip[0])
        
        #Define file_names
        color_clip_file = output_dir+"/"+clip[0]+".mp4"
        depth_clip_file = output_dir+"/"+clip[0]+"_depth.mkv"
        mask_clip_file = output_dir+"/"+clip[0]+"_mask.mp4"
        audio_wav_file = output_dir+"/"+clip[0]+".wav"
        tracking_file = color_clip_file+"_tracking.json"
        transformation_file = depth_clip_file+"_transformations.json"
        background_file = depth_clip_file+"_background.npy"
        stereo3d_video = depth_clip_file+"_stereo.mkv"
        stereo3d_audio_video = depth_clip_file+"_audio_stereo.mp4"
        
        
        #Generate color, audio and mask clip files
        subprocess.run("ffmpeg -i "+color_file+" -ss "+clip[1]+" -t "+clip[2]+" -c:v libx264 -crf 6 -pix_fmt yuv420p " + color_clip_file, shell=True)
        subprocess.run("ffmpeg -i "+mask_file+" -ss "+clip[1]+" -t "+clip[2]+" -c:v libx264 -crf 23 -pix_fmt yuv420p " + mask_clip_file, shell=True)
        subprocess.run("ffmpeg -i "+color_file+" -ss "+clip[1]+" -t "+clip[2]+" "+audio_wav_file, shell=True)
        
        #Generate clip tracking data
        subprocess.run("python track_points_in_video.py --color_video "+color_clip_file, shell=True)
        
        #Generate clip depth file
        subprocess.run("cd Video-Depth-Anything;python video_metric_convert.py --output_dir "+output_dir+" --color_video "+color_clip_file, shell=True)
        
        #Align 3d points to generate a transformation file
        subprocess.run("python align_3d_points.py --depth_video "+args.depth_clip_file+"  --track_file "+args.tracking_file+" --xfov "+args.xfov+" --assume_stationary_camera --mask_video "+mask_clip_file, shell=True)
        
        #Generate stereo 3d video
        subprocess.run("python stereo_rerender.py --color_video "+color_clip_file+" --xfov "+args.xfov+" --depth_video "+args.depth_clip_file+" --remove_edges --transformation_file "+transformation_file+" --mask_depth "+args.mask_depth+" --load_background "+background_file, shell=True)
        
        #Finnal result in to a video player compatible format with audio
        subprocess.run("ffmpeg -i "+stereo3d_video+" -i "+audio_wav_file+" -c:v libx265 -crf 18 -tag:v hvc1 -pix_fmt yuv420p -c:a aac -map 0:v:0 -map 1:a:0 "+stereo3d_audio_video)
