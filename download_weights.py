import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def abs_path(rel):
    return os.path.join(SCRIPT_DIR, rel)

def download_file(url, dest):
    dest = abs_path(dest)
    if os.path.exists(dest):
        print(f"Already exists, skipping: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Done: {dest}")

def download_gdrive_zip_extract(file_id, zip_member, dest):
    """Download a zip from Google Drive and extract a single member from it."""
    dest = abs_path(dest)
    if os.path.exists(dest):
        print(f"Already exists, skipping: {dest}")
        return
    try:
        import gdown
    except ImportError:
        print("gdown is required for Google Drive downloads: pip install gdown")
        sys.exit(1)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.zip')
    os.close(tmp_fd)
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from Google Drive ({file_id}) -> extracting {zip_member}")
        gdown.download(url, tmp_path, quiet=False)
        with zipfile.ZipFile(tmp_path) as zf:
            with zf.open(zip_member) as src, open(dest, 'wb') as dst:
                dst.write(src.read())
        print(f"Done: {dest}")
    finally:
        os.unlink(tmp_path)

def git_clone(url, dest):
    dest = abs_path(dest)
    if os.path.exists(dest):
        print(f"Already exists, skipping: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Cloning {url} -> {dest}")
    subprocess.run(["git", "clone", url, dest], check=True)
    print(f"Done: {dest}")

def convert_pth_to_safetensors(url, pth_dest, safetensors_dest):
    """Download a .pth file then convert it to safetensors, deleting the .pth afterwards."""
    safetensors_dest_abs = abs_path(safetensors_dest)
    if os.path.exists(safetensors_dest_abs):
        print(f"Already exists, skipping: {safetensors_dest_abs}")
        return
    download_file(url, pth_dest)
    pth_dest_abs = abs_path(pth_dest)
    print(f"Converting {pth_dest_abs} -> {safetensors_dest_abs}")
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"torch and safetensors are required for conversion: {e}")
        sys.exit(1)
    state_dict = torch.load(pth_dest_abs, map_location="cpu")
    save_file(state_dict, safetensors_dest_abs)
    os.remove(pth_dest_abs)
    print(f"Done: {safetensors_dest_abs}")


MODELS = {
    'vda': lambda: (
        download_file(
            "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
            "Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth"
        ),
        download_file(
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth",
            "Video-Depth-Anything/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
        ),
    ),
    'stereocrafter': lambda: (
        git_clone(
            "https://huggingface.co/TencentARC/StereoCrafter",
            "StereoCrafter/weights/StereoCrafter"
        ),
    ),
    'm2svid': lambda: (
        download_file(
            "https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.pt",
            "ckpts/m2svid_weights.pt"
        ),
        download_gdrive_zip_extract(
            "1j_NEG2CPhFeRetYziWK6Qe62R5h7lG_V",
            "ckpts/open_clip_pytorch_model.bin",
            "ckpts/open_clip_pytorch_model.bin"
        ),
    ),
    'inspatio_world': lambda: (
        download_file(
            "https://huggingface.co/inspatio/world/resolve/main/InSpatio-World-1.3B.safetensors",
            "inspatio-world/checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
        ),
        download_file(
            "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth",
            "inspatio-world/checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
        ),
        convert_pth_to_safetensors(
            "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
            "inspatio-world/checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "inspatio-world/checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.safetensors"
        ),
    ),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download model weights')
    parser.add_argument('model', choices=list(MODELS.keys()), help='Model to download weights for')
    args = parser.parse_args()
    MODELS[args.model]()
