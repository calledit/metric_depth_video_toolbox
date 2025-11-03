# Metric depth video toolbox (MDVToolbox)

Tools for Generating and working with metric 3D depth videos.

![gif_banner](https://github.com/user-attachments/assets/4d737bb3-6fb6-4135-b01e-b35528371d22)

_Banner created with 3d_view_depthfile.py_

## 📽️ Demos

**Showcase video:** https://youtu.be/nEiUloZ591Q

**Movie → 3D conversion demo**: https://www.youtube.com/watch?v=PLFjoNgkZDY

**Sample stereo clips:**
https://github.com/calledit/metric_depth_video_toolbox/releases/tag/Showcase

---

## ✨ Features

### Depth Generation (Metric)

Convert videos into true metric depth using multiple SOTA models:

- Depth-Anything series  
- MoGe  
- UniDepth  
- UniK3D  
- DepthPro  
- DepthCrafter  
- MVSAnywhere  

### Stereo / 3D Conversion

- 2D movies → 3D (`movie_2_3D.py`)
- Stereo rendering from depth
- Parallax infill & diffusion-based stereo inpainting

### Visualization

- Real-time 3D viewer (`3d_view_depthfile.py`)
- Novel-view rendering from depth video

### Camera Tracking & 3D Reconstruction

- Depth-assisted camera tracking
- Long-term point tracking (CoTracker3)
- Pose extraction & alignment tools

### Export Tools

Export to standard formats for DCC tools:

- `.ply` point clouds
- `.obj` meshes
- Blender `.blend` & Alembic `.abc` camera tracks
- 8-bit / 16-bit depth maps
- Depth rescaling based on triangulation

### Masking & Cleanup

- Automatic human masking
- Subtitle/logo inpainting

---

## 📚 Documentation

| Topic | Link |
|---|---|
Beginner guide | [`HOWTO.md`](./HOWTO.md)  
Movie → 3D guide | [`HOWTO_movie2_3d.md`](./HOWTO_movie2_3d.md)  
Full tool reference | [`USAGE.md`](./USAGE.md)  
GUI tutorial video | https://youtu.be/BE_aJCI7DHI  

---

## 🧠 Depth Video Format

MDVT uses **RGB-encoded 16-bit metric depth**:

- **Red + Green** = upper 8 bits (duplicated for visibility)  
- **Blue** = lower 8 bits  
- Default range: **100 meters**
- Resolution: **~1.5mm depth precision**

Future upgrade options: **24-bit depth** or **log-encoded depth** for long-range accuracy.

---
---

## ⚙️ Installation

### Windows
1. install git https://git-scm.com/downloads/win
2. Install miniconda https://docs.conda.io/en/latest/
3. Open the Anacoda prompt(miniconda) from the start menu
4. run
```batch
git clone https://github.com/calledit/metric_depth_video_toolbox
cd metric_depth_video_toolbox
windows_installer.bat
```
5. use `conda activate mdvt` to activate conda and use the tools. Some ML models are not supported on windows but the most esential ones are supported like Sterecrafter and depth anything.

### Ubuntu/Debian and OSX

```bash


git clone https://github.com/calledit/metric_depth_video_toolbox
cd metric_depth_video_toolbox

# on linux
sudo apt-get install -y libgl1
./install_mdvtoolbox.sh

#Optional (only required for some tools)
./install_mdvtoolbox.sh -megasam
./install_mdvtoolbox.sh -geometrycrafter
./install_mdvtoolbox.sh -unik3d
./install_mdvtoolbox.sh -depthpro
./install_mdvtoolbox.sh -stereocrafter
./install_mdvtoolbox.sh -madpose
./install_mdvtoolbox.sh -unidepth
./install_mdvtoolbox.sh -moge
./install_mdvtoolbox.sh -promptda

# if using headless linux you need to start a virtual x11 server
apt-get install xvfb
Xvfb :2 &
export DISPLAY=:2

# OSX (OSX only supports post processing of depth videos not generation of them. As the ML models need CUDA)
# (open3d requires python3.11 on OSX (as of 2025)))
pip3.11 install open3d numpy opencv-python

```

### Requirements
The tools that reuire ML models have been tested on machines with nvida 3090 cards that support Cuda 12.4 and Torch 2.5.1 on [vast.ai](https://cloud.vast.ai/?ref_id=148636) using "template PyTorch (cuDNN Devel)"

### Next steps
- Currently waiting for new stable and accurate depth models.

### Contributing
Is appreciated. Even for simple things like spelling.
