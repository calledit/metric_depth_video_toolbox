# How to recover camera position

Recovering the camera position from video is a complex task that can be done by matching and tracking features in subseqvent frames. These tracked features are then used to recover the camera position in a process of minimizing the projection error between all frames.
This process is named SLAM and works best if there is lots of paralax between the frames, and to a large degree it is dependent on having tracking points that are perfectly still betwen frames.
It is hard for the SALM algorithm to know if the tracked points in the video is moving cause the camera as moving or if subjects in the video are moving.

Newer methods use Nural networks to recover cemera position and as of 2025 the besst library to find the camera position is either
https://github.com/facebookresearch/vggt or https://github.com/mega-sam/mega-sam.
Where mega-sam is video only while vggt can do both images and video.
