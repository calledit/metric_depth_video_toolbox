# Depth models

There are as of writing this in apr 2025 many depth models avalible.
Diffrent ones are generally good at diffrent things.
The depth models can generally be classified in to the two types - Relative and Metric.

### Relative depth models
Relative depth models are trained in a way that they output Normalized Inverse Depth aka as Normalized disparity.
A Disparity map is a image where each pixel descibes how much movment that pixel has moved betwen two parallel stereo cameras.
The relationsip betwen depth and disparity looks like this:

Depth=f×B/disparity

where:
f is the focal length of the camera,
B is the baseline (distance between the two cameras),

If you assume that B = 1 and f=1 you get disparity = Inverse Depth (This asumtion is often done when talking about diparity maps in the context of depth estimation)

This article has a good explaination of what Normalized Inverse Depth is:
https://github.com/heyoeyo/muggled_dpt/blob/main/.readme_assets/results_explainer.md

The point of outputing Normalized Inverse Depth is to output depth data in a [ordinal](https://en.wikipedia.org/wiki/Ordinal_data) format
which more or less lets you ignore FOV. You know if a pixel is closer or further away its neighbour pixels but not really how much further or closer.

Luckily for us who want metric data the Normalized Inverse Depth that is the output of the models actually does contain information about how much further
away each pixel is. Unfortunately this information has not been **directly** rewarded durring training so the information is **not very accurate** and to
even use the that information we need to apply de-normalization first. De-normalization parameters need to be obtained either from ground truth or
from a metric model.

The metric depth video toolbox has tools that can de-normalize Relative depth in to metric. The result of often not as accurate as the result one gets from
a true metric model but it it often good enogh. The reason we sill might want to use a relative model is that there are as of writing this (Apr 2025)
there are more realtive models and the realtive models that have been trained have reacived far more traning data than the avalible metric models.
So they tend to give more stable result.


### Metric depth models 
Metric models are as the name implied tranied to output depth in meters. The avalible models as of writing this all strugle with this in one way or another (Apr 2025).
They often strugle with many scenes and while the newer ones (like UniK3D) can often deduce the generall shape of the thing the image can still get the scale
of the entire image wrong.

## Models

### Depth-Anything
WIP
### Depth-Anything-Metric
WIP
### Video-Depth-Anything
WIP
### Video-Depth-Anything-Metric
WIP
### DeptPro
WIP
### MoGe
WIP
### UniDepth
WIP
### UniK3D
WIP
### DepthCrafter
WIP
### GeometryCrafter
WIP
