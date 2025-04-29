# Depth models

There are as of writing this in apr 2025 many depth models available.
Different ones are generally good at different things.
The depth models can generally be classified in to the two types - Relative and Metric.

### Relative depth models
Relative depth models are trained in a way that they output Normalized Inverse Depth aka as Normalized disparity.
A Disparity map is a image where each pixel describes how much movement that pixel has moved between two parallel stereo cameras.
The relationship between depth and disparity looks like this:

Depth=f×B/disparity

where:
f is the focal length of the camera,
B is the baseline (distance between the two cameras),

If you assume that B = 1 and f=1 you get disparity = Inverse Depth (This assumption is often done when talking about disparity maps in the context of depth estimation)

This article has a good explanation of what Normalized Inverse Depth is:
https://github.com/heyoeyo/muggled_dpt/blob/main/.readme_assets/results_explainer.md

The point of outputting Normalized Inverse Depth is to output depth data in a [ordinal](https://en.wikipedia.org/wiki/Ordinal_data) format
which more or less lets you ignore FOV. You know if a pixel is closer or further away its neighbour pixels but not really how much further or closer.

Luckily for us who want metric data the Normalized Inverse Depth that is the output of the models actually does contain information about how much further
away each pixel is. Unfortunately this information has not been **directly** rewarded durring training so the information is **not very accurate** and to
even use the that information we need to apply de-normalization first. De-normalization parameters need to be obtained either from ground truth or
from a metric model.

The metric depth video toolbox has tools that can de-normalize Relative depth in to metric. The result of often not as accurate as the result one gets from
a true metric model but it it often good enough. The reason we sill might want to use a relative model is that there are as of writing this (Apr 2025)
there are more relative models and the relative models that have been trained have received far more training data than the available metric models.
So they tend to give more stable result.


### Metric depth models 
Metric models are as the name implied trained to output depth in meters. The available models as of writing this all struggle with this in one way or another (Apr 2025).
They often struggle with many scenes and while the newer ones (like UniK3D) can often deduce the general shape of the thing the image can still get the scale
of the entire image wrong.

## Models

### Single frame model
Single frame model takes a picture and outputs depth some like UniDepth and UniK3D also takes FOV as input which allows them to have better accuracy. Single frame models tend to suffer from jitter as each frame is evaluated without reference to the next.

#### Depth-Anything-Metric
A metric model that was converted from the relative version of Depth-Anything. The model has a tendency to sometimes produce jaged edges and strange outputs. Especially when there is fog or blurriness.

#### DeptPro
Produces very nice outputs, very sharp edges, quite allot of jitter, and will sometimes break down completely, and produce garbage results.

#### MoGe
One most accurate metric model (of the models that don’t take FOV as input) as of Apr 2025 

#### UniDepth
The first model with true FOV input. Produces okay results but seams to suffer from lack of training data.

#### UniK3D
The upgrade from UniDepth Produces very nice outputs when FOV is given, is also quite good at predicting FOV when it is not given. tend to have some issues with scale it sometime makes everything very very small. Scenes that should be 100m deep may not extend further than 2m.

### Video models
Video models take multiple frames as input and outputs multiple frames of depth. The advantage of this type of model as that the don’t no have the jitter that Single frame model has as the output depth is stabilised over many frames.

#### DepthCrafter
A relative model that is based on stable diffusion(slow). Due to how it is implemented it has issues with longer scenes.

#### GeometryCrafter
Geometry crafter can be thought of more like a depth stabiliser, it takes unstable single frame depth images and a color image and tries to output stable video depth. Built on the same code base as DepthCrafter.

#### Video-Depth-Anything
A relative video model that tend to produce nice results has issus with blurry images.

#### Video-Depth-Anything-Metric
Similar to is relative sibling but has a tendency to break down and produce garbage to a higher extent than the relative version.
