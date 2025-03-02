# How to recover FOV

FOV is the x and the y angles of the light cone that the camera captures. It is described in various ways that are all mathematically interchangeable.
FOV may be talked about as:
- Camera intrinsics a 3x3 matrix with fx, fy, cx, and cy. (cx and cy are not really part of fov but describes the image center point for cases when the image has been cropped so that the camera center no longer aligns with the image center)
- Lens Focal Length combined with sensor size (55 mm is a common focal length which with a 35mm sensor xfov: 36.2 yfov: 24.6)
- Diagnoal FOV

A table of common conversions is available here: https://www.nikonians.org/reviews/fov-tables

If an image is undistorted within reason (like most images are in the modern world unless someone has applied scalling in one direction and not the other) you can calculate xfov from yfov and back again as the realtionship betwen them is only determined by the image aspect ratio for undistorted images.

## Determining FOV

### Manualy
One way to determine the FOV is to use vanishing points a software to help with this is https://github.com/stuffmatic/fSpy the use of use vanishing points is finiky and requires that the image has large objects with clear lines like walls. These walls may not be aligned with the cameras own axis. Ie the it won't work if you are looking straight at a wall of if the wall is going straigt down the camera lens.

### ML models to determine FOV from single images
There are ML models that try to determine FOV they see: https://huggingface.co/spaces/jinlinyi/PerspectiveFields these are okay, but tend to be about as about as acurate as the maunual use of vanishing points to determine FOV. Ie great for some images not so great for other images.

If ML models does well on recovering FOV really depends on what is in the image and if the model has "knowlage" of the objects true size. And even more so to determine FOV you need to know the true size of atleast two objects that are located at diffrent depths from eachother.

Some depth models try to give estimates of FOV these FOV values tend to be unrealiable. This happens to be one of the major flaws with many metric depth models as of today (Mar 2025). Since FOV is is hard to determine the models make guesses and the resulting depth map can often be very wrong due to this. Some models have API's that allwo you to distort the generated depth map (depth pro and MoGe) to mitigate some of these issues. Unfortionally While the distorted output is better in some ways like overall depth the proccess of distorting also effects the shape of objects in the depth map. The result is that objects in the image end up at a more reasonable overall depth from the camera but the objects instead end up looking like pankakes or stretched out. Another model whish outputs FOV is Unidepth it does not suffer from the same distortion issues like the oher models but it is still not very good at estimating FOV.

### FOV from video
Recovering FOV from video is a better option than using a single image. Here we can use SLAM to estimate the FOV. But yeat again there is a catch SLAM can only recover the FOV if there is paralax ie the camera has translated up or down or side to side. I can not determine the FOV when the camera has only moved back and forward. If the camera has not moved at all and only rotated recovering FOV is yeat again not posible with SLAM.
So keypoint is SLAM will fix it, unless there is a lack of camera movment.

If there is a lack of camera movment. If the camera has no movment. Then your best shot is simply estimating FOV by your own eye. Or generate many versions of an image from the scene with difrent FOVs with unidepth. Then look at the resulting models and decide which looks most reasonable. If you know the true size of two objects in the image and the distance betwen them you can messure in the image then calculate the FOV from those values.

If the video only has movment in the depth direction. The most accurate way to recover FOV is to run slam with varoius FOVs then look at the result and see where the width of objects (as represented by a triangulated point cloud) matches the depth of objects in the scence. 

