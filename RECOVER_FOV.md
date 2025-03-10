# How to recover FOV

FOV is the x and the y angles of the light cone that the camera captures. It is described in various ways that are all mathematically interchangeable.
FOV may be talked about as:
- Camera intrinsics a 3x3 matrix with fx, fy, cx, and cy. (cx and cy are not really part of fov but describes the image center point for cases when the image has been cropped so that the camera center no longer aligns with the image center)
- Lens Focal Length combined with sensor size (55 mm is a common focal length which with a 35mm sensor gives xfov: 36.2 and yfov: 24.6). The relationship is FOV_x=2arctan(sensor_width/2*focal_length)
- Diagnoal FOV

A table of common conversions is available here: https://www.nikonians.org/reviews/fov-tables

If an image is undistorted within reason (like most images are in the modern world unless someone has applied scalling in one direction and not the other) you can calculate xfov from yfov and back again as the realtionship betwen them is only determined by the image aspect ratio for undistorted images.

## Determining FOV

### Manually
One way to determine the FOV is to use vanishing points; a software to help with this is [fSpy](https://github.com/stuffmatic/fSpy). The use of use vanishing points is finiky and requires that the image has large objects with clear lines like walls. These walls may not be aligned with the cameras own axis. Ie using vanishing points won't work if you are looking straight at a wall of if the wall is going straigt down the cameras point of view. If you know the true size of two objects in the image and the distance betwen them you can messure in the image then calculate the FOV from those values.

### ML models to determine FOV from single images
There are ML models that try to determine FOV an example is [PerspectiveFields](https://huggingface.co/spaces/jinlinyi/PerspectiveFields) which works okay but tend to be about as about as acurate as the maunual use of vanishing points to determine FOV. Ie. great for some images, not so great for other images.

If ML models does well on recovering FOV really depends on what is in the image and if the model has "knowlage" of the objects true size. To determine FOV you need to know the true size of atleast two objects that are located at a known depth from eachother in the image.

Some depth models try to give estimates of FOV these FOV values tend to be unrealiable. This happens to be one of the major flaws with many metric depth models as of today (Mar 2025). Since FOV is is hard to determine, the models make guesses and the resulting depth map can often be very wrong due to this. Some models (like depth pro and MoGe) have API's that allows you to distort the generated depth map to mitigate some of these issues. Unfortionally while the distorted output depth map is better in some ways. Like overall depth the proccess of distorting also effects the shape of objects in the depth map. The result is that objects in the image end up at a more reasonable overall depth from the camera but the objects instead end up looking like pankakes, stretched out or erroneously rotated. Another model which outputs FOV is Unidepth; it does not suffer from the same distortion issues like the oher models but it is still not very good at estimating FOV.

### FOV from video
Recovering FOV from video is a better option than using a single image. In this case we can use SLAM to estimate the FOV. But yeat again there is a catch; SLAM can only recover the FOV if there is paralax ie. the camera has translated up or down or side to side. I can not determine the FOV when the camera has only moved back and forward. If the camera has not moved at all and only rotated while staying in place recovering FOV is yeat again not posible with SLAM.
So the keypoint is SLAM will fix it, unless there is a lack of camera movment.

If there is a lack of camera movment, then your best shot is simply estimating FOV by your own eye. Or generate many versions of an scene from the image with difrent FOVs with unidepth. Then look at the resulting scenes and decide which looks most reasonable.

If the video only has movment in the depth direction. The most accurate way to recover FOV is to run slam with varoius FOVs then look at the result and see in which FOV the width of objects (as represented by a triangulated point cloud) matches the depth of objects in the scence.

