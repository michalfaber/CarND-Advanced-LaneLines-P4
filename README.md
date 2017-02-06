**Advanced Lane Finding Project**
=================================


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output_images/undistorted.png "Road Transformed"
[image3]: ./output_images/thresholded.png "Binary Example"
[image4]: ./output_images/bird_eye_view.png "Warp Example"
[image5]: ./output_images/lane_pixels.png "Fit Visual"
[image6]: ./output_images/projection.png "Output"
[video1]: ./output.mp4 "Video"

---

### Camera Calibration

#### 1. Camera matrix and distortion coefficients.

The code for this step is contained in `project.py` - function `calibration_matrix` (line 25). 
The first step is to prepare object points - x,y,z coordinates of the chessboard corners in the world `objpoints` . Here I am assuming the chessboard 
is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. `imgpoints` will be appended 
with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection `findChessboardCorners`.
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` 
function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied `cv2.undistort` to the example image.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.
Image is converted to the HLS colorspace - function `color_thresh` (line 90) Only `S` channel is used. Values outside of the range (170, 255) are zeroed.
Gradient threshold is performed using function `abs_sobel_thresh`. Values outside of the range (20, 100) are zeroed. 
Result of both thresholding is combined in the final binary image. Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform includes a function called `get_persp_transform()` (line 99). The function returns perspective
transformation matrix. Matrix is calculated using `cv2.getPerspectiveTransform`,  source `src` and destination `dst` points. I chose
the hardcode the source and destination points in the following manner:

```
    src = np.float32([[220, 719], [1220, 719], [750, 477], [553, 480]])
  
    dst = np.float32([[240, 719], [1040, 719], [1040, 300], [240, 300]])
```

Final image is created using `cv2.warpPerspective` function.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In this step I identified lane-line pixels - function `lane_points` (line 140) At first I identified histogram peaks
at the bottom stripe of image for the first half of the image and for the second half of the image. Then I used sliding
window technique to determine lane pixels. In the next step I used the points to fit with a polynomial.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines starting from 204 in my code in `project.py` - function `curvature`. I used the same points
but scaled them so that I got values in meters. Position of the vehicle is determined in the lines 288 - function `offset` 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 266 in my code in `project.py` in the function `draw_lanes()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Algorithm in its basic form is not very robust for sequences with shadows, occlusions, noisy edges. Additionally, hardcoded values for perspective transformation
may pose a problem in image sequences where distance between lanes is not US standard 3.7 m.

Possible improvement includes dynamic mapping between src and dst points for perspective transformation, detecting lines using hugh transform,
excluding lines with incorrect slope, more robust thresholding using different colorspaces RGB YUV, use RANSAC instead of fiting polynomial
