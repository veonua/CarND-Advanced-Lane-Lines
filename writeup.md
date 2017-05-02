# **Advanced Lane Finding Project**

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

[undistort_output]: ./examples/undistort.png "Undistorted"
[image2]: ./examples/undistort_test1.jpg "Road Transformed"
[image3]: ./examples/warped_straight_lines.jpg "Warp Example"
[image4]: ./examples/edges.png "Binary Example"
[image5]: ./examples/color_fit_lines.png "Fit Visual"
[image6]: ./examples/example_output.png "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `calibrate.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistort_output]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_birdview()`, which appears in the file `pipeline.py` The `get_birdview()` function takes as inputs an image (`img`), I chose the hardcode the source points see method `get_persp_points`:

I verified that my perspective transform was working as expected by drawing the birdview of straight_lines.jpg alog with parallel lines (see `genereate_straight_lanes` method in pipeline.py).

![alt text][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a pair of convolutions to extract lines to generate an image with probabilities of lane position (see `get_lines` method in `pipeline.py`).  Here's an example of my output for this step.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I had to rewrite window search method to make it work with sharp turns, and make it more noise robust and fit my lane lines with a 2nd order polynomial kinda like this (see `patches.py`):

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `calculate_road_info` in `pipeline.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code in `pipeline.py` in the function `draw_marks()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/6wIJN2k7LVE)

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

the biggest and not solved properly issue is curve detection, convolutions worked pretty well, but sliding window approach goes off pretty often.
so in cases when diamon marking appears nearby sliding window detects it as left turn and one or two frames are spoiled

On harder challenge video there are many sharp turns, and sliding window must be tuned to allow to shift window further.

But the idea of sliding window requires a lot of manual tuning, and most likely dead-end. Unfortunately I did not find better than chain of Hought lines approach, so I believe robust curve extraction is good topic for a reseach
