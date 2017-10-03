

# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: hog_extraction.png
[image2]: found_images.PNG
[image3]: boxes_drawn.PNG
[image4]: boxes_drawn2.PNG
[image5]: heatmap1.PNG
[image6]: heatmap2.PNG
[image7]: result.PNG
[video1]: ./final_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the detect.py file after all the function definitions (starting on line 173). I first started by reading in all the images (car and not car) for the training data. From here, I ran the extract_features() function that extracts, color, hog, spatial, and histogram features from the image.

From here, I made a testing set of 10% and trained a SVM classifier and measured the accuracy. Using this accuracy metric, I tweaked the parameters for feature extraction to find the best combination.

I settled on the following parameters:

* color_space = 'YCrCb'
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = "ALL"
* spatial_size = (32,32)
* hist_bins = 32
* spatial_feat = True
* hist_feat = True
* hog_feat = True

Below is an example of a training image and it's HOG feature extraction:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

Since my testing code had print statements for testing accuracy, I tweaked the parameters (mostly trial and error) to find the highest accuracy. I started with the parameters found below and got a 94.5% accuracy.

* color_space = 'RGB'
* orient = 6
* pix_per_cell = 6
* cell_per_block = 1
* hog_channel = 1
* spatial_size = (16,16)
* hist_bins = 32
* spatial_feat = True
* hist_feat = True
* hog_feat = True

From this starting point, I started to tweak one parameter at a time and see the effect. I first changed the color space and managed to get 97%; a big increase. I notice that increasing the orient produced better accuracy,up until about 9. From here, my accuracy was about 96.2%. I then decided to change a few parameters to see if I could iterate faster. My next set of parameters is shown below with an accuracy of 98.1%.

* color_space = 'YCrCb'
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 1
* spatial_size = (32,32)
* hist_bins = 32
* spatial_feat = True
* hist_feat = True
* hog_feat = True

Finally, I changed the hog_channel to "ALL" and was able to get 99% accuracy on the testing set. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a subset of the training data supplied in the project resources. There were about the same number of car and not car images; so the training data was symmetrical and did not need to be augmented.

Output of loading the training data:

![alt text][image2]

I started with a subsample of 1000 images so I could iterate faster when parameter tuning and refining the pipeline. Once the training data was loaded and the features were extracted as explained in the above section, I ran the features of both car and notcar sets through from  the sklearn LinearSVM function. This trains a linear SVM, which took about 90 seconds with the above parameters and training sets. The resulting SVM classifier could then be used to detect cars on features extracted from a frame of video.

The SVM classifier code is in detect.py after the function definitions starting on line 195.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My first iteration of the sliding window search was to scan the whole image. This was very slow... Code for Sliding Window Search can be found in the helperfunctions.py file starting on line 108. This code was given by the lectures and was used for testing on still images. There is sliding window search code in the find_car() function in detect.py starting on line 45 that was used when processing the video.

I used a scale parameter to change the window size (64) and ultimately ended up running three sizes on the bottom ~50% of the image. This was chosen since the top half is the horizon and not useful for searching for cars. A scale value of 1.0 was first used and could detect intermediate distance for cars. I also ended up using 1.5 and 0.5 scale factors to get a smaller size and larger size window for searching when cars were much closer (just getting into the camera's field of view) or much farther (when they are smaller).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are some images showing the boxes drawn on test images:

![alt text][image3]
![alt text][image4]

Again, I tried to optimizer the performance of my classifier by using different window sizes for variations in size, a change in color space for variations in car colors, and tweaking the parameters that maximized the test set accuracy.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](final_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Positive detections were saved for each frame of the video. A heatmap was created since there were multiple window sizes, meaning certain part of the images could be positive more than once and have overlaps. A threshold was applied to remove the false positives in the heatmap (line 119 in detect.py). Having a filtered heatmap, I used scipy.ndimage.measurements.label() to get the blobs or where multiple positives (overlaps) were reocrded. This would mean that a car was detected, with higher values in the heatmap indicating more confidence.

Boxes were then drawn over the thresholded heatmap, asssuming that anything left was a detected car. Below is an example of test images with boxes drawn and the accompanying heatmap:

![alt text][image5]
![alt text][image6]

And finally an example of the result; a box drawn around a detected car from the video:

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A big issue with this approach is computation time. This could not run in real time as it was very slow to process the whole video. Typically a longer feature extraction vector would mean better accuracy, but it would also mean more time to process.

Since I used only three difference window sizes for searching, the pipeline could fail when the cars are too close, hence big, or too far (small) for the classifier. More sizes for searching would produce better results, but again, this would mean a longer time to train and process.

Lastly, an average function for the heatmaps would make the box boundaries smoother and decrease false positives. I am currently debugging the averagin function, but each iteration is taking about an hour.

