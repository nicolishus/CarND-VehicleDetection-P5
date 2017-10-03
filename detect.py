import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from helperfunctions import *
from scipy.ndimage.measurements import label
# from parameters import *

def read_images():
    # First get all car images
    vehicle_directory = "vehicles/"
    image_types = os.listdir(vehicle_directory)
    cars = []
    for image_type in image_types:
        cars.extend(glob.glob(vehicle_directory + image_type + "/*"))

    print("Found {} vehicle images.".format(len(cars)))
    with open("cars.txt", 'w') as f:
        for filename in cars:
            f.write(filename + "\n")

    # second get all non car images
    not_directory = "non-vehicles/"
    image_types = os.listdir(not_directory)
    non_cars = []
    for image_type in image_types:
        non_cars.extend(glob.glob(not_directory + image_type + "/*"))

    print("Found {} non-vehicle images.".format(len(non_cars)))
    with open("non-cars.txt", 'w') as f:
        for filename in non_cars:
            f.write(filename + "\n")
    return cars, non_cars

def find_cars(img, scale):
    ystart = 302
    ystop = 656
    xstart = 656
    xstop = 1280
    img_boxes = []
    count = 0
    draw_img = np.copy(img)
    #make a heatmap of zeroes
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv="RGB2YCrCb")
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale),np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # extract hog for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255))
                img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
    return draw_img, heatmap

def convert_color(img, conv="RGB2YCrCb"):
    if conv == "RGB2YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == "BGR2YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == "RGB2LUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def apply_threshold(heatmap, threshold):
    # zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # return the image
    return img

def process_image(img):
    out_img1, heat_map1 = find_cars(img, 1.0)
    out_img2, heat_map2 = find_cars(img, 1.5)
    out_img3, heat_map3 = find_cars(img, 2.0)
    out_img = out_img1 + out_img2
    heat_map = heat_map1 + heat_map2
    heat_map = apply_threshold(heat_map, 1)
    labels = label(heat_map)
    # draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

# main code that reads the training images, extracts features, trains the SVM,
# and then processes the video.
# get all images
cars, notcars = read_images()
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # can be 0, 1, 2, or "ALL"
spatial_size = (32,32)  # spatial binning dimensions
hist_bins = 32  # number of histogram bins
spatial_feat = True # spatial features on or off
hist_feat = True    # histogram features on or off
hog_feat = True     # HOG features on or off

t=time.time()
n_samples = 2000
random_idxs = np.random.randint(0, len(cars), n_samples)
test_cars = cars # np.array(cars)[random_idxs]     # = cars
test_notcars = notcars # np.array(notcars)[random_idxs]  # = notcars

car_features = extract_features(test_cars, color_space=color_space, 
                            spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat, vis=True) 

notcar_features = extract_features(test_notcars, color_space=color_space, 
                            spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

print(time.time()-t, "Seconds to compute features...")
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state) # change test set size

print('Using:',orient,'orientations,',pix_per_cell,
    'pixels per cell,', cell_per_block,'cells per block,',
    hist_bins, "histogram_bins, and", spatial_size,"spatial sampling")
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

out_images = []
out_titles = []
out_maps = []

from moviepy.editor import VideoFileClip
from IPython.display import HTML

output_filename = "final_output.mp4"
# clip = VideoFileClip("test_video.mp4")
clip = VideoFileClip("project_video.mp4")  # .subclip(25,30)
output_clip = clip.fl_image(process_image)
output_clip.write_videofile(output_filename, audio=False)