# file that holds all the parameters for feature extraction and detection

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