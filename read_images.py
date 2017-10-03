import os
import glob

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