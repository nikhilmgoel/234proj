""" Creates and compresses datasets from annotations
"""
from utils import *
import pickle
import sys
import os
import numpy as np
import copy

VALID_SCENES = [0]

START_HEIGHT = 1948
START_WIDTH = 1630
SCALED_HEIGHT = 240
SCALED_WIDTH = 201

CENTER_Y = 138
CENTER_X = 102

# The following buildings are blocked off
# Formatted as: [YMIN, YMAX, XMIN, XMAX]
LANG_CORNER = [162, 240, 0, 74]
CUBBERLY = [0, 72, 0, 72]
CLOCK_TOWER = [77, 110, 117, 180]
MECH = [176, 240, 148, 201]
OTHER = [0, 100, 187, 201]
OTHER_2 = [0, 19, 162, 201]
BIKE_RACKS = [104, 123, 0, 76]
WALL = [0, 50, 130, 134]
CENTER = [133, 142, 96, 107]

def save_processed_scene(scene, s):
	path = 'train_data/processed/scene' + str(s)
	if not os.path.exists(path):
		os.makedirs(path)
	pickle.dump(scene, open(path + '/scene.pickle', 'wb'))

def load_processed_scene(s):
	return pickle.load(open('train_data/pooling/scene' + str(s) + '/scene.pickle', 'rb'))

def read_training_data():
	""" Convert all data to a new set of frames
	"""
	start_scene = np.zeros((240, 201))

	for YMIN, YMAX, XMIN, XMAX in [LANG_CORNER, CUBBERLY, CLOCK_TOWER, MECH, OTHER, OTHER_2, BIKE_RACKS, WALL, CENTER]:
		for y in range(YMIN, YMAX):
			for x in range(XMIN, XMAX):
				start_scene[y][x] = 1.

	# from matplotlib import pyplot as plt
	# plt.imshow(start_scene, interpolation='nearest')
	# plt.show()

	for s in VALID_SCENES:
		#load annotations
		dataset = open('annotations/deathCircle/video' + str(s) + '/annotations.txt')
		#dictionary to hold parsed details
		scene = {}

		while True:
			line = dataset.readline()
			if line == '':
				break
			row = line.split(" ")
			frame = int(row[5])

			if frame not in scene:
				scene[frame] = copy.deepcopy(start_scene)

			divisor = 1948./240
			x_min = int(int(row[1]) / divisor)
			x_max = int(int(row[3]) / divisor)
			y_min = int(int(row[2]) / divisor)
			y_max = int(int(row[4]) / divisor)

			for y in range(y_min, y_max):
				for x in range(x_min, x_max):
					scene[frame][y][x] = 4.

		# from matplotlib import pyplot as plt
		# plt.imshow(scene[frame], interpolation='nearest')
		# plt.show()

		save_processed_scene(scene, s)


if __name__=="__main__":
	read_training_data()