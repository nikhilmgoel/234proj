""" Creates and compresses datasets from annotations
"""
from utils import *
import pickle
import sys
import os
import numpy as np
import copy

VALID_SCENES = [0,1,3]

START_HEIGHT = 1948
START_WIDTH = 1630
SCALED_HEIGHT = 98
SCALED_WIDTH = 82

# The following buildings are blocked off
# Formatted as: [YMIN, YMAX, XMIN, XMAX]
LANG_CORNER = [66, 98, 0, int(36/1.22448979)]
CUBBERLY = [0, int(36/1.22448979), 0, int(36/1.22448979)]
CLOCK_TOWER = [int(39/1.22448979), int(55/1.22448979), int(58./1.22448979), int(90./1.22448979)]
MECH = [int(88/1.22448979), 98, int(75/1.22448979), 82]
OTHER = [0, int(50/1.22448979), int(94/1.22448979), 82]
OTHER_2 = [0, int(9/1.22448979), int(81/1.22448979), 82]
BIKE_RACKS = [int(52/1.22448979), int(60/1.22448979), 0, int(37/1.22448979)]
WALL = [0, int(25/1.22448979), int(65/1.22448979), int(67/1.22448979)]
CENTER = [int(69/1.22448979), int(74/1.22448979), int(48/1.22448979), int(53/1.22448979)]

def save_processed_scene(scene, s):
	path = 'train_data/processed/scene' + str(s)
	if not os.path.exists(path):
		os.makedirs(path)
	pickle.dump(scene, open(path + '/scene.pickle', 'wb'))

def load_processed_scene(s):
	return pickle.load(open('train_data/processed/scene' + str(s) + '/scene.pickle', 'rb'))

def read_training_data():
	""" Convert all data to a new set of frames
	"""
	start_scene = np.zeros((98, 82))

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
		# for _ in range(200):
			line = dataset.readline()
			if line == '':
				break
			row = line.split(" ")
			frame = int(row[5])

			if frame not in scene:
				scene[frame] = copy.deepcopy(start_scene)

			divisor = float(START_HEIGHT)/SCALED_HEIGHT
			x_min = int(int(row[1]) / divisor)
			x_max = int(int(row[3]) / divisor)
			y_min = int(int(row[2]) / divisor)
			y_max = int(int(row[4]) / divisor)

			for y in range(y_min, y_max):
				for x in range(x_min, x_max):
					scene[frame][y][x] = 1.

		# from matplotlib import pyplot as plt
		# plt.imshow(scene[frame], interpolation='nearest')
		# plt.show()

		save_processed_scene(scene, s)


if __name__=="__main__":
	read_training_data()