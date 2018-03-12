""" Creates and compresses datasets from annotations
"""
from utils import *
import pickle
import sys
import os
import numpy as np

VALID_SCENES = [0,1,3]

def save_processed_scene(scene, s):
	path = 'train_data/processed/scene' + str(s)
	if not os.path.exists(path):
		os.makedirs(path)
	pickle.dump(scene, open(path + '/scene.pickle', 'wb'))

def load_processed_scene(s):
	return pickle.load(open('train_data/pooling/scene' + str(s) + '/scene.pickle', 'rb'))

def read_training_data():
	"""
	"""
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
				scene[frame] = np.zeros((240, 201))

			x_min = int(row[1])
			x_max = int(row[3])
			y_min = int(row[2])
			y_max = int(row[4])

			x = int((x_min + x_max) / 2. / 8.15)
			y = int((y_min + y_max) / 2. / 8.15)

			scene[frame][y][x] = 1.

		save_processed_scene(scene, s)

			# x = (x_min + x_max) / 2
			# y = (y_min + y_max) / 2
			# label = row[-1][1:-2]
			# #skip sparse busses and resolve cars as carts
			# if label == "Bus":
			# 	continue
			# if label == "Car":
			# 	label = "Cart"
			# member_id = int(row[0])
			# info = [member_id, (x,y), label]
			# if frame in scene:
			# 	scene[frame].append(info)
			# else:
			# 	scene[frame] = [info]

		# separate parsed info into the three dictionaries (reduces complexity while training)
		# outlay_dict: position per frame. 
		# class_dict: classification per member-id.
		# path_dict: path thus far per member-id
		# outlay_dict, class_dict, path_dict = {}, {}, {}
		# frames = scene.keys()
		# frames = sorted(frames)
		# for frame in frames:
		# 	outlay_dict[frame], path_dict[frame] = {}, {}
		# 	for obj in scene[frame]:
		# 		outlay_dict[frame][obj[0]] = obj[1] # in a frame, set {member-id: positions} of all objects
		# 		class_dict[obj[0]] = obj[2] # frame doesn't matter. set {member-id: classification} for all objects in the scene

		# 		# initial frame
		# 		if frame == 0:
		# 			path_dict[frame][obj[0]] = [obj[1]] # set {member-id: position} for all objects in the first frame
		# 			continue

		# 		prev_frame = frames[frames.index(frame) - 1]
		# 		if obj[0] not in path_dict[prev_frame]:
		# 			path_dict[frame][obj[0]] = [obj[1]]
		# 		else:
		# 			path_dict[frame][obj[0]] = path_dict[prev_frame][obj[0]] + [obj[1]]

		# save_processed_scene([outlay_dict, class_dict, path_dict], s)
