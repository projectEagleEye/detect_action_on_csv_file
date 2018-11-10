import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Author: Evan Pan
# Function: pulling the data from csv file and fit into a matrix.
# To use: The get_datafunction will return a 4xn array, with n of 4 length samples, you would have to enter the ame of hte csv file being read 

def get_raw_data(csv_file):

	"""
	function that extracts data from csv file located at "csv_file" directory
	:param csv_file: STRING - file path directory
	:return: STRING 2-D LIST - raw csv data
	"""
	raw_muse_data = []
	try:
		with open(csv_file, newline='') as file:
			# muse_data looks like [[row],[row],[row]]
			data_reader = csv.reader(file, delimiter=',')
			for row_num in data_reader:
				raw_muse_data += [row_num]

		return raw_muse_data
	except:
		print("ERROR - get_data_from_csv(csv_file): unable to read csv file")
def get_Specifc_data(dataType, storage):
	rtv = []
	for item in storage:
		if item[1] == dataType:
			rtv = rtv + [item[2:6]]
	# print(len(rtv[1]))
	return rtv
def get_data(name_of_file):

	thing = get_raw_data(name_of_file)
	thing = get_Specifc_data(" Person0/notch_filtered_eeg",thing)

	data = [[],[],[],[]]
	for i in range (0, len(thing)):
		for j in range (0,4):
			data[j] = data[j] + [round(float(thing[i][j]),3)]
	print (thing)
	return thing;

data = get_data("C:/Users/evan1/Desktop/Programming_Stuff/Eagle Eye/detect_action_on_csv_file/raw_data/Blinks50.csv");

x = list(range (0, len(data[0])))
plt.plot(x,data[0]);
plt.show()   