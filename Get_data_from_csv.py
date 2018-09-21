import csv
import numpy as np
from plot_with_scrolling_bar import *


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
			rtv = rtv + [item]
	return rtv

thing = get_raw_data('Blinks1.csv')
thing = get_Specifc_data(" Person0/notch_filtered_eeg",thing)


# for i in thing:
# 	print(i)
# k = - float(thing[0][0]) + float(thing[len(thing)-1][0])

counter = 0
for i in range (0, len(thing)):
	for j in (2, 5):
		try:
			thing[i][j] = float(thing[i][j])
			thing[i][j] = round(thing[i][j], 2)
		except:
			counter = counter+1
data = np.array(thing)
scrollingGraph(data[:,5])

