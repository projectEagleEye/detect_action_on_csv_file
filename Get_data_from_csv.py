import csv
import numpy as np
import pyplot 
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

def

for i in thing:
	print(i)
k = - float(thing[0][0]) + float(thing[len(thing)-1][0])
print(k)

