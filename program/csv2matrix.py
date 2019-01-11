import csv
import pickle
# Author: Evan Pan
# Function: pulling the data from csv file and fit into a matrix.
# To use: The get_datafunction will return a 4xn array, with n of 4 length samples, you would have to enter the ame of hte csv file being read 
# note that you can optionally save the data into a pickle file if you feel like it
# also note that you can choose the range of data by modifying the get_specific_data function
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
	thing = get_Specifc_data(" /muse/notch_filtered_eeg",thing)
	data = [[],[],[],[]]
	for i in range (0, len(thing)):
		for j in range (0,4):
			data[j] = data[j] + [round(float(thing[i][j]),3)]
	print (thing)
	return thing
#save listt into a pickle file! make sure the filename ends with '.pkl'" 
def toPKL(directory_and_name,listt):
	with open(directory_and_name, 'wb') as output:
		pickle.dump(listt, output, pickle.HIGHEST_PROTOCOL)
#load the saved save.pkl        
def fromPKL(directory_and_name):
	with open(directory_and_name, 'rb') as input:
		retn = pickle.load(input)
	return retn;

