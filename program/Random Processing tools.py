import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from biosppy.signals.tools import filter_signal as Filter
from biosppy.signals.tools import normalize as normalize

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
def Decrease_sampling(n_4_matrix, frequency):
	rtl = []
	interval = int(256/frequency)
	for i in range (0, len(n_4_matrix), interval):
		rtl = rtl + [n_4_matrix[i]]
	return rtl 
# def normalize(data):
# 	avg = sum(data)/len(data)	
# 	for i in range (0, len(data)):
# 		data[i] = data[i] - avg;
# 	return data;
def get_deri(x,y,freq):
	dx = (1/freq)
	rtv = []
	avg = sum(y[0:10])/10
	rtv = rtv + [0]
	for i in range(0, len(y)-1):
		deriv_val = y[i+1] - y[i]
		rtv = rtv + [deriv_val]
	print(len(rtv))
	print(len(y))
	return rtv


####################################### get signal ########################################### 

thing = get_raw_data('Blinks1.csv')
thing = get_Specifc_data(" Person0/notch_filtered_eeg",thing)

# getting the transverse matrix 
data = [[],[],[],[]]
for i in range (0, len(thing)):
	for j in range (0,4):
		data[j] = data[j] + [round(float(thing[i][j]),3)]

####################################### Process data ########################################### 
processed_data = [[],[],[],[]] 
# normalize the data, making them all ossillate around 0;
for i in range (0, 4):
	data[i] = normalize(data[i])[0]
# find sum and difference of signals to amplify 
for i in range (0, len(data[0])):
	processed_data[0] = processed_data[0] + [data[3][i] + data[0][i]] # sum of temple data
	processed_data[1] = processed_data[1] + [data[2][i] + data[1][i]] # sum of brow data
	processed_data[2] = processed_data[2] + [data[3][i] - data[0][i]] # difference of temple data
	processed_data[3] = processed_data[3] + [data[2][i] - data[1][i]] # difference of brow data
for i in range (0, 4):
	processed_data[i] = (Filter(processed_data[i], order = 4, frequency = 10)[0])	

####################################### plotting signal  ########################################### 

t = list(range (0, len(processed_data[0])))
fig, axs = plt.subplots(4, 2)
for k in range (0, 4):

	axs[k][0].plot(t, data[k])
	axs[k][0].set_xlabel('time')
	axs[k][0].set_ylabel('raw data')
	axs[k][0].grid(True)

	axs[k][1].plot(t, processed_data[k])
	axs[k][1].set_xlabel('time')
	axs[k][1].set_ylabel('processed data')
	axs[k][1].grid(True)

fig.tight_layout()
plt.show()    

####################################### plotting signal end #############################################