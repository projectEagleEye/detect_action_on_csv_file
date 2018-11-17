import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
def get_raw_data(csv_file):

	"""
	function that extracts data from csv file located at "csv_file" directory
	:param csv_file: STRING - file path directory
	:return: STRING 2-D LIST - raw csv data
	"""
	raw_muse_data = []
	try:
		with open(csv_file, newline='') as file:
			# muse_data looks like [[row],[row].........[row]]
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
	print(len(rtv[1]))
	return rtv
def Decrease_sampling(n_4_matrix, frequency):
	rtl = []
	interval = int(256/frequency)
	for i in range (0, len(n_4_matrix), interval):
		rtl = rtl + [n_4_matrix[i]]
	return rtl 
def normalize(data):
	avg = sum(data)/len(data)	
	for i in range (0, len(data)):
		data[i] = data[i] - avg;
	return data;
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
def FFT(data, cut_off = 10):
	cut_off = (cut_off/256.0)
	b = 0.08
	N = int(np.ceil((4 / b)))
	if not N % 2: N += 1
	n = np.arange(N) 
	sinc_func = np.sinc(2 * cut_off * (n - (N - 1) / 2.))
	window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
	sinc_func = sinc_func * window
	sinc_func = sinc_func / np.sum(sinc_func)

	s = list(data)
	new_signal = np.convolve(s, sinc_func)
	return new_signal

thing = get_raw_data('Blinks1.csv')
thing = get_Specifc_data(" Person0/notch_filtered_eeg",thing)

data = [[],[],[],[]]
for i in range (0, len(thing)):
	for j in range (0,4):
		data[j] = data[j] + [round(float(thing[i][j]),3)]

print (thing)
y = data[0]
x = list(range (0, len(y)))
y = normalize(y);


# trace_data = [trace1]
# fig = go.Figure(data=trace_data, layout=layout)


plt.plot(x,y);
plt.show()   

y_prime = FFT(y)
x_prime = list(range (0, len(y_prime)))
plt.plot(x_prime,y_prime)
plt.show()   

y_prime = get_deri(x_prime,y_prime,256)
plt.plot(x_prime,y_prime)
plt.show()   

# py.iplot(fig)

# regularPlot(thing[3], 30)

# for i in thing:
# 	print(i)
# k = - float(thing[0][0]) + float(thing[len(thing)-1][0])
# counter = 0
# for i in range (0, len(thing)):
# 	for j in (2, 5):
# 		try:
# 			thing[i][j] = float(thing[i][j])
# 			thing[i][j] = round(thing[i][j], 2)
# 		except:
# 			counter = counter+1
# data = np.array(thing)
# scrollingGraph(data[:,5])