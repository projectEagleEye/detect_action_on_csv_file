# Author: Evan Pan
# Date: 2018-09-20
# Description: THis piece of code contains a function that plots x and y to a Scrollable graph. The limitation is that it only plots one graph, as supposed to be four. 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pickle
class GraphingData():
	def __init__(self, x, y, interval):
		self.x = x
		self.y = y
		self.length = interval
		self.step = 10
		self.x_dis = x[0:self.length]
		self.y_dis = y[0:self.length]
	def move(self,val):
		val = int(val)
		difference = len(self.x) - (int(val) + self.length)
		if difference > 0: 
			new_x = self.x[val:(val + self.length)]
			new_y = self.y[val:(val + self.length)]
		else:
			new_x = self.x[val:(len(self.x)-1)]
			new_y = np.append(self.y[val:(len(self.x)-1)],np.zeros(self.length -len(self.y[val:(len(self.x)-1)])))
		return (new_x,new_y)
def scrollingGraph(y,interval = 500):
	x = np.arange(0, len(y),1)
	minY = 10000
	maxY = 0
	for i in y:
		i = float(i)
		if i > maxY:
			maxY = i
		if i < minY:
			minY = i
	maxY = maxY + 5
	minY = minY - 5
	MaxX = len(x)
	# print(minY, maxY)
	fig, ax = plt.subplots()
	plt.subplots_adjust(left=0.25, bottom=0.25)
	thing = GraphingData(x,y,interval)
	delta_f = int(interval/50)
	# print(y[0:interval])
	l, = plt.plot(x[0:interval], y[0:interval], lw=0.5, color='red')
	plt.axis([0, interval, minY, maxY])

	axcolor = 'lightgoldenrodyellow'
	axfreq = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
	# axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

	sfreq = Slider(axfreq, 'Freq', 0, MaxX, valinit=0, valstep=delta_f)
	# samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


	def update(val):
		# amp = samp.val
		freq = sfreq.val
		l.set_ydata(thing.move(val)[1])
		fig.canvas.draw_idle()
	sfreq.on_changed(update)
	# samp.on_changed(update)

	resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
	button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


	def reset(event):
		sfreq.reset()
		# samp.reset()
	button.on_clicked(reset)

	rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
	radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


	def colorfunc(label):
		l.set_color(label)
		fig.canvas.draw_idle()
	radio.on_clicked(colorfunc)

	plt.show()

def regularPlot(y, place, window = 1000):
	x = np.arange(0, window)
	y = np.array(y)[place: place + window]
	plt.plot(x,y)
	plt.show()
def normalize(data, window):
	avg = np.sum(data)/len(data)	
	for i in range (0, len(data)):
		data[i] = data[i] - avg;
	return data;


# x = np.arange(0.0, 2000, 1)
# y = np.arange(0.0, 2000, 1)
# scrollingGraph(x,y,500)

def fromPKL(directory_and_name):
	with open(directory_and_name, 'rb') as input:
		retn = pickle.load(input)
	return retn;
file = "C:/Users/evan1/Desktop/Eye_tracking_experiment/result/Processed_Wavelets/Dad_left_far_notched.pkl"
data = fromPKL(file)
print(data)
normalized = normalize(data[0])
regularPlot(normalized, 10000)

