# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# lenth = 1000

# fig, ax = plt.subplots()

# t = np.arange(0, lenth*500, 500)

# s = np.arange(0, lenth, 1)
# l, = plt.plot(t,s)
# plt.axis([0, 10, -500, 1000])

# axcolor = 'lightgoldenrodyellow'
# axpos = plt.axes([0.2, 0.1, 0.65, 0.03])

# spos = Slider(axpos, 'Pos', 0, 1000)

# def update(val):
#     pos = spos.val
#     ax.axis([pos,pos+1000,0,1000])
#     fig.canvas.draw_idle()

# spos.on_changed(update)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

x = np.arange(0.0, 1000, 1)
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
			new_x = self.x[val:(len(x)-1)]
			new_y = np.append(self.y[val:(len(self.x)-1)],np.zeros(self.length -len(self.y[val:(len(self.x)-1)])))
		return (new_x,new_y)
def scrollingGraph(x,y,interval):
	x = np.array(x)
	y = np.array(y)
	minY = 1000
	maxY = 0
	for i in y:
		if i > maxY:
			maxY = i
		if i < minY:
			minY = i
	maxY = maxY + 200
	minY = minY - 200
	MaxX = len(x)

	fig, ax = plt.subplots()
	plt.subplots_adjust(left=0.25, bottom=0.25)
	thing = GraphingData(x,y,interval)
	delta_f = interval/50

	l, = plt.plot(x[0:interval], y[0:interval], lw=2, color='red')
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


x = np.arange(0.0, 2000, 1)
y = np.arange(0.0, 2000, 1)
scrollingGraph(x,y,500)