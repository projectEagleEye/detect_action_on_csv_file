# a test where different client would send a different word, then the server would join it into a sentance and print it out. 
import selectors
import socket
import types

host = '127.0.0.1'  # Standard loopback interface address (localhost)
port = 65436      # Port to listen on (non-privileged ports are > 1023)# ...
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # initialize server
sock.setblocking(False) # initiate server
sock.connect_ex((host, port)) # connect to server

for i in range (0, 10):
	text = str(i).encode('ascii')
	sock.send(text)

if 0 == 3:
	while True:
		text = input().encode('ascii')
		sock.send(text)
		try:
			recv_data = sock.recv(1024)
			print(recv_data.decode('ascii'))
		except:
			k = 3
		if text.decode('ascii') == 'end':	
			sock.close()
			print('closed')
			break



		