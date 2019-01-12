# a test where different client would send a different word, then the server would join it into a sentance and print it out. 

import selectors
import socket
import types

sel = selectors.DefaultSelector() # make the selector that respond to I/O events

host = '127.0.0.1'  # Standard loopback interface address (localhost)
port = 65436        # Port to listen on (non-privileged ports are > 1023)# ...

lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # initialize server
lsock.bind((host, port)) # bind host and port
lsock.listen() # start trying to reach clients
# print('listening on', (host, port))
lsock.setblocking(False) # make sure nothing is being blocked
sel.register(lsock, selectors.EVENT_READ, data=None) # respond to anything that lsock gives.

rtv = types.SimpleNamespace(word=b'')
def accept_wrapper(sock): # everytime a new client joins, a new thread is added. 
	conn, addr = sock.accept()  # Should be ready to read
	print('accepted connection from', addr)
	conn.setblocking(False)
	data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'', word = [] ) #name space is an object that stores lots of things
	events = selectors.EVENT_READ | selectors.EVENT_WRITE
	sel.register(conn, events, data=data) # add additional input to selector
def service_connection(key, mask): # deal with data from the client
	sock = key.fileobj # socket object
	data = key.data # data from server
	if mask & selectors.EVENT_READ:
		recv_data = sock.recv(1024)  # Should be ready to read
		# print("Msg:<", recv_data.decode('ascii'), ">was receive from client ", data.addr)
		if recv_data.decode('ascii') == "exit":
			print('closing connection to', data.addr) # if the recv_data is null, the socket will be closed, so it never waits
			sel.unregister(sock)# close socket and unregister selector
			sock.close()
		elif recv_data:
			print(recv_data.decode('ascii'))
			# data.word =  data.word + [recv_data.decode('ascii')]
			# for i in data.word:
			# 	data.outb = data.outb + " ".encode('ascii') + i.encode('ascii')
			# sent = sock.send(data.outb) # sent returns the characters it sent
			# data.outb = '0' 
	# if mask & selectors.EVENT_WRITE:





while True:
	events = sel.select(timeout=None) 
	for key, mask in events:
		if key.data is None: # if it's a listening socket
			accept_wrapper(key.fileobj)

		else:
			service_connection(key, mask)