import socket
import threading
import time
import traceback
import selectors
import types

class Tello:
    """Wrapper to simply interactions with the Ryze Tello drone."""

    def __init__(self, local_ip, local_port, imperial=True, command_timeout=.3, tello_ip='192.168.10.1', tello_port=8889):
        """Binds to the local IP/port and puts the Tello into command mode.
        Args:
            local_ip (str): Local IP address to bind.
            local_port (int): Local port to bind.
            imperial (bool): If True, speed is MPH and distance is feet.
                             If False, speed is KPH and distance is meters.
            command_timeout (int|float): Number of seconds to wait for a response to a command.
            tello_ip (str): Tello IP.
            tello_port (int): Tello port.
        Raises:
            RuntimeError: If the Tello rejects the attempt to enter command mode.
        """

        self.abort_flag = False
        self.command_timeout = command_timeout
        self.imperial = imperial
        self.response = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tello_address = (tello_ip, tello_port)

        self.socket.bind((local_ip, local_port))

        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon=True

        self.receive_thread.start()

        if self.send_command('command') != 'OK':
            raise RuntimeError('Tello rejected attempt to enter command mode')

    def __del__(self):
        """Closes the local socket."""

        self.socket.close()

    def _receive_thread(self):
        """Listens for responses from the Tello.
        Runs as a thread, sets self.response to whatever the Tello last returned.
        """
        while True:
            try:
                self.response, ip = self.socket.recvfrom(256)
            except Exception:
                break

    def flip(self, direction):
        """Flips.
        Args:
            direction (str): Direction to flip, 'l', 'r', 'f', 'b', 'lb', 'lf', 'rb' or 'rf'.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        return self.send_command('flip %s' % direction)

    def get_battery(self):
        """Returns percent battery life remaining.
        Returns:
            int: Percent battery life remaining.
        """

        battery = self.send_command('battery?')

        try:
            battery = int(battery)
        except:
            pass

        return battery


    def get_flight_time(self):
        """Returns the number of seconds elapsed during flight.
        Returns:
            int: Seconds elapsed during flight.
        """

        flight_time = self.send_command('time?')

        try:
            flight_time = int(flight_time)
        except:
            pass

        return flight_time

    def get_speed(self):
        """Returns the current speed.
        Returns:
            int: Current speed in KPH or MPH.
        """

        speed = self.send_command('speed?')

        try:
            speed = float(speed)

            if self.imperial is True:
                speed = round((speed / 44.704), 1)
            else:
                speed = round((speed / 27.7778), 1)
        except:
            pass

        return speed

    def land(self):
        """Initiates landing.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        return self.send_command('land')

    def move(self, direction, distance):
        """Moves in a direction for a distance.
        This method expects meters or feet. The Tello API expects distances
        from 20 to 500 centimeters.
        Metric: .1 to 5 meters
        Imperial: .7 to 16.4 feet
        Args:
            direction (str): Direction to move, 'forward', 'back', 'right' or 'left'.
            distance (int|float): Distance to move.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        distance = float(distance)

        if self.imperial is True:
            distance = int(round(distance * 30.48))
        else:
            distance = int(round(distance * 100))

        return self.send_command('%s %s' % (direction, distance))

    def move_backward(self, distance):
        """Moves backward for a distance.
        See comments for Tello.move().
        Args:
            distance (int): Distance to move.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        return self.move('back', distance)

    def move_down(self, distance):
        """Moves down for a distance.
        See comments for Tello.move().
        Args:
            distance (int): Distance to move.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        return self.move('down', distance)

    def move_forward(self, distance):
        """Moves forward for a distance.
        See comments for Tello.move().
        Args:
            distance (int): Distance to move.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        return self.move('forward', distance)

    def move_left(self, distance):
        """Moves left for a distance.
        See comments for Tello.move().
        Args:
            distance (int): Distance to move.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        return self.move('left', distance)

    def move_right(self, distance):
        """Moves right for a distance.
        See comments for Tello.move().
        Args:
            distance (int): Distance to move.
        """
        return self.move('right', distance)

    def move_up(self, distance):
        """Moves up for a distance.
        See comments for Tello.move().
        Args:
            distance (int): Distance to move.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        return self.move('up', distance)

    def send_command(self, command):
        """Sends a command to the Tello and waits for a response.
        If self.command_timeout is exceeded before a response is received,
        a RuntimeError exception is raised.
        Args:
            command (str): Command to send.
        Returns:
            str: Response from Tello.
        Raises:
            RuntimeError: If no response is received within self.timeout seconds.
        """

        self.abort_flag = False
        timer = threading.Timer(self.command_timeout, self.set_abort_flag)

        self.socket.sendto(command.encode('utf-8'), self.tello_address)

        timer.start()

        while self.response is None:
            if self.abort_flag is True:
                raise RuntimeError('No response to command')

        timer.cancel()

        response = self.response.decode('utf-8')
        self.response = None

        return response

    def set_abort_flag(self):
        """Sets self.abort_flag to True.
        Used by the timer in Tello.send_command() to indicate to that a response
        timeout has occurred.
        """

        self.abort_flag = True

    def set_speed(self, speed):
        """Sets speed.
        This method expects KPH or MPH. The Tello API expects speeds from
        1 to 100 centimeters/second.
        Metric: .1 to 3.6 KPH
        Imperial: .1 to 2.2 MPH
        Args:
            speed (int|float): Speed.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        speed = float(speed)

        if self.imperial is True:
            speed = int(round(speed * 44.704))
        else:
            speed = int(round(speed * 27.7778))

        return self.send_command('speed %s' % speed)

    def takeoff(self):
        """Initiates take-off.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        return self.send_command('takeoff')

    def rotate_cw(self, degrees):
        """Rotates clockwise.
        Args:
            degrees (int): Degrees to rotate, 1 to 360.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """

        return self.send_command('cw %s' % degrees)

    def rotate_ccw(self, degrees):
        """Rotates counter-clockwise.
        Args:
            degrees (int): Degrees to rotate, 1 to 360.
        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.
        """
        return self.send_command('ccw %s' % degrees)






sel = selectors.DefaultSelector() # make the selector that respond to I/O events

host = '127.0.0.1'  # Standard loopback interface address (localhost)
port = 65435        # Port to listen on (non-privileged ports are > 1023)# ...

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
            signal = (recv_data.decode('ascii'))
            if signal == "takeoff":
                tim.takeoff()
            elif signal == "land":
                tim.land()

            # data.word =  data.word + [recv_data.decode('ascii')]
            # for i in data.word:
            #   data.outb = data.outb + " ".encode('ascii') + i.encode('ascii')
            # sent = sock.send(data.outb) # sent returns the characters it sent
            # data.outb = '0' 
    # if mask & selectors.EVENT_WRITE:



tim = Tello("192.168.10.2", 2001)

while True:
    events = sel.select(timeout=None) 
    for key, mask in events:
        if key.data is None: # if it's a listening socket
            accept_wrapper(key.fileobj)

        else:
            service_connection(key, mask)

tim = Tello("192.168.10.2", 2001)
