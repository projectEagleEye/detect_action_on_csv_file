"""
    File Name: output_action_osc.py
    Author: Barney Wei
    Date Created: 09/02/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import argparse
import numpy as np
from pythonosc import dispatcher
from pythonosc import osc_server
import projection_classifier

# create global temporary array to hold osc data
temp_ports_stream_matrix = np.zeros((0, 4))

def eeg_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    """
    function that is called by the osc framework to deliver the osc data stream one row each call
    :param unused_addr: STRING
    :param args: STRING
    :param ch1: FLOAT
    :param ch2: FLOAT
    :param ch3: FLOAT
    :param ch4: FLOAT
    :return: NONE
    """
    # convert osc data to numpy array and append to temp_ports_stream_matrix
    ports_stream_array = np.array([ch1, ch2, ch3, ch4])
    num_cols = ports_stream_array.size
    global temp_ports_stream_matrix
    temp_ports_stream_matrix = np.append(temp_ports_stream_matrix, ports_stream_array).reshape((-1, num_cols))

    # call projection_classifier.get_action_osc
    classification_string = projection_classifier.get_action_osc(temp_ports_stream_matrix)
    
    # checks if empty, if not, then clear temp_ports_stream_matrix and print the action detected
    if not classification_string:
        # begin FIFO on temp_ports_stream_matrix once its .shape[0] exceeds a limit
        if temp_ports_stream_matrix.shape[0] > 800:
            temp_ports_stream_matrix = np.delete(temp_ports_stream_matrix, 0, 0)
        return
    else:
        temp_ports_stream_matrix = np.zeros((0, num_cols))
        print(classification_string)
        return


# CODE TESTING
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
                        default="127.0.0.1",
                        help="The ip to listen on")
    parser.add_argument("--port",
                        type=int,
                        default=5053,  # make sure you change the port every time you wanna run the code
                        # for example next one would be 5052 :)

                        help="The port to listen on")
    parser.add_argument("--serial",
                        default="",
                        help="Arduino serial port")
    args = parser.parse_args()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/debug", print)
    dispatcher.map("/muse/notch_filtered_eeg", eeg_handler, "EEG")

    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
