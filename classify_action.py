"""
    File Name: classify_action.py
    Author: Barney Wei
    Date Created: 09/02/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import numpy as np


def get_action(port_stream):
    """
    function that takes in "port_stream" parameter to classify a body action based on the data
    stream
    :param port_stream: FLOAT LIST - a stream of data from an individual port (sensor)
    :return: STRING - classified action
    """
    # checks if "port_stream" parameter is a list type
    if not isinstance(port_stream, list):
        print("ERROR - calibration_retrieve_mean(port_stream): port_stream is not a list type")
        return None

    # get mean value as base reference value for calibration
    mean_value = get_calibration_mean(port_stream)

    # iterate through "port_stream" parameter to classify body action
    for value in port_stream:
        # convert value from type string to float
        value = float(value)
        if value > 900:
            return "Blink detected"
    return None


def get_calibration_mean(port_stream):
    """
    function for calibration that finds the mean of the "port_stream" parameter to be used by
    "classify_action" function as a base reference value
    :param port_stream: FLOAT LIST
    :return: FLOAT
    """
    # try to find mean of "port_stream" parameter
    try:
        return np.mean(port_stream)
    except:
        print("ERROR - get_calibration_mean: port_stream is a list but unable"
              " to find mean")