"""
    File Name: classify_action_using_least_squares.py
    Author: Barney Wei
    Date Created: 09/02/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import numpy as np
import csv_analytics


def get_action(ports_stream, use_default_mean):
    """
    function that takes in "ports_stream" parameter to classify a body action based on the data
    stream
    :param ports_stream: NUMPY FLOAT ARRAY - a parallel stream of data from each of the sensors
    :return: STRING - classified action in a numpy list
    """

    # get mean value as base reference value for calibration
    mean_value = csv_analytics.get_calibration_mean(ports_stream, use_default_mean)

    # initiate counter numpy array for return
    counter = 0
    detected_action_arr = np.zeros((0, 2))

    # iterate through "port_stream" parameter to classify body action
    for value in ports_stream:
        counter += 1
        # TODO: write classification logic using the method of least squares from linear algebra
        if np.any(value > 1100):
            detected_action_arr = np.append(detected_action_arr, [[counter, "Blink detected"]], axis=0)

    return detected_action_arr


# CODE TESTING
if __name__ == "__main__":
    import csv_reader

    csv_file = "Blinks1.csv"
    sensor_cols = [2, 3, 4, 5]
    data_type_col = 1
    data_type = " Person0/eeg"
    transposition = False
    ports_stream = csv_reader.get_processed_data(csv_file,
                                                 sensor_cols,
                                                 data_type_col,
                                                 data_type,
                                                 transposition)
    use_default_mean = False

    action = get_action(ports_stream, use_default_mean)

    print(action)
    print("Total actions detected: ", action.shape[0])
    print("____________")
    print(ports_stream)
