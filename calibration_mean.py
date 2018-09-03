"""
    File Name: calibration_mean.py
    Author: Barney Wei
    Date Created: 09/03/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import numpy as np


def get_calibration_mean(ports_stream, use_default_mean=False):
    """
    function for calibration that finds the mean of the "port_stream" parameter to be used by the classify action
    function as a base reference value
    :param ports_stream: NUMPY FLOAT ARRAY
    :param use_default_mean: BOOLEAN - use default calibration mean (default=842.6)
    :return: NUMPY FLOAT ARRAY - default.shape=(1,) otherwise (num_ports,)
    """
    if use_default_mean:
        return np.array([842.6])

    # check if "port_stream" parameter is of type numpy array, if not display error message and return None
    if isinstance(ports_stream, np.ndarray):
        return np.mean(ports_stream, axis=0)
    else:
        print("ERROR - get_calibration_mean: port_stream is not of type numpy array")
        return None


# CODE TESTING
if __name__ == "__main__":
    # import libraries and files
    import csv_reader

    # initialize variables
    csv_file = "Blinks1.csv"
    sensor_cols = [2, 3, 4, 5]
    data_type_col = 1
    data_type = " Person0/eeg"
    transposition = False
    port = 1
    use_default_mean = False

    # call functions
    ports_stream = csv_reader.get_processed_data(csv_file,
                                                 sensor_cols,
                                                 data_type_col,
                                                 data_type,
                                                 transposition)
    calibration_mean = get_calibration_mean(ports_stream, use_default_mean)
    print(calibration_mean)
    print(calibration_mean.shape)
    print("_________________")
    print(ports_stream)
    print(ports_stream.shape)
