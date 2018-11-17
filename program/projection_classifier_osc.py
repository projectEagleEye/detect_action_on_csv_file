"""
    File Name: projection_classifier_osc.py
    Author: Barney Wei
    Date Created: 09/30/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import csv_reader
import numpy as np
from numpy.core.umath_tests import inner1d
import os
import projection_classifier
import csv_analytics


def get_action_osc(temp_ports_stream_matrix,
                   calibration_csv_file="ref_data/Calibration.csv",
                   action_name_array=("blink", "look down", "look left", "look right", "look up"),
                   data_type=" /muse/notch_filtered_eeg",
                   time_interval=20,
                   voltage_interval=(35, 35, 35, 35),
                   classification_threshold=2.4):
    """
    function that returns a list of performed actions based on eeg data from a csv file
    :param temp_ports_stream_matrix: NUMPY FLOAT 2D ARRAY
    :param calibration_csv_file: STRING - calibration file path directory (default="ref_data/Calibration.csv")
    :param action_name_array: STRING LIST - list of body actions defined by the reference_matrix in each file (default=
    ("blink", "look down", "look left", "look right", "look up"))
    :param data_type: STRING - name of the type of csv data to collect (default=" /muse/notch_filtered_eeg")
    :param time_interval: INT - the interval the signal holds above or below the voltage-interval on the
    calibration_mean to begin or end recording of the signal vector (default=20)
    :param voltage_interval: INT LIST - the interval above or below the calibration_mean which triggers recording
    of the signal vector when breached (for each signal port) (default=(35, 35, 35, 35))
    :param classification_threshold: FLOAT - projection scalars greater than this value will be indexed as -1 (not an
    action) (default=2.4)
    :return: STRING LIST - predicted actions that were performed based on eeg data
    """
    # convert data from csv file to a numpy array
    calibration_ports_stream = csv_reader.get_processed_data(calibration_csv_file,
                                                             sensor_cols=(2, 3, 4, 5),
                                                             data_type_col=1,
                                                             data_type=data_type,
                                                             transposition=False)
    # get calibration_mean
    calibration_mean = csv_analytics.get_calibration_mean(calibration_ports_stream)
    # filter out signal values from temp_ports_stream_matrix by calling the get_signal_3d_list function; if result is
    # empty, return []
    signal_3d_tensor = projection_classifier.get_signal_3d_list(temp_ports_stream_matrix,
                                                                calibration_mean,
                                                                time_interval,
                                                                voltage_interval)
    if not signal_3d_tensor:
        return []
    # read in reference vectors and their magnitudes
    reference_3d_tensor = projection_classifier.read_from_file(action_name_array,
                                                               footer='_ref',
                                                               file_type=".csv",
                                                               path=r"/ref_matrices/")
    reference_3d_tensor_magnitude = projection_classifier.read_from_file(action_name_array,
                                                                         footer='_mag',
                                                                         file_type=".csv",
                                                                         path=r"/ref_matrices/")
    # classify action by calling the get_projection_classification function
    classificaton_index = projection_classifier.get_projection_classification(signal_3d_tensor,
                                                                              reference_3d_tensor,
                                                                              reference_3d_tensor_magnitude,
                                                                              classification_threshold)

    # return a string list of classified actions from classification_index
    classificaton_string = []
    for index in classificaton_index:
        if index == -1:
            classificaton_string += ["detected something, but no reference action given"]
        else:
            classificaton_string += [action_name_array[index] + " detected"]

    return classificaton_string


# CODE TESTING
if __name__ == "__main__":
    # EXECUTION DURATION
    print("hi")
