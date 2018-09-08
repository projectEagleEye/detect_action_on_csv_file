"""
    File Name: projection_classifier.py
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
    :param ports_stream: PANDAS DATAFRAME
    :return: STRING - classified action in a numpy list
    """

    # get mean value as base reference value for calibration
    calibration_mean = csv_analytics.get_calibration_mean(ports_stream, use_default_mean)

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


def get_signal_vectors(ports_stream_df,
                       calibration_mean,
                       start_x_interval,
                       end_x_interval,
                       y_interval):
    """
    function that output signal vectors for each port found from the "ports_stream_df"
    :param ports_stream_df: PANDAS DATAFRAME
    :param calibration_mean: NUMPY FLOAT ARRAY
    :param start_x_interval: INT - the interval after the signal is and holds above or below the y-interval on the
    calibration_mean to begin recording of the signal vector
    :param end_x_interval: INT - the interval before the signal restores within the y-interval on the calibration_mean
    to end recording of the signal vector
    :param y_interval: INT - the interval above or below the calibration_mean which triggers recording of the signal
    vector when breached
    :return: FLOAT LIST - a 3D nested list that contains the signal vectors for each port
    """
    # TODO: get_signal_vectors()
    # get the number of rows and columns from the dataset
    num_rows = ports_stream_df.shape[0]
    num_cols = ports_stream_df.shape[1]

    # instantiate

    return


def get_reference_vectors(ports_stream_df,
                          calibration_mean,
                          start_x_interval,
                          end_x_interval,
                          y_interval):
    """
    function that output reference vectors for a body action
    :param ports_stream_df: PANDAS DATAFRAME - contain multiple repeated signal vectors of the desired body action
    :param calibration_mean: NUMPY FLOAT ARRAY
    :param start_x_interval: INT - the interval after the signal is and holds above or below the y-interval on the
    calibration_mean to begin recording of the signal vector
    :param end_x_interval: INT - the interval before the signal restores within the y-interval on the calibration_mean
    to end recording of the signal vector
    :param y_interval: INT - the interval above or below the calibration_mean which triggers recording of the signal
    vector when breached
    :return: FLOAT LIST - a 3D nested list that contains the signal vectors for each port
    """
    # TODO: get_reference_vectors()

    return


def get_projection_classification(signal_vectors, reference_vectors):
    """
    function that take each batch of signal vectors and compare to each batch of reference vectors by projection. The
    projection action with the highest value indicates highest similarity with the reference action
    :param signal_vectors: NUMPY ARRAY 2D LISTS - shape=(vector_length, num_ports, num_vectors)
    :param reference_vectors: NUMPY ARRAY 2D LISTS - shape=(vector_length, num_ports, num_actions)
    :return: INT - index into which action reference vector the scalar projection value was the highest
    """
    # TODO: get_projection_classification()

    return


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
