"""
    File Name: projection_classifier.py
    Author: Barney Wei
    Date Created: 09/02/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import numpy as np
from numpy.core.umath_tests import inner1d
import csv_analytics


def get_action(ports_stream, use_default_mean):
    """
    function that takes in "ports_stream" parameter to classify a body action based on the data
    stream
    :param ports_stream: NUMPY 2D ARRAY
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


def get_signal_3d_tensor(ports_stream,
                         calibration_mean,
                         time_interval,
                         voltage_interval):
    """
    function that output signal vectors for each port found from the "ports_stream_df"
    :param ports_stream: NUMPY 2D ARRAY
    :param calibration_mean: NUMPY 1D FLOAT ARRAY
    :param time_interval: INT - the interval the signal holds above or below the voltage-interval on the
    calibration_mean to begin or end recording of the signal vector
    :param voltage_interval: BOOLEAN LIST - the interval above or below the calibration_mean which triggers recording
    of the signal vector when breached (for each signal port)
    :return: LIST OF NUMPY 2D ARRAY - contains the signal vectors for each port as a list of 2D numpy arrays
    """

    # instantiate variables
    num_cols = ports_stream.shape[1]
    signal_3d_tensor = []
    temp_signal_matrix = np.zeros((0, num_cols))
    is_filled_by_signal = False
    is_iterate = True
    skip_check = time_interval  # countdown that allows iterate through rows in ports_stream without copying

    # instantiate a rolling FIFO numpy array to extract signal vectora once all values of in a column is greater or
    # less than calibration_mean +- voltage_interval, respectively. The shape of the array is (time_interval, num_cols)
    rolling_fifo = np.full((time_interval, num_cols), calibration_mean)

    # iterate through each row of the data to extract the signal matrix
    for row in ports_stream:
        if skip_check is not 0:
            skip_check -= 1

        row_keep_dims = row.reshape((1, num_cols))

        rolling_fifo = np.delete(rolling_fifo, 0, 0)
        rolling_fifo = np.append(rolling_fifo, row_keep_dims, axis=0)

        if (is_filled_by_signal is False and is_iterate is True and skip_check is 0) or\
                (is_filled_by_signal is True and is_iterate is True and skip_check is 0):
            # checks if all values of any column of the rolling_fifo is greater than the voltage_interval range
            for i in range(rolling_fifo.shape[1]):
                col_keep_dims = rolling_fifo.T[i].reshape((time_interval, 1))
                if (is_filled_by_signal is False) and (is_iterate is True):
                    if (np.all(col_keep_dims > (calibration_mean + voltage_interval[i])) or
                            np.all(col_keep_dims < (calibration_mean - voltage_interval[i]))):
                        skip_check = time_interval
                        is_filled_by_signal = True
                        is_iterate = False

            # checks if all values of all columns of the rolling_fifo is filled by mean
            if (is_filled_by_signal is True) and (is_iterate is True):
                if (np.all(rolling_fifo < (calibration_mean + voltage_interval)) and
                        np.all(rolling_fifo > (calibration_mean - voltage_interval))):
                    skip_check = time_interval
                    is_filled_by_signal = False
                    is_iterate = False

        # if all values of any column of the rolling_fifo is greater than the voltage_interval range, then add all of
        # rolling_fifo to temp signal matrix
        if (is_filled_by_signal is True) and (is_iterate is False):
            temp_signal_matrix = np.append(temp_signal_matrix, rolling_fifo, axis=0)

            is_iterate = True

        # if all values of all columns of the rolling_fifo is filled by mean, delete rolling_fifo section from temp
        # signal_matrix
        elif (is_filled_by_signal is False) and (is_iterate is False):
            num_keep_rows = temp_signal_matrix.shape[0] - time_interval
            temp_signal_matrix = temp_signal_matrix[0:num_keep_rows, :].reshape((num_keep_rows, num_cols))

            signal_3d_tensor += [temp_signal_matrix]
            temp_signal_matrix = np.zeros((0, num_cols))

            is_iterate = True

        # append each new appended line from rolling_fifo to temp_signal_matrix
        elif (is_filled_by_signal is True) and (is_iterate is True):
            rolling_fifo_last_row = rolling_fifo[rolling_fifo.shape[0]-1, :].reshape((1, num_cols))
            temp_signal_matrix = np.append(temp_signal_matrix, rolling_fifo_last_row, axis=0)

        # iterate through rows in ports_stream without copying
        elif (is_filled_by_signal is False) and (is_iterate is True):
            continue

        else:
            print("Error in get_signal_3d_tensor")

    return signal_3d_tensor


def get_reference_matrix(ports_stream,
                         calibration_mean,
                         time_interval,
                         voltage_interval):
    """
    function that output reference vectors for a body action
    :param ports_stream: NUMPY 2D ARRAY - contain multiple repeated signal vectors of the desired body action
    :param calibration_mean: NUMPY 1D FLOAT ARRAY
    :param time_interval: INT - the interval the signal holds above or below the voltage-interval on the
    calibration_mean to begin or end recording of the signal vector
    :param voltage_interval: INT - the interval above or below the calibration_mean which triggers recording of the signal
    vector when breached
    :return: NUMPY 2D ARRAY - signal vectors for each port
    """
    # get the number of rows and columns from the dataset
    num_cols = ports_stream.shape[1]

    # call get_signal_3d_tensor function to get a list of signal vectors before processing
    signal_3d_tensor = get_signal_3d_tensor(ports_stream, calibration_mean, time_interval, voltage_interval)

    # get max signal vector row length in the signal_3d_tensor list
    max_rows = 0
    for signal_matrix in signal_3d_tensor:
        if max_rows < signal_matrix.shape[0]:
            max_rows = signal_matrix.shape[0]

    # initialize temporary 3d tensor to hold all signal matrices after zero padding
    temp_signal_3d_tensor = np.zeros((max_rows, num_cols, 0))
    for signal_matrix in signal_3d_tensor:
        # resize signal vectors by adding 0s
        num_padding = max_rows - signal_matrix.shape[0]
        signal_matrix = np.pad(signal_matrix, ((0, num_padding), (0, 0)), "constant", constant_values=np.nan)

        temp_signal_3d_tensor = np.dstack((temp_signal_3d_tensor, signal_matrix))

    # get reference vector through finding the mean among vectors of the same column and skipping nan values
    reference_matrix = np.nanmean(temp_signal_3d_tensor, axis=2)

    return reference_matrix


def get_reference_matrix_magnitude(reference_matrix):
    """
    function that returns the magnitude of each column in the reference_matrix as a numpy array
    :param reference_matrix: NUMPY 2D ARRAY
    :return: NUMPY 1D ARRAY
    """
    reference_matrix_magnitude = np.sqrt(inner1d(reference_matrix.T, reference_matrix.T))

    return reference_matrix_magnitude.reshape(1, reference_matrix.shape[1])


def get_projection_classification(signal_3d_tensor, reference_3d_tensor, reference_3d_tensor_magnitude):
    """
    function that take each batch of signal vectors and compare to each batch of reference vectors by projection. The
    projection action with the highest value indicates highest similarity with the reference action
    :param signal_3d_tensor: LIST OF NUMPY 2D ARRAY - shape=(vector_length, num_ports, num_vectors)
    :param reference_3d_tensor: LIST OF NUMPY 2D ARRAY - shape=(vector_length, num_ports, num_actions)
    :param reference_3d_tensor_magnitude: NUMPY 1D ARRAY - shape(num_actions, num_ports)
    :return: NUMPY 1D ARRAY - index into which action reference vector the scalar projection value was the highest
    """
    # TODO: get_projection_classification()
    # instantiate variables
    num_cols = signal_3d_tensor[0].shape[1]
    projection_matrix = np.zeros((0, len(reference_3d_tensor)))

    # iterate through signal_3d_tensor for projection
    for signal_matrix in signal_3d_tensor:
        projection_array = []
        for i in range(len(reference_3d_tensor)):
            # 0-pad matrix to reach equal number of rows
            num_padding = abs(signal_matrix.shape[0] - reference_matrix.shape[0])
            if signal_matrix.shape[0] < reference_3d_tensor[i].shape[0]:
                temp_signal_matrix = np.pad(signal_matrix, ((0, num_padding), (0, 0)), "constant", constant_values=0)
            elif signal_matrix.shape[0] > reference_3d_tensor[i].shape[0]:
                temp_reference_matrix = np.pad(reference_3d_tensor[i], ((0, num_padding), (0, 0)), "constant", constant_values=0)

            # extract the scalar when projecting signal_matrix onto reference_matrix
            temp_element = inner1d(signal_matrix.T, signal_matrix.T)  # dot product
            temp_element = np.divide(temp_element, np.square(reference_3d_tensor_magnitude[i]))
            temp_element = np.sum(temp_element, axis=1)
            temp_element = np.subtract(temp_element, num_cols)
            temp_element = np.abs(temp_element)
            projection_array += temp_element
        projection_matrix = np.append(projection_matrix, projection_array, axis=0)

    # determine reference_3d_tensor index for classify action
    classificaton_index = np.argmin(projection_matrix, axis=1)
    
    return classificaton_index


def save_reference_matrix_to_txt(reference_matrix, action_name, path="/ref_matrix"):
    """
    function that saves reference_matrix to a text file
    :param reference_matrix: NUMPY 2D ARRAY
    :param action_name: STRING - the body action defined by the reference_matrix
    :param path: STRING - save path (default="/ref_matrix")
    :return: BOOLEAN - whether if save is successful
    """
    # TODO: save_reference_matrix_to_txt() - use np.savetxt()


def read_reference_matrix_from_txt(file_name, path="/ref_matrix"):
    """
    function that reads reference_matrix from text file
    :param file_name: STRING
    :param path: STRING - read path (default="/ref_matrix")
    :return: NUMPY ARRAY
    """
    # TODO: read_reference_matrix_from_txt() - use np.loadtxt()


# CODE TESTING
if __name__ == "__main__":
    # EXECUTION DURATION
    import time
    start_time = time.time()

    import csv_reader

    csv_file = "Blinks30.csv"
    sensor_cols = [2, 3, 4, 5]
    data_type_col = 1
    data_type = " Person0/eeg"
    transposition = False
    ports_stream = csv_reader.get_processed_data(csv_file,
                                                 sensor_cols,
                                                 data_type_col,
                                                 data_type,
                                                 transposition)
    calibration_mean = csv_analytics.get_calibration_mean(ports_stream)
    time_interval = 12
    voltage_interval = [80, 20, 25, 80]

    # TEST: get_signal_3d_tensor()
    print("TEST: get_signal_3d_tensor()")
    signal_3d_tensor = get_signal_3d_tensor(ports_stream, calibration_mean, time_interval, voltage_interval)
    # print(signal_3d_tensor)
    for i in range(len(signal_3d_tensor)):
        print(signal_3d_tensor[i].shape)
    print(len(signal_3d_tensor))
    print("My program took", time.time() - start_time, "seconds to run")
    print("____________________________")

    # TEST: get_reference_matrix()
    print("TEST: get_reference_matrix()")
    reference_matrix = get_reference_matrix(ports_stream, calibration_mean, time_interval, voltage_interval)
    # print(reference_matrix)
    print(reference_matrix.shape)
    print("____________________________")

    # TEST: get_reference_matrix_magnitude()
    reference_matrix_magnitude = get_reference_matrix_magnitude(reference_matrix)
    print(reference_matrix_magnitude)

    # TEST: get_projection_classification()
    """print("TEST: get_projection_classification()")
    reference_3d_tensor = [reference_matrix]
    print("____________________________")"""
