"""
    File Name: output_action.py
    Author: Barney Wei
    Date Created: 09/02/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import csv_reader
import classify_action


# prints detected action
def get_action_and_calibration_mean(csv_file, sensor_cols, data_type_col, data_type, transposition, port):
    """
    function that gets the list of action(s) detected and the calibration mean inside of a list
    :param csv_file: STRING - file path directory
    :param sensor_cols: INT ARRAY - integer index of sensor columns in csv file
    :param data_type_col: INT - integer index of "data_type"
    :param data_type: STRING - name of the type of csv data to collect (eg. "Person0/eeg"
    :param transposition: BOOLEAN - transpose the csv data output 2D numpy array (default=True)
    :param port: INT - port number to index into "processed_data"
    :return: 2D LIST - action_mean_data = [action, calibration_mean]
    """
    # get processed data
    processed_data = csv_reader.get_processed_data(csv_file, sensor_cols, data_type_col, data_type, transposition)
    # isolate data from one sensor
    port_stream = processed_data[port]

    # returns [action, calibration_mean]
    return classify_action.get_action(port_stream)


if __name__ == "__main__":
    # TESTING CODE
    csv_file = "Blinks1.csv"
    sensor_cols = [2, 3, 4, 5]
    data_type_col = 1
    data_type = " Person0/eeg"
    transposition = True
    port = 1

    action_mean_data = get_action_and_calibration_mean(csv_file,
                                                       sensor_cols,
                                                       data_type_col,
                                                       data_type,
                                                       transposition,
                                                       port)

    action = action_mean_data[0]
    calibration_mean = action_mean_data[1]

    print(action_mean_data)
    print("_______________________")
    print(action)
    print("_______________________")
    print(calibration_mean)
