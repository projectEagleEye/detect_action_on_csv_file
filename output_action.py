"""
    File Name: output_action.py
    Author: Barney Wei
    Date Created: 09/02/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import csv_reader
import classify_action_using_least_squares


def get_action(csv_file,
               sensor_cols,
               data_type_col,
               data_type,
               transposition,
               use_default_mean):
    """
    function that gets the list of action(s) detected
    :param csv_file: STRING - file path directory
    :param sensor_cols: INT ARRAY - integer index of sensor columns in csv file
    :param data_type_col: INT - integer index of "data_type"
    :param data_type: STRING - name of the type of csv data to collect (eg. "Person0/eeg"
    :param transposition: BOOLEAN - transpose the csv data output 2D numpy array (default=True)
    :return: NUMPY ARRAY - [integer, detected action]
    """
    # get processed data
    ports_stream = csv_reader.get_processed_data(csv_file,
                                                 sensor_cols,
                                                 data_type_col,
                                                 data_type,
                                                 transposition)

    return classify_action_using_least_squares.get_action(ports_stream, use_default_mean)


# CODE TESTING
if __name__ == "__main__":
    csv_file = "Blinks1.csv"
    sensor_cols = [2, 3, 4, 5]
    data_type_col = 1
    data_type = " Person0/eeg"
    transposition = False
    use_default_mean = False

    detected_action = get_action(csv_file,
                                 sensor_cols,
                                 data_type_col,
                                 data_type,
                                 transposition,
                                 use_default_mean)

    print(detected_action)
