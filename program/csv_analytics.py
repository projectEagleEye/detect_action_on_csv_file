"""
    File Name: csv_analytics.py
    Author: Barney Wei
    Date Created: 09/03/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv_reader

# declare default mean for get_calibration functions
default_mean = 847.39

def get_calibration_mean(ports_stream, use_default_mean=False):
    """
    function for calibration that finds the mean of the of the numpy array to be used by the classify action
    function as a base reference value
    :param ports_stream: NUMPY 2D ARRAY
    :param use_default_mean: BOOLEAN - use default calibration mean (default=842.6)
    :return: NUMPY FLOAT ARRAY - size of number of columns (ports) in the dataframe
    """
    if use_default_mean:
        return np.full((1, ports_stream.shape[1]), default_mean)
    else:
        return np.mean(ports_stream, axis=0, keepdims=True)


def get_calibration_mean_df(ports_stream_df, use_default_mean=False):
    """
    function for calibration that finds the mean of the of the dataframe to be used by the classify action
    function as a base reference value
    :param ports_stream_df: PANDAS DATAFRAME
    :param use_default_mean: BOOLEAN - use default calibration mean (default=842.6)
    :return: NUMPY FLOAT ARRAY - size of number of columns (ports) in the dataframe
    """
    if use_default_mean:
        return np.full((1, ports_stream_df.shape[1]), default_mean)
    else:
        return np.array(ports_stream_df.mean()).reshape((1, ports_stream_df.shape[1]))


def get_describe(ports_stream_df):
    """
    function that uses a pandas dataframe to get:
    1) min/max
    2) mode/mean
    3) 1st / 3rd quartile
    :param ports_stream_df: PANDAS DATAFRAME
    :return: PANDAS DATAFRAME
    """

    return ports_stream_df.describe()


def print_histogram(ports_stream_df,
                    file_name,
                    n_bins=250,
                    has_density=False,
                    xlim=None,
                    show_grid=True,
                    do_save_fig=False,
                    file_name_and_path=None,
                    dpi=None,
                    is_transparent=False):
    """
    function that uses a pandas dataframe to display a histogram for each data column
    :param ports_stream_df: PANDAS DATAFRAME
    :param file_name: STRING - file name to distinguish among other csv files
    :param n_bins: INT - set number of bin divisions (default=250)
    :param has_density: BOOLEAN - set density (y-axis): True=probability || False=count (default=False)
    :param xlim: DOUBLE LIST || NONE - (<x.min>, <x.max>) || None (default=None)
    :param show_grid: BOOLEAN - whether to show grid on histogram (default=True)
    :param do_save_fig: BOOLEAN - whether to save the figure (default=False)
    :param file_name_and_path: STRING - file directory, name and type (default=None)
    :param dpi: NONE || INT - dpi of exported figure (default=None)
    :param is_transparent: BOOLEAN - whether if exported figure is transparent (default=False)
    :return: BOOLEAN - whether if the function is successful
    """
    try:
        for col in range(ports_stream_df.shape[1]):
            # isolate each column of data
            data_col = ports_stream_df.iloc[:, col]

            # plot histogram
            plt.hist(x=data_col, bins=n_bins, density=has_density)

            # add histogram labels and characteristics
            plt.xlabel("Voltage (mV)")
            if has_density:
                plt.ylabel("Percentage (%)")
            else:
                plt.ylabel("Frequency")
            # retrieve column name from each column in the dataframe
            plt.title("Histogram: " + ports_stream_df.dtypes.index[col] + " (" + file_name + ")")
            plt.xlim(xlim)
            plt.grid(show_grid)

            # output histogram
            plt.show()

            # whether to save figure
            if do_save_fig:
                plt.savefig(fname=file_name_and_path, dpi=dpi, transparent=is_transparent)

        return True

    except:
        print("ERROR - print_histogram()")
        return False


def print_scatter_plot(ports_stream_df,
                       file_name,
                       marker="*",
                       linewidth=0.01,
                       show_grid=True,
                       do_save_fig=False,
                       file_name_and_path=None,
                       dpi=None,
                       is_transparent=False):
    """
    function that uses a pandas dataframe to display a scatter plot for each data column
    :param ports_stream_df: PANDAS DATAFRAME
    :param file_name: STRING - file name to distinguish among other csv files
    :param marker: STRING || NONE - marker type (default:'*')
    :param linewidth: INT || NONE - linewidth (default:0.01)
    :param show_grid: BOOLEAN - whether to show grid on scatter plot (default=True)
    :param do_save_fig: BOOLEAN - whether to save the figure (default=False)
    :param file_name_and_path: STRING - file directory, name and type (default=None)
    :param dpi: NONE || INT - dpi of exported figure (default=None)
    :param is_transparent: BOOLEAN - whether if exported figure is transparent (default=False)
    :return: BOOLEAN - whether if the function is successful
    """
    try:
        for col in range(ports_stream_df.shape[1]):
            # isolate each column of data
            data_col = ports_stream_df.iloc[:, col]

            # make a list of x values
            x_val = [range(1, 1 + data_col.shape[0])]

            # plot scatter plot
            plt.scatter(x=x_val, y=data_col, marker=marker, linewidths=linewidth)

            # add scatter plot labels and characteristics
            plt.xlabel("Time (1/256 sec)")
            plt.ylabel("Voltage (mV)")
            # retrieve column name from each column in the dataframe
            plt.title("Scatter Plot: " + ports_stream_df.dtypes.index[col] + " (" + file_name + ")")
            plt.grid(show_grid)

            # output scatter plot
            plt.show()

            # whether to save figure
            if do_save_fig:
                plt.savefig(fname=file_name_and_path, dpi=dpi, transparent=is_transparent)

        return True

    except:
        print("ERROR - print_scatter_plot()")
        return False


def get_dataframe(csv_file,
                  sensor_cols,
                  data_type_col,
                  data_type,
                  transposition):
    """
    function that returns a pandas dataframe from a csv file
    :param csv_file: STRING - file path directory
    :param sensor_cols: INT ARRAY - integer index of sensor columns in csv file
    :param data_type_col: INT - integer index of "data_type"
    :param data_type: STRING - name of the type of csv data to collect (eg. " Person0/eeg"
    :param transposition: BOOLEAN - transpose the csv data output 2D numpy array
    :return: PANDAS DATAFRAME
    """
    # get processed data from "csv_reader"
    ports_stream = csv_reader.get_processed_data(csv_file,
                                                 sensor_cols,
                                                 data_type_col,
                                                 data_type,
                                                 transposition)

    # create pandas DataFrame from numpy array
    row_index = [(i + 1) for i in range(ports_stream.shape[0])]
    col_index = ["port " + str(i + 1) for i in range(ports_stream.shape[1])]
    ports_stream_df = pd.DataFrame(data=ports_stream, index=row_index, columns=col_index)

    return ports_stream_df


# CODE TESTING
if __name__ == "__main__":
    # get dataframe parameters
    csv_file = "ref_data/Blinks30.csv"
    sensor_cols = [2, 3, 4, 5]
    data_type_col = 1
    data_type = " /muse/notch_filtered_eeg"
    transposition = False
    # get histogram parameters
    n_bins = 300
    has_density = False
    xlim = None
    show_grid = True


    # get ports_stream
    ports_stream = csv_reader.get_processed_data(csv_file,
                                                 sensor_cols,
                                                 data_type_col,
                                                 data_type,
                                                 transposition)


    # get dataframe
    ports_stream_df = get_dataframe(csv_file,
                                    sensor_cols,
                                    data_type_col,
                                    data_type,
                                    transposition)
    # get analytics
    ports_stream_analytics = get_describe(ports_stream_df)
    # get histogram
    # print_histogram(ports_stream_df, csv_file)
    # get scatter plot
    print_scatter_plot(ports_stream_df, csv_file)

    # get calibration mean
    calibration_mean_df = get_calibration_mean_df(ports_stream_df)
    calibration_mean = get_calibration_mean(ports_stream)

    print(ports_stream_analytics)
    print(ports_stream_df.shape)
    print(calibration_mean_df.shape)
    print("____________________")

    print(type(ports_stream_df.values))

