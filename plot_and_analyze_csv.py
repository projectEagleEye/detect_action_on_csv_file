"""
    File Name: plot_and_analyze_csv.py
    Author: Barney Wei
    Date Created: 09/03/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import numpy as np
import pandas as pd
import csv_reader

# get analysis info about csv data
def get_analytics(raw_data, sensor_cols, data_type_col, data_type, transposition):
    """
    function that analyzes processed csv data to get:
    1) min/max
    2) mode/mean
    3) 1st / 3rd quartile
    :param raw_data: STRING 2-D LIST - raw csv data
    :param sensor_cols: INT ARRAY - integer index of sensor columns in csv file
    :param data_type_col: INT - integer index of "data_type"
    :param data_type: STRING - name of the type of csv data to collect (eg. " Person0/eeg"
    :param data_type: BOOLEAN - transpose the csv data output 2D numpy array
    :return: NUMPY FLOAT ARRAY - processed data
    """


