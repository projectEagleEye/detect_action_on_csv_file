"""
    File Name: output_action.py
    Author: Barney Wei
    Date Created: 09/02/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import csv_reader
import classify_action

csv_file = "Blinks1.csv"
muse_data = csv_reader.get_data_from_csv(csv_file)
port_stream = muse_data[2]
output = classify_action.get_action(port_stream)
print(output)