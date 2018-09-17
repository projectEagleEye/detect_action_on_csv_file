import numpy as np
import csv_reader
import csv_analytics
from numpy.core.umath_tests import inner1d
import os

list = np.zeros((0, 3))
arr = np.array([1, 2, 3]).reshape((1, 3))

if __name__ == "__main__":
    csv_file = "ref_matrices/blink_ref.csv"
    ports_stream = np.array(csv_reader.get_raw_data(csv_file))
    ports_stream = ports_stream.astype(np.float)
    print(np.array(ports_stream))

