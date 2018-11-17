import csv2matrix
import numpy as np
list=csv2matrix.get_raw_data('Blinks1.csv')
print(list(np.transpose(list)))