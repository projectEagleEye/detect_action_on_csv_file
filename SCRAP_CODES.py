import numpy as np


list = np.zeros((0,))
list1 = [1,2,3,4]
list2 = [9,6,7,8]

hi = np.append(list, [list1], axis=0)
hi2 = np.append(hi, [list2], axis=0)
print(hi)
print(hi2)