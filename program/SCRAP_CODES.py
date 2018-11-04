import numpy as np

temp_signal_3d_tensor = np.zeros((7, 7, 0))

if __name__ == "__main__":
    print(np.dstack((temp_signal_3d_tensor, np.ones((7, 7)))))