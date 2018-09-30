"""
    File Name: data_manipulation.py
    Author: Barney Wei
    Date Created: 09/30/2018
    Python Version: 3.5
"""

# import necessary libraries and files
import numpy as np
import math


def stretch_padding(matrix,
                    new_len,
                    dual_dim=False,
                    new_len_2=None,
                    square=False):
    """
    function that stretches a matrix col-wise to a specified length, which can be less or greater than original length
    :param matrix: NUMPY 1||2D ARRAY
    :param new_len: INT
    :param dual_dim: BOOLEAN - whether or not to stretch matrix in 2D (default=False)
    :param new_len_2: INT - dual_dim must be True and stretches matrix in 2nd dim (default=None)
    :param square: BOOLEAN - whether to stretch matrix in 2D into a square matrix (default=False)
    :return: NUMPY 2D ARRAY
    """
    # if matrix is 1D then change to 2D || elif dim is other than 1 or 2, return None
    if matrix.ndim == 1:
        matrix = matrix.reshape((1, -1))
    elif matrix.ndim != 2:
        return None

    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]

    # return the input matrix if length == new length
    if num_cols == new_len:
        return matrix

    # instantiate stretched_matrix
    stretched_matrix = np.zeros((num_rows, new_len))
    len_ratio = (num_cols - 1) / (new_len - 1)

    # iterate through columns of the matrix and write to stretched_matrix
    for index_matrix_row in range(matrix.shape[0]):
        for index_stretched_matrix_col in range(new_len):
            index_fraction = len_ratio * index_stretched_matrix_col
            index_fraction_floor = math.floor(index_fraction)
            index_fraction_ceil = math.ceil(index_fraction)
            interval_val_diff = matrix[index_matrix_row][index_fraction_ceil] - \
                                matrix[index_matrix_row][index_fraction_floor]
            fraction = index_fraction % 1
            stretched_matrix[index_matrix_row][index_stretched_matrix_col] = \
                matrix[index_matrix_row][index_fraction_floor] + (interval_val_diff * fraction)

    # dual_dim = True
    if dual_dim and new_len_2 != (None or 0):
        return np.transpose(stretch_padding(np.transpose(stretched_matrix), new_len_2))

    # square = True
    if square:
        return np.transpose(stretch_padding(np.transpose(stretched_matrix), new_len))

    return stretched_matrix


# CODE TESTING
if __name__ == "__main__":
    array = np.array([1, 100, 1000, 10])
    new_len = 10
    stretched_matrix = stretch_padding(array, new_len)
    print(stretched_matrix)

    array2 = np.array([1, 34, 67, 100, 400, 700, 1000, 670, 340, 10])
    new_len2 = 4
    stretched_matrix2 = stretch_padding(array2, new_len2)
    print(stretched_matrix2)

    array3 = np.array([[1, 100, 1000, 10], [90, -79, 10, 0]])
    new_len3 = 2
    new_len3_1 = 4
    stretched_matrix3_1 = stretch_padding(array3, new_len3, dual_dim=True, new_len_2=5)
    stretched_matrix3_2 = stretch_padding(array3, new_len3, square=True)
    print(stretched_matrix3_1)
    print(stretched_matrix3_2)

