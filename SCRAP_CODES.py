import numpy as np

def get_reference_vectors(signal_vectors):
    """
    function that output reference vectors for a body action
    :param signal_vectors: PANDAS DATAFRAME - contain multiple repeated signal vectors of the desired body action
    :return: NUMPY ARRAY - signal vectors for each port
    """
    num_cols = signal_vectors[0].shape[1]

    # get max signal vector row length in the signal_vectors list
    max_rows = 0
    for signal_vector in signal_vectors:
        if max_rows < signal_vector.shape[0]:
            max_rows = signal_vector.shape[0]

    # initiate numpy array to store reference vectors
    reference_vectors = np.zeros((max_rows, 0))

    # initialize temporary 3d tensor to hold all signal matrices after zero padding
    signal_3d_tensor = np.zeros((max_rows, num_cols, 0))
    for signal_vector in signal_vectors:
        # resize signal vectors by adding 0s
        num_padding = max_rows - signal_vector.shape[0]
        signal_vector = np.pad(signal_vector, ((0, num_padding), (0, 0)), "constant", constant_values=0)

        signal_3d_tensor = np.dstack((signal_3d_tensor, signal_vector))

    # get reference vector through finding the mean among vectors of the same column
    reference_vectors = np.mean(signal_3d_tensor, axis=2)

    return reference_vectors


if __name__ == "__main__":
    signal_vectors = [np.full((1, 4), 0), np.full((2, 4), 10), np.full((3, 4), 20)]
    reference_vectors = get_reference_vectors(signal_vectors)
    print(reference_vectors)