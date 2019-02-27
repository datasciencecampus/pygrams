import numpy as np


def bisearch_csr(array, target, start, end):
    while start <= end:
        middle = (start + end) // 2
        midpoint = array[middle]
        if midpoint > target:
            end = middle - 1
        elif midpoint < target:
            start = middle + 1
        else:
            return middle, array[middle] == target
    return 0, False


def remove_all_null_rows(sparse_mat):
    nonzero_row_indices, _ = sparse_mat.nonzero()
    unique_nonzero_indices = np.unique(nonzero_row_indices)
    return sparse_mat[unique_nonzero_indices]
