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

def normalize_array(X, return_list = False):
    min_x, max_x = min(X), max(X)
    diff_x = (max_x - min_x)
    return (np.array(X) - min_x) / diff_x if not return_list else list((np.array(X) - min_x) / diff_x)