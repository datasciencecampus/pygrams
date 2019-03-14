import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile


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


def normalize_array(X, min_val=0.2, return_list=False):
    min_x, max_x = min(X), max(X)
    diff_x = (max_x - min_x)
    std_x = (np.array(X) - min_x) / diff_x
    x_scaled = std_x * (1 - min_val) + min_val
    return x_scaled if not return_list else list(x_scaled)


def w2vify(filein, fileout):
    glove_file = datapath(filein)
    tmp_file_name = get_tmpfile(fileout)
    _ = glove2word2vec(glove_file, tmp_file_name)
    return KeyedVectors.load_word2vec_format(tmp_file_name)
